"""Shared-index updates reach every worker without resetting provider state."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

import feedback_intelligence_agent.agent_snapshots as snapshots_module
import feedback_intelligence_agent.api as api_module
import feedback_intelligence_agent.factory as factory_module
from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.agent_snapshots import AgentSnapshots
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.index_persistence import write_index_json
from feedback_intelligence_agent.index_updates import update_json_index
from feedback_intelligence_agent.schemas import FeedbackRecord, MetadataFilters
from feedback_intelligence_agent.vector_store import InMemoryVectorStore, VectorStore

READ = {"X-Feedback-Agent-Key": "read-key"}
WRITE = {"X-Feedback-Agent-Key": "write-key"}
ADMIN = {"X-Feedback-Agent-Key": "admin-key"}


def test_qdrant_coordination_requires_an_explicit_nonblank_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variable = "FEEDBACK_AGENT_QDRANT_INGESTION_LOCK_PATH"
    monkeypatch.delenv(variable, raising=False)
    assert Settings(_env_file=None).qdrant_ingestion_lock_path is None
    monkeypatch.setenv(variable, str(tmp_path / "coordinator"))
    assert Settings(_env_file=None).qdrant_ingestion_lock_path == tmp_path / "coordinator"
    for blank in ("", "   "):
        monkeypatch.setenv(variable, blank)
        with pytest.raises(ValueError, match="cannot be blank"):
            Settings(_env_file=None)


def test_build_agent_accepts_a_snapshot_without_loading_the_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("An explicit snapshot must not load or rebuild the configured backend")

    monkeypatch.setattr(factory_module, "load_or_build_index", unexpected)
    agent = factory_module.build_agent(
        Settings(
            _env_file=None, vector_store="qdrant", llm_provider="local", telemetry_enabled=False
        ),
        vector_store=InMemoryVectorStore(dim=64),
    )
    assert agent.answer("What onboarding feedback exists?").citations == []


def record(identifier: str = "new") -> FeedbackRecord:
    return FeedbackRecord(
        tenant_id="acme",
        feedback_id=identifier,
        customer_segment="enterprise",
        channel="support_ticket",
        rating=2,
        text="Onboarding zirconium checklist needs an owner.",
        created_at="2026-10-29T10:00:00Z",
    )


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    data = tmp_path / "feedback.csv"
    data.write_text(
        "tenant_id,feedback_id,customer_segment,channel,rating,text,created_at\n"
        "acme,shared,enterprise,support_ticket,1,Acme onboarding feedback.,2026-05-01\n"
        "cobalt,shared,enterprise,support_ticket,2,Cobalt private feedback.,2026-05-02\n",
        encoding="utf-8",
    )
    config = Settings(
        _env_file=None,
        data_path=data,
        index_path=tmp_path / "index.json",
        conversation_store_path=tmp_path / "conversations",
        report_store_path=tmp_path / "reports",
        human_feedback_store_path=tmp_path / "human-feedback",
        active_learning_state_store_path=tmp_path / "states",
        job_store_path=tmp_path / "jobs",
        embedding_dim=64,
        vector_store="json",
        llm_provider="local",
        api_auth_enabled=True,
        api_reader_key="read-key",
        api_writer_key="write-key",
        api_admin_key="admin-key",
        api_tenant_bindings={"read-key": "acme", "write-key": "acme"},
        api_rate_limit_enabled=False,
        telemetry_enabled=False,
    )
    monkeypatch.setattr(api_module, "Settings", lambda: config)
    return config


def query(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/query", json={"question": "What onboarding feedback exists?", "top_k": 12}, headers=READ
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def update(settings: Settings, identifier: str = "new") -> None:
    update_json_index([record(identifier)], settings.index_path, embedding_dim=64)


@pytest.mark.parametrize("retriever", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize("writer", ["records", "external", "index", "job"])
def test_workers_reload_all_json_writers_and_keep_tenant_scope(
    settings: Settings, retriever: str, writer: str
) -> None:
    settings.retriever_type = retriever
    first, second = TestClient(api_module.create_app()), TestClient(api_module.create_app())
    old = query(second)
    assert {item["document_id"] for item in old["citations"]} == {"acme:shared"}
    if writer == "records":
        response = first.post(
            "/ingestion/records",
            json={"records": [record().model_dump(mode="json")]},
            headers=WRITE,
        )
        assert response.status_code == 200, response.text
    elif writer == "external":
        update(settings)
    else:
        with settings.data_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "acme,new,enterprise,support_ticket,2,Onboarding zirconium feedback.,2026-05-03\n"
            )
        response = first.post(
            "/index" if writer == "index" else "/ingestion/jobs",
            json={"input_path": str(settings.data_path), "index_path": str(settings.index_path)},
            headers=ADMIN,
        )
        assert response.status_code == (200 if writer == "index" else 202), response.text
        if writer == "job":
            job = first.get(f"/ingestion/jobs/{response.json()['job_id']}", headers=ADMIN)
            assert job.json()["status"] == "succeeded"
    for client in (first, second):
        answer = query(client)
        assert {item["document_id"] for item in answer["citations"]} == {"acme:shared", "acme:new"}
        summary = client.post("/query", json={"question": "Summarize sentiment"}, headers=READ)
        assert summary.json()["result"]["tool_run"]["output"]["total_records"] == 2


def test_streaming_and_existing_conversations_reload_external_updates(settings: Settings) -> None:
    client = TestClient(api_module.create_app())
    initial = client.post(
        "/chat", json={"message": "What onboarding feedback exists?"}, headers=WRITE
    )
    conversation_id = initial.json()["conversation_id"]
    update(settings)
    chat = client.post(
        "/chat",
        json={"message": "What onboarding feedback exists?", "conversation_id": conversation_id},
        headers=WRITE,
    )
    assert chat.status_code == 200, chat.text
    assert "acme:new" in {item["document_id"] for item in chat.json()["result"]["citations"]}
    update(settings, "stream")
    stream = client.post(
        "/query/stream",
        json={"question": "What onboarding feedback exists?", "top_k": 12},
        headers=READ,
    )
    assert stream.status_code == 200, stream.text
    metadata = next(
        block for block in stream.text.split("\n\n") if block.startswith("event: metadata")
    )
    payload = json.loads(metadata.split("data: ", 1)[1])
    assert {item["document_id"] for item in payload["citations"]} == {
        "acme:shared",
        "acme:new",
        "acme:stream",
    }


@pytest.mark.parametrize("failure", ["missing", "corrupt", "read", "prepare", "lock"])
def test_unavailable_index_fails_closed_and_recovers_without_restart(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client = TestClient(api_module.create_app())
    update(settings)
    committed = settings.index_path.read_text(encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"private detail at {settings.index_path}")

    with monkeypatch.context() as patch:
        if failure == "missing":
            settings.index_path.unlink()
        elif failure == "corrupt":
            write_index_json(settings.index_path, "invalid private payload")
        elif failure == "read":
            patch.setattr(snapshots_module.InMemoryVectorStore, "load", fail)
        elif failure == "prepare":
            patch.setattr(factory_module, "build_retriever", fail)
        else:
            patch.setattr(snapshots_module, "index_write_lock", fail)
        for route, payload in (
            ("/query", {"question": "What onboarding feedback exists?"}),
            ("/query/stream", {"question": "What onboarding feedback exists?"}),
            ("/chat", {"message": "What onboarding feedback exists?"}),
        ):
            response = client.post(route, json=payload, headers=WRITE)
            assert response.status_code == 503, response.text
            assert response.json() == {"detail": "Feedback index is temporarily unavailable"}
        assert client.get("/health").status_code == 200
        assert client.post("/query", json={"question": "Feedback?"}).status_code == 401
        assert (
            client.post(
                "/query", json={"question": "Feedback?", "tenant_id": "cobalt"}, headers=READ
            ).status_code
            == 403
        )
        if failure == "missing":
            assert not settings.index_path.exists()  # No silent rebuild from the source CSV.
    if failure in {"missing", "corrupt"}:
        write_index_json(settings.index_path, committed)
    assert "acme:new" in {item["document_id"] for item in query(client)["citations"]}


def test_same_size_and_timestamp_replacement_reloads(settings: Settings) -> None:
    client = TestClient(api_module.create_app())
    before = settings.index_path.stat()
    contents = settings.index_path.read_text(encoding="utf-8").replace(
        "Acme onboarding", "Acme zirconium "
    )
    write_index_json(settings.index_path, contents)
    os.utime(settings.index_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = settings.index_path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert "zirconium" in query(client)["citations"][0]["quote"]


def test_reload_reuses_policies_and_builds_only_once_per_version(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = AgentSnapshots(settings)
    original = snapshots.get()
    build = snapshots_module.build_agent_snapshot
    builds = 0

    def counted(
        config: Settings, store: VectorStore, *, previous: FeedbackInsightAgent
    ) -> FeedbackInsightAgent:
        nonlocal builds
        builds += 1
        return build(config, store, previous=previous)

    monkeypatch.setattr(snapshots_module, "build_agent_snapshot", counted)
    for _ in range(3):
        assert snapshots.get() is original
    assert builds == 0
    update(settings)
    with ThreadPoolExecutor(max_workers=6) as executor:
        agents = list(executor.map(lambda _: snapshots.get(), range(12)))
    refreshed = agents[0]
    assert builds == 1
    assert all(agent is refreshed for agent in agents)
    assert refreshed is not original
    assert refreshed.query_engine is not original.query_engine
    assert refreshed.tools is not original.tools
    for name in ("llm", "telemetry", "query_rewriter", "reranker", "hallucination_checker"):
        assert getattr(refreshed, name) is getattr(original, name)


def test_reload_and_ingestion_cannot_publish_out_of_order(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = AgentSnapshots(settings)
    original = snapshots.get()
    update(settings)
    entered, release = Event(), Event()
    build = snapshots_module.build_agent_snapshot

    def paused(
        config: Settings, store: VectorStore, *, previous: FeedbackInsightAgent
    ) -> FeedbackInsightAgent:
        entered.set()
        assert release.wait(10), "reload test timed out"
        return build(config, store, previous=previous)

    monkeypatch.setattr(snapshots_module, "build_agent_snapshot", paused)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reload = executor.submit(snapshots.get)
        try:
            assert entered.wait(10)
            ingest = executor.submit(snapshots.ingest, [record("later")], "acme")
        finally:
            release.set()
        reloaded = reload.result(timeout=10)
        assert ingest.result(timeout=10).inserted_records == 1
    for agent, expected in ((original, 1), (reloaded, 2), (snapshots.get(), 3)):
        answer = agent.answer("Summarize sentiment", filters=MetadataFilters(tenant_id="acme"))
        assert answer.tool_run is not None
        assert answer.tool_run.output["total_records"] == expected


def test_qdrant_does_not_probe_or_lock_a_json_index(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = factory_module.build_agent(settings)
    settings.vector_store = "qdrant"
    monkeypatch.setattr(snapshots_module, "build_agent", lambda _: agent)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Qdrant must not access the JSON index")

    monkeypatch.setattr(snapshots_module, "_file_version", unexpected)
    monkeypatch.setattr(snapshots_module, "index_write_lock", unexpected)
    snapshots = AgentSnapshots(settings)
    assert snapshots.get() is agent
    with pytest.raises(ValueError, match="JSON"):
        snapshots.ingest([record()], "acme")
