"""Tenant ingestion and consistent worker snapshots with the optional Qdrant SDK."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient

import feedback_intelligence_agent.api as api_module
import feedback_intelligence_agent.factory as factory_module
import feedback_intelligence_agent.qdrant_coordination as coordination_module
import feedback_intelligence_agent.qdrant_ingestion as ingestion_module
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.qdrant_coordination import qdrant_write_lock
from feedback_intelligence_agent.qdrant_ingestion import QdrantAgentSnapshots
from feedback_intelligence_agent.qdrant_store import QdrantVectorStore
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackRecord, MetadataFilters
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

qdrant = pytest.importorskip("qdrant_client")
READ = {"X-Feedback-Agent-Key": "read-key"}
WRITE = {"X-Feedback-Agent-Key": "write-key"}
ADMIN = {"X-Feedback-Agent-Key": "admin-key"}


def record(identifier: str = "new", **values: object) -> dict[str, Any]:
    return {
        "feedback_id": identifier,
        "customer_segment": "enterprise",
        "channel": "support_ticket",
        "rating": 2,
        "text": "Onboarding zirconium checklist needs an owner.",
        "created_at": "2026-10-30T10:00:00Z",
        **values,
    }


@dataclass
class Context:
    settings: Settings
    sdk: Any

    def app(self) -> TestClient:
        return TestClient(api_module.create_app())

    def points(self) -> list[dict[str, Any]]:
        points, _ = self.sdk.scroll(
            collection_name=self.settings.qdrant_collection,
            with_payload=True,
            with_vectors=True,
            limit=1000,
        )
        return [point.model_dump() for point in points]


@pytest.fixture(params=["memory", "server"])
def context(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Context]:
    if request.param == "server":
        if os.environ.get("FEEDBACK_AGENT_QDRANT_TEST", "").lower() not in {"1", "true", "yes"}:
            pytest.skip("set FEEDBACK_AGENT_QDRANT_TEST=1 for real-server API tests")
        sdk = qdrant.QdrantClient(
            url=os.environ.get("FEEDBACK_AGENT_QDRANT_URL", "http://localhost:6333")
        )
    else:
        sdk = qdrant.QdrantClient(location=":memory:")
    data = tmp_path / "feedback.csv"
    data.write_text(
        "tenant_id,feedback_id,customer_segment,channel,rating,text,created_at\n"
        f"acme,shared,enterprise,support_ticket,1,{'Acme onboarding feedback. ' * 60},2026-05-01\n"
        "cobalt,shared,enterprise,support_ticket,2,"
        "Cobalt private onboarding feedback.,2026-05-02\n",
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        data_path=data,
        index_path=tmp_path / "unused.json",
        embedding_dim=64,
        vector_store="qdrant",
        qdrant_collection="api_ingestion_" + uuid.uuid4().hex,
        qdrant_ingestion_lock_path=tmp_path / "coordinator",
        conversation_store_path=tmp_path / "conversations",
        report_store_path=tmp_path / "reports",
        human_feedback_store_path=tmp_path / "feedback",
        active_learning_state_store_path=tmp_path / "states",
        job_store_path=tmp_path / "jobs",
        llm_provider="local",
        telemetry_enabled=False,
        api_auth_enabled=True,
        api_reader_key="read-key",
        api_writer_key="write-key",
        api_admin_key="admin-key",
        api_tenant_bindings={"read-key": "acme", "write-key": "acme"},
        api_rate_limit_enabled=False,
    )
    original_init = QdrantVectorStore.__init__

    def injected_init(self: QdrantVectorStore, *args: object, **kwargs: object) -> None:
        kwargs["client"] = sdk
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(QdrantVectorStore, "__init__", injected_init)
    monkeypatch.setattr(api_module, "Settings", lambda: settings)
    try:
        yield Context(settings, sdk)
    finally:
        if sdk.collection_exists(settings.qdrant_collection):
            sdk.delete_collection(settings.qdrant_collection)
        sdk.close()


def query(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/query", json={"question": "What onboarding feedback exists?", "top_k": 12}, headers=READ
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.mark.parametrize("retriever", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize("cache_ttl", [0, 30])
def test_ingestion_refreshes_all_workers_preserves_tenants_and_removes_old_chunks(
    context: Context, retriever: str, cache_ttl: float
) -> None:
    context.settings.retriever_type = retriever
    context.settings.qdrant_snapshot_ttl_seconds = cache_ttl
    first, second = context.app(), context.app()
    query(first)
    query(second)
    foreign = [
        point for point in context.points() if point["payload"]["source_id"] == "cobalt:shared"
    ]
    response = first.post(
        "/ingestion/records", json={"records": [record("shared"), record()]}, headers=WRITE
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "tenant_id": "acme",
        "input_records": 2,
        "inserted_records": 1,
        "updated_records": 1,
        "skipped_records": 0,
        "added_chunks": 2,
        "removed_chunks": 3,
    }
    assert [
        point for point in context.points() if point["payload"]["source_id"] == "cobalt:shared"
    ] == foreign
    assert len(context.points()) == 3
    for client in (first, second):
        answer = query(client)
        assert {item["document_id"] for item in answer["citations"]} == {"acme:shared", "acme:new"}
        assert all("zirconium" in item["quote"] for item in answer["citations"])
        summary = client.post("/query", json={"question": "Summarize sentiment"}, headers=READ)
        assert summary.json()["result"]["tool_run"]["output"]["total_records"] == 2
    assert not context.settings.index_path.exists()
    assert Path(f"{context.settings.qdrant_ingestion_lock_path}.lock").exists()


@pytest.mark.parametrize(
    "problem",
    ["reader", "anonymous", "foreign_batch", "foreign_record", "paths", "empty", "unbound_writer"],
)
def test_authorization_and_validation_precede_remote_writes(context: Context, problem: str) -> None:
    client = context.app()
    before = context.points()
    headers = WRITE
    payload: dict[str, Any] = {"records": [record()]}
    expected = 403
    if problem == "reader":
        headers = READ
    elif problem == "anonymous":
        headers, expected = {}, 401
    elif problem == "foreign_batch":
        payload["tenant_id"] = "cobalt"
    elif problem == "foreign_record":
        payload["records"].append(record("foreign", tenant_id="cobalt"))
    elif problem == "paths":
        payload["index_path"], expected = "arbitrary.json", 422
    elif problem == "empty":
        payload["records"], expected = [], 422
    else:
        context.settings.api_tenant_bindings = {}
    assert client.post("/ingestion/records", json=payload, headers=headers).status_code == expected
    assert context.points() == before


def test_admin_scope_duplicates_redaction_and_retry(context: Context) -> None:
    client = context.app()
    payload = {
        "tenant_id": "cobalt",
        "records": [
            record("new", text="Discard this older record."),
            record("new", text="Onboarding owner@example.com needs an owner."),
        ],
    }
    first = client.post("/ingestion/records", json=payload, headers=ADMIN)
    second = client.post("/ingestion/records", json=payload, headers=ADMIN)
    assert first.status_code == second.status_code == 200
    assert first.json()["inserted_records"] == 1
    assert first.json()["skipped_records"] == 1
    assert second.json()["updated_records"] == 1
    stored = [
        point["payload"]["text"]
        for point in context.points()
        if point["payload"]["source_id"] == "cobalt:new"
    ]
    assert len(stored) == 1 and "[REDACTED_EMAIL]" in stored[0]
    assert "owner@example.com" not in stored[0] and "Discard" not in stored[0]
    assert "cobalt:new" not in {item["document_id"] for item in query(client)["citations"]}


@pytest.mark.parametrize("cache_ttl", [0, 30])
def test_stream_and_existing_chat_see_other_worker_updates(
    context: Context, cache_ttl: float
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = cache_ttl
    first, second = context.app(), context.app()
    initial = second.post(
        "/chat", json={"message": "What onboarding feedback exists?"}, headers=WRITE
    )
    conversation_id = initial.json()["conversation_id"]
    assert (
        first.post("/ingestion/records", json={"records": [record()]}, headers=WRITE).status_code
        == 200
    )
    chat = second.post(
        "/chat",
        json={
            "message": "What onboarding feedback exists?",
            "conversation_id": conversation_id,
            "top_k": 12,
        },
        headers=WRITE,
    )
    assert chat.status_code == 200, chat.text
    assert "acme:new" in {item["document_id"] for item in chat.json()["result"]["citations"]}
    stream = second.post(
        "/query/stream",
        json={"question": "What onboarding feedback exists?", "top_k": 12},
        headers=READ,
    )
    assert stream.status_code == 200, stream.text
    metadata = next(
        block for block in stream.text.split("\n\n") if block.startswith("event: metadata")
    )
    assert "acme:new" in {
        item["document_id"] for item in json.loads(metadata.split("data: ", 1)[1])["citations"]
    }


@pytest.mark.parametrize("failure", ["embedding", "prepare", "revision", "upsert"])
def test_precommit_failures_preserve_remote_data_and_hide_details(
    context: Context, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client = context.app()
    before = context.points()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"private credential at {context.settings.data_path}")

    with monkeypatch.context() as patch:
        if failure == "embedding":
            patch.setattr(HashingEmbeddingModel, "embed", fail)
        elif failure == "prepare":
            patch.setattr(factory_module, "build_retriever", fail)
        elif failure == "revision":
            patch.setattr(coordination_module, "write_index_json", fail)
        else:
            patch.setattr(QdrantVectorStore, "upsert_sources", fail)
        response = client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE)
    assert response.status_code == 500
    assert response.json() == {"detail": "Record ingestion failed"}
    assert context.points() == before
    assert (
        client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE).status_code
        == 200
    )


@pytest.mark.parametrize("cache_ttl", [0, 30])
def test_partial_remote_write_is_visible_and_retry_completes_cleanup(
    context: Context, monkeypatch: pytest.MonkeyPatch, cache_ttl: float
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = cache_ttl
    first, second = context.app(), context.app()
    foreign = [
        point for point in context.points() if point["payload"]["source_id"] == "cobalt:shared"
    ]

    def fail(**kwargs: object) -> None:
        raise RuntimeError("remote deletion failed")

    with monkeypatch.context() as patch:
        patch.setattr(context.sdk, "delete", fail)
        response = first.post(
            "/ingestion/records", json={"records": [record("shared")]}, headers=WRITE
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Record ingestion failed"}
    assert len(context.points()) == 4
    stored_texts = [point["payload"]["text"] for point in context.points()]
    assert any("zirconium" in text for text in stored_texts) and any(
        "Acme" in text for text in stored_texts
    )
    partial = second.post(
        "/query",
        json={"question": "What zirconium checklist needs an owner?", "top_k": 12},
        headers=READ,
    )
    assert partial.status_code == 200, partial.text
    assert any("zirconium" in item["quote"] for item in partial.json()["result"]["citations"])
    assert (
        first.post(
            "/ingestion/records", json={"records": [record("shared")]}, headers=WRITE
        ).status_code
        == 200
    )
    assert len(context.points()) == 2
    assert all("zirconium" in item["quote"] for item in query(second)["citations"])
    assert [
        point for point in context.points() if point["payload"]["source_id"] == "cobalt:shared"
    ] == foreign


def test_unavailable_snapshot_returns_503_and_retries(
    context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = context.app()

    def fail(self: QdrantVectorStore) -> None:
        raise RuntimeError("private backend details")

    with monkeypatch.context() as patch:
        patch.setattr(QdrantVectorStore, "snapshot", fail)
        for route, payload in (
            ("/query", {"question": "Feedback?"}),
            ("/query/stream", {"question": "Feedback?"}),
            ("/chat", {"message": "Feedback?"}),
        ):
            response = client.post(route, json=payload, headers=WRITE)
            assert response.status_code == 503, response.text
            assert response.json() == {"detail": "Feedback index is temporarily unavailable"}
        assert client.get("/health").status_code == 200
    assert query(client)["citations"]


def test_two_workers_serialize_same_source_replacements(
    context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = context.app(), context.app()
    entered, release, later_started = Event(), Event(), Event()
    original = QdrantVectorStore.upsert_sources
    calls: list[str] = []

    def paused(
        self: QdrantVectorStore, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]
    ) -> int:
        calls.append(chunks[0].text)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10), "ingestion test timed out"
        return original(self, chunks, vectors)

    def submit_later() -> int:
        later_started.set()
        return second.post(
            "/ingestion/records",
            json={"records": [record("shared", text="Later onboarding revision wins.")]},
            headers=WRITE,
        ).status_code

    monkeypatch.setattr(QdrantVectorStore, "upsert_sources", paused)
    with ThreadPoolExecutor(max_workers=2) as executor:
        earlier = executor.submit(
            first.post, "/ingestion/records", json={"records": [record("shared")]}, headers=WRITE
        )
        try:
            assert entered.wait(10)
            later = executor.submit(submit_later)
            assert later_started.wait(10)
            assert len(calls) == 1
        finally:
            release.set()
        assert earlier.result(timeout=10).status_code == 200
        assert later.result(timeout=10) == 200
    assert all("Later onboarding" in item["quote"] for item in query(first)["citations"])
    assert len(context.points()) == 2


def test_selected_snapshot_survives_updates_and_preserves_provider_state(context: Context) -> None:
    snapshots = QdrantAgentSnapshots(context.settings)
    previous = snapshots.get()
    snapshots.ingest([FeedbackRecord.model_validate(record(tenant_id="acme"))], "acme")
    refreshed = snapshots.get()
    for name in ("llm", "telemetry", "query_rewriter", "reranker", "hallucination_checker"):
        assert getattr(refreshed, name) is getattr(previous, name)
    for agent, count in ((previous, 1), (refreshed, 2)):
        answer = agent.answer("Summarize sentiment", filters=MetadataFilters(tenant_id="acme"))
        assert answer.tool_run is not None and answer.tool_run.output["total_records"] == count


def test_external_cooperating_update_and_empty_collection_refresh(context: Context) -> None:
    client = context.app()
    with index_write_lock(context.settings.qdrant_ingestion_lock_path):
        context.sdk.delete(
            collection_name=context.settings.qdrant_collection,
            points_selector=qdrant.models.Filter(),
            wait=True,
        )
    assert query(client)["citations"] == []
    assert context.points() == []  # Reads do not silently rebuild from the source CSV.


def test_requires_opt_in_and_valid_tenant_batches(context: Context) -> None:
    context.settings.qdrant_ingestion_lock_path = None
    client = context.app()
    before = context.points()
    assert (
        client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE).status_code
        == 409
    )
    assert context.points() == before

    with pytest.raises(ValueError, match="shared lock path"):
        QdrantAgentSnapshots(context.settings)
    context.settings.qdrant_ingestion_lock_path = context.settings.data_path.parent / "coordinator"
    snapshots = QdrantAgentSnapshots(context.settings)
    for records in ([], [FeedbackRecord.model_validate(record(tenant_id="cobalt"))]):
        with pytest.raises(ValueError, match="batch tenant"):
            snapshots.ingest(records, "acme")
    assert context.points() == before


@pytest.mark.parametrize("ttl", [0, 10])
def test_snapshot_reuse_and_exact_ttl_boundary(
    context: Context, monkeypatch: pytest.MonkeyPatch, ttl: float
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = ttl
    clock = [100.0]
    monkeypatch.setattr(ingestion_module, "monotonic", lambda: clock[0])
    snapshots = QdrantAgentSnapshots(context.settings)
    original = snapshots.get()
    read_snapshot = QdrantVectorStore.snapshot
    reads = 0

    def counted(self: QdrantVectorStore) -> InMemoryVectorStore:
        nonlocal reads
        reads += 1
        return read_snapshot(self)

    monkeypatch.setattr(QdrantVectorStore, "snapshot", counted)
    for _ in range(25):
        current = snapshots.get()
        assert (current is original) == (ttl > 0)
    assert reads == (0 if ttl > 0 else 25)
    if ttl > 0:
        clock[0] = 109.99
        assert snapshots.get() is original
        clock[0] = 110.0
        refreshed = snapshots.get()
        assert refreshed is not original and reads == 1
        assert snapshots.get() is refreshed and reads == 1
        clock[0] = 120.0
        assert snapshots.get() is not refreshed and reads == 2


def test_only_one_concurrent_reload_per_worker_generation(
    context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = 30
    snapshots = QdrantAgentSnapshots(context.settings)
    original = snapshots.get()
    read_snapshot = QdrantVectorStore.snapshot
    reads = 0

    def counted(self: QdrantVectorStore) -> InMemoryVectorStore:
        nonlocal reads
        reads += 1
        return read_snapshot(self)

    monkeypatch.setattr(QdrantVectorStore, "snapshot", counted)
    with qdrant_write_lock(context.settings.qdrant_ingestion_lock_path):
        pass
    with ThreadPoolExecutor(max_workers=4) as executor:
        agents = list(executor.map(lambda _: snapshots.get(), range(8)))
    assert reads == 1 and agents[0] is not original
    assert all(agent is agents[0] for agent in agents)


@pytest.mark.parametrize("invalidate", [False, True])
def test_external_writes_refresh_by_generation_or_expiry(
    context: Context, monkeypatch: pytest.MonkeyPatch, invalidate: bool
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = 10
    clock = [100.0]
    monkeypatch.setattr(ingestion_module, "monotonic", lambda: clock[0])
    client = context.app()
    assert query(client)["citations"]
    lock = qdrant_write_lock if invalidate else index_write_lock
    with lock(context.settings.qdrant_ingestion_lock_path):
        context.sdk.delete(
            collection_name=context.settings.qdrant_collection,
            points_selector=qdrant.models.Filter(),
            wait=True,
        )
    if invalidate:
        assert query(client)["citations"] == []
    else:
        assert query(client)["citations"]  # Legacy writers are visible at bounded expiry.
        clock[0] = 110.0
        assert query(client)["citations"] == []
    assert context.points() == []


@pytest.mark.parametrize("invalidate", [False, True])
def test_failed_refresh_does_not_extend_cache_lifetime_or_accept_new_generation(
    context: Context, monkeypatch: pytest.MonkeyPatch, invalidate: bool
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = 10
    clock = [100.0]
    monkeypatch.setattr(ingestion_module, "monotonic", lambda: clock[0])
    client = context.app()

    def fail(self: QdrantVectorStore) -> None:
        raise RuntimeError("private backend details")

    with monkeypatch.context() as patch:
        patch.setattr(QdrantVectorStore, "snapshot", fail)
        assert query(client)["citations"]  # A valid cache does not contact Qdrant.
        if invalidate:
            with qdrant_write_lock(context.settings.qdrant_ingestion_lock_path):
                pass
        else:
            clock[0] = 110.0
        for _ in range(2):
            response = client.post(
                "/query", json={"question": "Onboarding feedback?"}, headers=READ
            )
            assert response.status_code == 503
            assert response.json() == {"detail": "Feedback index is temporarily unavailable"}
    assert query(client)["citations"]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_revision_damage_never_silently_accepts_a_cached_snapshot(
    context: Context, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    context.settings.qdrant_snapshot_ttl_seconds = 30
    client = context.app()
    before = context.points()
    revision = Path(f"{context.settings.qdrant_ingestion_lock_path}.revision.json")
    with index_write_lock(context.settings.qdrant_ingestion_lock_path):
        if damage == "missing":
            revision.unlink()
        else:
            revision.write_text("private malformed marker", encoding="utf-8")
    reads = 0
    read_snapshot = QdrantVectorStore.snapshot

    def counted(self: QdrantVectorStore) -> InMemoryVectorStore:
        nonlocal reads
        reads += 1
        return read_snapshot(self)

    monkeypatch.setattr(QdrantVectorStore, "snapshot", counted)
    if damage == "corrupt":
        response = client.post("/query", json={"question": "Onboarding feedback?"}, headers=READ)
        assert response.status_code == 503 and reads == 0
        response = client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE)
        assert response.status_code == 500
        assert context.points() == before
        assert revision.read_text(encoding="utf-8") == "private malformed marker"
        with index_write_lock(context.settings.qdrant_ingestion_lock_path):
            revision.unlink()
        reads = 0
    assert query(client)["citations"] and reads == 1
