"""Tenant record ingestion preserves persisted and serving index boundaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

import feedback_intelligence_agent.api as api_module
import feedback_intelligence_agent.index_persistence as persistence_module
import feedback_intelligence_agent.record_ingestion as ingestion_module
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.factory import build_agent
from feedback_intelligence_agent.record_ingestion import ingest_record_batch
from feedback_intelligence_agent.retrieval import Retriever
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.vector_store import (
    InMemoryVectorStore,
    PersistedVectorStore,
    VectorStore,
)

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
        "created_at": "2026-10-29T10:00:00Z",
        **values,
    }


@pytest.fixture()
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Settings]:
    data = tmp_path / "feedback.csv"
    data.write_text(
        "tenant_id,feedback_id,customer_segment,channel,rating,text,created_at\n"
        "acme,shared,enterprise,support_ticket,1,Acme original onboarding feedback.,2026-05-01\n"
        "cobalt,shared,enterprise,support_ticket,2,"
        "Cobalt private onboarding feedback.,2026-05-02\n",
        encoding="utf-8",
    )
    settings = Settings(
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
    monkeypatch.setattr(api_module, "Settings", lambda: settings)
    return TestClient(api_module.create_app()), settings


def citations(client: TestClient, *, tenant: str = "acme") -> list[dict[str, Any]]:
    response = client.post(
        "/query",
        json={
            "question": "What onboarding feedback is available?",
            "tenant_id": tenant,
            "top_k": 12,
        },
        headers=ADMIN,
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]["citations"]


@pytest.mark.parametrize("retriever", ["dense", "lexical", "hybrid"])
def test_ingestion_preserves_other_tenants_and_refreshes_retrieval_and_tools(
    context: tuple[TestClient, Settings], retriever: str
) -> None:
    client, settings = context
    settings.retriever_type = retriever
    original = PersistedVectorStore.model_validate_json(
        settings.index_path.read_text(encoding="utf-8")
    )
    foreign = [
        (chunk.model_dump(), vector)
        for chunk, vector in zip(original.chunks, original.vectors, strict=True)
        if chunk.metadata["tenant_id"] == "cobalt"
    ]
    foreign_citations = citations(client, tenant="cobalt")
    response = client.post(
        "/ingestion/records",
        headers=WRITE,
        json={
            "records": [
                record("shared"),
                record("new", text="Onboarding new feedback needs support."),
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "tenant_id": "acme",
        "input_records": 2,
        "inserted_records": 1,
        "updated_records": 1,
        "skipped_records": 0,
        "added_chunks": 2,
        "removed_chunks": 1,
    }
    persisted = InMemoryVectorStore.load(settings.index_path)
    persisted_payload = PersistedVectorStore.model_validate_json(
        settings.index_path.read_text(encoding="utf-8")
    )
    assert [
        (chunk.model_dump(), vector)
        for chunk, vector in zip(persisted_payload.chunks, persisted_payload.vectors, strict=True)
        if chunk.metadata["tenant_id"] == "cobalt"
    ] == foreign
    assert {chunk.source_id for chunk in persisted.chunks} == {
        "acme:shared",
        "acme:new",
        "cobalt:shared",
    }
    assert "zirconium" in next(
        chunk.text for chunk in persisted.chunks if chunk.source_id == "acme:shared"
    )
    own_citations = citations(client)
    assert {item["document_id"] for item in own_citations} == {"acme:shared", "acme:new"}
    assert [
        (item["document_id"], item["quote"]) for item in citations(client, tenant="cobalt")
    ] == [(item["document_id"], item["quote"]) for item in foreign_citations]
    summary = client.post("/query", json={"question": "Summarize sentiment"}, headers=READ)
    assert summary.json()["result"]["tool_run"]["output"]["total_records"] == 2
    assert settings.data_path.read_text(encoding="utf-8").count("zirconium") == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"tenant_id": "cobalt", "records": [record()]},
        {"records": [record(), record("foreign", tenant_id="cobalt")]},
        {"records": [record(tenant_id="default")]},
    ],
)
def test_rejects_the_entire_foreign_tenant_batch_before_writing(
    context: tuple[TestClient, Settings], payload: dict[str, Any]
) -> None:
    client, settings = context
    before = settings.index_path.read_bytes()
    response = client.post("/ingestion/records", json=payload, headers=WRITE)
    assert response.status_code == 403
    assert settings.index_path.read_bytes() == before
    assert {item["document_id"] for item in citations(client)} == {"acme:shared"}


@pytest.mark.parametrize("headers", [{}, READ])
def test_requires_an_authenticated_writer(
    context: tuple[TestClient, Settings], headers: dict[str, str]
) -> None:
    client, settings = context
    before = settings.index_path.read_bytes()
    response = client.post("/ingestion/records", json={"records": [record()]}, headers=headers)
    assert response.status_code == (401 if not headers else 403)
    assert settings.index_path.read_bytes() == before


def test_unbound_writer_cannot_choose_a_tenant(context: tuple[TestClient, Settings]) -> None:
    client, settings = context
    settings.api_tenant_bindings = {}
    response = client.post(
        "/ingestion/records", json={"tenant_id": "acme", "records": [record()]}, headers=WRITE
    )
    assert response.status_code == 403


@pytest.mark.parametrize("bound", [False, True])
def test_admin_can_ingest_into_authorized_tenant(
    context: tuple[TestClient, Settings], bound: bool
) -> None:
    client, settings = context
    if bound:
        settings.api_tenant_bindings["admin-key"] = "cobalt"
    response = client.post(
        "/ingestion/records",
        json={"tenant_id": " Cobalt ", "records": [record(tenant_id="COBALT")]},
        headers=ADMIN,
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "cobalt"
    assert {item["document_id"] for item in citations(client, tenant="cobalt")} == {
        "cobalt:shared",
        "cobalt:new",
    }
    if bound:
        denied = client.post(
            "/ingestion/records", json={"tenant_id": "acme", "records": [record()]}, headers=ADMIN
        )
        assert denied.status_code == 403


def test_default_tenant_and_escaped_ids_stay_distinct(context: tuple[TestClient, Settings]) -> None:
    client, settings = context
    response = client.post(
        "/ingestion/records", json={"records": [record("acme:shared")]}, headers=ADMIN
    )
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "default"
    assert {chunk.source_id for chunk in InMemoryVectorStore.load(settings.index_path).chunks} == {
        "acme:shared",
        "cobalt:shared",
        "acme%3Ashared",
    }


def test_local_unauthenticated_mode_can_ingest_records(
    context: tuple[TestClient, Settings],
) -> None:
    client, settings = context
    settings.api_auth_enabled = False
    settings.api_tenant_bindings = {}
    response = client.post("/ingestion/records", json={"tenant_id": "acme", "records": [record()]})
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "acme"


def test_streams_and_existing_conversations_use_the_updated_snapshot(
    context: tuple[TestClient, Settings],
) -> None:
    client, _ = context
    first = client.post("/chat", json={"message": "What onboarding issues exist?"}, headers=WRITE)
    assert first.status_code == 200
    assert (
        client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE).status_code
        == 200
    )
    followup = client.post(
        "/chat",
        json={
            "message": "What about the onboarding zirconium checklist?",
            "conversation_id": first.json()["conversation_id"],
        },
        headers=WRITE,
    )
    assert followup.status_code == 200
    assert "acme:new" in {item["document_id"] for item in followup.json()["result"]["citations"]}
    stream = client.post(
        "/query/stream",
        json={"question": "What onboarding zirconium checklist feedback exists?"},
        headers=READ,
    )
    assert stream.status_code == 200
    assert '"document_id": "acme:new"' in stream.text
    assert "cobalt:shared" not in stream.text


@pytest.mark.parametrize(
    "payload",
    [
        {"records": []},
        {"records": [record()] * 1001},
        {"records": [record(), record("invalid", rating=6)]},
        {"tenant_id": " ", "records": [record()]},
        {"index_path": "other.json", "records": [record()]},
        {"input_path": "private.csv", "records": [record()]},
    ],
)
def test_batch_validation_and_path_rejection_are_atomic(
    context: tuple[TestClient, Settings], payload: dict[str, Any]
) -> None:
    client, settings = context
    before = settings.index_path.read_bytes()
    assert client.post("/ingestion/records", json=payload, headers=WRITE).status_code == 422
    assert settings.index_path.read_bytes() == before


def test_duplicate_records_and_retries_are_upserts_with_pii_redaction(
    context: tuple[TestClient, Settings],
) -> None:
    client, settings = context
    payload = {
        "records": [
            record(text="Discard this old text"),
            record(text="Onboarding email owner@example.com now."),
        ]
    }
    first = client.post("/ingestion/records", json=payload, headers=WRITE)
    second = client.post("/ingestion/records", json=payload, headers=WRITE)
    assert first.json()["inserted_records"] == 1
    assert first.json()["skipped_records"] == 1
    assert second.json()["inserted_records"] == 0
    assert second.json()["updated_records"] == 1
    stored = InMemoryVectorStore.load(settings.index_path)
    new = [chunk for chunk in stored.chunks if chunk.source_id == "acme:new"]
    assert len(new) == 1
    assert "[REDACTED_EMAIL]" in new[0].text
    assert "owner@example.com" not in new[0].text
    assert "Discard" not in new[0].text


def test_qdrant_configuration_cannot_write_to_a_json_index(
    context: tuple[TestClient, Settings],
) -> None:
    client, settings = context
    settings.vector_store = "qdrant"
    before = settings.index_path.read_bytes()
    response = client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE)
    assert response.status_code == 409
    assert settings.index_path.read_bytes() == before


@pytest.mark.parametrize("failure", ["embedding", "snapshot", "persist"])
def test_failed_ingestion_preserves_disk_and_serving_agent_without_leaking_errors(
    context: tuple[TestClient, Settings], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client, settings = context
    before = settings.index_path.read_bytes()
    old_citations = citations(client)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"private credential at {settings.index_path}")

    with monkeypatch.context() as patch:
        if failure == "embedding":
            patch.setattr(HashingEmbeddingModel, "embed", fail)
        elif failure == "snapshot":
            patch.setattr(ingestion_module, "build_retriever", fail)
        else:
            patch.setattr(persistence_module.os, "replace", fail)
        response = client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE)
    assert response.status_code == 500
    assert response.json() == {"detail": "Record ingestion failed"}
    assert settings.index_path.read_bytes() == before
    assert citations(client) == old_citations
    assert (
        client.post("/ingestion/records", json={"records": [record()]}, headers=WRITE).status_code
        == 200
    )


@pytest.mark.parametrize("invalid", ["backend", "empty", "foreign"])
def test_batch_helper_rejects_invalid_inputs_before_persistence(
    context: tuple[TestClient, Settings], invalid: str
) -> None:
    _, settings = context
    agent = build_agent(settings)
    before = settings.index_path.read_bytes()
    records = [FeedbackRecord.model_validate(record(tenant_id="acme"))]
    if invalid == "backend":
        settings.vector_store = "qdrant"
    elif invalid == "empty":
        records = []
    else:
        records[0].tenant_id = "cobalt"
    with pytest.raises(ValueError):
        ingest_record_batch(records, "acme", settings=settings, agent=agent)
    assert settings.index_path.read_bytes() == before


def test_snapshot_refresh_keeps_provider_and_policies(context: tuple[TestClient, Settings]) -> None:
    _, settings = context
    original = build_agent(settings)
    result, refreshed = ingest_record_batch(
        [FeedbackRecord.model_validate(record(tenant_id="acme"))],
        "acme",
        settings=settings,
        agent=original,
    )
    assert result.inserted_records == 1
    assert refreshed is not original
    assert refreshed.query_engine is not original.query_engine
    assert refreshed.tools is not original.tools
    for name in ("llm", "telemetry", "query_rewriter", "reranker", "hallucination_checker"):
        assert getattr(refreshed, name) is getattr(original, name)


def test_queries_keep_the_old_snapshot_until_ingestion_commits(
    context: tuple[TestClient, Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = context
    entered, release = Event(), Event()
    build_retriever = ingestion_module.build_retriever

    def paused_build(settings: Settings, store: VectorStore) -> Retriever:
        snapshot = build_retriever(settings, store)
        entered.set()
        assert release.wait(10), "snapshot test timed out"
        return snapshot

    monkeypatch.setattr(ingestion_module, "build_retriever", paused_build)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            client.post, "/ingestion/records", json={"records": [record()]}, headers=WRITE
        )
        try:
            assert entered.wait(10)
            assert {item["document_id"] for item in citations(client)} == {"acme:shared"}
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 200
    assert {item["document_id"] for item in citations(client)} == {"acme:shared", "acme:new"}


def test_concurrent_tenant_batches_preserve_every_record_and_latest_snapshot(
    context: tuple[TestClient, Settings],
) -> None:
    client, settings = context

    def submit(number: int) -> int:
        tenant = "acme" if number % 2 == 0 else "cobalt"
        response = client.post(
            "/ingestion/records",
            json={"tenant_id": tenant, "records": [record(str(number))]},
            headers=WRITE if tenant == "acme" else ADMIN,
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(submit, range(8))) == [200] * 8
    assert InMemoryVectorStore.load(settings.index_path).size == 10
    for tenant in ("acme", "cobalt"):
        summary = client.post(
            "/query", json={"question": "Summarize sentiment", "tenant_id": tenant}, headers=ADMIN
        )
        assert summary.json()["result"]["tool_run"]["output"]["total_records"] == 5
