from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import feedback_intelligence_agent.api as api_module
from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.lexical_search import BM25Retriever
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.metadata_filters import matches_metadata_filters
from feedback_intelligence_agent.retrieval import HybridRetriever, QueryEngine, Retriever
from feedback_intelligence_agent.schemas import DocumentChunk, MetadataFilters
from feedback_intelligence_agent.telemetry import InMemoryTelemetrySink, Telemetry
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

QUESTION = "onboarding checklist"
MATCHING_METADATA: dict[str, object] = {
    "tenant_id": "acme",
    "customer_segment": "Enterprise",
    "channel": "support_ticket",
    "rating": 2,
    "created_at": "2026-06-01T00:00:00+00:00",
}


def build_crowded_retriever(kind: str, excluded: dict[str, object]) -> Retriever:
    chunks = [
        DocumentChunk(
            chunk_id=f"excluded-{index}",
            source_id=f"excluded-{index}",
            text=QUESTION,
            metadata={**MATCHING_METADATA, **excluded},
        )
        for index in range(32)
    ]
    chunks.append(
        DocumentChunk(
            chunk_id="wanted",
            source_id="wanted",
            text="onboarding setup delay",
            metadata=MATCHING_METADATA,
        )
    )
    model = HashingEmbeddingModel(dim=128)
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    dense = QueryEngine(model, store)
    lexical = BM25Retriever(chunks)
    return {"dense": dense, "lexical": lexical, "hybrid": HybridRetriever(dense, lexical)}[kind]


@pytest.mark.parametrize("kind", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize(
    ("filters", "excluded"),
    [
        (MetadataFilters(tenant_id="Acme"), {"tenant_id": "cobalt"}),
        (MetadataFilters(customer_segment="enterprise"), {"customer_segment": "startup"}),
        (MetadataFilters(channel="support_ticket"), {"channel": "app_review"}),
        (MetadataFilters(min_rating=2), {"rating": 1}),
        (MetadataFilters(max_rating=2), {"rating": 5}),
        (
            MetadataFilters(created_after=datetime(2026, 6, 1)),
            {"created_at": "2026-01-01T00:00:00"},
        ),
        (
            MetadataFilters(created_before=datetime(2026, 6, 1, tzinfo=timezone.utc)),
            {"created_at": "2026-12-01T00:00:00"},
        ),
        (
            MetadataFilters(tenant_id="acme", customer_segment="enterprise", max_rating=2),
            {"tenant_id": "cobalt", "customer_segment": "startup", "rating": 5},
        ),
    ],
)
def test_filters_apply_before_candidate_limits(
    kind: str, filters: MetadataFilters, excluded: dict[str, object]
) -> None:
    retriever = build_crowded_retriever(kind, excluded)
    assert retriever.search(QUESTION, top_k=1)[0].chunk.source_id != "wanted"

    agent = FeedbackInsightAgent(retriever, DeterministicLLM())
    answer = agent.answer(QUESTION, top_k=1, filters=filters)

    assert [citation.document_id for citation in answer.citations] == ["wanted"]
    assert [
        hit.chunk.source_id for hit in retriever.search(QUESTION, top_k=1, filters=filters)
    ] == ["wanted"]


@pytest.mark.parametrize("kind", ["dense", "lexical", "hybrid"])
def test_empty_filter_preserves_ranking_and_no_matches_return_empty(kind: str) -> None:
    retriever = build_crowded_retriever(kind, {"tenant_id": "cobalt"})
    assert retriever.search(QUESTION, top_k=2, filters=MetadataFilters()) == retriever.search(
        QUESTION, top_k=2
    )
    assert retriever.search(QUESTION, top_k=2, filters=MetadataFilters(tenant_id="missing")) == []


@pytest.mark.parametrize(
    ("metadata", "filters", "expected"),
    [
        ({}, MetadataFilters(tenant_id="default", min_rating=5), True),
        ({}, MetadataFilters(tenant_id="acme"), False),
        ({"rating": "2"}, MetadataFilters(min_rating=2, max_rating=2), True),
        ({"rating": "unknown"}, MetadataFilters(min_rating=5), True),
        ({"rating": None}, MetadataFilters(max_rating=2), False),
        ({"created_at": "invalid"}, MetadataFilters(created_after=datetime(2026, 1, 1)), False),
        ({}, MetadataFilters(created_before=datetime(2026, 1, 1)), False),
    ],
)
def test_shared_filters_preserve_legacy_metadata_semantics(
    metadata: dict[str, object], filters: MetadataFilters, expected: bool
) -> None:
    assert matches_metadata_filters(metadata, filters) is expected


@pytest.mark.parametrize(
    "created_at",
    [
        "2026-06-01T00:00:00Z",
        "2026-06-01T01:00:00+01:00",
        datetime(2026, 6, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        datetime(2026, 6, 1),
    ],
)
def test_date_filters_use_inclusive_utc_bounds(created_at: str | datetime) -> None:
    filters = MetadataFilters(
        created_after=datetime(2026, 6, 1), created_before=datetime(2026, 6, 1)
    )
    assert matches_metadata_filters({"created_at": created_at}, filters)


@pytest.mark.parametrize("aware_after", [False, True])
def test_date_range_validation_accepts_mixed_timezone_bounds(aware_after: bool) -> None:
    after = datetime(2026, 6, 1, tzinfo=timezone.utc if aware_after else None)
    before = datetime(2026, 6, 2, tzinfo=None if aware_after else timezone.utc)
    filters = MetadataFilters(created_after=after, created_before=before)
    assert matches_metadata_filters({"created_at": "2026-06-01T12:00:00Z"}, filters)
    with pytest.raises(ValueError, match="created_after"):
        MetadataFilters(created_after=before, created_before=after)


@pytest.mark.parametrize("kind", ["dense", "lexical", "hybrid"])
@pytest.mark.parametrize("endpoint", ["/query", "/query/stream", "/chat"])
def test_api_returns_filtered_evidence_beyond_original_candidate_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, endpoint: str
) -> None:
    sink = InMemoryTelemetrySink()
    agent = FeedbackInsightAgent(
        build_crowded_retriever(kind, {"tenant_id": "cobalt"}),
        DeterministicLLM(),
        telemetry=Telemetry(sink=sink),
    )
    monkeypatch.setattr(
        api_module, "AgentSnapshots", lambda settings: SimpleNamespace(get=lambda: agent)
    )
    for store_name in ("CONVERSATION", "JOB", "REPORT", "HUMAN_FEEDBACK", "ACTIVE_LEARNING"):
        monkeypatch.setenv(f"FEEDBACK_AGENT_{store_name}_STORE_PATH", str(tmp_path / store_name))
    monkeypatch.setenv("FEEDBACK_AGENT_API_AUTH_ENABLED", "false")
    request = {
        "message" if endpoint == "/chat" else "question": QUESTION,
        "top_k": 1,
        "tenant_id": "acme",
        "created_after": "2026-06-01T00:00:00",
        "created_before": "2026-06-02T00:00:00Z",
    }
    with TestClient(api_module.create_app()) as client:
        response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    if endpoint == "/query/stream":
        metadata_event = next(
            block for block in response.text.split("\n\n") if block.startswith("event: metadata\n")
        )
        result = json.loads(metadata_event.split("\ndata: ", 1)[1])
    else:
        result = response.json()["result"]
    assert [citation["document_id"] for citation in result["citations"]] == ["wanted"]
    retrieval_event = next(event for event in sink.events if event.name == "retrieval_finished")
    assert retrieval_event.metadata["metadata_filters_applied_before_limit"] is True
