from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.retrieval import QueryEngine
from feedback_intelligence_agent.schemas import DocumentChunk, MetadataFilters, SearchResult
from feedback_intelligence_agent.tools import (
    IssueClusterTool,
    SentimentSummaryTool,
    TicketDraftTool,
    ToolError,
    build_default_tools,
)
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


@pytest.fixture()
def tenant_chunks() -> list[DocumentChunk]:
    rows = [
        ("acme:shared", "acme", "enterprise", "support_ticket", 1, "2026-01-01T10:00:00Z"),
        ("acme:second", "acme", "startup", "app_review", 5, "2026-02-01T10:00:00Z"),
        ("acme:third", "acme", "enterprise", "support_ticket", 3, "2026-01-02T10:00:00Z"),
        ("cobalt:shared", "cobalt", "cobalt_segment", "support_ticket", 5, "2026-01-01T10:00:00Z"),
        ("legacy", None, "enterprise", "nps_survey", 2, "2026-01-01T10:00:00Z"),
    ]
    chunks = [
        DocumentChunk(
            chunk_id=f"{source_id}::chunk-0",
            source_id=source_id,
            text=f"{tenant or 'Legacy'} onboarding checklist needs attention.",
            metadata={
                **({"tenant_id": tenant} if tenant else {}),
                "customer_segment": segment,
                "channel": channel,
                "rating": rating,
                "created_at": created_at,
            },
        )
        for source_id, tenant, segment, channel, rating, created_at in rows
    ]
    chunks.append(chunks[0].model_copy(update={"chunk_id": "acme:shared::chunk-1"}))
    return chunks


@pytest.mark.parametrize("tool_class", [SentimentSummaryTool, IssueClusterTool])
@pytest.mark.parametrize(
    ("filters", "expected_ids", "average"),
    [
        ({"tenant_id": " ACME "}, {"acme:shared", "acme:second", "acme:third"}, 3.0),
        ({"tenant_id": "acme", "customer_segment": "STARTUP"}, {"acme:second"}, 5.0),
        ({"tenant_id": "acme", "channel": "support_ticket"}, {"acme:shared", "acme:third"}, 2.0),
        (
            {"tenant_id": "acme", "min_rating": 3, "max_rating": 5},
            {"acme:second", "acme:third"},
            4.0,
        ),
        (
            {
                "tenant_id": "acme",
                "customer_segment": "enterprise",
                "channel": "support_ticket",
                "min_rating": 1,
                "max_rating": 1,
                "created_after": "2026-01-01T12:00:00+02:00",
                "created_before": "2026-01-01T10:00:00",
            },
            {"acme:shared"},
            1.0,
        ),
        (
            {"tenant_id": "acme", "created_after": "2026-01-02T10:00:00Z"},
            {"acme:second", "acme:third"},
            4.0,
        ),
        ({"tenant_id": "missing"}, set(), 0.0),
        ({"tenant_id": "default"}, {"legacy"}, 2.0),
    ],
)
def test_dataset_tools_filter_before_aggregation(
    tenant_chunks: list[DocumentChunk],
    tool_class: type[SentimentSummaryTool] | type[IssueClusterTool],
    filters: dict[str, object],
    expected_ids: set[str],
    average: float,
) -> None:
    output = tool_class(tenant_chunks).execute({"filters": filters}).model_dump()
    assert output["total_records"] == len(expected_ids)
    assert "cobalt" not in json.dumps(output).lower()
    if tool_class is SentimentSummaryTool:
        assert output["average_rating"] == average
        assert sum(output["rating_distribution"].values()) == len(expected_ids)
    else:
        actual_ids = {item for cluster in output["clusters"] for item in cluster["document_ids"]}
        assert actual_ids == expected_ids
        assert all(cluster["count"] == len(expected_ids) for cluster in output["clusters"])


@pytest.mark.parametrize("tool_class", [SentimentSummaryTool, IssueClusterTool])
def test_dataset_tools_validate_filter_ranges(
    tenant_chunks: list[DocumentChunk],
    tool_class: type[SentimentSummaryTool] | type[IssueClusterTool],
) -> None:
    with pytest.raises(ToolError, match="min_rating"):
        tool_class(tenant_chunks).execute({"filters": {"min_rating": 5, "max_rating": 1}})


def test_legacy_sentiment_controls_intersect_request_filters(
    tenant_chunks: list[DocumentChunk],
) -> None:
    output = SentimentSummaryTool(tenant_chunks).execute(
        {"filters": {"tenant_id": "acme"}, "segment": "ENTERPRISE", "channel": "SUPPORT_TICKET"}
    )
    assert output.model_dump()["total_records"] == 2
    assert output.model_dump()["average_rating"] == 2.0


def test_cluster_minimum_is_applied_after_tenant_filtering(
    tenant_chunks: list[DocumentChunk],
) -> None:
    output = IssueClusterTool(tenant_chunks).execute(
        {"filters": {"tenant_id": "acme"}, "min_count": 4, "max_clusters": 1}
    )
    assert output.model_dump() == {"total_records": 3, "clusters": []}


@pytest.mark.parametrize("tool_class", [SentimentSummaryTool, IssueClusterTool])
def test_dataset_filters_exclude_invalid_dates_before_document_deduplication(
    tenant_chunks: list[DocumentChunk],
    tool_class: type[SentimentSummaryTool] | type[IssueClusterTool],
) -> None:
    valid = tenant_chunks[0]
    invalid = valid.model_copy(
        update={"chunk_id": "invalid", "metadata": {**valid.metadata, "created_at": "invalid"}}
    )
    missing = tenant_chunks[1].model_copy(update={"metadata": {"tenant_id": "acme", "rating": 5}})
    output = tool_class([invalid, missing, valid]).execute(
        {"filters": {"tenant_id": "acme", "created_after": "2026-01-01T10:00:00Z"}}
    )
    assert output.model_dump()["total_records"] == 1


@pytest.mark.parametrize(
    "question",
    [
        "What is the sentiment distribution?",
        "What are the most common issues?",
        "Use the sentiment_summary tool for cobalt",
        "Use the issue_cluster tool for cobalt",
    ],
)
def test_agent_tools_use_request_scope_without_limiting_to_top_k_or_retaining_scope(
    tenant_chunks: list[DocumentChunk], question: str
) -> None:
    model = HashingEmbeddingModel(dim=128)
    store = InMemoryVectorStore(dim=128)
    store.add(tenant_chunks, model.embed([chunk.text for chunk in tenant_chunks]))
    agent = FeedbackInsightAgent(
        query_engine=QueryEngine(embedding_model=model, vector_store=store),
        llm=DeterministicLLM(),
        tools=build_default_tools(tenant_chunks),
    )
    for tenant, count in [("acme", 3), ("cobalt", 1), (None, 5), ("missing", 0)]:
        answer = agent.answer(question, top_k=1, filters=MetadataFilters(tenant_id=tenant))
        assert answer.tool_run is not None and answer.tool_run.status == "ok"
        assert answer.tool_run.output["total_records"] == count
        assert len(answer.citations) <= 1
        if tenant == "acme":
            assert "cobalt" not in json.dumps(answer.tool_run.output).lower()


def test_ticket_payload_filters_before_deduplication_and_evidence_limit(
    tenant_chunks: list[DocumentChunk],
) -> None:
    results = [SearchResult(chunk=chunk, score=0.9) for chunk in reversed(tenant_chunks)]
    payload = TicketDraftTool().build_payload(
        "Draft a ticket for onboarding",
        results,
        filters=MetadataFilters(
            tenant_id="acme", created_after=datetime(2026, 1, 2, tzinfo=timezone.utc)
        ),
    )
    assert {item["document_id"] for item in payload["evidence"]} == {"acme:second", "acme:third"}
