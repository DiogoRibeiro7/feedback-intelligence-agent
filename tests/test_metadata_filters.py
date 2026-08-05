from __future__ import annotations

from datetime import datetime

import pytest

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.retrieval import QueryEngine
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackChannel, MetadataFilters
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def build_filter_agent() -> FeedbackInsightAgent:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="enterprise-support-low",
            source_id="fb-enterprise-support-low",
            text="Feedback issue about onboarding checklist and setup delays.",
            metadata={
                "customer_segment": "enterprise",
                "channel": "support_ticket",
                "rating": 2,
                "created_at": "2026-01-08T09:20:00",
            },
        ),
        DocumentChunk(
            chunk_id="startup-review-high",
            source_id="fb-startup-review-high",
            text="Feedback issue about onboarding templates and fast setup.",
            metadata={
                "customer_segment": "startup",
                "channel": "app_review",
                "rating": 5,
                "created_at": "2026-01-20T11:00:00",
            },
        ),
        DocumentChunk(
            chunk_id="mid-market-community-neutral",
            source_id="fb-mid-market-community-neutral",
            text="Feedback issue about documentation and advanced automations.",
            metadata={
                "customer_segment": "mid_market",
                "channel": "community",
                "rating": 3,
                "created_at": "2026-02-18T18:40:00",
            },
        ),
        DocumentChunk(
            chunk_id="enterprise-nps-low",
            source_id="fb-enterprise-nps-low",
            text="Feedback issue about onboarding progress and fragmented setup.",
            metadata={
                "customer_segment": "enterprise",
                "channel": "nps_survey",
                "rating": 2,
                "created_at": "2026-03-03T08:45:00",
            },
        ),
        DocumentChunk(
            chunk_id="mid-market-nps-high",
            source_id="fb-mid-market-nps-high",
            text="Feedback issue about support responses and clearer ticket updates.",
            metadata={
                "customer_segment": "mid_market",
                "channel": "nps_survey",
                "rating": 4,
                "created_at": "2026-04-15T17:05:00",
            },
        ),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    return FeedbackInsightAgent(
        query_engine=QueryEngine(embedding_model=model, vector_store=store),
        llm=DeterministicLLM(),
    )


def test_filter_by_customer_segment() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions onboarding?",
        top_k=5,
        filters=MetadataFilters(customer_segment="enterprise"),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-enterprise-support-low",
        "fb-enterprise-nps-low",
    }


def test_filter_by_channel() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions onboarding?",
        top_k=5,
        filters=MetadataFilters(channel=FeedbackChannel.nps_survey),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-enterprise-nps-low",
        "fb-mid-market-nps-high",
    }


def test_filter_by_min_rating() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions setup?",
        top_k=5,
        filters=MetadataFilters(min_rating=4),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-startup-review-high",
        "fb-mid-market-nps-high",
    }


def test_filter_by_max_rating() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions setup?",
        top_k=5,
        filters=MetadataFilters(max_rating=2),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-enterprise-support-low",
        "fb-enterprise-nps-low",
    }


def test_filter_by_created_after() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions support?",
        top_k=5,
        filters=MetadataFilters(created_after=datetime(2026, 3, 1)),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-enterprise-nps-low",
        "fb-mid-market-nps-high",
    }


def test_filter_by_created_before() -> None:
    agent = build_filter_agent()
    answer = agent.answer(
        "Which feedback issue mentions setup?",
        top_k=5,
        filters=MetadataFilters(created_before=datetime(2026, 2, 1)),
    )
    assert answer.citations
    assert {citation.document_id for citation in answer.citations} == {
        "fb-enterprise-support-low",
        "fb-startup-review-high",
    }


def test_filter_ranges_are_validated() -> None:
    with pytest.raises(ValueError, match="min_rating"):
        MetadataFilters(min_rating=5, max_rating=2)
    with pytest.raises(ValueError, match="created_after"):
        MetadataFilters(
            created_after=datetime(2026, 4, 1),
            created_before=datetime(2026, 1, 1),
        )
