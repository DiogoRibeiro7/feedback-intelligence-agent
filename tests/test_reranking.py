from __future__ import annotations

import pytest

from feedback_intelligence_agent.reranking import DeterministicJudgeReranker
from feedback_intelligence_agent.schemas import DocumentChunk, SearchResult


def make_result(
    suffix: str,
    text: str,
    *,
    score: float,
    rating: int = 5,
    segment: str = "enterprise",
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=f"c-{suffix}",
            source_id=f"fb-{suffix}",
            text=text,
            metadata={"rating": rating, "customer_segment": segment},
        ),
        score=score,
    )


def test_deterministic_judge_reranker_promotes_route_relevant_candidates() -> None:
    reranker = DeterministicJudgeReranker()
    results = [
        make_result(
            "pricing",
            "Pricing renewal was hard to explain to finance.",
            score=0.52,
            rating=5,
        ),
        make_result(
            "onboarding",
            "Onboarding checklist was unclear and setup took too long.",
            score=0.42,
            rating=2,
        ),
    ]

    reranked = reranker.rerank(
        "Why are enterprise customers unhappy with onboarding?",
        results,
        top_k=2,
        route_keywords=("onboarding", "setup", "implementation", "checklist"),
    )

    assert reranked[0].chunk.source_id == "fb-onboarding"
    assert reranked[0].score > reranked[1].score


def test_deterministic_judge_reranker_respects_top_k() -> None:
    reranker = DeterministicJudgeReranker()
    results = [
        make_result("a", "Onboarding checklist was unclear.", score=0.1),
        make_result("b", "Setup took too long.", score=0.1),
        make_result("c", "Pricing renewal was difficult.", score=0.1),
    ]

    reranked = reranker.rerank(
        "What onboarding setup issues were reported?",
        results,
        top_k=2,
        route_keywords=("onboarding", "setup"),
    )

    assert len(reranked) == 2
    assert {result.chunk.source_id for result in reranked} == {"fb-a", "fb-b"}


def test_deterministic_judge_reranker_rejects_invalid_top_k() -> None:
    reranker = DeterministicJudgeReranker()
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("question", [], top_k=0)
