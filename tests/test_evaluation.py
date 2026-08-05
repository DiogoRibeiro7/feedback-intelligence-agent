from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.evaluation import (
    context_hit,
    evaluate_case,
    evaluate_system,
    groundedness_score,
    is_refusal,
    keyword_coverage,
    load_evaluation_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_correct,
)
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.retrieval import QueryEngine
from feedback_intelligence_agent.schemas import DocumentChunk, EvaluationCase
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def build_query_engine_and_agent(
    chunks: list[DocumentChunk] | None = None,
) -> tuple[QueryEngine, FeedbackInsightAgent]:
    model = HashingEmbeddingModel(dim=128)
    if chunks is None:
        chunks = [
            DocumentChunk(
                chunk_id="1", source_id="fb-a", text="export failed during reporting", metadata={}
            ),
            DocumentChunk(
                chunk_id="2", source_id="fb-b", text="onboarding setup checklist", metadata={}
            ),
        ]
    store = InMemoryVectorStore(dim=128)
    if chunks:
        store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    query_engine = QueryEngine(embedding_model=model, vector_store=store)
    agent = FeedbackInsightAgent(query_engine=query_engine, llm=DeterministicLLM())
    return query_engine, agent


# ---------------------------------------------------------------------------
# Retrieval metric functions
# ---------------------------------------------------------------------------


def test_precision_at_k_basic() -> None:
    assert precision_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(2 / 3)


def test_precision_at_k_no_retrieved_documents() -> None:
    assert precision_at_k([], ["a"], 4) == 0.0


def test_precision_at_k_ignores_duplicate_retrieved_documents() -> None:
    # Returning the same relevant document twice must not inflate precision.
    assert precision_at_k(["a", "a", "b"], ["a"], 3) == pytest.approx(1 / 2)


def test_precision_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)


def test_recall_at_k_basic() -> None:
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == pytest.approx(1 / 2)


def test_recall_at_k_empty_relevant_set_is_zero() -> None:
    assert recall_at_k(["a", "b"], [], 2) == 0.0


def test_recall_at_k_duplicates_count_once() -> None:
    assert recall_at_k(["a", "a"], ["a", "b"], 2) == pytest.approx(1 / 2)


def test_reciprocal_rank_first_hit_position() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == pytest.approx(1 / 2)


def test_reciprocal_rank_no_hit_is_zero() -> None:
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_skips_duplicates_when_ranking() -> None:
    # Duplicate "x" collapses, so "a" sits at rank 2 rather than rank 3.
    assert reciprocal_rank(["x", "x", "a"], ["a"]) == pytest.approx(1 / 2)


def test_context_hit() -> None:
    assert context_hit(["a", "b"], ["b"]) is True
    assert context_hit(["a"], ["b"]) is False
    assert context_hit([], ["b"]) is False


# ---------------------------------------------------------------------------
# Answer metric functions
# ---------------------------------------------------------------------------


def test_keyword_coverage_counts_case_insensitive_hits() -> None:
    assert keyword_coverage("Onboarding was SLOW.", ["onboarding", "slow", "export"]) == (
        pytest.approx(2 / 3)
    )


def test_keyword_coverage_empty_expected_keywords_is_full() -> None:
    assert keyword_coverage("any answer", []) == 1.0


def test_groundedness_fully_supported_answer() -> None:
    context = ["export failed during month-end reporting workflows"]
    assert groundedness_score("Export failed during reporting.", context) == 1.0


def test_groundedness_unsupported_answer_is_zero() -> None:
    context = ["export failed during reporting"]
    assert groundedness_score("Bananas turned purple yesterday evening.", context) == 0.0


def test_groundedness_empty_context_or_answer_is_zero() -> None:
    assert groundedness_score("Some claim here.", []) == 0.0
    assert groundedness_score("", ["export failed"]) == 0.0


def test_is_refusal_detects_deterministic_refusal_phrase() -> None:
    assert is_refusal("I could not find enough evidence to answer this question.") is True
    assert is_refusal("Exports fail during month-end reporting.") is False


def test_refusal_correct_matrix() -> None:
    refusal = "I could not find enough evidence to answer this question."
    answer = "Exports fail during reporting."
    assert refusal_correct(refusal, is_answerable=False) is True
    assert refusal_correct(answer, is_answerable=True) is True
    assert refusal_correct(refusal, is_answerable=True) is False
    assert refusal_correct(answer, is_answerable=False) is False


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def test_load_evaluation_cases_rich_format(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Why is onboarding slow?",
                "expected_keywords": ["onboarding"],
                "relevant_document_ids": ["fb-1"],
                "is_answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_evaluation_cases(path)
    assert cases[0].relevant_document_ids == ["fb-1"]
    assert cases[0].expected_keywords == ["onboarding"]
    assert cases[0].is_answerable is True


def test_load_evaluation_cases_accepts_legacy_alias(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "q", "relevant_source_ids": ["fb-1"]}\n',
        encoding="utf-8",
    )
    cases = load_evaluation_cases(path)
    assert cases[0].relevant_document_ids == ["fb-1"]
    assert cases[0].is_answerable is True
    assert cases[0].expected_keywords == []


def test_load_evaluation_cases_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "ok", "relevant_document_ids": []}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_evaluation_cases(path)


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------


def test_evaluate_case_answerable_question() -> None:
    query_engine, agent = build_query_engine_and_agent()
    case = EvaluationCase(
        question="reporting export failed",
        expected_keywords=["export"],
        relevant_document_ids=["fb-a"],
    )
    result = evaluate_case(query_engine, agent, case, top_k=1)
    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.context_hit is True
    assert result.keyword_coverage == 1.0
    assert 0.0 <= result.evidence_overlap <= 1.0
    assert result.hallucination_risk in {"low", "medium", "high"}
    assert result.refused is False
    assert result.refusal_correct is True


def test_evaluate_case_with_empty_store_refuses() -> None:
    # No retrieved documents: the agent refuses, which is correct for an
    # unanswerable question.
    query_engine, agent = build_query_engine_and_agent(chunks=[])
    case = EvaluationCase(question="anything at all", is_answerable=False)
    result = evaluate_case(query_engine, agent, case, top_k=2)
    assert result.retrieved_document_ids == []
    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.context_hit is False
    assert result.refused is True
    assert result.refusal_correct is True


def test_evaluate_system_aggregates_and_excludes_unanswerable_from_retrieval() -> None:
    query_engine, agent = build_query_engine_and_agent()
    cases = [
        EvaluationCase(
            question="onboarding checklist",
            expected_keywords=["onboarding"],
            relevant_document_ids=["fb-b"],
        ),
        EvaluationCase(question="weather on the moon", is_answerable=False),
    ]
    report = evaluate_system(query_engine, agent, cases, top_k=1)
    assert report.total_cases == 2
    assert report.retrieval.evaluated_cases == 1
    assert report.answers.evaluated_cases == 2
    assert report.retrieval.precision_at_k == 1.0
    assert report.retrieval.recall_at_k == 1.0
    assert report.retrieval.mean_reciprocal_rank == 1.0
    assert report.retrieval.context_hit_rate == 1.0
    assert 0.0 <= report.answers.groundedness <= 1.0
    assert 0.0 <= report.answers.evidence_overlap <= 1.0
    assert 0.0 <= report.answers.hallucination_rate <= 1.0
    assert 0.0 <= report.answers.judge_supported_rate <= 1.0
    assert 0.0 <= report.answers.refusal_correctness <= 1.0
    assert len(report.cases) == 2


def test_evaluate_system_empty_cases_returns_zeroed_report() -> None:
    query_engine, agent = build_query_engine_and_agent()
    report = evaluate_system(query_engine, agent, [], top_k=2)
    assert report.total_cases == 0
    assert report.retrieval.precision_at_k == 0.0
    assert report.answers.keyword_coverage == 0.0
    assert report.cases == []


def test_evaluate_system_is_deterministic() -> None:
    query_engine, agent = build_query_engine_and_agent()
    cases = [
        EvaluationCase(
            question="reporting export",
            expected_keywords=["export"],
            relevant_document_ids=["fb-a"],
        )
    ]
    first = evaluate_system(query_engine, agent, cases, top_k=2)
    second = evaluate_system(query_engine, agent, cases, top_k=2)
    assert first.model_dump() == second.model_dump()
