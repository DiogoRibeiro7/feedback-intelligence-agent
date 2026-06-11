"""Offline evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.schemas import EvaluationCase


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval metrics."""

    precision_at_k: float
    mean_reciprocal_rank: float
    evaluated_cases: int


class AnswerQualityMetrics(BaseModel):
    """Simple answer quality checks."""

    citation_coverage: float = Field(ge=0.0, le=1.0)
    grounded_answer_rate: float = Field(ge=0.0, le=1.0)
    evaluated_cases: int


class EvaluationReport(BaseModel):
    """Full evaluation report."""

    retrieval: RetrievalMetrics
    answer_quality: AnswerQualityMetrics


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load JSONL evaluation cases."""
    input_path = Path(path)
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - include line number in user-facing error.
            raise ValueError(f"Invalid evaluation case at line {line_number}: {exc}") from exc
    return cases


def evaluate_retrieval(
    query_engine: QueryEngine,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
) -> RetrievalMetrics:
    """Evaluate retrieval using precision@k and MRR."""
    if not cases:
        return RetrievalMetrics(precision_at_k=0.0, mean_reciprocal_rank=0.0, evaluated_cases=0)

    precision_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in cases:
        relevant = set(case.relevant_source_ids)
        results = query_engine.search(case.question, top_k=top_k)
        retrieved = [result.chunk.source_id for result in results]
        hits = sum(1 for source_id in retrieved if source_id in relevant)
        precision_sum += hits / max(top_k, 1)

        reciprocal_rank = 0.0
        for rank, source_id in enumerate(retrieved, start=1):
            if source_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_rank_sum += reciprocal_rank

    case_count = len(cases)
    return RetrievalMetrics(
        precision_at_k=round(precision_sum / case_count, 4),
        mean_reciprocal_rank=round(reciprocal_rank_sum / case_count, 4),
        evaluated_cases=case_count,
    )


def evaluate_answer_quality(
    agent: FeedbackInsightAgent,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
) -> AnswerQualityMetrics:
    """Evaluate citations and simple grounding for generated answers."""
    if not cases:
        return AnswerQualityMetrics(
            citation_coverage=0.0, grounded_answer_rate=0.0, evaluated_cases=0
        )

    citation_hits = 0
    grounded_hits = 0

    for case in cases:
        answer = agent.answer(case.question, top_k=top_k)
        cited_sources = {citation.source_id for citation in answer.citations}
        if cited_sources.intersection(case.relevant_source_ids):
            citation_hits += 1
        if answer.citations and any(citation.quote for citation in answer.citations):
            grounded_hits += 1

    case_count = len(cases)
    return AnswerQualityMetrics(
        citation_coverage=round(citation_hits / case_count, 4),
        grounded_answer_rate=round(grounded_hits / case_count, 4),
        evaluated_cases=case_count,
    )


def evaluate_system(
    query_engine: QueryEngine,
    agent: FeedbackInsightAgent,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
) -> EvaluationReport:
    """Run retrieval and answer-quality evaluation."""
    return EvaluationReport(
        retrieval=evaluate_retrieval(query_engine, cases, top_k=top_k),
        answer_quality=evaluate_answer_quality(agent, cases, top_k=top_k),
    )
