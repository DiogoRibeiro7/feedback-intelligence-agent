"""Offline evaluation for retrieval quality and answer quality.

The module exposes small, pure metric functions (precision@k, recall@k,
reciprocal rank, keyword coverage, groundedness, refusal correctness) plus an
orchestration layer that runs every evaluation case through the retriever and
the agent and aggregates results into a typed :class:`EvaluationReport`.

All metrics are deterministic when used with the local LLM provider, which
makes the report suitable for CI regression gates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.retrieval import Retriever
from ai_engineering_showcase.schemas import AgentAnswer, EvaluationCase

REFUSAL_MARKERS = (
    "could not find enough evidence",
    "cannot answer",
    "can't answer",
    "not enough evidence",
    "no relevant evidence",
    "unable to answer",
)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "was",
    "were",
    "are",
    "our",
    "but",
    "not",
    "from",
    "into",
    "have",
    "has",
    "had",
    "answer",
    "grounded",
    "feedback",
    "sources",
    "retrieved",
    "points",
    "repeated",
    "friction",
    "strongest",
    "signal",
    "around",
}


def _dedupe(items: list[str]) -> list[str]:
    """Return items without duplicates, preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant.

    Duplicate retrieved IDs are collapsed before scoring so a retriever cannot
    inflate precision by returning the same relevant document several times.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    unique_retrieved = _dedupe(retrieved_ids)[:k]
    if not unique_retrieved:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in unique_retrieved if doc_id in relevant)
    return hits / len(unique_retrieved)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the relevant documents found in the top-k results.

    Returns 0.0 when there are no relevant documents, so unanswerable cases
    do not artificially inflate recall.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    unique_retrieved = _dedupe(retrieved_ids)[:k]
    hits = sum(1 for doc_id in unique_retrieved if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal of the rank of the first relevant document, 0.0 if none."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(_dedupe(retrieved_ids), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def context_hit(retrieved_ids: list[str], relevant_ids: list[str]) -> bool:
    """True if at least one relevant document appears anywhere in the results."""
    return bool(set(retrieved_ids).intersection(relevant_ids))


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive).

    Returns 1.0 when no keywords are expected: an empty expectation set
    cannot be violated.
    """
    if not expected_keywords:
        return 1.0
    lower_answer = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lower_answer)
    return hits / len(expected_keywords)


def groundedness_score(answer: str, context_texts: list[str]) -> float:
    """Fraction of answer sentences whose content words appear in the context.

    A sentence counts as supported when at least half of its content tokens
    occur in the retrieved context. This is a lexical proxy for claim support:
    cheap, deterministic, and good at catching answers that drift away from
    the retrieved evidence.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    if not context_texts:
        return 0.0
    context_tokens = _content_tokens(" ".join(context_texts))
    supported = 0
    scored = 0
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        if not tokens:
            continue
        scored += 1
        overlap = len(tokens.intersection(context_tokens)) / len(tokens)
        if overlap >= 0.5:
            supported += 1
    if scored == 0:
        return 0.0
    return supported / scored


def is_refusal(answer: str) -> bool:
    """Detect whether an answer is a refusal / abstention."""
    lower_answer = answer.lower()
    return any(marker in lower_answer for marker in REFUSAL_MARKERS)


def refusal_correct(answer: str, *, is_answerable: bool) -> bool:
    """True when refusal behaviour matches answerability.

    Unanswerable questions should be refused; answerable questions should not.
    """
    refused = is_refusal(answer)
    return refused != is_answerable


def _content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return {token for token in tokens if token not in _STOPWORDS}


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval metrics across answerable cases."""

    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    context_hit_rate: float = Field(ge=0.0, le=1.0)
    evaluated_cases: int = Field(ge=0)


class AnswerMetrics(BaseModel):
    """Aggregate answer-quality metrics."""

    keyword_coverage: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    refusal_correctness: float = Field(ge=0.0, le=1.0)
    citation_alignment: float = Field(ge=0.0, le=1.0)
    evaluated_cases: int = Field(ge=0)


class CaseResult(BaseModel):
    """Per-case evaluation detail kept in the report for debugging."""

    question: str
    is_answerable: bool
    retrieved_document_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    context_hit: bool
    keyword_coverage: float
    groundedness: float
    citation_aligned: bool
    refused: bool
    refusal_correct: bool


class EvaluationReport(BaseModel):
    """Full evaluation report: aggregates plus per-case breakdown."""

    top_k: int = Field(ge=1)
    total_cases: int = Field(ge=0)
    retrieval: RetrievalMetrics
    answers: AnswerMetrics
    cases: list[CaseResult] = Field(default_factory=list)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load JSONL evaluation cases, reporting the line number on failure."""
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


def evaluate_case_detailed(
    query_engine: Retriever,
    agent: FeedbackInsightAgent,
    case: EvaluationCase,
    *,
    top_k: int = 4,
) -> tuple[CaseResult, AgentAnswer]:
    """Run one evaluation case and return both the metrics and the raw answer.

    Useful for callers (such as the experiment runner) that need the generated
    answer text in addition to the per-case metrics.
    """
    results = query_engine.search(case.question, top_k=top_k)
    retrieved_ids = [result.chunk.source_id for result in results]
    context_texts = [result.chunk.text for result in results]
    answer = agent.answer(case.question, top_k=top_k)
    refused = is_refusal(answer.answer)
    cited_ids = {citation.document_id for citation in answer.citations}
    case_result = CaseResult(
        question=case.question,
        is_answerable=case.is_answerable,
        retrieved_document_ids=_dedupe(retrieved_ids),
        precision_at_k=round(precision_at_k(retrieved_ids, case.relevant_document_ids, top_k), 4),
        recall_at_k=round(recall_at_k(retrieved_ids, case.relevant_document_ids, top_k), 4),
        reciprocal_rank=round(reciprocal_rank(retrieved_ids, case.relevant_document_ids), 4),
        context_hit=context_hit(retrieved_ids, case.relevant_document_ids),
        keyword_coverage=round(keyword_coverage(answer.answer, case.expected_keywords), 4),
        groundedness=round(groundedness_score(answer.answer, context_texts), 4),
        citation_aligned=bool(cited_ids.intersection(case.relevant_document_ids)),
        refused=refused,
        refusal_correct=refused != case.is_answerable,
    )
    return case_result, answer


def evaluate_case(
    query_engine: Retriever,
    agent: FeedbackInsightAgent,
    case: EvaluationCase,
    *,
    top_k: int = 4,
) -> CaseResult:
    """Run one evaluation case through retrieval and the agent."""
    case_result, _ = evaluate_case_detailed(query_engine, agent, case, top_k=top_k)
    return case_result


def evaluate_system(
    query_engine: Retriever,
    agent: FeedbackInsightAgent,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
) -> EvaluationReport:
    """Evaluate retrieval and answer quality and aggregate a typed report.

    Retrieval metrics are averaged over answerable cases only, because
    unanswerable cases have no relevant documents to retrieve. Answer metrics
    are averaged over all cases.
    """
    case_results = [evaluate_case(query_engine, agent, case, top_k=top_k) for case in cases]
    return aggregate_report(case_results, top_k=top_k)


def aggregate_report(case_results: list[CaseResult], *, top_k: int = 4) -> EvaluationReport:
    """Aggregate per-case results into a typed :class:`EvaluationReport`.

    Retrieval and citation metrics are averaged over answerable cases only;
    answer metrics are averaged over all cases.
    """
    answerable = [result for result in case_results if result.is_answerable]
    retrieval = RetrievalMetrics(
        precision_at_k=_mean([result.precision_at_k for result in answerable]),
        recall_at_k=_mean([result.recall_at_k for result in answerable]),
        mean_reciprocal_rank=_mean([result.reciprocal_rank for result in answerable]),
        context_hit_rate=_mean([1.0 if result.context_hit else 0.0 for result in answerable]),
        evaluated_cases=len(answerable),
    )
    answers = AnswerMetrics(
        keyword_coverage=_mean([result.keyword_coverage for result in case_results]),
        groundedness=_mean([result.groundedness for result in case_results]),
        refusal_correctness=_mean(
            [1.0 if result.refusal_correct else 0.0 for result in case_results]
        ),
        citation_alignment=_mean(
            [1.0 if result.citation_aligned else 0.0 for result in answerable]
        ),
        evaluated_cases=len(case_results),
    )
    return EvaluationReport(
        top_k=top_k,
        total_cases=len(case_results),
        retrieval=retrieval,
        answers=answers,
        cases=case_results,
    )


def _mean(values: list[float]) -> float:
    """Mean rounded to 4 decimals; 0.0 for an empty list."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
