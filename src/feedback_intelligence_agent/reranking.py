"""Local candidate reranking for retrieved feedback chunks."""

from __future__ import annotations

import re
from typing import Protocol

from feedback_intelligence_agent.schemas import SearchResult


class Reranker(Protocol):
    """Protocol for candidate rerankers."""

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
        *,
        top_k: int,
        route_keywords: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        """Return candidates ordered by judged relevance."""
        ...


class DeterministicJudgeReranker:
    """Deterministic local relevance judge for reranking retrieval candidates.

    This is a lightweight stand-in for a cross-encoder or LLM judge: it scores
    each candidate using the original retrieval score plus transparent lexical,
    route, segment, and low-rating relevance signals. It keeps CI and demos fully
    local while making the reranking step explicit and replaceable.
    """

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
        *,
        top_k: int,
        route_keywords: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        """Score and rank candidate chunks with deterministic relevance signals."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            SearchResult(
                chunk=result.chunk,
                score=self.score(question, result, route_keywords=route_keywords),
            )
            for result in results
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def score(
        self,
        question: str,
        result: SearchResult,
        *,
        route_keywords: tuple[str, ...] = (),
    ) -> float:
        """Return a deterministic relevance score for one retrieved chunk."""
        question_terms = self._tokens(question)
        metadata = result.chunk.metadata
        searchable = " ".join(
            [
                result.chunk.text,
                str(metadata.get("customer_segment", "")),
                str(metadata.get("channel", "")),
                str(metadata.get("rating", "")),
            ]
        ).lower()

        overlap = 0.0
        if question_terms:
            overlap = sum(1 for term in question_terms if term in searchable) / len(question_terms)

        route_hits = sum(1 for keyword in route_keywords if keyword in searchable)
        route_score = min(route_hits / 3.0, 1.0)

        segment_score = 0.0
        segment = str(metadata.get("customer_segment", "")).lower()
        if segment and segment.replace("_", " ") in question.lower():
            segment_score = 1.0

        low_rating_score = 0.0
        risk_terms = {"unhappy", "complain", "complaints", "risk", "churn", "bad", "poor"}
        if question_terms.intersection(risk_terms) and self._metadata_rating(metadata) <= 2:
            low_rating_score = 1.0

        return round(
            result.score
            + 0.22 * overlap
            + 0.18 * route_score
            + 0.08 * segment_score
            + 0.06 * low_rating_score,
            6,
        )

    def _tokens(self, text: str) -> set[str]:
        """Return compact query tokens used by reranking."""
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "are",
            "the",
            "for",
            "with",
            "and",
            "from",
            "should",
            "could",
            "would",
            "customers",
            "customer",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
            if token not in stopwords
        }

    def _metadata_rating(self, metadata: dict[str, object]) -> int:
        """Read a metadata rating, defaulting to the neutral upper bound."""
        value = metadata.get("rating", 5)
        if isinstance(value, int):
            return value
        if not isinstance(value, str):
            return 5
        try:
            return int(value)
        except ValueError:
            return 5
