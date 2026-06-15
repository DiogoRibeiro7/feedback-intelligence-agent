"""Retrieval helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_engineering_showcase.embeddings import EmbeddingModel
from ai_engineering_showcase.schemas import DocumentChunk, SearchResult
from ai_engineering_showcase.vector_store import VectorStore


class Retriever(Protocol):
    """Protocol implemented by all retrievers (dense, lexical, hybrid)."""

    def search(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        """Return the most relevant chunks for a natural-language question."""
        ...


class QueryEngine:
    """Embed user questions and retrieve relevant document chunks."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        """Wire the embedding model used for queries to the vector store."""
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        """Search for chunks relevant to a natural-language question."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        query_vector = self.embedding_model.embed([question])[0]
        return self.vector_store.search(query_vector, top_k=top_k)


class HybridRetriever:
    """Combine dense and lexical retrieval with weighted, normalized scores.

    Each underlying retriever is queried for an enlarged candidate pool. Scores
    are min-max normalized per result list so the two scales are comparable,
    then combined as a weighted sum. Documents appearing in both lists are
    de-duplicated by chunk ID and receive contributions from both retrievers.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        dense_weight: float = 0.6,
        lexical_weight: float = 0.4,
    ) -> None:
        """Validate and normalize the retriever weights."""
        if dense_weight < 0.0 or lexical_weight < 0.0:
            raise ValueError("weights must be non-negative")
        total = dense_weight + lexical_weight
        if total <= 0.0:
            raise ValueError("at least one weight must be positive")
        self.dense = dense
        self.lexical = lexical
        self.dense_weight = dense_weight / total
        self.lexical_weight = lexical_weight / total

    def search(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        """Return de-duplicated chunks ranked by the combined hybrid score."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_k = max(top_k * 2, top_k)
        contributions = (
            (self.dense.search(question, top_k=candidate_k), self.dense_weight),
            (self.lexical.search(question, top_k=candidate_k), self.lexical_weight),
        )
        combined_scores: dict[str, float] = {}
        chunks_by_id: dict[str, DocumentChunk] = {}
        for results, weight in contributions:
            normalized = min_max_normalize([result.score for result in results])
            for result, score in zip(results, normalized, strict=True):
                chunk_id = result.chunk.chunk_id
                chunks_by_id.setdefault(chunk_id, result.chunk)
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + weight * score

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk=chunks_by_id[chunk_id], score=round(score, 6))
            for chunk_id, score in ranked[:top_k]
        ]


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    """Scale scores to the [0, 1] range; constant lists map to all ones."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]
