"""A small vector store with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from ai_engineering_showcase.schemas import DocumentChunk, SearchResult


class PersistedVectorStore(BaseModel):
    """Serializable representation of the vector store."""

    dim: int
    chunks: list[DocumentChunk]
    vectors: list[list[float]] = Field(default_factory=list)


class InMemoryVectorStore:
    """Cosine-similarity vector store.

    Vectors are assumed to be L2-normalised. The embedding model in this project
    returns normalised vectors, so search can use a simple dot product.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._chunks: list[DocumentChunk] = []
        self._vectors = np.empty((0, dim), dtype=np.float64)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return indexed chunks."""
        return list(self._chunks)

    def add(self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]) -> None:
        """Add chunks and their vectors to the store."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with dim={self.dim}")
        if not chunks:
            return

        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float64, copy=False)])

    def search(
        self, query_vector: npt.NDArray[np.float64], *, top_k: int = 4
    ) -> list[SearchResult]:
        """Return the most similar chunks for a query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"expected query vector with dim={self.dim}")
        if self.size == 0:
            return []

        scores = self._vectors @ query_vector
        candidate_count = min(top_k, self.size)
        candidate_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        return [
            SearchResult(chunk=self._chunks[int(index)], score=float(scores[int(index)]))
            for index in sorted_indices
        ]

    def save(self, path: str | Path) -> None:
        """Persist the vector store as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = PersistedVectorStore(
            dim=self.dim,
            chunks=self._chunks,
            vectors=self._vectors.tolist(),
        )
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        """Load a vector store from JSON."""
        input_path = Path(path)
        payload = PersistedVectorStore.model_validate_json(input_path.read_text(encoding="utf-8"))
        store = cls(dim=payload.dim)
        if payload.chunks:
            vectors = np.asarray(payload.vectors, dtype=np.float64)
            store.add(payload.chunks, vectors)
        return store

    def to_json(self) -> str:
        """Return a compact JSON representation for debugging."""
        payload = {
            "dim": self.dim,
            "size": self.size,
            "source_ids": sorted({chunk.source_id for chunk in self._chunks}),
        }
        return json.dumps(payload, sort_keys=True)
