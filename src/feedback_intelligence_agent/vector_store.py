"""A small vector store with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrapping
from pydantic import BaseModel, Field, ValidationError

from feedback_intelligence_agent.schemas import DocumentChunk, SearchResult
from feedback_intelligence_agent.source_ids import SourceIdMigrationError, migrate_legacy_chunks


@runtime_checkable
class VectorStore(Protocol):
    """Interface shared by every vector store backend.

    Both the default :class:`InMemoryVectorStore` (JSON persistence) and the
    optional Qdrant-backed store satisfy this protocol, so the retrieval layer
    can treat them interchangeably. Vectors are assumed to be L2-normalised and
    scored by cosine similarity, with higher scores meaning more similar.
    """

    dim: int

    @property
    def size(self) -> int:
        """Number of indexed chunks."""

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return the indexed chunks."""

    def add(self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]) -> None:
        """Add chunks and their (L2-normalised) vectors to the store."""

    def search(
        self, query_vector: npt.NDArray[np.float64], *, top_k: int = 4
    ) -> list[SearchResult]:
        """Return the most similar chunks for a query vector, highest score first."""


class PersistedVectorStore(BaseModel):
    """Serializable representation of the vector store."""

    source_id_version: Literal[1, 2] = 1
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
        """Add chunks and their vectors to the store.

        Raises:
            ValueError: If vector shape or dimension does not match the chunks and store.
        """
        prepared_vectors = self._prepare_vectors(chunks, vectors)
        if not chunks:
            return

        combined_vectors = np.vstack([self._vectors, prepared_vectors])
        self._chunks.extend(chunks)
        self._vectors = combined_vectors

    def _prepare_vectors(
        self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Validate and convert incoming vectors before mutating store state."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with dim={self.dim}")
        return vectors.astype(np.float64, copy=False)

    def remove_source_ids(self, source_ids: set[str]) -> int:
        """Remove all chunks whose source document ID is in ``source_ids``.

        Returns the number of chunks removed. This supports incremental updates:
        a refreshed feedback record can replace every old chunk for the same
        ``feedback_id`` before the new chunks are appended.
        """
        if not source_ids or not self._chunks:
            return 0
        keep_indices = [
            index for index, chunk in enumerate(self._chunks) if chunk.source_id not in source_ids
        ]
        removed = len(self._chunks) - len(keep_indices)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[index] for index in keep_indices]
        self._vectors = self._vectors[keep_indices, :]
        return removed

    def upsert_sources(
        self,
        chunks: list[DocumentChunk],
        vectors: npt.NDArray[np.float64],
    ) -> int:
        """Replace existing source documents and append the new chunks.

        The replacement key is ``DocumentChunk.source_id``. All old chunks for
        any incoming source are replaced, which handles changed text that
        produces a different number of chunks. Invalid vectors leave the
        existing chunks and vectors unchanged.

        Returns:
            Number of chunks removed before adding the new chunks.

        Raises:
            ValueError: If vector shape or dimension does not match the chunks and store.
        """
        prepared_vectors = self._prepare_vectors(chunks, vectors)
        if not chunks:
            return 0
        source_ids = {chunk.source_id for chunk in chunks}
        keep_indices = [
            index for index, chunk in enumerate(self._chunks) if chunk.source_id not in source_ids
        ]
        replacement_chunks = [self._chunks[index] for index in keep_indices] + chunks
        replacement_vectors = np.vstack([self._vectors[keep_indices, :], prepared_vectors])
        removed = len(self._chunks) - len(keep_indices)
        self._chunks = replacement_chunks
        self._vectors = replacement_vectors
        return removed

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
        with wrapping(OSError, FileWriteError, path=str(output_path.parent)):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = PersistedVectorStore(
            source_id_version=2,
            dim=self.dim,
            chunks=self._chunks,
            vectors=self._vectors.tolist(),
        )
        with wrapping((OSError, UnicodeError), FileWriteError, path=str(output_path)):
            output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        """Load JSON and migrate legacy chunk IDs in memory without changing the file.

        Saving the returned store persists version-2 identifiers. Migration
        retains the original vectors and requires no embedding provider.

        Raises:
            DataLoadingError: If the payload or its legacy identities are invalid.
        """
        input_path = Path(path)
        with wrapping((OSError, UnicodeError), FileReadError, path=str(input_path)):
            contents = input_path.read_text(encoding="utf-8")
        with wrapping(
            (ValidationError, SourceIdMigrationError), DataLoadingError, source=str(input_path)
        ):
            payload = PersistedVectorStore.model_validate_json(contents)
            chunks = (
                migrate_legacy_chunks(payload.chunks)
                if payload.source_id_version == 1
                else payload.chunks
            )
        store = cls(dim=payload.dim)
        if chunks:
            vectors = np.asarray(payload.vectors, dtype=np.float64)
            store.add(chunks, vectors)
        return store

    def to_json(self) -> str:
        """Return a compact JSON representation for debugging."""
        payload = {
            "dim": self.dim,
            "size": self.size,
            "source_ids": sorted({chunk.source_id for chunk in self._chunks}),
        }
        return json.dumps(payload, sort_keys=True)
