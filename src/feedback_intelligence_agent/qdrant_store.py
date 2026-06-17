"""Optional Qdrant-backed vector store.

The store mirrors the semantics of the default ``InMemoryVectorStore`` but
persists vectors in a running Qdrant instance instead of a local JSON file.
It satisfies the same ``VectorStore`` protocol, so the retrieval layer treats both
stores identically.

``qdrant-client`` is an *optional* dependency (poetry extra ``qdrant``) and is
imported lazily, with an actionable error if it is not installed. This keeps the
default install lean and the deterministic local path free of any external
service.
"""

from __future__ import annotations

import uuid
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from feedback_intelligence_agent.schemas import DocumentChunk, SearchResult

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking.
    from qdrant_client import QdrantClient


class QdrantStoreError(RuntimeError):
    """Raised when the Qdrant store cannot be configured or reached."""


def _import_qdrant_client() -> ModuleType:
    """Import the optional ``qdrant_client`` package, with an actionable error."""
    try:
        import qdrant_client
    except ImportError as exc:
        raise QdrantStoreError(
            "The 'qdrant-client' package is required for the Qdrant vector store. "
            "Install it with: poetry install --extras qdrant "
            "(or: pip install qdrant-client)."
        ) from exc
    module: ModuleType = qdrant_client
    return module


# Deterministic namespace so a given chunk_id always maps to the same point ID.
_POINT_NAMESPACE = uuid.UUID("a3f1c2d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d")


def _chunk_to_point_id(chunk_id: str) -> str:
    """Map a stable chunk identifier to a deterministic Qdrant point UUID."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Cosine-similarity vector store backed by a Qdrant collection.

    The collection is created on demand using ``Distance.COSINE`` so the scoring
    orientation matches the in-memory store: higher scores mean more similar.
    """

    def __init__(
        self,
        dim: int,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "feedback_intelligence",
        client: QdrantClient | None = None,
    ) -> None:
        """Connect to Qdrant and ensure the target collection exists.

        ``client`` can be injected for testing; otherwise a real ``QdrantClient``
        is created against ``url``.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.url = url
        self.collection_name = collection_name
        self._qdrant = _import_qdrant_client()
        if client is None:
            try:
                client = self._qdrant.QdrantClient(url=url)
            except Exception as exc:  # pragma: no cover - network-dependent.
                raise QdrantStoreError(
                    f"Could not connect to Qdrant at {url}. "
                    "Is the service running (docker compose up qdrant)?"
                ) from exc
        self._client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection with cosine distance if it does not exist yet."""
        models = self._qdrant.models
        try:
            exists = self._client.collection_exists(self.collection_name)
        except Exception as exc:  # pragma: no cover - network-dependent.
            raise QdrantStoreError(f"Could not query collections on Qdrant at {self.url}.") from exc
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    @property
    def size(self) -> int:
        """Number of indexed chunks in the collection."""
        return int(self._client.count(collection_name=self.collection_name).count)

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Return all indexed chunks, reconstructed from point payloads."""
        chunks: list[DocumentChunk] = []
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            chunks.extend(_payload_to_chunk(point.payload or {}) for point in points)
            if offset is None:
                break
        return chunks

    def add(self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]) -> None:
        """Upsert chunks and their vectors into the collection."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with dim={self.dim}")
        if not chunks:
            return

        models = self._qdrant.models
        points = [
            models.PointStruct(
                id=_chunk_to_point_id(chunk.chunk_id),
                vector=vector.astype(np.float64, copy=False).tolist(),
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self, query_vector: npt.NDArray[np.float64], *, top_k: int = 4
    ) -> list[SearchResult]:
        """Return the most similar chunks for a query vector (highest score first)."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"expected query vector with dim={self.dim}")

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector.astype(np.float64, copy=False).tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk=_payload_to_chunk(point.payload or {}),
                score=float(point.score),
            )
            for point in response.points
        ]


def _chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """Serialise a chunk into a Qdrant point payload."""
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def _payload_to_chunk(payload: dict[str, Any]) -> DocumentChunk:
    """Reconstruct a chunk from a Qdrant point payload."""
    return DocumentChunk(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        text=str(payload["text"]),
        metadata=dict(payload.get("metadata") or {}),
    )
