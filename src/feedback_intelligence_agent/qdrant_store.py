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

from feedback_intelligence_agent.metadata_filters import matches_metadata_filters
from feedback_intelligence_agent.schemas import DocumentChunk, MetadataFilters, SearchResult
from feedback_intelligence_agent.source_ids import (
    SourceIdMigrationError,
    encode_source_id,
    migrate_legacy_chunks,
)
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

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
        if exists:
            # Validate legacy identities before any new points can overwrite them.
            _ = self.chunks
        else:
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

    def snapshot(self) -> InMemoryVectorStore:
        """Load chunks and vectors into an independent local retrieval snapshot.

        Callers must coordinate writers across the entire paginated read. This
        method does not supply a database transaction or lock external clients.
        The returned store makes no further Qdrant calls during retrieval.
        """
        chunks: list[DocumentChunk] = []
        vectors: list[npt.NDArray[np.float64]] = []
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=offset,
            )
            for point in points:
                if not isinstance(point.vector, list):
                    raise QdrantStoreError("Qdrant snapshot requires unnamed dense vectors")
                vector = np.asarray(point.vector, dtype=np.float64)
                if vector.shape != (self.dim,) or not np.isfinite(vector).all():
                    raise QdrantStoreError("Qdrant snapshot contains invalid dense vectors")
                chunks.append(_payload_to_chunk(point.payload or {}))
                vectors.append(vector)
            if offset is None:
                break
        store = InMemoryVectorStore(dim=self.dim)
        if chunks:
            store.add(chunks, np.vstack(vectors))
        return store

    def add(self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]) -> None:
        """Upsert chunk IDs and wait for completion; retain other chunks of each source."""
        points = self._prepare_points(chunks, vectors)
        if points:
            self._upsert_points(points)

    def _prepare_points(
        self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]
    ) -> list[Any]:
        """Validate every vector and serialize every point before sending mutations."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if vectors.shape[0] != len(chunks):
            raise ValueError("number of vectors must match number of chunks")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with dim={self.dim}")
        prepared = vectors.astype(np.float64, copy=False)
        if not np.isfinite(prepared).all():
            raise ValueError("vectors must contain only finite values")

        models = self._qdrant.models
        return [
            models.PointStruct(
                id=_chunk_to_point_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, prepared, strict=True)
        ]

    def _upsert_points(self, points: list[Any]) -> None:
        result = self._client.upsert(collection_name=self.collection_name, points=points, wait=True)
        if result.status != self._qdrant.models.UpdateStatus.COMPLETED:
            raise QdrantStoreError("Qdrant did not complete the point upsert")

    def upsert_sources(self, chunks: list[DocumentChunk], vectors: npt.NDArray[np.float64]) -> int:
        """Replace complete tenant-scoped sources, including obsolete trailing chunks.

        Supply chunks produced by ``feedback_to_chunks`` with canonical source,
        chunk, tenant, and feedback identities. All input is validated before any
        remote mutation. New points are upserted with ``wait=True`` before obsolete
        point IDs are deleted, also with ``wait=True``. Other sources are untouched.

        This is not a transaction: readers can see intermediate state and a remote
        failure can leave a partial update. Retrying the identical batch converges.
        Callers must serialize concurrent writes to the same sources, including
        writes through other clients. Disjoint sources may be updated independently.

        Returns the number of previous chunks belonging to the supplied sources,
        including overwritten chunk IDs, matching the in-memory store's counts.
        An empty batch is a no-op, not a request to delete a source.
        """
        _validate_source_chunks(chunks)
        points = self._prepare_points(chunks, vectors)
        if not points:
            return 0
        models = self._qdrant.models
        source_ids = {chunk.source_id for chunk in chunks}
        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id", match=models.MatchAny(any=sorted(source_ids))
                )
            ]
        )
        existing_ids: list[int | str] = []
        offset: Any = None
        while True:
            existing, offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=source_filter,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in existing:
                chunk = _payload_to_chunk(point.payload or {})
                if chunk.source_id not in source_ids:
                    raise QdrantStoreError(
                        "Qdrant returned a point outside the replacement sources"
                    )
                point_id = point.id
                # SDK transports can return UUID objects or differently cased strings.
                # Normalize before comparing IDs, or cleanup could delete new points.
                existing_ids.append(
                    point_id if isinstance(point_id, int) else str(uuid.UUID(str(point_id)))
                )
            if offset is None:
                break
        incoming_ids = {point.id for point in points}
        obsolete_ids = [point_id for point_id in existing_ids if point_id not in incoming_ids]
        self._upsert_points(points)
        if obsolete_ids:
            result = self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=obsolete_ids),
                wait=True,
            )
            if result.status != models.UpdateStatus.COMPLETED:
                raise QdrantStoreError("Qdrant did not complete obsolete chunk deletion")
        return len(existing_ids)

    def search(
        self,
        query_vector: npt.NDArray[np.float64],
        *,
        top_k: int = 4,
        filters: MetadataFilters | None = None,
    ) -> list[SearchResult]:
        """Page ranked points until enough eligible chunks are found or results end.

        Local matching preserves case-insensitive and legacy metadata semantics
        without requiring a payload migration. Selective filters may scan many
        pages; each page is bounded, but there is no fixed candidate cutoff.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"expected query vector with dim={self.dim}")

        filter_active = filters is not None and not filters.is_empty
        page_size = max(64, top_k) if filter_active else top_k
        offset = 0
        results: list[SearchResult] = []
        query = query_vector.astype(np.float64, copy=False).tolist()
        while True:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query,
                limit=page_size,
                offset=offset,
                with_payload=True,
            )
            for point in response.points:
                chunk = _payload_to_chunk(point.payload or {})
                if filters is not None and not matches_metadata_filters(chunk.metadata, filters):
                    continue
                results.append(SearchResult(chunk=chunk, score=float(point.score)))
                if len(results) == top_k:
                    return results
            if len(response.points) < page_size:
                return results
            offset += len(response.points)


def _validate_source_chunks(chunks: list[DocumentChunk]) -> None:
    """Reject ambiguous identities before a source replacement can overwrite points."""
    seen_ids: set[str] = set()
    for chunk in chunks:
        tenant = chunk.metadata.get("tenant_id")
        feedback_id = chunk.metadata.get("feedback_id")
        if (
            not isinstance(tenant, str)
            or not tenant
            or tenant != tenant.strip().lower()
            or not isinstance(feedback_id, str)
            or not feedback_id
            or chunk.source_id != encode_source_id(tenant, feedback_id)
        ):
            raise ValueError("source_id must match tenant_id and feedback_id metadata")
        prefix = f"{chunk.source_id}::chunk-"
        position = chunk.chunk_id.removeprefix(prefix)
        if (
            not chunk.chunk_id.startswith(prefix)
            or not position.isascii()
            or not position.isdecimal()
            or (len(position) > 1 and position.startswith("0"))
        ):
            raise ValueError("chunk_id must contain its source_id and a canonical chunk position")
        if chunk.chunk_id in seen_ids:
            raise ValueError("duplicate chunk_id in source replacement")
        seen_ids.add(chunk.chunk_id)


def _chunk_to_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """Serialise a chunk into a Qdrant point payload."""
    return {
        "source_id_version": 2,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def _payload_to_chunk(payload: dict[str, Any]) -> DocumentChunk:
    """Reconstruct a chunk from a Qdrant point payload."""
    chunk = DocumentChunk(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        text=str(payload["text"]),
        metadata=dict(payload.get("metadata") or {}),
    )
    version = payload.get("source_id_version", 1)
    if version == 2:
        return chunk
    if version != 1:
        raise QdrantStoreError("Unsupported Qdrant source ID version")
    rebuild_message = (
        "Legacy Qdrant source IDs require migration. Set FEEDBACK_AGENT_QDRANT_COLLECTION "
        "to a new collection and rebuild from the original feedback data."
    )
    try:
        [migrated] = migrate_legacy_chunks([chunk])
    except SourceIdMigrationError as exc:
        raise QdrantStoreError(rebuild_message) from exc
    if migrated.source_id != chunk.source_id or migrated.chunk_id != chunk.chunk_id:
        raise QdrantStoreError(rebuild_message)
    return chunk
