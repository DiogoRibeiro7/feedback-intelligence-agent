"""Coordinated Qdrant record ingestion and immutable API retrieval snapshots."""

from __future__ import annotations

from time import monotonic
from uuid import UUID

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.chunking import feedback_to_chunks, scoped_source_id
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.factory import (
    build_agent,
    build_agent_snapshot,
    build_qdrant_index,
    build_telemetry,
    chunk_to_embedding_text,
)
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.qdrant_coordination import (
    qdrant_snapshot_revision,
    qdrant_write_lock,
)
from feedback_intelligence_agent.record_ingestion import RecordIngestionResult
from feedback_intelligence_agent.schemas import FeedbackRecord


class QdrantAgentSnapshots:
    """Serve Qdrant through snapshots coordinated by one shared local lock path.

    All workers and external writers must cooperate on that path. Snapshot reads
    hold the lock and release it before answering. A positive cache TTL reuses
    unchanged snapshots; cooperating writes invalidate them immediately through
    a shared generation marker. Retrieval and tools share immutable local data.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.vector_store != "qdrant" or settings.qdrant_ingestion_lock_path is None:
            raise ValueError("Qdrant ingestion requires a shared lock path")
        self._settings = settings
        self._lock_path = settings.qdrant_ingestion_lock_path
        # Startup can seed an empty collection, so invalidate before initializing.
        with qdrant_write_lock(self._lock_path) as generation:
            telemetry = build_telemetry(settings)
            self._store = build_qdrant_index(settings, telemetry=telemetry)
            self._agent = build_agent(
                settings, telemetry=telemetry, vector_store=self._store.snapshot()
            )
            self._generation: UUID | None = generation
            self._snapshot_loaded_at: float | None = monotonic()

    def get(self) -> FeedbackInsightAgent:
        """Reuse an unexpired, unchanged snapshot or refresh retrieval and tools together."""
        with index_write_lock(self._lock_path) as canonical_path:
            generation = qdrant_snapshot_revision(canonical_path)
            if (
                self._snapshot_loaded_at is not None
                and generation == self._generation
                and monotonic() - self._snapshot_loaded_at
                < self._settings.qdrant_snapshot_ttl_seconds
            ):
                return self._agent
            snapshot = self._store.snapshot()
            agent = build_agent_snapshot(self._settings, snapshot, previous=self._agent)
            self._agent = agent
            self._generation = generation
            self._snapshot_loaded_at = monotonic()
            return agent

    def ingest(self, records: list[FeedbackRecord], tenant_id: str) -> RecordIngestionResult:
        """Replace one tenant's records and publish the prepared snapshot on success.

        Validation, embedding, and snapshot preparation precede remote mutation.
        Remote failures can leave partial updates; retry the identical batch.
        Subsequent reads load the actual stored state, including partial writes.
        """
        if not records or any(record.tenant_id != tenant_id for record in records):
            raise ValueError("All records must belong to the batch tenant")
        deduped = list({record.feedback_id: record for record in records}.values())
        with index_write_lock(self._lock_path):
            snapshot = self._store.snapshot()
            existing_sources = {chunk.source_id for chunk in snapshot.chunks}
            chunks = feedback_to_chunks(deduped)
            vectors = HashingEmbeddingModel(dim=self._settings.embedding_dim).embed(
                [chunk_to_embedding_text(chunk) for chunk in chunks]
            )
            snapshot.upsert_sources(chunks, vectors)
            agent = build_agent_snapshot(self._settings, snapshot, previous=self._agent)
            with qdrant_write_lock(self._lock_path):
                removed = self._store.upsert_sources(chunks, vectors)
            self._agent = agent
            # Read persisted vectors on the next query; Qdrant may normalize them.
            self._snapshot_loaded_at = None
            updated = sum(scoped_source_id(record) in existing_sources for record in deduped)
            return RecordIngestionResult(
                tenant_id=tenant_id,
                input_records=len(records),
                inserted_records=len(deduped) - updated,
                updated_records=updated,
                skipped_records=len(records) - len(deduped),
                added_chunks=len(chunks),
                removed_chunks=removed,
            )
