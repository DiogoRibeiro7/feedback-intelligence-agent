"""Coordinated Qdrant record ingestion and immutable API retrieval snapshots."""

from __future__ import annotations

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
from feedback_intelligence_agent.record_ingestion import RecordIngestionResult
from feedback_intelligence_agent.schemas import FeedbackRecord


class QdrantAgentSnapshots:
    """Serve Qdrant through snapshots coordinated by one shared local lock path.

    All workers and external writers must cooperate on that path. Every request
    downloads the collection's chunks and vectors under the lock, then releases
    it before answering. Retrieval and dataset tools share that snapshot, while
    other workers can ingest and in-flight answers retain their selected data.
    This mode trades remote search scalability for consistent local snapshots.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.vector_store != "qdrant" or settings.qdrant_ingestion_lock_path is None:
            raise ValueError("Qdrant ingestion requires a shared lock path")
        self._settings = settings
        self._lock_path = settings.qdrant_ingestion_lock_path
        with index_write_lock(self._lock_path):
            telemetry = build_telemetry(settings)
            self._store = build_qdrant_index(settings, telemetry=telemetry)
            self._agent = build_agent(
                settings, telemetry=telemetry, vector_store=self._store.snapshot()
            )

    def get(self) -> FeedbackInsightAgent:
        """Refresh retrieval and tools together without replacing provider state."""
        with index_write_lock(self._lock_path):
            snapshot = self._store.snapshot()
            agent = build_agent_snapshot(self._settings, snapshot, previous=self._agent)
            self._agent = agent
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
            removed = self._store.upsert_sources(chunks, vectors)
            self._agent = agent
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
