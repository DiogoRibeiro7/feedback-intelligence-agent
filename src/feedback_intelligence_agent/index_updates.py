"""Incremental index update helpers.

The default JSON index can be updated without rebuilding from the full dataset:
validated feedback records are chunked, embedded, and merged into the persisted
vector store by tenant-scoped source ID. Existing chunks for the same source document are
replaced before new chunks are appended.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from feedback_intelligence_agent.chunking import feedback_to_chunks, scoped_source_id
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.factory import chunk_to_embedding_text
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.telemetry import Telemetry
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


class IncrementalIndexResult(BaseModel):
    """Summary of an incremental JSON index update."""

    index_path: str
    created: bool
    input_records: int = Field(ge=0)
    inserted_records: int = Field(ge=0)
    updated_records: int = Field(ge=0)
    skipped_records: int = Field(ge=0)
    added_chunks: int = Field(ge=0)
    removed_chunks: int = Field(ge=0)
    total_chunks: int = Field(ge=0)


def update_json_index(
    records: list[FeedbackRecord],
    index_path: str | Path,
    *,
    embedding_dim: int,
    chunk_size: int = 80,
    chunk_overlap: int = 16,
    telemetry: Telemetry | None = None,
    prepare_snapshot: Callable[[InMemoryVectorStore], None] | None = None,
) -> IncrementalIndexResult:
    """Merge records under a writer lock, then atomically replace the JSON index.

    An optional consumer can prepare a retrieval snapshot before persistence.
    If preparation raises, the previous index is preserved. The consumer must
    not publish the snapshot until this function returns successfully.
    """
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    telemetry = telemetry or Telemetry()
    with index_write_lock(index_path) as output_path:
        created = not output_path.exists()
        store = (
            InMemoryVectorStore(dim=embedding_dim)
            if created
            else InMemoryVectorStore.load(output_path)
        )
        if store.dim != embedding_dim:
            raise ValueError(f"index dim={store.dim} does not match embedding_dim={embedding_dim}")

        existing_source_ids = {chunk.source_id for chunk in store.chunks}
        deduped = _dedupe_records(records)
        skipped = len(records) - len(deduped)
        chunks = feedback_to_chunks(deduped, max_words=chunk_size, overlap_words=chunk_overlap)

        correlation_id = telemetry.new_correlation_id()
        with telemetry.span(
            "index_update_started",
            "index_update_finished",
            correlation_id=correlation_id,
            metadata={
                "index_path": str(index_path),
                "input_records": len(records),
                "deduped_records": len(deduped),
                "created": created,
            },
        ) as span:
            removed_chunks = 0
            if chunks:
                embedding_model = HashingEmbeddingModel(dim=embedding_dim)
                vectors = embedding_model.embed(
                    [chunk_to_embedding_text(chunk) for chunk in chunks]
                )
                removed_chunks = store.upsert_sources(chunks, vectors)
            if prepare_snapshot is not None:
                prepare_snapshot(store)
            store.save(output_path)
            updated_records = sum(
                1 for record in deduped if scoped_source_id(record) in existing_source_ids
            )
            inserted_records = len(deduped) - updated_records
            result = IncrementalIndexResult(
                index_path=str(index_path),
                created=created,
                input_records=len(records),
                inserted_records=inserted_records,
                updated_records=updated_records,
                skipped_records=skipped,
                added_chunks=len(chunks),
                removed_chunks=removed_chunks,
                total_chunks=store.size,
            )
            span["inserted_records"] = result.inserted_records
            span["updated_records"] = result.updated_records
            span["skipped_records"] = result.skipped_records
            span["added_chunks"] = result.added_chunks
            span["removed_chunks"] = result.removed_chunks
            span["total_chunks"] = result.total_chunks
        return result


def _dedupe_records(records: list[FeedbackRecord]) -> list[FeedbackRecord]:
    """Keep the last event per tenant and feedback ID, in first-seen key order."""
    by_id: dict[tuple[str, str], FeedbackRecord] = {}
    for record in records:
        by_id[(record.tenant_id, record.feedback_id)] = record
    return list(by_id.values())
