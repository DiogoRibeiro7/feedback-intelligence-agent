"""Bounded record batches and live retrieval snapshots for JSON ingestion."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import build_agent_snapshot
from feedback_intelligence_agent.index_updates import update_json_index
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


class RecordIngestionRequest(BaseModel):
    """A single tenant's records, with no client-selectable filesystem paths."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str | None = Field(default=None, min_length=1)
    records: list[FeedbackRecord] = Field(min_length=1, max_length=1000)

    @field_validator("tenant_id")
    @classmethod
    def normalise_tenant_id(cls, value: str | None) -> str | None:
        """Normalize explicit batch tenants and reject blank names."""
        if value is None:
            return None
        tenant = value.strip().lower()
        if not tenant:
            raise ValueError("tenant_id cannot be blank")
        return tenant


class RecordIngestionResult(BaseModel):
    """Batch-local counts without global index totals or server paths."""

    tenant_id: str
    input_records: int = Field(ge=0)
    inserted_records: int = Field(ge=0)
    updated_records: int = Field(ge=0)
    skipped_records: int = Field(ge=0)
    added_chunks: int = Field(ge=0)
    removed_chunks: int = Field(ge=0)


def ingest_record_batch(
    records: list[FeedbackRecord],
    tenant_id: str,
    *,
    settings: Settings,
    agent: FeedbackInsightAgent,
) -> tuple[RecordIngestionResult, FeedbackInsightAgent]:
    """Persist a validated tenant batch and return its prepared agent snapshot.

    The API holds the index writer lock until it publishes the returned agent.
    Preparation precedes the atomic save, so failed preparation or persistence
    never replaces the serving agent. Existing provider and policy objects are
    retained, including circuit-breaker state.
    """
    if settings.vector_store != "json":
        raise ValueError("Record ingestion requires the JSON vector store")
    if not records or any(record.tenant_id != tenant_id for record in records):
        raise ValueError("All records must belong to the batch tenant")

    prepared_agent = agent

    def prepare_snapshot(store: InMemoryVectorStore) -> None:
        nonlocal prepared_agent
        prepared_agent = build_agent_snapshot(settings, store, previous=agent)

    result = update_json_index(
        records,
        settings.index_path,
        embedding_dim=settings.embedding_dim,
        telemetry=agent.telemetry,
        prepare_snapshot=prepare_snapshot,
    )
    return RecordIngestionResult(
        tenant_id=tenant_id,
        input_records=result.input_records,
        inserted_records=result.inserted_records,
        updated_records=result.updated_records,
        skipped_records=result.skipped_records,
        added_chunks=result.added_chunks,
        removed_chunks=result.removed_chunks,
    ), prepared_agent
