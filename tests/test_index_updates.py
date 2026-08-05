from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.index_updates import update_json_index
from feedback_intelligence_agent.schemas import FeedbackChannel, FeedbackRecord
from feedback_intelligence_agent.telemetry import InMemoryTelemetrySink, Telemetry
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def make_record(feedback_id: str, text: str) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=feedback_id,
        customer_segment="enterprise",
        channel=FeedbackChannel.support_ticket,
        rating=2,
        text=text,
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def test_update_json_index_creates_new_index(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"

    result = update_json_index(
        [make_record("fb-new", "streamed onboarding checklist update")],
        index_path,
        embedding_dim=128,
    )

    assert result.created is True
    assert result.inserted_records == 1
    assert result.updated_records == 0
    assert result.total_chunks == 1
    store = InMemoryVectorStore.load(index_path)
    assert store.size == 1
    assert store.chunks[0].source_id == "fb-new"


def test_update_json_index_replaces_existing_source_without_rebuilding_all(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "vector_store.json"
    update_json_index(
        [
            make_record("fb-1", "old onboarding checklist"),
            make_record("fb-2", "pricing renewal finance"),
        ],
        index_path,
        embedding_dim=128,
    )

    result = update_json_index(
        [make_record("fb-1", "new export dashboard failure")],
        index_path,
        embedding_dim=128,
    )

    assert result.created is False
    assert result.inserted_records == 0
    assert result.updated_records == 1
    assert result.removed_chunks == 1
    assert result.total_chunks == 2
    store = InMemoryVectorStore.load(index_path)
    assert {chunk.source_id for chunk in store.chunks} == {"fb-1", "fb-2"}
    assert "old onboarding" not in " ".join(chunk.text for chunk in store.chunks)
    result_chunks = store.search(
        query_vector=HashingEmbeddingModel(dim=128).embed(["export dashboard"])[0],
        top_k=1,
    )
    assert result_chunks[0].chunk.source_id == "fb-1"


def test_update_json_index_deduplicates_input_records_last_wins(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"

    result = update_json_index(
        [
            make_record("fb-1", "old event text"),
            make_record("fb-1", "new event text"),
        ],
        index_path,
        embedding_dim=128,
    )

    assert result.input_records == 2
    assert result.skipped_records == 1
    assert result.inserted_records == 1
    store = InMemoryVectorStore.load(index_path)
    assert store.chunks[0].text == "new event text"


def test_update_json_index_emits_telemetry(tmp_path: Path) -> None:
    sink = InMemoryTelemetrySink()

    result = update_json_index(
        [make_record("fb-1", "telemetry update")],
        tmp_path / "vector_store.json",
        embedding_dim=128,
        telemetry=Telemetry(sink=sink),
    )

    assert result.total_chunks == 1
    assert sink.event_names() == ["index_update_started", "index_update_finished"]
    assert sink.events[-1].metadata["inserted_records"] == 1


def test_update_json_index_rejects_dimension_mismatch(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    update_json_index([make_record("fb-1", "first")], index_path, embedding_dim=128)

    with pytest.raises(ValueError, match="does not match"):
        update_json_index([make_record("fb-2", "second")], index_path, embedding_dim=256)
