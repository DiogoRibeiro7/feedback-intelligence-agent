from __future__ import annotations

from pathlib import Path

import pytest
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.schemas import DocumentChunk
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def test_vector_store_persistence_errors_have_file_context(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileReadError) as missing_error:
        InMemoryVectorStore.load(missing)
    assert missing_error.value.path == str(missing)
    assert missing_error.value.original is missing_error.value.__cause__

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(DataLoadingError) as parse_error:
        InMemoryVectorStore.load(malformed)
    assert parse_error.value.source == str(malformed)
    assert parse_error.value.original is parse_error.value.__cause__

    parent_file = tmp_path / "parent-file"
    parent_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileWriteError) as write_error:
        InMemoryVectorStore(dim=8).save(parent_file / "store.json")
    assert write_error.value.path == str(parent_file)
    assert write_error.value.original is write_error.value.__cause__


def test_hashing_embeddings_are_deterministic() -> None:
    model = HashingEmbeddingModel(dim=128)
    first = model.embed(["onboarding checklist"])
    second = model.embed(["onboarding checklist"])
    assert first.shape == (1, 128)
    assert (first == second).all()


def test_vector_store_returns_most_similar_chunk() -> None:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="1", source_id="fb-1", text="onboarding checklist setup", metadata={}
        ),
        DocumentChunk(chunk_id="2", source_id="fb-2", text="pricing renewal finance", metadata={}),
    ]
    vectors = model.embed([chunk.text for chunk in chunks])
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, vectors)

    query_vector = model.embed(["setup checklist"])[0]
    results = store.search(query_vector, top_k=1)

    assert results[0].chunk.source_id == "fb-1"


def test_vector_store_upsert_replaces_existing_source_chunks() -> None:
    model = HashingEmbeddingModel(dim=128)
    initial = [
        DocumentChunk(
            chunk_id="fb-1::chunk-0", source_id="fb-1", text="old onboarding", metadata={}
        ),
        DocumentChunk(
            chunk_id="fb-2::chunk-0", source_id="fb-2", text="pricing renewal", metadata={}
        ),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(initial, model.embed([chunk.text for chunk in initial]))
    replacement = [
        DocumentChunk(
            chunk_id="fb-1::chunk-0",
            source_id="fb-1",
            text="new export dashboard",
            metadata={},
        )
    ]

    removed = store.upsert_sources(
        replacement,
        model.embed([chunk.text for chunk in replacement]),
    )

    assert removed == 1
    assert store.size == 2
    assert {chunk.text for chunk in store.chunks} == {"new export dashboard", "pricing renewal"}
    results = store.search(model.embed(["export dashboard"])[0], top_k=1)
    assert results[0].chunk.source_id == "fb-1"
