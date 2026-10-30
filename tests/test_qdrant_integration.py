"""Exercise the optional SDK against a real, isolated Qdrant collection."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from feedback_intelligence_agent.chunking import feedback_to_chunks
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.qdrant_store import QdrantVectorStore
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackRecord, MetadataFilters

pytestmark = pytest.mark.skipif(
    os.environ.get("FEEDBACK_AGENT_QDRANT_TEST", "").lower() not in {"1", "true", "yes"},
    reason="set FEEDBACK_AGENT_QDRANT_TEST=1 to run against a real Qdrant server",
)


@pytest.fixture()
def collection() -> Iterator[tuple[QdrantVectorStore, Any]]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.environ.get("FEEDBACK_AGENT_QDRANT_URL", "http://localhost:6333"))
    name = "feedback_source_test_" + uuid.uuid4().hex
    try:
        yield QdrantVectorStore(dim=64, client=client, collection_name=name), client
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


def chunks(tenant: str, identifier: str = "shared", *, words: int = 6) -> list[DocumentChunk]:
    record = FeedbackRecord.model_validate(
        {
            "tenant_id": tenant,
            "feedback_id": identifier,
            "customer_segment": "enterprise",
            "channel": "support_ticket",
            "rating": 2,
            "text": "onboarding " * words,
            "created_at": "2026-10-29T10:00:00Z",
        }
    )
    return feedback_to_chunks([record], max_words=2, overlap_words=0)


@pytest.mark.parametrize("words", [6, 1030])
def test_remote_source_replacement_removes_trailing_chunks_and_preserves_foreign_vectors(
    collection: tuple[QdrantVectorStore, Any], words: int
) -> None:
    store, client = collection
    initial = chunks("acme", words=words) + chunks("cobalt") + chunks("default", "acme:shared")
    model = HashingEmbeddingModel(dim=64)
    store.add(initial, model.embed([chunk.text for chunk in initial]))
    before = store.chunks
    foreign_points, _ = client.scroll(
        collection_name=store.collection_name, limit=1024, with_payload=True, with_vectors=True
    )
    foreign = {
        point.id: point.model_dump()
        for point in foreign_points
        if point.payload["source_id"] != "acme:shared"
    }

    incoming = chunks("acme", words=2)
    assert (
        store.upsert_sources(incoming, model.embed([chunk.text for chunk in incoming]))
        == words // 2
    )
    assert store.size == 7
    assert len(before) > len(store.chunks)
    after = client.retrieve(
        collection_name=store.collection_name,
        ids=list(foreign),
        with_vectors=True,
        with_payload=True,
    )
    assert {point.id: point.model_dump() for point in after} == foreign
    results = store.search(
        model.embed(["onboarding"])[0], top_k=12, filters=MetadataFilters(tenant_id="acme")
    )
    assert [result.chunk.chunk_id for result in results] == [incoming[0].chunk_id]
    assert store.upsert_sources(incoming, model.embed([incoming[0].text])) == 1
    assert store.size == 7


def test_remote_replacement_of_multiple_sources_and_empty_batch(
    collection: tuple[QdrantVectorStore, Any],
) -> None:
    store, _ = collection
    original = chunks("acme") + chunks("cobalt")
    store.add(original, np.ones((len(original), 64)))
    incoming = chunks("acme", words=2) + chunks("cobalt", words=8) + chunks("acme", "new", words=2)
    assert store.upsert_sources(incoming, np.ones((len(incoming), 64))) == 6
    assert {chunk.chunk_id for chunk in store.chunks} == {chunk.chunk_id for chunk in incoming}
    before = store.chunks
    assert store.upsert_sources([], np.empty((0, 64))) == 0
    assert store.chunks == before


def test_remote_invalid_batch_preserves_all_stored_points(
    collection: tuple[QdrantVectorStore, Any],
) -> None:
    store, client = collection
    original = chunks("acme") + chunks("cobalt")
    store.add(original, np.ones((len(original), 64)))
    before, _ = client.scroll(collection_name=store.collection_name, with_vectors=True)
    incoming = chunks("acme", words=2)
    with pytest.raises(ValueError, match="finite"):
        store.upsert_sources(incoming, np.full((1, 64), float("nan")))
    incoming[0].chunk_id = "cobalt:shared::chunk-0"
    with pytest.raises(ValueError, match="chunk_id"):
        store.upsert_sources(incoming, np.ones((1, 64)))
    after, _ = client.scroll(collection_name=store.collection_name, with_vectors=True)
    assert after == before
