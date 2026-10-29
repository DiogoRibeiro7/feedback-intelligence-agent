"""Tests for the optional Qdrant-backed vector store.

Unit tests use a fake ``qdrant_client`` module injected into ``sys.modules`` (and
an injected fake client), so they never require a running Qdrant or the optional
SDK. Integration tests against a real Qdrant are skipped unless
``FEEDBACK_AGENT_QDRANT_TEST`` is set, keeping the default test run green with no
external service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from feedback_intelligence_agent.chunking import feedback_to_chunks
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.factory import load_or_build_index
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackRecord, MetadataFilters

# ---------------------------------------------------------------------------
# Fake qdrant_client module + client for unit tests
# ---------------------------------------------------------------------------


def _make_fake_qdrant_module() -> ModuleType:
    """Build a stand-in ``qdrant_client`` module with the ``models`` namespace."""

    class VectorParams:
        def __init__(self, *, size: int, distance: object) -> None:
            self.size = size
            self.distance = distance

    class Distance:
        COSINE = "Cosine"

    class PointStruct:
        def __init__(self, *, id: str, vector: list[float], payload: dict[str, Any]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    models = SimpleNamespace(
        VectorParams=VectorParams,
        Distance=Distance,
        PointStruct=PointStruct,
    )

    module = type(sys)("qdrant_client")
    module.models = models  # type: ignore[attr-defined]
    # QdrantClient is provided by the test via dependency injection, but expose a
    # placeholder so attribute access does not fail.
    module.QdrantClient = object  # type: ignore[attr-defined]
    return module


class FakeQdrantClient:
    """In-memory stand-in for ``QdrantClient`` recording calls and points."""

    def __init__(self) -> None:
        self.collections: dict[str, Any] = {}
        self.points: dict[str, Any] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, int]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.collections[collection_name] = vectors_config
        self.create_calls.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )

    def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})
        for point in points:
            self.points[point.id] = point

    def count(self, *, collection_name: str) -> SimpleNamespace:
        return SimpleNamespace(count=len(self.points))

    def scroll(
        self,
        *,
        collection_name: str,
        with_payload: bool,
        with_vectors: bool,
        limit: int,
        offset: object,
    ) -> tuple[list[object], object]:
        items = list(self.points.values())
        return items, None

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        offset: int = 0,
    ) -> SimpleNamespace:
        self.query_calls.append({"offset": offset, "limit": limit})
        query_vec = np.asarray(query, dtype=np.float64)
        scored = []
        for point in self.points.values():
            vec = np.asarray(point.vector, dtype=np.float64)
            score = float(vec @ query_vec)
            scored.append(SimpleNamespace(payload=point.payload, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scored[offset : offset + limit])


@pytest.fixture()
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _make_fake_qdrant_module()
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return module


def _sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="fb-1::chunk-0",
            source_id="fb-1",
            text="onboarding checklist setup",
            metadata={"channel": "support_ticket"},
        ),
        DocumentChunk(
            chunk_id="fb-2::chunk-0",
            source_id="fb-2",
            text="pricing renewal finance",
            metadata={"channel": "nps_survey"},
        ),
    ]


# ---------------------------------------------------------------------------
# Unit tests (no running Qdrant)
# ---------------------------------------------------------------------------


def test_qdrant_store_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(QdrantStoreError, match="poetry install --extras qdrant"):
        QdrantVectorStore(dim=16, client=None)


def test_qdrant_store_creates_collection_with_cosine(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    QdrantVectorStore(dim=32, collection_name="feedback", client=client)

    assert client.create_calls
    config = client.create_calls[0]["vectors_config"]
    assert config.size == 32
    assert config.distance == fake_qdrant.models.Distance.COSINE  # type: ignore[attr-defined]


def test_qdrant_store_upsert_payloads(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=64, client=client)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    vectors = model.embed([chunk.text for chunk in chunks])

    store.add(chunks, vectors)

    assert len(client.upsert_calls) == 1
    points = client.upsert_calls[0]["points"]
    assert len(points) == 2
    first = points[0]
    assert first.payload["source_id_version"] == 2
    assert first.payload["chunk_id"] == "fb-1::chunk-0"
    assert first.payload["source_id"] == "fb-1"
    assert first.payload["text"] == "onboarding checklist setup"
    assert first.payload["metadata"] == {"channel": "support_ticket"}
    assert len(first.vector) == 64
    assert store.size == 2


def test_qdrant_store_search_maps_to_search_results(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=64, client=client)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    query_vector = model.embed(["setup checklist"])[0]
    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    # Highest cosine score first, mapped back to the matching chunk.
    assert results[0].chunk.source_id == "fb-1"
    assert results[0].score >= results[1].score


def test_qdrant_store_chunks_roundtrip(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=64, client=client)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    recovered_ids = sorted(chunk.chunk_id for chunk in store.chunks)
    assert recovered_ids == ["fb-1::chunk-0", "fb-2::chunk-0"]


def test_qdrant_store_validates_inputs(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    with pytest.raises(ValueError, match="dim must be positive"):
        QdrantVectorStore(dim=0, client=FakeQdrantClient())

    store = QdrantVectorStore(dim=8, client=FakeQdrantClient())
    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search(np.zeros(8, dtype=np.float64), top_k=0)
    with pytest.raises(ValueError, match="dim=8"):
        store.search(np.zeros(4, dtype=np.float64), top_k=1)


@pytest.mark.parametrize("matching_count", [0, 1, 3])
def test_qdrant_filtered_search_pages_until_matches_or_exhaustion(
    fake_qdrant: ModuleType, matching_count: int
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=2, client=client)
    excluded = [
        DocumentChunk(
            chunk_id=f"other-{index}",
            source_id=f"other-{index}",
            text="Other tenant",
            metadata={"tenant_id": "cobalt"},
        )
        for index in range(130)
    ]
    matching = [
        DocumentChunk(
            chunk_id=f"wanted-{index}",
            source_id=f"wanted-{index}",
            text="Requested tenant",
            metadata={"tenant_id": "acme", "customer_segment": "Enterprise", "rating": "2"},
        )
        for index in range(matching_count)
    ]
    vectors = np.asarray(
        [[1.0, 0.0]] * 130 + [[0.6 - index / 10, 0.8] for index in range(matching_count)]
    )
    store.add(excluded + matching, vectors)

    results = store.search(
        np.asarray([1.0, 0.0]),
        top_k=2,
        filters=MetadataFilters(tenant_id="ACME", customer_segment="enterprise", max_rating=2),
    )

    assert [result.chunk for result in results] == matching[:2]
    assert client.query_calls == [
        {"offset": 0, "limit": 64},
        {"offset": 64, "limit": 64},
        {"offset": 128, "limit": 64},
    ]
    client.query_calls.clear()
    assert len(store.search(np.asarray([1.0, 0.0]), top_k=2, filters=MetadataFilters())) == 2
    assert client.query_calls == [{"offset": 0, "limit": 2}]


def test_qdrant_preserves_records_with_previously_colliding_ids(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=64, client=client)
    records = [
        FeedbackRecord.model_validate(
            {
                "tenant_id": tenant,
                "feedback_id": feedback_id,
                "customer_segment": "enterprise",
                "channel": "support_ticket",
                "rating": 2,
                "text": "Tenant feedback about onboarding",
                "created_at": "2026-10-28T00:00:00Z",
            }
        )
        for tenant, feedback_id in [("default", "acme:shared"), ("acme", "shared")]
    ]
    chunks = feedback_to_chunks(records)
    model = HashingEmbeddingModel(dim=64)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    assert store.size == 2
    assert QdrantVectorStore(dim=64, client=client).chunks == chunks
    assert {result.chunk.source_id for result in store.search(model.embed(["onboarding"])[0])} == {
        "acme%3Ashared",
        "acme:shared",
    }


@pytest.mark.parametrize("legacy_id", ["fb-1", "acme:shared", "fb%3A1"])
def test_qdrant_checks_legacy_collection_before_writes(
    fake_qdrant: ModuleType, legacy_id: str
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    client = FakeQdrantClient()
    client.collections["feedback_intelligence"] = object()
    payload = {
        "chunk_id": f"{legacy_id}::chunk-0",
        "source_id": legacy_id,
        "text": "Legacy feedback",
        "metadata": {"tenant_id": "default", "feedback_id": legacy_id},
    }
    client.points["legacy-point"] = SimpleNamespace(payload=payload, vector=[1.0] * 64)

    if legacy_id == "fb-1":
        store = QdrantVectorStore(dim=64, client=client)
        assert store.chunks[0].source_id == "fb-1"
    else:
        with pytest.raises(QdrantStoreError, match="new collection and rebuild"):
            QdrantVectorStore(dim=64, client=client)
    assert client.upsert_calls == []
    assert len(client.points) == 1


@pytest.mark.parametrize("problem", ["version", "metadata"])
def test_qdrant_rejects_invalid_identity_payloads(fake_qdrant: ModuleType, problem: str) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    client = FakeQdrantClient()
    client.collections["feedback_intelligence"] = object()
    payload = _sample_chunks()[0].model_dump()
    if problem == "version":
        payload["source_id_version"] = 99
    else:
        payload["metadata"]["feedback_id"] = "different"
    client.points["legacy-point"] = SimpleNamespace(payload=payload)

    with pytest.raises(QdrantStoreError):
        QdrantVectorStore(dim=64, client=client)
    assert client.upsert_calls == []


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_factory_defaults_to_json_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from feedback_intelligence_agent.vector_store import InMemoryVectorStore

    monkeypatch.delenv("FEEDBACK_AGENT_VECTOR_STORE", raising=False)
    settings = Settings(index_path=tmp_path / "vector_store.json")
    assert settings.vector_store == "json"

    store = load_or_build_index(settings)
    assert isinstance(store, InMemoryVectorStore)


def test_factory_uses_qdrant_when_configured(
    fake_qdrant: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feedback_intelligence_agent.qdrant_store as qdrant_store
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    original_init = QdrantVectorStore.__init__

    def patched_init(self: QdrantVectorStore, *args: object, **kwargs: object) -> None:
        kwargs["client"] = client
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(qdrant_store.QdrantVectorStore, "__init__", patched_init)

    settings = Settings(vector_store="qdrant", embedding_dim=64)
    store = load_or_build_index(settings)

    assert isinstance(store, QdrantVectorStore)
    assert store.size > 0


# ---------------------------------------------------------------------------
# Integration tests (skipped unless a real Qdrant is available)
# ---------------------------------------------------------------------------

_QDRANT_TEST = os.environ.get("FEEDBACK_AGENT_QDRANT_TEST", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _QDRANT_TEST, reason="set FEEDBACK_AGENT_QDRANT_TEST=1 to run")
def test_qdrant_integration_roundtrip() -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    url = os.environ.get("FEEDBACK_AGENT_QDRANT_URL", "http://localhost:6333")
    collection = "feedback_intelligence_test_" + os.urandom(4).hex()
    store = QdrantVectorStore(dim=64, url=url, collection_name=collection)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    results = store.search(model.embed(["setup checklist"])[0], top_k=1)
    assert results
    assert results[0].chunk.source_id == "fb-1"
