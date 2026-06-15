"""Tests for the optional Qdrant-backed vector store.

Unit tests use a fake ``qdrant_client`` module injected into ``sys.modules`` (and
an injected fake client), so they never require a running Qdrant or the optional
SDK. Integration tests against a real Qdrant are skipped unless
``AI_SHOWCASE_QDRANT_TEST`` is set, keeping the default test run green with no
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

from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.factory import load_or_build_index
from ai_engineering_showcase.schemas import DocumentChunk

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
        self, *, collection_name: str, query: list[float], limit: int, with_payload: bool
    ) -> SimpleNamespace:
        query_vec = np.asarray(query, dtype=np.float64)
        scored = []
        for point in self.points.values():
            vec = np.asarray(point.vector, dtype=np.float64)
            score = float(vec @ query_vec)
            scored.append(SimpleNamespace(payload=point.payload, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])


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
    from ai_engineering_showcase.qdrant_store import QdrantStoreError, QdrantVectorStore

    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(QdrantStoreError, match="poetry install --extras qdrant"):
        QdrantVectorStore(dim=16, client=None)


def test_qdrant_store_creates_collection_with_cosine(fake_qdrant: ModuleType) -> None:
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    QdrantVectorStore(dim=32, collection_name="feedback", client=client)

    assert client.create_calls
    config = client.create_calls[0]["vectors_config"]
    assert config.size == 32
    assert config.distance == fake_qdrant.models.Distance.COSINE  # type: ignore[attr-defined]


def test_qdrant_store_upsert_payloads(fake_qdrant: ModuleType) -> None:
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

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
    assert first.payload["chunk_id"] == "fb-1::chunk-0"
    assert first.payload["source_id"] == "fb-1"
    assert first.payload["text"] == "onboarding checklist setup"
    assert first.payload["metadata"] == {"channel": "support_ticket"}
    assert len(first.vector) == 64
    assert store.size == 2


def test_qdrant_store_search_maps_to_search_results(fake_qdrant: ModuleType) -> None:
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

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
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=64, client=client)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    recovered_ids = sorted(chunk.chunk_id for chunk in store.chunks)
    assert recovered_ids == ["fb-1::chunk-0", "fb-2::chunk-0"]


def test_qdrant_store_validates_inputs(fake_qdrant: ModuleType) -> None:
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

    with pytest.raises(ValueError, match="dim must be positive"):
        QdrantVectorStore(dim=0, client=FakeQdrantClient())

    store = QdrantVectorStore(dim=8, client=FakeQdrantClient())
    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search(np.zeros(8, dtype=np.float64), top_k=0)
    with pytest.raises(ValueError, match="dim=8"):
        store.search(np.zeros(4, dtype=np.float64), top_k=1)


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_factory_defaults_to_json_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_engineering_showcase.vector_store import InMemoryVectorStore

    monkeypatch.delenv("AI_SHOWCASE_VECTOR_STORE", raising=False)
    settings = Settings(index_path=tmp_path / "vector_store.json")
    assert settings.vector_store == "json"

    store = load_or_build_index(settings)
    assert isinstance(store, InMemoryVectorStore)


def test_factory_uses_qdrant_when_configured(
    fake_qdrant: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai_engineering_showcase.qdrant_store as qdrant_store
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

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

_QDRANT_TEST = os.environ.get("AI_SHOWCASE_QDRANT_TEST", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _QDRANT_TEST, reason="set AI_SHOWCASE_QDRANT_TEST=1 to run")
def test_qdrant_integration_roundtrip() -> None:
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

    url = os.environ.get("AI_SHOWCASE_QDRANT_URL", "http://localhost:6333")
    collection = "ai_showcase_test_" + os.urandom(4).hex()
    store = QdrantVectorStore(dim=64, url=url, collection_name=collection)
    model = HashingEmbeddingModel(dim=64)
    chunks = _sample_chunks()
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))

    results = store.search(model.embed(["setup checklist"])[0], top_k=1)
    assert results
    assert results[0].chunk.source_id == "fb-1"
