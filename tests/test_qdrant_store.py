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
import uuid
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
        Filter=SimpleNamespace,
        FieldCondition=SimpleNamespace,
        MatchAny=SimpleNamespace,
        PointIdsList=SimpleNamespace,
        UpdateStatus=SimpleNamespace(COMPLETED="completed"),
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
        self.delete_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.collections[collection_name] = vectors_config
        self.create_calls.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )

    def upsert(self, *, collection_name: str, points: list[Any], wait: bool) -> SimpleNamespace:
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": points, "wait": wait}
        )
        for point in points:
            self.points[point.id] = point
        return SimpleNamespace(status="completed")

    def delete(
        self, *, collection_name: str, points_selector: SimpleNamespace, wait: bool
    ) -> SimpleNamespace:
        self.delete_calls.append({"points": points_selector.points, "wait": wait})
        for point_id in points_selector.points:
            self.points.pop(point_id, None)
        return SimpleNamespace(status="completed")

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
        scroll_filter: SimpleNamespace | None = None,
    ) -> tuple[list[object], object]:
        self.scroll_calls.append({"filter": scroll_filter, "offset": offset, "limit": limit})
        items = list(self.points.values())
        if scroll_filter is not None:
            sources = scroll_filter.must[0].match.any
            items = [point for point in items if point.payload["source_id"] in sources]
        start = int(offset) if isinstance(offset, int) else 0
        following = start + limit
        return items[start:following], following if following < len(items) else None

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


def test_qdrant_snapshot_is_paginated_and_independent(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    assert store.snapshot().size == 0
    chunks = _source_chunks(words=1030)
    store.add(chunks, np.ones((len(chunks), 8)))
    snapshot = store.snapshot()
    assert snapshot.size == len(chunks) == 515
    assert [call["offset"] for call in client.scroll_calls] == [None, None, 256, 512]
    store.upsert_sources(_source_chunks(words=2), np.ones((1, 8)))
    assert store.size == 1 and snapshot.size == 515


@pytest.mark.parametrize(
    "vector", [None, {"named": [1.0] * 8}, [1.0], [[1.0] * 8], [float("nan")] * 8]
)
def test_qdrant_snapshot_rejects_missing_or_invalid_vectors(
    fake_qdrant: ModuleType, vector: object
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    store.add(_source_chunks(words=2), np.ones((1, 8)))
    next(iter(client.points.values())).vector = vector
    with pytest.raises(QdrantStoreError):
        store.snapshot()
    assert len(client.upsert_calls) == 1


def test_qdrant_store_validates_inputs(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    with pytest.raises(ValueError, match="dim must be positive"):
        QdrantVectorStore(dim=0, client=FakeQdrantClient())

    store = QdrantVectorStore(dim=8, client=FakeQdrantClient())
    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search(np.zeros(8, dtype=np.float64), top_k=0)
    with pytest.raises(ValueError, match="dim=8"):
        store.search(np.zeros(4, dtype=np.float64), top_k=1)


def _source_chunks(
    tenant: str = "acme", feedback_id: str = "shared", *, words: int = 6
) -> list[DocumentChunk]:
    record = FeedbackRecord.model_validate(
        {
            "tenant_id": tenant,
            "feedback_id": feedback_id,
            "customer_segment": "enterprise",
            "channel": "support_ticket",
            "rating": 2,
            "text": "onboarding " * words,
            "created_at": "2026-10-29T10:00:00Z",
        }
    )
    return feedback_to_chunks([record], max_words=2, overlap_words=0)


def test_source_replacement_shrinks_expands_and_preserves_other_tenants(
    fake_qdrant: ModuleType,
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    original = (
        _source_chunks() + _source_chunks("cobalt") + _source_chunks("default", "acme:shared")
    )
    store.add(original, np.ones((len(original), 8)))
    foreign = {
        key: point
        for key, point in client.points.items()
        if point.payload["source_id"] != "acme:shared"
    }

    for words, previous in ((2, 3), (8, 1), (8, 4)):
        incoming = _source_chunks(words=words)
        incoming[0].text = "revised onboarding feedback"
        assert store.upsert_sources(incoming, np.full((len(incoming), 8), 2.0)) == previous
        assert sorted(
            chunk.chunk_id for chunk in store.chunks if chunk.source_id == "acme:shared"
        ) == [chunk.chunk_id for chunk in incoming]
        assert {key: client.points[key] for key in foreign} == foreign
        assert store.size == len(foreign) + len(incoming)
        replaced = [
            point for point in client.points.values() if point.payload["source_id"] == "acme:shared"
        ]
        assert any(point.payload["text"] == "revised onboarding feedback" for point in replaced)
        assert all(point.vector == [2.0] * 8 for point in replaced)
    assert all(call["wait"] is True for call in client.upsert_calls + client.delete_calls)
    assert len(client.delete_calls) == 1


def test_source_replacement_pages_all_matching_chunks_and_uses_actual_point_ids(
    fake_qdrant: ModuleType,
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    original = _source_chunks(words=1030) + _source_chunks("cobalt")
    store.add(original, np.ones((len(original), 8)))
    # Legacy integrations may have used a different point ID for the same chunk.
    point = next(iter(client.points.values()))
    client.points.pop(point.id)
    point.id = 1234
    client.points[1234] = point
    incoming = _source_chunks(words=2)
    assert store.upsert_sources(incoming, np.ones((1, 8))) == 515
    assert 1234 in client.delete_calls[0]["points"]
    assert [call["offset"] for call in client.scroll_calls] == [None, 256, 512]
    assert all(call["filter"].must[0].match.any == ["acme:shared"] for call in client.scroll_calls)
    assert store.size == 4


@pytest.mark.parametrize(
    "problem",
    [
        "rank",
        "count",
        "dim",
        "conversion",
        "nan",
        "inf",
        "duplicate",
        "source",
        "tenant",
        "feedback",
        "chunk",
        "position",
    ],
)
def test_invalid_source_replacement_does_not_read_or_mutate_remote_points(
    fake_qdrant: ModuleType, problem: str
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    original = _source_chunks()
    store.add(original, np.ones((len(original), 8)))
    before = dict(client.points)
    chunks = _source_chunks(words=2)
    vectors = np.ones((1, 8))
    if problem == "rank":
        vectors = np.ones(8)
    elif problem == "count":
        vectors = np.ones((2, 8))
    elif problem == "dim":
        vectors = np.ones((1, 4))
    elif problem == "conversion":
        vectors = np.full((1, 8), "invalid", dtype=object)
    elif problem in {"nan", "inf"}:
        vectors[0, 0] = float(problem)
    elif problem == "duplicate":
        chunks *= 2
        vectors = np.ones((2, 8))
    elif problem == "source":
        chunks[0].source_id = "cobalt:shared"
    elif problem == "tenant":
        chunks[0].metadata["tenant_id"] = "ACME"
    elif problem == "feedback":
        chunks[0].metadata.pop("feedback_id")
    elif problem == "chunk":
        chunks[0].chunk_id = "cobalt:shared::chunk-0"
    else:
        chunks[0].chunk_id = "acme:shared::chunk-01"
    with pytest.raises(ValueError):
        store.upsert_sources(chunks, vectors)
    assert client.points == before
    assert len(client.upsert_calls) == 1
    assert not client.delete_calls
    assert not client.scroll_calls


def test_empty_source_batch_is_a_noop(fake_qdrant: ModuleType) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    assert store.upsert_sources([], np.empty((0, 8))) == 0
    assert not client.upsert_calls and not client.delete_calls and not client.scroll_calls


@pytest.mark.parametrize("representation", ["uuid", "uppercase"])
def test_point_id_normalization_never_deletes_the_replacement(
    fake_qdrant: ModuleType, monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    incoming = _source_chunks(words=2)
    store.add(incoming, np.ones((1, 8)))
    point = next(iter(client.points.values()))
    alternate_id = uuid.UUID(point.id) if representation == "uuid" else point.id.upper()
    monkeypatch.setattr(
        client,
        "scroll",
        lambda **kwargs: ([SimpleNamespace(id=alternate_id, payload=point.payload)], None),
    )
    assert store.upsert_sources(incoming, np.ones((1, 8))) == 1
    assert not client.delete_calls
    assert store.size == 1


@pytest.mark.parametrize(
    "failure",
    [
        "scroll",
        "upsert",
        "upsert_status",
        "upsert_after_commit",
        "delete",
        "delete_status",
        "delete_after_commit",
    ],
)
def test_failed_replacement_can_be_retried_without_losing_other_sources(
    fake_qdrant: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    original = _source_chunks() + _source_chunks("cobalt")
    store.add(original, np.ones((len(original), 8)))
    before = dict(client.points)
    incoming = _source_chunks(words=2)

    def fail(**kwargs: object) -> None:
        raise RuntimeError("remote write failed")

    def incomplete(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="acknowledged")

    method = failure.removesuffix("_status").removesuffix("_after_commit")
    original_method = getattr(client, method)

    def lost_response(**kwargs: object) -> None:
        original_method(**kwargs)
        raise RuntimeError("response lost after commit")

    with monkeypatch.context() as patch:
        replacement = lost_response if failure.endswith("_after_commit") else fail
        patch.setattr(client, method, incomplete if failure.endswith("_status") else replacement)
        with pytest.raises((RuntimeError, QdrantStoreError)):
            store.upsert_sources(incoming, np.ones((1, 8)))
    if method in {"scroll", "upsert"} and not failure.endswith("_after_commit"):
        assert client.points == before
        assert not client.delete_calls
    # A deletion failure leaves the already-written new points plus old trailing chunks.
    assert store.upsert_sources(incoming, np.ones((1, 8))) == (
        1 if failure == "delete_after_commit" else 3
    )
    assert store.size == 4
    for key, point in before.items():
        if point.payload["source_id"] == "cobalt:shared":
            assert client.points[key] is point


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_add_rejects_nonfinite_vectors_before_writes(fake_qdrant: ModuleType, value: float) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    with pytest.raises(ValueError, match="finite"):
        store.add(_source_chunks(words=2), np.full((1, 8), value))
    assert not client.upsert_calls


def test_unexpected_foreign_scroll_result_stops_replacement_before_writes(
    fake_qdrant: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from feedback_intelligence_agent.qdrant_store import QdrantStoreError, QdrantVectorStore

    client = FakeQdrantClient()
    store = QdrantVectorStore(dim=8, client=client)
    store.add(_source_chunks("cobalt", words=2), np.ones((1, 8)))
    monkeypatch.setattr(client, "scroll", lambda **kwargs: (list(client.points.values()), None))
    with pytest.raises(QdrantStoreError, match="outside the replacement sources"):
        store.upsert_sources(_source_chunks(words=2), np.ones((1, 8)))
    assert len(client.upsert_calls) == 1
    assert not client.delete_calls


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
    from qdrant_client import QdrantClient

    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    url = os.environ.get("FEEDBACK_AGENT_QDRANT_URL", "http://localhost:6333")
    collection = "feedback_intelligence_test_" + os.urandom(4).hex()
    client = QdrantClient(url=url)
    try:
        store = QdrantVectorStore(dim=64, client=client, collection_name=collection)
        model = HashingEmbeddingModel(dim=64)
        chunks = _sample_chunks()
        store.add(chunks, model.embed([chunk.text for chunk in chunks]))

        results = store.search(model.embed(["setup checklist"])[0], top_k=1)
        assert results
        assert results[0].chunk.source_id == "fb-1"
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()
