from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from dataexcept import DataLoadingError

from feedback_intelligence_agent.chunking import feedback_to_chunks, scoped_source_id
from feedback_intelligence_agent.index_updates import update_json_index
from feedback_intelligence_agent.schemas import DocumentChunk, FeedbackRecord
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def record(tenant: str, feedback_id: str) -> FeedbackRecord:
    return FeedbackRecord(
        tenant_id=tenant,
        feedback_id=feedback_id,
        customer_segment="enterprise",
        channel="support_ticket",
        rating=2,
        text=f"Feedback belonging to {tenant} with identifier {feedback_id}",
        created_at="2026-10-28T00:00:00Z",
    )


@pytest.mark.parametrize(
    ("tenant", "feedback_id", "expected"),
    [
        ("default", "fb-1", "fb-1"),
        ("acme", "fb-1", "acme:fb-1"),
        ("default", "acme:shared", "acme%3Ashared"),
        ("acme:eu", "shared", "acme%3Aeu:shared"),
        ("acme", "eu:shared", "acme:eu%3Ashared"),
        ("default", "acme%3Ashared", "acme%253Ashared"),
        ("default", "fb::chunk-0", "fb%3A%3Achunk-0"),
    ],
)
def test_source_ids_escape_components(tenant: str, feedback_id: str, expected: str) -> None:
    [chunk] = feedback_to_chunks([record(tenant, feedback_id)])
    assert chunk.source_id == expected
    assert chunk.chunk_id == f"{expected}::chunk-0"
    assert chunk.metadata["feedback_id"] == feedback_id


def test_source_ids_are_unique_for_delimiters_percent_escapes_and_unicode() -> None:
    values = ["default", "a", "b", "a:b", "b:a", "a%3Ab", "%", "::chunk-0", "a/b", "é", "空"]
    records = [record(tenant, feedback_id) for tenant, feedback_id in product(values, repeat=2)]
    assert len({scoped_source_id(item) for item in records}) == len(records)
    assert len({chunk.chunk_id for chunk in feedback_to_chunks(records)}) == len(records)


def legacy_payload(records: list[FeedbackRecord]) -> dict[str, object]:
    chunks = []
    for item in records:
        [chunk] = feedback_to_chunks([item])
        old_source = (
            item.feedback_id
            if item.tenant_id == "default"
            else f"{item.tenant_id}:{item.feedback_id}"
        )
        chunks.append(
            chunk.model_copy(
                update={"source_id": old_source, "chunk_id": f"{old_source}::chunk-0"}
            ).model_dump()
        )
    return {"dim": 64, "chunks": chunks, "vectors": np.eye(len(chunks), 64).tolist()}


@pytest.mark.parametrize("explicit_version", [False, True])
def test_loading_legacy_index_migrates_collisions_without_writing_or_reembedding(
    tmp_path: Path, explicit_version: bool
) -> None:
    records = [
        record("default", "acme:shared"),
        record("acme", "shared"),
        record("acme:eu", "shared"),
        record("acme", "eu:shared"),
        record("default", "acme%3Ashared"),
    ]
    payload = legacy_payload(records)
    if explicit_version:
        payload["source_id_version"] = 1
    path = tmp_path / "legacy.json"
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")

    store = InMemoryVectorStore.load(path)

    assert path.read_text(encoding="utf-8") == original
    expected = feedback_to_chunks(records)
    assert store.chunks == expected
    for index, vector in enumerate(np.eye(len(records), 64)):
        [match] = store.search(vector, top_k=1)
        assert match.chunk == expected[index]
        assert match.score == 1.0
    migrated = tmp_path / "migrated.json"
    store.save(migrated)
    saved = json.loads(migrated.read_text(encoding="utf-8"))
    assert saved["source_id_version"] == 2
    assert saved["vectors"] == payload["vectors"]
    assert InMemoryVectorStore.load(migrated).chunks == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_updates_preserve_other_records_when_legacy_ids_collide(
    tmp_path: Path, legacy: bool
) -> None:
    records = [record("default", "acme:shared"), record("acme", "shared")]
    path = tmp_path / "index.json"
    if legacy:
        path.write_text(json.dumps(legacy_payload(records)), encoding="utf-8")
    else:
        for item in records:
            result = update_json_index([item], path, embedding_dim=64)
            assert result.inserted_records == 1
            assert result.updated_records == 0
    replacement = records[1].model_copy(update={"text": "Corrected tenant feedback"})

    result = update_json_index([replacement], path, embedding_dim=64)

    assert result.updated_records == 1
    assert result.inserted_records == 0
    assert result.removed_chunks == 1
    assert result.total_chunks == 2
    assert InMemoryVectorStore.load(path).chunks == feedback_to_chunks([records[0], replacement])


@pytest.mark.parametrize("tenant", ["default", "acme:eu"])
def test_legacy_index_without_feedback_id_metadata_migrates(tmp_path: Path, tenant: str) -> None:
    prefix = "" if tenant == "default" else f"{tenant}:"
    chunk = DocumentChunk(
        source_id=f"{prefix}feedback:1",
        chunk_id=f"{prefix}feedback:1::chunk-2",
        text="Legacy feedback",
        metadata={} if tenant == "default" else {"tenant_id": tenant},
    )
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"dim": 1, "chunks": [chunk.model_dump()], "vectors": [[1.0]]}),
        encoding="utf-8",
    )
    [migrated] = InMemoryVectorStore.load(path).chunks
    encoded_prefix = "" if tenant == "default" else "acme%3Aeu:"
    assert migrated.source_id == f"{encoded_prefix}feedback%3A1"
    assert migrated.chunk_id == f"{encoded_prefix}feedback%3A1::chunk-2"


@pytest.mark.parametrize(
    "problem", ["metadata", "chunk_id", "duplicate", "version", "tenant", "missing_identity"]
)
def test_invalid_legacy_identity_fails_without_overwriting_index(
    tmp_path: Path, problem: str
) -> None:
    payload = legacy_payload([record("acme", "eu:shared")])
    chunks = payload["chunks"]
    assert isinstance(chunks, list)
    if problem == "metadata":
        chunks[0]["metadata"]["feedback_id"] = "different"
    elif problem == "tenant":
        chunks[0]["metadata"]["tenant_id"] = None
    elif problem == "missing_identity":
        chunks[0]["metadata"].pop("feedback_id")
        chunks[0]["source_id"] = "unqualified"
    elif problem == "chunk_id":
        chunks[0]["chunk_id"] = "unrelated"
    elif problem == "duplicate":
        chunks.append(chunks[0])
        payload["vectors"] = np.eye(2, 64).tolist()
    else:
        payload["source_id_version"] = 99
    path = tmp_path / "invalid.json"
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")

    with pytest.raises(DataLoadingError):
        update_json_index([record("acme", "new")], path, embedding_dim=64)

    assert path.read_text(encoding="utf-8") == original
