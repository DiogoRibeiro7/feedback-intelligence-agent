"""Local generation markers invalidate cache before potentially partial remote writes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest

import feedback_intelligence_agent.qdrant_coordination as coordination_module
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.qdrant_coordination import (
    qdrant_snapshot_revision,
    qdrant_write_lock,
)


def test_generation_changes_before_remote_work_and_survives_failure(tmp_path: Path) -> None:
    path = tmp_path / "coordinator"
    assert qdrant_snapshot_revision(path) is None
    with qdrant_write_lock(path) as first:
        assert qdrant_snapshot_revision(path) == first
    with pytest.raises(RuntimeError, match="remote failure"), qdrant_write_lock(path) as second:
        assert qdrant_snapshot_revision(path) == second != first
        raise RuntimeError("remote failure")
    assert qdrant_snapshot_revision(path) == second
    assert json.loads(Path(f"{path}.revision.json").read_text(encoding="utf-8")) == {
        "version": 1,
        "generation": str(second),
    }
    assert not path.exists()  # The logical lock path is never overwritten.


def test_failed_marker_save_never_enters_remote_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coordinator"
    with qdrant_write_lock(path) as previous:
        pass

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("marker persistence failed")

    monkeypatch.setattr(coordination_module, "write_index_json", fail)
    with pytest.raises(OSError, match="marker persistence failed"), qdrant_write_lock(path):
        pytest.fail("Remote work must not start when invalidation cannot be persisted")
    assert qdrant_snapshot_revision(path) == previous


@pytest.mark.parametrize(
    "contents",
    [
        "broken json",
        "{}",
        '{"version":2,"generation":"invalid"}',
        '{"generation":"00000000-0000-0000-0000-000000000001"}',
        '{"version":1,"generation":"00000000-0000-0000-0000-000000000001","extra":"data"}',
    ],
)
def test_unrecognized_marker_is_not_overwritten(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "coordinator"
    revision = Path(f"{path}.revision.json")
    revision.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        qdrant_snapshot_revision(path)
    with pytest.raises(ValueError), qdrant_write_lock(path):
        pytest.fail("Invalid metadata must prevent remote writes")
    assert revision.read_text(encoding="utf-8") == contents


def test_nested_and_concurrent_writers_share_one_canonical_generation(tmp_path: Path) -> None:
    path = tmp_path / "coordinator"
    alias = tmp_path / "alias" / ".." / "coordinator"
    with index_write_lock(alias), qdrant_write_lock(path) as first:
        assert qdrant_snapshot_revision(alias) == first
    observed: list[UUID] = []

    def write(_: int) -> None:
        with qdrant_write_lock(alias) as generation:
            assert qdrant_snapshot_revision(path) == generation
            observed.append(generation)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(12)))
    assert len(set(observed)) == 12
    assert first not in observed
    assert qdrant_snapshot_revision(path) == observed[-1]


def test_cache_configuration_defaults_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FEEDBACK_AGENT_QDRANT_SNAPSHOT_TTL_SECONDS", raising=False)
    assert Settings(_env_file=None).qdrant_snapshot_ttl_seconds == 0
    monkeypatch.setenv("FEEDBACK_AGENT_QDRANT_SNAPSHOT_TTL_SECONDS", "15")
    settings = Settings(
        _env_file=None, vector_store="qdrant", qdrant_ingestion_lock_path=tmp_path / "coordinator"
    )
    assert settings.qdrant_snapshot_ttl_seconds == 15
    with pytest.raises(ValueError, match="coordinated Qdrant"):
        Settings(_env_file=None, vector_store="qdrant", qdrant_ingestion_lock_path=None)
    with pytest.raises(ValueError, match="coordinated Qdrant"):
        Settings(
            _env_file=None, vector_store="json", qdrant_ingestion_lock_path=tmp_path / "coordinator"
        )


@pytest.mark.parametrize("ttl", [-1, 3601, float("inf"), float("nan")])
def test_cache_configuration_rejects_invalid_lifetimes(tmp_path: Path, ttl: float) -> None:
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            vector_store="qdrant",
            qdrant_ingestion_lock_path=tmp_path / "coordinator",
            qdrant_snapshot_ttl_seconds=ttl,
        )
