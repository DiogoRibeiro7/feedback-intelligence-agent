"""Shared local write generations for invalidating cached Qdrant snapshots."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from feedback_intelligence_agent.index_persistence import index_write_lock, write_index_json


class _SnapshotRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    generation: UUID


def _revision_path(lock_path: str | Path) -> Path:
    return Path(f"{Path(lock_path).resolve()}.revision.json")


def qdrant_snapshot_revision(lock_path: str | Path) -> UUID | None:
    """Read the shared generation while the caller holds ``index_write_lock``.

    A missing marker invalidates an existing generation. Malformed markers raise
    instead of silently accepting cached data or overwriting unrelated files.
    """
    try:
        contents = _revision_path(lock_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return _SnapshotRevision.model_validate_json(contents).generation


@contextmanager
def qdrant_write_lock(lock_path: str | Path) -> Iterator[UUID]:
    """Lock writers and invalidate cached snapshots before any remote mutation.

    Use this for every external update to a coordinated collection. The marker
    is atomically replaced before yielding, so a marker-write failure prevents
    the remote write and remote failures still invalidate other workers' caches.
    Hold the context through the complete operation, including cleanup. This is
    a cooperative local lock, not a distributed lock or Qdrant transaction.
    """
    with index_write_lock(lock_path) as canonical_path:
        qdrant_snapshot_revision(canonical_path)
        revision = _SnapshotRevision(version=1, generation=uuid4())
        write_index_json(_revision_path(canonical_path), revision.model_dump_json())
        yield revision.generation
