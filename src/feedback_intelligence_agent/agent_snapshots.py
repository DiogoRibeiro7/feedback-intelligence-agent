"""Request-time JSON index refresh with immutable, process-local agent snapshots."""

from __future__ import annotations

from pathlib import Path

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import build_agent, build_agent_snapshot
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.record_ingestion import RecordIngestionResult, ingest_record_batch
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

_FileVersion = tuple[int, int, int, int, int]


def _file_version(path: Path) -> _FileVersion:
    stat = path.stat()
    # File identity detects atomic replacements even with equal size/timestamps.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class IndexSnapshotUnavailableError(RuntimeError):
    """The committed JSON index cannot currently supply a valid agent snapshot."""


class AgentSnapshots:
    """Keep one agent per committed JSON index version in each API worker.

    Cooperating writers use atomic replacement and the shared index writer lock.
    The unchanged-file fast path needs only a stat call; it does not wait for a
    writer preparing an update. Changed files are loaded and published under
    that same lock, serializing reloads with ingestion and preventing stale
    publication. Each request retains its selected agent for its whole answer.
    Qdrant keeps its existing startup behavior without touching a JSON path.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state: tuple[FeedbackInsightAgent, _FileVersion | None]
        if settings.vector_store == "json":
            with index_write_lock(settings.index_path):
                agent = build_agent(settings)
                self._state = agent, _file_version(settings.index_path)
        else:
            self._state = build_agent(settings), None

    def get(self) -> FeedbackInsightAgent:
        """Return the current snapshot, or fail closed until an invalid index recovers.

        A failed reload leaves the last good snapshot intact for in-flight
        requests but raises for new requests. It never rebuilds from the CSV;
        the next request retries the committed file, even after preparation fails.
        """
        if self._settings.vector_store != "json":
            return self._state[0]
        try:
            agent, version = self._state
            if _file_version(self._settings.index_path) == version:
                return agent
            with index_write_lock(self._settings.index_path):
                agent, version = self._state
                current_version = _file_version(self._settings.index_path)
                if current_version != version:
                    store = InMemoryVectorStore.load(self._settings.index_path)
                    agent = build_agent_snapshot(self._settings, store, previous=agent)
                    self._state = agent, current_version
                return agent
        except Exception as exc:
            raise IndexSnapshotUnavailableError(
                "Feedback index is temporarily unavailable"
            ) from exc

    def ingest(self, records: list[FeedbackRecord], tenant_id: str) -> RecordIngestionResult:
        """Persist a tenant batch and publish its prepared snapshot before unlocking."""
        if self._settings.vector_store != "json":
            raise ValueError("Record ingestion requires the JSON vector store")
        with index_write_lock(self._settings.index_path):
            result, agent = ingest_record_batch(
                records, tenant_id, settings=self._settings, agent=self._state[0]
            )
            self._state = agent, _file_version(self._settings.index_path)
            return result
