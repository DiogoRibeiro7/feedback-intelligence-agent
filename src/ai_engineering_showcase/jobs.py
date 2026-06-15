"""Asynchronous ingestion jobs with status tracking.

This module decouples ingestion from the request/response cycle. A client
submits an ingestion request, receives a job identifier immediately, and polls
for the terminal status (``succeeded`` / ``failed``) without blocking on the
embedding and indexing work.

The pieces are:

1. Typed job models (:class:`JobStatus`, :class:`JobRequest`, :class:`JobResult`).
2. A :class:`JobStore` protocol with two backends: an in-memory store guarded by
   a lock (suitable for FastAPI ``BackgroundTasks``) and a JSON-backed store that
   persists one file per job.
3. A single :func:`run_ingestion_job` pipeline that loads + validates a CSV,
   chunks it, builds embeddings, persists the vector store, and records success
   or a clean failure message. The full exception detail is logged server-side
   but never surfaced to the client, so secrets, stack traces, and filesystem
   internals do not leak.

The pipeline is a plain function so it is directly unit-testable and is also the
exact callable scheduled by the API's ``BackgroundTasks``.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.data_contracts import DataContractError
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.factory import chunk_to_embedding_text
from ai_engineering_showcase.ingestion import FeedbackIngestionError, load_feedback_csv
from ai_engineering_showcase.telemetry import Telemetry, get_logger
from ai_engineering_showcase.vector_store import InMemoryVectorStore

__all__ = [
    "InMemoryJobStore",
    "JobNotFoundError",
    "JobRequest",
    "JobResult",
    "JobStatus",
    "JobStore",
    "JsonJobStore",
    "new_job_id",
    "run_ingestion_job",
]

_CLEAN_INGESTION_ERROR = (
    "Ingestion failed: the input data could not be loaded or validated. "
    "Check that the CSV path is correct and the rows satisfy the data contract."
)
_CLEAN_UNEXPECTED_ERROR = "Ingestion failed due to an unexpected internal error."


def _utcnow() -> datetime:
    """Return the current UTC time (extracted for deterministic testing)."""
    return datetime.now(timezone.utc)


def new_job_id() -> str:
    """Return a fresh job identifier."""
    return uuid.uuid4().hex


class JobStatus(str, Enum):
    """Lifecycle status of an ingestion job."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobRequest(BaseModel):
    """Request describing an ingestion job to run.

    Attributes:
        input_path: Path to the feedback CSV to ingest.
        index_path: Optional output path for the JSON vector store. When omitted
            the caller's configured default index path is used.
    """

    input_path: str = Field(min_length=1)
    index_path: str | None = None


class JobResult(BaseModel):
    """Status and outcome of an ingestion job.

    ``error`` carries a clean, non-leaky message on failure (the full exception
    is logged server-side, never stored here). ``chunks`` is populated on
    success with the number of indexed chunks.
    """

    job_id: str
    status: JobStatus
    request: JobRequest
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    chunks: int | None = None
    index_path: str | None = None
    error: str | None = None


class JobStore(Protocol):
    """Protocol implemented by job persistence backends."""

    def create(self, request: JobRequest) -> JobResult:
        """Create and persist a new ``pending`` job, returning it."""
        ...

    def get(self, job_id: str) -> JobResult | None:
        """Return the stored job, or ``None`` when unknown."""
        ...

    def update(self, result: JobResult) -> None:
        """Persist the full (mutated) job result."""
        ...


class JobNotFoundError(KeyError):
    """Raised when updating a job that does not exist in the store."""


class InMemoryJobStore:
    """Thread-safe, dict-backed job store.

    A lock guards every mutation so the store is safe to share across FastAPI
    ``BackgroundTasks`` (which may run on a worker thread) and request handlers.
    Stored results are deep-copied on the way in and out, so callers cannot
    mutate persisted state by accident.
    """

    def __init__(self) -> None:
        """Initialise the empty store."""
        self._jobs: dict[str, JobResult] = {}
        self._lock = threading.Lock()

    def create(self, request: JobRequest) -> JobResult:
        """Create and store a new ``pending`` job."""
        result = JobResult(job_id=new_job_id(), status=JobStatus.pending, request=request)
        with self._lock:
            self._jobs[result.job_id] = result.model_copy(deep=True)
        return result.model_copy(deep=True)

    def get(self, job_id: str) -> JobResult | None:
        """Return a deep copy of the stored job, or ``None``."""
        with self._lock:
            result = self._jobs.get(job_id)
            return result.model_copy(deep=True) if result is not None else None

    def update(self, result: JobResult) -> None:
        """Persist a deep copy of the job, refreshing ``updated_at``."""
        with self._lock:
            if result.job_id not in self._jobs:
                raise JobNotFoundError(result.job_id)
            result.updated_at = _utcnow()
            self._jobs[result.job_id] = result.model_copy(deep=True)


class JsonJobStore:
    """Job store that persists one JSON file per job under a root directory.

    Files are written as ``{root}/{job_id}.json`` with indented JSON, so jobs are
    easy to inspect and survive process restarts. A lock guards in-process
    concurrent writes.
    """

    def __init__(self, root: str | Path) -> None:
        """Create the store and ensure the root directory exists."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        """Return the JSON file path for a job identifier."""
        if not job_id:
            raise JobNotFoundError(job_id)
        return self.root / f"{job_id}.json"

    def create(self, request: JobRequest) -> JobResult:
        """Create and persist a new ``pending`` job."""
        result = JobResult(job_id=new_job_id(), status=JobStatus.pending, request=request)
        self._write(result)
        return result

    def get(self, job_id: str) -> JobResult | None:
        """Load a job from its JSON file, or ``None`` when missing."""
        path = self._path(job_id)
        if not path.exists():
            return None
        return JobResult.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, result: JobResult) -> None:
        """Persist the job, refreshing ``updated_at``."""
        if not self._path(result.job_id).exists():
            raise JobNotFoundError(result.job_id)
        result.updated_at = _utcnow()
        self._write(result)

    def _write(self, result: JobResult) -> None:
        """Write the job as an indented JSON file under the lock."""
        path = self._path(result.job_id)
        with self._lock:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def run_ingestion_job(
    job_id: str,
    store: JobStore,
    *,
    embedding_dim: int,
    default_index_path: str | Path,
    telemetry: Telemetry | None = None,
) -> JobResult:
    """Execute one ingestion job and record its terminal status.

    The pipeline mirrors the synchronous ingestion path: load + validate the CSV
    (reusing :func:`load_feedback_csv` and the data contract), chunk the records,
    embed them, and persist the resulting JSON vector store. The job status
    transitions ``pending`` -> ``running`` -> ``succeeded``/``failed``.

    On failure a clean, non-leaky message is stored on the job; the full
    exception is logged server-side via the package logger but never surfaced to
    the client (no secrets, stack traces, or filesystem internals leak).

    Args:
        job_id: Identifier of a job already created in ``store``.
        store: Job store holding the job; updated in place as the job runs.
        embedding_dim: Dimension for the hashing embedding model.
        default_index_path: Index path used when the request omits one.
        telemetry: Optional telemetry emitter passed through to ingestion.

    Returns:
        The terminal :class:`JobResult`.

    Raises:
        JobNotFoundError: If ``job_id`` is unknown to the store.
    """
    result = store.get(job_id)
    if result is None:
        raise JobNotFoundError(job_id)

    result.status = JobStatus.running
    store.update(result)

    index_path = result.request.index_path or str(default_index_path)
    try:
        records = load_feedback_csv(result.request.input_path, telemetry=telemetry)
        chunks = feedback_to_chunks(records)
        embedding_model = HashingEmbeddingModel(dim=embedding_dim)
        vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
        vector_store = InMemoryVectorStore(dim=embedding_dim)
        vector_store.add(chunks, vectors)
        vector_store.save(index_path)
    except (FeedbackIngestionError, DataContractError, FileNotFoundError, ValueError):
        get_logger().exception("ingestion_job_failed", extra={"job_id": job_id})
        result.status = JobStatus.failed
        result.error = _CLEAN_INGESTION_ERROR
        store.update(result)
        return store.get(job_id) or result
    except Exception:  # noqa: BLE001 - never leak unexpected internals to the client.
        get_logger().exception("ingestion_job_failed", extra={"job_id": job_id})
        result.status = JobStatus.failed
        result.error = _CLEAN_UNEXPECTED_ERROR
        store.update(result)
        return store.get(job_id) or result

    result.status = JobStatus.succeeded
    result.chunks = vector_store.size
    result.index_path = index_path
    result.error = None
    store.update(result)
    return store.get(job_id) or result
