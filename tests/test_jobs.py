from __future__ import annotations

from pathlib import Path

import pytest

from ai_engineering_showcase.jobs import (
    InMemoryJobStore,
    JobNotFoundError,
    JobRequest,
    JobResult,
    JobStatus,
    JsonJobStore,
    run_ingestion_job,
)
from ai_engineering_showcase.vector_store import InMemoryVectorStore

SAMPLE_CSV = "data/sample_feedback.csv"


def test_successful_job_transitions_and_populates_vector_store(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    index_path = tmp_path / "vector_store.json"
    job = store.create(JobRequest(input_path=SAMPLE_CSV, index_path=str(index_path)))
    assert job.status == JobStatus.pending

    result = run_ingestion_job(
        job.job_id,
        store,
        embedding_dim=512,
        default_index_path=str(tmp_path / "unused.json"),
    )

    assert result.status == JobStatus.succeeded
    assert result.error is None
    assert result.chunks is not None and result.chunks > 0
    assert result.index_path == str(index_path)

    # The vector store was actually persisted and is populated.
    assert index_path.exists()
    loaded = InMemoryVectorStore.load(index_path)
    assert loaded.size == result.chunks

    # The store reflects the terminal status.
    stored = store.get(job.job_id)
    assert stored is not None
    assert stored.status == JobStatus.succeeded


def test_job_uses_default_index_path_when_request_omits_one(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    default_index = tmp_path / "default.json"
    job = store.create(JobRequest(input_path=SAMPLE_CSV))
    result = run_ingestion_job(
        job.job_id,
        store,
        embedding_dim=512,
        default_index_path=str(default_index),
    )
    assert result.status == JobStatus.succeeded
    assert result.index_path == str(default_index)
    assert default_index.exists()


def test_failed_job_has_clean_non_leaky_error(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    missing = tmp_path / "does-not-exist.csv"
    job = store.create(JobRequest(input_path=str(missing), index_path=str(tmp_path / "out.json")))

    result = run_ingestion_job(
        job.job_id,
        store,
        embedding_dim=512,
        default_index_path=str(tmp_path / "out.json"),
    )

    assert result.status == JobStatus.failed
    assert result.chunks is None
    assert result.error is not None
    # No filesystem internals, paths, or stack traces leak into the message.
    assert str(missing) not in result.error
    assert "Traceback" not in result.error
    assert ".csv" not in result.error
    assert "Ingestion failed" in result.error


def test_failed_job_on_invalid_data_does_not_leak_contents(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("not,the,right,columns\n1,2,3,4\n", encoding="utf-8")
    store = InMemoryJobStore()
    job = store.create(JobRequest(input_path=str(bad_csv)))

    result = run_ingestion_job(
        job.job_id,
        store,
        embedding_dim=512,
        default_index_path=str(tmp_path / "out.json"),
    )
    assert result.status == JobStatus.failed
    assert result.error is not None
    assert str(bad_csv) not in result.error


def test_run_unknown_job_raises(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    with pytest.raises(JobNotFoundError):
        run_ingestion_job(
            "unknown",
            store,
            embedding_dim=512,
            default_index_path=str(tmp_path / "out.json"),
        )


def test_in_memory_store_round_trip_and_isolation() -> None:
    store = InMemoryJobStore()
    job = store.create(JobRequest(input_path=SAMPLE_CSV))
    fetched = store.get(job.job_id)
    assert fetched is not None
    # Mutating the returned copy does not change stored state.
    fetched.status = JobStatus.succeeded
    again = store.get(job.job_id)
    assert again is not None
    assert again.status == JobStatus.pending
    assert store.get("missing") is None


def test_in_memory_update_unknown_raises() -> None:
    store = InMemoryJobStore()
    orphan = JobResult(
        job_id="nope", status=JobStatus.running, request=JobRequest(input_path=SAMPLE_CSV)
    )
    with pytest.raises(JobNotFoundError):
        store.update(orphan)


def test_json_store_round_trip(tmp_path: Path) -> None:
    store = JsonJobStore(tmp_path / "jobs")
    job = store.create(JobRequest(input_path=SAMPLE_CSV, index_path="idx.json"))
    assert (tmp_path / "jobs" / f"{job.job_id}.json").exists()

    loaded = store.get(job.job_id)
    assert loaded is not None
    assert loaded.job_id == job.job_id
    assert loaded.request.input_path == SAMPLE_CSV
    assert loaded.status == JobStatus.pending

    loaded.status = JobStatus.succeeded
    loaded.chunks = 7
    store.update(loaded)

    reloaded = store.get(job.job_id)
    assert reloaded is not None
    assert reloaded.status == JobStatus.succeeded
    assert reloaded.chunks == 7
    assert reloaded.updated_at >= reloaded.created_at

    assert store.get("missing") is None


def test_json_store_update_unknown_raises(tmp_path: Path) -> None:
    store = JsonJobStore(tmp_path / "jobs")
    orphan = JobResult(
        job_id="nope", status=JobStatus.running, request=JobRequest(input_path=SAMPLE_CSV)
    )
    with pytest.raises(JobNotFoundError):
        store.update(orphan)
