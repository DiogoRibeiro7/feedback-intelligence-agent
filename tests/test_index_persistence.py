from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from threading import Event
from typing import TextIO

import numpy as np
import numpy.typing as npt
import pytest
from dataexcept import FileWriteError
from portalocker.exceptions import LockException

import feedback_intelligence_agent.factory as factory_module
import feedback_intelligence_agent.index_persistence as persistence_module
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.index_persistence import index_write_lock
from feedback_intelligence_agent.index_updates import IncrementalIndexResult, update_json_index
from feedback_intelligence_agent.jobs import (
    InMemoryJobStore,
    JobRequest,
    JobStatus,
    run_ingestion_job,
)
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def record(source_id: str) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=source_id,
        text=f"onboarding {source_id}",
        rating=2,
        channel="support_ticket",
        customer_segment="enterprise",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("failure_point", ["write", "fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_preserves_previous_index_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str, existing: bool
) -> None:
    path = tmp_path / "index.json"
    if existing:
        update_json_index([record("original")], path, embedding_dim=16)
    previous = path.read_bytes() if existing else None

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated persistence failure")

    if failure_point == "write":
        original_temporary_file = persistence_module.NamedTemporaryFile

        @contextmanager
        def failing_temporary_file(*args: object, **kwargs: object) -> Iterator[TextIO]:
            with original_temporary_file(*args, **kwargs) as temporary:
                original_write = temporary.write

                def partial_write(contents: str) -> int:
                    original_write(contents[:10])
                    temporary.flush()
                    raise OSError("simulated partial write")

                temporary.write = partial_write
                yield temporary

        monkeypatch.setattr(persistence_module, "NamedTemporaryFile", failing_temporary_file)
    else:
        monkeypatch.setattr(os, failure_point, fail)
    with pytest.raises(FileWriteError) as error:
        InMemoryVectorStore(dim=16).save(path)

    assert error.value.path == str(path)
    assert error.value.original is error.value.__cause__
    if existing:
        assert path.read_bytes() == previous
        assert InMemoryVectorStore.load(path).chunks[0].source_id == "original"
    else:
        assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_thread_updates_keep_both_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "index.json"
    if existing:
        update_json_index([record("original")], path, embedding_dim=16)
    first_embedded = Event()
    release_first = Event()
    second_started = Event()
    second_embedded = Event()
    original_embed = HashingEmbeddingModel.embed

    def controlled_embed(self: HashingEmbeddingModel, texts: list[str]) -> npt.NDArray[np.float64]:
        if "first" in texts[0]:
            first_embedded.set()
            assert release_first.wait(10), "first update was never released"
        elif "second" in texts[0]:
            second_embedded.set()
        return original_embed(self, texts)

    def second_update() -> IncrementalIndexResult:
        second_started.set()
        return update_json_index([record("second")], path, embedding_dim=16)

    monkeypatch.setattr(HashingEmbeddingModel, "embed", controlled_embed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(update_json_index, [record("first")], path, embedding_dim=16)
        try:
            assert first_embedded.wait(10)
            second = executor.submit(second_update)
            assert second_started.wait(10)
            overlapped = second_embedded.wait(0.3)
        finally:
            release_first.set()
        first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)

    assert not overlapped, "second update read the index before the first committed"
    assert first_result.created is not existing
    assert second_result.created is False
    expected = {"first", "second"} | ({"original"} if existing else set())
    assert {chunk.source_id for chunk in InMemoryVectorStore.load(path).chunks} == expected


def process_update(path: str, source_id: str) -> None:
    update_json_index([record(source_id)], path, embedding_dim=16)


def test_updates_from_separate_processes_keep_every_batch(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=process_update, args=(str(path), f"batch-{index}"))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    assert {chunk.source_id for chunk in InMemoryVectorStore.load(path).chunks} == {
        f"batch-{index}" for index in range(4)
    }


def test_readers_see_complete_snapshots_at_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.json"
    update_json_index([record("original")], path, embedding_dim=16)
    original_replace = os.replace
    replaced = False

    def inspect_replace(source: Path, destination: Path) -> None:
        nonlocal replaced
        assert source.parent == destination.parent == tmp_path
        assert InMemoryVectorStore.load(path).size == 1
        assert InMemoryVectorStore.load(source).size == 2
        original_replace(source, destination)
        replaced = True

    monkeypatch.setattr(os, "replace", inspect_replace)
    update_json_index([record("second")], path, embedding_dim=16)
    assert replaced
    assert InMemoryVectorStore.load(path).size == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_is_reentrant_for_relative_and_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "index.json"
    with index_write_lock(path), index_write_lock("index.json", timeout=0) as canonical:
        assert canonical == path
        update_json_index([record("nested")], "index.json", embedding_dim=16)
    assert InMemoryVectorStore.load(path).size == 1


def test_lock_timeout_and_unrelated_index_progress(tmp_path: Path) -> None:
    path = tmp_path / "index.json"

    def attempt_locked_write() -> None:
        with index_write_lock(path, timeout=0):
            pytest.fail("a different thread acquired the held lock")

    with index_write_lock(path), ThreadPoolExecutor(max_workers=1) as executor:
        attempt = executor.submit(attempt_locked_write)
        with pytest.raises(FileWriteError) as error:
            attempt.result(timeout=5)
        assert error.value.path == str(path)
        assert isinstance(error.value.original, LockException)
        unrelated = executor.submit(
            update_json_index, [record("other")], tmp_path / "other.json", embedding_dim=16
        )
        assert unrelated.result(timeout=5).total_chunks == 1
    with index_write_lock(path, timeout=0):
        pass


def hold_process_lock(path: str, acquired: ProcessEvent, release: ProcessEvent) -> None:
    with index_write_lock(path):
        acquired.set()
        release.wait(30)


def test_process_exit_releases_lock(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    process = context.Process(target=hold_process_lock, args=(str(path), acquired, release))
    process.start()
    try:
        assert acquired.wait(15)
        with pytest.raises(FileWriteError) as error, index_write_lock(path, timeout=0):
            pytest.fail("another process acquired the held lock")
        assert isinstance(error.value.original, LockException)
    finally:
        process.terminate()
        process.join(timeout=10)
    assert not process.is_alive()
    with index_write_lock(path, timeout=2):
        pass


def test_initializer_rechecks_index_after_waiting_for_an_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.json"
    attempted = Event()

    @contextmanager
    def observed_lock(target: str | Path) -> Iterator[Path]:
        attempted.set()
        with index_write_lock(target) as canonical:
            yield canonical

    monkeypatch.setattr(factory_module, "index_write_lock", observed_lock)
    settings = Settings(index_path=path, embedding_dim=64)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with index_write_lock(path):
            initializer = executor.submit(factory_module.load_or_build_index, settings)
            assert attempted.wait(10)
            update_json_index([record("first")], path, embedding_dim=64)
        result = initializer.result(timeout=10)
    assert {chunk.source_id for chunk in result.chunks} == {"first"}
    assert {chunk.source_id for chunk in InMemoryVectorStore.load(path).chunks} == {"first"}


@pytest.mark.parametrize("writer", ["save", "rebuild", "job"])
def test_all_index_writers_respect_an_existing_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    path = tmp_path / "index.json"
    attempted = Event()

    @contextmanager
    def observed_lock(target: str | Path) -> Iterator[Path]:
        attempted.set()
        with index_write_lock(target) as canonical:
            yield canonical

    def write() -> None:
        if writer == "save":
            InMemoryVectorStore(dim=16).save(path)
        elif writer == "rebuild":
            factory_module.build_index("data/sample_feedback.csv", path, embedding_dim=16)
        else:
            jobs = InMemoryJobStore()
            job = jobs.create(JobRequest(input_path="data/sample_feedback.csv"))
            result = run_ingestion_job(job.job_id, jobs, embedding_dim=16, default_index_path=path)
            assert result.status == JobStatus.succeeded

    monkeypatch.setattr(persistence_module, "index_write_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with index_write_lock(path):
            pending = executor.submit(write)
            assert attempted.wait(10)
            assert not pending.done()
            assert not path.exists()
        pending.result(timeout=10)
    assert path.exists()
    InMemoryVectorStore.load(path)


def test_failed_update_releases_lock_and_allows_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.json"
    update_json_index([record("original")], path, embedding_dim=16)
    before = path.read_bytes()
    with monkeypatch.context() as fault:

        def fail_embed(self: HashingEmbeddingModel, texts: list[str]) -> npt.NDArray[np.float64]:
            raise ValueError("simulated embedding failure")

        fault.setattr(HashingEmbeddingModel, "embed", fail_embed)
        with pytest.raises(ValueError, match="embedding failure"):
            update_json_index([record("failed")], path, embedding_dim=16)
    assert path.read_bytes() == before
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(update_json_index, [record("recovered")], path, embedding_dim=16)
        assert result.result(timeout=5).total_chunks == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not supported on Windows")
def test_save_preserves_existing_file_permissions(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    InMemoryVectorStore(dim=16).save(path)
    path.chmod(0o640)
    InMemoryVectorStore(dim=16).save(path)
    assert path.stat().st_mode & 0o777 == 0o640
