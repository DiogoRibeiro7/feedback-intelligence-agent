"""Atomic JSON index replacement and cooperative locking on local filesystems."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from stat import S_IMODE
from tempfile import NamedTemporaryFile
from threading import Lock
from weakref import WeakValueDictionary

from dataexcept import FileWriteError, wrapping
from filelock import BaseFileLock, FileLock, Timeout

_LOCKS: WeakValueDictionary[str, BaseFileLock] = WeakValueDictionary()
_REGISTRY_LOCK = Lock()


@contextmanager
def index_write_lock(path: str | Path, *, timeout: float = 60.0) -> Iterator[Path]:
    """Lock a canonical index path across threads and processes, yielding that path.

    Hold this lock across the entire read/modify/write operation. Nested calls
    on the same thread are reentrant, including saves inside an update. The
    sidecar ``<index>.lock`` belongs to the locking library; do not remove it
    while processes may use the index. Lock acquisition failures retain file
    context in ``FileWriteError``. Locks require a cooperating local filesystem.
    """
    output_path = Path(path)
    with wrapping(OSError, FileWriteError, path=str(output_path.parent)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with wrapping(OSError, FileWriteError, path=str(output_path)):
        output_path = output_path.resolve()
    key = os.path.normcase(str(output_path))
    # Serialize registry creation as well as writes: two first-time callers
    # must share the same reentrant object when a save nests inside an update.
    with _REGISTRY_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = FileLock(f"{key}.lock", mode=0o600, thread_local=True)
            _LOCKS[key] = lock
    with wrapping((OSError, Timeout), FileWriteError, path=str(output_path)):
        lock.acquire(timeout=timeout)
    try:
        yield output_path
    finally:
        lock.release()


def write_index_json(path: str | Path, contents: str) -> None:
    """Flush JSON to a sibling temporary file, then atomically replace the index.

    Serialization must finish before calling this function. Failed writes,
    flushes, or replacements leave the previous index intact and clean up the
    temporary file where possible. Readers can keep using the old snapshot
    until replacement succeeds. Existing file permission bits are retained.
    """
    with index_write_lock(path) as output_path:
        temporary_path: Path | None = None
        with wrapping((OSError, UnicodeError), FileWriteError, path=str(output_path)):
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=output_path.parent,
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    if output_path.exists():
                        os.chmod(temporary_path, S_IMODE(output_path.stat().st_mode))
                    temporary.write(contents)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, output_path)
            finally:
                if temporary_path is not None:
                    with suppress(OSError):
                        temporary_path.unlink(missing_ok=True)
