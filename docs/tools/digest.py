"""Content digests that do not depend on the platform or on git settings.

The documentation manifest records a SHA-256 of every generated artifact so
that drift between the committed manual and the source is detectable. A digest
taken over raw bytes cannot serve that purpose in a repository checked out on
more than one platform: git converts line endings on checkout, so the same
content hashes differently on Windows and on Linux and every digest appears to
have drifted.

Line endings are therefore normalised before hashing. The rule lives here, in
one place, because the generator and the validator must apply exactly the same
one or the manifest is worthless.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["content_digest", "normalise_newlines"]


def normalise_newlines(data: bytes) -> bytes:
    """Return ``data`` with CRLF and lone CR line endings reduced to LF."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def content_digest(path: Path) -> str:
    """Return the SHA-256 of a file's content, ignoring line-ending style."""
    return hashlib.sha256(normalise_newlines(path.read_bytes())).hexdigest()
