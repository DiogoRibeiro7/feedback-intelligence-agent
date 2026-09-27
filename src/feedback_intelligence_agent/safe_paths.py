"""Constrain JSON record identifiers to one file under a configured store root."""

from __future__ import annotations

import os
from pathlib import Path


def json_record_path(root: Path, identifier: str) -> Path:
    """Return a store file, rejecting traversal and symlinks outside the root."""
    if not (
        1 <= len(identifier) <= 64
        and identifier.isascii()
        and identifier[0].isalnum()
        and all(character.isalnum() or character in "._-" for character in identifier)
    ):
        raise ValueError("invalid record identifier")
    directory = os.path.realpath(root)
    destination = os.path.realpath(os.path.join(directory, f"{identifier}.json"))
    prefix = directory if directory.endswith(os.sep) else directory + os.sep
    if not destination.startswith(prefix) or os.path.dirname(destination) != directory:
        raise ValueError("record path must stay inside its store directory")
    return Path(destination)
