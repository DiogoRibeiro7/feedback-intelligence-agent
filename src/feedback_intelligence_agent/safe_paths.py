"""Constrain JSON record identifiers to one file under a configured store root."""

from __future__ import annotations

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
    directory = root.resolve()
    destination = (directory / f"{identifier}.json").resolve()
    if destination.parent != directory:
        raise ValueError("record path must stay inside its store directory")
    return destination
