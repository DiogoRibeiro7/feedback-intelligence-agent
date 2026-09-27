"""JSON store paths stay inside their configured directory."""

from pathlib import Path

import pytest

from feedback_intelligence_agent.safe_paths import json_record_path


@pytest.mark.parametrize("identifier", ["../private", "record/other", "record\n", "é", ""])
def test_record_path_rejects_unsafe_identifier(tmp_path: Path, identifier: str) -> None:
    with pytest.raises(ValueError):
        json_record_path(tmp_path, identifier)


def test_record_path_rejects_symlink_to_outside_directory(tmp_path: Path) -> None:
    root = tmp_path / "records"
    root.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text("private", encoding="utf-8")
    (root / "record.json").symlink_to(outside)

    with pytest.raises(ValueError, match="inside its store directory"):
        json_record_path(root, "record")
    assert outside.read_text(encoding="utf-8") == "private"


def test_record_path_accepts_direct_child(tmp_path: Path) -> None:
    assert json_record_path(tmp_path, "feedback-1") == tmp_path / "feedback-1.json"
