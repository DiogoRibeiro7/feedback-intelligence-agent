from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_release.py"


@pytest.fixture()
def release_repo(tmp_path: Path) -> Path:
    files = {
        "pyproject.toml": '[tool.poetry]\nversion = "0.6.0"\n',
        "src/feedback_intelligence_agent/__init__.py": (
            'raise RuntimeError("application must not be imported")\n__version__ = "0.6.0"\n'
        ),
        "CITATION.cff": 'version: 0.6.0\ndate-released: "2026-10-29"\n',
        ".zenodo.json": json.dumps({"version": "0.6.0", "publication_date": "2026-10-29"}),
        "frontend/package.json": json.dumps({"version": "0.6.0"}),
        "frontend/package-lock.json": json.dumps(
            {"version": "0.6.0", "packages": {"": {"version": "0.6.0"}}}
        ),
        "CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n## 0.6.0 - 2026-10-29\n",
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return tmp_path


def run_check(root: Path, tag: str = "v0.6.0") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), "--tag", tag],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_release_gate_accepts_consistent_metadata_without_importing_application(
    release_repo: Path,
) -> None:
    result = run_check(release_repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "src/feedback_intelligence_agent/__init__.py",
        "CITATION.cff",
        ".zenodo.json",
        "frontend/package.json",
        "frontend/package-lock.json",
    ],
)
def test_release_gate_rejects_version_drift(release_repo: Path, name: str) -> None:
    path = release_repo / name
    path.write_text(path.read_text(encoding="utf-8").replace("0.6.0", "0.5.0"), encoding="utf-8")
    result = run_check(release_repo)
    assert result.returncode == 1
    assert "does not match" in result.stdout


def test_release_gate_checks_nested_frontend_lock_version(release_repo: Path) -> None:
    path = release_repo / "frontend/package-lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    lock["packages"][""]["version"] = "0.3.0"
    path.write_text(json.dumps(lock), encoding="utf-8")
    result = run_check(release_repo)
    assert result.returncode == 1
    assert "packages['']" in result.stdout


def test_release_gate_rejects_wrong_tag(release_repo: Path) -> None:
    result = run_check(release_repo, "v0.5.0")
    assert result.returncode == 1
    assert "tag 'v0.5.0'" in result.stdout


@pytest.mark.parametrize("name", ["CITATION.cff", ".zenodo.json", "CHANGELOG.md"])
def test_release_gate_rejects_date_drift(release_repo: Path, name: str) -> None:
    path = release_repo / name
    path.write_text(
        path.read_text(encoding="utf-8").replace("2026-10-29", "2026-10-28"), encoding="utf-8"
    )
    assert run_check(release_repo).returncode == 1


@pytest.mark.parametrize("contents", ["{broken json", "{}"])
def test_release_gate_reports_invalid_or_missing_metadata(
    release_repo: Path, contents: str
) -> None:
    (release_repo / ".zenodo.json").write_text(contents, encoding="utf-8")
    result = run_check(release_repo)
    assert result.returncode == 1
    assert "could not be validated" in result.stdout
    assert "Traceback" not in result.stderr
