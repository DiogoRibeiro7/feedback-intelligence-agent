"""Shared pytest configuration.

The documentation toolchain lives in ``docs/tools`` rather than inside the
distributed package, because it is a build tool and not part of the public
API. Adding ``docs`` to ``sys.path`` here makes it importable as ``tools`` in
tests, matching how the Makefile invokes it (``PYTHONPATH=docs python -m
tools.generate_docs``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

if str(DOCS_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCS_ROOT))


@pytest.fixture
def symlink_supported(tmp_path: Path) -> None:
    """Skip symlink tests only when Windows denies symlink creation privileges."""
    target = tmp_path / "symlink-probe-target"
    target.write_text("probe", encoding="utf-8")
    link = tmp_path / "symlink-probe-link"
    try:
        link.symlink_to(target)
    except OSError as exc:
        if sys.platform == "win32" and exc.winerror == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or elevation")
        raise
    finally:
        link.unlink(missing_ok=True)
        target.unlink()
