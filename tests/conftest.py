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

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

if str(DOCS_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCS_ROOT))
