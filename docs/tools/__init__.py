"""Documentation generation toolchain for the feedback intelligence agent.

The toolchain is split into two layers that never mix:

Extraction
    :mod:`tools.inspect_repository`, :mod:`tools.extract_api`, and
    :mod:`tools.extract_dependencies` read the repository with static analysis
    only and emit a normalized documentation model as JSON under
    ``docs/metadata/``.

Rendering
    :mod:`tools.render_reference`, :mod:`tools.render_architecture`, and
    :mod:`tools.render_engineering` turn that JSON model into LaTeX fragments
    under ``docs/latex/generated/``. Rendering never re-reads application
    source code.

:mod:`tools.generate_docs` orchestrates both layers and
:mod:`tools.validate_docs` checks the result.
"""

from __future__ import annotations

import sys

__all__ = ["MINIMUM_PYTHON", "__version__"]

__version__ = "1.0.0"

#: The toolchain reads ``pyproject.toml`` with :mod:`tomllib`, which entered
#: the standard library in Python 3.11. The documented package itself supports
#: 3.10, so this requirement applies to documentation generation only and is
#: stated up front rather than surfacing as a bare ``ModuleNotFoundError``.
MINIMUM_PYTHON = (3, 11)

if sys.version_info < MINIMUM_PYTHON:  # pragma: no cover - environment guard
    raise RuntimeError(
        "The documentation toolchain requires Python "
        f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer because it reads "
        "pyproject.toml with the standard-library tomllib module; this "
        f"interpreter is {sys.version_info.major}.{sys.version_info.minor}. "
        "The feedback_intelligence_agent package itself still supports 3.10 - "
        "only 'make docs' needs the newer interpreter."
    )
