"""Validate release versions, dates, changelog entry, and an optional Git tag."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import date
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: installed with the DataExcept dependency.
    import tomli as tomllib


def _package_version(path: Path) -> str:
    """Read the literal version without importing application code."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"{path}: missing literal __version__")


def validate_release(root: Path, *, tag: str | None = None) -> list[str]:
    """Return inconsistencies between the release's published metadata sources."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["tool"]["poetry"]["version"]
    if not isinstance(expected, str) or not expected:
        raise ValueError("pyproject.toml: version must be a nonempty string")
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    frontend = json.loads((root / "frontend/package.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "frontend/package-lock.json").read_text(encoding="utf-8"))
    versions = {
        "src/feedback_intelligence_agent/__init__.py": _package_version(
            root / "src/feedback_intelligence_agent/__init__.py"
        ),
        "CITATION.cff": citation["version"],
        ".zenodo.json": zenodo["version"],
        "frontend/package.json": frontend["version"],
        "frontend/package-lock.json": lock["version"],
        "frontend/package-lock.json packages['']": lock["packages"][""]["version"],
    }
    errors = [
        f"{name}: version {actual!r} does not match {expected!r}"
        for name, actual in versions.items()
        if actual != expected
    ]
    if tag and tag != f"v{expected}":
        errors.append(f"tag {tag!r} does not match 'v{expected}'")

    release_date = str(citation["date-released"])
    if date.fromisoformat(release_date).isoformat() != release_date:
        errors.append("CITATION.cff: date-released must use YYYY-MM-DD")
    if zenodo["publication_date"] != release_date:
        errors.append(".zenodo.json: publication_date does not match CITATION.cff date-released")
    heading = f"## {expected} - {release_date}"
    if heading not in (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        errors.append(f"CHANGELOG.md: missing release heading {heading!r}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Fail a release gate with actionable metadata errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", default=None, help="Expected release tag, for example v0.6.0.")
    args = parser.parse_args(argv)
    try:
        errors = validate_release(args.repo_root, tag=args.tag)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, yaml.YAMLError) as exc:
        print(f"Release metadata could not be validated: {exc}")
        return 1
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Release metadata versions, dates, changelog, and optional tag agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
