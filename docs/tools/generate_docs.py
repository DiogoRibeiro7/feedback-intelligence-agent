"""Orchestrate the full documentation pipeline: inspect, extract, render, manifest.

The pipeline runs in a fixed order so the model is always complete before any
LaTeX is rendered::

    repository source
        -> tools.inspect_repository   (docs/metadata/repository-inventory.json)
        -> tools.extract_api          (docs/metadata/api-index.json)
        -> tools.extract_dependencies (docs/metadata/module-dependencies.json)
        -> tools.render_reference     (docs/latex/generated/**)
        -> tools.render_architecture
        -> tools.render_engineering
        -> docs/metadata/documentation-manifest.json

The manifest records a SHA-256 digest of every generated file together with the
repository revision the run described, which makes drift between the committed
documentation and the source detectable.

Documentation timestamps are taken from the HEAD commit date rather than the
wall clock, so two runs over the same revision produce byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools import __version__ as generator_version
from tools import (
    extract_api,
    extract_dependencies,
    inspect_repository,
    render_architecture,
    render_engineering,
    render_reference,
)
from tools.latex_utils import escape_latex, file_header

METADATA_ROOT = Path("docs/metadata")
LATEX_ROOT = Path("docs/latex")


def render_document_metadata(inventory: dict[str, Any]) -> str:
    """Render ``docs/latex/metadata.tex``: title-page and header macros."""
    project = inventory["project"]
    title = " ".join(part.capitalize() for part in project["name"].split("-"))
    # CITATION.cff is the authoritative contact record; the pyproject author
    # string is the fallback for a repository that has no citation metadata.
    maintainers = project.get("maintainers") or []
    authors = "; ".join(
        f"{person['name']} <{person['email']}>" if person["email"] else person["name"]
        for person in maintainers
    ) or (", ".join(project["authors"]) or "Unknown")
    revision_date = project["revision_date"] or "unknown"
    macros = [
        ("projectname", title),
        ("projectversion", project["version"]),
        ("projectrevision", project["revision"]),
        ("projectrevisionshort", project["revision_short"]),
        ("projectrevisiondate", revision_date),
        ("projectlicense", project["license"]),
        ("projectdoi", project["doi"]),
        ("projectruntime", f"Python {project['python_requires']}"),
        ("projectrepository", project["repository_url"]),
        ("generatorversion", generator_version),
    ]
    lines = [
        r"% Title-page and running-header macros for the technical manual.",
        "",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{escape_latex(value)}}}" for name, value in macros)
    lines.extend(
        [
            "",
            r"\title{%",
            r"  \Huge \projectname \\[6pt]",
            r"  \Large Technical Reference Manual \\[10pt]",
            r"  \normalsize Reference Manual \textbullet{} Software Architecture "
            r"\textbullet{} Engineering",
            r"}",
            f"\\author{{{escape_latex(authors)}}}",
            r"\date{%",
            r"  Version \projectversion{} \quad Revision \texttt{\projectrevisionshort} \\[4pt]",
            r"  \small Documented revision dated \projectrevisiondate",
            r"}",
        ]
    )
    return "\n".join(lines)


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(repo_root: Path, generated: list[Path], inventory: dict[str, Any]) -> Path:
    """Write the documentation manifest describing every generated artifact."""
    entries = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in sorted(generated)
    ]
    metadata_files = sorted(
        (repo_root / METADATA_ROOT).glob("*.json"),
    )
    manifest = {
        "generator": {
            "name": "docs/tools",
            "version": generator_version,
            "entry_point": "python -m tools.generate_docs",
        },
        "revision": inventory["project"]["revision"],
        "revision_date": inventory["project"]["revision_date"],
        "project_version": inventory["project"]["version"],
        "counts": {
            "generated_latex_files": len(entries),
            "metadata_files": len(
                [path for path in metadata_files if path.name != "documentation-manifest.json"]
            ),
        },
        "generated_files": entries,
        "metadata_files": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
            for path in metadata_files
            if path.name != "documentation-manifest.json"
        ],
    }
    output = repo_root / METADATA_ROOT / "documentation-manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def _clean_generated(repo_root: Path) -> None:
    """Remove previously generated LaTeX so deleted objects do not linger."""
    generated_root = repo_root / LATEX_ROOT / "generated"
    if not generated_root.exists():
        return
    for path in sorted(generated_root.rglob("*.tex")):
        path.unlink()
    for directory in sorted(generated_root.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def generate(repo_root: Path, revision: str | None = None) -> dict[str, Any]:
    """Run the whole pipeline and return a small summary dictionary.

    Args:
        repo_root: Repository to document.
        revision: Revision to record. Defaults to ``HEAD``. A checker that
            wants to verify the committed documentation is current passes the
            revision the documentation itself records, so that the comparison
            isolates content drift from the revision simply having moved on.
    """
    inventory = inspect_repository.build_inventory(repo_root, revision)
    (repo_root / METADATA_ROOT).mkdir(parents=True, exist_ok=True)
    (repo_root / METADATA_ROOT / "repository-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )

    model = extract_api.build_api_index(repo_root)
    (repo_root / METADATA_ROOT / "api-index.json").write_text(
        json.dumps(model, indent=2) + "\n", encoding="utf-8"
    )

    graph = extract_dependencies.build_dependency_graph(repo_root)
    (repo_root / METADATA_ROOT / "module-dependencies.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8"
    )

    _clean_generated(repo_root)
    generated: list[Path] = []
    generated.extend(render_reference.render_all(repo_root, model, inventory))
    generated.extend(render_architecture.render_all(repo_root, graph, model))
    generated.extend(render_engineering.render_all(repo_root, inventory, model, graph))

    metadata_tex = repo_root / LATEX_ROOT / "metadata.tex"
    metadata_tex.parent.mkdir(parents=True, exist_ok=True)
    metadata_tex.write_text(
        f"{file_header('generate_docs.py')}\n{render_document_metadata(inventory)}\n",
        encoding="utf-8",
    )
    generated.append(metadata_tex)

    manifest = write_manifest(repo_root, generated, inventory)
    return {
        "inventory": inventory,
        "model": model,
        "graph": graph,
        "generated": generated,
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    """Regenerate the complete LaTeX documentation source tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the validation pass that normally runs after generation.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Revision to record instead of HEAD. Used to verify that committed "
            "documentation is current: regenerating as of the revision the "
            "documentation claims leaves only genuine content differences."
        ),
    )
    arguments = parser.parse_args(argv)
    repo_root = arguments.repo_root.resolve()

    summary = generate(repo_root, arguments.revision)
    inventory = summary["inventory"]
    model = summary["model"]
    graph = summary["graph"]

    print(
        "documentation model: "
        f"{inventory['files']['total']} files, "
        f"{len(model['modules'])} modules, "
        f"{len(model['classes'])} public classes, "
        f"{len(model['functions'])} public functions, "
        f"{len(model['commands'])} CLI commands, "
        f"{len(model['endpoints'])} HTTP endpoints, "
        f"{graph['edge_count']} internal import edges"
    )
    print(
        f"generated {len(summary['generated'])} LaTeX files; manifest written to "
        f"{summary['manifest'].relative_to(repo_root).as_posix()}"
    )
    print(f"documented revision: {inventory['project']['revision'] or 'unknown'}")

    if arguments.no_validate:
        return 0

    from tools import validate_docs

    return validate_docs.main(["--repo-root", str(repo_root)])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
