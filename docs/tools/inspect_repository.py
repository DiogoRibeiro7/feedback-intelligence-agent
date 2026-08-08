"""Static inspection of the repository into a machine-readable inventory.

The inventory is the engineering half of the documentation model: files,
languages, dependencies, CI workflows, container definitions, quality gates,
tests, datasets, and release metadata. Like the API extractor it performs no
application imports and no network access; the only external process invoked
is ``git`` for revision facts.

Running the module writes ``docs/metadata/repository-inventory.json``::

    python -m tools.inspect_repository --repo-root . \
        --output docs/metadata/repository-inventory.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
import yaml

from tools.digest import normalise_newlines

#: Path prefixes mapped to the role a file plays in the repository.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("src/", "python-source"),
    ("tests/snapshots/", "test-fixture"),
    ("tests/", "python-test"),
    ("scripts/", "python-script"),
    ("docs/tools/", "documentation-tool"),
    ("docs/latex/", "documentation-latex"),
    ("docs/metadata/", "documentation-metadata"),
    ("docs/", "documentation"),
    ("data/", "dataset"),
    ("examples/", "example-data"),
    ("frontend/", "frontend"),
    ("deploy/", "deployment"),
    (".github/workflows/", "ci-workflow"),
    (".github/", "repository-governance"),
)

#: Output of the documentation system itself, excluded from the file census.
#: See :func:`build_file_index` for why.
GENERATED_DOCUMENTATION_PREFIXES: tuple[str, ...] = (
    "docs/metadata/",
    "docs/latex/generated/",
)

_ROOT_CATEGORIES: dict[str, str] = {
    "pyproject.toml": "build-configuration",
    "poetry.lock": "dependency-lock",
    "Makefile": "build-interface",
    "Dockerfile": "container",
    "docker-compose.yml": "container",
    ".env.example": "configuration-template",
    ".editorconfig": "editor-configuration",
    ".gitattributes": "repository-configuration",
    ".gitignore": "repository-configuration",
    "CITATION.cff": "citation-metadata",
    ".zenodo.json": "citation-metadata",
    "LICENSE": "license",
    "README.md": "documentation",
    "CHANGELOG.md": "documentation",
    "ROADMAP.md": "documentation",
    "CONTRIBUTING.md": "documentation",
    "CODE_OF_CONDUCT.md": "documentation",
    "SECURITY.md": "documentation",
    "SUPPORT.md": "documentation",
    "AGENTS.md": "documentation",
}

_EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".css": "CSS",
    ".html": "HTML",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".csv": "CSV",
    ".md": "Markdown",
    ".tex": "LaTeX",
    ".cff": "YAML",
}

#: Directories described in the annotated repository tree.
_DIRECTORY_ROLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "src/feedback_intelligence_agent",
        "Python package: the entire importable API.",
        "Modules, no subpackages.",
        "authored",
    ),
    (
        "tests",
        "Pytest suite mirroring the package module-for-module.",
        "test_*.py plus prompt snapshots.",
        "authored",
    ),
    (
        "scripts",
        "Executable entry points that are not part of the package.",
        "Deterministic demo and benchmark drivers.",
        "authored",
    ),
    (
        "data",
        "Version-controlled sample dataset used by demos, tests, and CI.",
        "sample_feedback.csv.",
        "authored",
    ),
    (
        "examples",
        "Input fixtures for evaluation, streaming, tools, and experiments.",
        "JSONL and YAML.",
        "authored",
    ),
    (
        "docs",
        "Documentation sources: Markdown notes, the LaTeX manual, and its generator.",
        "Markdown, LaTeX, JSON metadata, Python tools.",
        "mixed",
    ),
    (
        "docs/latex/generated",
        "LaTeX fragments produced by the documentation generator.",
        "One fragment per documented object.",
        "generated",
    ),
    (
        "docs/metadata",
        "Normalized documentation model in JSON.",
        "Repository inventory, API index, dependency graph, manifest.",
        "generated",
    ),
    (
        "docs/tools",
        "Extraction and LaTeX rendering layers of the documentation system.",
        "Python modules.",
        "authored",
    ),
    (
        "frontend",
        "Vite/TypeScript demo client for the HTTP API.",
        "Source, build config, Dockerfile.",
        "authored",
    ),
    (
        "deploy",
        "Deployment descriptors for production-like targets.",
        "Compose, ECS task definition, Fly.io config.",
        "authored",
    ),
    (
        ".github",
        "Repository governance and CI/CD definitions.",
        "Workflows, issue templates, Dependabot.",
        "authored",
    ),
    (
        ".artifacts",
        "Local run outputs: index, telemetry, jobs, reports, experiments.",
        "JSON and JSONL.",
        "generated",
    ),
    ("dist", "Built Python distributions.", "sdist and wheel.", "generated"),
)


@dataclass
class FileEntry:
    """One version-controlled file and its classification."""

    path: str
    category: str
    language: str | None
    lines: int | None
    bytes: int


def _run_git(repo_root: Path, *arguments: str) -> str | None:
    """Run a read-only git command, returning ``None`` when git is unavailable."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def tracked_files(repo_root: Path) -> list[str]:
    """List the files git would consider part of the project.

    Tracked files plus untracked files that are not ignored, so a freshly
    generated documentation tree is described accurately before it is
    committed. Ignored build output (``.artifacts``, ``dist``, caches) is
    excluded because ``--exclude-standard`` honours ``.gitignore``.
    """
    listing = _run_git(repo_root, "ls-files", "--cached", "--others", "--exclude-standard")
    if listing:
        return sorted(line.strip() for line in listing.splitlines() if line.strip())
    ignored = {".git", ".venv", "node_modules", "__pycache__", ".artifacts", "dist"}
    found: list[str] = []
    for path in repo_root.rglob("*"):
        if path.is_dir() or any(part in ignored for part in path.parts):
            continue
        found.append(path.relative_to(repo_root).as_posix())
    return sorted(found)


def classify(path: str) -> str:
    """Classify a repository-relative path into a content category."""
    for prefix, category in _CATEGORY_RULES:
        if path.startswith(prefix):
            return category
    if path in _ROOT_CATEGORIES:
        return _ROOT_CATEGORIES[path]
    return "other"


def _content_size(path: Path) -> int:
    """Return a file's size with line endings normalised.

    The size on disk depends on the checkout: git rewrites line endings, so
    the same file is larger on a Windows working tree than on a Linux one.
    Recording the normalised size keeps the inventory identical across
    platforms, which the documentation drift check depends on.
    """
    try:
        return len(normalise_newlines(path.read_bytes()))
    except OSError:  # pragma: no cover - unreadable file
        return 0


def _count_lines(path: Path) -> int | None:
    """Count text lines, returning ``None`` for binary or unreadable files."""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (UnicodeDecodeError, OSError):
        return None


def build_file_index(repo_root: Path) -> list[FileEntry]:
    """Build the classified file index of the authored repository.

    The documentation system's own output is excluded. Including it would
    make the census self-referential: the inventory records the size of the
    manifest, the manifest records a digest of the inventory, and the pair has
    no stable fixed point, so two runs over one revision could differ. The
    census therefore measures the authored repository, and the count of
    excluded generated files is reported alongside it rather than hidden.
    """
    entries: list[FileEntry] = []
    for relative in tracked_files(repo_root):
        if relative.startswith(GENERATED_DOCUMENTATION_PREFIXES):
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        entries.append(
            FileEntry(
                path=relative,
                category=classify(relative),
                language=_EXTENSION_LANGUAGES.get(suffix),
                lines=_count_lines(path) if suffix in _EXTENSION_LANGUAGES else None,
                bytes=_content_size(path),
            )
        )
    return entries


def project_metadata(repo_root: Path, revision: str | None = None) -> dict[str, Any]:
    """Read project identity from pyproject, CITATION.cff, and git."""
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject.get("tool", {}).get("poetry", {})

    citation: dict[str, Any] = {}
    citation_path = repo_root / "CITATION.cff"
    if citation_path.exists():
        citation = yaml.safe_load(citation_path.read_text(encoding="utf-8")) or {}

    zenodo: dict[str, Any] = {}
    zenodo_path = repo_root / ".zenodo.json"
    if zenodo_path.exists():
        zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))

    # CITATION.cff is the authoritative contact record: it carries the
    # maintainer's institutional email, ORCID, and affiliation, whereas the
    # pyproject author string carries only a name and a single address.
    maintainers = [
        {
            "name": " ".join(
                part
                for part in (author.get("given-names", ""), author.get("family-names", ""))
                if part
            ).strip(),
            "email": author.get("email", ""),
            "orcid": author.get("orcid", ""),
            "affiliation": author.get("affiliation", ""),
        }
        for author in (citation.get("authors") or [])
    ]

    # A commit cannot contain its own hash, so documentation committed
    # alongside a change necessarily records the revision it was generated
    # from -- the parent. Passing an explicit revision lets a checker
    # regenerate exactly as of the revision the committed documentation
    # claims, so any resulting difference is genuine content drift rather
    # than the revision moving on.
    # The branch is deliberately not recorded: a commit can be reachable from
    # any number of branches, and a CI checkout is detached, so the value
    # would differ between environments describing the same revision. The
    # revision hash is the identifier that matters.
    revision = revision or _run_git(repo_root, "rev-parse", "HEAD")
    revision_date = _run_git(repo_root, "log", "-1", "--format=%cI", revision or "HEAD")
    remote = _run_git(repo_root, "config", "--get", "remote.origin.url")

    return {
        "name": poetry.get("name", ""),
        "version": poetry.get("version", ""),
        "description": poetry.get("description", ""),
        "authors": poetry.get("authors", []),
        "maintainers": maintainers,
        "readme": poetry.get("readme", ""),
        "license": citation.get("license") or zenodo.get("license") or "",
        "doi": citation.get("doi", ""),
        "repository_url": citation.get("repository-code") or (remote or ""),
        "orcid": (citation.get("authors") or [{}])[0].get("orcid", ""),
        "affiliation": (citation.get("authors") or [{}])[0].get("affiliation", ""),
        "keywords": citation.get("keywords", []),
        "date_released": str(citation.get("date-released", "")),
        "python_requires": poetry.get("dependencies", {}).get("python", ""),
        "revision": revision or "",
        "revision_short": (revision or "")[:12],
        "revision_date": revision_date or "",
        "packages": poetry.get("packages", []),
        "console_scripts": poetry.get("scripts", {}),
    }


def dependency_metadata(repo_root: Path) -> dict[str, Any]:
    """Read runtime, optional, and development dependencies from pyproject."""
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject.get("tool", {}).get("poetry", {})
    raw_runtime = poetry.get("dependencies", {})
    extras = poetry.get("extras", {})
    optional_names = {
        name
        for name, spec in raw_runtime.items()
        if isinstance(spec, dict) and spec.get("optional")
    }

    def _describe(name: str, spec: object) -> dict[str, Any]:
        if isinstance(spec, dict):
            return {
                "name": name,
                "constraint": str(spec.get("version", "")),
                "optional": bool(spec.get("optional", False)),
                "extras": list(spec.get("extras", [])),
            }
        return {"name": name, "constraint": str(spec), "optional": False, "extras": []}

    runtime = [
        _describe(name, spec)
        for name, spec in sorted(raw_runtime.items())
        if name != "python" and name not in optional_names
    ]
    optional = [
        _describe(name, spec)
        for name, spec in sorted(raw_runtime.items())
        if name in optional_names
    ]
    development = [
        _describe(name, spec)
        for name, spec in sorted(
            poetry.get("group", {}).get("dev", {}).get("dependencies", {}).items()
        )
    ]

    lock_path = repo_root / "poetry.lock"
    lock_packages = 0
    if lock_path.exists():
        lock_packages = len(
            re.findall(
                r"^\[\[package\]\]$", lock_path.read_text(encoding="utf-8"), flags=re.MULTILINE
            )
        )

    frontend: dict[str, Any] = {}
    package_json = repo_root / "frontend" / "package.json"
    if package_json.exists():
        payload = json.loads(package_json.read_text(encoding="utf-8"))
        frontend = {
            "name": payload.get("name", ""),
            "scripts": payload.get("scripts", {}),
            "dependencies": payload.get("dependencies", {}),
            "dev_dependencies": payload.get("devDependencies", {}),
        }

    return {
        "python_constraint": str(raw_runtime.get("python", "")),
        "runtime": runtime,
        "optional": optional,
        "development": development,
        "extras": {name: sorted(values) for name, values in sorted(extras.items())},
        "lock_file": {
            "path": "poetry.lock",
            "present": lock_path.exists(),
            "locked_packages": lock_packages,
        },
        "frontend": frontend,
    }


def tooling_configuration(repo_root: Path) -> dict[str, Any]:
    """Read linter, type-checker, and test-runner configuration."""
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    tool = pyproject.get("tool", {})
    ruff = tool.get("ruff", {})
    mypy = tool.get("mypy", {})
    pytest_options = tool.get("pytest", {}).get("ini_options", {})
    return {
        "ruff": {
            "line_length": ruff.get("line-length"),
            "target_version": ruff.get("target-version"),
            "src": ruff.get("src", []),
            "select": ruff.get("lint", {}).get("select", []),
            "ignore": ruff.get("lint", {}).get("ignore", []),
        },
        "mypy": {
            "python_version": mypy.get("python_version"),
            "strict": mypy.get("strict", False),
            "plugins": mypy.get("plugins", []),
            "overrides": [
                {
                    "module": entry.get("module", []),
                    "ignore_missing_imports": entry.get("ignore_missing_imports", False),
                }
                for entry in tool.get("mypy", {}).get("overrides", [])
            ],
        },
        "pytest": {
            "testpaths": pytest_options.get("testpaths", []),
            "addopts": pytest_options.get("addopts", ""),
        },
        "build_system": pyproject.get("build-system", {}),
    }


def makefile_targets(repo_root: Path) -> list[dict[str, Any]]:
    """Parse Makefile targets and their recipe lines."""
    path = repo_root / "Makefile"
    if not path.exists():
        return []
    targets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("\t"):
            if current is not None:
                current["recipe"].append(line.strip())
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", line)
        if match is None:
            continue
        name = match.group(1)
        if name == ".PHONY":
            continue
        current = {
            "name": name,
            "prerequisites": match.group(2).split(),
            "recipe": [],
        }
        targets.append(current)
    return targets


def ci_workflows(repo_root: Path) -> list[dict[str, Any]]:
    """Parse GitHub Actions workflow definitions."""
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.exists():
        return []
    workflows: list[dict[str, Any]] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # PyYAML parses the unquoted ``on:`` key as the boolean True.
        triggers = payload.get("on", payload.get(True, {}))
        jobs = []
        for job_id, job in (payload.get("jobs") or {}).items():
            steps = [
                {
                    "name": step.get("name", ""),
                    "uses": step.get("uses", ""),
                    "run": " ".join(str(step.get("run", "")).split()),
                    "working_directory": step.get("working-directory", ""),
                }
                for step in job.get("steps", [])
            ]
            matrix = job.get("strategy", {}).get("matrix", {})
            jobs.append(
                {
                    "id": job_id,
                    "name": job.get("name", job_id),
                    "runs_on": job.get("runs-on", ""),
                    "needs": job.get("needs", [])
                    if isinstance(job.get("needs"), list)
                    else ([job["needs"]] if job.get("needs") else []),
                    "matrix": dict(matrix),
                    "steps": steps,
                }
            )
        workflows.append(
            {
                "file": path.relative_to(repo_root).as_posix(),
                "name": payload.get("name", path.stem),
                "triggers": _normalise_triggers(triggers),
                "permissions": payload.get("permissions", {}),
                "concurrency": payload.get("concurrency", {}),
                "jobs": jobs,
            }
        )
    return workflows


def _normalise_triggers(triggers: object) -> dict[str, Any]:
    """Normalise a workflow trigger block into a plain mapping."""
    if isinstance(triggers, dict):
        return {str(key): value for key, value in triggers.items()}
    if isinstance(triggers, list):
        return {str(item): {} for item in triggers}
    if triggers:
        return {str(triggers): {}}
    return {}


def container_metadata(repo_root: Path) -> dict[str, Any]:
    """Read Dockerfile stages and Compose service definitions."""
    result: dict[str, Any] = {"dockerfiles": [], "compose": []}
    for relative in ("Dockerfile", "frontend/Dockerfile"):
        path = repo_root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        result["dockerfiles"].append(
            {
                "path": relative,
                "base_images": re.findall(r"^FROM\s+(\S+)", text, flags=re.MULTILINE),
                "exposed_ports": re.findall(r"^EXPOSE\s+(\d+)", text, flags=re.MULTILINE),
                "entrypoint": next(
                    iter(re.findall(r"^(?:CMD|ENTRYPOINT)\s+(.+)$", text, flags=re.MULTILINE)), ""
                ),
                "user": next(iter(re.findall(r"^USER\s+(\S+)", text, flags=re.MULTILINE)), ""),
            }
        )
    for relative in ("docker-compose.yml", "deploy/docker-compose.prod.yml"):
        path = repo_root / relative
        if not path.exists():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        result["compose"].append(
            {
                "path": relative,
                "services": [
                    {
                        "name": name,
                        "image": service.get("image", ""),
                        "build": bool(service.get("build")),
                        "ports": service.get("ports", []),
                        "command": service.get("command", ""),
                        "healthcheck": bool(service.get("healthcheck")),
                    }
                    for name, service in (payload.get("services") or {}).items()
                ],
            }
        )
    return result


def deployment_targets(repo_root: Path) -> list[dict[str, str]]:
    """List deployment descriptors present in the repository."""
    descriptors = {
        "deploy/docker-compose.prod.yml": "Production-like Docker Compose stack.",
        "deploy/ecs-fargate-task-definition.json": "AWS ECS Fargate task definition.",
        "deploy/fly.toml": "Fly.io application configuration.",
    }
    return [
        {"path": path, "purpose": purpose}
        for path, purpose in sorted(descriptors.items())
        if (repo_root / path).exists()
    ]


_TEST_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("test_prompt_snapshot", "snapshot regression"),
    ("test_retrieval_regressions", "retrieval regression"),
    ("test_metadata_filters", "behavioural regression"),
    ("test_api", "integration (HTTP)"),
    ("test_cli", "integration (CLI)"),
    ("test_data_contracts", "data validation"),
    ("test_evaluation", "numerical / metric"),
    ("test_benchmarking", "performance measurement"),
    ("test_lexical_and_hybrid", "numerical / retrieval"),
    ("test_embeddings_and_store", "numerical / retrieval"),
)


def test_inventory(repo_root: Path) -> list[dict[str, Any]]:
    """Inventory the pytest suite: file, category, and test counts."""
    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        return []
    inventory: list[dict[str, Any]] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
        ]
        parametrised = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "parametrize"
        )
        category = "unit"
        for marker, name in _TEST_CATEGORIES:
            if path.stem.startswith(marker):
                category = name
                break
        inventory.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "module_under_test": _module_under_test(path.stem),
                "category": category,
                "test_functions": len(functions),
                "parametrized_cases": parametrised,
            }
        )
    return inventory


def _module_under_test(stem: str) -> str:
    """Map a test module name to the package module it primarily covers."""
    candidate = stem.removeprefix("test_")
    special = {
        "embeddings_and_store": "embeddings, vector_store",
        "lexical_and_hybrid": "lexical_search, retrieval",
        "llm_providers": "llm",
        "metadata_filters": "agent, schemas",
        "prompt_snapshot": "prompts, prompt_registry",
        "retrieval_regressions": "retrieval, reranking",
        "qdrant_store": "qdrant_store",
    }
    return special.get(candidate, candidate)


def dataset_inventory(repo_root: Path) -> list[dict[str, Any]]:
    """Inventory version-controlled datasets and example inputs."""
    datasets: list[dict[str, Any]] = []
    for relative in sorted(
        path.relative_to(repo_root).as_posix()
        for directory in ("data", "examples")
        for path in (repo_root / directory).glob("*")
        if (repo_root / directory).exists() and path.is_file()
    ):
        path = repo_root / relative
        entry: dict[str, Any] = {
            "path": relative,
            "format": path.suffix.lstrip("."),
            "bytes": path.stat().st_size,
            "records": None,
            "columns": [],
        }
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                entry["columns"] = header
                entry["records"] = sum(1 for _ in reader)
        elif path.suffix == ".jsonl":
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            entry["records"] = len(lines)
            if lines:
                first = json.loads(lines[0])
                entry["columns"] = sorted(first) if isinstance(first, dict) else []
        datasets.append(entry)
    return datasets


def env_template(repo_root: Path) -> list[dict[str, str]]:
    """Read the documented environment variables from ``.env.example``."""
    path = repo_root / ".env.example"
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    comment: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment.append(stripped.lstrip("# ").strip())
            continue
        if not stripped or "=" not in stripped:
            comment = []
            continue
        name, _, value = stripped.partition("=")
        entries.append(
            {"name": name.strip(), "example": value.strip(), "comment": " ".join(comment)}
        )
        comment = []
    return entries


def quality_gates(repo_root: Path) -> list[dict[str, Any]]:
    """Derive the implemented quality gates from the Makefile and CI workflows."""
    targets = {target["name"]: target for target in makefile_targets(repo_root)}
    workflows = ci_workflows(repo_root)
    ci_steps = {
        step["name"]: workflow["name"]
        for workflow in workflows
        for job in workflow["jobs"]
        for step in job["steps"]
    }

    definitions: tuple[tuple[str, str, str, str], ...] = (
        ("Formatting", "ruff format", "format-check", "Format check"),
        ("Linting", "ruff check", "lint", "Lint"),
        ("Type checking", "mypy (strict)", "typecheck", "Type check"),
        ("Tests and coverage", "pytest, pytest-cov", "coverage", "Test with coverage"),
        ("Deterministic demo", "scripts/run_demo.py", "demo", "Run deterministic demo"),
        ("Package build", "poetry build", "build", "Build package"),
    )
    gates: list[dict[str, Any]] = []
    for gate, tool, target, ci_step in definitions:
        entry = targets.get(target)
        gates.append(
            {
                "gate": gate,
                "tool": tool,
                "make_target": target if entry else "",
                "local_command": entry["recipe"][0] if entry and entry["recipe"] else "",
                "ci_step": ci_step if ci_step in ci_steps else "",
                "ci_workflow": ci_steps.get(ci_step, ""),
                "blocking": ci_step in ci_steps,
            }
        )
    static_analysis = any(
        step["uses"].startswith("github/codeql-action")
        for workflow in workflows
        for job in workflow["jobs"]
        for step in job["steps"]
    )
    if static_analysis:
        gates.append(
            {
                "gate": "Security static analysis",
                "tool": "CodeQL",
                "make_target": "",
                "local_command": "",
                "ci_step": "Perform CodeQL analysis",
                "ci_workflow": "Security",
                "blocking": True,
            }
        )
    dependency_review = any(
        step["uses"].startswith("actions/dependency-review-action")
        for workflow in workflows
        for job in workflow["jobs"]
        for step in job["steps"]
    )
    if dependency_review:
        gates.append(
            {
                "gate": "Dependency review",
                "tool": "actions/dependency-review-action",
                "make_target": "",
                "local_command": "",
                "ci_step": "Review dependency changes",
                "ci_workflow": "Dependency Review",
                "blocking": True,
            }
        )
    return gates


def directory_roles(repo_root: Path) -> list[dict[str, str]]:
    """Return the annotated responsibilities of significant directories."""
    return [
        {
            "path": path,
            "responsibility": responsibility,
            "contents": contents,
            "provenance": provenance,
            "version_controlled": "yes"
            if (repo_root / path).exists() and not path.startswith((".artifacts", "dist"))
            else "no",
        }
        for path, responsibility, contents, provenance in _DIRECTORY_ROLES
    ]


def build_inventory(repo_root: Path, revision: str | None = None) -> dict[str, Any]:
    """Build the complete repository inventory."""
    files = build_file_index(repo_root)
    languages: dict[str, dict[str, int]] = {}
    categories: dict[str, int] = {}
    for entry in files:
        categories[entry.category] = categories.get(entry.category, 0) + 1
        if entry.language is None:
            continue
        bucket = languages.setdefault(entry.language, {"files": 0, "lines": 0})
        bucket["files"] += 1
        bucket["lines"] += entry.lines or 0

    return {
        "project": project_metadata(repo_root, revision),
        "files": {
            "total": len(files),
            "excluded_generated_documentation": len(
                [
                    name
                    for name in tracked_files(repo_root)
                    if name.startswith(GENERATED_DOCUMENTATION_PREFIXES)
                ]
            ),
            "by_category": dict(sorted(categories.items())),
            "by_language": dict(sorted(languages.items())),
            "entries": [entry.__dict__ for entry in files],
        },
        "directories": directory_roles(repo_root),
        "dependencies": dependency_metadata(repo_root),
        "tooling": tooling_configuration(repo_root),
        "make_targets": makefile_targets(repo_root),
        "ci_workflows": ci_workflows(repo_root),
        "containers": container_metadata(repo_root),
        "deployments": deployment_targets(repo_root),
        "tests": test_inventory(repo_root),
        "datasets": dataset_inventory(repo_root),
        "environment_template": env_template(repo_root),
        "quality_gates": quality_gates(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the repository inventory JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("docs/metadata/repository-inventory.json")
    )
    arguments = parser.parse_args(argv)

    inventory = build_inventory(arguments.repo_root)
    output = arguments.repo_root / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"repository-inventory: {inventory['files']['total']} files, "
        f"{len(inventory['ci_workflows'])} workflows, {len(inventory['tests'])} test modules "
        f"-> {arguments.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
