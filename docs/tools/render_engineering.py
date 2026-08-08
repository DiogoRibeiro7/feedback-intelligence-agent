"""Render engineering LaTeX fragments from the repository inventory.

These fragments answer how the software is developed, verified, built,
reproduced, packaged, and released: repository structure, dependency
engineering, the test inventory, quality gates, CI/CD workflows, container and
deployment descriptors, data provenance, and the traceability tables that link
documentation to source and source to tests.

Only controls that exist in the repository are rendered. Nothing here proposes
a control as if it were implemented; recommendations live in an authored
section of the engineering part instead.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.latex_utils import (
    escape_inline_code,
    escape_latex,
    file_header,
    itemize,
    join_blocks,
    label,
    longtable,
    render_prose,
    verbatim,
)
from tools.render_reference import endpoint_key

GENERATED_ROOT = Path("docs/latex/generated/engineering")


def _write(path: Path, body: str, *, written: list[Path]) -> None:
    """Write a generated fragment and record it in the manifest list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{file_header('render_engineering.py')}\n{body.rstrip()}\n", encoding="utf-8")
    written.append(path)


def build_tree(paths: list[str], *, max_depth: int = 2) -> str:
    """Render an ASCII directory tree of the repository, collapsed at a depth."""
    children: dict[str, set[str]] = defaultdict(set)
    files: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for path in paths:
        parts = path.split("/")
        for depth in range(len(parts) - 1):
            parent = "/".join(parts[:depth]) if depth else ""
            children[parent].add("/".join(parts[: depth + 1]))
            counts["/".join(parts[: depth + 1])] += 1
        parent = "/".join(parts[:-1])
        files[parent].append(parts[-1])

    lines = ["."]

    def _walk(directory: str, prefix: str, depth: int) -> None:
        entries: list[tuple[str, bool]] = [
            (child, True) for child in sorted(children.get(directory, set()))
        ]
        if depth < max_depth:
            entries.extend((name, False) for name in sorted(files.get(directory, [])))
        else:
            file_count = len(files.get(directory, []))
            if file_count:
                entries.append((f"({file_count} files)", False))
        for index, (entry, is_directory) in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            name = entry.rsplit("/", 1)[-1] if is_directory else entry
            suffix = "/" if is_directory else ""
            if is_directory and depth + 1 > max_depth:
                lines.append(f"{prefix}{connector}{name}{suffix}  ({counts[entry]} files)")
                continue
            lines.append(f"{prefix}{connector}{name}{suffix}")
            if is_directory:
                extension = "    " if index == len(entries) - 1 else "|   "
                _walk(entry, prefix + extension, depth + 1)

    _walk("", "", 0)
    return "\n".join(lines)


def render_repository_structure(inventory: dict[str, Any]) -> str:
    """Render the annotated repository tree and the directory responsibility table."""
    paths = [entry["path"] for entry in inventory["files"]["entries"]]
    tree = build_tree(paths, max_depth=2)
    directory_rows = [
        [
            escape_inline_code(entry["path"]),
            escape_latex(entry["responsibility"]),
            escape_latex(entry["contents"]),
            escape_latex(entry["provenance"]),
            escape_latex(entry["version_controlled"]),
        ]
        for entry in inventory["directories"]
    ]
    category_rows = [
        [escape_latex(category), str(count)]
        for category, count in inventory["files"]["by_category"].items()
    ]
    language_rows = [
        [escape_latex(language), str(stats["files"]), str(stats["lines"])]
        for language, stats in inventory["files"]["by_language"].items()
    ]
    return join_blocks(
        [
            "\\section{Repository tree}",
            "The tree below is generated from the version-controlled file list at the "
            "documented revision and is collapsed below the second level.",
            verbatim(tree),
            "\\section{Directory responsibilities}",
            longtable(
                column_spec=(
                    r"p{0.20\textwidth} p{0.28\textwidth} p{0.22\textwidth} "
                    r"p{0.11\textwidth} p{0.11\textwidth}"
                ),
                header=[
                    "Directory",
                    "Responsibility",
                    "Expected contents",
                    "Provenance",
                    "In VCS",
                ],
                rows=directory_rows,
                caption="Responsibility and provenance of each significant directory.",
                table_label="tab:directories",
                escape_cells=False,
            ),
            "\\section{File census}",
            "The census covers the authored repository. The "
            f"{inventory['files']['excluded_generated_documentation']} files this "
            "documentation system generates under \\texttt{docs/metadata/} and "
            "\\texttt{docs/latex/generated/} are excluded: a census that measured "
            "its own output could not be reproducible, because the inventory "
            "records the manifest's size while the manifest records the "
            "inventory's digest.",
            longtable(
                column_spec=r"p{0.45\textwidth} r",
                header=["Category", "Files"],
                rows=category_rows,
                caption="Version-controlled files by role.",
                table_label="tab:file-categories",
            ),
            longtable(
                column_spec=r"p{0.35\textwidth} r r",
                header=["Language", "Files", "Lines"],
                rows=language_rows,
                caption="Version-controlled files by language.",
                table_label="tab:file-languages",
            ),
        ]
    )


def render_dependencies(inventory: dict[str, Any]) -> str:
    """Render runtime, optional, and development dependency tables."""
    dependencies = inventory["dependencies"]

    def _rows(items: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                escape_inline_code(item["name"]),
                escape_inline_code(item["constraint"] or "*"),
                ", ".join(escape_inline_code(extra) for extra in item["extras"]) or "--",
            ]
            for item in items
        ]

    extras_rows = [
        [escape_inline_code(name), ", ".join(escape_inline_code(item) for item in packages)]
        for name, packages in dependencies["extras"].items()
    ]
    frontend = dependencies["frontend"]
    frontend_rows = [
        [escape_inline_code(name), escape_inline_code(constraint), "runtime"]
        for name, constraint in sorted(frontend.get("dependencies", {}).items())
    ] + [
        [escape_inline_code(name), escape_inline_code(constraint), "development"]
        for name, constraint in sorted(frontend.get("dev_dependencies", {}).items())
    ]

    return join_blocks(
        [
            "\\section{Python dependencies}",
            f"The package targets Python {escape_inline_code(dependencies['python_constraint'])}.",
            longtable(
                column_spec=r"p{0.30\textwidth} p{0.30\textwidth} p{0.32\textwidth}",
                header=["Distribution", "Version constraint", "Extras requested"],
                rows=_rows(dependencies["runtime"]),
                caption="Required runtime dependencies.",
                table_label="tab:deps-runtime",
                escape_cells=False,
            ),
            longtable(
                column_spec=r"p{0.30\textwidth} p{0.30\textwidth} p{0.32\textwidth}",
                header=["Distribution", "Version constraint", "Extras requested"],
                rows=_rows(dependencies["optional"]),
                caption="Optional runtime dependencies, installed through extras.",
                table_label="tab:deps-optional",
                escape_cells=False,
            ),
            longtable(
                column_spec=r"p{0.35\textwidth} p{0.55\textwidth}",
                header=["Extra", "Distributions"],
                rows=extras_rows,
                caption="Poetry extras and the distributions they install.",
                table_label="tab:deps-extras",
                escape_cells=False,
            ),
            longtable(
                column_spec=r"p{0.30\textwidth} p{0.30\textwidth} p{0.32\textwidth}",
                header=["Distribution", "Version constraint", "Extras requested"],
                rows=_rows(dependencies["development"]),
                caption="Development dependencies (Poetry dev group).",
                table_label="tab:deps-dev",
                escape_cells=False,
            ),
            "\\section{Dependency locking}",
            "Exact resolved versions are pinned in "
            f"{escape_inline_code(dependencies['lock_file']['path'])}, which locks "
            f"{dependencies['lock_file']['locked_packages']} packages. The lock file is "
            "version controlled, so a checkout of a given revision resolves to the same "
            "dependency set.",
            "\\section{Frontend dependencies}",
            longtable(
                column_spec=r"p{0.35\textwidth} p{0.30\textwidth} p{0.27\textwidth}",
                header=["Package", "Version constraint", "Group"],
                rows=frontend_rows,
                caption="Frontend npm dependencies.",
                table_label="tab:deps-frontend",
                escape_cells=False,
            ),
        ]
    )


def render_testing(inventory: dict[str, Any]) -> str:
    """Render the test inventory grouped by category."""
    tests = inventory["tests"]
    total_functions = sum(entry["test_functions"] for entry in tests)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in tests:
        by_category[entry["category"]].append(entry)

    category_rows = [
        [
            escape_latex(category),
            str(len(entries)),
            str(sum(entry["test_functions"] for entry in entries)),
        ]
        for category, entries in sorted(by_category.items())
    ]
    inventory_rows = [
        [
            escape_inline_code(entry["path"]),
            escape_inline_code(entry["module_under_test"]),
            escape_latex(entry["category"]),
            str(entry["test_functions"]),
            str(entry["parametrized_cases"]),
        ]
        for entry in tests
    ]
    coverage: dict[str, Any] = next(
        (gate for gate in inventory["quality_gates"] if gate["gate"] == "Tests and coverage"),
        {},
    )
    return join_blocks(
        [
            "\\section{Test suite at a glance}",
            f"The suite contains {len(tests)} test modules and {total_functions} test "
            "functions, all executed by a single pytest invocation.",
            longtable(
                column_spec=r"p{0.45\textwidth} r r",
                header=["Category", "Modules", "Test functions"],
                rows=category_rows,
                caption="Test modules and functions by category.",
                table_label="tab:test-categories",
            ),
            "\\section{Execution}",
            "Every category runs from the same command; the suite is not split by marker "
            "or path in either the Makefile or CI:",
            escape_inline_code(coverage.get("local_command", "poetry run pytest")) + ".",
            "The suite requires no network access, no database, and no API key: the "
            "deterministic local LLM provider and the JSON vector store are the defaults, "
            "and optional integrations are exercised through injected fakes.",
            "\\section{Test module inventory}",
            longtable(
                column_spec=(
                    r"p{0.28\textwidth} p{0.24\textwidth} p{0.18\textwidth} "
                    r"p{0.10\textwidth} p{0.12\textwidth}"
                ),
                header=[
                    "Test module",
                    "Primary subject",
                    "Category",
                    "Tests",
                    "Parametrised",
                ],
                rows=inventory_rows,
                caption="Every test module, its subject, and its size.",
                table_label="tab:test-inventory",
                escape_cells=False,
            ),
        ]
    )


def render_quality_gates(inventory: dict[str, Any]) -> str:
    """Render the implemented quality gates table."""
    rows = [
        [
            escape_latex(gate["gate"]),
            escape_inline_code(gate["tool"]),
            escape_inline_code(gate["local_command"]) if gate["local_command"] else "--",
            escape_latex(gate["ci_workflow"] or "--"),
            "yes" if gate["blocking"] else "no",
        ]
        for gate in inventory["quality_gates"]
    ]
    tooling = inventory["tooling"]
    return join_blocks(
        [
            longtable(
                column_spec=(
                    r"p{0.20\textwidth} p{0.16\textwidth} p{0.30\textwidth} "
                    r"p{0.16\textwidth} p{0.08\textwidth}"
                ),
                header=["Gate", "Tool", "Local command", "CI workflow", "Blocking"],
                rows=rows,
                caption="Quality gates implemented in the repository.",
                table_label="tab:quality-gates",
                escape_cells=False,
            ),
            "\\section{Gate configuration}",
            longtable(
                column_spec=r"p{0.25\textwidth} p{0.65\textwidth}",
                header=["Setting", "Value"],
                rows=[
                    ["ruff line-length", escape_inline_code(str(tooling["ruff"]["line_length"]))],
                    [
                        "ruff target-version",
                        escape_inline_code(str(tooling["ruff"]["target_version"])),
                    ],
                    [
                        "ruff rule sets",
                        escape_inline_code(", ".join(tooling["ruff"]["select"])),
                    ],
                    [
                        "ruff ignored rules",
                        escape_inline_code(", ".join(tooling["ruff"]["ignore"]) or "none"),
                    ],
                    [
                        "mypy python\\_version",
                        escape_inline_code(str(tooling["mypy"]["python_version"])),
                    ],
                    ["mypy strict", "yes" if tooling["mypy"]["strict"] else "no"],
                    [
                        "mypy plugins",
                        escape_inline_code(", ".join(tooling["mypy"]["plugins"]) or "none"),
                    ],
                    [
                        "mypy overrides",
                        f"{len(tooling['mypy']['overrides'])} modules with "
                        "\\texttt{ignore\\_missing\\_imports} (optional extras)",
                    ],
                    [
                        "pytest testpaths",
                        escape_inline_code(", ".join(tooling["pytest"]["testpaths"])),
                    ],
                    ["pytest addopts", escape_inline_code(tooling["pytest"]["addopts"])],
                    [
                        "build backend",
                        escape_inline_code(str(tooling["build_system"].get("build-backend", ""))),
                    ],
                ],
                caption="Configured behaviour of each quality tool.",
                table_label="tab:gate-configuration",
                escape_cells=False,
            ),
        ]
    )


def render_ci_cd(inventory: dict[str, Any]) -> str:
    """Render the CI/CD workflow reference and the pipeline figure."""
    workflows = inventory["ci_workflows"]
    overview_rows = []
    for workflow in workflows:
        triggers = ", ".join(sorted(workflow["triggers"]))
        overview_rows.append(
            [
                escape_latex(workflow["name"]),
                escape_inline_code(workflow["file"]),
                escape_latex(triggers),
                str(len(workflow["jobs"])),
                "yes" if workflow["concurrency"] else "no",
            ]
        )

    sections = [
        longtable(
            column_spec=(
                r"p{0.18\textwidth} p{0.28\textwidth} p{0.26\textwidth} "
                r"p{0.08\textwidth} p{0.14\textwidth}"
            ),
            header=["Workflow", "File", "Triggers", "Jobs", "Concurrency group"],
            rows=overview_rows,
            caption="GitHub Actions workflows.",
            table_label="tab:ci-workflows",
            escape_cells=False,
        ),
        render_pipeline_figure(inventory),
    ]

    for workflow in workflows:
        blocks = [
            f"\\section{{Workflow: {escape_latex(workflow['name'])}}}\n"
            f"\\label{{{label('ci', workflow['file'])}}}"
        ]
        trigger_items = []
        for trigger, configuration in sorted(workflow["triggers"].items()):
            if isinstance(configuration, dict) and configuration:
                detail = "; ".join(
                    f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}"
                    for key, value in sorted(configuration.items())
                )
                trigger_items.append(f"{trigger} ({detail})")
            else:
                trigger_items.append(str(trigger))
        blocks.append("\\subsection*{Triggers}\n" + itemize(trigger_items))
        permissions = workflow["permissions"]
        if permissions:
            blocks.append(
                "\\subsection*{Permissions}\n"
                + itemize(
                    [f"{key}: {value}" for key, value in sorted(permissions.items())]
                    if isinstance(permissions, dict)
                    else [str(permissions)]
                )
            )
        for job in workflow["jobs"]:
            matrix = "; ".join(
                f"{key}: {', '.join(map(str, value))}"
                if isinstance(value, list)
                else f"{key}: {value}"
                for key, value in sorted(job["matrix"].items())
            )
            step_rows = [
                [
                    str(index + 1),
                    escape_latex(step["name"] or "(unnamed)"),
                    escape_inline_code(step["uses"]) if step["uses"] else "--",
                    escape_inline_code(step["run"]) if step["run"] else "--",
                ]
                for index, step in enumerate(job["steps"])
            ]
            blocks.append(
                f"\\subsection*{{Job \\texttt{{{escape_latex(job['id'])}}}}}\n"
                + itemize(
                    [
                        f"Runner: {job['runs_on']}",
                        f"Matrix: {matrix or 'none'}",
                        f"Depends on: {', '.join(job['needs']) if job['needs'] else 'nothing'}",
                        f"Steps: {len(job['steps'])}",
                    ]
                )
            )
            blocks.append(
                longtable(
                    column_spec=(
                        r"p{0.04\textwidth} p{0.24\textwidth} p{0.26\textwidth} "
                        r"p{0.36\textwidth}"
                    ),
                    header=["\\#", "Step", "Action", "Command"],
                    rows=step_rows,
                    escape_cells=False,
                )
            )
        sections.append(join_blocks(blocks))
    return join_blocks(sections)


def render_pipeline_figure(inventory: dict[str, Any]) -> str:
    """Generate a TikZ figure of the actual CI quality pipeline."""
    quality = next(
        (
            job
            for workflow in inventory["ci_workflows"]
            if workflow["name"] == "CI"
            for job in workflow["jobs"]
        ),
        None,
    )
    steps = [step["name"] for step in quality["steps"]] if quality else []
    gate_steps = [
        name
        for name in steps
        if name
        in {
            "Lint",
            "Format check",
            "Type check",
            "Test with coverage",
            "Run deterministic demo",
            "Build package",
        }
    ]
    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}[",
        r"  stage/.style={rectangle, draw=black, rounded corners=2pt, align=center,",
        r"    minimum width=32mm, minimum height=8mm, font=\small},",
        r"  flow/.style={-{Latex[length=2mm]}, thick},",
        r"  node distance=6mm",
        r"]",
        r"  \node[stage] (trigger) {Push / pull request\\ \texttt{main}, \texttt{develop}};",
    ]
    previous = "trigger"
    for index, name in enumerate(gate_steps):
        node = f"s{index}"
        lines.append(f"  \\node[stage, below=of {previous}] ({node}) {{{escape_latex(name)}}};")
        lines.append(f"  \\draw[flow] ({previous}) -- ({node});")
        previous = node
    lines.append(
        r"  \node[stage, below=of " + previous + r"] (matrix) "
        r"{Repeated for Python\\ 3.10, 3.11, 3.12};"
    )
    lines.append(f"  \\draw[flow] ({previous}) -- (matrix);")
    lines.extend(
        [
            r"  \node[stage, right=28mm of trigger] (sec) {Security workflow\\ CodeQL "
            r"(python, js/ts)};",
            r"  \node[stage, below=of sec] (dep) {Dependency review\\ (pull requests)};",
            r"  \node[stage, below=of dep] (docker) {Docker workflow\\ build + smoke test};",
            r"  \node[stage, below=of docker] (front) {Frontend workflow\\ npm ci, audit, "
            r"build};",
            r"  \node[stage, below=of front] (rel) {Release workflow\\ tag \texttt{v*} or "
            r"manual};",
            r"  \draw[flow] (trigger) -- (sec);",
            r"  \draw[flow] (sec) -- (dep);",
            r"  \draw[flow] (dep) -- (docker);",
            r"  \draw[flow] (docker) -- (front);",
            r"  \draw[flow] (front) -- (rel);",
            r"\end{tikzpicture}",
            r"\caption{Continuous integration as implemented. The left column is the "
            r"blocking \texttt{quality} job of the CI workflow, executed once per Python "
            r"version in the matrix; the right column lists the independent workflows and "
            r"their own triggers. No workflow declares a \texttt{needs} dependency on "
            r"another, so the columns run concurrently.}",
            r"\label{fig:ci-pipeline}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def render_containers(inventory: dict[str, Any]) -> str:
    """Render container images, Compose stacks, and deployment descriptors."""
    containers = inventory["containers"]
    image_rows = [
        [
            escape_inline_code(entry["path"]),
            ", ".join(escape_inline_code(image) for image in entry["base_images"]),
            ", ".join(escape_latex(port) for port in entry["exposed_ports"]) or "--",
            escape_inline_code(entry["entrypoint"]) if entry["entrypoint"] else "--",
        ]
        for entry in containers["dockerfiles"]
    ]
    service_rows = [
        [
            escape_inline_code(compose["path"]),
            escape_inline_code(service["name"]),
            escape_inline_code(service["image"]) if service["image"] else "built locally",
            ", ".join(escape_latex(str(port)) for port in service["ports"]) or "--",
            "yes" if service["healthcheck"] else "no",
        ]
        for compose in containers["compose"]
        for service in compose["services"]
    ]
    deployment_rows = [
        [escape_inline_code(entry["path"]), escape_latex(entry["purpose"])]
        for entry in inventory["deployments"]
    ]
    return join_blocks(
        [
            "\\section{Container images}",
            longtable(
                column_spec=(
                    r"p{0.22\textwidth} p{0.24\textwidth} p{0.10\textwidth} p{0.36\textwidth}"
                ),
                header=["Dockerfile", "Base image(s)", "Ports", "Entrypoint"],
                rows=image_rows,
                caption="Container images defined in the repository.",
                table_label="tab:container-images",
                escape_cells=False,
            ),
            "\\section{Compose stacks}",
            longtable(
                column_spec=(
                    r"p{0.26\textwidth} p{0.18\textwidth} p{0.20\textwidth} "
                    r"p{0.14\textwidth} p{0.12\textwidth}"
                ),
                header=["File", "Service", "Image", "Ports", "Healthcheck"],
                rows=service_rows,
                caption="Compose services.",
                table_label="tab:compose-services",
                escape_cells=False,
            ),
            "\\section{Deployment descriptors}",
            longtable(
                column_spec=r"p{0.40\textwidth} p{0.52\textwidth}",
                header=["Descriptor", "Target"],
                rows=deployment_rows,
                caption="Deployment descriptors present in the repository.",
                table_label="tab:deployments",
                escape_cells=False,
            ),
        ]
    )


def render_data_provenance(inventory: dict[str, Any]) -> str:
    """Render dataset provenance and the artifact lineage table."""
    dataset_rows = [
        [
            escape_inline_code(entry["path"]),
            escape_latex(entry["format"]),
            str(entry["records"]) if entry["records"] is not None else "--",
            escape_inline_code(", ".join(entry["columns"])) if entry["columns"] else "--",
            str(entry["bytes"]),
        ]
        for entry in inventory["datasets"]
    ]

    lineage = [
        [
            "data/sample\\_feedback.csv",
            "Authored in the repository (no external source)",
            "\\texttt{load\\_feedback\\_csv}",
            "\\texttt{validate\\_feedback\\_csv} (data contract)",
            "Validated \\texttt{FeedbackRecord} list",
        ],
        [
            "Validated records",
            "\\texttt{load\\_feedback\\_csv}",
            "\\texttt{feedback\\_to\\_chunks}",
            "PII redaction via \\texttt{redact\\_pii}",
            "\\texttt{DocumentChunk} list",
        ],
        [
            "Document chunks",
            "\\texttt{feedback\\_to\\_chunks}",
            "\\texttt{HashingEmbeddingModel.embed}",
            "Dimension check in \\texttt{InMemoryVectorStore.add}",
            ".artifacts/vector\\_store.json",
        ],
        [
            "Vector index",
            "\\texttt{build\\_index}",
            "\\texttt{QueryEngine} / \\texttt{BM25Retriever} / \\texttt{HybridRetriever}",
            "Guardrail context check",
            "Ranked \\texttt{SearchResult} list",
        ],
        [
            "Search results",
            "Retriever",
            "\\texttt{FeedbackInsightAgent.answer}",
            "Evidence-overlap hallucination check",
            "\\texttt{AgentAnswer} with citations",
        ],
        [
            "Agent answers",
            "\\texttt{FeedbackInsightAgent}",
            "\\texttt{evaluate\\_system}",
            "Metric aggregation over evaluation cases",
            "\\texttt{EvaluationReport} JSON",
        ],
        [
            "Validated records",
            "\\texttt{load\\_feedback\\_csv}",
            "\\texttt{export\\_feedback\\_lakehouse}",
            "Partition column validation, record redaction",
            "Partitioned JSONL + table metadata",
        ],
        [
            "Stream events",
            "\\texttt{JsonlFeedbackStream} / Kafka / Kinesis",
            "\\texttt{consume\\_feedback\\_stream}",
            "Per-event contract validation",
            "\\texttt{StreamIngestionResult} + CSV",
        ],
    ]

    return join_blocks(
        [
            "\\section{Version-controlled datasets}",
            "Every dataset used by demos, tests, and CI is authored inside the repository. "
            "The repository records no external data source, no acquisition script, no "
            "download URL, and no checksum manifest, so there is no external provenance "
            "chain to document.",
            longtable(
                column_spec=(
                    r"p{0.26\textwidth} p{0.08\textwidth} p{0.09\textwidth} "
                    r"p{0.39\textwidth} p{0.10\textwidth}"
                ),
                header=["Artifact", "Format", "Records", "Fields", "Bytes"],
                rows=dataset_rows,
                caption="Datasets and example inputs under version control.",
                table_label="tab:datasets",
                escape_cells=False,
            ),
            "\\section{Artifact lineage}",
            longtable(
                column_spec=(
                    r"p{0.17\textwidth} p{0.18\textwidth} p{0.20\textwidth} "
                    r"p{0.22\textwidth} p{0.17\textwidth}"
                ),
                header=[
                    "Source artifact",
                    "Producer",
                    "Transformation",
                    "Validation performed",
                    "Destination artifact",
                ],
                rows=lineage,
                caption="Transformation chain from raw CSV to published artifacts.",
                table_label="tab:lineage",
                escape_cells=False,
            ),
        ]
    )


def render_traceability(
    inventory: dict[str, Any], model: dict[str, Any], graph: dict[str, Any]
) -> str:
    """Render documentation-to-source-to-test traceability tables."""
    tests_by_module: dict[str, list[str]] = defaultdict(list)
    for entry in inventory["tests"]:
        for name in entry["module_under_test"].split(","):
            tests_by_module[name.strip()].append(entry["path"])

    module_rows = []
    untested: list[str] = []
    for node in graph["nodes"]:
        name = node["name"]
        covering = sorted(tests_by_module.get(name, []))
        if not covering:
            untested.append(name)
        module_label = (
            label("module", node["qualified_name"])
            if name != "__init__"
            else label("module", "feedback_intelligence_agent")
        )
        module_rows.append(
            [
                f"\\cref{{{module_label}}}",
                escape_inline_code(node["source"]),
                ", ".join(escape_inline_code(path) for path in covering) or "--",
            ]
        )

    command_rows = [
        [
            f"\\cref{{{label('cli', command['qualified_name'])}}}",
            escape_inline_code(command["function"]),
            escape_inline_code("tests/test_cli.py"),
        ]
        for command in model["commands"]
    ]
    endpoint_rows = [
        [
            f"\\cref{{{label('http', endpoint_key(endpoint))}}}",
            escape_inline_code(f"{endpoint['source']}:{endpoint['line']}"),
            escape_inline_code("tests/test_api.py"),
        ]
        for endpoint in model["endpoints"]
    ]

    blocks = [
        "\\section{Module traceability}",
        "Each documented module maps to exactly one source file and to the test modules "
        "that exercise it. The mapping from test module to subject is derived from the "
        "test file names, which mirror the package modules one-for-one.",
        longtable(
            column_spec=r"p{0.26\textwidth} p{0.32\textwidth} p{0.34\textwidth}",
            header=["Documentation object", "Source", "Tests"],
            rows=module_rows,
            caption="Module to source to test traceability.",
            table_label="tab:traceability-modules",
            escape_cells=False,
        ),
    ]
    if untested:
        blocks.append(
            "The following modules have no test module named after them. They are "
            "exercised indirectly through the integration suites rather than by a "
            "dedicated module:"
        )
        blocks.append(itemize([escape_inline_code(name) for name in untested], escape_items=False))
    blocks.extend(
        [
            "\\section{Command traceability}",
            longtable(
                column_spec=r"p{0.30\textwidth} p{0.36\textwidth} p{0.26\textwidth}",
                header=["Command", "Handler", "Tests"],
                rows=command_rows,
                caption="CLI command to handler to test traceability.",
                table_label="tab:traceability-cli",
                escape_cells=False,
            ),
            "\\section{Endpoint traceability}",
            longtable(
                column_spec=r"p{0.34\textwidth} p{0.36\textwidth} p{0.22\textwidth}",
                header=["Endpoint", "Handler", "Tests"],
                rows=endpoint_rows,
                caption="HTTP endpoint to handler to test traceability.",
                table_label="tab:traceability-http",
                escape_cells=False,
            ),
        ]
    )
    return join_blocks(blocks)


def render_build_interface(inventory: dict[str, Any]) -> str:
    """Render the Makefile target reference."""
    rows = [
        [
            escape_inline_code(target["name"]),
            ", ".join(escape_inline_code(name) for name in target["prerequisites"]) or "--",
            escape_inline_code(" && ".join(target["recipe"])) if target["recipe"] else "--",
        ]
        for target in inventory["make_targets"]
    ]
    return longtable(
        column_spec=r"p{0.18\textwidth} p{0.24\textwidth} p{0.50\textwidth}",
        header=["Target", "Prerequisites", "Recipe"],
        rows=rows,
        caption="Makefile targets: the documented build interface.",
        table_label="tab:make-targets",
        escape_cells=False,
    )


def render_environment(inventory: dict[str, Any]) -> str:
    """Render the ``.env.example`` template with its inline commentary."""
    rows = [
        [
            escape_inline_code(entry["name"]),
            escape_inline_code(entry["example"]) if entry["example"] else "(empty)",
            render_prose(entry["comment"]),
        ]
        for entry in inventory["environment_template"]
    ]
    return join_blocks(
        [
            "The template below is the version-controlled "
            "\\texttt{.env.example}. Every credential slot ships empty; no secret value "
            "exists anywhere in the repository or in this manual.",
            longtable(
                column_spec=r"p{0.34\textwidth} p{0.24\textwidth} p{0.34\textwidth}",
                header=["Variable", "Template value", "Documented purpose"],
                rows=rows,
                caption="Environment template shipped with the repository.",
                table_label="tab:env-template",
                escape_cells=False,
            ),
        ]
    )


def render_all(
    repo_root: Path,
    inventory: dict[str, Any],
    model: dict[str, Any],
    graph: dict[str, Any],
) -> list[Path]:
    """Render every engineering fragment and return the written paths."""
    generated = repo_root / GENERATED_ROOT
    written: list[Path] = []
    _write(
        generated / "repository-structure.tex",
        render_repository_structure(inventory),
        written=written,
    )
    _write(generated / "dependencies.tex", render_dependencies(inventory), written=written)
    _write(generated / "testing.tex", render_testing(inventory), written=written)
    _write(generated / "quality-gates.tex", render_quality_gates(inventory), written=written)
    _write(generated / "ci-cd.tex", render_ci_cd(inventory), written=written)
    _write(generated / "containers.tex", render_containers(inventory), written=written)
    _write(generated / "data-provenance.tex", render_data_provenance(inventory), written=written)
    _write(
        generated / "traceability.tex",
        render_traceability(inventory, model, graph),
        written=written,
    )
    _write(generated / "build-interface.tex", render_build_interface(inventory), written=written)
    _write(generated / "environment.tex", render_environment(inventory), written=written)
    return written


def main(argv: list[str] | None = None) -> int:
    """Render the engineering fragments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)

    repo_root = arguments.repo_root
    inventory = json.loads(
        (repo_root / "docs/metadata/repository-inventory.json").read_text(encoding="utf-8")
    )
    model = json.loads((repo_root / "docs/metadata/api-index.json").read_text(encoding="utf-8"))
    graph = json.loads(
        (repo_root / "docs/metadata/module-dependencies.json").read_text(encoding="utf-8")
    )
    written = render_all(repo_root, inventory, model, graph)
    print(f"engineering: {len(written)} LaTeX fragments -> {GENERATED_ROOT.as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
