"""Render architecture LaTeX fragments from the dependency and API models.

The fragments produced here describe structure rather than individual API
objects: the component inventory per architectural layer, the internal module
dependency graph, detected cycles, and TikZ figures generated directly from
the extracted edges.

Diagrams are emitted as TikZ so they compile with a plain LaTeX toolchain and
need no external diagram renderer.
"""

from __future__ import annotations

import argparse
import json
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
)

GENERATED_ROOT = Path("docs/latex/generated/architecture")

#: Human-readable description of each architectural layer.
LAYER_DESCRIPTIONS: dict[str, str] = {
    "interface": "Public entry points: the importable package surface, the Typer CLI, "
    "and the FastAPI application.",
    "application": "Orchestration and workflow: wiring components together, running the "
    "agent loop, jobs, experiments, evaluation, and product workflows.",
    "domain": "Computation and domain logic: retrieval scoring, reranking, embeddings, "
    "prompting, parsing, guardrails, and answer-support checks.",
    "data": "Reading, validating, persisting, and exporting feedback data and vector indexes.",
    "infrastructure": "Adapters to processes and services outside the domain: LLM "
    "providers, telemetry sinks, resilience, authorization, rate limiting.",
    "foundation": "Types and configuration shared by every other layer.",
    "unclassified": "Modules with no assigned layer.",
}


def _write(path: Path, body: str, *, written: list[Path]) -> None:
    """Write a generated fragment and record it in the manifest list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{file_header('render_architecture.py')}\n{body.rstrip()}\n", encoding="utf-8")
    written.append(path)


def _module_summaries(model: dict[str, Any]) -> dict[str, str]:
    """Map short module names to their one-line docstring summaries."""
    summaries: dict[str, str] = {}
    for module in model["modules"]:
        short = module["name"].removeprefix("feedback_intelligence_agent.")
        if module["name"] == "feedback_intelligence_agent":
            short = "__init__"
        summaries[short] = module["summary"]
    return summaries


def render_components(graph: dict[str, Any], model: dict[str, Any]) -> str:
    """Render one section per architectural layer with its component table."""
    summaries = _module_summaries(model)
    dependants: dict[str, list[str]] = {node["name"]: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        dependants.setdefault(edge["target"], []).append(edge["source"])
    dependencies: dict[str, list[str]] = {node["name"]: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        dependencies.setdefault(edge["source"], []).append(edge["target"])

    blocks: list[str] = []
    for layer in graph["layer_order"]:
        members = graph["layers"].get(layer)
        if not members:
            continue
        rows = []
        for name in members:
            node = next(item for item in graph["nodes"] if item["name"] == name)
            module_label = (
                label("module", node["qualified_name"])
                if name != "__init__"
                else label("module", "feedback_intelligence_agent")
            )
            rows.append(
                [
                    f"\\cref{{{module_label}}}",
                    render_prose(summaries.get(name, "")),
                    ", ".join(
                        escape_inline_code(target)
                        for target in sorted(set(dependencies.get(name, [])))
                    )
                    or "--",
                    ", ".join(
                        escape_inline_code(source)
                        for source in sorted(set(dependants.get(name, [])))
                    )
                    or "--",
                ]
            )
        blocks.append(
            f"\\section{{{escape_latex(layer.title())} layer}}\n\\label{{{label('layer', layer)}}}"
        )
        blocks.append(escape_latex(LAYER_DESCRIPTIONS.get(layer, "")))
        blocks.append(
            longtable(
                column_spec=(
                    r"p{0.17\textwidth} p{0.33\textwidth} p{0.22\textwidth} p{0.22\textwidth}"
                ),
                header=["Component", "Responsibility", "Depends on", "Depended on by"],
                rows=rows,
                caption=f"Components of the {layer} layer.",
                table_label=f"tab:layer-{layer}",
                escape_cells=False,
            )
        )
    return join_blocks(blocks)


def render_layer_diagram(graph: dict[str, Any]) -> str:
    """Generate the TikZ layer diagram from the extracted layer edges."""
    layers = [layer for layer in graph["layer_order"] if graph["layers"].get(layer)]
    positions = {layer: index for index, layer in enumerate(layers)}

    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}[",
        r"  layerbox/.style={rectangle, draw=black, rounded corners=2pt, align=center,",
        r"    text width=0.72\textwidth, inner sep=5pt, font=\small},",
        r"  down/.style={-{Latex[length=2mm]}, thick},",
        r"  side/.style={-{Latex[length=2mm]}, thick, dashed},",
        r"  node distance=8mm",
        r"]",
    ]
    for index, layer in enumerate(layers):
        members = ", ".join(graph["layers"][layer])
        content = (
            rf"\textbf{{{escape_latex(layer.title())}}}\\[2pt]"
            rf"\texttt{{\scriptsize {escape_latex(members)}}}"
        )
        placement = "" if index == 0 else f", below=of L{index - 1}"
        lines.append(f"  \\node[layerbox{placement}] (L{index}) {{{content}}};")
    for index in range(len(layers) - 1):
        lines.append(f"  \\draw[down] (L{index}) -- (L{index + 1});")

    upward: list[tuple[str, str]] = []
    for source, targets in graph["layer_edges"].items():
        for target in targets:
            if (
                source in positions
                and target in positions
                and positions[source] > positions[target]
            ):
                upward.append((source, target))
    for source, target in sorted(upward):
        source_index = positions[source]
        target_index = positions[target]
        lines.append(
            f"  \\draw[side] (L{source_index}.east) to[out=20, in=-20] (L{target_index}.east);"
        )
    lines.extend(
        [
            r"\end{tikzpicture}",
            r"\caption{Architectural layers of the package. Solid arrows show the dominant "
            r"downward dependency direction; dashed arrows on the right show the upward "
            r"edges that exist in the source.}",
            r"\label{fig:architecture-layers}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def render_core_graph(graph: dict[str, Any], *, limit: int = 14) -> str:
    """Generate a TikZ figure of the most connected modules and their real edges."""
    ranked = sorted(
        graph["nodes"],
        key=lambda node: (-(node["fan_in"] + node["fan_out"]), node["name"]),
    )[:limit]
    selected = {node["name"] for node in ranked}
    order = [layer for layer in graph["layer_order"] if graph["layers"].get(layer)]
    rows: dict[str, list[str]] = {layer: [] for layer in order}
    for node in sorted(ranked, key=lambda item: item["name"]):
        rows.setdefault(node["layer"], []).append(node["name"])

    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tikzpicture}[",
        r"  mod/.style={rectangle, draw=black, rounded corners=2pt, font=\scriptsize\ttfamily,",
        r"    minimum height=6mm, inner xsep=3pt},",
        r"  dep/.style={-{Latex[length=1.6mm]}, gray, thin},",
        r"  lazy/.style={-{Latex[length=1.6mm]}, gray, thin, dashed}",
        r"]",
    ]
    coordinates: dict[str, tuple[float, float]] = {}
    for row_index, layer in enumerate(order):
        members = rows.get(layer, [])
        if not members:
            continue
        width = max(len(members) - 1, 0)
        for column_index, name in enumerate(members):
            x = (column_index - width / 2.0) * 3.0
            y = -row_index * 2.0
            coordinates[name] = (x, y)
            lines.append(
                f"  \\node[mod] ({_node_id(name)}) at ({x:.2f},{y:.2f}) {{{escape_latex(name)}}};"
            )
        lines.append(
            f"  \\node[font=\\scriptsize\\bfseries, anchor=west] at "
            f"({(-width / 2.0) * 3.0 - 3.2:.2f},{-row_index * 2.0:.2f}) "
            f"{{{escape_latex(layer)}}};"
        )
    seen: set[tuple[str, str]] = set()
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in selected or target not in selected or source == target:
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        style = "dep" if edge["context"] == "eager" else "lazy"
        lines.append(f"  \\draw[{style}] ({_node_id(source)}) -- ({_node_id(target)});")
    lines.extend(
        [
            r"\end{tikzpicture}%",
            r"}",
            r"\caption{Internal dependency graph of the "
            + str(len(ranked))
            + r" most connected modules. Solid edges are module-level (eager) imports; "
            r"dashed edges are deferred imports performed inside function bodies or "
            r"under \texttt{TYPE\_CHECKING}.}",
            r"\label{fig:module-dependency-core}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def _node_id(name: str) -> str:
    """Return a TikZ-safe node identifier for a module name."""
    return f"n{name.replace('_', 'X')}"


def _paired_rows(rows: list[list[str]]) -> list[list[str]]:
    """Fold a long narrow table into two side-by-side halves.

    A 151-row table of three narrow columns wastes most of the page width. The
    first half of the rows is placed beside the second half, halving the page
    count without changing the reading order down each side.
    """
    if not rows:
        return []
    width = len(rows[0])
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right += [[""] * width] * (len(left) - len(right))
    return [list(first) + list(second) for first, second in zip(left, right, strict=True)]


def render_dependencies(graph: dict[str, Any]) -> str:
    """Render the module dependency tables, coupling metrics, and cycle report."""
    # The imported-name column was dropped: it made this table four pages long
    # while duplicating information the per-component tables already carry.
    edge_rows = [
        [
            escape_inline_code(edge["source"]),
            escape_inline_code(edge["target"]),
            escape_latex(edge["context"]),
        ]
        for edge in graph["edges"]
    ]
    coupling_rows = [
        [
            escape_inline_code(node["name"]),
            escape_latex(node["layer"]),
            str(node["fan_in"]),
            str(node["fan_out"]),
            ", ".join(escape_inline_code(name) for name in node["external_dependencies"]) or "--",
        ]
        for node in sorted(
            graph["nodes"], key=lambda item: (-item["fan_in"], -item["fan_out"], item["name"])
        )
    ]

    runtime_cycles = graph["cycles"]["runtime"]
    deferred_cycles = graph["cycles"]["including_deferred"]
    cycle_blocks = ["\\section{Circular dependencies}"]
    if runtime_cycles:
        cycle_blocks.append(
            "The following strongly connected components exist among module-level "
            "(eager) imports and therefore constitute runtime import cycles:"
        )
        cycle_blocks.append(
            itemize(
                [", ".join(escape_inline_code(name) for name in cycle) for cycle in runtime_cycles],
                escape_items=False,
            )
        )
    else:
        cycle_blocks.append(
            "No cycle exists among module-level imports: the eager import graph is a "
            "directed acyclic graph."
        )
    if deferred_cycles:
        cycle_blocks.append(
            "Counting deferred imports (function-local imports and imports guarded by "
            "\\texttt{TYPE\\_CHECKING}) as edges, the following components become cyclic:"
        )
        cycle_blocks.append(
            itemize(
                [
                    ", ".join(escape_inline_code(name) for name in cycle)
                    for cycle in deferred_cycles
                ],
                escape_items=False,
            )
        )
        cycle_blocks.append(
            "These are not runtime cycles. The deferred import style is the mechanism "
            "that keeps them out of the eager graph; see the dependency-direction "
            "discussion in this chapter. Reporting them is deliberate: they mark the "
            "places where the eager graph is acyclic only because of an import "
            "placement decision."
        )
    else:
        cycle_blocks.append("Counting deferred imports as edges introduces no additional cycles.")

    lazy_rows = [
        [
            escape_inline_code(node["name"]),
            ", ".join(escape_inline_code(name) for name in node["lazy_imports"]) or "--",
            ", ".join(escape_inline_code(name) for name in node["typing_only_imports"]) or "--",
            ", ".join(escape_inline_code(name) for name in node["optional_dependencies"]) or "--",
        ]
        for node in graph["nodes"]
        if node["lazy_imports"] or node["typing_only_imports"] or node["optional_dependencies"]
    ]

    return join_blocks(
        [
            "\\section{Coupling}",
            "Fan-in counts the internal modules that import a module; fan-out counts the "
            "internal modules it imports. High fan-in with low fan-out identifies shared "
            "foundations; high fan-out identifies composition roots.",
            longtable(
                column_spec=(
                    r"p{0.20\textwidth} p{0.15\textwidth} p{0.08\textwidth} "
                    r"p{0.08\textwidth} p{0.35\textwidth}"
                ),
                header=["Module", "Layer", "Fan-in", "Fan-out", "External packages"],
                rows=coupling_rows,
                caption="Internal coupling and external package usage per module.",
                table_label="tab:module-coupling",
                escape_cells=False,
            ),
            "\\section{Deferred imports}",
            "Deferred imports are the mechanism this package uses to break cycles and to "
            "keep optional extras out of the default import path.",
            longtable(
                column_spec=(
                    r"p{0.20\textwidth} p{0.26\textwidth} p{0.26\textwidth} p{0.20\textwidth}"
                ),
                header=[
                    "Module",
                    "Function-local imports",
                    "TYPE\\_CHECKING imports",
                    "Optional packages",
                ],
                rows=lazy_rows,
                caption="Modules that defer imports, and the optional distributions they use.",
                table_label="tab:deferred-imports",
                escape_cells=False,
            ),
            *cycle_blocks,
            "\\section{Complete internal import graph}",
            "Every internal import edge, with the context that determines whether it "
            "can participate in a runtime cycle. The names each edge imports are in "
            "the source; the per-component tables in \\cref{ch:components} give the "
            "same edges grouped by component.",
            longtable(
                column_spec=(
                    r"p{0.14\textwidth} p{0.14\textwidth} p{0.08\textwidth} "
                    r"p{0.14\textwidth} p{0.14\textwidth} p{0.08\textwidth}"
                ),
                header=["Source", "Imports", "Context", "Source", "Imports", "Context"],
                rows=_paired_rows(edge_rows),
                caption="Every internal import edge of the package.",
                table_label="tab:module-edges",
                escape_cells=False,
            ),
        ]
    )


def render_layer_violations(graph: dict[str, Any]) -> str:
    """Report edges that point from an inner layer back towards an outer layer."""
    rank = {layer: index for index, layer in enumerate(graph["layer_order"])}
    node_layers = {node["name"]: node["layer"] for node in graph["nodes"]}
    rows = []
    for edge in graph["edges"]:
        source_layer = node_layers.get(edge["source"], "unclassified")
        target_layer = node_layers.get(edge["target"], "unclassified")
        if rank.get(source_layer, 0) > rank.get(target_layer, 0):
            rows.append(
                [
                    escape_inline_code(edge["source"]),
                    escape_latex(source_layer),
                    escape_inline_code(edge["target"]),
                    escape_latex(target_layer),
                    escape_latex(edge["context"]),
                ]
            )
    body = [
        "The layering in this chapter is a classification applied to the observed "
        "modules, not a constraint enforced by any tool in the repository. The table "
        "below lists every import that runs from an inner layer towards an outer one. "
        "These are reported as facts about the current source; the repository contains "
        "no evidence about why each was introduced.",
        longtable(
            column_spec=(
                r"p{0.20\textwidth} p{0.15\textwidth} p{0.20\textwidth} "
                r"p{0.15\textwidth} p{0.14\textwidth}"
            ),
            header=["Source", "Source layer", "Target", "Target layer", "Context"],
            rows=rows,
            caption="Imports directed from an inner layer towards an outer layer.",
            table_label="tab:layer-inversions",
            escape_cells=False,
        ),
    ]
    return join_blocks(body)


def render_all(repo_root: Path, graph: dict[str, Any], model: dict[str, Any]) -> list[Path]:
    """Render every architecture fragment and return the written paths."""
    generated = repo_root / GENERATED_ROOT
    written: list[Path] = []
    _write(generated / "components.tex", render_components(graph, model), written=written)
    _write(generated / "layer-diagram.tex", render_layer_diagram(graph), written=written)
    _write(generated / "module-graph.tex", render_core_graph(graph), written=written)
    _write(generated / "dependencies.tex", render_dependencies(graph), written=written)
    _write(generated / "layer-violations.tex", render_layer_violations(graph), written=written)
    return written


def main(argv: list[str] | None = None) -> int:
    """Render the architecture fragments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)

    repo_root = arguments.repo_root
    graph = json.loads(
        (repo_root / "docs/metadata/module-dependencies.json").read_text(encoding="utf-8")
    )
    model = json.loads((repo_root / "docs/metadata/api-index.json").read_text(encoding="utf-8"))
    written = render_all(repo_root, graph, model)
    print(f"architecture: {len(written)} LaTeX fragments -> {GENERATED_ROOT.as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
