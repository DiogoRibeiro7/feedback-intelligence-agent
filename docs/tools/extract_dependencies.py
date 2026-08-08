"""Internal module dependency extraction for the documented package.

Imports are classified by the context in which they appear, because the
package deliberately uses two deferred import styles:

``eager``
    A module-level ``from feedback_intelligence_agent.x import y``.

``lazy``
    An import inside a function body, used to break import cycles and to keep
    optional extras out of the default import path.

``typing``
    An import guarded by ``if TYPE_CHECKING:``, which has no runtime effect.

Only eager edges can participate in a runtime import cycle, so cycle detection
is run over the eager subgraph and reported separately for the full graph.

Running the module writes ``docs/metadata/module-dependencies.json``::

    python -m tools.extract_dependencies --repo-root .
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_NAME = "feedback_intelligence_agent"
PACKAGE_DIR = Path("src") / PACKAGE_NAME

#: Architectural layer assigned to every module of the package.
#:
#: This mapping is an authored classification: it records the layer each
#: module was observed to occupy, and is the single place where the
#: architecture chapter's layering is defined. Any module missing from the
#: mapping is reported as ``unclassified`` rather than silently defaulted.
MODULE_LAYERS: dict[str, str] = {
    "__init__": "interface",
    "api": "interface",
    "cli": "interface",
    "agent": "application",
    "factory": "application",
    "jobs": "application",
    "experiments": "application",
    "benchmarking": "application",
    "evaluation": "application",
    "reports": "application",
    "human_feedback": "application",
    "active_learning": "application",
    "email_summaries": "application",
    "retrieval": "domain",
    "lexical_search": "domain",
    "reranking": "domain",
    "embeddings": "domain",
    "chunking": "domain",
    "query_expansion": "domain",
    "citations": "domain",
    "prompts": "domain",
    "prompt_registry": "domain",
    "output_parser": "domain",
    "guardrails": "domain",
    "hallucination": "domain",
    "memory": "domain",
    "tools": "domain",
    "synthetic_data": "domain",
    "privacy": "domain",
    "ingestion": "data",
    "data_contracts": "data",
    "vector_store": "data",
    "qdrant_store": "data",
    "lakehouse": "data",
    "streaming_ingestion": "data",
    "index_updates": "data",
    "llm": "infrastructure",
    "telemetry": "infrastructure",
    "resilience": "infrastructure",
    "auth": "infrastructure",
    "rate_limit": "infrastructure",
    "config": "foundation",
    "schemas": "foundation",
}

#: Rendering order of the layers, from the outside of the system inwards.
LAYER_ORDER: tuple[str, ...] = (
    "interface",
    "application",
    "domain",
    "data",
    "infrastructure",
    "foundation",
    "unclassified",
)

#: Third-party distributions that may only be imported lazily, because they
#: are declared as optional Poetry extras.
OPTIONAL_IMPORTS = frozenset(
    {"anthropic", "boto3", "confluent_kafka", "mlflow", "opentelemetry", "qdrant_client"}
)

_STDLIB_HINT = frozenset(
    {
        "abc",
        "argparse",
        "ast",
        "base64",
        "collections",
        "csv",
        "dataclasses",
        "datetime",
        "email",
        "enum",
        "functools",
        "hashlib",
        "http",
        "importlib",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "random",
        "re",
        "secrets",
        "smtplib",
        "socket",
        "statistics",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "tomllib",
        "types",
        "typing",
        "urllib",
        "uuid",
        "warnings",
    }
)


def _import_context(tree: ast.Module) -> dict[int, str]:
    """Map every import node id to ``eager``, ``lazy``, or ``typing``."""
    context: dict[int, str] = {}

    def _visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_context = current
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                child_context = "lazy"
            elif isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                child_context = "typing"
            if isinstance(child, ast.Import | ast.ImportFrom):
                context[id(child)] = current
            _visit(child, child_context)

    _visit(tree, "eager")
    return context


def module_imports(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return the internal and external imports of one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    contexts = _import_context(tree)
    internal: list[dict[str, str]] = []
    external: list[dict[str, str]] = []
    seen_internal: set[tuple[str, str]] = set()
    seen_external: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            module = node.names[0].name
            names = [alias.name for alias in node.names]
        else:
            continue
        context = contexts.get(id(node), "eager")
        root = module.split(".", 1)[0]
        if module.startswith(PACKAGE_NAME):
            parts = module.split(".")
            target = parts[1] if len(parts) > 1 else "__init__"
            key = (target, context)
            if key not in seen_internal:
                seen_internal.add(key)
                internal.append(
                    {"target": target, "context": context, "names": ", ".join(sorted(names))}
                )
        elif root and root not in _STDLIB_HINT and root != "__future__":
            key = (root, context)
            if key not in seen_external:
                seen_external.add(key)
                external.append({"target": root, "context": context})
    internal.sort(key=lambda item: (item["target"], item["context"]))
    external.sort(key=lambda item: (item["target"], item["context"]))
    return internal, external


def find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Find strongly connected components with more than one member.

    Uses an iterative Tarjan traversal so the result is deterministic and
    independent of the recursion limit.
    """
    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    components: list[list[str]] = []

    for root in sorted(edges):
        if root in indices:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work[-1]
            if child_index == 0:
                indices[node] = index_counter
                lowlink[node] = index_counter
                index_counter += 1
                stack.append(node)
                on_stack[node] = True
            successors = sorted(edges.get(node, set()))
            if child_index < len(successors):
                work[-1] = (node, child_index + 1)
                successor = successors[child_index]
                if successor not in indices:
                    work.append((successor, 0))
                elif on_stack.get(successor):
                    lowlink[node] = min(lowlink[node], indices[successor])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == indices[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    components.append(sorted(component))
    return sorted(components)


def find_self_loops(edges: dict[str, set[str]]) -> list[str]:
    """Return modules that import themselves."""
    return sorted(node for node, targets in edges.items() if node in targets)


def build_dependency_graph(repo_root: Path) -> dict[str, Any]:
    """Build the internal module dependency graph of the package."""
    package_dir = repo_root / PACKAGE_DIR
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    eager_edges: dict[str, set[str]] = defaultdict(set)
    all_edges: dict[str, set[str]] = defaultdict(set)
    fan_in: dict[str, int] = defaultdict(int)

    module_names = sorted(path.stem for path in package_dir.glob("*.py"))
    for path in sorted(package_dir.glob("*.py")):
        name = path.stem
        internal, external = module_imports(path)
        for edge in internal:
            edges.append(
                {
                    "source": name,
                    "target": edge["target"],
                    "context": edge["context"],
                    "names": edge["names"],
                }
            )
            all_edges[name].add(edge["target"])
            if edge["context"] == "eager":
                eager_edges[name].add(edge["target"])
            fan_in[edge["target"]] += 1
        nodes.append(
            {
                "name": name,
                "qualified_name": f"{PACKAGE_NAME}.{name}" if name != "__init__" else PACKAGE_NAME,
                "source": path.relative_to(repo_root).as_posix(),
                "layer": MODULE_LAYERS.get(name, "unclassified"),
                "fan_out": len({edge["target"] for edge in internal}),
                "external_dependencies": sorted({item["target"] for item in external}),
                "optional_dependencies": sorted(
                    {item["target"] for item in external if item["target"] in OPTIONAL_IMPORTS}
                ),
                "lazy_imports": sorted(
                    {edge["target"] for edge in internal if edge["context"] == "lazy"}
                ),
                "typing_only_imports": sorted(
                    {edge["target"] for edge in internal if edge["context"] == "typing"}
                ),
            }
        )

    for node in nodes:
        node["fan_in"] = fan_in.get(node["name"], 0)

    edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["context"]))
    layers = {
        layer: sorted(node["name"] for node in nodes if node["layer"] == layer)
        for layer in LAYER_ORDER
    }
    layer_edges: dict[str, set[str]] = defaultdict(set)
    node_layers = {node["name"]: node["layer"] for node in nodes}
    for edge in edges:
        source_layer = node_layers.get(edge["source"], "unclassified")
        target_layer = node_layers.get(edge["target"], "unclassified")
        if source_layer != target_layer:
            layer_edges[source_layer].add(target_layer)

    return {
        "package": PACKAGE_NAME,
        "module_count": len(module_names),
        "edge_count": len(edges),
        "layer_order": list(LAYER_ORDER),
        "layers": {layer: names for layer, names in layers.items() if names},
        "layer_edges": {layer: sorted(targets) for layer, targets in sorted(layer_edges.items())},
        "nodes": sorted(nodes, key=lambda node: node["name"]),
        "edges": edges,
        "cycles": {
            "runtime": find_cycles(dict(eager_edges)),
            "including_deferred": find_cycles(dict(all_edges)),
            "self_loops": find_self_loops(dict(all_edges)),
        },
        "most_depended_upon": sorted(
            (
                {"module": node["name"], "fan_in": node["fan_in"], "fan_out": node["fan_out"]}
                for node in nodes
            ),
            key=lambda item: (-item["fan_in"], item["module"]),
        )[:10],
    }


def main(argv: list[str] | None = None) -> int:
    """Write the module dependency graph JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("docs/metadata/module-dependencies.json")
    )
    arguments = parser.parse_args(argv)

    graph = build_dependency_graph(arguments.repo_root)
    output = arguments.repo_root / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    runtime_cycles = len(graph["cycles"]["runtime"])
    print(
        f"module-dependencies: {graph['module_count']} modules, {graph['edge_count']} edges, "
        f"{runtime_cycles} runtime cycle(s) -> {arguments.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
