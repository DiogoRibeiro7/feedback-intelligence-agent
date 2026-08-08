"""Render the reference manual LaTeX fragments from the normalized API model.

This module is the rendering half of the documentation system: it reads
``docs/metadata/api-index.json`` and ``docs/metadata/repository-inventory.json``
and writes LaTeX under ``docs/latex/generated/``. It never parses application
source code, so extraction bugs and rendering bugs stay separable.

Entries follow the layout CRAN reference manuals use: a heading carrying the
object name, the summary immediately underneath, then run-in bold field labels
(``Usage``, ``Arguments``, ``Value``, ``Raises``, ``Details``, ``Source``)
rather than a numbered subsection per field. Arguments are hanging-indent lists
rather than full-width tables, which is the single largest difference in page
count.

Every fragment is deterministic: objects are emitted in ascending
fully-qualified-name order and every table row order is derived from the model.
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
    index_entry,
    join_blocks,
    label,
    listing,
    longtable,
    render_prose,
)

GENERATED_ROOT = Path("docs/latex/generated")


def endpoint_key(endpoint: dict[str, Any]) -> str:
    """Return the stable label key of an HTTP endpoint, e.g. ``GET-/query``."""
    return f"{endpoint['method']}-{endpoint['path']}"


def _write(path: Path, body: str, *, written: list[Path]) -> None:
    """Write a generated LaTeX fragment and record it in the manifest list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{file_header('render_reference.py')}\n{body.rstrip()}\n", encoding="utf-8", newline="\n"
    )
    written.append(path)


def field(name: str, content: str) -> str:
    """Render a run-in bold field label followed by its content."""
    if not content:
        return ""
    return f"\\apifield{{{name}}}{content}"


def usage(signature: str) -> str:
    """Render a signature as a compact, unframed usage block."""
    return listing(signature, language="Python", style="usage")


def _short_module(name: str) -> str:
    """Strip the package prefix from a module name for display."""
    return name.removeprefix("feedback_intelligence_agent.")


def _source_reference(source: str, line: int, repository_url: str, revision: str) -> str:
    """Render a local source path plus, when available, a revision-pinned URL."""
    local = escape_inline_code(f"{source}:{line}")
    if not repository_url or not revision:
        return local
    url = f"{repository_url.rstrip('/')}/blob/{revision}/{source}#L{line}"
    return f"{local}~(\\href{{{url}}}{{online}})"


def _argument_list(parameters: list[dict[str, Any]]) -> str:
    """Render callable parameters as a hanging-indent list.

    The receiver (``self``/``cls``) and parameters carrying no information
    beyond their name are still listed, because a reader consulting a
    reference needs the complete signature vocabulary; what is dropped
    relative to a table is the per-parameter row overhead.
    """
    items: list[str] = []
    for parameter in parameters:
        if parameter["name"] in {"self", "cls"}:
            continue
        annotations: list[str] = []
        if parameter["annotation"]:
            annotations.append(escape_inline_code(parameter["annotation"]))
        if parameter["default"] is not None:
            annotations.append(f"default {escape_inline_code(parameter['default'])}")
        elif parameter["kind"] not in {"var-positional", "var-keyword"}:
            annotations.append("required")
        if parameter["kind"] == "keyword-only":
            annotations.append("keyword-only")
        elif parameter["kind"] == "positional-only":
            annotations.append("positional-only")

        prefix = ""
        if parameter["kind"] == "var-positional":
            prefix = "*"
        elif parameter["kind"] == "var-keyword":
            prefix = "**"

        detail = ", ".join(annotations)
        description = render_prose(parameter["description"])
        body = f"{detail}. {description}" if description else f"{detail}."
        items.append(f"  \\item[{escape_inline_code(prefix + parameter['name'])}] {body}")
    if not items:
        return ""
    return "\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}"


def _raises_inline(raises: list[dict[str, str]]) -> str:
    """Render raised exceptions inline, or as a list when each is described."""
    if not raises:
        return ""
    if not any(entry["description"] for entry in raises):
        return ", ".join(escape_inline_code(entry["exception"]) for entry in raises)
    items = [
        f"  \\item[{escape_inline_code(entry['exception'])}] "
        + (render_prose(entry["description"]) or "Raised by this callable.")
        for entry in raises
    ]
    return "\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}"


def callable_body(documented: dict[str, Any], *, context: dict[str, Any], origin: str = "") -> str:
    """Render the compact body shared by functions, methods, and properties.

    ``origin`` is folded into the Source field rather than occupying a line of
    its own, which saves one typeset line for every documented object.
    """
    parts: list[str] = [render_prose(documented["summary"], fallback="Undocumented.")]

    parts.append(field("Usage", "\n" + usage(documented["signature"])))

    arguments = _argument_list(documented["parameters"])
    if arguments:
        parts.append(field("Arguments", "\n" + arguments))

    returns = documented["returns"]
    if returns["annotation"] and returns["annotation"] != "None":
        description = render_prose(returns["description"])
        annotation = escape_inline_code(returns["annotation"])
        parts.append(field("Value", f"{annotation}. {description}" if description else annotation))
    elif returns["description"]:
        parts.append(field("Value", render_prose(returns["description"])))

    raises = _raises_inline(documented["raises"])
    if raises:
        separator = "\n" if raises.startswith("\\begin") else ""
        parts.append(field("Raises", separator + raises))

    details = " ".join(
        block
        for block in (
            render_prose(documented["description"]),
            render_prose(documented["notes"]),
        )
        if block
    )
    if details:
        parts.append(field("Details", details))

    if documented["examples"]:
        parts.append(field("Examples", "\n" + usage(documented["examples"])))

    source = _source_reference(
        documented["source"],
        documented["line"],
        context["repository_url"],
        context["revision"],
    )
    parts.append(field("Source", f"{source}, {origin}" if origin else source))
    return "\n".join(part for part in parts if part)


def render_function(documented: dict[str, Any], context: dict[str, Any]) -> str:
    """Render a complete entry for one public module-level function."""
    qualified = documented["qualified_name"]
    module = documented["module"]
    heading = (
        "\\apibreak\n"
        f"\\section{{\\texttt{{{escape_latex(documented['name'])}}}}}\n"
        f"\\label{{{label('func', qualified)}}}"
        f"{index_entry(documented['name'], category='Functions')}\n"
    )
    return (
        heading
        + "\n"
        + callable_body(
            documented, context=context, origin=f"module \\cref{{{label('module', module)}}}"
        )
    )


def _member_list(attributes: list[dict[str, Any]], *, is_enum: bool) -> str:
    """Render class attributes, Pydantic fields, or enum members compactly."""
    items: list[str] = []
    for attribute in attributes:
        details: list[str] = []
        if is_enum:
            details.append(f"value {escape_inline_code(attribute['default'] or '--')}")
        else:
            if attribute["annotation"]:
                details.append(escape_inline_code(attribute["annotation"]))
            if attribute["default"] is not None:
                details.append(f"default {escape_inline_code(attribute['default'])}")
            elif attribute["required"]:
                details.append("required")
            constraints = ", ".join(
                f"{key}={value}" for key, value in sorted(attribute["constraints"].items())
            )
            if constraints:
                details.append(escape_inline_code(constraints))
        description = render_prose(attribute["description"])
        body = ", ".join(details)
        if description:
            body = f"{body}. {description}" if body else description
        items.append(f"  \\item[{escape_inline_code(attribute['name'])}] {body}")
    if not items:
        return ""
    return "\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}"


def render_class(documented: dict[str, Any], context: dict[str, Any]) -> str:
    """Render a complete entry for one public class."""
    qualified = documented["qualified_name"]
    is_enum = documented["kind"] == "enumeration"
    labels = f"\\label{{{label('class', qualified)}}}"
    if documented["kind"] == "pydantic model":
        labels += f"\\label{{{label('schema', documented['name'])}}}"

    heading = (
        "\\apibreak\n"
        f"\\section{{\\texttt{{{escape_latex(documented['name'])}}}}}\n"
        f"{labels}{index_entry(documented['name'], category='Classes')}\n"
    )
    origin = (
        f"{escape_latex(documented['kind'])}, module "
        f"\\cref{{{label('module', documented['module'])}}}"
    )

    parts: list[str] = [render_prose(documented["summary"], fallback="Undocumented.")]

    if documented["bases"]:
        parts.append(
            field(
                "Inherits",
                ", ".join(escape_inline_code(base) for base in documented["bases"]),
            )
        )

    constructor = documented["constructor"]
    if constructor is not None:
        parts.append(
            field(
                "Usage",
                "\n" + usage(constructor["signature"].replace("__init__", documented["name"], 1)),
            )
        )
        arguments = _argument_list(constructor["parameters"])
        if arguments:
            parts.append(field("Arguments", "\n" + arguments))
        constructor_raises = _raises_inline(constructor["raises"])
        if constructor_raises:
            separator = "\n" if constructor_raises.startswith("\\begin") else ""
            parts.append(field("Raises", separator + constructor_raises))

    members = _member_list(documented["attributes"], is_enum=is_enum)
    if members:
        title = (
            "Members"
            if is_enum
            else ("Fields" if documented["kind"] == "pydantic model" else "Attributes")
        )
        parts.append(field(title, "\n" + members))

    if documented["properties"]:
        items = [
            f"  \\item[{escape_inline_code(prop['name'])}] "
            + (render_prose(prop["summary"]) or "Undocumented.")
            for prop in documented["properties"]
        ]
        parts.append(
            field("Properties", "\n\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}")
        )

    details = " ".join(
        block
        for block in (
            render_prose(documented["description"]),
            render_prose(documented["notes"]),
        )
        if block
    )
    if details:
        parts.append(field("Details", details))

    source = _source_reference(
        documented["source"],
        documented["line"],
        context["repository_url"],
        context["revision"],
    )
    parts.append(field("Source", f"{source}, {origin}"))

    body = "\n".join(part for part in parts if part)

    method_blocks: list[str] = []
    for method in documented["methods"]:
        method_blocks.append(
            f"\\subsection{{\\texttt{{{escape_latex(method['name'])}}}}}\n"
            f"\\label{{{label('method', method['qualified_name'])}}}"
            f"{index_entry(method['name'], category='Methods')}\n"
            + callable_body(method, context=context)
        )
    if method_blocks:
        return heading + "\n" + body + "\n\n" + "\n\n".join(method_blocks)
    return heading + "\n" + body


def _reference_list(items: list[str]) -> str:
    """Render cross-references as a comma-separated run of links."""
    return ", ".join(items)


def render_modules(model: dict[str, Any], context: dict[str, Any]) -> str:
    """Render the module chapter: an overview table plus one entry per module."""
    modules = model["modules"]
    overview = longtable(
        column_spec=r"p{0.24\textwidth} r r p{0.52\textwidth}",
        header=["Module", "Lines", "Objects", "Summary"],
        rows=[
            [
                escape_inline_code(_short_module(module["name"])),
                str(module["line_count"]),
                str(
                    len(module["class_names"])
                    + len(module["function_names"])
                    + len(module["constant_names"])
                ),
                render_prose(module["summary"]),
            ]
            for module in modules
        ],
        caption="Modules of the feedback_intelligence_agent package.",
        table_label="tab:modules",
        escape_cells=False,
    )

    sections: list[str] = [overview]
    for module in modules:
        module_name = module["name"]
        short = _short_module(module_name)
        parts = [
            "\\apibreak\n"
            f"\\section{{\\texttt{{{escape_latex(short)}}}}}\n"
            f"\\label{{{label('module', module_name)}}}"
            f"{index_entry(short, category='Modules')}\n",
            render_prose(module["summary"], fallback="Undocumented module."),
            field(
                "Source", escape_inline_code(module["source"]) + f", {module['line_count']} lines"
            ),
        ]
        if module["exports"]:
            parts.append(
                field(
                    "Exports",
                    ", ".join(escape_inline_code(name) for name in module["exports"]),
                )
            )
        if module["imports_internal"]:
            parts.append(
                field(
                    "Imports",
                    ", ".join(escape_inline_code(name) for name in module["imports_internal"]),
                )
            )
        if module["class_names"]:
            parts.append(
                field(
                    "Classes",
                    _reference_list(
                        [
                            f"\\cref{{{label('class', f'{module_name}.{name}')}}}"
                            for name in module["class_names"]
                        ]
                    ),
                )
            )
        if module["function_names"]:
            parts.append(
                field(
                    "Functions",
                    _reference_list(
                        [
                            f"\\cref{{{label('func', f'{module_name}.{name}')}}}"
                            for name in module["function_names"]
                        ]
                    ),
                )
            )
        if module["constant_names"]:
            parts.append(
                field(
                    "Constants",
                    ", ".join(escape_inline_code(name) for name in module["constant_names"]),
                )
            )
        if module["description"]:
            parts.append(field("Details", render_prose(module["description"])))
        sections.append("\n".join(part for part in parts if part))
    _ = context
    return join_blocks(sections)


def render_types(model: dict[str, Any]) -> str:
    """Render enumerations, protocols, and type aliases as navigation tables."""
    enumerations = [item for item in model["types"] if item.get("kind") == "enumeration"]
    protocols = [item for item in model["types"] if item.get("kind") == "protocol"]
    aliases = [item for item in model["types"] if item.get("kind") == "type alias"]

    return join_blocks(
        [
            "\\section{Enumerations}",
            longtable(
                column_spec=r"p{0.26\textwidth} p{0.30\textwidth} p{0.36\textwidth}",
                header=["Enumeration", "Members", "Summary"],
                rows=[
                    [
                        f"\\cref{{{label('class', item['qualified_name'])}}}",
                        ", ".join(
                            escape_inline_code(member["name"]) for member in item["attributes"]
                        ),
                        render_prose(item["summary"]),
                    ]
                    for item in enumerations
                ],
                caption="Enumerations exposed by the package.",
                table_label="tab:enumerations",
                escape_cells=False,
            ),
            "\\section{Protocols}",
            "Protocols define the structural interfaces the package programs against. "
            "Any object providing the listed methods can be substituted without inheritance.",
            longtable(
                column_spec=r"p{0.26\textwidth} p{0.28\textwidth} p{0.38\textwidth}",
                header=["Protocol", "Required methods", "Summary"],
                rows=[
                    [
                        f"\\cref{{{label('class', item['qualified_name'])}}}",
                        ", ".join(escape_inline_code(method["name"]) for method in item["methods"])
                        or "--",
                        render_prose(item["summary"]),
                    ]
                    for item in protocols
                ],
                caption="Structural protocols exposed by the package.",
                table_label="tab:protocols",
                escape_cells=False,
            ),
            "\\section{Type Aliases}",
            longtable(
                column_spec=r"p{0.24\textwidth} p{0.48\textwidth} p{0.20\textwidth}",
                header=["Alias", "Definition", "Module"],
                rows=[
                    [
                        escape_inline_code(item["name"]),
                        escape_inline_code(item["value"]),
                        escape_inline_code(_short_module(item["module"])),
                    ]
                    for item in aliases
                ],
                caption="Type aliases exposed by the package.",
                table_label="tab:type-aliases",
                escape_cells=False,
            ),
        ]
    )


def render_constants(model: dict[str, Any]) -> str:
    """Render the public module-level constants."""
    return longtable(
        column_spec=r"p{0.24\textwidth} p{0.16\textwidth} p{0.52\textwidth}",
        header=["Constant", "Module", "Value"],
        rows=[
            [
                escape_inline_code(item["name"]),
                escape_inline_code(_short_module(item["module"])),
                escape_inline_code(
                    item["value"] if len(item["value"]) <= 160 else f"{item['value'][:157]}..."
                ),
            ]
            for item in model["constants"]
        ],
        caption="Public module-level constants.",
        table_label="tab:constants",
        escape_cells=False,
    )


def render_schemas(model: dict[str, Any]) -> str:
    """Render the data-contract overview.

    Field-by-field detail lives in each model's class entry rather than being
    repeated here; this chapter is the index into those entries.
    """
    schemas = [item for item in model["classes"] if item["kind"] == "pydantic model"]
    return longtable(
        column_spec=r"p{0.28\textwidth} r p{0.16\textwidth} p{0.42\textwidth}",
        header=["Model", "Fields", "Module", "Purpose"],
        rows=[
            [
                f"\\cref{{{label('class', item['qualified_name'])}}}",
                str(len(item["attributes"])),
                escape_inline_code(_short_module(item["module"])),
                render_prose(item["summary"]),
            ]
            for item in schemas
        ],
        caption="Typed data contracts (Pydantic models) exposed by the package.",
        table_label="tab:schemas",
        escape_cells=False,
    )


def render_cli(model: dict[str, Any], context: dict[str, Any]) -> str:
    """Render the CLI reference: the command tree and one entry per command."""
    commands = model["commands"]
    console_scripts = context["console_scripts"]
    entry_point = next(iter(console_scripts), "feedback-agent")

    overview = longtable(
        column_spec=r"p{0.30\textwidth} p{0.12\textwidth} p{0.46\textwidth}",
        header=["Command", "Group", "Purpose"],
        rows=[
            [
                escape_inline_code(f"{entry_point} {command['qualified_name']}"),
                escape_inline_code(command["group"]) if command["group"] else "--",
                render_prose(command["summary"]),
            ]
            for command in commands
        ],
        caption="Commands exposed by the console script.",
        table_label="tab:cli-commands",
        escape_cells=False,
    )

    sections = [overview]
    for command in commands:
        invocation = f"{entry_point} {command['qualified_name']}"
        options = [
            parameter for parameter in command["parameters"] if parameter["kind"] == "option"
        ]
        arguments = [
            parameter for parameter in command["parameters"] if parameter["kind"] == "argument"
        ]
        syntax_parts = [invocation]
        syntax_parts.extend(parameter["cli_name"] for parameter in arguments)
        if options:
            syntax_parts.append("[OPTIONS]")

        parts = [
            "\\apibreak\n"
            f"\\section{{\\texttt{{{escape_latex(command['qualified_name'])}}}}}\n"
            f"\\label{{{label('cli', command['qualified_name'])}}}"
            f"{index_entry(command['qualified_name'], category='CLI commands')}\n",
            render_prose(command["summary"], fallback="Undocumented command."),
            field("Usage", "\n" + listing(" ".join(syntax_parts), language="bash", style="usage")),
        ]
        if arguments:
            items = [
                f"  \\item[{escape_inline_code(parameter['cli_name'])}] "
                + escape_inline_code(parameter["annotation"] or "--")
                + (f". {render_prose(parameter['help'])}" if parameter["help"] else ".")
                for parameter in arguments
            ]
            parts.append(
                field("Arguments", "\n\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}")
            )
        if options:
            items = []
            for parameter in options:
                detail = [escape_inline_code(parameter["annotation"] or "--")]
                detail.append(
                    f"default {escape_inline_code(parameter['default'])}"
                    if parameter["default"]
                    else "required"
                )
                help_text = render_prose(parameter["help"])
                body = ", ".join(detail)
                items.append(
                    f"  \\item[{escape_inline_code(parameter['cli_name'])}] "
                    + (f"{body}. {help_text}" if help_text else f"{body}.")
                )
            parts.append(
                field("Options", "\n\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}")
            )
        if command["description"]:
            parts.append(field("Details", render_prose(command["description"])))
        parts.append(
            field(
                "Source",
                f"{escape_inline_code(command['function'])}, "
                + _source_reference(
                    command["source"],
                    command["line"],
                    context["repository_url"],
                    context["revision"],
                ),
            )
        )
        sections.append("\n".join(part for part in parts if part))
    return join_blocks(sections)


def render_http_api(model: dict[str, Any], context: dict[str, Any]) -> str:
    """Render the HTTP API reference from the extracted FastAPI routes."""
    endpoints = model["endpoints"]
    overview = longtable(
        column_spec=(r"p{0.07\textwidth} p{0.30\textwidth} r p{0.10\textwidth} p{0.35\textwidth}"),
        header=["Method", "Path", "Status", "Role", "Purpose"],
        rows=[
            [
                escape_latex(endpoint["method"]),
                escape_inline_code(endpoint["path"]),
                escape_latex(endpoint["status_code"]),
                escape_latex(endpoint["required_role"] or "open"),
                render_prose(endpoint["summary"]),
            ]
            for endpoint in endpoints
        ],
        caption="HTTP routes registered by create_app.",
        table_label="tab:http-endpoints",
        escape_cells=False,
    )

    sections = [overview]
    for endpoint in endpoints:
        key = endpoint_key(endpoint)
        parts = [
            "\\apibreak\n"
            f"\\section{{\\texttt{{{escape_latex(endpoint['method'])} "
            f"{escape_latex(endpoint['path'])}}}}}\n"
            f"\\label{{{label('http', key)}}}"
            f"{index_entry(endpoint['path'], category='HTTP endpoints')}\n",
            render_prose(endpoint["summary"], fallback="Undocumented endpoint."),
        ]
        contract: list[str] = [
            f"status {escape_latex(endpoint['status_code'])}",
            f"role {escape_latex(endpoint['required_role'] or 'open')}",
        ]
        if endpoint["request_model"]:
            contract.append(f"request {escape_inline_code(endpoint['request_model'])}")
        if endpoint["response_model"]:
            contract.append(f"response {escape_inline_code(endpoint['response_model'])}")
        parts.append(field("Contract", ", ".join(contract)))

        query_parameters = [
            parameter
            for parameter in endpoint["query_parameters"]
            if parameter["name"] not in {"self", "background_tasks"}
        ]
        if query_parameters:
            items = []
            for parameter in query_parameters:
                location = "path" if f"{{{parameter['name']}}}" in endpoint["path"] else "query"
                detail = [escape_inline_code(parameter["annotation"] or "--"), location]
                detail.append(
                    f"default {escape_inline_code(parameter['default'])}"
                    if parameter["default"]
                    else "required"
                )
                items.append(
                    f"  \\item[{escape_inline_code(parameter['name'])}] {', '.join(detail)}."
                )
            parts.append(
                field("Parameters", "\n\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}")
            )
        if endpoint["description"]:
            parts.append(field("Details", render_prose(endpoint["description"])))
        parts.append(
            field(
                "Source",
                f"{escape_inline_code(endpoint['operation'])}, "
                + _source_reference(
                    endpoint["source"],
                    endpoint["line"],
                    context["repository_url"],
                    context["revision"],
                ),
            )
        )
        sections.append("\n".join(part for part in parts if part))
    return join_blocks(sections)


def render_configuration(model: dict[str, Any], inventory: dict[str, Any]) -> str:
    """Render the configuration reference from Settings and ``.env.example``."""
    fields = model["configuration"]
    comments = {entry["name"]: entry["comment"] for entry in inventory["environment_template"]}
    template_names = {entry["name"] for entry in inventory["environment_template"]}

    rows = []
    for item in fields:
        constraints = ", ".join(
            f"{key}={value}" for key, value in sorted(item["constraints"].items())
        )
        purpose = item["description"] or comments.get(item["env_var"], "")
        purpose_text = render_prose(purpose)
        if constraints:
            constraint_text = escape_inline_code(constraints)
            purpose_text = f"{purpose_text} {constraint_text}".strip()
        rows.append(
            [
                escape_inline_code(item["env_var"]),
                escape_inline_code(item["annotation"]),
                escape_inline_code(item["default"]) if item["default"] else "--",
                "yes" if item["sensitive"] else "no",
                purpose_text or "--",
            ]
        )

    settings_table = longtable(
        column_spec=(
            r"p{0.24\textwidth} p{0.14\textwidth} p{0.15\textwidth} "
            r"p{0.06\textwidth} p{0.31\textwidth}"
        ),
        header=["Variable", "Type", "Default", "Secret", "Purpose / Constraints"],
        rows=rows,
        caption="Environment variables read by the Settings model.",
        table_label="tab:configuration",
        escape_cells=False,
    )

    undocumented = sorted({item["env_var"] for item in fields} - template_names)
    unused = sorted(template_names - {item["env_var"] for item in fields})

    coverage: list[str] = []
    if undocumented:
        coverage.append(
            "Read by the application but absent from \\texttt{.env.example}: "
            + ", ".join(escape_inline_code(name) for name in undocumented)
            + "."
        )
    if unused:
        coverage.append(
            "Present in \\texttt{.env.example} but not read by the \\texttt{Settings} "
            "model (consumed by third-party SDKs or by deployment descriptors): "
            + ", ".join(escape_inline_code(name) for name in unused)
            + "."
        )
    if not coverage:
        coverage.append(
            "Every setting read by the application appears in \\texttt{.env.example} "
            "and vice versa."
        )

    sensitive = [item["env_var"] for item in fields if item["sensitive"]]
    return join_blocks(
        [
            settings_table,
            "\\section{Template coverage}",
            *coverage,
            "\\section{Sensitive values}",
            "The variables below carry credentials. This manual documents their names, "
            "types, and defaults only; no value is read from the environment during "
            "documentation generation, and the declared default of every credential "
            "field in source is \\texttt{None}: "
            + ", ".join(escape_inline_code(name) for name in sensitive)
            + ".",
        ]
    )


def _multicolumn_list(title: str, items: list[str], *, columns: int = 2) -> str:
    """Render a navigation list in balanced columns."""
    if not items:
        return f"\\section{{{title}}}\n\\emph{{none}}"
    body = "\\begin{arglist}\n" + "\n".join(items) + "\n\\end{arglist}"
    return (
        f"\\section{{{title}}}\n"
        f"\\begin{{multicols}}{{{columns}}}\n\\small\n{body}\n\\end{{multicols}}"
    )


def render_indexes(model: dict[str, Any]) -> str:
    """Render alphabetical navigation indexes for every documented object kind."""
    return join_blocks(
        [
            _multicolumn_list(
                "Modules",
                [
                    f"  \\item[{escape_inline_code(_short_module(item['name']))}] "
                    f"\\cref{{{label('module', item['name'])}}}"
                    for item in model["modules"]
                ],
                columns=3,
            ),
            _multicolumn_list(
                "Classes",
                [
                    f"  \\item[{escape_inline_code(item['name'])}] "
                    f"\\cref{{{label('class', item['qualified_name'])}}}"
                    for item in model["classes"]
                ],
                columns=3,
            ),
            _multicolumn_list(
                "Functions",
                [
                    f"  \\item[{escape_inline_code(item['name'])}] "
                    f"\\cref{{{label('func', item['qualified_name'])}}}"
                    for item in model["functions"]
                ],
                columns=3,
            ),
            _multicolumn_list(
                "CLI commands",
                [
                    f"  \\item[{escape_inline_code(item['qualified_name'])}] "
                    f"\\cref{{{label('cli', item['qualified_name'])}}}"
                    for item in model["commands"]
                ],
                columns=2,
            ),
            _multicolumn_list(
                "HTTP endpoints",
                [
                    "  \\item["
                    + escape_inline_code(f"{item['method']} {item['path']}")
                    + f"] \\cref{{{label('http', endpoint_key(item))}}}"
                    for item in model["endpoints"]
                ],
                columns=2,
            ),
            _multicolumn_list(
                "Configuration variables",
                [
                    f"  \\item[{escape_inline_code(item['env_var'])}] "
                    f"{escape_inline_code(item['annotation'])}"
                    for item in model["configuration"]
                ],
                columns=2,
            ),
        ]
    )


def render_project_metadata(inventory: dict[str, Any], model: dict[str, Any]) -> str:
    """Render the project front matter from repository metadata."""
    project = inventory["project"]
    dependencies = inventory["dependencies"]
    maintainers = project.get("maintainers") or []
    if maintainers:
        maintainer_text = "; ".join(
            ", ".join(
                part
                for part in (
                    escape_latex(person["name"]),
                    escape_inline_code(person["email"]) if person["email"] else "",
                    escape_latex(person["affiliation"]),
                )
                if part
            )
            for person in maintainers
        )
        orcid = maintainers[0].get("orcid", "")
    else:
        maintainer_text = ", ".join(escape_latex(author) for author in project["authors"])
        orcid = project.get("orcid", "")

    rows = [
        ["Project", escape_latex(project["name"])],
        ["Description", escape_latex(project["description"])],
        ["Version", escape_inline_code(project["version"])],
        ["Revision", escape_inline_code(project["revision"] or "unknown")],
        ["Revision date", escape_latex(project["revision_date"] or "unknown")],
        ["Runtime", f"Python {escape_inline_code(project['python_requires'])}"],
        ["License", escape_latex(project["license"])],
        ["Maintainer", maintainer_text],
        ["ORCID", escape_latex(orcid)],
        ["DOI", escape_latex(project["doi"])],
        ["Repository", escape_inline_code(project["repository_url"])],
        [
            "Console script",
            ", ".join(
                escape_inline_code(f"{name} = {target}")
                for name, target in project["console_scripts"].items()
            ),
        ],
        [
            "Runtime dependencies",
            escape_inline_code(", ".join(item["name"] for item in dependencies["runtime"])),
        ],
        ["Optional extras", escape_inline_code(", ".join(sorted(dependencies["extras"])))],
        ["Installation", escape_inline_code("poetry install")],
    ]

    counts = longtable(
        column_spec=r"p{0.45\textwidth} r",
        header=["Documented object kind", "Count"],
        rows=[
            ["Modules", str(len(model["modules"]))],
            ["Public classes", str(len(model["classes"]))],
            ["Public functions", str(len(model["functions"]))],
            ["Public types (enums, protocols, aliases)", str(len(model["types"]))],
            ["Public constants", str(len(model["constants"]))],
            ["CLI commands", str(len(model["commands"]))],
            ["HTTP endpoints", str(len(model["endpoints"]))],
            ["Configuration variables", str(len(model["configuration"]))],
        ],
        caption="Size of the documented public surface at this revision.",
        table_label="tab:api-surface",
    )
    return join_blocks(
        [
            longtable(
                column_spec=r"p{0.22\textwidth} p{0.70\textwidth}",
                header=["Field", "Value"],
                rows=[row for row in rows if row[1]],
                escape_cells=False,
            ),
            "\\section*{Documented surface}",
            counts,
        ]
    )


def render_all(repo_root: Path, model: dict[str, Any], inventory: dict[str, Any]) -> list[Path]:
    """Render every reference fragment and return the written paths."""
    generated = repo_root / GENERATED_ROOT
    written: list[Path] = []
    project = inventory["project"]
    context = {
        "repository_url": project["repository_url"],
        "revision": project["revision"],
        "console_scripts": list(project["console_scripts"]),
    }

    _write(
        generated / "project-metadata.tex",
        render_project_metadata(inventory, model),
        written=written,
    )
    _write(generated / "modules.tex", render_modules(model, context), written=written)

    class_includes: list[str] = []
    for item in model["classes"]:
        name = f"{item['qualified_name']}.tex"
        _write(generated / "api" / name, render_class(item, context), written=written)
        # The explicit .tex suffix is required: the file stem contains dots, so
        # TeX would otherwise treat the last component as the extension.
        class_includes.append(f"\\input{{generated/api/{item['qualified_name']}.tex}}")
    _write(
        generated / "api" / "_classes.tex",
        "\n".join(class_includes) or "\\emph{No public classes were extracted.}",
        written=written,
    )

    function_includes: list[str] = []
    for item in model["functions"]:
        name = f"{item['qualified_name']}.tex"
        _write(generated / "api" / name, render_function(item, context), written=written)
        function_includes.append(f"\\input{{generated/api/{item['qualified_name']}.tex}}")
    _write(
        generated / "api" / "_functions.tex",
        "\n".join(function_includes) or "\\emph{No public functions were extracted.}",
        written=written,
    )

    _write(generated / "types.tex", render_types(model), written=written)
    _write(generated / "constants.tex", render_constants(model), written=written)
    _write(generated / "schemas.tex", render_schemas(model), written=written)
    _write(generated / "cli.tex", render_cli(model, context), written=written)
    _write(generated / "http-api.tex", render_http_api(model, context), written=written)
    _write(
        generated / "configuration.tex",
        render_configuration(model, inventory),
        written=written,
    )
    _write(generated / "indexes.tex", render_indexes(model), written=written)
    return written


def main(argv: list[str] | None = None) -> int:
    """Render the reference manual fragments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)

    repo_root = arguments.repo_root
    model = json.loads((repo_root / "docs/metadata/api-index.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (repo_root / "docs/metadata/repository-inventory.json").read_text(encoding="utf-8")
    )
    written = render_all(repo_root, model, inventory)
    print(f"reference: {len(written)} LaTeX fragments -> {GENERATED_ROOT.as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
