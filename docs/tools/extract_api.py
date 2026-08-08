"""Static extraction of the public API into a normalized documentation model.

The extractor is strictly AST based. It never imports the application package,
so generating documentation cannot open network connections, touch a vector
store, contact an LLM provider, or execute any other application side effect.

Running the module writes ``docs/metadata/api-index.json``::

    python -m tools.extract_api --repo-root . --output docs/metadata/api-index.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_NAME = "feedback_intelligence_agent"
PACKAGE_DIR = Path("src") / PACKAGE_NAME

#: Base classes that identify a Pydantic model.
_PYDANTIC_BASES = frozenset({"BaseModel", "BaseSettings", "ToolOutput"})

#: Section headings recognised in Google-style docstrings.
_SECTION_HEADINGS = (
    "Args",
    "Arguments",
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Example",
    "Examples",
    "Note",
    "Notes",
    "See Also",
)

_SECTION_RE = re.compile(rf"^({'|'.join(_SECTION_HEADINGS)}):\s*$")
_FIELD_RE = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_][\w.\[\], |]*)\s*(?:\((?P<type>[^)]*)\))?:\s*"
    r"(?P<text>.*)$"
)


@dataclass
class Parameter:
    """One callable parameter as declared in the source signature."""

    name: str
    annotation: str | None
    default: str | None
    kind: str
    description: str = ""


@dataclass
class RaisesEntry:
    """One exception type a callable can raise."""

    exception: str
    description: str = ""


@dataclass
class FunctionDoc:
    """A module-level function or a method of a class."""

    name: str
    qualified_name: str
    kind: str
    module: str
    signature: str
    source: str
    line: int
    public: bool
    is_async: bool
    decorators: list[str]
    summary: str
    description: str
    parameters: list[Parameter]
    returns: dict[str, str]
    raises: list[RaisesEntry]
    examples: str = ""
    notes: str = ""
    see_also: list[str] = field(default_factory=list)


@dataclass
class Attribute:
    """A class attribute, Pydantic field, or enum member."""

    name: str
    annotation: str | None
    default: str | None
    required: bool
    description: str = ""
    constraints: dict[str, str] = field(default_factory=dict)


@dataclass
class ClassDoc:
    """A class, Pydantic model, enum, protocol, dataclass, or exception."""

    name: str
    qualified_name: str
    kind: str
    module: str
    bases: list[str]
    source: str
    line: int
    public: bool
    decorators: list[str]
    summary: str
    description: str
    attributes: list[Attribute]
    methods: list[FunctionDoc]
    properties: list[FunctionDoc]
    constructor: FunctionDoc | None = None
    notes: str = ""


@dataclass
class ConstantDoc:
    """A module-level constant or type alias."""

    name: str
    qualified_name: str
    module: str
    kind: str
    annotation: str | None
    value: str
    source: str
    line: int
    public: bool
    description: str = ""


@dataclass
class ModuleDoc:
    """One Python module of the documented package."""

    name: str
    source: str
    line_count: int
    public: bool
    summary: str
    description: str
    exports: list[str]
    imports_internal: list[str]
    class_names: list[str]
    function_names: list[str]
    constant_names: list[str]


@dataclass
class CommandParameter:
    """One CLI option or argument."""

    name: str
    cli_name: str
    kind: str
    annotation: str | None
    default: str | None
    help: str = ""


@dataclass
class CommandDoc:
    """One Typer command exposed by the console script."""

    name: str
    group: str | None
    qualified_name: str
    function: str
    source: str
    line: int
    summary: str
    description: str
    parameters: list[CommandParameter]


@dataclass
class EndpointDoc:
    """One HTTP route registered on the FastAPI application."""

    method: str
    path: str
    operation: str
    source: str
    line: int
    status_code: str
    response_model: str | None
    required_role: str | None
    summary: str
    description: str
    request_model: str | None
    query_parameters: list[Parameter]


def _text(node: ast.AST | None) -> str | None:
    """Unparse an AST node back to source text."""
    if node is None:
        return None
    return ast.unparse(node)


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Return the source text of every decorator applied to a definition."""
    return [ast.unparse(decorator) for decorator in node.decorator_list]


def _is_public(name: str, exports: list[str] | None) -> bool:
    """Decide whether a name belongs to the public API of its module."""
    if name.startswith("_"):
        return False
    if exports is None:
        return True
    return name in exports


def split_docstring(docstring: str | None) -> dict[str, Any]:
    """Split a Google-style docstring into a structured mapping.

    Args:
        docstring: Raw docstring text, or ``None``.

    Returns:
        A mapping with ``summary``, ``description``, ``sections`` (heading to
        raw block text), and ``fields`` (heading to ordered name/description
        pairs) entries.
    """
    if not docstring:
        return {"summary": "", "description": "", "sections": {}, "fields": {}}

    lines = docstring.strip("\n").splitlines()
    body_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in lines:
        heading = _SECTION_RE.match(raw_line.strip())
        if heading is not None:
            current = heading.group(1)
            sections.setdefault(current, [])
            continue
        if current is None:
            body_lines.append(raw_line)
        else:
            sections[current].append(raw_line)

    body = "\n".join(body_lines).strip()
    parts = re.split(r"\n\s*\n", body, maxsplit=1)
    summary = " ".join(parts[0].split()) if parts and parts[0].strip() else ""
    description = parts[1].strip() if len(parts) > 1 else ""

    fields: dict[str, list[tuple[str, str]]] = {}
    for name, block in sections.items():
        if name in {"Args", "Arguments", "Parameters", "Raises", "Attributes"}:
            fields[name] = _parse_field_block(block)
    rendered_sections = {name: "\n".join(block).strip() for name, block in sections.items()}
    return {
        "summary": summary,
        "description": description,
        "sections": rendered_sections,
        "fields": fields,
    }


def _parse_field_block(block: list[str]) -> list[tuple[str, str]]:
    """Parse an indented ``name: description`` docstring block."""
    entries: list[tuple[str, str]] = []
    base_indent: int | None = None
    for raw_line in block:
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if base_indent is None:
            base_indent = indent
        stripped = raw_line.strip()
        match = _FIELD_RE.match(stripped)
        if indent <= base_indent and match is not None:
            entries.append((match.group("name").strip(), match.group("text").strip()))
        elif entries:
            name, text = entries[-1]
            entries[-1] = (name, f"{text} {stripped}".strip())
    return entries


def render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, *, name: str) -> str:
    """Render a readable, source-faithful signature for a callable."""
    parameters = extract_parameters(node)
    rendered: list[str] = []
    seen_keyword_only = False
    for parameter in parameters:
        if parameter.kind == "keyword-only" and not seen_keyword_only:
            rendered.append("*")
            seen_keyword_only = True
        text = parameter.name
        if parameter.kind == "var-positional":
            text = f"*{text}"
            seen_keyword_only = True
        elif parameter.kind == "var-keyword":
            text = f"**{text}"
        if parameter.annotation:
            text = f"{text}: {parameter.annotation}"
        if parameter.default is not None:
            text = (
                f"{text} = {parameter.default}"
                if parameter.annotation
                else (f"{text}={parameter.default}")
            )
        rendered.append(text)
    returns = _text(node.returns)
    suffix = f" -> {returns}" if returns else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{name}({', '.join(rendered)}){suffix}"


def extract_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Parameter]:
    """Extract every declared parameter of a callable, in source order."""
    args = node.args
    parameters: list[Parameter] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults.extend(args.defaults)
    for argument, default in zip(positional, defaults, strict=True):
        kind = "positional-only" if argument in args.posonlyargs else "positional-or-keyword"
        parameters.append(
            Parameter(
                name=argument.arg,
                annotation=_text(argument.annotation),
                default=_text(default),
                kind=kind,
            )
        )
    if args.vararg is not None:
        parameters.append(
            Parameter(
                name=args.vararg.arg,
                annotation=_text(args.vararg.annotation),
                default=None,
                kind="var-positional",
            )
        )
    for argument, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parameters.append(
            Parameter(
                name=argument.arg,
                annotation=_text(argument.annotation),
                default=_text(kw_default),
                kind="keyword-only",
            )
        )
    if args.kwarg is not None:
        parameters.append(
            Parameter(
                name=args.kwarg.arg,
                annotation=_text(args.kwarg.annotation),
                default=None,
                kind="var-keyword",
            )
        )
    return parameters


def collect_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Collect exception type names raised directly in a callable body.

    Results are ordered by the line the ``raise`` appears on, so the generated
    documentation lists exceptions in the order a reader meets them in the
    source rather than in AST traversal order.
    """
    found: list[str] = []
    raises = sorted(
        (child for child in ast.walk(node) if isinstance(child, ast.Raise)),
        key=lambda child: (child.lineno, child.col_offset),
    )
    for child in raises:
        if child.exc is None:
            continue
        exception = child.exc
        if isinstance(exception, ast.Call):
            exception = exception.func
        name = None
        if isinstance(exception, ast.Name):
            name = exception.id
        elif isinstance(exception, ast.Attribute):
            name = exception.attr
        if name is not None and name not in found:
            found.append(name)
    return found


def build_function_doc(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    module: str,
    source: str,
    owner: str | None = None,
    exports: list[str] | None = None,
) -> FunctionDoc:
    """Build the documentation record for a function or method."""
    parsed = split_docstring(ast.get_docstring(node))
    documented = dict(parsed["fields"].get("Args", []) + parsed["fields"].get("Parameters", []))
    parameters = extract_parameters(node)
    for parameter in parameters:
        parameter.description = documented.get(parameter.name, "")

    documented_raises = dict(parsed["fields"].get("Raises", []))
    raised = collect_raises(node)
    for name in documented_raises:
        if name not in raised:
            raised.append(name)
    raises = [
        RaisesEntry(exception=name, description=documented_raises.get(name, "")) for name in raised
    ]

    qualified_owner = f"{module}.{owner}" if owner else module
    decorators = _decorator_names(node)
    kind = "method" if owner else "function"
    if "property" in decorators:
        kind = "property"
    elif any(decorator.endswith("setter") for decorator in decorators):
        kind = "property-setter"
    elif "staticmethod" in decorators:
        kind = "static method"
    elif "classmethod" in decorators:
        kind = "class method"

    return FunctionDoc(
        name=node.name,
        qualified_name=f"{qualified_owner}.{node.name}",
        kind=kind,
        module=module,
        signature=render_signature(node, name=node.name),
        source=source,
        line=node.lineno,
        public=_is_public(node.name, exports) if owner is None else not node.name.startswith("_"),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=decorators,
        summary=parsed["summary"],
        description=parsed["description"],
        parameters=parameters,
        returns={
            "annotation": _text(node.returns) or "",
            "description": _clean_block(parsed["sections"].get("Returns", "")),
        },
        raises=raises,
        examples=_clean_block(
            parsed["sections"].get("Examples", parsed["sections"].get("Example", ""))
        ),
        notes=_clean_block(parsed["sections"].get("Notes", parsed["sections"].get("Note", ""))),
        see_also=[
            line.strip("- ").strip()
            for line in parsed["sections"].get("See Also", "").splitlines()
            if line.strip()
        ],
    )


def _clean_block(block: str) -> str:
    """Normalise the indentation of a docstring section block."""
    if not block:
        return ""
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return ""
    indent = min(len(line) - len(line.lstrip()) for line in lines)
    return "\n".join(line[indent:] for line in lines).strip()


def _field_constraints(call: ast.Call) -> tuple[str | None, dict[str, str], str]:
    """Read the default, constraints, and description from a ``Field(...)`` call."""
    default: str | None = None
    constraints: dict[str, str] = {}
    description = ""
    for index, positional in enumerate(call.args):
        if index == 0:
            default = ast.unparse(positional)
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        value = ast.unparse(keyword.value)
        if keyword.arg == "default":
            default = value
        elif keyword.arg == "default_factory":
            default = f"{value}()"
        elif keyword.arg == "description":
            description = (
                ast.literal_eval(keyword.value)
                if isinstance(keyword.value, ast.Constant)
                else value
            )
        else:
            constraints[keyword.arg] = value
    return default, constraints, str(description)


def _class_kind(node: ast.ClassDef, bases: list[str]) -> str:
    """Classify a class definition from its bases and decorators."""
    decorators = _decorator_names(node)
    base_names = {base.split("[")[0].split(".")[-1] for base in bases}
    if base_names & _PYDANTIC_BASES:
        return "pydantic model"
    if "Protocol" in base_names:
        return "protocol"
    if any(name in {"Enum", "IntEnum", "StrEnum"} or name.endswith("Enum") for name in base_names):
        return "enumeration"
    if base_names & {"Exception", "ValueError", "RuntimeError", "LookupError", "TypeError"}:
        return "exception"
    if any(decorator.startswith("dataclass") for decorator in decorators):
        return "dataclass"
    if "ABC" in base_names:
        return "abstract base class"
    return "class"


def build_class_doc(
    node: ast.ClassDef,
    *,
    module: str,
    source: str,
    exports: list[str] | None,
) -> ClassDoc:
    """Build the documentation record for a class."""
    parsed = split_docstring(ast.get_docstring(node))
    bases = [ast.unparse(base) for base in node.bases]
    kind = _class_kind(node, bases)
    documented_attributes = dict(parsed["fields"].get("Attributes", []))

    attributes: list[Attribute] = []
    methods: list[FunctionDoc] = []
    properties: list[FunctionDoc] = []
    constructor: FunctionDoc | None = None

    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            annotation = _text(statement.annotation)
            default: str | None = None
            constraints: dict[str, str] = {}
            description = documented_attributes.get(name, "")
            if isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func).endswith(
                "Field"
            ):
                default, constraints, field_description = _field_constraints(statement.value)
                description = field_description or description
            elif statement.value is not None:
                default = ast.unparse(statement.value)
            attributes.append(
                Attribute(
                    name=name,
                    annotation=annotation,
                    default=default,
                    required=default is None,
                    description=description,
                    constraints=constraints,
                )
            )
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.startswith("__"):
                continue
            attributes.append(
                Attribute(
                    name=name,
                    annotation=None,
                    default=ast.unparse(statement.value),
                    required=False,
                    description=documented_attributes.get(name, ""),
                )
            )
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            documented = build_function_doc(
                statement, module=module, source=source, owner=node.name
            )
            if statement.name == "__init__":
                constructor = documented
            elif documented.kind.startswith("property"):
                properties.append(documented)
            elif documented.public:
                methods.append(documented)

    return ClassDoc(
        name=node.name,
        qualified_name=f"{module}.{node.name}",
        kind=kind,
        module=module,
        bases=bases,
        source=source,
        line=node.lineno,
        public=_is_public(node.name, exports),
        decorators=_decorator_names(node),
        summary=parsed["summary"],
        description=parsed["description"],
        attributes=attributes,
        methods=methods,
        properties=properties,
        constructor=constructor,
        notes=_clean_block(parsed["sections"].get("Notes", parsed["sections"].get("Note", ""))),
    )


def _module_exports(tree: ast.Module) -> list[str] | None:
    """Read ``__all__`` from a module, if it declares one."""
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            try:
                value = ast.literal_eval(statement.value)
            except ValueError:
                return None
            if isinstance(value, list | tuple):
                return [str(item) for item in value]
    return None


def _internal_imports(tree: ast.Module) -> list[str]:
    """Return the internal package modules imported by a module."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(PACKAGE_NAME) and node.module != PACKAGE_NAME:
                target = node.module.split(".", 1)[1]
                if target not in found:
                    found.append(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_NAME}."):
                    target = alias.name.split(".", 1)[1]
                    if target not in found:
                        found.append(target)
    return sorted(found)


def _constant_kind(name: str, annotation: str | None, value: str) -> str | None:
    """Classify a module-level assignment as a constant, type alias, or nothing."""
    if name.startswith("_"):
        return None
    if value.startswith(("Literal[", "Union[")) or annotation == "TypeAlias":
        return "type alias"
    if name.isupper() or re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        return "constant"
    if annotation is None and re.fullmatch(r"[A-Z][A-Za-z]*", name) and value.startswith("Literal"):
        return "type alias"
    return None


def extract_module(
    path: Path, repo_root: Path
) -> tuple[ModuleDoc, list[ClassDoc], list[FunctionDoc], list[ConstantDoc]]:
    """Extract every documented object from one module file."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    relative = path.relative_to(repo_root).as_posix()
    module_name = f"{PACKAGE_NAME}.{path.stem}" if path.stem != "__init__" else PACKAGE_NAME
    exports = _module_exports(tree)
    parsed = split_docstring(ast.get_docstring(tree))

    classes: list[ClassDoc] = []
    functions: list[FunctionDoc] = []
    constants: list[ConstantDoc] = []

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef):
            classes.append(
                build_class_doc(statement, module=module_name, source=relative, exports=exports)
            )
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append(
                build_function_doc(statement, module=module_name, source=relative, exports=exports)
            )
        elif isinstance(statement, ast.AnnAssign | ast.Assign):
            if isinstance(statement, ast.AnnAssign):
                if not isinstance(statement.target, ast.Name) or statement.value is None:
                    continue
                name = statement.target.id
                annotation = _text(statement.annotation)
                value = ast.unparse(statement.value)
            else:
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    continue
                name = statement.targets[0].id
                annotation = None
                value = ast.unparse(statement.value)
            kind = _constant_kind(name, annotation, value)
            if kind is None:
                continue
            constants.append(
                ConstantDoc(
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    module=module_name,
                    kind=kind,
                    annotation=annotation,
                    value=value,
                    source=relative,
                    line=statement.lineno,
                    public=_is_public(name, exports),
                )
            )

    module_doc = ModuleDoc(
        name=module_name,
        source=relative,
        line_count=len(text.splitlines()),
        public=not path.stem.startswith("_") or path.stem == "__init__",
        summary=parsed["summary"],
        description=parsed["description"],
        exports=exports or [],
        imports_internal=_internal_imports(tree),
        class_names=[item.name for item in classes if item.public],
        function_names=[item.name for item in functions if item.public],
        constant_names=[item.name for item in constants if item.public],
    )
    return module_doc, classes, functions, constants


def extract_cli_commands(path: Path, repo_root: Path) -> list[CommandDoc]:
    """Extract the Typer command tree from the CLI module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative = path.relative_to(repo_root).as_posix()

    groups: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_typer" or not node.args:
            continue
        target = node.args[0]
        if not isinstance(target, ast.Name):
            continue
        for keyword in node.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                groups[target.id] = str(keyword.value.value)

    commands: list[CommandDoc] = []
    for statement in tree.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        for decorator in statement.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr != "command" or not isinstance(decorator.func.value, ast.Name):
                continue
            app_variable = decorator.func.value.id
            explicit = next(
                (
                    str(argument.value)
                    for argument in decorator.args
                    if isinstance(argument, ast.Constant)
                ),
                None,
            )
            name = explicit or statement.name.replace("_", "-")
            group = groups.get(app_variable)
            parsed = split_docstring(ast.get_docstring(statement))
            commands.append(
                CommandDoc(
                    name=name,
                    group=group,
                    qualified_name=f"{group} {name}" if group else name,
                    function=f"{PACKAGE_NAME}.cli.{statement.name}",
                    source=relative,
                    line=statement.lineno,
                    summary=parsed["summary"],
                    description=parsed["description"],
                    parameters=_command_parameters(statement),
                )
            )
    commands.sort(key=lambda command: command.qualified_name)
    return commands


def _command_parameters(node: ast.FunctionDef) -> list[CommandParameter]:
    """Extract Typer options and arguments from a command function."""
    parameters: list[CommandParameter] = []
    for parameter in extract_parameters(node):
        annotation = parameter.annotation or ""
        kind = "option"
        help_text = ""
        explicit_name: str | None = None
        marker = (
            "typer.Argument"
            if "typer.Argument" in annotation
            else ("typer.Option" if "typer.Option" in annotation else None)
        )
        if marker == "typer.Argument":
            kind = "argument"
        if marker is not None:
            help_text, explicit_name = _typer_metadata(annotation)
        base_annotation = annotation.split(",", 1)[0].removeprefix("Annotated[").strip()
        if kind == "argument":
            cli_name = parameter.name.upper()
        else:
            cli_name = explicit_name or f"--{parameter.name.replace('_', '-')}"
        parameters.append(
            CommandParameter(
                name=parameter.name,
                cli_name=cli_name,
                kind=kind,
                annotation=base_annotation or None,
                default=parameter.default,
                help=help_text,
            )
        )
    return parameters


def _typer_metadata(annotation: str) -> tuple[str, str | None]:
    """Read the help text and explicit flag name from a Typer annotation."""
    try:
        expression = ast.parse(annotation, mode="eval").body
    except SyntaxError:  # pragma: no cover - defensive, annotations come from ast.unparse
        return "", None
    help_text = ""
    explicit_name: str | None = None
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue
        function_name = ast.unparse(node.func)
        if not function_name.endswith(("Option", "Argument")):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("-"):
                explicit_name = str(argument.value)
        for keyword in node.keywords:
            if keyword.arg == "help" and isinstance(keyword.value, ast.Constant):
                help_text = str(keyword.value.value)
    return help_text, explicit_name


_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


def extract_endpoints(path: Path, repo_root: Path) -> list[EndpointDoc]:
    """Extract the HTTP routes registered inside ``create_app``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative = path.relative_to(repo_root).as_posix()

    factory = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_app"
        ),
        None,
    )
    if factory is None:
        return []

    roles: dict[str, str] = {}
    for statement in factory.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        rendered = ast.unparse(statement.value)
        match = re.search(r"ApiRole\.(\w+)", rendered)
        if match is not None:
            roles[target.id] = match.group(1)

    endpoints: list[EndpointDoc] = []
    for node in ast.walk(factory):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr
            if method not in _HTTP_METHODS or not decorator.args:
                continue
            route = decorator.args[0]
            if not isinstance(route, ast.Constant):
                continue
            keywords = {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in decorator.keywords
                if keyword.arg is not None
            }
            required_role = None
            dependencies = keywords.get("dependencies", "")
            for variable, role in roles.items():
                if variable in dependencies:
                    required_role = role
            parsed = split_docstring(ast.get_docstring(node))
            request_model = None
            query_parameters: list[Parameter] = []
            for parameter in extract_parameters(node):
                annotation = parameter.annotation or ""
                if annotation.endswith(("Request", "BackgroundTasks")):
                    if annotation != "BackgroundTasks":
                        request_model = annotation
                    continue
                query_parameters.append(parameter)
            endpoints.append(
                EndpointDoc(
                    method=method.upper(),
                    path=str(route.value),
                    operation=node.name,
                    source=relative,
                    line=node.lineno,
                    status_code=keywords.get("status_code", "200"),
                    response_model=keywords.get("response_model"),
                    required_role=required_role,
                    summary=parsed["summary"],
                    description=parsed["description"],
                    request_model=request_model,
                    query_parameters=query_parameters,
                )
            )
    endpoints.sort(key=lambda endpoint: (endpoint.path, endpoint.method))
    return endpoints


def extract_settings_fields(path: Path) -> list[dict[str, Any]]:
    """Extract configuration fields and their environment variables."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    settings = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Settings"),
        None,
    )
    if settings is None:
        return []

    prefix = "FEEDBACK_AGENT_"
    for statement in settings.body:
        if isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == (
            "model_config"
        ):
            match = re.search(r"env_prefix=['\"]([^'\"]*)['\"]", ast.unparse(statement.value))
            if match is not None:
                prefix = match.group(1)

    # Suffix matching, not substring matching: a field such as
    # ``bedrock_max_tokens`` is a size limit, not a credential.
    sensitive_suffixes = (
        "_key",
        "_password",
        "_secret",
        "_token",
        "_credential",
        "_credentials",
    )
    fields: list[dict[str, Any]] = []
    for statement in settings.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        if name == "model_config":
            continue
        annotation = _text(statement.annotation) or ""
        default: str | None = None
        constraints: dict[str, str] = {}
        description = ""
        alias: str | None = None
        if isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func).endswith(
            "Field"
        ):
            default, constraints, description = _field_constraints(statement.value)
            raw_alias = constraints.pop("validation_alias", None)
            if raw_alias is not None:
                alias = raw_alias.strip("'\"")
        elif statement.value is not None:
            default = ast.unparse(statement.value)
        fields.append(
            {
                "name": name,
                "env_var": alias or f"{prefix}{name.upper()}",
                "annotation": annotation,
                "default": default,
                "required": default is None,
                "constraints": constraints,
                "description": description,
                "sensitive": name.lower().endswith(sensitive_suffixes),
                "line": statement.lineno,
            }
        )
    return fields


def build_api_index(repo_root: Path) -> dict[str, Any]:
    """Build the complete normalized API model for the repository."""
    package_dir = repo_root / PACKAGE_DIR
    modules: list[ModuleDoc] = []
    classes: list[ClassDoc] = []
    functions: list[FunctionDoc] = []
    constants: list[ConstantDoc] = []

    for path in sorted(package_dir.glob("*.py")):
        module_doc, module_classes, module_functions, module_constants = extract_module(
            path, repo_root
        )
        modules.append(module_doc)
        classes.extend(module_classes)
        functions.extend(module_functions)
        constants.extend(module_constants)

    modules.sort(key=lambda item: item.name)
    classes.sort(key=lambda item: item.qualified_name)
    functions.sort(key=lambda item: item.qualified_name)
    constants.sort(key=lambda item: item.qualified_name)

    schemas = [item for item in classes if item.kind == "pydantic model" and item.public]
    types = [item for item in classes if item.kind in {"enumeration", "protocol"} and item.public]
    return {
        "package": PACKAGE_NAME,
        "modules": [asdict(item) for item in modules],
        "classes": [asdict(item) for item in classes if item.public],
        "functions": [asdict(item) for item in functions if item.public],
        "types": [asdict(item) for item in types]
        + [asdict(item) for item in constants if item.kind == "type alias" and item.public],
        "constants": [
            asdict(item) for item in constants if item.kind == "constant" and item.public
        ],
        "commands": [
            asdict(item) for item in extract_cli_commands(package_dir / "cli.py", repo_root)
        ],
        "endpoints": [
            asdict(item) for item in extract_endpoints(package_dir / "api.py", repo_root)
        ],
        "schemas": [item.qualified_name for item in schemas],
        "configuration": extract_settings_fields(package_dir / "config.py"),
        "private_objects": {
            "classes": [item.qualified_name for item in classes if not item.public],
            "functions": [item.qualified_name for item in functions if not item.public],
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write the API index JSON for the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("docs/metadata/api-index.json"))
    arguments = parser.parse_args(argv)

    index = build_api_index(arguments.repo_root)
    output = arguments.repo_root / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"api-index: {len(index['modules'])} modules, {len(index['classes'])} classes, "
        f"{len(index['functions'])} functions, {len(index['commands'])} commands, "
        f"{len(index['endpoints'])} endpoints -> {arguments.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
