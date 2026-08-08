"""Tests for the LaTeX documentation toolchain in ``docs/tools``.

The suite is split the same way the toolchain is: pure LaTeX rendering
primitives, static extraction from source, and the validation layer. Extraction
tests run against both synthetic source (for precise control) and the real
package (so the extractor cannot silently stop matching the repository).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):  # pragma: no cover - environment guard
    pytest.skip(
        "the documentation toolchain requires Python 3.11+ (tomllib)",
        allow_module_level=True,
    )

from tools import (  # noqa: E402
    check_latex_log,
    digest,
    extract_api,
    extract_dependencies,
    inspect_repository,
    latex_utils,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "feedback_intelligence_agent"
METADATA_DIR = REPO_ROOT / "docs" / "metadata"

SPECIAL_CHARACTERS = "\\ { } # $ % & _ ~ ^ < > |"


# ---------------------------------------------------------------------------
# LaTeX escaping and rendering primitives.
# ---------------------------------------------------------------------------


def test_escape_latex_handles_every_special_character() -> None:
    """Each LaTeX special character is replaced by a safe sequence."""
    escaped = latex_utils.escape_latex(SPECIAL_CHARACTERS)
    assert r"\textbackslash{}" in escaped
    for expected in (r"\{", r"\}", r"\#", r"\$", r"\%", r"\&", r"\_"):
        assert expected in escaped
    assert r"\textasciitilde{}" in escaped
    assert r"\textasciicircum{}" in escaped
    assert r"\textless{}" in escaped
    assert r"\textgreater{}" in escaped
    assert r"\textbar{}" in escaped


def test_escape_latex_does_not_double_escape_its_own_output() -> None:
    """The backslash sentinel prevents replacement text being escaped again."""
    escaped = latex_utils.escape_latex("100% & _rising_")
    assert escaped == r"100\% \& \_rising\_"
    assert "\\\\%" not in escaped


@pytest.mark.parametrize("value", ["", None])
def test_escape_latex_accepts_empty_input(value: str | None) -> None:
    """Empty and missing text render as the empty string."""
    assert latex_utils.escape_latex(value) == ""


def test_escape_latex_leaves_plain_text_untouched() -> None:
    """Text with no special characters is returned unchanged."""
    assert latex_utils.escape_latex("plain ASCII text 123") == "plain ASCII text 123"


def test_escape_inline_code_wraps_and_adds_break_opportunities() -> None:
    """Qualified names are wrapped in texttt and may wrap across lines."""
    rendered = latex_utils.escape_inline_code("feedback_intelligence_agent.agent")
    assert rendered.startswith(r"\texttt{")
    assert rendered.endswith("}")
    assert r"\_\allowbreak{}" in rendered
    assert r".\allowbreak{}" in rendered


def test_escape_inline_code_of_empty_text_is_empty() -> None:
    """No stray texttt braces are emitted for missing values."""
    assert latex_utils.escape_inline_code(None) == ""


def test_render_prose_converts_literals_and_roles() -> None:
    """reStructuredText literals and Sphinx roles become inline code."""
    rendered = latex_utils.render_prose("Uses ``top_k`` and :class:`Retriever` for search.")
    assert r"\texttt{top\_\allowbreak{}k}" in rendered
    assert r"\texttt{Retriever}" in rendered
    assert "``" not in rendered


def test_render_prose_splits_paragraphs_and_collapses_newlines() -> None:
    """Blank lines separate paragraphs; single newlines are collapsed."""
    rendered = latex_utils.render_prose("first line\nstill first\n\nsecond block")
    assert rendered == "first line still first\n\nsecond block"


def test_render_prose_uses_fallback_when_empty() -> None:
    """An undocumented object renders its fallback text."""
    assert latex_utils.render_prose("", fallback="Undocumented.") == "Undocumented."


def test_label_sanitizes_disallowed_characters() -> None:
    """Label keys keep only characters that are safe in a reference key."""
    assert latex_utils.label("func", "pkg.mod.name") == "func:pkg.mod.name"
    assert latex_utils.label("http", "GET-/query/stream") == "http:GET--query-stream"
    assert latex_utils.label("cli", "reports email-summary") == "cli:reports-email-summary"
    assert " " not in latex_utils.label("cli", "reports email-summary")


def test_longtable_renders_booktabs_structure() -> None:
    """The table carries rules, a repeating header, a caption, and a label."""
    table = latex_utils.longtable(
        column_spec=r"p{0.5\textwidth} l",
        header=["Name", "Value"],
        rows=[["alpha", "1"], ["beta", "2"]],
        caption="Example table",
        table_label="tab:example",
    )
    assert r"\begin{longtable}" in table
    assert r"\toprule" in table and r"\midrule" in table and r"\bottomrule" in table
    assert r"\endhead" in table
    assert r"\caption{Example table}\label{tab:example}" in table
    assert "alpha & 1 \\\\" in table


def test_longtable_makes_columns_ragged_and_small() -> None:
    """Justified p columns overflow around long identifiers; tables are dense."""
    table = latex_utils.longtable(
        column_spec=r"p{0.5\textwidth} l",
        header=["a", "b"],
        rows=[],
    )
    assert r">{\raggedright\arraybackslash\small}p{0.5\textwidth}" in table
    assert r">{\small}l" in table


def test_longtable_leaves_an_explicit_column_prefix_untouched() -> None:
    """A caller that already declared column formatting keeps control of it."""
    spec = r">{\bfseries}l p{0.4\textwidth}"
    table = latex_utils.longtable(column_spec=spec, header=["a", "b"], rows=[])
    assert f"\\begin{{longtable}}{{{spec}}}" in table


def test_longtable_escapes_cells_by_default_and_can_be_disabled() -> None:
    """Cells are escaped unless the caller already emitted LaTeX markup."""
    escaped = latex_utils.longtable(column_spec="l", header=["h"], rows=[["a_b"]])
    assert r"a\_b" in escaped
    raw = latex_utils.longtable(
        column_spec="l", header=["h"], rows=[[r"\texttt{a}"]], escape_cells=False
    )
    assert r"\texttt{a}" in raw


def test_longtable_renders_an_explicit_empty_body() -> None:
    """An empty table says so rather than producing a malformed body."""
    table = latex_utils.longtable(column_spec="l l", header=["a", "b"], rows=[])
    assert r"\emph{none}" in table


def test_listing_neutralises_its_own_end_marker() -> None:
    """Source text cannot terminate the listing environment early."""
    rendered = latex_utils.listing("code \\end{lstlisting} more")
    assert rendered.count(r"\end{lstlisting}") == 1


def test_itemize_and_description_list_render_environments() -> None:
    """List helpers emit the expected environments and escape their content."""
    assert latex_utils.itemize([]) == ""
    items = latex_utils.itemize(["a_b"])
    assert r"\begin{itemize}[nosep]" in items and r"a\_b" in items
    described = latex_utils.description_list([("Key", "value_1")])
    assert r"\item[Key] value\_1" in described


def test_index_entry_escapes_the_makeindex_separator() -> None:
    """An exclamation mark in a term must not create a spurious subentry."""
    assert latex_utils.index_entry("a!b") == "\\index{a\\string!b}"
    assert latex_utils.index_entry("name", category="Functions") == r"\index{Functions!name}"


def test_join_blocks_drops_empty_blocks() -> None:
    """Optional sections that render empty leave no blank gaps."""
    assert latex_utils.join_blocks(["a", "", "  ", "b"]) == "a\n\nb"


def test_file_header_marks_generated_files() -> None:
    """Generated files carry an unmistakable do-not-edit banner."""
    header = latex_utils.file_header("render_reference.py")
    assert "GENERATED FILE" in header
    assert "make docs" in header


# ---------------------------------------------------------------------------
# Docstring parsing and signature rendering.
# ---------------------------------------------------------------------------


def test_split_docstring_separates_summary_description_and_sections() -> None:
    """A Google-style docstring is decomposed into its parts."""
    parsed = extract_api.split_docstring(
        """Do the thing.

        Longer explanation spanning
        two source lines.

        Args:
            first: The first parameter.
            second: The second parameter,
                continued on the next line.

        Returns:
            Something useful.

        Raises:
            ValueError: If the input is wrong.
        """
    )
    assert parsed["summary"] == "Do the thing."
    assert "Longer explanation" in parsed["description"]
    arguments = dict(parsed["fields"]["Args"])
    assert arguments["first"] == "The first parameter."
    assert arguments["second"].endswith("continued on the next line.")
    assert dict(parsed["fields"]["Raises"])["ValueError"] == "If the input is wrong."
    assert "Something useful" in parsed["sections"]["Returns"]


def test_split_docstring_handles_missing_docstring() -> None:
    """An undocumented object yields empty parts rather than raising."""
    parsed = extract_api.split_docstring(None)
    assert parsed == {"summary": "", "description": "", "sections": {}, "fields": {}}


def _function_node(source: str) -> ast.FunctionDef:
    """Parse a single function definition out of a source snippet."""
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_render_signature_preserves_annotations_defaults_and_keyword_only() -> None:
    """The rendered signature is faithful to the declaration."""
    node = _function_node("def search(self, question: str, *, top_k: int = 4) -> list[str]: ...")
    signature = extract_api.render_signature(node, name="search")
    assert signature == "search(self, question: str, *, top_k: int = 4) -> list[str]"


def test_render_signature_handles_varargs_and_async() -> None:
    """Star-args, double-star-args, and async definitions all render."""
    node = ast.parse("async def run(*args: int, **kwargs: str) -> None: ...").body[0]
    assert isinstance(node, ast.AsyncFunctionDef)
    signature = extract_api.render_signature(node, name="run")
    assert signature == "async run(*args: int, **kwargs: str) -> None"


def test_extract_parameters_reports_parameter_kinds() -> None:
    """Each parameter carries the kind it was declared with."""
    node = _function_node("def f(a, /, b, *c, d=1, **e): ...")
    kinds = {parameter.name: parameter.kind for parameter in extract_api.extract_parameters(node)}
    assert kinds == {
        "a": "positional-only",
        "b": "positional-or-keyword",
        "c": "var-positional",
        "d": "keyword-only",
        "e": "var-keyword",
    }


def test_collect_raises_finds_directly_raised_exceptions() -> None:
    """Exception types raised in the body are collected without duplicates."""
    node = _function_node(
        "def f(x):\n"
        "    if x:\n"
        "        raise ValueError('bad')\n"
        "    if not x:\n"
        "        raise ValueError('also bad')\n"
        "    raise custom.ToolError()\n"
    )
    assert extract_api.collect_raises(node) == ["ValueError", "ToolError"]


def test_build_function_doc_merges_documented_and_raised_exceptions() -> None:
    """Documented exceptions that are not raised syntactically are kept."""
    node = _function_node(
        "def f(path):\n"
        '    """Load a file.\n'
        "\n"
        "    Raises:\n"
        "        FileNotFoundError: If the file is absent.\n"
        '    """\n'
        '    raise ValueError("x")\n'
    )
    documented = extract_api.build_function_doc(node, module="m", source="m.py")
    exceptions = [entry.exception for entry in documented.raises]
    assert exceptions == ["ValueError", "FileNotFoundError"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("class A(BaseModel):\n    x: int = 1\n", "pydantic model"),
        ("class A(Protocol):\n    pass\n", "protocol"),
        ("class A(str, Enum):\n    a = 'a'\n", "enumeration"),
        ("class A(RuntimeError):\n    pass\n", "exception"),
        ("@dataclass(frozen=True)\nclass A:\n    x: int\n", "dataclass"),
        ("class A(ABC):\n    pass\n", "abstract base class"),
        ("class A:\n    pass\n", "class"),
    ],
)
def test_build_class_doc_classifies_class_kinds(source: str, expected: str) -> None:
    """Classes are classified from their bases and decorators."""
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ClassDef)
    documented = extract_api.build_class_doc(node, module="m", source="m.py", exports=None)
    assert documented.kind == expected


def test_build_class_doc_reads_pydantic_field_constraints() -> None:
    """Field bounds and defaults are captured for the schema tables."""
    node = ast.parse(
        "class A(BaseModel):\n"
        "    rating: int = Field(ge=1, le=5)\n"
        "    text: str\n"
        "    tags: list[str] = Field(default_factory=list)\n"
    ).body[0]
    assert isinstance(node, ast.ClassDef)
    documented = extract_api.build_class_doc(node, module="m", source="m.py", exports=None)
    fields = {attribute.name: attribute for attribute in documented.attributes}
    assert fields["rating"].constraints == {"ge": "1", "le": "5"}
    assert fields["text"].required is True
    assert fields["tags"].default == "list()"


def test_public_api_respects_dunder_all() -> None:
    """A module declaring __all__ narrows its public surface."""
    assert extract_api._is_public("visible", ["visible"]) is True
    assert extract_api._is_public("hidden", ["visible"]) is False
    assert extract_api._is_public("_private", None) is False
    assert extract_api._is_public("public", None) is True


# ---------------------------------------------------------------------------
# Extraction against the real package.
# ---------------------------------------------------------------------------


def test_extract_cli_commands_matches_the_real_cli() -> None:
    """Typer commands, groups, and option help are extracted from source."""
    commands = extract_api.extract_cli_commands(PACKAGE_DIR / "cli.py", REPO_ROOT)
    names = {command.qualified_name for command in commands}
    assert {"query", "index", "evaluate", "serve"} <= names
    assert "reports save" in names
    assert "answer-feedback analytics" in names

    query = next(command for command in commands if command.qualified_name == "query")
    options = {parameter.cli_name for parameter in query.parameters}
    assert "--top-k" in options
    top_k = next(parameter for parameter in query.parameters if parameter.cli_name == "--top-k")
    assert top_k.help
    assert commands == sorted(commands, key=lambda command: command.qualified_name)


def test_extract_endpoints_matches_the_real_api() -> None:
    """Routes, status codes, and required roles are extracted from decorators."""
    endpoints = extract_api.extract_endpoints(PACKAGE_DIR / "api.py", REPO_ROOT)
    by_key = {(endpoint.method, endpoint.path): endpoint for endpoint in endpoints}

    assert by_key[("GET", "/health")].required_role is None
    assert by_key[("GET", "/ready")].required_role is None
    assert by_key[("POST", "/query")].required_role == "reader"
    assert by_key[("POST", "/chat")].required_role == "writer"
    assert by_key[("POST", "/index")].required_role == "admin"
    assert by_key[("POST", "/ingestion/jobs")].status_code == "202"
    assert by_key[("POST", "/query")].request_model == "QueryRequest"


def test_extract_settings_fields_reports_aliases_and_sensitivity() -> None:
    """Credential fields use vendor variable names and are marked sensitive."""
    fields = {
        item["name"]: item
        for item in extract_api.extract_settings_fields(PACKAGE_DIR / "config.py")
    }
    assert fields["openai_api_key"]["env_var"] == "OPENAI_API_KEY"
    assert fields["openai_api_key"]["sensitive"] is True
    assert fields["data_path"]["env_var"] == "FEEDBACK_AGENT_DATA_PATH"
    assert fields["data_path"]["sensitive"] is False
    assert fields["embedding_dim"]["constraints"] == {"ge": "64", "le": "8192"}


def test_settings_defaults_never_contain_a_credential_value() -> None:
    """Documentation cannot leak a secret because none is declared in source."""
    for item in extract_api.extract_settings_fields(PACKAGE_DIR / "config.py"):
        if item["sensitive"]:
            assert item["default"] in (None, "None")


def test_build_api_index_covers_the_public_surface() -> None:
    """Known public objects appear in the model with the right classification."""
    index = extract_api.build_api_index(REPO_ROOT)

    module_names = {module["name"] for module in index["modules"]}
    assert "feedback_intelligence_agent.agent" in module_names
    assert "feedback_intelligence_agent" in module_names

    classes = {item["qualified_name"]: item for item in index["classes"]}
    assert "feedback_intelligence_agent.agent.FeedbackInsightAgent" in classes
    assert classes["feedback_intelligence_agent.schemas.AgentAnswer"]["kind"] == "pydantic model"
    assert classes["feedback_intelligence_agent.retrieval.Retriever"]["kind"] == "protocol"

    functions = {item["qualified_name"] for item in index["functions"]}
    assert "feedback_intelligence_agent.evaluation.precision_at_k" in functions
    # Private helpers must stay out of the public model.
    assert not any(item["name"].startswith("_") for item in index["functions"])
    assert not any(item["name"].startswith("_") for item in index["classes"])


def test_dunder_all_modules_hide_non_exported_names() -> None:
    """ingestion declares __all__, so its private helpers are not documented."""
    index = extract_api.build_api_index(REPO_ROOT)
    ingestion = next(
        module
        for module in index["modules"]
        if module["name"] == "feedback_intelligence_agent.ingestion"
    )
    assert set(ingestion["function_names"]) <= set(ingestion["exports"])


def test_extracted_model_is_sorted_for_deterministic_rendering() -> None:
    """Rendering order must not depend on filesystem or dict ordering."""
    index = extract_api.build_api_index(REPO_ROOT)
    for key, sort_key in (
        ("modules", "name"),
        ("classes", "qualified_name"),
        ("functions", "qualified_name"),
        ("commands", "qualified_name"),
    ):
        values = [item[sort_key] for item in index[key]]
        assert values == sorted(values)


# ---------------------------------------------------------------------------
# Dependency extraction.
# ---------------------------------------------------------------------------


def test_module_imports_classifies_import_context() -> None:
    """Eager, function-local, and TYPE_CHECKING imports are distinguished."""
    internal, _ = extract_dependencies.module_imports(PACKAGE_DIR / "factory.py")
    contexts = {(edge["target"], edge["context"]) for edge in internal}
    assert ("agent", "eager") in contexts
    assert ("jobs", "lazy") in contexts
    assert ("jobs", "typing") in contexts


def test_find_cycles_detects_only_real_components() -> None:
    """Tarjan components of size one are not reported as cycles."""
    assert extract_dependencies.find_cycles({"a": {"b"}, "b": {"c"}, "c": set()}) == []
    assert extract_dependencies.find_cycles({"a": {"b"}, "b": {"a"}}) == [["a", "b"]]
    cycles = extract_dependencies.find_cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}})
    assert cycles == [["a", "b", "c"]]


def test_find_self_loops() -> None:
    """A module importing itself is reported separately."""
    assert extract_dependencies.find_self_loops({"a": {"a"}, "b": {"a"}}) == ["a"]


def test_dependency_graph_has_no_runtime_cycles() -> None:
    """The eager import graph of the package is acyclic."""
    graph = extract_dependencies.build_dependency_graph(REPO_ROOT)
    assert graph["cycles"]["runtime"] == []
    assert graph["cycles"]["self_loops"] == []


def test_every_module_has_an_assigned_layer() -> None:
    """A new module must be classified rather than silently defaulted."""
    graph = extract_dependencies.build_dependency_graph(REPO_ROOT)
    unclassified = [node["name"] for node in graph["nodes"] if node["layer"] == "unclassified"]
    assert unclassified == []


# ---------------------------------------------------------------------------
# LaTeX log analysis.
# ---------------------------------------------------------------------------


def test_latex_log_analysis_reports_a_clean_build(tmp_path: Path) -> None:
    """A log with no errors and no undefined references is clean."""
    log = tmp_path / "main.log"
    log.write_text("Output written on main.pdf (10 pages, 1234 bytes).\n", encoding="utf-8")
    report = check_latex_log.analyse(log)
    assert report.ok is True
    assert report.pages == 10
    assert report.errors == []


def test_content_digest_ignores_line_ending_style(tmp_path: Path) -> None:
    """The manifest must verify on every platform.

    Git converts line endings on checkout, so a digest taken over raw bytes
    makes every artifact look changed when the repository is checked out on a
    different platform from the one that generated it.
    """
    unix = tmp_path / "unix.tex"
    windows = tmp_path / "windows.tex"
    unix.write_bytes(b"\\section{a}\nbody\n")
    windows.write_bytes(b"\\section{a}\r\nbody\r\n")
    assert digest.content_digest(unix) == digest.content_digest(windows)


def test_content_digest_still_distinguishes_different_content(tmp_path: Path) -> None:
    """Normalising newlines must not make different content collide."""
    first = tmp_path / "a.tex"
    second = tmp_path / "b.tex"
    first.write_bytes(b"alpha\n")
    second.write_bytes(b"beta\n")
    assert digest.content_digest(first) != digest.content_digest(second)


def test_generated_fragments_are_written_with_unix_line_endings() -> None:
    """Generation must be byte-identical across platforms, not just per-platform."""
    _require_generated_docs()
    sample = sorted((REPO_ROOT / "docs" / "latex" / "generated" / "api").glob("*.tex"))[:20]
    assert sample, "no generated API fragments found"
    for path in sample:
        assert b"\r\n" not in path.read_bytes(), f"{path.name} was written with CRLF"


def test_latex_log_analysis_detects_file_line_error_format(tmp_path: Path) -> None:
    """Errors must be found in both of TeX's reporting formats.

    The build passes ``-file-line-error``, which prints errors as
    ``file:line: message`` with no leading exclamation mark. Matching only the
    ``!`` form reports a broken build as clean.
    """
    log = tmp_path / "main.log"
    log.write_text(
        "diagrams/architecture/data-flow.tex:18: Package pgfkeys Error: "
        "The key '/tikz/step' requires a value.\n"
        "diagrams/engineering/lifecycle.tex:25: LaTeX Error: Not allowed in LR mode.\n"
        "Output written on main.pdf (200 pages, 900 bytes).\n",
        encoding="utf-8",
    )
    report = check_latex_log.analyse(log)
    assert len(report.errors) == 2
    assert report.ok is False


def test_latex_log_analysis_ignores_windows_paths_in_log_lines(tmp_path: Path) -> None:
    """A drive letter's colon must not make an ordinary log line look like an error."""
    log = tmp_path / "main.log"
    log.write_text(
        "<C:/Users/x/miktex/fonts/type1/public/lm/lmr10.pfb>\n"
        "Output written on main.pdf (2 pages, 10 bytes).\n",
        encoding="utf-8",
    )
    assert check_latex_log.analyse(log).errors == []


def test_latex_log_analysis_rejects_empty_contents_lists(tmp_path: Path) -> None:
    """A pass that never read the .toc typesets an empty table of contents.

    This failure reports no error and no undefined reference, so it has to be
    detected from the missing auxiliary file or it ships silently.
    """
    log = tmp_path / "main.log"
    log.write_text(
        "No file main.toc.\n"
        "No file main.lof.\n"
        "No file main.lot.\n"
        "Output written on main.pdf (100 pages, 5000 bytes).\n",
        encoding="utf-8",
    )
    report = check_latex_log.analyse(log)
    assert report.errors == []
    assert report.undefined_references == []
    assert report.missing_contents == ["main.lof", "main.lot", "main.toc"]
    assert report.ok is False


def test_latex_log_analysis_ignores_unrelated_missing_files(tmp_path: Path) -> None:
    """Only the contents-bearing auxiliary files gate the build."""
    log = tmp_path / "main.log"
    log.write_text(
        "No file main.xyz.\nOutput written on main.pdf (3 pages, 10 bytes).\n",
        encoding="utf-8",
    )
    report = check_latex_log.analyse(log)
    assert report.missing_contents == []
    assert report.ok is True


def test_latex_log_analysis_reports_errors_and_undefined_references(tmp_path: Path) -> None:
    """Real failures are surfaced, including references that never resolved."""
    log = tmp_path / "main.log"
    log.write_text(
        "! Undefined control sequence.\n"
        "LaTeX Warning: Reference `func:missing' on page 3 undefined on input line 7.\n"
        "LaTeX Warning: Citation `nobody2020' on page 4 undefined on input line 9.\n"
        "Overfull \\hbox (3.0pt too wide) in paragraph\n"
        "Output written on main.pdf (5 pages, 42 bytes).\n",
        encoding="utf-8",
    )
    report = check_latex_log.analyse(log)
    assert report.ok is False
    assert report.errors == ["Undefined control sequence."]
    assert report.undefined_references == ["func:missing"]
    assert report.undefined_citations == ["nobody2020"]
    assert report.boxes == 1


# ---------------------------------------------------------------------------
# Validation of the committed documentation tree.
# ---------------------------------------------------------------------------


def _require_generated_docs() -> None:
    """Skip when the documentation has not been generated in this checkout."""
    if not (METADATA_DIR / "api-index.json").exists():
        pytest.skip("documentation not generated; run 'make docs'")


def test_generated_documentation_passes_every_validation_check() -> None:
    """The committed manual is complete, escaped, unique, and secret-free."""
    _require_generated_docs()
    from tools import validate_docs

    failures = [check for check in validate_docs.run_checks(REPO_ROOT) if not check.passed]
    assert failures == [], "\n".join(check.render() for check in failures)


def test_manifest_is_ordered_by_posix_path() -> None:
    """Ordering must not depend on the platform's path comparison.

    Sorting Path objects orders case-insensitively on Windows and
    case-sensitively elsewhere, which would reorder the whole manifest between
    platforms and make every entry look as though it had drifted.
    """
    _require_generated_docs()
    manifest = json.loads(
        (METADATA_DIR / "documentation-manifest.json").read_text(encoding="utf-8")
    )
    for key in ("generated_files", "metadata_files"):
        paths = [entry["path"] for entry in manifest[key]]
        assert paths == sorted(paths), f"{key} is not ordered by POSIX path"


def test_recorded_sizes_ignore_line_ending_style(tmp_path: Path) -> None:
    """Recorded sizes must reflect content, not the checkout's line endings.

    Asserted against controlled files rather than the committed inventory,
    which is only current as of the last ``make docs`` run; keeping the
    committed model current is the drift check's job, not this test's.
    """
    unix = tmp_path / "unix.py"
    windows = tmp_path / "windows.py"
    unix.write_bytes(b"one\ntwo\nthree\n")
    windows.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    assert windows.stat().st_size != unix.stat().st_size
    assert inspect_repository._content_size(windows) == inspect_repository._content_size(unix)
    assert inspect_repository._content_size(unix) == len(b"one\ntwo\nthree\n")


def test_metadata_files_are_valid_json_and_describe_one_revision() -> None:
    """Every metadata artifact parses and agrees on the documented revision."""
    _require_generated_docs()
    manifest = json.loads(
        (METADATA_DIR / "documentation-manifest.json").read_text(encoding="utf-8")
    )
    inventory = json.loads((METADATA_DIR / "repository-inventory.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == inventory["project"]["revision"]
    assert manifest["counts"]["generated_latex_files"] == len(manifest["generated_files"])
