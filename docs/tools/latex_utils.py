"""Pure helpers for emitting valid LaTeX from extracted repository metadata.

Every function here is deterministic and side-effect free: the same input
always produces the same LaTeX string. Escaping is the critical
responsibility, because docstrings and source text are arbitrary and must
never be able to break the LaTeX build.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

#: Replacements applied, in order, by :func:`escape_latex`.
#:
#: The backslash is handled first via a private sentinel so the replacement
#: text introduced for other characters is not escaped a second time.
_BACKSLASH_SENTINEL = "\x00BACKSLASH\x00"

_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\\", _BACKSLASH_SENTINEL),
    ("{", r"\{"),
    ("}", r"\}"),
    ("#", r"\#"),
    ("$", r"\$"),
    ("%", r"\%"),
    ("&", r"\&"),
    ("_", r"\_"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
    ("<", r"\textless{}"),
    (">", r"\textgreater{}"),
    ("|", r"\textbar{}"),
)

_LABEL_ALLOWED = re.compile(r"[^A-Za-z0-9:.\-_]+")
_WHITESPACE = re.compile(r"\s+")


def escape_latex(text: str | None) -> str:
    """Escape every LaTeX special character in ``text``.

    Args:
        text: Arbitrary text, typically a docstring fragment or a source
            identifier. ``None`` is treated as the empty string.

    Returns:
        Text that is safe to place in LaTeX running text.
    """
    if not text:
        return ""
    escaped = text
    for character, replacement in _ESCAPES:
        escaped = escaped.replace(character, replacement)
    return escaped.replace(_BACKSLASH_SENTINEL, r"\textbackslash{}")


#: Characters after which a long identifier may be broken across lines.
_BREAK_AFTER: tuple[str, ...] = ("\\_", ".", "/", ",", "]", ")", "-")


def escape_inline_code(text: str | None) -> str:
    r"""Escape ``text`` and wrap it in ``\texttt`` for inline code.

    TeX does not hyphenate typewriter text, so a long dotted identifier such
    as a fully-qualified name would overflow a narrow table column. Explicit
    ``\allowbreak`` opportunities are inserted after the separator characters
    of identifiers and paths, which lets such a name wrap without introducing
    a hyphen that would change what the name reads as.
    """
    if not text:
        return ""
    escaped = escape_latex(text)
    for separator in _BREAK_AFTER:
        escaped = escaped.replace(separator, f"{separator}\\allowbreak{{}}")
    return rf"\texttt{{{escaped}}}"


def collapse_whitespace(text: str | None) -> str:
    """Collapse all runs of whitespace into single spaces."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def paragraphs(text: str | None) -> list[str]:
    """Split text into paragraphs on blank lines, collapsing inner newlines."""
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text.strip())
    return [collapse_whitespace(block) for block in blocks if collapse_whitespace(block)]


def render_prose(text: str | None, *, fallback: str = "") -> str:
    r"""Render docstring prose as escaped LaTeX paragraphs.

    Double-backtick reStructuredText literals (```like this``) and Sphinx
    cross-reference roles (``:class:`Name```) are converted to ``\\texttt``
    before escaping, because they are pervasive in this repository's
    docstrings and read poorly when shown verbatim.
    """
    blocks = paragraphs(text)
    if not blocks:
        return escape_latex(fallback)
    return "\n\n".join(_render_inline_markup(block) for block in blocks)


_ROLE_PATTERN = re.compile(r":[a-zA-Z]+:`~?([^`]+)`")
_LITERAL_PATTERN = re.compile(r"``([^`]+)``")


def _render_inline_markup(block: str) -> str:
    """Convert inline docstring markup to LaTeX, escaping everything else."""
    placeholders: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(1))
        return f"\x01{len(placeholders) - 1}\x01"

    staged = _LITERAL_PATTERN.sub(_stash, block)
    staged = _ROLE_PATTERN.sub(_stash, staged)
    escaped = escape_latex(staged)
    for index, literal in enumerate(placeholders):
        escaped = escaped.replace(f"\x01{index}\x01", escape_inline_code(literal))
    return escaped


def label(kind: str, name: str) -> str:
    """Build a stable LaTeX label key such as ``func:package.module.name``."""
    sanitized = _LABEL_ALLOWED.sub("-", name).strip("-")
    return f"{kind}:{sanitized}"


def cref(kind: str, name: str) -> str:
    """Reference a labelled object with ``\\cref``."""
    return rf"\cref{{{label(kind, name)}}}"


def listing(code: str, *, language: str = "Python", style: str | None = None) -> str:
    """Wrap code in a ``lstlisting`` environment.

    Args:
        code: Verbatim body.
        language: Listings language for syntax highlighting.
        style: Optional named ``lstdefinestyle``. The reference entries use
            ``usage`` for compact, unframed signature blocks.

    ``listings`` reproduces its body verbatim, so no escaping is applied. The
    only unsafe content would be the literal end-of-environment marker, which
    is defensively neutralised.
    """
    body = code.replace(r"\end{lstlisting}", r"\end {lstlisting}")
    options = f"language={language}"
    if style is not None:
        options = f"style={style}, {options}"
    return f"\\begin{{lstlisting}}[{options}]\n{body}\n\\end{{lstlisting}}"


def verbatim(text: str) -> str:
    """Wrap text in a ``verbatim`` environment (used for directory trees)."""
    body = text.replace(r"\end{verbatim}", r"\end {verbatim}")
    return f"\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}"


def longtable(
    *,
    column_spec: str,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    caption: str | None = None,
    table_label: str | None = None,
    escape_cells: bool = True,
) -> str:
    r"""Render a ``longtable`` with booktabs rules and a repeating header.

    Args:
        column_spec: LaTeX column specification, e.g. ``p{0.3\\textwidth}l``.
        header: Header cell contents (escaped unless ``escape_cells`` is off).
        rows: Body rows; every row must match the header length.
        caption: Optional caption; when given the table appears in the list
            of tables.
        table_label: Optional label key placed after the caption.
        escape_cells: Escape cell content. Disable when cells already carry
            LaTeX markup produced by other helpers in this module.

    Returns:
        A complete ``longtable`` environment.
    """

    def _cells(values: Sequence[str]) -> str:
        rendered = [escape_latex(value) if escape_cells else (value or "") for value in values]
        return " & ".join(rendered)

    lines = [f"\\begin{{longtable}}{{{_ragged_columns(column_spec)}}}"]
    if caption is not None:
        label_markup = f"\\label{{{table_label}}}" if table_label else ""
        lines.append(f"\\caption{{{escape_latex(caption)}}}{label_markup}\\\\")
    lines.append(r"\toprule")
    lines.append(f"{_cells(header)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(f"{_cells(header)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{_column_count(column_spec)}}}{{r}}"
        r"{\footnotesize continued on next page}\\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    body_rows = list(rows)
    if not body_rows:
        columns = _column_count(column_spec)
        lines.append(f"\\multicolumn{{{columns}}}{{l}}{{\\emph{{none}}}} \\\\")
    for row in body_rows:
        lines.append(f"{_cells(row)} \\\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


_COLUMN_TOKEN_RE = re.compile(r"p\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}|[lrc]")


def _ragged_columns(column_spec: str) -> str:
    r"""Set every column ragged-right and one size smaller.

    Two adjustments are applied to each declared column. Justified ``p``
    columns cannot stretch enough around long unbreakable identifiers and
    overflow instead, so they are set ragged-right; and reference tables are
    set in ``\small`` so a dense table costs proportionally fewer pages.
    Specifications that already carry a ``>{...}`` prefix are left untouched.
    """
    if r">{" in column_spec:
        return column_spec

    def _decorate(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith("p"):
            return rf">{{\raggedright\arraybackslash\small}}{token}"
        return rf">{{\small}}{token}"

    return _COLUMN_TOKEN_RE.sub(_decorate, column_spec)


def _column_count(column_spec: str) -> int:
    """Count the columns declared by a LaTeX column specification."""
    without_args = re.sub(r"\{[^{}]*\}", "", column_spec)
    return sum(1 for character in without_args if character in "lcrp")


def description_list(items: Sequence[tuple[str, str]], *, escape_values: bool = True) -> str:
    """Render a ``description`` environment from term/value pairs."""
    lines = [r"\begin{description}"]
    for term, value in items:
        rendered = escape_latex(value) if escape_values else value
        lines.append(f"  \\item[{escape_latex(term)}] {rendered}")
    lines.append(r"\end{description}")
    return "\n".join(lines)


def itemize(items: Sequence[str], *, escape_items: bool = True) -> str:
    """Render an ``itemize`` environment, or nothing for an empty sequence."""
    if not items:
        return ""
    lines = [r"\begin{itemize}[nosep]"]
    for item in items:
        lines.append(f"  \\item {escape_latex(item) if escape_items else item}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def index_entry(term: str, *, category: str | None = None) -> str:
    """Build an ``\\index`` entry, optionally nested under a category."""
    escaped = escape_latex(term).replace("!", r"\string!")
    if category is None:
        return f"\\index{{{escaped}}}"
    return f"\\index{{{escape_latex(category)}!{escaped}}}"


def file_header(source: str) -> str:
    """Return the standard warning comment placed atop every generated file."""
    return (
        "% !TEX root = ../main.tex\n"
        "% ---------------------------------------------------------------\n"
        "% GENERATED FILE - DO NOT EDIT.\n"
        f"% Produced by docs/tools/{source} from the repository source.\n"
        "% Regenerate with: make docs\n"
        "% ---------------------------------------------------------------\n"
    )


def join_blocks(blocks: Iterable[str]) -> str:
    """Join non-empty LaTeX blocks with blank lines between them."""
    return "\n\n".join(block for block in blocks if block and block.strip())
