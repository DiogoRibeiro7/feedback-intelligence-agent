"""Inspect a LaTeX log and decide whether the PDF build actually succeeded.

The exit status of a LaTeX engine is not a reliable success signal: MiKTeX
returns a non-zero status for a run that merely reported undefined references,
which every first pass does by construction. ``latexmk`` propagates that
status and stops before the later passes that would resolve them.

The build therefore runs ``latexmk -f`` (force through all passes) and this
module decides the outcome from the final log: real errors and unresolved
references fail the build, while typographic warnings are reported as counts.

Usage::

    python -m tools.check_latex_log docs/latex/main.log
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

#: TeX reports errors in one of two formats, and the build uses both.
#:
#: Without ``-file-line-error`` an error line begins with ``! ``. With it --
#: which is what the Makefile passes, because it makes errors far easier to
#: locate -- the same error is printed as ``file:line: message`` and never
#: carries the exclamation mark. Matching only the first form silently reports
#: a broken build as clean, so both are matched here.
_BANG_ERROR_RE = re.compile(r"^! (.+)$", flags=re.MULTILINE)
#: ``[^\s:]+`` deliberately excludes Windows absolute paths (``C:/...``), whose
#: drive colon would otherwise make ordinary log lines look like errors.
_FILE_LINE_ERROR_RE = re.compile(r"^([^\s:]+:\d+): (.+)$", flags=re.MULTILINE)
_UNDEFINED_REF_RE = re.compile(r"Reference `([^']*)' on page [^ ]* undefined")
_UNDEFINED_CITE_RE = re.compile(r"Citation `([^']*)' on page [^ ]* undefined")
_OVERFULL_RE = re.compile(r"^(Overfull|Underfull) \\[hv]box", flags=re.MULTILINE)
_PAGES_RE = re.compile(r"Output written on \S+ \((\d+) pages?, (\d+) bytes\)")
_RERUN_RE = re.compile(r"Rerun to get|Label\(s\) may have changed")
_NO_FILE_RE = re.compile(r"^No file (\S+)\.$", flags=re.MULTILINE)

#: Auxiliary files whose absence during a pass means a list was typeset empty.
#: A run that reports these produced a document with no table of contents, no
#: list of figures, no list of tables, or no index -- a silent, easily missed
#: failure, because the run itself reports no error.
_CONTENTS_SUFFIXES = (".toc", ".lof", ".lot", ".ind")


@dataclass
class LogReport:
    """Summary of one LaTeX log."""

    path: Path
    errors: list[str]
    undefined_references: list[str]
    undefined_citations: list[str]
    boxes: int
    pages: int
    size_bytes: int
    needs_rerun: bool
    missing_contents: list[str]

    @property
    def ok(self) -> bool:
        """Whether the build produced a complete, fully cross-referenced document."""
        return not (
            self.errors
            or self.undefined_references
            or self.undefined_citations
            or self.needs_rerun
            or self.missing_contents
            or self.pages == 0
        )

    def render(self) -> str:
        """Format the report for terminal output."""
        lines = [
            f"latex log: {self.path.as_posix()}",
            f"  pages                : {self.pages}",
            f"  output size          : {self.size_bytes} bytes",
            f"  errors               : {len(self.errors)}",
            f"  undefined references : {len(self.undefined_references)}",
            f"  undefined citations  : {len(self.undefined_citations)}",
            f"  over/underfull boxes : {self.boxes} (typographic warnings only)",
            f"  rerun required       : {'yes' if self.needs_rerun else 'no'}",
            f"  empty contents lists : {len(self.missing_contents)}",
        ]
        for name in self.missing_contents:
            lines.append(
                f"  EMPTY LIST {name} was absent during this pass, so the "
                "corresponding list was typeset empty; the engine must run again"
            )
        for error in self.errors[:10]:
            lines.append(f"  ERROR {error}")
        for reference in sorted(set(self.undefined_references))[:10]:
            lines.append(f"  UNDEFINED REFERENCE {reference}")
        for citation in sorted(set(self.undefined_citations))[:10]:
            lines.append(f"  UNDEFINED CITATION {citation}")
        return "\n".join(lines)


def _collect_errors(text: str) -> list[str]:
    """Collect engine errors in both of TeX's error-reporting formats."""
    errors = [line.strip() for line in _BANG_ERROR_RE.findall(text)]
    errors.extend(
        f"{location}: {message.strip()}" for location, message in _FILE_LINE_ERROR_RE.findall(text)
    )
    return errors


def analyse(path: Path) -> LogReport:
    """Parse a LaTeX log file into a :class:`LogReport`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pages_match = _PAGES_RE.search(text)
    return LogReport(
        path=path,
        errors=_collect_errors(text),
        undefined_references=_UNDEFINED_REF_RE.findall(text),
        undefined_citations=_UNDEFINED_CITE_RE.findall(text),
        boxes=len(_OVERFULL_RE.findall(text)),
        pages=int(pages_match.group(1)) if pages_match else 0,
        size_bytes=int(pages_match.group(2)) if pages_match else 0,
        needs_rerun=bool(_RERUN_RE.search(text)),
        missing_contents=sorted(
            name for name in _NO_FILE_RE.findall(text) if name.endswith(_CONTENTS_SUFFIXES)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Report on a LaTeX log and exit non-zero when the build was not clean."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to the .log file produced by the engine.")
    arguments = parser.parse_args(argv)

    if not arguments.log.exists():
        print(f"latex log not found: {arguments.log.as_posix()}")
        print("run 'make docs-pdf', or install a LaTeX toolchain providing latexmk")
        return 1

    report = analyse(arguments.log)
    print(report.render())
    if report.ok:
        print("PDF build clean: no errors, no unresolved cross-references")
        return 0
    print("PDF build not clean")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
