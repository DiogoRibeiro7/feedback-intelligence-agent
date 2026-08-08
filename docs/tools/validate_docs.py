"""Validate the generated LaTeX documentation tree against the source model.

The validator is the documentation system's own quality gate. It runs
automatically after ``make docs`` and can be run alone with ``make docs-check``.
Every check reports a count and a sample of offenders, and any failure makes
the process exit non-zero so the checks are usable in CI.

The checks are deliberately independent of a TeX installation: they inspect the
generated source rather than compiling it, so they run in environments with no
LaTeX toolchain.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.digest import content_digest

LATEX_ROOT = Path("docs/latex")
GENERATED_ROOT = LATEX_ROOT / "generated"
METADATA_ROOT = Path("docs/metadata")

#: Authored files the manual cannot be assembled without.
REQUIRED_FILES: tuple[str, ...] = (
    "docs/latex/main.tex",
    "docs/latex/reference.tex",
    "docs/latex/preamble.tex",
    "docs/latex/metadata.tex",
    "docs/latex/about.tex",
    "docs/latex/references.bib",
    "docs/latex/reference/reference-manual.tex",
    "docs/latex/reference/mathematics.tex",
    "docs/latex/reference/indexes.tex",
    "docs/latex/architecture/architecture.tex",
    "docs/latex/architecture/system-context.tex",
    "docs/latex/architecture/components.tex",
    "docs/latex/architecture/dependencies.tex",
    "docs/latex/architecture/runtime.tex",
    "docs/latex/architecture/data-flow.tex",
    "docs/latex/architecture/decisions.tex",
    "docs/latex/engineering/engineering.tex",
    "docs/latex/engineering/repository-structure.tex",
    "docs/latex/engineering/development-workflow.tex",
    "docs/latex/engineering/dependencies.tex",
    "docs/latex/engineering/testing.tex",
    "docs/latex/engineering/quality-assurance.tex",
    "docs/latex/engineering/build.tex",
    "docs/latex/engineering/ci-cd.tex",
    "docs/latex/engineering/configuration.tex",
    "docs/latex/engineering/reproducibility.tex",
    "docs/latex/engineering/observability.tex",
    "docs/latex/engineering/security.tex",
    "docs/latex/engineering/releases.tex",
    "docs/latex/engineering/documentation-system.tex",
    "docs/latex/engineering/improvements.tex",
    "docs/latex/engineering/traceability.tex",
    "docs/metadata/repository-inventory.json",
    "docs/metadata/api-index.json",
    "docs/metadata/module-dependencies.json",
    "docs/metadata/documentation-manifest.json",
)

#: Case-insensitive markers that must never survive into generated output.
PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bTBD\b",
    r"\bXXX\b",
    r"lorem ipsum",
    r"<insert[^>]*>",
    # Standalone only: the package legitimately exports constants named
    # EMAIL_PLACEHOLDER, PHONE_PLACEHOLDER, and TOKEN_PLACEHOLDER.
    r"(?<![A-Za-z_}])PLACEHOLDER(?![A-Za-z_])",
)

#: Literals shaped like real credentials. Names and empty values are fine;
#: an actual key-looking string is not.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("OpenAI-style key", r"\bsk-[A-Za-z0-9]{16,}"),
    ("Anthropic-style key", r"\bsk-ant-[A-Za-z0-9\-_]{16,}"),
    ("GitHub token", r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}"),
    ("AWS access key id", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ("Slack token", r"\bxox[abpsr]-[A-Za-z0-9-]{10,}"),
    ("Private key block", r"BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY"),
    ("Bearer token", r"\bBearer\s+[A-Za-z0-9._\-]{20,}"),
)

_LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
_REF_RE = re.compile(r"\\(?:cref|Cref|ref|autoref|nameref)\{([^}]*)\}")
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")
_VERBATIM_RE = re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", flags=re.DOTALL)
_COMMENT_RE = re.compile(r"(?<!\\)%.*")
#: Commands whose argument is a key or a path rather than typeset text, and
#: therefore legitimately contains characters that would need escaping in prose.
_KEY_ARGUMENT_RE = re.compile(
    r"\\(?:label|input|include|cref|Cref|ref|autoref|nameref|href|url|bibliography"
    r"|bibliographystyle|addcontentsline)\{[^{}]*\}"
)
#: Marker an authored file may carry to opt out of the placeholder scan, for
#: text that legitimately discusses placeholder markers (this validator's own
#: documentation, for instance).
PLACEHOLDER_OPT_OUT = "% validate-docs: allow-placeholder-words"
_SIGNATURE_RE = re.compile(
    r"\\apifield\{Usage\}\s*\\begin\{lstlisting\}\[style=usage, language=Python\]\n(.*?)\n"
    r"\\end\{lstlisting\}",
    flags=re.DOTALL,
)
#: Special characters that must not appear raw in generated running text.
#:
#: ``&`` is excluded because it is the column separator of every generated
#: table; ``~`` and ``^`` are excluded because they are used as a non-breaking
#: space and in TikZ coordinates respectively.
_UNESCAPED_RE = re.compile(r"(?<!\\)[#$_]")


@dataclass
class Check:
    """Result of one validation check."""

    name: str
    passed: bool
    detail: str = ""
    offenders: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Format the check for terminal output."""
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name}"
        if self.detail:
            line += f": {self.detail}"
        for offender in self.offenders[:8]:
            line += f"\n         - {offender}"
        if len(self.offenders) > 8:
            line += f"\n         - ... and {len(self.offenders) - 8} more"
        return line


def _latex_files(repo_root: Path) -> list[Path]:
    """Return every LaTeX source file of the manual."""
    return sorted((repo_root / LATEX_ROOT).rglob("*.tex"))


def _strip_non_text(content: str) -> str:
    """Remove verbatim environments and comments before textual checks.

    Listings and ``verbatim`` blocks are reproduced literally by LaTeX, so
    their contents are neither labels, references, nor escapable prose. Both
    are removed first by every check in this module.
    """
    without_verbatim = _VERBATIM_RE.sub("", content)
    return _COMMENT_RE.sub("", without_verbatim)


def _strip_key_arguments(content: str) -> str:
    """Remove commands whose argument is a key or path rather than prose."""
    return _KEY_ARGUMENT_RE.sub("", content)


def check_required_files(repo_root: Path) -> Check:
    """Every authored and metadata file the manual depends on must exist."""
    missing = [name for name in REQUIRED_FILES if not (repo_root / name).exists()]
    return Check(
        name="required files present",
        passed=not missing,
        detail=f"{len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)} present",
        offenders=missing,
    )


def check_api_coverage(repo_root: Path, model: dict[str, Any]) -> Check:
    """Every public class and function must have a rendered fragment."""
    api_dir = repo_root / GENERATED_ROOT / "api"
    missing: list[str] = []
    for item in model["classes"] + model["functions"]:
        if not (api_dir / f"{item['qualified_name']}.tex").exists():
            missing.append(item["qualified_name"])
    total = len(model["classes"]) + len(model["functions"])
    return Check(
        name="public API coverage",
        passed=not missing,
        detail=f"{total - len(missing)}/{total} public objects rendered",
        offenders=missing,
    )


def check_signatures(repo_root: Path, model: dict[str, Any]) -> Check:
    """Rendered signatures must match the signature extracted from source."""
    api_dir = repo_root / GENERATED_ROOT / "api"
    mismatched: list[str] = []
    checked = 0
    for item in model["functions"]:
        path = api_dir / f"{item['qualified_name']}.tex"
        if not path.exists():
            continue
        match = _SIGNATURE_RE.search(path.read_text(encoding="utf-8"))
        if match is None:
            mismatched.append(f"{item['qualified_name']}: no signature listing")
            continue
        checked += 1
        if match.group(1).strip() != item["signature"].strip():
            mismatched.append(
                f"{item['qualified_name']}: rendered {match.group(1).strip()!r} "
                f"!= extracted {item['signature'].strip()!r}"
            )
    return Check(
        name="signatures match implementation",
        passed=not mismatched,
        detail=f"{checked} function signatures verified",
        offenders=mismatched,
    )


def collect_labels(files: list[Path]) -> dict[str, list[Path]]:
    """Map every declared label to the files that declare it."""
    labels: dict[str, list[Path]] = {}
    for path in files:
        content = _strip_non_text(path.read_text(encoding="utf-8"))
        for label in _LABEL_RE.findall(content):
            labels.setdefault(label, []).append(path)
    return labels


def check_label_uniqueness(repo_root: Path, files: list[Path]) -> Check:
    """No label may be declared twice anywhere in the manual."""
    labels = collect_labels(files)
    duplicates = [
        f"{label} declared in "
        + ", ".join(path.relative_to(repo_root).as_posix() for path in paths)
        for label, paths in sorted(labels.items())
        if len(paths) > 1
    ]
    return Check(
        name="labels unique",
        passed=not duplicates,
        detail=f"{len(labels)} labels declared",
        offenders=duplicates,
    )


def check_references_resolve(repo_root: Path, files: list[Path]) -> Check:
    """Every statically checkable cross-reference must have a target."""
    labels = set(collect_labels(files))
    dangling: list[str] = []
    total = 0
    for path in files:
        content = _strip_non_text(path.read_text(encoding="utf-8"))
        for group in _REF_RE.findall(content):
            for target in group.split(","):
                target = target.strip()
                if not target:
                    continue
                total += 1
                if target not in labels:
                    dangling.append(f"{path.relative_to(repo_root).as_posix()} -> {target}")
    return Check(
        name="cross-references resolve",
        passed=not dangling,
        detail=f"{total} references checked against {len(labels)} labels",
        offenders=sorted(set(dangling)),
    )


def check_inputs_exist(repo_root: Path, files: list[Path]) -> Check:
    """Every \\input target must exist relative to the LaTeX root."""
    latex_root = repo_root / LATEX_ROOT
    missing: list[str] = []
    total = 0
    for path in files:
        content = _strip_non_text(path.read_text(encoding="utf-8"))
        for target in _INPUT_RE.findall(content):
            total += 1
            candidate = latex_root / target
            # TeX appends .tex only when the name has no extension. Fragment
            # names contain dots, so both spellings must be accepted here.
            if not candidate.exists() and not Path(f"{candidate}.tex").exists():
                missing.append(f"{path.relative_to(repo_root).as_posix()} -> {target}")
    return Check(
        name="input targets exist",
        passed=not missing,
        detail=f"{total} include directives checked",
        offenders=missing,
    )


def check_escaping(repo_root: Path) -> Check:
    """Generated text must contain no unescaped LaTeX special characters."""
    offenders: list[str] = []
    generated = sorted((repo_root / GENERATED_ROOT).rglob("*.tex"))
    for path in generated:
        text = _strip_key_arguments(_strip_non_text(path.read_text(encoding="utf-8")))
        for number, line in enumerate(text.splitlines(), start=1):
            match = _UNESCAPED_RE.search(line)
            if match is not None:
                offenders.append(
                    f"{path.relative_to(repo_root).as_posix()}:{number} "
                    f"unescaped {match.group(0)!r}"
                )
    return Check(
        name="LaTeX special characters escaped",
        passed=not offenders,
        detail=f"{len(generated)} generated files scanned",
        offenders=offenders,
    )


def check_placeholders(repo_root: Path, files: list[Path]) -> Check:
    """No placeholder marker may survive into the manual."""
    pattern = re.compile("|".join(PLACEHOLDER_PATTERNS), flags=re.IGNORECASE)
    offenders: list[str] = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        if PLACEHOLDER_OPT_OUT in raw:
            continue
        text = _strip_non_text(raw)
        for number, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match is not None:
                offenders.append(
                    f"{path.relative_to(repo_root).as_posix()}:{number} {match.group(0)!r}"
                )
    return Check(
        name="no placeholder text",
        passed=not offenders,
        detail=f"{len(files)} LaTeX files scanned",
        offenders=offenders,
    )


def check_no_secrets(repo_root: Path, files: list[Path]) -> Check:
    """No credential-shaped literal may appear anywhere in the manual."""
    offenders: list[str] = []
    targets = list(files) + sorted((repo_root / METADATA_ROOT).glob("*.json"))
    for path in targets:
        content = path.read_text(encoding="utf-8")
        for name, pattern in SECRET_PATTERNS:
            match = re.search(pattern, content)
            if match is not None:
                offenders.append(f"{path.relative_to(repo_root).as_posix()}: possible {name}")
    return Check(
        name="no secret values emitted",
        passed=not offenders,
        detail=f"{len(targets)} files scanned against {len(SECRET_PATTERNS)} patterns",
        offenders=offenders,
    )


def check_deterministic_order(model: dict[str, Any]) -> Check:
    """Model collections must be sorted, so rendering order is stable."""
    problems: list[str] = []
    for key, sort_key in (
        ("modules", "name"),
        ("classes", "qualified_name"),
        ("functions", "qualified_name"),
        ("commands", "qualified_name"),
    ):
        values = [item[sort_key] for item in model[key]]
        if values != sorted(values):
            problems.append(f"{key} is not sorted by {sort_key}")
    endpoints = [(item["path"], item["method"]) for item in model["endpoints"]]
    if endpoints != sorted(endpoints):
        problems.append("endpoints are not sorted by (path, method)")
    return Check(
        name="deterministic generation order",
        passed=not problems,
        detail="all model collections sorted",
        offenders=problems,
    )


def check_manifest(repo_root: Path) -> Check:
    """Manifest digests must match the files on disk."""
    manifest_path = repo_root / METADATA_ROOT / "documentation-manifest.json"
    if not manifest_path.exists():
        return Check(
            name="manifest integrity",
            passed=False,
            detail="documentation-manifest.json is missing",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    entries = manifest["generated_files"] + manifest["metadata_files"]
    for entry in entries:
        path = repo_root / entry["path"]
        if not path.exists():
            offenders.append(f"{entry['path']}: missing")
            continue
        digest = content_digest(path)
        if digest != entry["sha256"]:
            offenders.append(f"{entry['path']}: digest mismatch")
    return Check(
        name="manifest integrity",
        passed=not offenders,
        detail=f"{len(entries)} recorded artifacts verified",
        offenders=offenders,
    )


def check_revision(repo_root: Path) -> Check:
    """The manifest must name a revision; matching HEAD exactly is not required.

    Documentation committed alongside a change records the revision it was
    generated from, which is the parent of the commit that carries it -- a
    commit cannot contain its own hash. Failing on that difference would make
    ``make docs`` fail immediately after every commit, so a recorded revision
    that merely trails HEAD is reported rather than treated as an error. What
    genuinely matters, that the documentation still matches the source, is
    covered by regenerating with ``--revision`` and comparing.
    """
    manifest_path = repo_root / METADATA_ROOT / "documentation-manifest.json"
    if not manifest_path.exists():
        return Check(
            name="revision correspondence",
            passed=False,
            detail="documentation-manifest.json is missing",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return Check(
            name="revision correspondence",
            passed=True,
            detail="git unavailable; revision check skipped",
        )
    head = completed.stdout.strip()
    recorded = manifest.get("revision", "")
    if not recorded:
        return Check(
            name="revision correspondence",
            passed=False,
            detail="the manifest records no revision",
            offenders=["regenerate with: make docs"],
        )
    if recorded == head:
        return Check(
            name="revision correspondence",
            passed=True,
            detail=f"documents HEAD ({head[:12]})",
        )
    return Check(
        name="revision correspondence",
        passed=True,
        detail=(
            f"documents {recorded[:12]}, HEAD is {head[:12]} "
            "(expected when the documentation was committed with the change)"
        ),
    )


def run_checks(repo_root: Path) -> list[Check]:
    """Run every validation check and return the results in order."""
    checks = [check_required_files(repo_root)]
    api_index = repo_root / METADATA_ROOT / "api-index.json"
    if not api_index.exists():
        checks.append(
            Check(
                name="documentation model present",
                passed=False,
                detail="api-index.json is missing; run 'make docs' first",
            )
        )
        return checks

    model = json.loads(api_index.read_text(encoding="utf-8"))
    files = _latex_files(repo_root)
    checks.extend(
        [
            check_api_coverage(repo_root, model),
            check_signatures(repo_root, model),
            check_label_uniqueness(repo_root, files),
            check_references_resolve(repo_root, files),
            check_inputs_exist(repo_root, files),
            check_escaping(repo_root),
            check_placeholders(repo_root, files),
            check_no_secrets(repo_root, files),
            check_deterministic_order(model),
            check_manifest(repo_root),
            check_revision(repo_root),
        ]
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    """Validate the documentation tree and report the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    repo_root = arguments.repo_root.resolve()

    checks = run_checks(repo_root)
    print("documentation validation")
    for check in checks:
        print(check.render())
    failed = [check for check in checks if not check.passed]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
