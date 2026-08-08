# Documentation

This directory holds two kinds of documentation.

- **Markdown notes** (`architecture.md`, `case_study.md`, `deployment.md`,
  `evaluation.md`, `prompts.md`) — short, hand-written topic notes.
- **The LaTeX technical reference manual** (`latex/`) — a complete, generated
  reference for the repository, plus the toolchain that produces it.

The canonical deliverable of the LaTeX system is the `.tex` source. The PDF is
a downstream build artifact.

## Quick start

```bash
make docs                # inspect, extract, render, validate
make docs-check          # validate only, no regeneration
make docs-pdf            # the above, then build the complete manual (~263 pp.)
make docs-pdf-reference  # build the CRAN-style reference only (~179 pp.)
make docs-clean          # remove LaTeX auxiliary files
```

## Two documents, one source

There are two master documents. They share the preamble, the metadata, and
every chapter file, so they cannot describe different revisions and nothing is
duplicated between them.

| Document | Contents | Pages |
| --- | --- | --- |
| `latex/reference.tex` | The documented objects only — modules, classes, functions, schemas, CLI, HTTP API, configuration, mathematics. This is the CRAN-style reference manual. | ~179 |
| `latex/main.tex` | The same reference, plus the software architecture and engineering parts and the traceability appendix. | ~263 |

Entries use CRAN's layout: the object name as the heading, the summary
immediately below it, then run-in bold field labels (`Usage`, `Arguments`,
`Value`, `Raises`, `Details`, `Source`). Arguments are hanging-indent lists
rather than full-width tables — that one choice, plus a 10pt body and a
two-column contents, is most of the difference between a 573-page draft and
the current sizes.

`make docs` performs **no application imports**: extraction is AST-based, so
generating documentation cannot open a network connection, contact an LLM
provider, build a vector index, or write outside `docs/`. The only subprocess
it runs is `git`, for revision facts.

### Requirement

The toolchain needs **Python 3.11 or newer** because it reads `pyproject.toml`
with the standard-library `tomllib` module. The `feedback_intelligence_agent`
package itself still supports 3.10 — only documentation generation needs the
newer interpreter, and it says so clearly if run on an older one.

A LaTeX distribution providing `latexmk` is needed only for `make docs-pdf`.

## Layout

```text
docs/
├── README.md                  this file
├── architecture.md            hand-written topic notes
├── case_study.md
├── deployment.md
├── evaluation.md
├── prompts.md
│
├── latex/                     the technical manual
│   ├── main.tex               master document, all three parts (authored)
│   ├── reference.tex          master document, reference only (authored)
│   ├── preamble.tex           shared preamble (authored)
│   ├── metadata.tex           title-page macros (GENERATED)
│   ├── about.tex              how to read the manual (authored)
│   ├── references.bib         bibliography (authored)
│   ├── reference/             Part I  - what the repository exposes (authored)
│   ├── architecture/          Part II - how the system is structured (authored)
│   │   └── adr/               architecture decision records (authored)
│   ├── engineering/           Part III - how it is built and shipped (authored)
│   ├── diagrams/              authored TikZ figures
│   └── generated/             every derived fragment (GENERATED)
│
├── metadata/                  the normalized documentation model (GENERATED)
│   ├── repository-inventory.json
│   ├── api-index.json
│   ├── module-dependencies.json
│   └── documentation-manifest.json
│
└── tools/                     the documentation system itself
    ├── inspect_repository.py  files, deps, CI, tests, datasets, gates
    ├── extract_api.py         AST extraction of the public API
    ├── extract_dependencies.py internal import graph and cycles
    ├── latex_utils.py         escaping and table primitives (pure)
    ├── render_reference.py    model -> reference fragments
    ├── render_architecture.py model -> architecture fragments
    ├── render_engineering.py  model -> engineering fragments
    ├── generate_docs.py       orchestration and manifest
    ├── validate_docs.py       the documentation quality gate
    └── check_latex_log.py     decides whether a PDF build was clean
```

## Authored vs generated

**Never edit anything under `latex/generated/` or `metadata/`.** Every file
there carries a generated-file banner and is deleted and rewritten by
`make docs`.

Authored files — the chapter prose, the mathematical exposition, the ADRs, and
the TikZ diagrams under `latex/diagrams/` — are never touched by the generator.

The split is deliberate: signatures, parameter tables, field tables, CLI
options, endpoint tables, environment variables, dependency lists, the module
graph, the file census, and all indexes are derived from the repository, so
nobody has to maintain the same fact in two places.

## Architecture of the toolchain

```text
repository source
      |
      +--> inspect_repository.py ---+
      +--> extract_api.py -----------+--> docs/metadata/*.json   (the model)
      +--> extract_dependencies.py --+
                                     |
      +------------------------------+
      |
      +--> render_reference.py ------+
      +--> render_architecture.py ---+--> docs/latex/generated/**
      +--> render_engineering.py ----+
                                     |
                              docs/latex/main.tex
                                     |
                              PDF (built separately)
```

Extraction and rendering never mix: a renderer never parses Python, and an
extractor never emits markup. An incorrect signature is therefore an extractor
bug and a broken table is a renderer bug, and both layers are tested
separately in `tests/test_docs_tools.py`.

## Validation

`make docs` runs `validate_docs.py` automatically. It checks that:

1. every public class and function has a rendered fragment;
2. every rendered signature matches the extracted signature;
3. every LaTeX label is unique;
4. every `\cref`/`\ref` target is defined;
5. every `\input` target exists;
6. generated files contain no unescaped LaTeX special characters;
7. every required authored file exists;
8. no placeholder text survives;
9. no credential-shaped literal appears anywhere;
10. the manifest digests match the files on disk;
11. the manifest revision matches the current `HEAD`.

CI runs the same command and then fails if regeneration changed any tracked
file, which enforces the invariant that the committed manual describes the
committed source.

## Determinism

Two runs over the same revision produce byte-identical output: objects are
emitted in fully-qualified-name order, table rows come from the model rather
than dictionary iteration, and timestamps are taken from the HEAD commit date
rather than the wall clock. That is what makes the SHA-256 digests in
`metadata/documentation-manifest.json` meaningful as a drift detector.

## Secrets

No secret value appears in the manual. The configuration reference documents
the *names*, types, defaults, and sensitivity of credential settings; the
declared default of every credential field in source is `None`, and the
generator reads source text rather than the process environment. The validator
additionally scans every generated file for credential-shaped literals.
