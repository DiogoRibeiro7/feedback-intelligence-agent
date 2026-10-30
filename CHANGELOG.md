# Changelog

All notable changes to this project are documented here.

## Unreleased

## 0.7.0 - 2026-10-30

See [upgrade notes](docs/releases/v0.7.0.md) for tenant access policies, record
ingestion, shared-index refresh, and coordinated Qdrant configuration.

### Added

- Optional Qdrant snapshot caching reuses unchanged retrieval and tool snapshots
  until a cooperating write or configured expiry. The default zero TTL preserves
  refresh on every request; positive lifetimes require coordinated Qdrant mode.
- Shared generation markers invalidate other workers before remote writes, including
  partial failures. External writers can use `qdrant_write_lock()`; legacy writers
  become visible at cache expiry. Failed refreshes never extend stale cache lifetimes.
- Qdrant API record ingestion can be enabled with an explicit shared lock path.
  Tenant authorization, bounded batches, deduplication, PII redaction, and counts
  match JSON ingestion; preparation precedes writes and remote failures are retryable.
- Coordinated Qdrant workers select immutable local snapshots for retrieval and tools
  on every query, stream, or chat request, preserving provider state and in-flight
  answers. This requires cooperating local locks and memory for the full collection;
  remote failures can leave partial updates, and independent hosts are not coordinated.
- Qdrant source replacement removes obsolete chunks while preserving other tenants,
  validates identities and vectors before writes, and supports retrying interrupted
  replacements. Callers must serialize writers to the same source; updates are not atomic.
- Qdrant integration CI uses a real server on Python 3.10 and 3.12 and cleans up
  temporary test collections. Qdrant writes now wait for completion and reject
  nonfinite vectors before sending requests.
- API workers automatically reload changed JSON indexes before query, streaming,
  and chat requests, including updates from other workers, CLI tools, and ingestion
  jobs. Retrieval and dataset tools refresh together while preserving provider state.
- Missing or invalid committed indexes return a generic `503` on new answer requests
  and recover on the next request after repair, without rebuilding from source CSV.
- Tenant-bound writers and admins can ingest validated record batches through
  `POST /ingestion/records`. JSON upserts preserve other tenants and refresh the
  handling API instance's retriever and aggregation tools after persistence.
- Batch ingestion rejects foreign tenants, path overrides, and invalid batches;
  responses expose batch-local counts without global index sizes or server paths.

- Optional API-key tenant bindings scope queries, streaming, conversations,
  reports, exports, email summaries, feedback, and active-learning workflows.
  Foreign resource IDs return 404, and explicit tenant overrides return 403.
- Tenant-bound keys cannot rebuild shared indexes, submit ingestion jobs, or read
  global job artifacts. Unbound administrators retain these operations.
- Invalid nonempty tenant policies fail closed. The default empty policy retains
  existing API role behavior and local CLI workflows.

## 0.6.0 - 2026-10-29

See [upgrade notes](docs/releases/v0.6.0.md) for index, conversation, extension,
and MLflow compatibility changes.

### Added

- Release metadata validation checks Python, frontend, citation, and Zenodo
  versions, release dates, the changelog entry, and release tags in CI.

### Changed

- The `mlflow` extra installs the tracking client (`mlflow-skinny`) with SQLite
  support, removing unused server, model-training, and image dependencies while
  retaining Python 3.10 support. Host the MLflow server/UI in a separate environment.
- MLflow 3.17 requires `MLFLOW_ALLOW_FILE_STORE=true` for existing `file:` tracking
  URIs; the documented local setup now uses SQLite.

### Fixed

- Normalize dataset sizes in generated documentation so Windows and Linux
  checkouts produce the same release manual.
- Feedback ingestion and vector persistence use DataExcept context, and API
  file access is confined to the configured data and artifact directories.
- Refresh vulnerable Python dependencies, including AnyIO, Certifi, Cryptography,
  GitPython, SQLParse, Mako, MLflow's tracking client, OpenTelemetry, and urllib3.
- Exercise real MLflow parameter, metric, tag, and artifact round trips against
  local file and SQLite stores in CI on Python 3.10 and 3.12.
- Sentiment and issue-cluster tools apply request metadata filters before
  aggregation, keeping counts, distributions, document IDs, and example quotes
  within the selected tenant and cohort, including inherited chat tenant scope.
- Ticket evidence applies the same filters before deduplication and limiting.
- Authenticated conversation reads and follow-ups use a private namespace per
  API key, returning 404 for unknown or another caller's conversation IDs.
- API-key conversation identities use PBKDF2-HMAC-SHA256 with 600,000 iterations
  instead of a fast credential digest; cached identities still require current
  key authorization on every request.
- Conversation follow-ups inherit their saved tenant filter and reject changes
  to that tenant before rewriting or answer generation.
- Use `portalocker` for JSON writer locks to retain Python 3.10 support without
  the vulnerable `filelock` dependency flagged by dependency review.
- JSON index saves flush a sibling temporary file before atomic replacement,
  keeping the last committed snapshot intact when writes fail.
- Incremental JSON updates lock the full load/merge/save transaction across
  threads and processes; initial index creation rechecks after acquiring the
  same lock to avoid rebuilding over a concurrent update.
- Metadata filters apply before candidate limits in dense, lexical, and hybrid
  retrieval; Qdrant pages until enough eligible results are found or exhausted.
- Shared metadata matching preserves legacy defaults and normalizes timestamp
  objects and ISO strings, including trailing `Z`, to inclusive UTC date bounds.
- Date filter range validation accepts mixed naive and timezone-aware bounds
  using UTC, and reports reversed ranges as validation errors.
- Source and chunk IDs escape tenant/feedback components to prevent delimiter
  collisions, with in-memory legacy JSON migration and a persisted version marker.
- Qdrant refuses legacy collections requiring identifier migration before writes,
  with instructions to rebuild in a new collection from the original dataset.
- Incremental indexing preserves feedback IDs shared by different tenants and
  correctly counts updates to non-default-tenant records in results and telemetry.
- Invalid vector replacements leave existing chunks and search results intact.
- Generated documentation retains LF line endings on Windows checkouts; symlink
  tests explicitly skip when Windows denies the required creation privilege.
- Frontend dependency audit findings resolved by refreshing the lockfile within
  existing version ranges, including Vite 8.3.5 and patched transitive dependencies.

## 0.5.0 - 2026-08-08

### Added

- LaTeX technical documentation system under `docs/`, generated from the
  repository by an AST-based toolchain that performs no application imports and
  no network access.
- Two compiled documents sharing one source tree: `docs/latex/reference.tex`,
  a CRAN-style reference covering every public object, and
  `docs/latex/main.tex`, which adds the software architecture and engineering
  parts and the traceability appendix.
- Machine-readable documentation model in `docs/metadata/`: repository
  inventory, API index, module dependency graph with cycle analysis, and a
  manifest recording a SHA-256 digest of every generated artifact.
- Documentation validator covering API coverage, signature agreement, label
  uniqueness, cross-reference resolution, LaTeX escaping, placeholder text,
  and credential-shaped literals.
- Make targets `docs`, `docs-check`, `docs-pdf`, `docs-pdf-reference`,
  `docs-clean`, and `typecheck-docs`.
- Documentation job in CI that regenerates as of the documented revision and
  fails when the committed manual has drifted from the source.
- `tests/test_docs_tools.py`, covering LaTeX escaping and rendering, docstring
  and signature extraction, CLI and endpoint extraction, dependency and cycle
  analysis, and LaTeX log validation.

### Changed

- Maintainer contact in package metadata now uses the institutional address
  recorded in `CITATION.cff`.
- Ruff's `src` setting includes `docs`, so the documentation toolchain is
  treated as first-party for import sorting.

## 0.4.0 - 2026-08-06

### Added

- Human answer feedback analytics with useful/not-useful rates, comment counts,
  and route/tenant breakdowns through the API and CLI.
- Markdown exports for saved insight reports through the API and CLI.
- Active-learning queue ranking for not-useful and low-confidence reviewed
  answers through the API and CLI.
- Assignment and workflow-state transitions for active-learning queue items
  with a JSON-backed local state store.
- Optional role-based API-key authorization for reader, writer, and admin routes.
- Optional in-process fixed-window API rate limiting for deployed services.

## 0.3.0 - 2026-08-06

### Added

- AWS Bedrock Runtime Converse API provider via `FEEDBACK_AGENT_LLM_PROVIDER=bedrock`.
- Retry, timeout, and circuit-breaker wrapper for remote LLM providers.
- Structured LLM output validation with deterministic repair diagnostics.
- Optional MLflow experiment tracking for experiment runs.
- Optional OpenTelemetry span backend for ingestion, embedding, retrieval, generation,
  and response parsing traces.
- Hallucination diagnostics using deterministic evidence overlap plus optional
  LLM-as-judge support.
- Frontend session dashboard for latency, retrieval-score distribution, and
  citation coverage.
- Streaming feedback ingestion primitives with local JSONL, Kafka, and Kinesis
  adapters plus dead-letter handling.
- Incremental JSON vector index updates for CSV batches and accepted stream events.
- Deterministic PII and credential redaction before chunking, indexing, and
  stream artifact persistence.
- Local lakehouse-style feedback export with partitioned JSONL data files plus
  Delta-style or Iceberg-style metadata manifests.
- Saved insight reports with JSON persistence, API endpoints, CLI commands, and
  frontend save/list controls.
- Email digests for saved insight reports with render-only dry runs and optional
  SMTP delivery.
- Human answer feedback capture with JSON persistence, API endpoints, CLI
  commands, and frontend useful/not useful controls.
- Multi-tenant feedback scoping through optional `tenant_id` ingestion fields,
  retrieval filters, tenant-aware reports, answer feedback, and lakehouse exports.

## 0.2.0 - 2026-08-05

### Added

- Metadata filters for retrieval by segment, channel, rating, and creation date.
- Deterministic product terminology query expansion for retrieval.
- Deterministic judge reranking for cited evidence selection.
- Retrieval regression tests over known sample questions.
- OpenAI Responses API provider via `FEEDBACK_AGENT_LLM_PROVIDER=openai_responses`.
- Repository health files and release validation workflows.

### Changed

- Documented retrieval regression coverage and the OpenAI Responses provider.
- Updated roadmap status for completed retrieval and OpenAI provider milestones.
- Kept `openai` mapped to Chat Completions for OpenAI-compatible gateway support.

## 0.1.1 - 2026-08-05

### Added

- Zenodo and citation metadata for archival releases.

## 0.1.0 - 2026-08-05

### Added

- Initial Feedback Intelligence Agent release.
