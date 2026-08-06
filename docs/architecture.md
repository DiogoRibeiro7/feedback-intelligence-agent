# Architecture

## Goal

The system converts unstructured customer feedback into evidence-grounded product insights.

It is intentionally small, but it uses boundaries that mirror a real production AI engineering system.

## Flow

```text
CSV feedback / feedback streams
   │
   ▼
Validation and ingestion
   │
   ├───────────────► Lakehouse export
   │
   ▼
PII redaction
   │
   ▼
Chunking
   │
   ▼
Hashing embeddings
   │
   ▼
Local vector store
   │
   ▼
Query engine
   │
   ▼
Route selection
   │
   ▼
Tool router (keyword intent matching)
   │            │
   │            ▼
   │      Local tool run (sentiment / clusters / ticket draft)
   │            │
   ▼            ▼
Grounded prompt + tool output
   │
   ▼
LLM provider
   │
   ▼
Cited answer + tool metadata + recommended actions + diagnostics
   │
   ├───────────────► Saved insight reports
   │                       │
   │                       ▼
   │                  Email summaries
   │
   └───────────────► Human answer feedback
```

## Components

### Ingestion

`ingestion.py` loads CSV data and validates each row with Pydantic. `tenant_id` is optional and defaults to `default`; duplicate `feedback_id`s are checked within each tenant. `streaming_ingestion.py` validates bounded JSONL, Kafka, or Kinesis feedback batches against the same schema, checkpoints accepted offsets, and can write invalid messages to a dead-letter JSONL file. `index_updates.py` merges validated records into the persisted JSON vector index by replacing chunks for matching tenant-scoped source IDs. `lakehouse.py` exports validated, redacted records into partitioned local data files with Delta-style or Iceberg-style metadata. Invalid rows or messages are reported with clear locations, which makes data quality issues easier to debug.

### Chunking

`chunking.py` splits feedback into overlapping word chunks and carries tenant metadata into every chunk. Default-tenant source IDs preserve the original `feedback_id`; non-default tenants use tenant-qualified source IDs to avoid collisions. Before chunking, `privacy.py` redacts emails, phone numbers, and obvious access tokens so persisted chunks do not carry raw PII or pasted credentials. The current dataset has short feedback, but the logic also works for longer support tickets or interview transcripts.

### Privacy

`privacy.py` provides deterministic regex redaction for emails, phone numbers, and obvious access tokens. It is applied before chunking/indexing and when writing accepted stream CSVs or dead-letter payloads.

### Lakehouse export

`lakehouse.py` writes dependency-light local tables for downstream analytics. It stores partitioned JSONL data files plus a top-level manifest and either Delta-style `_delta_log` actions or Iceberg-style `metadata/v1.metadata.json` metadata. Records are redacted before export, and partition columns are validated against known fields such as `tenant_id`, `created_date`, `created_month`, `channel`, `customer_segment`, and `rating`.

### Embeddings

`embeddings.py` implements deterministic feature hashing with unigrams and bigrams. This is useful for local development because it avoids external APIs. In production, this component can be replaced by OpenAI embeddings, Bedrock embeddings, or an internal embedding service.

### Vector store

`vector_store.py` provides cosine search, JSON persistence, and source-level upsert support for incremental updates. It is deliberately simple. Production alternatives include pgvector, OpenSearch, Pinecone, Weaviate, Qdrant, or FAISS.

### Agent

`agent.py` performs query routing, retrieval, tool routing and execution, prompt building, generation, structured output parsing and repair, citation construction, hallucination checks, and confidence scoring.

The full agent flow per question is:

1. **Input guardrail gate** (`guardrails.check_input`): unsafe questions are refused before any retrieval or tool use.
2. **Route selection**: a keyword router classifies the question into a stable route for observability and prompts.
3. **Retrieval + reranking**: the configured retriever gathers candidate chunks; `reranking.py` applies a deterministic local judge using lexical, route, segment, and low-rating signals.
4. **Context guardrail gate** (`guardrails.check_context`): retrieved chunks carrying injection-style content are dropped.
5. **Tool routing** (`tools.ToolRouter`): a deterministic keyword router selects at most one local tool. Explicit requests for unknown tools (`use the <name> tool`) are refused gracefully and the run continues as plain RAG.
6. **Tool execution**: the selected tool validates its Pydantic input schema and runs locally, wrapped in `tool_run_started`/`tool_run_finished` telemetry spans. Tool failures degrade to an `error` record instead of failing the run.
7. **Answer generation**: the LLM provider produces the cited answer; a successful tool run appends a `Tool insight (...)` line to the answer text.
8. **Hallucination checks**: `hallucination.py` scores sentence-level evidence overlap and can optionally use the configured LLM as a semantic judge.
9. **Response assembly**: the answer carries citations, the guardrail decision, the `tool_run` record (name, status, summary, structured output), parser diagnostics, hallucination diagnostics, and retrieval diagnostics. The same metadata is returned by the API `/query` response.

### Tools

`tools.py` implements a small deterministic tool-use framework. Every tool conforms to a typed interface: a stable `name`, a `description`, a Pydantic `input_schema`, a Pydantic `output_schema`, and a validated `execute` entry point. Three local tools ship by default:

- **`sentiment_summary`** (`SentimentSummaryTool`): aggregates the rating and sentiment distribution (positive ≥ 4, neutral = 3, negative ≤ 2) over the indexed feedback, with optional segment and channel filters.
- **`issue_cluster`** (`IssueClusterTool`): groups recurring customer issues using deterministic keyword/term clusters (onboarding, integrations, reporting, pricing, support, performance, documentation) and reports counts, supporting documents, and example quotes.
- **`ticket_draft`** (`TicketDraftTool`): drafts a support ticket (title, body, priority, references, tags) from the question and the retrieved evidence chunks.

Routing is keyword/intent based (`TOOL_ROUTES`) with no function-calling API, so tool selection is reproducible in tests and CI. Tools run against the same indexed chunks used for retrieval, so they need no extra data source. Example tool queries live in `examples/tool_queries.jsonl`.

### Guardrails

`guardrails.py` provides a deterministic safety layer with two gates: an input check before retrieval (empty queries, prompt injection, system-prompt disclosure, context-override and unsupported data access requests) and a context check before generation that drops retrieved chunks carrying instruction-override content. Decisions are regex-based, typed (`GuardrailDecision`), and attached to every agent answer and API response.

### LLM provider

`llm.py` defines an `LLMProvider` protocol with per-provider capability metadata (streaming, tool calling, JSON mode, context size). The default provider is deterministic and local, so no API key is ever required. Optional providers — OpenAI Chat Completions-compatible endpoints, OpenAI's Responses API, Anthropic via the official SDK (optional `anthropic` extra), AWS Bedrock Runtime via the optional `bedrock` extra, and a local Ollama server — connect to external inference while keeping the rest of the system unchanged. `factory.build_llm` selects the provider from configuration and fails fast with actionable errors on missing keys or unknown provider names. Remote providers are wrapped by `resilience.py`, which applies per-attempt timeouts, bounded retries, and an in-memory circuit breaker while leaving the deterministic local provider unwrapped.

### Evaluation

`evaluation.py` measures retrieval quality (precision@k, recall@k, MRR, context hit rate) and answer quality (keyword coverage, groundedness, citation alignment, evidence overlap, hallucination rate, judge support, refusal correctness) and aggregates them into a typed `EvaluationReport`. This is important because AI engineering should not stop at prompt writing: retrieval quality, grounding, and abstention behaviour need continuous measurement. See [evaluation.md](evaluation.md) for details.

### API

`api.py` exposes the system through FastAPI. The API validates input and returns typed responses.

### Saved reports

`reports.py` persists generated `AgentAnswer` payloads as saved insight reports with a tenant ID, title, tags, notes, citations, diagnostics, and creation time. The default `JsonInsightReportStore` writes one report file per ID under `.artifacts/reports`, and the API exposes create, tenant-filtered list, and fetch endpoints for the frontend.

### Human feedback

`human_feedback.py` persists reviewer judgements on generated `AgentAnswer` payloads. Records capture the tenant ID, question, full answer package, useful/not useful rating, optional comment, linked report ID, tags, and creation time. The default `JsonHumanFeedbackStore` writes one record file per ID under `.artifacts/human_feedback`, and the API and CLI expose create, tenant-filtered list, and fetch workflows.

### Email summaries

`email_summaries.py` renders saved reports into concise plain-text email digests. The default path is render-only for local previews and CI; optional SMTP delivery is enabled by `FEEDBACK_AGENT_EMAIL_*` settings.

## Extension points

- Replace the hashing embedding model with a neural embedding provider.
- Replace the local vector store with a managed vector database.
- Add durable stream checkpoints across process restarts.
- Replace local JSONL lakehouse files with Parquet once a production table runtime is selected.
- Export telemetry spans through additional OpenTelemetry collectors or vendors.
- Add feedback analytics and active-learning queues from captured reviewer judgements.
- Add report sharing and export formats for stakeholder handoff.
- Add regression tests for prompts and retrieval behavior.
- Replace the deterministic reranker with a cross-encoder or external LLM judge.
- Add role-based access control around the API.

## Production considerations

A production version should include:

- Tenant isolation.
- Broader privacy policy coverage for names, addresses, and tenant-specific identifiers.
- Rate limiting and authentication.
- Prompt injection checks (a deterministic baseline ships in `guardrails.py`).
- Data lineage for every generated answer.
- Human feedback loops with reviewer attribution and lifecycle states.
- Monitoring for retrieval drift and answer degradation.
- Canary evaluation before prompt or model changes.
