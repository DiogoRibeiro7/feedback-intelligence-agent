# Changelog

All notable changes to this project are documented here.

## Unreleased

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
