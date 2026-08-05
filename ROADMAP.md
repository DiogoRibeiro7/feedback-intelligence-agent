# Roadmap

This roadmap is designed to evolve the repository from a compact portfolio project into a stronger AI engineering reference system.

## Phase 1 — Core AI system

- [x] Validate feedback data with typed schemas.
- [x] Implement deterministic embeddings for local development.
- [x] Implement vector search and persistence.
- [x] Build an evidence-grounded RAG agent.
- [x] Expose a FastAPI query endpoint.
- [x] Add offline retrieval and answer-quality evaluation.

## Phase 2 — Better retrieval

- [x] Add hybrid retrieval with lexical BM25 plus vector search.
- [x] Add metadata filters by segment, rating, channel, and date.
- [x] Add query expansion for product-specific terminology.
- [x] Add reranking with a cross-encoder or LLM judge.
- [x] Add retrieval regression tests for known questions.

## Phase 3 — Production LLM integration

- [x] Add OpenAI Responses API provider.
- [x] Add AWS Bedrock provider.
- [x] Add retry, timeout, and circuit-breaker policies.
- [x] Add prompt versioning with a validated registry, CLI inspection, and snapshot tests.
- [x] Add structured JSON output validation with automatic repair.

## Phase 4 — Evaluation and observability

- [x] Add a local experiment runner for comparing retrieval and generation configurations.
- [x] Add MLflow or Weights & Biases experiment tracking.
- [x] Add OpenTelemetry traces for retrieval and generation.
- [x] Add deterministic guardrails with prompt-injection test cases.
- [x] Add hallucination checks using evidence overlap and LLM-as-judge.
- [x] Add dashboards for latency, retrieval score distribution, and citation coverage.

## Phase 5 — Data engineering

- [x] Add streaming ingestion with Kafka or Kinesis.
- [x] Add incremental index updates.
- [x] Add data contracts for feedback producers.
- [ ] Add PII redaction before storage.
- [ ] Add lakehouse export to Iceberg or Delta tables.

## Phase 6 — Product layer

- [x] Add a small UI for analysts and product managers.
- [ ] Add saved insight reports.
- [ ] Add Slack or email summaries.
- [ ] Add human feedback capture on generated answers.
- [ ] Add multi-tenant support.
