# AI Engineering Showcase

[![CI](https://github.com/DiogoRibeiro7/ai-engineering-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/ai-engineering-showcase/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-style AI engineering repository that demonstrates how to build, evaluate, and serve an LLM-powered insight system.

The project implements a **customer feedback intelligence agent**. It ingests raw feedback, builds a lightweight vector index, retrieves relevant evidence, generates grounded answers, exposes a FastAPI service, and includes evaluation tests for retrieval and answer quality.

It is designed as a portfolio project: small enough to read, but structured like real production work.

## What this showcases

- Agentic RAG workflow with retrieval, routing, evidence selection, and cited responses.
- Clean LLM provider abstraction with a deterministic local fallback.
- Embedding and vector search implemented without a managed vector database.
- FastAPI inference service with typed request and response schemas.
- Offline evaluation for retrieval quality and answer grounding.
- Reproducible development setup with Poetry, Docker, tests, linting, and CI.
- Clear architecture boundaries that can be extended to OpenAI, Bedrock, LangGraph, Kafka, or a real vector database.

## Repository structure

```text
ai-engineering-showcase/
├── src/ai_engineering_showcase/
│   ├── agent.py              # RAG agent orchestration
│   ├── api.py                # FastAPI app
│   ├── chunking.py           # Text chunking utilities
│   ├── cli.py                # Typer CLI
│   ├── config.py             # Runtime configuration
│   ├── data_contracts.py     # Dataset validation and data contracts
│   ├── embeddings.py         # Hashing embedding model
│   ├── evaluation.py         # Retrieval and answer-quality metrics
│   ├── experiments.py        # Repeatable experiment runner
│   ├── ingestion.py          # CSV feedback loader
│   ├── lexical_search.py     # BM25 lexical retriever
│   ├── llm.py                # LLM abstraction and local fallback
│   ├── prompts.py            # Prompt construction
│   ├── retrieval.py          # Query engine and hybrid retriever
│   ├── schemas.py            # Domain schemas
│   ├── telemetry.py          # Structured logging helpers
│   └── vector_store.py       # In-memory vector store with JSON persistence
├── data/sample_feedback.csv  # Demo dataset
├── examples/queries.jsonl    # Example evaluation set
├── docs/architecture.md      # Architecture notes
├── scripts/run_demo.py       # One-command demo script
├── tests/                    # Unit tests
├── .github/workflows/ci.yml  # CI pipeline
├── AGENTS.md                 # Instructions for coding agents
├── ROADMAP.md                # Future roadmap
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Quick start

```bash
poetry install
poetry run ai-showcase index --input data/sample_feedback.csv --index-path .artifacts/vector_store.json
poetry run ai-showcase query "Why are enterprise customers unhappy with onboarding?" --index-path .artifacts/vector_store.json
```

## Retrieval strategies

Three retrievers are available behind a common interface:

- `dense` (default): cosine similarity over hashing embeddings. Good for paraphrased questions.
- `lexical`: a local BM25 index built from the same chunks. Good for exact domain terms such as product names, integration names, or error codes.
- `hybrid`: queries both, min-max normalizes each score list, de-duplicates documents, and combines them as `dense_weight * dense + lexical_weight * lexical` (weights are normalized to sum to 1).

Select the retriever when querying:

```bash
# Default dense retrieval (unchanged behaviour).
poetry run ai-showcase query "Why are enterprise customers unhappy with onboarding?"

# Exact-term lookup with BM25.
poetry run ai-showcase query "Which Salesforce integration problems were reported?" --retriever lexical

# Hybrid retrieval with custom weights.
poetry run ai-showcase query "Which Salesforce integration problems were reported?" \
  --retriever hybrid --dense-weight 0.5 --lexical-weight 0.5
```

The same options work for `ai-showcase evaluate`, so retrieval strategies can be compared offline:

```bash
poetry run ai-showcase evaluate --queries examples/queries.jsonl --retriever hybrid
```

The API uses the retriever configured through the environment (`AI_SHOWCASE_RETRIEVER_TYPE`, `AI_SHOWCASE_DENSE_WEIGHT`, `AI_SHOWCASE_LEXICAL_WEIGHT`).

## Data validation

Ingested datasets are checked against a data contract (`data_contracts.py`) before indexing. The contract requires the columns `feedback_id`, `customer_segment`, `channel`, `rating`, `text`, and `created_at`, and accepts optional `sentiment` and `label` columns. Validation reports missing columns, empty text, duplicate IDs, and invalid timestamps.

Validate a CSV from the CLI:

```bash
poetry run ai-showcase validate-data data/sample_feedback.csv
poetry run ai-showcase validate-data data/sample_feedback.csv --strict
```

The command prints a JSON report with total, valid, and invalid row counts plus row-level errors and warnings. In strict mode (`--strict`, also the default during indexing) any contract violation fails the run; in non-strict mode invalid rows are skipped and the valid rows are kept.

Run the demo:

```bash
poetry run python scripts/run_demo.py
```

## Evaluation

The project ships an offline evaluation harness that measures retrieval quality (precision@k, recall@k, MRR, context hit rate) and answer quality (keyword coverage, groundedness, citation alignment, refusal correctness) over a JSONL dataset:

```bash
poetry run ai-showcase evaluate --queries examples/queries.jsonl --output evaluation_report.json
```

The default output path is `.artifacts/evaluation_report.json`. The run is fully deterministic with the local provider, so the report can be used as a CI regression gate. See [docs/evaluation.md](docs/evaluation.md) for the dataset format and why each metric matters in production RAG systems.

## Experiments

The experiment runner compares retrieval and answer-generation configurations in a repeatable way. An experiment is described by a YAML file covering chunking (`chunk_size`, `chunk_overlap`), embeddings (`embedding_provider`, `embedding_dim`), retrieval (`retriever_type`, `dense_weight`, `lexical_weight`, `top_k`), the LLM provider, and the dataset and query files:

```bash
poetry run ai-showcase experiment run --config examples/experiment_config.yaml
```

The run builds a fresh in-memory index from the configured dataset (the persisted application index is untouched) and writes three files to the configured `output_dir`:

- `results.json`: the configuration plus per-query answers, citations, and metrics.
- `metrics.json`: aggregate retrieval and answer-quality metrics.
- `run_metadata.json`: timestamp, git commit, Python and package versions.

With the local deterministic provider, `results.json` and `metrics.json` are bit-for-bit reproducible; environment-specific values live only in `run_metadata.json`. To compare configurations, copy `examples/experiment_config.yaml`, change one parameter (for example `retriever_type: dense` vs `hybrid`), point `output_dir` at a new folder, and diff the resulting `metrics.json` files.

Run the API:

```bash
poetry run uvicorn ai_engineering_showcase.api:create_app --factory --reload
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What should we improve in onboarding?","top_k":4}'
```

Run tests:

```bash
poetry run pytest
```

Run quality checks (the same gates as CI):

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest --cov=ai_engineering_showcase --cov-fail-under=63
poetry build
```

Or, with `make`:

```bash
make ci
```

## Docker

```bash
docker build -t ai-engineering-showcase .
docker run --rm -p 8000:8000 ai-engineering-showcase
```

## Configuration

The default mode is fully local and deterministic. It does not require an API key.

Environment variables:

| Variable | Default | Description |
|---|---:|---|
| `AI_SHOWCASE_DATA_PATH` | `data/sample_feedback.csv` | CSV file loaded by the API at startup. |
| `AI_SHOWCASE_INDEX_PATH` | `.artifacts/vector_store.json` | Local vector index path. |
| `AI_SHOWCASE_EMBEDDING_DIM` | `512` | Dimension used by the hashing embedding model. |
| `AI_SHOWCASE_RETRIEVER_TYPE` | `dense` | Retrieval strategy: `dense`, `lexical`, or `hybrid`. |
| `AI_SHOWCASE_DENSE_WEIGHT` | `0.6` | Dense score weight used by the hybrid retriever. |
| `AI_SHOWCASE_LEXICAL_WEIGHT` | `0.4` | Lexical (BM25) score weight used by the hybrid retriever. |
| `AI_SHOWCASE_LLM_PROVIDER` | `local` | `local` or `openai`. |
| `OPENAI_API_KEY` | empty | Required only when using the optional OpenAI provider. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name for the optional OpenAI provider. |

Create a local `.env` from `.env.example` if needed.

## Why this project is useful in interviews

This repository lets you discuss AI engineering from multiple angles:

1. **Product thinking**: the system turns unstructured feedback into evidence-backed decisions.
2. **ML engineering**: retrieval, ranking, evaluation, and deterministic tests are first-class components.
3. **Software engineering**: code is typed, modular, tested, and deployable.
4. **Responsible AI**: generated answers include citations and simple grounding checks.
5. **Extensibility**: each layer can be swapped without rewriting the whole system.

## Citation-aware answers

Every generated answer embeds bracketed citation markers (`[1]`, `[2]`) that refer to a
machine-readable citation list built from the actually retrieved chunks. Each citation
carries a stable `citation_id`, the `document_id` and `chunk_id` of the evidence, the
`source` channel, a compact quoted evidence span, and the retrieval score. The agent never
cites a document that was not retrieved, and the local deterministic provider emits the
markers deterministically, so citation output is reproducible in tests and CI.

The same metadata is returned by the API (`result.citations` in the `/query` response) and
rendered as a readable block by the CLI `query` command.

## Example output

```text
$ poetry run ai-showcase query "Why are enterprise customers unhappy with onboarding?"

{
  "question": "Why are enterprise customers unhappy with onboarding?",
  "answer": "The strongest signal is around onboarding [1]. The retrieved feedback points
             to repeated friction in onboarding, checklist, did, know. The answer is
             grounded in feedback sources fb-001 [1], fb-007 [2], fb-009 [3].",
  "recommended_actions": [
    "Create a clearer onboarding checklist with owners, milestones, and escalation rules.",
    "Add proactive support when implementation or setup exceeds the expected timeline.",
    "Group feedback by segment and quantify how often this issue appears."
  ],
  "citations": [
    {
      "citation_id": 1,
      "document_id": "fb-001",
      "chunk_id": "fb-001::chunk-0",
      "source": "support_ticket",
      "quote": "Implementation took three weeks longer than expected. We also had no clear
                onboarding checklist and did not know who owned each setup step.",
      "score": 0.532497
    },
    ...
  ],
  "route": "onboarding",
  "confidence": 0.649
}

Citations:
  [1] fb-001 (support_ticket, chunk fb-001::chunk-0, score 0.532): "Implementation took
      three weeks longer than expected. We also had no clear onboarding checklist..."
  [2] fb-007 (nps_survey, chunk fb-007::chunk-0, score 0.502): "Onboarding felt fragmented..."
  [3] fb-009 (support_ticket, chunk fb-009::chunk-0, score 0.435): "We did not know who
      owned the onboarding checklist..."
```

## License

MIT.
