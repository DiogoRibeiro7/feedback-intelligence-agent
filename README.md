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
│   ├── embeddings.py         # Hashing embedding model
│   ├── evaluation.py         # Retrieval and answer-quality metrics
│   ├── ingestion.py          # CSV feedback loader
│   ├── llm.py                # LLM abstraction and local fallback
│   ├── prompts.py            # Prompt construction
│   ├── retrieval.py          # Query engine
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

Run the demo:

```bash
poetry run python scripts/run_demo.py
```

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

## Example output

```text
Question: Why are enterprise customers unhappy with onboarding?

Answer:
Enterprise customers are mainly unhappy because onboarding feels slow, handoffs are unclear,
and success criteria are not visible inside the product. The strongest evidence comes from
feedback mentioning setup delays, unclear ownership, and missing progress tracking.

Recommended actions:
- Add an onboarding progress dashboard.
- Create clearer ownership between sales, support, and customer success.
- Trigger proactive follow-ups when setup exceeds the expected timeline.

Citations:
- fb-001: "Implementation took three weeks longer than expected..."
- fb-009: "We did not know who owned the onboarding checklist..."
```

## License

MIT.
