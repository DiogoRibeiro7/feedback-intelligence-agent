# Contributing

This project is intentionally local-first and deterministic. Changes should keep
the default path working without external APIs, managed services, or private
data.

## Development setup

```bash
poetry install
cd frontend
npm install
```

## Before opening a pull request

Run the same checks used by CI:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest --cov=feedback_intelligence_agent --cov-fail-under=63
poetry run python scripts/run_demo.py
```

For frontend changes, also run:

```bash
cd frontend
npm audit
npm run build
```

## Engineering expectations

- Keep modules small, typed, and explicit.
- Add tests for new behavior.
- Preserve deterministic behavior for the local provider, local embeddings, and
  evaluation path.
- Prefer Pydantic models for external request and response schemas.
- Do not commit secrets, real customer data, generated datasets, or local
  artifacts.

## Pull request scope

Keep each pull request focused on one change. If a fix requires a broader
refactor, describe the reason clearly in the PR summary and call out any
behavior that changed.
