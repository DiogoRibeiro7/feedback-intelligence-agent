.PHONY: install test lint format-check typecheck typecheck-docs coverage build quality ci demo api docs docs-check docs-pdf docs-pdf-reference docs-clean docker-build docker-run compose-prod-up compose-prod-down clean

COVERAGE_THRESHOLD := 63

# The documentation toolchain lives outside the distributed package, so it is
# imported as the top-level `tools` package with docs/ on the import path.
DOCS_ENV := PYTHONPATH=docs
# Documentation generation reads pyproject.toml with tomllib (Python 3.11+),
# while the package itself supports 3.10; mypy is told which version to assume.
DOCS_PYTHON_VERSION := 3.11
LATEX_DIR := docs/latex

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

format-check:
	poetry run ruff format --check .

typecheck:
	poetry run mypy src

typecheck-docs:
	poetry run mypy --python-version $(DOCS_PYTHON_VERSION) docs/tools

coverage:
	poetry run pytest --cov=feedback_intelligence_agent --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

build:
	poetry build

quality: lint typecheck typecheck-docs test

ci: lint format-check typecheck typecheck-docs coverage build

# Regenerate the LaTeX technical manual from the repository: inspect, extract
# the API and dependency model, render every fragment, then validate the tree.
# Performs no application imports and no network access.
docs:
	$(DOCS_ENV) poetry run python -m tools.generate_docs --repo-root . \
		$(if $(REVISION),--revision $(REVISION),)

# Validate the committed documentation without regenerating it.
docs-check:
	$(DOCS_ENV) poetry run python -m tools.validate_docs --repo-root .

# latexmk runs with -f because MiKTeX reports a non-zero status for a pass that
# merely had undefined references, which every first pass has; latexmk would
# otherwise stop before the passes that resolve them. It is invoked twice
# because a run that stops early leaves the .toc/.lof/.lot written but not yet
# read back, which silently produces empty contents lists. check_latex_log then
# judges the outcome from the final log and fails the target if anything --
# an error, an unresolved reference, or an empty contents list -- remains.
LATEXMK := latexmk -f -pdf -interaction=nonstopmode -file-line-error

# Compile the complete manual: reference, architecture, and engineering.
docs-pdf: docs
	cd $(LATEX_DIR) && $(LATEXMK) main.tex || true
	cd $(LATEX_DIR) && $(LATEXMK) main.tex || true
	$(DOCS_ENV) poetry run python -m tools.check_latex_log $(LATEX_DIR)/main.log

# The CRAN-style standalone: the documented objects only, without the
# architecture and engineering parts.
docs-pdf-reference: docs
	cd $(LATEX_DIR) && $(LATEXMK) reference.tex || true
	cd $(LATEX_DIR) && $(LATEXMK) reference.tex || true
	$(DOCS_ENV) poetry run python -m tools.check_latex_log $(LATEX_DIR)/reference.log

docs-clean:
	cd $(LATEX_DIR) && latexmk -C main.tex reference.tex

demo:
	poetry run python scripts/run_demo.py

api:
	poetry run uvicorn feedback_intelligence_agent.api:create_app --factory --reload

docker-build:
	docker build -t feedback-intelligence-agent .

docker-run:
	docker run --rm -p 8000:8000 feedback-intelligence-agent

# Production-like Docker Compose (built image, gunicorn workers, healthcheck).
# Requires a deploy/.env.prod file (copy from .env.example) and the 'latest'
# image tag, e.g. `make docker-build && docker tag feedback-intelligence-agent feedback-intelligence-agent:latest`.
compose-prod-up:
	docker compose -f deploy/docker-compose.prod.yml up -d

compose-prod-down:
	docker compose -f deploy/docker-compose.prod.yml down

clean:
	rm -rf .artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
