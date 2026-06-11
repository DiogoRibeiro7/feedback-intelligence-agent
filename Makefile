.PHONY: install test lint format-check typecheck coverage build quality ci demo api docker-build docker-run clean

COVERAGE_THRESHOLD := 63

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

coverage:
	poetry run pytest --cov=ai_engineering_showcase --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

build:
	poetry build

quality: lint typecheck test

ci: lint format-check typecheck coverage build

demo:
	poetry run python scripts/run_demo.py

api:
	poetry run uvicorn ai_engineering_showcase.api:create_app --factory --reload

docker-build:
	docker build -t ai-engineering-showcase .

docker-run:
	docker run --rm -p 8000:8000 ai-engineering-showcase

clean:
	rm -rf .artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
