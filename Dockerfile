FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY data ./data
COPY docs ./docs
COPY scripts ./scripts

RUN poetry install --only main --no-root \
    && poetry install --only-root

EXPOSE 8000

CMD ["uvicorn", "feedback_intelligence_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
