"""Repeatable experiment runner for comparing RAG configurations.

An experiment is described by a YAML file (parsed into a typed
:class:`ExperimentConfig`), executed fully locally against a feedback CSV and a
JSONL query set, and persisted as three JSON artifacts:

- ``results.json``: configuration plus per-query answers and metrics.
- ``metrics.json``: aggregate retrieval and answer-quality metrics.
- ``run_metadata.json``: timestamp, git commit, Python and package versions.

With the deterministic local LLM provider, ``results.json`` and
``metrics.json`` are bit-for-bit reproducible; everything environment-specific
lives in ``run_metadata.json``. This makes experiment outputs directly
comparable across configurations and usable as CI regression baselines.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ai_engineering_showcase import __version__
from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.config import RetrieverType, Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.evaluation import (
    AnswerMetrics,
    CaseResult,
    RetrievalMetrics,
    aggregate_report,
    evaluate_case_detailed,
    load_evaluation_cases,
)
from ai_engineering_showcase.factory import build_llm, build_retriever, chunk_to_embedding_text
from ai_engineering_showcase.ingestion import load_feedback_csv
from ai_engineering_showcase.vector_store import InMemoryVectorStore

RESULTS_FILENAME = "results.json"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "run_metadata.json"


class ExperimentConfig(BaseModel):
    """Typed description of one experiment run.

    The configuration covers the full pipeline: chunking, embedding,
    retrieval strategy, answer generation, and the dataset and query files
    used for evaluation.
    """

    name: str = Field(default="experiment", min_length=1)
    description: str = ""
    dataset_path: Path = Path("data/sample_feedback.csv")
    queries_path: Path = Path("examples/queries.jsonl")
    output_dir: Path = Path(".artifacts/experiments/default")
    chunk_size: int = Field(default=80, ge=1, description="Maximum words per chunk.")
    chunk_overlap: int = Field(default=16, ge=0, description="Words shared between chunks.")
    top_k: int = Field(default=4, ge=1, le=12)
    embedding_provider: Literal["hashing"] = "hashing"
    embedding_dim: int = Field(default=512, ge=64, le=8192)
    llm_provider: Literal["local", "openai"] = "local"
    retriever_type: RetrieverType = "dense"
    dense_weight: float = Field(default=0.6, ge=0.0)
    lexical_weight: float = Field(default=0.4, ge=0.0)

    @model_validator(mode="after")
    def check_chunk_overlap(self) -> ExperimentConfig:
        """Reject overlaps that would prevent chunking from terminating."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Experiment config must be a YAML mapping, got: {type(raw).__name__}")
        return cls.model_validate(raw)


class QueryResult(BaseModel):
    """Outcome of a single evaluation query within an experiment."""

    question: str
    answer: str
    cited_source_ids: list[str]
    metrics: CaseResult


class AggregateMetrics(BaseModel):
    """Aggregate retrieval and answer metrics for one experiment run."""

    top_k: int = Field(ge=1)
    total_cases: int = Field(ge=0)
    retrieval: RetrievalMetrics
    answers: AnswerMetrics


class ExperimentResult(BaseModel):
    """Full, deterministic outcome of one experiment run."""

    config: ExperimentConfig
    metrics: AggregateMetrics
    query_results: list[QueryResult] = Field(default_factory=list)


class RunMetadata(BaseModel):
    """Environment-specific metadata captured alongside an experiment."""

    timestamp: str
    git_commit: str | None
    python_version: str
    package_version: str
    config: ExperimentConfig


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Execute one experiment end to end and return the typed result.

    The index is built in memory from the configured dataset using the
    configured chunking and embedding parameters, so experiments never touch
    the persisted application index.
    """
    settings = Settings(
        data_path=config.dataset_path,
        embedding_dim=config.embedding_dim,
        retriever_type=config.retriever_type,
        dense_weight=config.dense_weight,
        lexical_weight=config.lexical_weight,
        llm_provider=config.llm_provider,
    )
    vector_store = _build_in_memory_index(config)
    retriever = build_retriever(settings, vector_store)
    agent = FeedbackInsightAgent(query_engine=retriever, llm=build_llm(settings))
    cases = load_evaluation_cases(config.queries_path)

    query_results: list[QueryResult] = []
    for case in cases:
        case_result, answer = evaluate_case_detailed(retriever, agent, case, top_k=config.top_k)
        query_results.append(
            QueryResult(
                question=case.question,
                answer=answer.answer,
                cited_source_ids=[citation.document_id for citation in answer.citations],
                metrics=case_result,
            )
        )

    report = aggregate_report([result.metrics for result in query_results], top_k=config.top_k)
    metrics = AggregateMetrics(
        top_k=report.top_k,
        total_cases=report.total_cases,
        retrieval=report.retrieval,
        answers=report.answers,
    )
    return ExperimentResult(config=config, metrics=metrics, query_results=query_results)


def collect_run_metadata(config: ExperimentConfig) -> RunMetadata:
    """Capture environment metadata for reproducibility audits."""
    return RunMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit_hash(),
        python_version=platform.python_version(),
        package_version=__version__,
        config=config,
    )


def write_experiment_outputs(
    result: ExperimentResult,
    metadata: RunMetadata,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write results, metrics, and run metadata JSON files.

    Args:
        result: Deterministic experiment result.
        metadata: Environment metadata for this run.
        output_dir: Destination directory; defaults to the configured one.

    Returns:
        Mapping of artifact name to the path written.
    """
    destination = output_dir if output_dir is not None else result.config.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {
        RESULTS_FILENAME: result.model_dump_json(indent=2),
        METRICS_FILENAME: result.metrics.model_dump_json(indent=2),
        METADATA_FILENAME: metadata.model_dump_json(indent=2),
    }
    paths: dict[str, Path] = {}
    for filename, payload in artifacts.items():
        path = destination / filename
        path.write_text(payload + "\n", encoding="utf-8")
        paths[filename] = path
    return paths


def _build_in_memory_index(config: ExperimentConfig) -> InMemoryVectorStore:
    """Build a fresh in-memory vector index from the experiment configuration."""
    records = load_feedback_csv(config.dataset_path)
    chunks = feedback_to_chunks(
        records,
        max_words=config.chunk_size,
        overlap_words=config.chunk_overlap,
    )
    embedding_model = _build_embedding_model(config)
    vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
    vector_store = InMemoryVectorStore(dim=config.embedding_dim)
    vector_store.add(chunks, vectors)
    return vector_store


def _build_embedding_model(config: ExperimentConfig) -> HashingEmbeddingModel:
    """Construct the configured embedding provider."""
    if config.embedding_provider != "hashing":
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    return HashingEmbeddingModel(dim=config.embedding_dim)


def _git_commit_hash() -> str | None:
    """Return the current git commit hash, or None when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None
