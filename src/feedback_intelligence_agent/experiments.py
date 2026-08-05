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
from types import ModuleType
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from feedback_intelligence_agent import __version__
from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.chunking import feedback_to_chunks
from feedback_intelligence_agent.config import RetrieverType, Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.evaluation import (
    AnswerMetrics,
    CaseResult,
    RetrievalMetrics,
    aggregate_report,
    evaluate_case_detailed,
    load_evaluation_cases,
)
from feedback_intelligence_agent.factory import build_llm, build_retriever, chunk_to_embedding_text
from feedback_intelligence_agent.ingestion import load_feedback_csv
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

RESULTS_FILENAME = "results.json"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "run_metadata.json"
TrackingProvider = Literal["none", "mlflow"]


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
    llm_provider: Literal["local", "openai", "openai_responses", "bedrock"] = "local"
    retriever_type: RetrieverType = "dense"
    dense_weight: float = Field(default=0.6, ge=0.0)
    lexical_weight: float = Field(default=0.4, ge=0.0)
    tracking_provider: TrackingProvider = "none"
    tracking_uri: str | None = None
    tracking_experiment_name: str = "feedback-intelligence-agent"

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


class ExperimentTrackingError(RuntimeError):
    """Raised when an optional experiment tracker cannot log a run."""


def track_experiment_run(
    result: ExperimentResult,
    metadata: RunMetadata,
    artifacts: dict[str, Path],
) -> str | None:
    """Log an experiment run to the configured tracking backend.

    ``tracking_provider='none'`` is the default and returns ``None``. The MLflow
    backend is imported lazily so the default install remains lightweight.
    """
    if result.config.tracking_provider == "none":
        return None
    if result.config.tracking_provider == "mlflow":
        return _track_with_mlflow(result, metadata, artifacts)
    raise ExperimentTrackingError(
        f"Unsupported tracking provider: {result.config.tracking_provider}"
    )


def _track_with_mlflow(
    result: ExperimentResult,
    metadata: RunMetadata,
    artifacts: dict[str, Path],
) -> str:
    """Log params, metrics, tags, and artifacts to MLflow."""
    mlflow = _import_mlflow()
    config = result.config
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.tracking_experiment_name)
    with mlflow.start_run(run_name=config.name) as run:
        mlflow.log_params(_tracking_params(config))
        mlflow.log_metrics(_tracking_metrics(result.metrics))
        if metadata.git_commit:
            mlflow.set_tag("git_commit", metadata.git_commit)
        mlflow.set_tag("package_version", metadata.package_version)
        mlflow.set_tag("description", config.description)
        for path in artifacts.values():
            mlflow.log_artifact(str(path))
    run_id = getattr(getattr(run, "info", None), "run_id", None)
    return str(run_id) if run_id else ""


def _import_mlflow() -> ModuleType:
    """Import optional MLflow with an actionable setup message."""
    try:
        import mlflow
    except ImportError as exc:
        raise ExperimentTrackingError(
            "The 'mlflow' package is required for experiment tracking. "
            "Install it with: poetry install --extras mlflow (or: pip install mlflow)."
        ) from exc
    module: ModuleType = mlflow
    return module


def _tracking_params(config: ExperimentConfig) -> dict[str, str | int | float]:
    """Flatten experiment config fields into MLflow-safe parameters."""
    return {
        "dataset_path": str(config.dataset_path),
        "queries_path": str(config.queries_path),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "embedding_provider": config.embedding_provider,
        "embedding_dim": config.embedding_dim,
        "llm_provider": config.llm_provider,
        "retriever_type": config.retriever_type,
        "dense_weight": config.dense_weight,
        "lexical_weight": config.lexical_weight,
    }


def _tracking_metrics(metrics: AggregateMetrics) -> dict[str, float]:
    """Flatten aggregate metrics into MLflow scalar metrics."""
    return {
        "total_cases": float(metrics.total_cases),
        "retrieval_precision_at_k": metrics.retrieval.precision_at_k,
        "retrieval_recall_at_k": metrics.retrieval.recall_at_k,
        "retrieval_mrr": metrics.retrieval.mean_reciprocal_rank,
        "retrieval_context_hit_rate": metrics.retrieval.context_hit_rate,
        "answer_keyword_coverage": metrics.answers.keyword_coverage,
        "answer_groundedness": metrics.answers.groundedness,
        "answer_refusal_correctness": metrics.answers.refusal_correctness,
        "answer_citation_alignment": metrics.answers.citation_alignment,
    }


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
