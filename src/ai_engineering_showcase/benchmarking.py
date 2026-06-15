"""Latency and throughput benchmarks for the RAG pipeline.

This module times the four phases that dominate end-to-end cost in a retrieval
system and reports robust summary statistics per phase:

- ``indexing``: load the CSV, chunk it, embed every chunk, and build the in-memory
  vector store (one timed unit per repetition).
- ``embedding``: embed the benchmark queries with the hashing embedding model.
- ``retrieval``: run every query through the retriever and collect the top-k chunks.
- ``agent_response``: run every query through the full agent (retrieval plus answer
  generation with the deterministic local provider).

Each phase is run ``warmup`` times to prime caches and then ``repetitions`` times for
measurement. Durations come from :func:`time.perf_counter`, so they are wall-clock
times and inherently non-deterministic. Only the *structure* of the report is
deterministic; never snapshot the timing values.

The summary statistics (mean, median, p95, min, max) are exposed as pure functions so
they can be unit-tested without timing anything. Results serialise to both JSON and a
small Markdown table via :func:`write_benchmark_outputs`.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.evaluation import load_evaluation_cases
from ai_engineering_showcase.factory import build_llm, build_retriever, chunk_to_embedding_text
from ai_engineering_showcase.ingestion import load_feedback_csv
from ai_engineering_showcase.vector_store import InMemoryVectorStore

RESULTS_JSON_FILENAME = "benchmark_results.json"
RESULTS_MARKDOWN_FILENAME = "benchmark_results.md"

PHASE_INDEXING = "indexing"
PHASE_EMBEDDING = "embedding"
PHASE_RETRIEVAL = "retrieval"
PHASE_AGENT_RESPONSE = "agent_response"


def percentile(values: list[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``values`` (e.g. 0.95 for p95).

    Uses the *nearest-rank* method on the sorted samples: the rank is
    ``ceil(fraction * n)`` (clamped to ``[1, n]``) and the value at that 1-based
    rank is returned. Nearest-rank is chosen over interpolation because it always
    returns an actually-observed sample, which is easy to reason about for latency
    SLOs and avoids fabricating values between measurements.

    Args:
        values: Non-empty list of samples (order does not matter).
        fraction: Percentile as a fraction in ``[0.0, 1.0]``.

    Returns:
        The sample at the nearest rank for ``fraction``.

    Raises:
        ValueError: If ``values`` is empty or ``fraction`` is outside ``[0, 1]``.
    """
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    ordered = sorted(values)
    count = len(ordered)
    if fraction == 0.0:
        return ordered[0]
    # Nearest-rank: rank = ceil(fraction * n), clamped into [1, n].
    rank = max(1, min(math.ceil(fraction * count), count))
    return ordered[rank - 1]


class PhaseStats(BaseModel):
    """Summary statistics for one benchmarked phase, in milliseconds."""

    phase: str
    samples: int = Field(ge=1)
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def summarize(phase: str, durations_ms: list[float]) -> PhaseStats:
    """Compute summary statistics for one phase from its per-repetition durations.

    Args:
        phase: Phase name used as a label in the report.
        durations_ms: Per-repetition durations in milliseconds (non-empty).

    Returns:
        A :class:`PhaseStats` with mean, median, p95, min, and max, each rounded
        to four decimal places.

    Raises:
        ValueError: If ``durations_ms`` is empty.
    """
    if not durations_ms:
        raise ValueError(f"phase {phase!r} has no samples to summarize")
    return PhaseStats(
        phase=phase,
        samples=len(durations_ms),
        mean_ms=round(statistics.fmean(durations_ms), 4),
        median_ms=round(statistics.median(durations_ms), 4),
        p95_ms=round(percentile(durations_ms, 0.95), 4),
        min_ms=round(min(durations_ms), 4),
        max_ms=round(max(durations_ms), 4),
    )


class BenchmarkReport(BaseModel):
    """Full benchmark report: configuration plus per-phase statistics."""

    repetitions: int = Field(ge=1)
    warmup: int = Field(ge=0)
    top_k: int = Field(ge=1)
    dataset_path: str
    queries_path: str
    num_chunks: int = Field(ge=0)
    num_queries: int = Field(ge=0)
    phases: list[PhaseStats] = Field(default_factory=list)


def _time_calls(func: Callable[[], object], *, repetitions: int, warmup: int) -> list[float]:
    """Run ``func`` ``warmup`` then ``repetitions`` times, returning timings in ms.

    Warmup calls are executed but discarded; only the measured repetitions are
    returned. Timing uses :func:`time.perf_counter`.
    """
    for _ in range(warmup):
        func()
    durations_ms: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        func()
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    return durations_ms


def run_benchmark(
    *,
    dataset_path: str | Path,
    queries_path: str | Path,
    repetitions: int = 5,
    warmup: int = 1,
    top_k: int = 4,
    settings: Settings | None = None,
) -> BenchmarkReport:
    """Benchmark the four pipeline phases and return a typed report.

    The benchmark builds a fresh in-memory index from ``dataset_path`` (the
    persisted application index is never touched) and uses the deterministic
    local LLM provider unless ``settings`` selects another one.

    Args:
        dataset_path: Feedback CSV used to build the index.
        queries_path: JSONL evaluation cases supplying benchmark questions.
        repetitions: Number of measured repetitions per phase (>= 1).
        warmup: Number of discarded warmup repetitions per phase (>= 0).
        top_k: Number of chunks retrieved per query.
        settings: Optional settings override; defaults to a local configuration.

    Returns:
        A :class:`BenchmarkReport` with per-phase summary statistics.

    Raises:
        ValueError: If ``repetitions`` < 1, ``warmup`` < 0, or there are no queries.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    dataset_path = Path(dataset_path)
    queries_path = Path(queries_path)
    resolved_settings = settings or Settings(data_path=dataset_path)
    embedding_dim = resolved_settings.embedding_dim

    cases = load_evaluation_cases(queries_path)
    questions = [case.question for case in cases]
    if not questions:
        raise ValueError(f"no benchmark queries found in {queries_path}")

    embedding_model = HashingEmbeddingModel(dim=embedding_dim)

    def build_store() -> InMemoryVectorStore:
        records = load_feedback_csv(dataset_path)
        chunks = feedback_to_chunks(records)
        vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
        store = InMemoryVectorStore(dim=embedding_dim)
        store.add(chunks, vectors)
        return store

    # Build a reference index once, reused by the retrieval and agent phases.
    vector_store = build_store()
    retriever = build_retriever(resolved_settings, vector_store)
    agent = FeedbackInsightAgent(query_engine=retriever, llm=build_llm(resolved_settings))

    def index_phase() -> None:
        build_store()

    def embedding_phase() -> None:
        embedding_model.embed(questions)

    def retrieval_phase() -> None:
        for question in questions:
            retriever.search(question, top_k=top_k)

    def agent_phase() -> None:
        for question in questions:
            agent.answer(question, top_k=top_k)

    phase_funcs: list[tuple[str, Callable[[], object]]] = [
        (PHASE_INDEXING, index_phase),
        (PHASE_EMBEDDING, embedding_phase),
        (PHASE_RETRIEVAL, retrieval_phase),
        (PHASE_AGENT_RESPONSE, agent_phase),
    ]
    phases = [
        summarize(name, _time_calls(func, repetitions=repetitions, warmup=warmup))
        for name, func in phase_funcs
    ]

    return BenchmarkReport(
        repetitions=repetitions,
        warmup=warmup,
        top_k=top_k,
        dataset_path=dataset_path.as_posix(),
        queries_path=queries_path.as_posix(),
        num_chunks=vector_store.size,
        num_queries=len(questions),
        phases=phases,
    )


def report_to_markdown(report: BenchmarkReport) -> str:
    """Render a benchmark report as a small Markdown document with a results table."""
    lines = [
        "# Benchmark results",
        "",
        f"- Dataset: `{report.dataset_path}`",
        f"- Queries: `{report.queries_path}` ({report.num_queries} questions)",
        f"- Chunks indexed: {report.num_chunks}",
        f"- Repetitions: {report.repetitions} (warmup: {report.warmup})",
        f"- top_k: {report.top_k}",
        "",
        "| Phase | Samples | Mean (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in report.phases:
        lines.append(
            f"| {phase.phase} | {phase.samples} | {phase.mean_ms} | {phase.median_ms} "
            f"| {phase.p95_ms} | {phase.min_ms} | {phase.max_ms} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_outputs(report: BenchmarkReport, output_dir: str | Path) -> dict[str, Path]:
    """Write the report as JSON and Markdown into ``output_dir``.

    Args:
        report: The benchmark report to persist.
        output_dir: Destination directory (created if missing).

    Returns:
        Mapping of artifact filename to the path written.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / RESULTS_JSON_FILENAME
    markdown_path = destination / RESULTS_MARKDOWN_FILENAME
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return {RESULTS_JSON_FILENAME: json_path, RESULTS_MARKDOWN_FILENAME: markdown_path}
