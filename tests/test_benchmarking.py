from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_engineering_showcase.benchmarking import (
    PHASE_AGENT_RESPONSE,
    PHASE_EMBEDDING,
    PHASE_INDEXING,
    PHASE_RETRIEVAL,
    BenchmarkReport,
    PhaseStats,
    percentile,
    report_to_markdown,
    run_benchmark,
    summarize,
    write_benchmark_outputs,
)
from ai_engineering_showcase.cli import app

runner = CliRunner()

DATASET = "data/sample_feedback.csv"
QUERIES = "examples/queries.jsonl"


# ---------------------------------------------------------------------------
# percentile (pure)
# ---------------------------------------------------------------------------


def test_percentile_single_sample() -> None:
    assert percentile([42.0], 0.95) == 42.0
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 1.0) == 42.0


def test_percentile_identical_samples() -> None:
    assert percentile([5.0, 5.0, 5.0, 5.0], 0.95) == 5.0


def test_percentile_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # ceil(0.95 * 10) = 10 -> the largest value.
    assert percentile(values, 0.95) == 10.0
    # ceil(0.5 * 10) = 5 -> the 5th smallest value.
    assert percentile(values, 0.5) == 5.0
    # Ordering of the input does not matter.
    assert percentile(list(reversed(values)), 0.95) == 10.0


def test_percentile_clamps_to_max_rank() -> None:
    assert percentile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        percentile([], 0.95)


def test_percentile_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], 1.5)
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], -0.1)


# ---------------------------------------------------------------------------
# summarize (pure)
# ---------------------------------------------------------------------------


def test_summarize_basic_statistics() -> None:
    stats = summarize("retrieval", [10.0, 20.0, 30.0, 40.0])
    assert stats.phase == "retrieval"
    assert stats.samples == 4
    assert stats.mean_ms == 25.0
    assert stats.median_ms == 25.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 40.0
    # ceil(0.95 * 4) = 4 -> the largest sample.
    assert stats.p95_ms == 40.0


def test_summarize_single_sample() -> None:
    stats = summarize("embedding", [12.5])
    assert stats.samples == 1
    assert stats.mean_ms == 12.5
    assert stats.median_ms == 12.5
    assert stats.p95_ms == 12.5
    assert stats.min_ms == 12.5
    assert stats.max_ms == 12.5


def test_summarize_identical_samples() -> None:
    stats = summarize("indexing", [7.0, 7.0, 7.0])
    assert stats.mean_ms == 7.0
    assert stats.median_ms == 7.0
    assert stats.p95_ms == 7.0
    assert stats.min_ms == 7.0
    assert stats.max_ms == 7.0


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no samples"):
        summarize("retrieval", [])


# ---------------------------------------------------------------------------
# Markdown rendering (pure, no timing values asserted)
# ---------------------------------------------------------------------------


def _fixture_report() -> BenchmarkReport:
    return BenchmarkReport(
        repetitions=3,
        warmup=1,
        top_k=4,
        dataset_path=DATASET,
        queries_path=QUERIES,
        num_chunks=12,
        num_queries=5,
        phases=[
            PhaseStats(
                phase=PHASE_INDEXING,
                samples=3,
                mean_ms=1.0,
                median_ms=1.0,
                p95_ms=1.0,
                min_ms=1.0,
                max_ms=1.0,
            )
        ],
    )


def test_report_to_markdown_structure() -> None:
    markdown = report_to_markdown(_fixture_report())
    assert markdown.startswith("# Benchmark results")
    assert "| Phase | Samples | Mean (ms)" in markdown
    assert f"`{DATASET}`" in markdown
    assert PHASE_INDEXING in markdown
    assert "Chunks indexed: 12" in markdown


# ---------------------------------------------------------------------------
# End-to-end benchmark on the sample dataset (structure only; never timings)
# ---------------------------------------------------------------------------


def test_run_benchmark_reports_all_phases() -> None:
    report = run_benchmark(
        dataset_path=DATASET,
        queries_path=QUERIES,
        repetitions=2,
        warmup=1,
        top_k=4,
    )
    assert report.repetitions == 2
    assert report.warmup == 1
    assert report.num_queries == 5
    assert report.num_chunks > 0
    phases = {phase.phase for phase in report.phases}
    assert phases == {
        PHASE_INDEXING,
        PHASE_EMBEDDING,
        PHASE_RETRIEVAL,
        PHASE_AGENT_RESPONSE,
    }
    for phase in report.phases:
        assert phase.samples == 2
        # Durations are non-negative and ordered min <= max; never assert exact values.
        assert 0.0 <= phase.min_ms <= phase.max_ms
        assert phase.min_ms <= phase.mean_ms <= phase.max_ms


def test_run_benchmark_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        run_benchmark(dataset_path=DATASET, queries_path=QUERIES, repetitions=0)
    with pytest.raises(ValueError, match="warmup"):
        run_benchmark(dataset_path=DATASET, queries_path=QUERIES, warmup=-1)


def test_run_benchmark_rejects_empty_queries(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no benchmark queries"):
        run_benchmark(dataset_path=DATASET, queries_path=empty)


def test_write_benchmark_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_benchmark(
        dataset_path=DATASET,
        queries_path=QUERIES,
        repetitions=2,
        warmup=0,
    )
    paths = write_benchmark_outputs(report, tmp_path / "nested" / "out")
    assert set(paths) == {"benchmark_results.json", "benchmark_results.md"}
    payload = json.loads(paths["benchmark_results.json"].read_text(encoding="utf-8"))
    assert payload["num_queries"] == 5
    assert len(payload["phases"]) == 4
    markdown = paths["benchmark_results.md"].read_text(encoding="utf-8")
    assert markdown.startswith("# Benchmark results")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_benchmark_cli_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--queries",
            QUERIES,
            "--output",
            str(output_dir),
            "--repetitions",
            "2",
            "--warmup",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_results.md").exists()
    assert '"num_queries": 5' in result.output
