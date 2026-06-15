"""Run the RAG latency benchmark from the command line.

Thin wrapper over :mod:`ai_engineering_showcase.benchmarking`. The CLI command
``ai-showcase benchmark`` offers the same functionality with more options; this
script exists for a quick one-command run against the sample dataset.
"""

from __future__ import annotations

from pathlib import Path

from ai_engineering_showcase.benchmarking import run_benchmark, write_benchmark_outputs


def main() -> None:
    """Benchmark the local pipeline and write JSON + Markdown results."""
    report = run_benchmark(
        dataset_path=Path("data/sample_feedback.csv"),
        queries_path=Path("examples/queries.jsonl"),
        repetitions=5,
        warmup=1,
    )
    paths = write_benchmark_outputs(report, Path(".artifacts/benchmark_results"))
    print(report.model_dump_json(indent=2))
    for filename, path in paths.items():
        print(f"{filename} written to {path}")


if __name__ == "__main__":
    main()
