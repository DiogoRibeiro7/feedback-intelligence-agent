from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from feedback_intelligence_agent import __version__
from feedback_intelligence_agent.cli import app
from feedback_intelligence_agent.evaluation import AnswerMetrics, CaseResult, RetrievalMetrics
from feedback_intelligence_agent.experiments import (
    AggregateMetrics,
    ExperimentConfig,
    ExperimentResult,
    QueryResult,
    RunMetadata,
    collect_run_metadata,
    run_experiment,
    track_experiment_run,
    write_experiment_outputs,
)

runner = CliRunner()


def write_config_yaml(path: Path, output_dir: Path, **overrides: object) -> Path:
    """Write a minimal experiment YAML pointing at the sample dataset."""
    values: dict[str, object] = {
        "name": "test-experiment",
        "dataset_path": "data/sample_feedback.csv",
        "queries_path": "examples/queries.jsonl",
        "output_dir": output_dir.as_posix(),
        "retriever_type": "hybrid",
        **overrides,
    }
    lines = [f'{key}: "{value}"' for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_from_yaml_parses_fields_and_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: my-experiment",
                "chunk_size: 60",
                "chunk_overlap: 10",
                "top_k: 3",
                "retriever_type: lexical",
            ]
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig.from_yaml(config_path)
    assert config.name == "my-experiment"
    assert config.chunk_size == 60
    assert config.chunk_overlap == 10
    assert config.top_k == 3
    assert config.retriever_type == "lexical"
    # Unspecified values fall back to defaults.
    assert config.embedding_provider == "hashing"
    assert config.embedding_dim == 512
    assert config.llm_provider == "local"
    assert config.tracking_provider == "none"
    assert config.tracking_experiment_name == "feedback-intelligence-agent"
    assert config.dataset_path == Path("data/sample_feedback.csv")


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        ExperimentConfig.from_yaml(config_path)


def test_config_rejects_overlap_not_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ExperimentConfig(chunk_size=20, chunk_overlap=20)


def test_config_rejects_unknown_providers() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(llm_provider="anthropic")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExperimentConfig(embedding_provider="openai")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExperimentConfig(retriever_type="graph")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExperimentConfig(tracking_provider="wandb")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def build_small_result() -> ExperimentResult:
    case = CaseResult(
        question="Why is onboarding slow?",
        is_answerable=True,
        retrieved_document_ids=["fb-001"],
        precision_at_k=1.0,
        recall_at_k=0.5,
        reciprocal_rank=1.0,
        context_hit=True,
        keyword_coverage=1.0,
        groundedness=0.8,
        citation_aligned=True,
        refused=False,
        refusal_correct=True,
    )
    return ExperimentResult(
        config=ExperimentConfig(name="serialization-test"),
        metrics=AggregateMetrics(
            top_k=4,
            total_cases=1,
            retrieval=RetrievalMetrics(
                precision_at_k=1.0,
                recall_at_k=0.5,
                mean_reciprocal_rank=1.0,
                context_hit_rate=1.0,
                evaluated_cases=1,
            ),
            answers=AnswerMetrics(
                keyword_coverage=1.0,
                groundedness=0.8,
                refusal_correctness=1.0,
                citation_alignment=1.0,
                evaluated_cases=1,
            ),
        ),
        query_results=[
            QueryResult(
                question="Why is onboarding slow?",
                answer="Onboarding is slow because of unclear handoffs.",
                cited_source_ids=["fb-001"],
                metrics=case,
            )
        ],
    )


def test_experiment_result_json_roundtrip() -> None:
    result = build_small_result()
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result
    payload = json.loads(result.model_dump_json())
    assert payload["config"]["name"] == "serialization-test"
    assert payload["metrics"]["retrieval"]["precision_at_k"] == 1.0
    assert payload["query_results"][0]["cited_source_ids"] == ["fb-001"]


def test_run_metadata_contents() -> None:
    config = ExperimentConfig(name="metadata-test")
    metadata = collect_run_metadata(config)
    # ISO-8601 timestamp with explicit UTC offset.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", metadata.timestamp)
    assert metadata.timestamp.endswith("+00:00")
    assert metadata.package_version == __version__
    assert re.match(r"^\d+\.\d+\.\d+", metadata.python_version)
    # Repo tests run inside git, but None is tolerated (e.g. sdist installs).
    assert metadata.git_commit is None or re.fullmatch(r"[0-9a-f]{40}", metadata.git_commit)
    assert metadata.config.name == "metadata-test"
    restored = RunMetadata.model_validate_json(metadata.model_dump_json())
    assert restored == metadata


# ---------------------------------------------------------------------------
# End-to-end experiment with the deterministic local provider
# ---------------------------------------------------------------------------


def test_run_experiment_end_to_end_is_deterministic(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="e2e",
        output_dir=tmp_path / "out",
        retriever_type="hybrid",
        top_k=4,
    )
    first = run_experiment(config)
    second = run_experiment(config)
    assert first.model_dump_json() == second.model_dump_json()
    assert first.metrics.total_cases == 5
    assert first.metrics.retrieval.evaluated_cases == 4
    assert 0.0 <= first.metrics.retrieval.precision_at_k <= 1.0
    assert 0.0 <= first.metrics.answers.groundedness <= 1.0
    assert len(first.query_results) == 5
    assert all(result.answer for result in first.query_results)


def test_write_experiment_outputs_creates_three_files(tmp_path: Path) -> None:
    config = ExperimentConfig(name="outputs", output_dir=tmp_path / "nested" / "out")
    result = run_experiment(config)
    metadata = collect_run_metadata(config)
    paths = write_experiment_outputs(result, metadata)
    assert set(paths) == {"results.json", "metrics.json", "run_metadata.json"}
    for path in paths.values():
        assert path.exists()
    results_payload = json.loads(paths["results.json"].read_text(encoding="utf-8"))
    metrics_payload = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
    metadata_payload = json.loads(paths["run_metadata.json"].read_text(encoding="utf-8"))
    assert results_payload["config"]["name"] == "outputs"
    assert metrics_payload == results_payload["metrics"]
    # Environment-specific values live only in run_metadata.json.
    assert "timestamp" in metadata_payload
    assert "timestamp" not in results_payload
    assert "timestamp" not in metrics_payload


def test_tracking_provider_none_is_noop(tmp_path: Path) -> None:
    config = ExperimentConfig(name="tracking-none", output_dir=tmp_path / "out")
    result = build_small_result()
    result.config = config
    metadata = collect_run_metadata(config)

    run_id = track_experiment_run(result, metadata, {})

    assert run_id is None


def test_mlflow_tracking_logs_params_metrics_tags_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    config = ExperimentConfig(
        name="mlflow-test",
        description="compare hybrid retrieval",
        output_dir=tmp_path / "out",
        tracking_provider="mlflow",
        tracking_uri=tmp_path.as_uri(),
        tracking_experiment_name="feedback-agent-tests",
    )
    result = build_small_result()
    result.config = config
    metadata = collect_run_metadata(config)
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")

    run_id = track_experiment_run(result, metadata, {"metrics.json": artifact})

    assert run_id == "run-123"
    assert fake_mlflow.tracking_uri == tmp_path.as_uri()  # type: ignore[attr-defined]
    assert fake_mlflow.experiment_name == "feedback-agent-tests"  # type: ignore[attr-defined]
    assert fake_mlflow.run_names == ["mlflow-test"]  # type: ignore[attr-defined]
    assert fake_mlflow.params["retriever_type"] == "dense"  # type: ignore[attr-defined]
    assert fake_mlflow.metrics["answer_groundedness"] == 0.8  # type: ignore[attr-defined]
    assert fake_mlflow.tags["description"] == "compare hybrid retrieval"  # type: ignore[attr-defined]
    assert fake_mlflow.artifacts == [str(artifact)]  # type: ignore[attr-defined]


def test_mlflow_tracking_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    result = build_small_result()
    result.config = ExperimentConfig(tracking_provider="mlflow")
    metadata = collect_run_metadata(result.config)

    with pytest.raises(RuntimeError, match="poetry install --extras mlflow"):
        track_experiment_run(result, metadata, {})


def test_experiment_run_cli_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "run-a"
    config_path = write_config_yaml(tmp_path / "config.yaml", output_dir)
    result = runner.invoke(app, ["experiment", "run", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    for filename in ("results.json", "metrics.json", "run_metadata.json"):
        assert (output_dir / filename).exists()
    # Aggregate metrics are echoed for scripting.
    assert '"total_cases": 5' in result.output

    # A second run with the same config produces identical deterministic outputs.
    output_dir_b = tmp_path / "run-b"
    config_path_b = write_config_yaml(tmp_path / "config_b.yaml", output_dir_b)
    result_b = runner.invoke(app, ["experiment", "run", "--config", str(config_path_b)])
    assert result_b.exit_code == 0, result_b.output
    assert (output_dir / "metrics.json").read_text(encoding="utf-8") == (
        output_dir_b / "metrics.json"
    ).read_text(encoding="utf-8")


def test_experiment_run_cli_logs_to_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _make_fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    output_dir = tmp_path / "tracked"
    config_path = write_config_yaml(
        tmp_path / "tracked.yaml",
        output_dir,
        tracking_provider="mlflow",
        tracking_experiment_name="cli-tests",
    )

    result = runner.invoke(app, ["experiment", "run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "tracking run id: run-123" in result.output
    assert fake_mlflow.experiment_name == "cli-tests"  # type: ignore[attr-defined]


def test_experiment_run_cli_fails_on_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("chunk_size: 10\nchunk_overlap: 99\n", encoding="utf-8")
    result = runner.invoke(app, ["experiment", "run", "--config", str(config_path)])
    assert result.exit_code != 0


def _make_fake_mlflow() -> ModuleType:
    """Build a minimal fake MLflow fluent module for tracking tests."""

    class RunInfo:
        run_id = "run-123"

    class Run:
        info = RunInfo()

        def __enter__(self) -> Run:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

    module = type(sys)("mlflow")
    module.tracking_uri = None  # type: ignore[attr-defined]
    module.experiment_name = None  # type: ignore[attr-defined]
    module.run_names = []  # type: ignore[attr-defined]
    module.params = {}  # type: ignore[attr-defined]
    module.metrics = {}  # type: ignore[attr-defined]
    module.tags = {}  # type: ignore[attr-defined]
    module.artifacts = []  # type: ignore[attr-defined]

    def set_tracking_uri(uri: str) -> None:
        module.tracking_uri = uri  # type: ignore[attr-defined]

    def set_experiment(name: str) -> None:
        module.experiment_name = name  # type: ignore[attr-defined]

    def start_run(*, run_name: str) -> Run:
        module.run_names.append(run_name)  # type: ignore[attr-defined]
        return Run()

    def log_params(params: dict[str, object]) -> None:
        module.params.update(params)  # type: ignore[attr-defined]

    def log_metrics(metrics: dict[str, float]) -> None:
        module.metrics.update(metrics)  # type: ignore[attr-defined]

    def set_tag(key: str, value: str) -> None:
        module.tags[key] = value  # type: ignore[attr-defined]

    def log_artifact(path: str) -> None:
        module.artifacts.append(path)  # type: ignore[attr-defined]

    module.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    module.set_experiment = set_experiment  # type: ignore[attr-defined]
    module.start_run = start_run  # type: ignore[attr-defined]
    module.log_params = log_params  # type: ignore[attr-defined]
    module.log_metrics = log_metrics  # type: ignore[attr-defined]
    module.set_tag = set_tag  # type: ignore[attr-defined]
    module.log_artifact = log_artifact  # type: ignore[attr-defined]
    return module
