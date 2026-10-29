from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from feedback_intelligence_agent.experiments import (
    ExperimentConfig,
    collect_run_metadata,
    run_experiment,
    track_experiment_run,
    write_experiment_outputs,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FEEDBACK_AGENT_MLFLOW_TEST") != "1",
    reason="install the mlflow extra and set FEEDBACK_AGENT_MLFLOW_TEST=1 to run",
)


@pytest.mark.parametrize("backend", ["file", "sqlite", "default"])
def test_mlflow_tracking_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    # Disable upstream usage reporting; the integration uses local stores only.
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "true")
    # MLflow 3.17 requires an explicit opt-in for legacy file tracking stores.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", str(backend == "file").lower())
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    import mlflow
    from mlflow.tracking import MlflowClient

    repo_root = Path(__file__).resolve().parents[1]
    uri = {
        "file": (tmp_path / "mlruns").as_uri(),
        "sqlite": "sqlite:///tracking.db",
        "default": None,
    }[backend]
    # Reset MLflow's process-wide URI between independent backends.
    mlflow.set_tracking_uri(None)
    config = ExperimentConfig(
        name=f"integration-{backend}",
        description="Verify local experiment tracking",
        dataset_path=repo_root / "data/sample_feedback.csv",
        queries_path=repo_root / "examples/queries.jsonl",
        output_dir=tmp_path / "experiment",
        tracking_provider="mlflow",
        tracking_uri=uri,
        tracking_experiment_name=f"integration-{backend}",
    )
    result = run_experiment(config)
    metadata = collect_run_metadata(config)
    artifacts = write_experiment_outputs(result, metadata)
    try:
        run_id = track_experiment_run(result, metadata, artifacts)
        assert run_id
        client = MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.params["retriever_type"] == config.retriever_type
        assert run.data.metrics["total_cases"] == result.metrics.total_cases
        assert run.data.tags["description"] == config.description
        assert run.data.tags["package_version"] == metadata.package_version
        assert {item.path for item in client.list_artifacts(run_id)} == set(artifacts)
        artifact = client.download_artifacts(run_id, "metrics.json", str(tmp_path))
        assert json.loads(Path(artifact).read_text(encoding="utf-8")) == json.loads(
            artifacts["metrics.json"].read_text(encoding="utf-8")
        )
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(None)
