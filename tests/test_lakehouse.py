from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from feedback_intelligence_agent.lakehouse import (
    LakehouseTableFormat,
    export_feedback_lakehouse,
)
from feedback_intelligence_agent.schemas import FeedbackChannel, FeedbackRecord
from feedback_intelligence_agent.telemetry import InMemoryTelemetrySink, Telemetry


def make_record(
    feedback_id: str,
    *,
    text: str = "Onboarding update",
    channel: FeedbackChannel = FeedbackChannel.support_ticket,
) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=feedback_id,
        customer_segment="enterprise",
        channel=channel,
        rating=2,
        text=text,
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def test_export_feedback_lakehouse_writes_delta_style_table(tmp_path: Path) -> None:
    result = export_feedback_lakehouse(
        [
            make_record("fb-2", channel=FeedbackChannel.nps_survey),
            make_record("fb-1", text="Contact owner@example.com with token=abc123def456ghi789."),
        ],
        tmp_path / "feedback",
        table_format=LakehouseTableFormat.delta,
    )

    assert result.records == 2
    assert result.partitions == 2
    assert result.table_format == LakehouseTableFormat.delta
    assert Path(result.manifest_path).exists()
    assert Path(result.metadata_path).name == "00000000000000000000.json"

    data_text = "\n".join(
        Path(data_file.path).read_text(encoding="utf-8") for data_file in result.data_files
    )
    assert "owner@example.com" not in data_text
    assert "abc123def456ghi789" not in data_text
    assert "[REDACTED_EMAIL]" in data_text
    assert "[REDACTED_TOKEN]" in data_text

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["table_format"] == "delta"
    assert manifest["partition_columns"] == ["created_date", "channel"]

    delta_actions = Path(result.metadata_path).read_text(encoding="utf-8").splitlines()
    assert json.loads(delta_actions[0])["protocol"]["minReaderVersion"] == 1
    assert any("add" in json.loads(line) for line in delta_actions)


def test_export_feedback_lakehouse_writes_iceberg_style_metadata(tmp_path: Path) -> None:
    result = export_feedback_lakehouse(
        [make_record("fb-1")],
        tmp_path / "feedback",
        table_format=LakehouseTableFormat.iceberg,
        partition_columns=("created_month",),
    )

    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))

    assert result.table_format == LakehouseTableFormat.iceberg
    assert Path(result.metadata_path).name == "v1.metadata.json"
    assert metadata["format-version"] == 2
    assert metadata["partition-specs"][0]["fields"][0]["name"] == "created_month"


def test_export_feedback_lakehouse_rejects_unknown_partitions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown partition column"):
        export_feedback_lakehouse(
            [make_record("fb-1")],
            tmp_path / "feedback",
            partition_columns=("account_id",),
        )


def test_export_feedback_lakehouse_emits_telemetry(tmp_path: Path) -> None:
    sink = InMemoryTelemetrySink()

    result = export_feedback_lakehouse(
        [make_record("fb-1")],
        tmp_path / "feedback",
        telemetry=Telemetry(sink=sink),
    )

    assert result.records == 1
    assert sink.event_names() == ["lakehouse_export_started", "lakehouse_export_finished"]
    assert sink.events[-1].metadata["data_files"] == 1
