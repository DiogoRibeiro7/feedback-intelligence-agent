from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_engineering_showcase.cli import app
from ai_engineering_showcase.data_contracts import (
    DataContractError,
    ValidationReport,
    validate_feedback_csv,
)
from ai_engineering_showcase.ingestion import FeedbackIngestionError, load_feedback_csv

HEADER = "feedback_id,customer_segment,channel,rating,text,created_at"

VALID_ROWS = [
    'fb-001,enterprise,support_ticket,2,"Onboarding took too long.",2026-01-08T09:20:00',
    'fb-002,startup,nps_survey,5,"Setup was fast and easy.",2026-01-20T11:00:00',
]


def write_csv(path: Path, lines: list[str]) -> Path:
    csv_path = path / "feedback.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def test_valid_file_passes(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [HEADER, *VALID_ROWS])
    report, records = validate_feedback_csv(csv_path, strict=True)
    assert report.is_valid
    assert report.total_rows == 2
    assert report.valid_rows == 2
    assert report.invalid_rows == 0
    assert not report.errors
    assert [record.feedback_id for record in records] == ["fb-001", "fb-002"]


def test_missing_columns_reported(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        ["feedback_id,text", 'fb-001,"Some feedback text."'],
    )
    report, records = validate_feedback_csv(csv_path)
    assert not report.is_valid
    assert records == []
    assert report.valid_rows == 0
    missing = {issue.column for issue in report.errors}
    assert missing == {"customer_segment", "channel", "rating", "created_at"}


def test_empty_text_reported(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER, VALID_ROWS[0], "fb-003,startup,app_review,4,,2026-02-01T10:00:00"],
    )
    report, records = validate_feedback_csv(csv_path)
    assert report.invalid_rows == 1
    assert len(records) == 1
    [issue] = report.errors
    assert issue.column == "text"
    assert issue.row == 3
    assert issue.message == "text is empty"


def test_duplicate_ids_reported(tmp_path: Path) -> None:
    duplicate = 'fb-001,startup,app_review,4,"Another row with a reused id.",2026-02-01T10:00:00'
    csv_path = write_csv(tmp_path, [HEADER, *VALID_ROWS, duplicate])
    report, records = validate_feedback_csv(csv_path)
    assert report.invalid_rows == 1
    assert len(records) == 2
    [issue] = report.errors
    assert issue.column == "feedback_id"
    assert issue.row == 4
    assert "duplicate feedback_id 'fb-001'" in issue.message


def test_invalid_timestamp_reported(tmp_path: Path) -> None:
    bad_row = 'fb-003,startup,app_review,4,"Decent product overall.",not-a-date'
    csv_path = write_csv(tmp_path, [HEADER, VALID_ROWS[0], bad_row])
    report, records = validate_feedback_csv(csv_path)
    assert report.invalid_rows == 1
    assert len(records) == 1
    [issue] = report.errors
    assert issue.column == "created_at"
    assert "timestamp" in issue.message


def test_strict_mode_raises_with_report(tmp_path: Path) -> None:
    bad_row = 'fb-003,startup,app_review,9,"Rating out of range here.",2026-02-01T10:00:00'
    csv_path = write_csv(tmp_path, [HEADER, bad_row])
    with pytest.raises(DataContractError) as excinfo:
        validate_feedback_csv(csv_path, strict=True)
    report = excinfo.value.report
    assert isinstance(report, ValidationReport)
    assert report.invalid_rows == 1
    assert "rating" in str(excinfo.value)


def test_non_strict_mode_skips_invalid_rows(tmp_path: Path) -> None:
    bad_row = "fb-003,startup,unknown_channel,4,bad channel value here,2026-02-01T10:00:00"
    csv_path = write_csv(tmp_path, [HEADER, *VALID_ROWS, bad_row])
    report, records = validate_feedback_csv(csv_path, strict=False)
    assert report.total_rows == 3
    assert report.valid_rows == 2
    assert report.invalid_rows == 1
    assert [record.feedback_id for record in records] == ["fb-001", "fb-002"]


def test_unexpected_column_is_warning_not_error(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [
            HEADER + ",internal_notes",
            VALID_ROWS[0] + ",keep this private",
        ],
    )
    report, records = validate_feedback_csv(csv_path, strict=True)
    assert report.is_valid
    assert len(records) == 1
    assert any(issue.column == "internal_notes" for issue in report.warnings)


def test_unknown_sentiment_value_is_warning(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER + ",sentiment", VALID_ROWS[0] + ",elated"],
    )
    report, _ = validate_feedback_csv(csv_path, strict=True)
    assert report.is_valid
    assert any(issue.column == "sentiment" for issue in report.warnings)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_feedback_csv(tmp_path / "missing.csv")


def test_report_summary_mentions_counts(tmp_path: Path) -> None:
    bad_row = 'fb-003,startup,app_review,4,"Valid text here.",not-a-date'
    csv_path = write_csv(tmp_path, [HEADER, VALID_ROWS[0], bad_row])
    report, _ = validate_feedback_csv(csv_path)
    summary = report.summary()
    assert "1/2 rows valid" in summary
    assert "ERROR" in summary


def test_ingestion_strict_raises(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER, "fb-003,startup,app_review,4,,2026-02-01T10:00:00"],
    )
    with pytest.raises(FeedbackIngestionError):
        load_feedback_csv(csv_path)


def test_ingestion_non_strict_skips_bad_rows(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER, *VALID_ROWS, "fb-003,startup,app_review,4,,2026-02-01T10:00:00"],
    )
    records = load_feedback_csv(csv_path, strict=False)
    assert [record.feedback_id for record in records] == ["fb-001", "fb-002"]


def test_ingestion_valid_file_loads_all_rows(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [HEADER, *VALID_ROWS])
    records = load_feedback_csv(csv_path)
    assert len(records) == 2


def test_cli_validate_data_valid_file(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [HEADER, *VALID_ROWS])
    result = CliRunner().invoke(app, ["validate-data", str(csv_path)])
    assert result.exit_code == 0
    assert '"valid_rows": 2' in result.stdout


def test_cli_validate_data_strict_fails_on_bad_rows(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER, "fb-003,startup,app_review,4,,2026-02-01T10:00:00"],
    )
    result = CliRunner().invoke(app, ["validate-data", str(csv_path), "--strict"])
    assert result.exit_code == 1
    assert '"invalid_rows": 1' in result.stdout


def test_cli_validate_data_non_strict_reports_and_succeeds(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [HEADER, VALID_ROWS[0], "fb-003,startup,app_review,4,,2026-02-01T10:00:00"],
    )
    result = CliRunner().invoke(app, ["validate-data", str(csv_path)])
    assert result.exit_code == 0
    assert '"invalid_rows": 1' in result.stdout
