"""Data contracts and dataset validation for customer feedback CSV files.

This module defines the expected schema for ingested feedback datasets and
validates CSV input before it reaches chunking and indexing. Validation
produces a structured report with clear, row-level errors for missing
columns, empty text, duplicate identifiers, and invalid timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ai_engineering_showcase.schemas import FeedbackChannel, FeedbackRecord

REQUIRED_COLUMNS: tuple[str, ...] = (
    "feedback_id",
    "customer_segment",
    "channel",
    "rating",
    "text",
    "created_at",
)

OPTIONAL_COLUMNS: tuple[str, ...] = ("sentiment", "label")

KNOWN_SENTIMENTS: frozenset[str] = frozenset({"positive", "negative", "neutral", "mixed"})

_FIELD_ERROR_MESSAGES: dict[str, str] = {
    "feedback_id": "feedback_id must be a non-empty string",
    "customer_segment": "customer_segment must be a non-empty string",
    "channel": "channel must be one of: "
    + ", ".join(sorted(channel.value for channel in FeedbackChannel)),
    "rating": "rating must be an integer between 1 and 5",
    "text": "text must contain at least 3 characters of feedback",
    "created_at": "created_at must be a valid ISO 8601 timestamp",
}


class ValidationIssue(BaseModel):
    """Single validation finding for a feedback dataset."""

    severity: Literal["error", "warning"]
    row: int | None = Field(
        default=None,
        description="1-based CSV line number (header is line 1); None for dataset-level issues.",
    )
    column: str | None = None
    message: str

    def render(self) -> str:
        """Format the issue as a human-readable line."""
        location = f"row {self.row}" if self.row is not None else "dataset"
        column = f", column '{self.column}'" if self.column else ""
        return f"{location}{column}: {self.message}"


class ValidationReport(BaseModel):
    """Summary of validating a dataset against the feedback data contract."""

    source: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the dataset passed validation without any errors."""
        return not self.errors

    def summary(self, *, max_issues: int = 10) -> str:
        """Build a human-readable summary of the validation outcome."""
        lines = [
            f"{self.source}: {self.valid_rows}/{self.total_rows} rows valid "
            f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        ]
        lines.extend(f"- ERROR {issue.render()}" for issue in self.errors[:max_issues])
        if len(self.errors) > max_issues:
            lines.append(f"- ... and {len(self.errors) - max_issues} more error(s)")
        lines.extend(f"- WARNING {issue.render()}" for issue in self.warnings[:max_issues])
        return "\n".join(lines)


class DataContractError(ValueError):
    """Raised in strict mode when a dataset violates the feedback data contract."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.summary())
        self.report = report


def _row_issues(payload: dict[str, Any], line_number: int) -> list[ValidationIssue]:
    """Validate a single row payload and translate failures into clear issues."""
    issues: list[ValidationIssue] = []
    try:
        FeedbackRecord.model_validate(payload)
    except ValidationError as exc:
        seen_fields: set[str] = set()
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else "row"
            if field in seen_fields:
                continue
            seen_fields.add(field)
            message = _FIELD_ERROR_MESSAGES.get(field, str(error["msg"]))
            if field == "text" and not str(payload.get("text", "")).strip():
                message = "text is empty"
            issues.append(
                ValidationIssue(severity="error", row=line_number, column=field, message=message)
            )
    return issues


def validate_feedback_csv(
    path: str | Path, *, strict: bool = False
) -> tuple[ValidationReport, list[FeedbackRecord]]:
    """Validate a feedback CSV file against the data contract.

    Args:
        path: Path to the CSV file to validate.
        strict: When True, raise :class:`DataContractError` if any error is found.
            When False, invalid rows are skipped and reported.

    Returns:
        A tuple of the validation report and the feedback records that passed
        validation (all rows in a clean dataset, the valid subset otherwise).

    Raises:
        FileNotFoundError: If the file does not exist.
        DataContractError: In strict mode, if the dataset has any validation error.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    records: list[FeedbackRecord] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    for column in missing_columns:
        errors.append(
            ValidationIssue(
                severity="error", column=column, message=f"required column '{column}' is missing"
            )
        )

    known_columns = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
    for column in frame.columns:
        if str(column) not in known_columns:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    column=str(column),
                    message=f"unexpected column '{column}' is ignored",
                )
            )

    total_rows = len(frame)
    if total_rows == 0:
        warnings.append(ValidationIssue(severity="warning", message="dataset contains no rows"))

    if missing_columns:
        report = ValidationReport(
            source=str(csv_path),
            total_rows=total_rows,
            valid_rows=0,
            invalid_rows=total_rows,
            errors=errors,
            warnings=warnings,
        )
        if strict:
            raise DataContractError(report)
        return report, []

    seen_ids: set[str] = set()
    invalid_rows = 0
    raw_rows: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()} for row in frame.to_dict(orient="records")
    ]

    for position, raw_row in enumerate(raw_rows):
        line_number = position + 2  # 1-based, accounting for the header line.
        payload = {column: raw_row.get(column) for column in REQUIRED_COLUMNS}
        row_issues = _row_issues(payload, line_number)

        feedback_id = str(raw_row.get("feedback_id", "")).strip()
        if feedback_id and feedback_id in seen_ids:
            row_issues.append(
                ValidationIssue(
                    severity="error",
                    row=line_number,
                    column="feedback_id",
                    message=f"duplicate feedback_id '{feedback_id}'",
                )
            )
        if feedback_id:
            seen_ids.add(feedback_id)

        sentiment = str(raw_row.get("sentiment", "")).strip().lower()
        if sentiment and sentiment not in KNOWN_SENTIMENTS:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    row=line_number,
                    column="sentiment",
                    message=f"unknown sentiment value '{sentiment}'",
                )
            )

        if row_issues:
            errors.extend(row_issues)
            invalid_rows += 1
        else:
            records.append(FeedbackRecord.model_validate(payload))

    report = ValidationReport(
        source=str(csv_path),
        total_rows=total_rows,
        valid_rows=total_rows - invalid_rows,
        invalid_rows=invalid_rows,
        errors=errors,
        warnings=warnings,
    )
    if strict and errors:
        raise DataContractError(report)
    return report, records
