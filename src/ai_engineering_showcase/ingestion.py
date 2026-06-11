"""Feedback ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_engineering_showcase.schemas import FeedbackRecord

REQUIRED_COLUMNS = {
    "feedback_id",
    "customer_segment",
    "channel",
    "rating",
    "text",
    "created_at",
}


class FeedbackIngestionError(ValueError):
    """Raised when input feedback data cannot be loaded safely."""


def load_feedback_csv(path: str | Path) -> list[FeedbackRecord]:
    """Load feedback records from a CSV file.

    Args:
        path: Path to a CSV file containing the required feedback columns.

    Returns:
        Validated feedback records.

    Raises:
        FileNotFoundError: If the file does not exist.
        FeedbackIngestionError: If required columns are missing or rows are invalid.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise FeedbackIngestionError(f"Missing required columns: {missing}")

    records: list[FeedbackRecord] = []
    errors: list[str] = []

    for position, (_, row) in enumerate(frame.iterrows()):
        payload = row.to_dict()
        try:
            records.append(FeedbackRecord.model_validate(payload))
        except Exception as exc:  # noqa: BLE001 - aggregate validation details for the caller.
            errors.append(f"row={position + 2}: {exc}")

    if errors:
        joined_errors = "\n".join(errors[:10])
        raise FeedbackIngestionError(f"Invalid feedback rows:\n{joined_errors}")

    return records
