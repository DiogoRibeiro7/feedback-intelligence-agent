"""Feedback ingestion utilities."""

from __future__ import annotations

from pathlib import Path

from ai_engineering_showcase.data_contracts import (
    REQUIRED_COLUMNS,
    DataContractError,
    validate_feedback_csv,
)
from ai_engineering_showcase.schemas import FeedbackRecord

__all__ = ["REQUIRED_COLUMNS", "FeedbackIngestionError", "load_feedback_csv"]


class FeedbackIngestionError(ValueError):
    """Raised when input feedback data cannot be loaded safely."""


def load_feedback_csv(path: str | Path, *, strict: bool = True) -> list[FeedbackRecord]:
    """Load feedback records from a CSV file, validating the data contract first.

    Args:
        path: Path to a CSV file containing the required feedback columns.
        strict: When True (default), any contract violation aborts ingestion.
            When False, invalid rows are skipped and the valid rows are returned.

    Returns:
        Validated feedback records.

    Raises:
        FileNotFoundError: If the file does not exist.
        FeedbackIngestionError: In strict mode, if required columns are missing
            or any row violates the data contract.
    """
    try:
        _, records = validate_feedback_csv(path, strict=strict)
    except DataContractError as exc:
        raise FeedbackIngestionError(str(exc)) from exc
    return records
