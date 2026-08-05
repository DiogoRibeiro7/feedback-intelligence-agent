"""Deterministic PII redaction helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from feedback_intelligence_agent.schemas import FeedbackRecord

__all__ = [
    "EMAIL_PLACEHOLDER",
    "PHONE_PLACEHOLDER",
    "TOKEN_PLACEHOLDER",
    "redact_feedback_record",
    "redact_feedback_records",
    "redact_pii",
]

EMAIL_PLACEHOLDER = "[REDACTED_EMAIL]"
PHONE_PLACEHOLDER = "[REDACTED_PHONE]"
TOKEN_PLACEHOLDER = "[REDACTED_TOKEN]"

_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
_TOKEN_PATTERNS = [
    re.compile(r"\b(?:sk|pk|ghp|github_pat|xoxb|xoxp|xoxa|xapp|AKIA|ASIA)[-_A-Za-z0-9]{12,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|token)"
        r"\s*[:=]\s*['\"]?[-_A-Za-z0-9./+=]{12,}['\"]?"
    ),
    re.compile(r"(?i)\bbearer\s+[-_A-Za-z0-9./+=]{12,}\b"),
]


def redact_pii(text: str) -> str:
    """Redact common PII and obvious credentials from free text."""
    redacted = _EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)
    redacted = _PHONE_PATTERN.sub(PHONE_PLACEHOLDER, redacted)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(TOKEN_PLACEHOLDER, redacted)
    return " ".join(redacted.split())


def redact_feedback_record(record: FeedbackRecord) -> FeedbackRecord:
    """Return a copy of a feedback record with redacted free-text content."""
    redacted_text = redact_pii(record.text)
    if redacted_text == record.text:
        return record
    return record.model_copy(update={"text": redacted_text})


def redact_feedback_records(records: Iterable[FeedbackRecord]) -> list[FeedbackRecord]:
    """Return feedback records with redacted free-text content."""
    return [redact_feedback_record(record) for record in records]
