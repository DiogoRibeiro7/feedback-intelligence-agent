from __future__ import annotations

import pytest

from feedback_intelligence_agent.chunking import chunk_text, feedback_to_chunks
from feedback_intelligence_agent.schemas import FeedbackRecord


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    chunks = chunk_text("one two three", max_words=5, overlap_words=1)
    assert chunks == ["one two three"]


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("one two three four five six", max_words=3, overlap_words=1)
    assert chunks == ["one two three", "three four five", "five six"]


def test_chunk_text_validates_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_words must be smaller"):
        chunk_text("one two", max_words=2, overlap_words=2)


def test_feedback_to_chunks_preserves_metadata() -> None:
    record = FeedbackRecord.model_validate(
        {
            "feedback_id": "fb-test",
            "customer_segment": "enterprise",
            "channel": "support_ticket",
            "rating": 2,
            "text": "Onboarding was unclear and slow.",
            "created_at": "2026-01-01T10:00:00",
        }
    )
    chunks = feedback_to_chunks([record])
    assert len(chunks) == 1
    assert chunks[0].source_id == "fb-test"
    assert chunks[0].metadata["customer_segment"] == "enterprise"


def test_feedback_to_chunks_redacts_pii_before_storage() -> None:
    record = FeedbackRecord.model_validate(
        {
            "feedback_id": "fb-pii",
            "customer_segment": "enterprise",
            "channel": "support_ticket",
            "rating": 2,
            "text": "Contact owner@example.com or +1 555-123-4567 about setup.",
            "created_at": "2026-01-01T10:00:00",
        }
    )

    chunks = feedback_to_chunks([record])

    assert "owner@example.com" not in chunks[0].text
    assert "555-123-4567" not in chunks[0].text
    assert "[REDACTED_EMAIL]" in chunks[0].text
    assert "[REDACTED_PHONE]" in chunks[0].text
