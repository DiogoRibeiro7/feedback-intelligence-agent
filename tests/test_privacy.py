from __future__ import annotations

from feedback_intelligence_agent.privacy import redact_feedback_record, redact_pii
from feedback_intelligence_agent.schemas import FeedbackRecord


def test_redact_pii_replaces_email_addresses() -> None:
    text = redact_pii("Contact maria.santos@example.com for onboarding follow-up.")

    assert "maria.santos@example.com" not in text
    assert "[REDACTED_EMAIL]" in text


def test_redact_pii_replaces_phone_numbers() -> None:
    text = redact_pii("Call +1 (555) 123-4567 before the renewal meeting.")

    assert "555" not in text
    assert "[REDACTED_PHONE]" in text


def test_redact_pii_replaces_obvious_access_tokens() -> None:
    text = redact_pii("The customer pasted api_key=abc123def456ghi789 into the ticket.")

    assert "abc123def456ghi789" not in text
    assert "[REDACTED_TOKEN]" in text


def test_redact_feedback_record_keeps_metadata_and_redacts_text() -> None:
    record = FeedbackRecord.model_validate(
        {
            "feedback_id": "privacy-1",
            "customer_segment": "enterprise",
            "channel": "support_ticket",
            "rating": 2,
            "text": "Email owner@example.com about the setup issue.",
            "created_at": "2026-08-01T09:00:00Z",
        }
    )

    redacted = redact_feedback_record(record)

    assert redacted.feedback_id == "privacy-1"
    assert redacted.customer_segment == "enterprise"
    assert "owner@example.com" not in redacted.text
    assert "[REDACTED_EMAIL]" in redacted.text
