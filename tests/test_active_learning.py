from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.active_learning import (
    ActiveLearningStatus,
    InMemoryActiveLearningStateStore,
    JsonActiveLearningStateStore,
    UpdateActiveLearningStateRequest,
    apply_active_learning_states,
)
from feedback_intelligence_agent.human_feedback import (
    HumanFeedbackRating,
    HumanFeedbackSummary,
    build_active_learning_queue,
)


def make_summary(
    feedback_id: str = "feedback-1",
    *,
    confidence: float = 0.4,
    rating: HumanFeedbackRating = HumanFeedbackRating.not_useful,
) -> HumanFeedbackSummary:
    return HumanFeedbackSummary(
        feedback_id=feedback_id,
        question="Why is onboarding slow?",
        rating=rating,
        created_at="2026-08-06T09:00:00Z",
        tenant_id="acme",
        route="onboarding",
        confidence=confidence,
        report_id="report-1",
        has_comment=True,
        tags=["onboarding"],
    )


def test_active_learning_state_store_updates_and_lists() -> None:
    store = InMemoryActiveLearningStateStore()

    state = store.update(
        "feedback-1",
        UpdateActiveLearningStateRequest(
            status=ActiveLearningStatus.assigned,
            assignee=" Diogo Ribeiro ",
            notes=" Review retrieval coverage. ",
        ),
    )

    assert store.get("feedback-1") == state
    assert state.status == ActiveLearningStatus.assigned
    assert state.assignee == "Diogo Ribeiro"
    assert state.notes == "Review retrieval coverage."
    assert store.list(status=ActiveLearningStatus.assigned)[0].feedback_id == "feedback-1"
    assert store.list(status=ActiveLearningStatus.resolved) == []


def test_json_active_learning_state_store_persists_files(tmp_path: Path) -> None:
    store = JsonActiveLearningStateStore(tmp_path)

    state = store.update(
        "feedback-1",
        UpdateActiveLearningStateRequest(status=ActiveLearningStatus.in_progress),
    )
    reloaded = JsonActiveLearningStateStore(tmp_path).get("feedback-1")

    assert reloaded == state
    assert (tmp_path / "feedback-1.json").exists()


def test_active_learning_state_store_rejects_unsafe_ids(tmp_path: Path) -> None:
    store = JsonActiveLearningStateStore(tmp_path)

    with pytest.raises(ValueError, match="invalid feedback_id"):
        store.get("../bad")


def test_apply_active_learning_states_overlays_queue_items() -> None:
    store = InMemoryActiveLearningStateStore()
    store.update(
        "feedback-1",
        UpdateActiveLearningStateRequest(
            status=ActiveLearningStatus.resolved,
            assignee="diogo",
            notes="Added a retrieval regression.",
        ),
    )
    queue = build_active_learning_queue([make_summary()])

    enriched = apply_active_learning_states(queue, store.list())

    assert enriched.items[0].workflow_status == "resolved"
    assert enriched.items[0].assignee == "diogo"
    assert enriched.items[0].state_notes == "Added a retrieval regression."
    assert enriched.items[0].state_updated_at is not None
