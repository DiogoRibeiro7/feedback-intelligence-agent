from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.human_feedback import (
    HumanFeedbackRating,
    InMemoryHumanFeedbackStore,
    JsonHumanFeedbackStore,
    SubmitHumanFeedbackRequest,
)
from feedback_intelligence_agent.schemas import AgentAnswer


def make_answer(question: str = "Why is onboarding slow?") -> AgentAnswer:
    return AgentAnswer(
        question=question,
        answer="Onboarding is slow because setup ownership is unclear [1].",
        recommended_actions=["Create a clearer onboarding checklist."],
        citations=[],
        route="onboarding",
        confidence=0.72,
        diagnostics={"retrieved_chunks": 1},
    )


def test_in_memory_human_feedback_store_saves_and_lists_summaries() -> None:
    store = InMemoryHumanFeedbackStore()

    record = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.useful,
            comment=" Grounded and actionable. ",
            report_id="report-1",
            tags=[" Enterprise ", "enterprise", "Onboarding"],
        )
    )

    assert store.get(record.feedback_id) == record
    assert record.comment == "Grounded and actionable."
    assert record.tags == ["enterprise", "onboarding"]
    summaries = store.list()
    assert len(summaries) == 1
    assert summaries[0].feedback_id == record.feedback_id
    assert summaries[0].rating == HumanFeedbackRating.useful
    assert summaries[0].has_comment is True
    assert summaries[0].report_id == "report-1"


def test_json_human_feedback_store_persists_record_files(tmp_path: Path) -> None:
    store = JsonHumanFeedbackStore(tmp_path)

    record = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.not_useful,
            comment="Missed the integration issue.",
        )
    )
    reloaded = JsonHumanFeedbackStore(tmp_path).get(record.feedback_id)

    assert reloaded == record
    assert (tmp_path / f"{record.feedback_id}.json").exists()
    assert JsonHumanFeedbackStore(tmp_path).list()[0].rating == HumanFeedbackRating.not_useful


def test_json_human_feedback_store_rejects_unsafe_ids(tmp_path: Path) -> None:
    store = JsonHumanFeedbackStore(tmp_path)

    with pytest.raises(ValueError, match="invalid feedback_id"):
        store.get("../bad")


def test_human_feedback_request_rejects_unsafe_report_ids() -> None:
    with pytest.raises(ValueError, match="invalid report_id"):
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.useful,
            report_id="../bad",
        )
