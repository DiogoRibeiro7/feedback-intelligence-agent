from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.human_feedback import (
    HumanFeedbackRating,
    InMemoryHumanFeedbackStore,
    JsonHumanFeedbackStore,
    SubmitHumanFeedbackRequest,
    build_active_learning_queue,
    summarise_human_feedback,
)
from feedback_intelligence_agent.schemas import AgentAnswer


def make_answer(
    question: str = "Why is onboarding slow?",
    *,
    confidence: float = 0.72,
    route: str = "onboarding",
) -> AgentAnswer:
    return AgentAnswer(
        question=question,
        answer="Onboarding is slow because setup ownership is unclear [1].",
        recommended_actions=["Create a clearer onboarding checklist."],
        citations=[],
        route=route,
        confidence=confidence,
        diagnostics={"retrieved_chunks": 1},
    )


def test_in_memory_human_feedback_store_saves_and_lists_summaries() -> None:
    store = InMemoryHumanFeedbackStore()

    record = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.useful,
            tenant_id="Acme",
            comment=" Grounded and actionable. ",
            report_id="report-1",
            tags=[" Enterprise ", "enterprise", "Onboarding"],
        )
    )

    assert store.get(record.feedback_id) == record
    assert record.comment == "Grounded and actionable."
    assert record.tenant_id == "acme"
    assert record.tags == ["enterprise", "onboarding"]
    summaries = store.list()
    assert len(summaries) == 1
    assert summaries[0].feedback_id == record.feedback_id
    assert summaries[0].rating == HumanFeedbackRating.useful
    assert summaries[0].tenant_id == "acme"
    assert summaries[0].has_comment is True
    assert summaries[0].report_id == "report-1"
    assert store.list(tenant_id="acme")[0].feedback_id == record.feedback_id
    assert store.list(tenant_id="cobalt") == []


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


def test_summarise_human_feedback_groups_by_route_and_tenant() -> None:
    store = InMemoryHumanFeedbackStore()
    store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.useful,
            tenant_id="acme",
            comment="Useful.",
        )
    )
    store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.not_useful,
            tenant_id="acme",
        )
    )
    store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer("Which integrations failed?").model_copy(
                update={"route": "integrations"}
            ),
            rating=HumanFeedbackRating.useful,
            tenant_id="cobalt",
        )
    )

    analytics = summarise_human_feedback(store.list())

    assert analytics.total == 3
    assert analytics.useful == 2
    assert analytics.not_useful == 1
    assert analytics.useful_rate == 0.667
    assert analytics.with_comments == 1
    assert [(item.key, item.total, item.useful_rate) for item in analytics.by_route] == [
        ("onboarding", 2, 0.5),
        ("integrations", 1, 1.0),
    ]
    assert [(item.key, item.total, item.useful_rate) for item in analytics.by_tenant] == [
        ("acme", 2, 0.5),
        ("cobalt", 1, 1.0),
    ]


def test_build_active_learning_queue_prioritises_not_useful_and_low_confidence() -> None:
    store = InMemoryHumanFeedbackStore()
    useful_high = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer("What worked?", confidence=0.91),
            rating=HumanFeedbackRating.useful,
            tenant_id="acme",
        )
    )
    useful_low = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer("What was uncertain?", confidence=0.42),
            rating=HumanFeedbackRating.useful,
            tenant_id="acme",
            report_id="report-1",
        )
    )
    not_useful = store.save(
        SubmitHumanFeedbackRequest(
            result=make_answer("What was wrong?", confidence=0.81, route="pricing"),
            rating=HumanFeedbackRating.not_useful,
            tenant_id="acme",
            comment="Missed the renewal issue.",
            tags=["Pricing"],
        )
    )

    queue = build_active_learning_queue(store.list(), low_confidence_threshold=0.65)

    assert queue.total_candidates == 2
    assert [item.feedback_id for item in queue.items] == [
        not_useful.feedback_id,
        useful_low.feedback_id,
    ]
    assert useful_high.feedback_id not in {item.feedback_id for item in queue.items}
    assert queue.items[0].reasons == ["not_useful"]
    assert queue.items[0].priority_score == 1.1
    assert queue.items[0].route == "pricing"
    assert queue.items[0].has_comment is True
    assert queue.items[0].tags == ["pricing"]
    assert queue.items[1].reasons == ["low_confidence"]
    assert queue.items[1].report_id == "report-1"


def test_build_active_learning_queue_validates_options() -> None:
    with pytest.raises(ValueError, match="max_items"):
        build_active_learning_queue([], max_items=0)
    with pytest.raises(ValueError, match="low_confidence_threshold"):
        build_active_learning_queue([], low_confidence_threshold=1.1)


def test_human_feedback_request_rejects_unsafe_report_ids() -> None:
    with pytest.raises(ValueError, match="invalid report_id"):
        SubmitHumanFeedbackRequest(
            result=make_answer(),
            rating=HumanFeedbackRating.useful,
            report_id="../bad",
        )
