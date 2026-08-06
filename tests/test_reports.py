from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.reports import (
    InMemoryInsightReportStore,
    JsonInsightReportStore,
    SaveInsightReportRequest,
    render_report_markdown,
)
from feedback_intelligence_agent.schemas import AgentAnswer, Citation


def make_answer(question: str = "Why is onboarding slow?") -> AgentAnswer:
    return AgentAnswer(
        question=question,
        answer="Onboarding is slow because setup ownership is unclear [1].",
        recommended_actions=["Create a clearer onboarding checklist."],
        citations=[
            Citation(
                citation_id=1,
                document_id="fb-001",
                chunk_id="fb-001:0",
                source="support_ticket",
                quote="Setup ownership is unclear during onboarding.",
                score=0.82,
            )
        ],
        route="onboarding",
        confidence=0.72,
        diagnostics={"retrieved_chunks": 1},
    )


def test_in_memory_report_store_saves_and_lists_summaries() -> None:
    store = InMemoryInsightReportStore()

    report = store.save(
        SaveInsightReportRequest(
            title="Onboarding report",
            result=make_answer(),
            tenant_id="Acme",
            tags=[" Enterprise ", "enterprise", "Onboarding"],
            notes="Review next planning cycle.",
        )
    )

    assert store.get(report.report_id) == report
    summaries = store.list()
    assert len(summaries) == 1
    assert summaries[0].report_id == report.report_id
    assert summaries[0].tenant_id == "acme"
    assert summaries[0].tags == ["enterprise", "onboarding"]
    assert store.list(tenant_id="acme")[0].report_id == report.report_id
    assert store.list(tenant_id="cobalt") == []


def test_json_report_store_persists_report_files(tmp_path: Path) -> None:
    store = JsonInsightReportStore(tmp_path)

    report = store.save(SaveInsightReportRequest(title="Saved answer", result=make_answer()))
    reloaded = JsonInsightReportStore(tmp_path).get(report.report_id)

    assert reloaded == report
    assert reloaded is not None
    assert reloaded.tenant_id == "default"
    assert (tmp_path / f"{report.report_id}.json").exists()
    assert JsonInsightReportStore(tmp_path).list()[0].title == "Saved answer"


def test_json_report_store_rejects_unsafe_ids(tmp_path: Path) -> None:
    store = JsonInsightReportStore(tmp_path)

    with pytest.raises(ValueError, match="invalid report_id"):
        store.get("../bad")


def test_render_report_markdown_includes_handoff_fields() -> None:
    store = InMemoryInsightReportStore()
    report = store.save(
        SaveInsightReportRequest(
            title="Onboarding report",
            result=make_answer(),
            tenant_id="Acme",
            tags=["Onboarding"],
            notes="Share with the activation squad.",
        )
    )

    markdown = render_report_markdown(report)

    assert markdown.startswith("# Onboarding report\n")
    assert f"- Report ID: `{report.report_id}`" in markdown
    assert "- Tenant: `acme`" in markdown
    assert "- Question: Why is onboarding slow?" in markdown
    assert "## Notes" in markdown
    assert "Share with the activation squad." in markdown
    assert "## Answer" in markdown
    assert "1. Create a clearer onboarding checklist." in markdown
    assert "### [1] fb-001" in markdown
    assert "> Setup ownership is unclear during onboarding." in markdown
    assert '"retrieved_chunks": 1' in markdown
