from __future__ import annotations

from email.message import EmailMessage

import pytest

from feedback_intelligence_agent.email_summaries import (
    EmailSummaryRequest,
    deliver_email_summary,
    render_email_summary,
    select_reports_for_summary,
)
from feedback_intelligence_agent.reports import (
    InMemoryInsightReportStore,
    SaveInsightReportRequest,
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


def test_render_email_summary_includes_report_details() -> None:
    store = InMemoryInsightReportStore()
    report = store.save(
        SaveInsightReportRequest(
            title="Onboarding report",
            result=make_answer(),
            tags=["onboarding"],
        )
    )

    summary = render_email_summary(
        [report],
        recipients=["pm@example.com"],
    )

    assert summary.subject == "Feedback insight: Onboarding report"
    assert summary.recipients == ["pm@example.com"]
    assert summary.report_ids == [report.report_id]
    assert "Question: Why is onboarding slow?" in summary.body_text
    assert "Tenant: default" in summary.body_text
    assert "Recommended actions:" in summary.body_text
    assert "Tags: onboarding" in summary.body_text


def test_select_reports_for_summary_uses_latest_when_ids_are_omitted() -> None:
    store = InMemoryInsightReportStore()
    first = store.save(SaveInsightReportRequest(title="First", result=make_answer("First?")))
    second = store.save(SaveInsightReportRequest(title="Second", result=make_answer("Second?")))

    selected = select_reports_for_summary(store, report_ids=[], max_reports=1)

    assert len(selected) == 1
    assert selected[0].report_id in {first.report_id, second.report_id}


def test_select_reports_for_summary_reports_missing_ids() -> None:
    store = InMemoryInsightReportStore()

    with pytest.raises(LookupError, match="report not found"):
        select_reports_for_summary(store, report_ids=["missing"], max_reports=5)


def test_select_reports_for_summary_filters_by_tenant() -> None:
    store = InMemoryInsightReportStore()
    acme = store.save(
        SaveInsightReportRequest(title="Acme", result=make_answer("Acme?"), tenant_id="acme")
    )
    store.save(
        SaveInsightReportRequest(title="Cobalt", result=make_answer("Cobalt?"), tenant_id="cobalt")
    )

    selected = select_reports_for_summary(
        store,
        report_ids=[],
        max_reports=5,
        tenant_id="acme",
    )

    assert [report.report_id for report in selected] == [acme.report_id]


def test_email_summary_request_validates_recipients() -> None:
    with pytest.raises(ValueError, match="invalid email recipient"):
        EmailSummaryRequest(recipients=["not-an-email"])


def test_deliver_email_summary_sends_with_fake_smtp() -> None:
    summary = render_email_summary(
        [
            InMemoryInsightReportStore().save(
                SaveInsightReportRequest(title="One", result=make_answer())
            )
        ],
        recipients=["pm@example.com"],
    )
    fake = FakeSmtpFactory()

    result = deliver_email_summary(
        summary,
        smtp_host="smtp.example.com",
        smtp_port=587,
        from_address="agent@example.com",
        username="agent",
        password="secret",
        use_tls=True,
        smtp_factory=fake,
    )

    assert result.sent is True
    assert fake.instance is not None
    assert fake.instance.started_tls is True
    assert fake.instance.login_args == ("agent", "secret")
    assert fake.instance.message is not None
    assert fake.instance.message["To"] == "pm@example.com"


class FakeSmtpFactory:
    def __init__(self) -> None:
        self.instance: FakeSmtp | None = None

    def __call__(self, host: str, port: int) -> FakeSmtp:
        self.instance = FakeSmtp(host, port)
        return self.instance


class FakeSmtp:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.started_tls = False
        self.login_args: tuple[str, str] | None = None
        self.message: EmailMessage | None = None

    def __enter__(self) -> FakeSmtp:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def starttls(self) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.login_args = (username, password)

    def send_message(self, message: EmailMessage) -> None:
        self.message = message
