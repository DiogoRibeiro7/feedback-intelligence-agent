"""Email summaries for saved insight reports."""

from __future__ import annotations

import smtplib
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from email.message import EmailMessage

from pydantic import BaseModel, Field, field_validator

from feedback_intelligence_agent.reports import InsightReportStore, SavedInsightReport


class EmailSummaryRequest(BaseModel):
    """Request to render or send a saved-report email summary."""

    recipients: list[str] = Field(min_length=1)
    report_ids: list[str] = Field(default_factory=list)
    subject: str | None = Field(default=None, max_length=160)
    max_reports: int = Field(default=5, ge=1, le=20)
    send: bool = False

    @field_validator("recipients")
    @classmethod
    def validate_recipients(cls, value: list[str]) -> list[str]:
        """Trim and lightly validate email recipients without extra dependencies."""
        recipients: list[str] = []
        for item in value:
            email = item.strip()
            if "@" not in email or email.startswith("@") or email.endswith("@"):
                raise ValueError(f"invalid email recipient {item!r}")
            if email not in recipients:
                recipients.append(email)
        return recipients


class EmailSummary(BaseModel):
    """Rendered email digest for saved insight reports."""

    subject: str
    body_text: str
    recipients: list[str]
    report_ids: list[str]
    report_count: int = Field(ge=0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmailSummaryDelivery(BaseModel):
    """Result of rendering and optionally sending an email summary."""

    summary: EmailSummary
    sent: bool
    smtp_host: str | None = None


SmtpFactory = Callable[[str, int], smtplib.SMTP]


def render_email_summary(
    reports: Sequence[SavedInsightReport],
    *,
    recipients: list[str],
    subject: str | None = None,
) -> EmailSummary:
    """Render saved reports into a concise plain-text email digest."""
    ordered = sorted(reports, key=lambda report: report.created_at, reverse=True)
    resolved_subject = subject or _default_subject(ordered)
    body_lines = [
        resolved_subject,
        "",
        f"Reports: {len(ordered)}",
        "",
    ]
    for index, report in enumerate(ordered, start=1):
        body_lines.extend(_report_lines(index, report))
        body_lines.append("")
    return EmailSummary(
        subject=resolved_subject,
        body_text="\n".join(body_lines).strip() + "\n",
        recipients=recipients,
        report_ids=[report.report_id for report in ordered],
        report_count=len(ordered),
    )


def select_reports_for_summary(
    store: InsightReportStore,
    *,
    report_ids: list[str],
    max_reports: int,
) -> list[SavedInsightReport]:
    """Resolve explicit report IDs or the latest saved reports for a summary."""
    selected_ids = report_ids or [summary.report_id for summary in store.list()[:max_reports]]
    reports: list[SavedInsightReport] = []
    missing: list[str] = []
    for report_id in selected_ids[:max_reports]:
        report = store.get(report_id)
        if report is None:
            missing.append(report_id)
            continue
        reports.append(report)
    if missing:
        raise LookupError(f"report not found: {', '.join(missing)}")
    if not reports:
        raise ValueError("no saved reports are available for an email summary")
    return reports


def deliver_email_summary(
    summary: EmailSummary,
    *,
    smtp_host: str | None,
    smtp_port: int,
    from_address: str,
    username: str | None = None,
    password: str | None = None,
    use_tls: bool = True,
    smtp_factory: SmtpFactory = smtplib.SMTP,
) -> EmailSummaryDelivery:
    """Send a rendered email summary through SMTP."""
    if not smtp_host:
        raise ValueError("smtp_host is required when sending email summaries")
    message = EmailMessage()
    message["From"] = from_address
    message["To"] = ", ".join(summary.recipients)
    message["Subject"] = summary.subject
    message.set_content(summary.body_text)

    with smtp_factory(smtp_host, smtp_port) as smtp:
        if use_tls:
            smtp.starttls()
        if username:
            smtp.login(username, password or "")
        smtp.send_message(message)
    return EmailSummaryDelivery(summary=summary, sent=True, smtp_host=smtp_host)


def _default_subject(reports: Sequence[SavedInsightReport]) -> str:
    if len(reports) == 1:
        return f"Feedback insight: {reports[0].title}"
    return f"Feedback insight digest: {len(reports)} reports"


def _report_lines(index: int, report: SavedInsightReport) -> list[str]:
    answer = " ".join(report.result.answer.split())
    actions = report.result.recommended_actions[:3]
    lines = [
        f"{index}. {report.title}",
        f"Question: {report.question}",
        f"Route: {report.result.route}",
        f"Confidence: {report.result.confidence:.3f}",
        f"Citations: {len(report.result.citations)}",
        f"Answer: {answer}",
    ]
    if actions:
        lines.append("Recommended actions:")
        lines.extend(f"- {action}" for action in actions)
    if report.tags:
        lines.append(f"Tags: {', '.join(report.tags)}")
    return lines
