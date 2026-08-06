"""Saved insight reports for product and analyst workflows."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from feedback_intelligence_agent.schemas import AgentAnswer

_REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def new_report_id() -> str:
    """Return a fresh URL-safe report identifier."""
    return uuid.uuid4().hex


class SaveInsightReportRequest(BaseModel):
    """Request to persist one generated agent answer as an insight report."""

    title: str = Field(min_length=1, max_length=120)
    result: AgentAnswer
    tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, value: list[str]) -> list[str]:
        """Trim, lowercase, deduplicate, and bound report tags."""
        normalised: list[str] = []
        for tag in value:
            clean = " ".join(tag.lower().split())
            if not clean or clean in normalised:
                continue
            normalised.append(clean)
            if len(normalised) >= 12:
                break
        return normalised


class SavedInsightReport(BaseModel):
    """Persisted answer package that can be shared or revisited later."""

    report_id: str
    title: str
    question: str
    result: AgentAnswer
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    def summary(self) -> InsightReportSummary:
        """Return the lightweight list view for this report."""
        return InsightReportSummary(
            report_id=self.report_id,
            title=self.title,
            question=self.question,
            created_at=self.created_at,
            route=self.result.route,
            confidence=self.result.confidence,
            citations=len(self.result.citations),
            tags=self.tags,
        )


class InsightReportSummary(BaseModel):
    """Lightweight report metadata for list endpoints and CLI output."""

    report_id: str
    title: str
    question: str
    created_at: datetime
    route: str
    confidence: float
    citations: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)


class InsightReportStore(Protocol):
    """Protocol implemented by report persistence backends."""

    def save(self, request: SaveInsightReportRequest) -> SavedInsightReport:
        """Persist a report and return the stored model."""
        ...

    def get(self, report_id: str) -> SavedInsightReport | None:
        """Return a saved report, or None when unknown."""
        ...

    def list(self) -> list[InsightReportSummary]:
        """Return saved report summaries sorted newest first."""
        ...


class InMemoryInsightReportStore:
    """Dict-backed report store for tests."""

    def __init__(self) -> None:
        """Initialise the empty report store."""
        self._reports: dict[str, SavedInsightReport] = {}

    def save(self, request: SaveInsightReportRequest) -> SavedInsightReport:
        """Persist a report in memory."""
        report = _report_from_request(request)
        self._reports[report.report_id] = report.model_copy(deep=True)
        return report.model_copy(deep=True)

    def get(self, report_id: str) -> SavedInsightReport | None:
        """Return a deep copy of a stored report, or None."""
        _validate_report_id(report_id)
        report = self._reports.get(report_id)
        return report.model_copy(deep=True) if report is not None else None

    def list(self) -> list[InsightReportSummary]:
        """Return report summaries sorted newest first."""
        return _sort_summaries(report.summary() for report in self._reports.values())


class JsonInsightReportStore:
    """Report store that persists one JSON file per report."""

    def __init__(self, root: str | Path) -> None:
        """Create the store and ensure the root directory exists."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, report_id: str) -> Path:
        return self.root / f"{_validate_report_id(report_id)}.json"

    def save(self, request: SaveInsightReportRequest) -> SavedInsightReport:
        """Persist a report as indented JSON."""
        report = _report_from_request(request)
        self._path(report.report_id).write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report

    def get(self, report_id: str) -> SavedInsightReport | None:
        """Load a report from disk, or None when missing."""
        path = self._path(report_id)
        if not path.exists():
            return None
        return SavedInsightReport.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[InsightReportSummary]:
        """Return saved report summaries sorted newest first."""
        reports = [
            SavedInsightReport.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*.json")
        ]
        return _sort_summaries(report.summary() for report in reports)


def _report_from_request(request: SaveInsightReportRequest) -> SavedInsightReport:
    return SavedInsightReport(
        report_id=new_report_id(),
        title=request.title,
        question=request.result.question,
        result=request.result,
        tags=request.tags,
        notes=request.notes,
    )


def _sort_summaries(summaries: Iterable[InsightReportSummary]) -> list[InsightReportSummary]:
    return sorted(
        summaries,
        key=lambda report: (report.created_at, report.report_id),
        reverse=True,
    )


def _validate_report_id(report_id: str) -> str:
    if not _REPORT_ID_PATTERN.match(report_id):
        raise ValueError(
            f"invalid report_id {report_id!r}: expected 1-64 characters "
            "from [A-Za-z0-9._-] starting with a letter or digit"
        )
    return report_id
