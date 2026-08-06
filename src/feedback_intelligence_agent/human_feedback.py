"""Human review feedback on generated agent answers."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from feedback_intelligence_agent.schemas import AgentAnswer

_FEEDBACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class HumanFeedbackRating(str, Enum):
    """Allowed human judgement labels for an answer."""

    useful = "useful"
    not_useful = "not_useful"


def new_human_feedback_id() -> str:
    """Return a fresh URL-safe human feedback identifier."""
    return uuid.uuid4().hex


class SubmitHumanFeedbackRequest(BaseModel):
    """Request to persist one human judgement on a generated answer."""

    result: AgentAnswer
    rating: HumanFeedbackRating
    tenant_id: str = Field(default="default", min_length=1)
    comment: str | None = Field(default=None, max_length=2000)
    report_id: str | None = Field(default=None, max_length=64)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tenant_id")
    @classmethod
    def normalise_tenant_id(cls, value: str) -> str:
        """Normalize tenant identifiers for deterministic filtering."""
        return value.strip().lower()

    @field_validator("comment")
    @classmethod
    def normalise_comment(cls, value: str | None) -> str | None:
        """Trim empty feedback comments to None."""
        if value is None:
            return None
        clean = value.strip()
        return clean or None

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, value: str | None) -> str | None:
        """Reject unsafe report identifiers when a feedback record is linked."""
        if value is None:
            return None
        return _validate_human_feedback_id(value, field_name="report_id")

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, value: list[str]) -> list[str]:
        """Trim, lowercase, deduplicate, and bound feedback tags."""
        normalised: list[str] = []
        for tag in value:
            clean = " ".join(tag.lower().split())
            if not clean or clean in normalised:
                continue
            normalised.append(clean)
            if len(normalised) >= 12:
                break
        return normalised


class HumanFeedbackRecord(BaseModel):
    """Persisted human judgement tied to a generated answer."""

    feedback_id: str
    question: str
    result: AgentAnswer
    rating: HumanFeedbackRating
    tenant_id: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    comment: str | None = None
    report_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    def summary(self) -> HumanFeedbackSummary:
        """Return the lightweight list view for this feedback record."""
        return HumanFeedbackSummary(
            feedback_id=self.feedback_id,
            question=self.question,
            rating=self.rating,
            created_at=self.created_at,
            tenant_id=self.tenant_id,
            route=self.result.route,
            confidence=self.result.confidence,
            report_id=self.report_id,
            has_comment=self.comment is not None,
            tags=self.tags,
        )


class HumanFeedbackSummary(BaseModel):
    """Lightweight feedback metadata for list endpoints and CLI output."""

    feedback_id: str
    question: str
    rating: HumanFeedbackRating
    created_at: datetime
    tenant_id: str = "default"
    route: str
    confidence: float
    report_id: str | None = None
    has_comment: bool
    tags: list[str] = Field(default_factory=list)


class HumanFeedbackStore(Protocol):
    """Protocol implemented by human feedback persistence backends."""

    def save(self, request: SubmitHumanFeedbackRequest) -> HumanFeedbackRecord:
        """Persist a feedback record and return the stored model."""
        ...

    def get(self, feedback_id: str) -> HumanFeedbackRecord | None:
        """Return a feedback record, or None when unknown."""
        ...

    def list(self, *, tenant_id: str | None = None) -> list[HumanFeedbackSummary]:
        """Return feedback summaries sorted newest first."""
        ...


class InMemoryHumanFeedbackStore:
    """Dict-backed human feedback store for tests."""

    def __init__(self) -> None:
        """Initialise the empty feedback store."""
        self._records: dict[str, HumanFeedbackRecord] = {}

    def save(self, request: SubmitHumanFeedbackRequest) -> HumanFeedbackRecord:
        """Persist a feedback record in memory."""
        record = _record_from_request(request)
        self._records[record.feedback_id] = record.model_copy(deep=True)
        return record.model_copy(deep=True)

    def get(self, feedback_id: str) -> HumanFeedbackRecord | None:
        """Return a deep copy of a stored feedback record, or None."""
        _validate_human_feedback_id(feedback_id)
        record = self._records.get(feedback_id)
        return record.model_copy(deep=True) if record is not None else None

    def list(self, *, tenant_id: str | None = None) -> list[HumanFeedbackSummary]:
        """Return feedback summaries sorted newest first."""
        return _sort_summaries(
            record.summary()
            for record in self._records.values()
            if _matches_tenant(record.tenant_id, tenant_id)
        )


class JsonHumanFeedbackStore:
    """Human feedback store that persists one JSON file per record."""

    def __init__(self, root: str | Path) -> None:
        """Create the store and ensure the root directory exists."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, feedback_id: str) -> Path:
        return self.root / f"{_validate_human_feedback_id(feedback_id)}.json"

    def save(self, request: SubmitHumanFeedbackRequest) -> HumanFeedbackRecord:
        """Persist a feedback record as indented JSON."""
        record = _record_from_request(request)
        self._path(record.feedback_id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return record

    def get(self, feedback_id: str) -> HumanFeedbackRecord | None:
        """Load a feedback record from disk, or None when missing."""
        path = self._path(feedback_id)
        if not path.exists():
            return None
        return HumanFeedbackRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, *, tenant_id: str | None = None) -> list[HumanFeedbackSummary]:
        """Return feedback summaries sorted newest first."""
        records = [
            HumanFeedbackRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*.json")
        ]
        return _sort_summaries(
            record.summary() for record in records if _matches_tenant(record.tenant_id, tenant_id)
        )


def _record_from_request(request: SubmitHumanFeedbackRequest) -> HumanFeedbackRecord:
    return HumanFeedbackRecord(
        feedback_id=new_human_feedback_id(),
        question=request.result.question,
        result=request.result,
        rating=request.rating,
        tenant_id=request.tenant_id,
        comment=request.comment,
        report_id=request.report_id,
        tags=request.tags,
    )


def _sort_summaries(
    summaries: Iterable[HumanFeedbackSummary],
) -> list[HumanFeedbackSummary]:
    return sorted(
        summaries,
        key=lambda record: (record.created_at, record.feedback_id),
        reverse=True,
    )


def _validate_human_feedback_id(feedback_id: str, *, field_name: str = "feedback_id") -> str:
    if not _FEEDBACK_ID_PATTERN.match(feedback_id):
        raise ValueError(
            f"invalid {field_name} {feedback_id!r}: expected 1-64 characters "
            "from [A-Za-z0-9._-] starting with a letter or digit"
        )
    return feedback_id


def _matches_tenant(value: str, tenant_id: str | None) -> bool:
    if tenant_id is None:
        return True
    return value.lower() == tenant_id.strip().lower()
