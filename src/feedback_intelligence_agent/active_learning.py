"""Workflow state for active-learning follow-up items."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from feedback_intelligence_agent.human_feedback import ActiveLearningQueue

_FEEDBACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ActiveLearningStatus(str, Enum):
    """Workflow states for active-learning queue items."""

    open = "open"
    assigned = "assigned"
    in_progress = "in_progress"
    resolved = "resolved"
    dismissed = "dismissed"


class UpdateActiveLearningStateRequest(BaseModel):
    """Request to assign or transition one active-learning queue item."""

    status: ActiveLearningStatus | None = None
    assignee: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("assignee", "notes")
    @classmethod
    def normalise_optional_text(cls, value: str | None) -> str | None:
        """Trim empty optional text fields to None."""
        if value is None:
            return None
        clean = " ".join(value.split())
        return clean or None


class ActiveLearningState(BaseModel):
    """Persisted workflow state for one reviewed answer."""

    feedback_id: str
    status: ActiveLearningStatus = ActiveLearningStatus.open
    assignee: str | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("feedback_id")
    @classmethod
    def validate_feedback_id(cls, value: str) -> str:
        """Reject unsafe feedback identifiers."""
        return validate_feedback_id(value)


class ActiveLearningStateStore(Protocol):
    """Protocol implemented by active-learning state stores."""

    def get(self, feedback_id: str) -> ActiveLearningState | None:
        """Return persisted workflow state, or None when unknown."""
        ...

    def update(
        self, feedback_id: str, request: UpdateActiveLearningStateRequest
    ) -> ActiveLearningState:
        """Create or update workflow state for one feedback record."""
        ...

    def list(self, *, status: ActiveLearningStatus | None = None) -> list[ActiveLearningState]:
        """Return workflow states sorted by most recently updated first."""
        ...


class InMemoryActiveLearningStateStore:
    """Dict-backed active-learning state store for tests."""

    def __init__(self) -> None:
        """Initialise the empty state store."""
        self._states: dict[str, ActiveLearningState] = {}

    def get(self, feedback_id: str) -> ActiveLearningState | None:
        """Return a deep copy of a state record, or None."""
        validate_feedback_id(feedback_id)
        state = self._states.get(feedback_id)
        return state.model_copy(deep=True) if state is not None else None

    def update(
        self, feedback_id: str, request: UpdateActiveLearningStateRequest
    ) -> ActiveLearningState:
        """Create or update workflow state in memory."""
        state = _apply_state_update(self._states.get(feedback_id), feedback_id, request)
        self._states[state.feedback_id] = state.model_copy(deep=True)
        return state.model_copy(deep=True)

    def list(self, *, status: ActiveLearningStatus | None = None) -> list[ActiveLearningState]:
        """Return state records sorted by most recently updated first."""
        return _sort_states(
            state for state in self._states.values() if status is None or state.status == status
        )


class JsonActiveLearningStateStore:
    """JSON-backed active-learning state store."""

    def __init__(self, root: str | Path) -> None:
        """Create the store and ensure the root directory exists."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, feedback_id: str) -> Path:
        return self.root / f"{validate_feedback_id(feedback_id)}.json"

    def get(self, feedback_id: str) -> ActiveLearningState | None:
        """Load one state record from disk, or None when missing."""
        path = self._path(feedback_id)
        if not path.exists():
            return None
        return ActiveLearningState.model_validate_json(path.read_text(encoding="utf-8"))

    def update(
        self, feedback_id: str, request: UpdateActiveLearningStateRequest
    ) -> ActiveLearningState:
        """Create or update one JSON state record."""
        state = _apply_state_update(self.get(feedback_id), feedback_id, request)
        self._path(state.feedback_id).write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return state

    def list(self, *, status: ActiveLearningStatus | None = None) -> list[ActiveLearningState]:
        """Return state records sorted by most recently updated first."""
        states = [
            ActiveLearningState.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*.json")
        ]
        return _sort_states(state for state in states if status is None or state.status == status)


def apply_active_learning_states(
    queue: ActiveLearningQueue,
    states: Iterable[ActiveLearningState],
) -> ActiveLearningQueue:
    """Overlay persisted workflow state onto active-learning queue items."""
    state_by_feedback_id = {state.feedback_id: state for state in states}
    return queue.model_copy(
        update={
            "items": [
                item.model_copy(
                    update=_state_item_update(state_by_feedback_id.get(item.feedback_id))
                )
                for item in queue.items
            ]
        }
    )


def validate_feedback_id(feedback_id: str) -> str:
    """Validate a URL-safe feedback identifier."""
    if not _FEEDBACK_ID_PATTERN.match(feedback_id):
        raise ValueError(
            f"invalid feedback_id {feedback_id!r}: expected 1-64 characters "
            "from [A-Za-z0-9._-] starting with a letter or digit"
        )
    return feedback_id


def _apply_state_update(
    current: ActiveLearningState | None,
    feedback_id: str,
    request: UpdateActiveLearningStateRequest,
) -> ActiveLearningState:
    validate_feedback_id(feedback_id)
    now = datetime.now(timezone.utc)
    state = current or ActiveLearningState(feedback_id=feedback_id, created_at=now)
    updates: dict[str, object] = {"updated_at": now}
    if request.status is not None:
        updates["status"] = request.status
    if "assignee" in request.model_fields_set:
        updates["assignee"] = request.assignee
    if "notes" in request.model_fields_set:
        updates["notes"] = request.notes
    return state.model_copy(update=updates)


def _state_item_update(state: ActiveLearningState | None) -> dict[str, object]:
    if state is None:
        return {
            "workflow_status": ActiveLearningStatus.open.value,
            "assignee": None,
            "state_notes": None,
            "state_updated_at": None,
        }
    return {
        "workflow_status": state.status.value,
        "assignee": state.assignee,
        "state_notes": state.notes,
        "state_updated_at": state.updated_at,
    }


def _sort_states(states: Iterable[ActiveLearningState]) -> list[ActiveLearningState]:
    return sorted(
        (state.model_copy(deep=True) for state in states),
        key=lambda state: (state.updated_at, state.feedback_id),
        reverse=True,
    )
