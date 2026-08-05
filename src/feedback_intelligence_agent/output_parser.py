"""Structured LLM output parsing and deterministic repair."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

OutputFormat = Literal["json", "repaired_json", "sectioned", "raw"]


class GeneratedAnswerPayload(BaseModel):
    """Validated answer payload expected from an LLM provider."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    recommended_actions: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("recommended_actions", "recommendedActions", "actions"),
    )

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        """Reject answers that collapse to empty text."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer cannot be empty")
        return stripped

    @field_validator("recommended_actions", mode="before")
    @classmethod
    def coerce_actions(cls, value: object) -> object:
        """Accept strings or string lists for recommended actions."""
        if value is None:
            return []
        if isinstance(value, str):
            return value.splitlines()
        return value

    @field_validator("recommended_actions")
    @classmethod
    def strip_actions(cls, value: list[str]) -> list[str]:
        """Normalize action bullets and limit the public action list."""
        actions = []
        for item in value:
            action = re.sub(r"^[-*]\s*", "", item).strip()
            if action:
                actions.append(action)
        return actions[:5]


class ParsedLLMOutput(BaseModel):
    """Parsed answer plus diagnostics about validation and repair."""

    payload: GeneratedAnswerPayload
    output_format: OutputFormat
    repair_applied: bool = False
    validation_error: str | None = None


def parse_llm_output(raw_response: str) -> ParsedLLMOutput:
    """Parse provider output as structured JSON, repairing common fallback formats."""
    stripped = raw_response.strip()
    json_error: str | None = None
    for candidate, repaired in _json_candidates(stripped):
        try:
            data = json.loads(candidate)
            payload = GeneratedAnswerPayload.model_validate(data)
            return ParsedLLMOutput(
                payload=payload,
                output_format="repaired_json" if repaired else "json",
                repair_applied=repaired,
                validation_error=json_error,
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            json_error = _compact_error(exc)

    sectioned = _parse_sectioned_response(stripped)
    if sectioned is not None:
        return ParsedLLMOutput(
            payload=sectioned,
            output_format="sectioned",
            repair_applied=True,
            validation_error=json_error,
        )

    return ParsedLLMOutput(
        payload=GeneratedAnswerPayload(answer=stripped or "I could not parse the model response."),
        output_format="raw",
        repair_applied=True,
        validation_error=json_error,
    )


def _json_candidates(text: str) -> list[tuple[str, bool]]:
    """Return possible JSON payloads ordered from strict to repaired."""
    candidates: list[tuple[str, bool]] = [(text, False)]
    fenced = _extract_fenced_json(text)
    if fenced is not None and fenced != text:
        candidates.append((fenced, True))
    embedded = _extract_first_json_object(text)
    if embedded is not None and embedded not in {candidate for candidate, _ in candidates}:
        candidates.append((embedded, True))
    return candidates


def _extract_fenced_json(text: str) -> str | None:
    """Extract a ```json fenced block when present."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced top-level JSON object from surrounding prose."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_sectioned_response(text: str) -> GeneratedAnswerPayload | None:
    """Repair the legacy sectioned response format into a typed payload."""
    answer = _section(text, "Answer", ["Recommended actions", "Citations"])
    actions_text = _section(text, "Recommended actions", ["Citations"])
    if not answer and not actions_text:
        return None
    actions = [
        re.sub(r"^[-*]\s*", "", line).strip() for line in actions_text.splitlines() if line.strip()
    ]
    return GeneratedAnswerPayload(answer=answer.strip() or text, recommended_actions=actions)


def _section(text: str, heading: str, next_headings: list[str]) -> str:
    """Extract a simple markdown-like section from text."""
    start_pattern = re.compile(rf"{re.escape(heading)}\s*:\s*", flags=re.IGNORECASE)
    start_match = start_pattern.search(text)
    if not start_match:
        return ""
    start = start_match.end()
    end = len(text)
    for next_heading in next_headings:
        next_pattern = re.compile(
            rf"\n\s*{re.escape(next_heading)}\s*:\s*",
            flags=re.IGNORECASE,
        )
        next_match = next_pattern.search(text, pos=start)
        if next_match:
            end = min(end, next_match.start())
    return text[start:end].strip()


def _compact_error(exc: Exception) -> str:
    """Return a compact validation error for diagnostics."""
    message = str(exc).replace("\n", " ")
    return message[:240]
