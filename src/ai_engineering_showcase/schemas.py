"""Typed domain schemas used across the project."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ai_engineering_showcase.guardrails import GuardrailDecision


class FeedbackChannel(str, Enum):
    """Supported feedback channels."""

    support_ticket = "support_ticket"
    app_review = "app_review"
    sales_call = "sales_call"
    nps_survey = "nps_survey"
    community = "community"


class FeedbackRecord(BaseModel):
    """Single raw customer feedback record."""

    model_config = ConfigDict(str_strip_whitespace=True)

    feedback_id: str = Field(min_length=1)
    customer_segment: str = Field(min_length=1)
    channel: FeedbackChannel
    rating: int = Field(ge=1, le=5)
    text: str = Field(min_length=3)
    created_at: datetime

    @field_validator("text")
    @classmethod
    def normalise_text(cls, value: str) -> str:
        """Collapse repeated whitespace in feedback text."""
        return " ".join(value.split())


class DocumentChunk(BaseModel):
    """Searchable chunk derived from feedback."""

    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any]


class SearchResult(BaseModel):
    """A retrieved chunk with similarity score."""

    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Evidence citation exposed to agent and API consumers.

    ``citation_id`` matches the bracketed markers (``[1]``, ``[2]``) embedded in
    the generated answer text, so every claim can be traced back to the exact
    retrieved document and chunk that supports it.
    """

    citation_id: int = Field(ge=1)
    document_id: str
    chunk_id: str
    source: str
    quote: str
    score: float


class ToolRunRecord(BaseModel):
    """Metadata about one tool invocation performed during an agent run.

    ``status`` is ``ok`` when the tool ran successfully, ``refused`` when an
    unknown or unavailable tool was requested, and ``error`` when a registered
    tool failed. ``output`` carries the structured tool output so API and CLI
    consumers can inspect exactly what the tool produced.
    """

    tool_name: str
    status: Literal["ok", "refused", "error"]
    summary: str
    output: dict[str, Any] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    """Final answer returned by the insight agent.

    ``guardrail`` records the deterministic safety decision made for the
    request, so API and CLI consumers can see whether the question was
    answered normally or refused (and why). ``tool_run`` records the tool
    invoked for the question, if any, making tool use visible in the agent
    answer and the API response.
    """

    question: str
    answer: str
    recommended_actions: list[str]
    citations: list[Citation]
    route: str
    confidence: float
    guardrail: GuardrailDecision | None = None
    tool_run: ToolRunRecord | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """API request for the `/query` endpoint."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=12)


class QueryResponse(BaseModel):
    """API response for the `/query` endpoint."""

    result: AgentAnswer


class StreamMetadata(BaseModel):
    """Payload of the final ``metadata`` SSE event of the `/query/stream` endpoint.

    Sent after all ``content`` chunks so streaming clients can render the
    answer incrementally and then attach evidence, scores, and timing:
    ``provider`` names the LLM provider class, ``latency_ms`` measures the
    full answer-generation time, ``sources`` lists the cited document IDs in
    citation order, and ``retrieval_scores`` carries the per-citation
    retrieval scores.
    """

    provider: str
    latency_ms: float
    route: str
    confidence: float
    sources: list[str]
    retrieval_scores: list[float]
    citations: list[Citation]
    recommended_actions: list[str] = Field(default_factory=list)
    guardrail: GuardrailDecision | None = None


class ChatRequest(BaseModel):
    """API request for the `/chat` endpoint.

    ``conversation_id`` is optional: omit it to start a new conversation, or
    pass the identifier returned by a previous `/chat` call to continue with
    the stored turns as context.
    """

    message: str = Field(min_length=1)
    conversation_id: str | None = None
    top_k: int = Field(default=4, ge=1, le=12)


class ChatResponse(BaseModel):
    """API response for the `/chat` endpoint."""

    conversation_id: str
    result: AgentAnswer


class IndexRequest(BaseModel):
    """API request for rebuilding the index from a CSV path."""

    input_path: str = Field(min_length=1)
    index_path: str | None = None


class JobSubmitResponse(BaseModel):
    """API response returned when an ingestion job is submitted.

    The job runs asynchronously, so only the identifier and the initial status
    are returned immediately. Poll ``GET /ingestion/jobs/{job_id}`` for the
    terminal status and result.
    """

    job_id: str
    status: str


class EvaluationCase(BaseModel):
    """Single evaluation case covering retrieval and answer quality.

    The ``relevant_source_ids`` field name is accepted as a legacy alias for
    ``relevant_document_ids`` so older evaluation files keep loading.
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=1)
    expected_keywords: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("relevant_document_ids", "relevant_source_ids"),
    )
    is_answerable: bool = True
