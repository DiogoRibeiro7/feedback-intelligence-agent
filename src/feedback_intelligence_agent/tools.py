"""Deterministic tool-use framework for the feedback intelligence agent.

The module provides:

1. A typed tool interface (:class:`Tool`) where every tool declares a name,
   a description, a Pydantic input schema, and a Pydantic output schema.
2. Three local tools that run fully offline: :class:`SentimentSummaryTool`,
   :class:`IssueClusterTool`, and :class:`TicketDraftTool`.
3. A registry (:class:`ToolRegistry`) and a deterministic keyword router
   (:class:`ToolRouter`) that selects at most one tool per query.

No function-calling API is involved: routing and execution are keyword and
rule based, so tool behaviour is reproducible in tests and CI. Unknown or
unavailable tool requests are surfaced as a graceful refusal instead of an
error, and the agent falls back to the plain RAG answer.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

from feedback_intelligence_agent.citations import summarize_evidence
from feedback_intelligence_agent.metadata_filters import matches_metadata_filters
from feedback_intelligence_agent.schemas import DocumentChunk, MetadataFilters, SearchResult


class ToolError(RuntimeError):
    """Raised when a tool receives invalid input or cannot complete its run."""


class ToolOutput(BaseModel):
    """Base class for tool outputs that can render a human-readable summary."""

    def render(self) -> str:
        """Return a compact human-readable summary of the tool output."""
        raise NotImplementedError


class Tool(ABC):
    """Typed interface implemented by all local tools.

    Every tool declares a stable ``name``, a human-readable ``description``,
    and Pydantic ``input_schema``/``output_schema`` models. :meth:`execute`
    validates the payload before running, so malformed inputs surface as a
    :class:`ToolError` instead of an unhandled exception.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[ToolOutput]]

    def build_payload(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        filters: MetadataFilters | None = None,
    ) -> dict[str, Any]:
        """Build tool input with request filters, independently of retrieval limits."""
        del question, results
        return {"filters": filters.model_dump(mode="json")} if filters is not None else {}

    def execute(self, payload: Mapping[str, Any]) -> ToolOutput:
        """Validate the payload against the input schema and run the tool."""
        try:
            data = self.input_schema.model_validate(dict(payload))
        except ValidationError as exc:
            raise ToolError(f"invalid input for tool '{self.name}': {exc}") from exc
        return self._run(data)

    @abstractmethod
    def _run(self, payload: BaseModel) -> ToolOutput:
        """Run the tool with an already-validated payload."""


@dataclass(frozen=True)
class _FeedbackEntry:
    """Deduplicated per-document view of the indexed feedback used by tools."""

    document_id: str
    segment: str
    channel: str
    rating: int
    text: str


def _entries_from_chunks(
    chunks: Sequence[DocumentChunk], filters: MetadataFilters
) -> list[_FeedbackEntry]:
    """Filter indexed chunks before building one entry per matching document."""
    entries: list[_FeedbackEntry] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not matches_metadata_filters(chunk.metadata, filters):
            continue
        if chunk.source_id in seen:
            continue
        seen.add(chunk.source_id)
        metadata = chunk.metadata
        try:
            rating = int(metadata.get("rating", 0))
        except (TypeError, ValueError):
            rating = 0
        entries.append(
            _FeedbackEntry(
                document_id=chunk.source_id,
                segment=str(metadata.get("customer_segment", "unknown")),
                channel=str(metadata.get("channel", "unknown")),
                rating=rating,
                text=chunk.text,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# SentimentSummaryTool
# ---------------------------------------------------------------------------


class SentimentSummaryInput(BaseModel):
    """Optional filters applied before summarising sentiment."""

    filters: MetadataFilters = Field(default_factory=MetadataFilters)
    segment: str | None = None
    channel: str | None = None


class SentimentSummaryOutput(ToolOutput):
    """Aggregated sentiment and rating distribution over the feedback dataset."""

    total_records: int
    average_rating: float
    rating_distribution: dict[str, int]
    sentiment_distribution: dict[str, int]
    average_rating_by_segment: dict[str, float]

    def render(self) -> str:
        """Summarise the distribution in one sentence."""
        if not self.total_records:
            return "No feedback records matched the requested filters."
        dist = self.sentiment_distribution
        return (
            f"Sentiment across {self.total_records} feedback records: "
            f"{dist.get('positive', 0)} positive, {dist.get('neutral', 0)} neutral, "
            f"{dist.get('negative', 0)} negative; average rating "
            f"{self.average_rating:.2f}/5."
        )


class SentimentSummaryTool(Tool):
    """Summarise the sentiment and rating distribution of the indexed feedback.

    Ratings of 4-5 count as positive, 3 as neutral, and 1-2 as negative, so
    the summary is fully deterministic and needs no model call.
    """

    name = "sentiment_summary"
    description = "Summarise sentiment and rating distribution across the feedback dataset."
    input_schema = SentimentSummaryInput
    output_schema = SentimentSummaryOutput

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        """Bind the tool to the indexed feedback chunks."""
        self._chunks = tuple(chunks)

    def _run(self, payload: BaseModel) -> SentimentSummaryOutput:
        """Aggregate ratings into distributions, applying optional filters."""
        if not isinstance(payload, SentimentSummaryInput):
            raise ToolError(f"tool '{self.name}' received an unexpected payload type")
        entries = _entries_from_chunks(self._chunks, payload.filters)
        if payload.segment:
            entries = [e for e in entries if e.segment.lower() == payload.segment.lower()]
        if payload.channel:
            entries = [e for e in entries if e.channel.lower() == payload.channel.lower()]

        rating_distribution = {str(value): 0 for value in range(1, 6)}
        sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        segment_ratings: dict[str, list[int]] = {}
        rated = [entry for entry in entries if 1 <= entry.rating <= 5]
        for entry in rated:
            rating_distribution[str(entry.rating)] += 1
            sentiment_distribution[_sentiment_bucket(entry.rating)] += 1
            segment_ratings.setdefault(entry.segment, []).append(entry.rating)
        average = round(sum(e.rating for e in rated) / len(rated), 2) if rated else 0.0
        by_segment = {
            segment: round(sum(ratings) / len(ratings), 2)
            for segment, ratings in sorted(segment_ratings.items())
        }
        return SentimentSummaryOutput(
            total_records=len(rated),
            average_rating=average,
            rating_distribution=rating_distribution,
            sentiment_distribution=sentiment_distribution,
            average_rating_by_segment=by_segment,
        )


def _sentiment_bucket(rating: int) -> str:
    """Map a 1-5 rating to a deterministic sentiment bucket."""
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


# ---------------------------------------------------------------------------
# IssueClusterTool
# ---------------------------------------------------------------------------

ISSUE_TERM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("onboarding", ("onboarding", "checklist", "setup", "implementation")),
    ("integrations", ("integration", "salesforce", "hubspot", "api")),
    ("reporting_and_exports", ("export", "report", "reporting", "analytics")),
    ("pricing_and_renewal", ("pricing", "price", "renewal", "cost")),
    ("support_experience", ("support", "ticket", "response", "incident")),
    ("performance", ("latency", "slow", "performance", "delay")),
    ("documentation", ("documentation", "docs", "examples", "templates")),
)
"""Deterministic keyword groups used to cluster recurring customer issues."""


class IssueClusterInput(BaseModel):
    """Metadata filters and controls for the issue clustering output size."""

    filters: MetadataFilters = Field(default_factory=MetadataFilters)
    max_clusters: int = Field(default=5, ge=1, le=10)
    min_count: int = Field(default=1, ge=1)


class IssueCluster(BaseModel):
    """One recurring issue cluster with supporting documents."""

    label: str
    count: int
    matched_terms: list[str]
    document_ids: list[str]
    example_quote: str


class IssueClusterOutput(ToolOutput):
    """Recurring issue clusters detected in the feedback dataset."""

    total_records: int
    clusters: list[IssueCluster]

    def render(self) -> str:
        """Summarise the top clusters in one sentence."""
        if not self.clusters:
            return "No recurring issue clusters were found in the feedback dataset."
        top = ", ".join(f"{cluster.label} ({cluster.count})" for cluster in self.clusters[:3])
        return (
            f"Identified {len(self.clusters)} recurring issue clusters across "
            f"{self.total_records} feedback records; top clusters: {top}."
        )


class IssueClusterTool(Tool):
    """Group recurring customer issues with deterministic keyword clustering."""

    name = "issue_cluster"
    description = "Group recurring customer issues into deterministic keyword clusters."
    input_schema = IssueClusterInput
    output_schema = IssueClusterOutput

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        """Bind the tool to the indexed feedback chunks."""
        self._chunks = tuple(chunks)

    def _run(self, payload: BaseModel) -> IssueClusterOutput:
        """Match every feedback entry against the keyword groups and rank clusters."""
        if not isinstance(payload, IssueClusterInput):
            raise ToolError(f"tool '{self.name}' received an unexpected payload type")
        entries = _entries_from_chunks(self._chunks, payload.filters)
        clusters: list[IssueCluster] = []
        for label, terms in ISSUE_TERM_GROUPS:
            document_ids: list[str] = []
            matched_terms: set[str] = set()
            example = ""
            for entry in entries:
                hits = [term for term in terms if term in entry.text.lower()]
                if hits:
                    document_ids.append(entry.document_id)
                    matched_terms.update(hits)
                    if not example:
                        example = summarize_evidence(entry.text, max_chars=140)
            if len(document_ids) >= payload.min_count:
                clusters.append(
                    IssueCluster(
                        label=label,
                        count=len(document_ids),
                        matched_terms=sorted(matched_terms),
                        document_ids=document_ids,
                        example_quote=example,
                    )
                )
        clusters.sort(key=lambda cluster: (-cluster.count, cluster.label))
        return IssueClusterOutput(
            total_records=len(entries),
            clusters=clusters[: payload.max_clusters],
        )


# ---------------------------------------------------------------------------
# TicketDraftTool
# ---------------------------------------------------------------------------


class TicketEvidence(BaseModel):
    """One retrieved evidence item attached to a ticket draft."""

    document_id: str
    source: str
    text: str


class TicketDraftInput(BaseModel):
    """Question and retrieved evidence used to draft a support ticket."""

    question: str = Field(min_length=3)
    evidence: list[TicketEvidence] = Field(default_factory=list)


class TicketDraftOutput(ToolOutput):
    """A drafted support ticket grounded in retrieved feedback."""

    title: str
    body: str
    priority: Literal["low", "medium", "high"]
    references: list[str]
    tags: list[str]

    def render(self) -> str:
        """Summarise the drafted ticket in one sentence."""
        references = ", ".join(self.references) or "none"
        return (
            f"Drafted support ticket '{self.title}' "
            f"(priority: {self.priority}, references: {references})."
        )


class TicketDraftTool(Tool):
    """Draft a support ticket from the question and retrieved feedback context."""

    name = "ticket_draft"
    description = "Draft a support ticket grounded in retrieved customer feedback."
    input_schema = TicketDraftInput
    output_schema = TicketDraftOutput

    _TICKET_PREFIX = re.compile(
        r"^\s*(?:please\s+)?(?:draft|create|open|file|write|prepare)\s+(?:a|an|the)?\s*"
        r"(?:new\s+)?(?:support\s+|jira\s+)?tickets?\s*(?:for|about|on|regarding)?\s*",
        re.IGNORECASE,
    )

    def build_payload(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        filters: MetadataFilters | None = None,
    ) -> dict[str, Any]:
        """Filter retrieved chunks before deduplicating and limiting ticket evidence."""
        evidence: list[dict[str, str]] = []
        seen: set[str] = set()
        for result in results:
            if filters is not None and not matches_metadata_filters(result.chunk.metadata, filters):
                continue
            document_id = result.chunk.source_id
            if document_id in seen:
                continue
            seen.add(document_id)
            evidence.append(
                {
                    "document_id": document_id,
                    "source": str(result.chunk.metadata.get("channel", "feedback")),
                    "text": result.chunk.text,
                }
            )
        return {"question": question, "evidence": evidence[:4]}

    def _run(self, payload: BaseModel) -> TicketDraftOutput:
        """Assemble a deterministic ticket draft from the question and evidence."""
        if not isinstance(payload, TicketDraftInput):
            raise ToolError(f"tool '{self.name}' received an unexpected payload type")
        topic = self._topic(payload.question)
        combined = " ".join([payload.question, *(item.text for item in payload.evidence)]).lower()
        tags = sorted(
            label for label, terms in ISSUE_TERM_GROUPS if any(term in combined for term in terms)
        )
        if len(payload.evidence) >= 3:
            priority: Literal["low", "medium", "high"] = "high"
        elif payload.evidence:
            priority = "medium"
        else:
            priority = "low"
        lines = [f"Summary: {topic}", "", "Evidence from retrieved customer feedback:"]
        if payload.evidence:
            lines.extend(
                f"- [{item.document_id} | {item.source}] "
                f"{summarize_evidence(item.text, max_chars=160)}"
                for item in payload.evidence
            )
        else:
            lines.append("- No retrieved evidence; please triage manually.")
        lines.extend(
            [
                "",
                "Suggested next steps:",
                "- Reproduce the reported issue and confirm its scope.",
                "- Quantify impact by customer segment and channel.",
                "- Assign an owner and track resolution in the next review.",
            ]
        )
        return TicketDraftOutput(
            title=f"Customer feedback follow-up: {summarize_evidence(topic, max_chars=70)}",
            body="\n".join(lines),
            priority=priority,
            references=[item.document_id for item in payload.evidence],
            tags=tags,
        )

    def _topic(self, question: str) -> str:
        """Extract a compact ticket topic from the user question."""
        cleaned = self._TICKET_PREFIX.sub("", question).strip().rstrip("?.!").strip()
        return cleaned or question.strip()


# ---------------------------------------------------------------------------
# Registry and router
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of available tools keyed by their stable names."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        """Register the provided tools, rejecting duplicate names."""
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry; duplicate names raise ``ValueError``."""
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool registered under ``name``, or ``None`` if unknown."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return the sorted names of all registered tools."""
        return sorted(self._tools)


class ToolSelection(BaseModel):
    """Outcome of routing one user query to a tool.

    ``status`` is ``selected`` when a registered tool matched, ``no_tool``
    when the query should be answered by plain RAG, and ``unknown_tool`` when
    the user explicitly requested a tool that is not available.
    """

    status: Literal["selected", "no_tool", "unknown_tool"]
    tool_name: str | None = None
    reason: str


def _compile_patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive routing patterns."""
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


@dataclass(frozen=True)
class ToolRoute:
    """Deterministic routing rule mapping query patterns to one tool."""

    tool_name: str
    patterns: tuple[re.Pattern[str], ...]


TOOL_ROUTES: tuple[ToolRoute, ...] = (
    ToolRoute(
        "ticket_draft",
        _compile_patterns(
            r"\b(?:draft|create|open|file|write|prepare)\b[^.?!]{0,60}?\btickets?\b",
            r"\bsupport\s+ticket\b",
            r"\bjira\s+(?:ticket|issue)\b",
        ),
    ),
    ToolRoute(
        "issue_cluster",
        _compile_patterns(
            r"\b(?:cluster|group|categori[sz]e)\b[^.?!]{0,60}?"
            r"\b(?:issues?|complaints?|feedback|problems?|themes?)\b",
            r"\b(?:recurring|common|frequent|top|biggest)\s+"
            r"(?:issues?|problems?|complaints?|themes?)\b",
            r"\bissue\s+clusters?\b",
            r"\bthemes?\s+in\s+(?:the\s+)?feedback\b",
        ),
    ),
    ToolRoute(
        "sentiment_summary",
        _compile_patterns(
            r"\bsentiment\b",
            r"\bratings?\s+distribution\b",
            r"\bdistribution\s+of\s+ratings?\b",
            r"\baverage\s+rating\b",
            r"\bhow\s+do\s+customers\s+feel\b",
            r"\bhow\s+(?:happy|satisfied)\s+are\b",
        ),
    ),
)
"""Ordered routing rules; the first matching rule wins."""

_EXPLICIT_TOOL_REQUEST = re.compile(
    r"\b(?:use|run|call|invoke|execute)\s+(?:the\s+)?([a-z0-9_\-]+)\s+tool\b",
    re.IGNORECASE,
)
"""Detects explicit tool requests such as ``use the sentiment_summary tool``."""


class ToolRouter:
    """Deterministic keyword router that selects at most one tool per query.

    Explicit tool requests (``use the <name> tool``) are honoured when the
    tool exists and refused gracefully when it does not. Otherwise, the
    ordered :data:`TOOL_ROUTES` keyword rules decide; queries matching no
    rule fall through to the plain RAG flow.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        """Bind the router to the registry it can select from."""
        self.registry = registry

    def select(self, question: str) -> ToolSelection:
        """Select a tool for the question, refusing unknown explicit requests."""
        explicit = _EXPLICIT_TOOL_REQUEST.search(question)
        if explicit:
            requested = explicit.group(1).lower()
            if self.registry.get(requested) is not None:
                return ToolSelection(
                    status="selected",
                    tool_name=requested,
                    reason=f"explicit request for tool '{requested}'",
                )
            return ToolSelection(
                status="unknown_tool",
                tool_name=requested,
                reason=f"requested tool '{requested}' is not registered",
            )
        for route in TOOL_ROUTES:
            if self.registry.get(route.tool_name) is None:
                continue
            for pattern in route.patterns:
                if pattern.search(question):
                    return ToolSelection(
                        status="selected",
                        tool_name=route.tool_name,
                        reason=f"matched pattern {pattern.pattern!r}",
                    )
        return ToolSelection(
            status="no_tool",
            tool_name=None,
            reason="no tool route matched; answering with plain RAG",
        )


def build_default_tools(chunks: Sequence[DocumentChunk]) -> ToolRegistry:
    """Build the default local tool registry over the indexed feedback chunks."""
    return ToolRegistry(
        (
            SentimentSummaryTool(chunks),
            IssueClusterTool(chunks),
            TicketDraftTool(),
        )
    )
