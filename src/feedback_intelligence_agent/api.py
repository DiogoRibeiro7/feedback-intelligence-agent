"""FastAPI application."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from feedback_intelligence_agent import __version__
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.email_summaries import (
    EmailSummaryDelivery,
    EmailSummaryRequest,
    deliver_email_summary,
    render_email_summary,
    select_reports_for_summary,
)
from feedback_intelligence_agent.factory import (
    build_agent,
    build_conversation_store,
    build_human_feedback_store,
    build_index,
    build_job_store,
    build_report_store,
)
from feedback_intelligence_agent.human_feedback import (
    HumanFeedbackRecord,
    HumanFeedbackSummary,
    SubmitHumanFeedbackRequest,
)
from feedback_intelligence_agent.jobs import JobRequest, JobResult, run_ingestion_job
from feedback_intelligence_agent.memory import ConversationMemory
from feedback_intelligence_agent.reports import (
    InsightReportSummary,
    SavedInsightReport,
    SaveInsightReportRequest,
)
from feedback_intelligence_agent.schemas import (
    AgentAnswer,
    ChatRequest,
    ChatResponse,
    IndexRequest,
    JobSubmitResponse,
    QueryRequest,
    QueryResponse,
    StreamMetadata,
)
from feedback_intelligence_agent.telemetry import configure_logging, log_event


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event with a JSON payload.

    The payload is JSON-encoded onto a single ``data:`` line, so answer text
    containing newlines never breaks the SSE framing.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _split_for_streaming(text: str, *, words_per_chunk: int = 8) -> list[str]:
    """Split text into whitespace-preserving chunks for simulated streaming.

    Splitting keeps every separator, so concatenating the chunks reproduces
    the original text byte-for-byte. Used when the configured provider does
    not expose true token streaming (for example the deterministic local
    provider): the final answer is replayed as small chunks instead.
    """
    parts = [part for part in re.split(r"(\s+)", text) if part]
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for part in parts:
        current.append(part)
        if not part.isspace():
            words += 1
            if words >= words_per_chunk:
                chunks.append("".join(current))
                current = []
                words = 0
    if current:
        chunks.append("".join(current))
    return chunks


def _build_stream_metadata(
    result: AgentAnswer, *, provider: str, latency_ms: float
) -> dict[str, Any]:
    """Build the JSON payload of the final ``metadata`` SSE event."""
    metadata = StreamMetadata(
        provider=provider,
        latency_ms=latency_ms,
        route=result.route,
        confidence=result.confidence,
        sources=[citation.document_id for citation in result.citations],
        retrieval_scores=[citation.score for citation in result.citations],
        citations=result.citations,
        recommended_actions=result.recommended_actions,
        guardrail=result.guardrail,
    )
    return metadata.model_dump(mode="json")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = Settings()
    agent = build_agent(settings)
    conversation_store = build_conversation_store(settings)
    job_store = build_job_store(settings)
    report_store = build_report_store(settings)
    human_feedback_store = build_human_feedback_store(settings)

    app = FastAPI(
        title="Feedback Intelligence Agent API",
        version=__version__,
        description="Evidence-grounded customer feedback intelligence agent.",
    )

    cors_origins = settings.cors_origins
    if cors_origins:
        allow_all = cors_origins == ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return service liveness.

        Liveness probe: confirms the process is up and able to serve
        requests. It performs no dependency checks, so an orchestrator can
        use it to decide whether to restart the container.
        """
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """Return service readiness.

        Readiness probe: the agent, conversation store, and job store were
        all constructed during application startup (above). Reaching this
        handler proves the app object is fully built and able to serve
        traffic, so it reports ``ready`` without faking external dependency
        checks.
        """
        return {"status": "ready"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Answer a question using the feedback insight agent."""
        try:
            result = agent.answer(
                request.question,
                top_k=request.top_k,
                filters=request.metadata_filters(),
            )
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("query_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return QueryResponse(result=result)

    @app.post("/query/stream")
    def query_stream(request: QueryRequest) -> StreamingResponse:
        """Answer a question as a Server-Sent Events stream.

        Emits ``content`` events whose JSON ``text`` fields concatenate to the
        same answer `/query` would return, followed by one final ``metadata``
        event carrying citations, sources, retrieval scores, the provider
        name, and the answer latency in milliseconds.

        Providers without true token streaming (such as the deterministic
        local provider) are supported by replaying the final answer in small
        chunks, so the endpoint works without external LLM APIs.
        """
        started = time.perf_counter()
        try:
            result = agent.answer(
                request.question,
                top_k=request.top_k,
                filters=request.metadata_filters(),
            )
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("query_stream_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        def event_stream() -> Iterator[str]:
            for chunk in _split_for_streaming(result.answer):
                yield _sse_event("content", {"text": chunk})
            yield _sse_event(
                "metadata",
                _build_stream_metadata(
                    result, provider=type(agent.llm).__name__, latency_ms=latency_ms
                ),
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Answer a message within a stored conversation.

        Omit ``conversation_id`` to start a new conversation; pass it back to
        continue the same conversation with previous turns as context.
        """
        try:
            result, conversation_id = agent.chat(
                request.message,
                store=conversation_store,
                conversation_id=request.conversation_id,
                top_k=request.top_k,
                filters=request.metadata_filters(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("chat_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(conversation_id=conversation_id, result=result)

    @app.get("/conversations/{conversation_id}", response_model=ConversationMemory)
    def get_conversation(conversation_id: str) -> ConversationMemory:
        """Return the stored turns of one conversation."""
        try:
            memory = conversation_store.get(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if memory is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return memory

    @app.post("/reports", response_model=SavedInsightReport, status_code=201)
    def save_report(request: SaveInsightReportRequest) -> SavedInsightReport:
        """Persist a generated insight answer as a saved report."""
        return report_store.save(request)

    @app.get("/reports", response_model=list[InsightReportSummary])
    def list_reports() -> list[InsightReportSummary]:
        """Return saved insight report summaries."""
        return report_store.list()

    @app.post("/reports/email-summary", response_model=EmailSummaryDelivery)
    def email_report_summary(request: EmailSummaryRequest) -> EmailSummaryDelivery:
        """Render or send an email digest for saved insight reports."""
        try:
            reports = select_reports_for_summary(
                report_store,
                report_ids=request.report_ids,
                max_reports=request.max_reports,
            )
            summary = render_email_summary(
                reports,
                recipients=request.recipients,
                subject=request.subject,
            )
            if not request.send:
                return EmailSummaryDelivery(summary=summary, sent=False)
            return deliver_email_summary(
                summary,
                smtp_host=settings.email_smtp_host,
                smtp_port=settings.email_smtp_port,
                from_address=settings.email_from_address,
                username=settings.email_smtp_username,
                password=settings.email_smtp_password,
                use_tls=settings.email_smtp_use_tls,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/reports/{report_id}", response_model=SavedInsightReport)
    def get_report(report_id: str) -> SavedInsightReport:
        """Return one saved insight report."""
        try:
            report = report_store.get(report_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(status_code=404, detail="report not found")
        return report

    @app.post("/answer-feedback", response_model=HumanFeedbackRecord, status_code=201)
    def submit_answer_feedback(
        request: SubmitHumanFeedbackRequest,
    ) -> HumanFeedbackRecord:
        """Persist a human judgement on a generated answer."""
        return human_feedback_store.save(request)

    @app.get("/answer-feedback", response_model=list[HumanFeedbackSummary])
    def list_answer_feedback() -> list[HumanFeedbackSummary]:
        """Return human feedback summaries."""
        return human_feedback_store.list()

    @app.get("/answer-feedback/{feedback_id}", response_model=HumanFeedbackRecord)
    def get_answer_feedback(feedback_id: str) -> HumanFeedbackRecord:
        """Return one human feedback record."""
        try:
            record = human_feedback_store.get(feedback_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="answer feedback not found")
        return record

    @app.post("/index")
    def index(request: IndexRequest) -> dict[str, str | int]:
        """Rebuild the local vector index from a CSV path."""
        index_path = request.index_path or str(settings.index_path)
        try:
            vector_store = build_index(
                request.input_path,
                index_path,
                embedding_dim=settings.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001 - convert to API-safe response.
            log_event("index_failed", {"error": str(exc)})
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "indexed", "chunks": vector_store.size, "index_path": index_path}

    @app.post(
        "/ingestion/jobs",
        response_model=JobSubmitResponse,
        status_code=202,
    )
    def submit_ingestion_job(
        request: JobRequest, background_tasks: BackgroundTasks
    ) -> JobSubmitResponse:
        """Submit an asynchronous ingestion job.

        Creates a ``pending`` job, schedules the ingestion pipeline (load +
        validate CSV, chunk, embed, persist the vector store) via
        ``BackgroundTasks``, and returns the job id immediately. Poll
        ``GET /ingestion/jobs/{job_id}`` for the terminal status.
        """
        job = job_store.create(request)
        background_tasks.add_task(
            run_ingestion_job,
            job.job_id,
            job_store,
            embedding_dim=settings.embedding_dim,
            default_index_path=str(settings.index_path),
        )
        return JobSubmitResponse(job_id=job.job_id, status=job.status.value)

    @app.get("/ingestion/jobs/{job_id}", response_model=JobResult)
    def get_ingestion_job(job_id: str) -> JobResult:
        """Return the status and result of an ingestion job; 404 if unknown."""
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    return app
