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

from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import (
    build_agent,
    build_conversation_store,
    build_index,
    build_job_store,
)
from feedback_intelligence_agent.jobs import JobRequest, JobResult, run_ingestion_job
from feedback_intelligence_agent.memory import ConversationMemory
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

    app = FastAPI(
        title="Feedback Intelligence Agent API",
        version="0.1.0",
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
            result = agent.answer(request.question, top_k=request.top_k)
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
            result = agent.answer(request.question, top_k=request.top_k)
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
