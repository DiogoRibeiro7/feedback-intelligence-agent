from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.evaluation import evaluate_system
from feedback_intelligence_agent.factory import build_telemetry
from feedback_intelligence_agent.ingestion import load_feedback_csv
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.retrieval import QueryEngine
from feedback_intelligence_agent.schemas import DocumentChunk, EvaluationCase
from feedback_intelligence_agent.telemetry import (
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    Telemetry,
    TelemetryEvent,
)
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

SAMPLE_CSV = Path("data/sample_feedback.csv")


def build_agent_with_sink() -> tuple[FeedbackInsightAgent, QueryEngine, InMemoryTelemetrySink]:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="1",
            source_id="fb-1",
            text="Onboarding checklist was unclear and setup took too long.",
            metadata={"rating": 2},
        ),
        DocumentChunk(
            chunk_id="2",
            source_id="fb-2",
            text="Pricing renewal was hard to explain to finance.",
            metadata={"rating": 2},
        ),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    query_engine = QueryEngine(embedding_model=model, vector_store=store)
    sink = InMemoryTelemetrySink()
    agent = FeedbackInsightAgent(
        query_engine=query_engine, llm=DeterministicLLM(), telemetry=Telemetry(sink=sink)
    )
    return agent, query_engine, sink


# ---------------------------------------------------------------------------
# Telemetry core
# ---------------------------------------------------------------------------


def test_telemetry_is_noop_without_sink() -> None:
    telemetry = Telemetry()
    assert not telemetry.enabled
    # Must not raise even though no sink is attached.
    telemetry.emit("agent_run_started", correlation_id="abc", metadata={"x": 1})
    with telemetry.span("retrieval_started", "retrieval_finished", correlation_id="abc"):
        pass


def test_telemetry_emit_records_event_fields() -> None:
    sink = InMemoryTelemetrySink()
    telemetry = Telemetry(sink=sink)
    assert telemetry.enabled
    telemetry.emit(
        "llm_call_finished",
        correlation_id="cid-1",
        duration_ms=12.5,
        metadata={"provider": "DeterministicLLM"},
    )
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.name == "llm_call_finished"
    assert event.correlation_id == "cid-1"
    assert event.duration_ms == 12.5
    assert event.metadata == {"provider": "DeterministicLLM"}
    assert event.timestamp  # ISO-8601 string, exact value not asserted.


def test_telemetry_span_emits_started_and_finished_with_duration() -> None:
    sink = InMemoryTelemetrySink()
    telemetry = Telemetry(sink=sink)
    correlation_id = telemetry.new_correlation_id()
    with telemetry.span(
        "retrieval_started",
        "retrieval_finished",
        correlation_id=correlation_id,
        metadata={"top_k": 4},
    ) as span:
        span["results"] = 2
    assert sink.event_names() == ["retrieval_started", "retrieval_finished"]
    started, finished = sink.events
    assert started.duration_ms is None
    assert finished.duration_ms is not None
    assert finished.duration_ms >= 0.0
    assert finished.correlation_id == correlation_id == started.correlation_id
    assert finished.metadata["results"] == 2
    assert finished.metadata["status"] == "ok"


def test_telemetry_span_records_error_and_reraises() -> None:
    sink = InMemoryTelemetrySink()
    telemetry = Telemetry(sink=sink)
    with (
        pytest.raises(ValueError, match="boom"),
        telemetry.span("llm_call_started", "llm_call_finished", correlation_id="cid-err"),
    ):
        raise ValueError("boom")
    assert sink.event_names() == ["llm_call_started", "llm_call_finished"]
    finished = sink.events[-1]
    assert finished.metadata["status"] == "error"
    assert finished.metadata["error"] == "boom"
    assert finished.duration_ms is not None


def test_new_correlation_ids_are_unique() -> None:
    telemetry = Telemetry()
    ids = {telemetry.new_correlation_id() for _ in range(50)}
    assert len(ids) == 50


# ---------------------------------------------------------------------------
# JSONL sink
# ---------------------------------------------------------------------------


def test_jsonl_sink_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "telemetry.jsonl"
    sink = JsonlTelemetrySink(path)
    telemetry = Telemetry(sink=sink)
    telemetry.emit("ingestion_started", correlation_id="cid-1", metadata={"path": "x.csv"})
    telemetry.emit("ingestion_finished", correlation_id="cid-1", duration_ms=3.2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["name"] for record in records] == ["ingestion_started", "ingestion_finished"]
    for record in records:
        assert set(record) == {"name", "timestamp", "correlation_id", "duration_ms", "metadata"}
        assert record["correlation_id"] == "cid-1"
    assert records[1]["duration_ms"] == 3.2


def test_jsonl_sink_appends_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    event = TelemetryEvent(name="agent_run_started", timestamp="t", correlation_id="c")
    JsonlTelemetrySink(path).emit(event)
    JsonlTelemetrySink(path).emit(event)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


# ---------------------------------------------------------------------------
# Integration: agent, ingestion, evaluation
# ---------------------------------------------------------------------------


def test_agent_run_emits_correlated_events_in_order() -> None:
    agent, _, sink = build_agent_with_sink()
    agent.answer("Why is onboarding slow?", top_k=1)
    assert sink.event_names() == [
        "agent_run_started",
        "retrieval_started",
        "retrieval_finished",
        "llm_call_started",
        "llm_call_finished",
        "agent_run_finished",
    ]
    correlation_ids = {event.correlation_id for event in sink.events}
    assert len(correlation_ids) == 1
    finished = {event.name: event for event in sink.events}
    assert finished["retrieval_finished"].metadata["results"] == 1
    assert finished["llm_call_finished"].metadata["provider"] == "DeterministicLLM"
    assert finished["llm_call_finished"].metadata["response_chars"] > 0
    assert finished["agent_run_finished"].metadata["route"] == "onboarding"
    assert finished["agent_run_finished"].metadata["citations"] >= 1
    for name in ("retrieval_finished", "llm_call_finished", "agent_run_finished"):
        duration = finished[name].duration_ms
        assert duration is not None
        assert duration >= 0.0


def test_agent_runs_use_distinct_correlation_ids() -> None:
    agent, _, sink = build_agent_with_sink()
    agent.answer("Why is onboarding slow?", top_k=1)
    agent.answer("Why is onboarding slow?", top_k=1)
    correlation_ids = {event.correlation_id for event in sink.events}
    assert len(correlation_ids) == 2


def test_ingestion_emits_started_and_finished_events() -> None:
    sink = InMemoryTelemetrySink()
    records = load_feedback_csv(SAMPLE_CSV, telemetry=Telemetry(sink=sink))
    assert sink.event_names() == ["ingestion_started", "ingestion_finished"]
    started, finished = sink.events
    assert started.metadata["path"] == str(SAMPLE_CSV)
    assert started.metadata["strict"] is True
    assert finished.metadata["records"] == len(records)
    assert finished.duration_ms is not None


def test_evaluation_emits_finished_event_with_aggregates() -> None:
    agent, query_engine, agent_sink = build_agent_with_sink()
    eval_sink = InMemoryTelemetrySink()
    cases = [
        EvaluationCase(
            question="Why is onboarding slow?",
            relevant_document_ids=["fb-1"],
            expected_keywords=["onboarding"],
            is_answerable=True,
        )
    ]
    report = evaluate_system(
        query_engine, agent, cases, top_k=1, telemetry=Telemetry(sink=eval_sink)
    )
    assert eval_sink.event_names() == ["evaluation_finished"]
    event = eval_sink.events[0]
    assert event.duration_ms is not None
    assert event.metadata["total_cases"] == report.total_cases == 1
    assert event.metadata["precision_at_k"] == report.retrieval.precision_at_k
    assert event.metadata["groundedness"] == report.answers.groundedness
    # The agent telemetry still captured the underlying run events.
    assert "agent_run_finished" in agent_sink.event_names()


# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------


def test_build_telemetry_disabled_by_default() -> None:
    telemetry = build_telemetry(Settings())
    assert not telemetry.enabled


def test_build_telemetry_enabled_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    settings = Settings(telemetry_enabled=True, telemetry_path=path)
    telemetry = build_telemetry(settings)
    assert telemetry.enabled
    telemetry.emit("agent_run_started", correlation_id="cid-1")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["name"] == "agent_run_started"


def test_settings_read_telemetry_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("FEEDBACK_AGENT_TELEMETRY_PATH", "traces/run.jsonl")
    settings = Settings()
    assert settings.telemetry_enabled is True
    assert settings.telemetry_path == Path("traces/run.jsonl")
