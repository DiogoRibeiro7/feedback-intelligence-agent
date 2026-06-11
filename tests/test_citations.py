from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.api import create_app
from ai_engineering_showcase.citations import (
    build_citations,
    citation_marker,
    render_citations,
    summarize_evidence,
)
from ai_engineering_showcase.cli import app
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.llm import DeterministicLLM
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.schemas import DocumentChunk, SearchResult
from ai_engineering_showcase.vector_store import InMemoryVectorStore

runner = CliRunner()

MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def make_chunk(chunk_id: str, source_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        metadata={"channel": "support_ticket", "rating": 2},
    )


def make_results() -> list[SearchResult]:
    return [
        SearchResult(chunk=make_chunk("fb-1::chunk-0", "fb-1", "onboarding setup slow"), score=0.9),
        SearchResult(chunk=make_chunk("fb-1::chunk-1", "fb-1", "still onboarding"), score=0.8),
        SearchResult(chunk=make_chunk("fb-2::chunk-0", "fb-2", "pricing was unclear"), score=0.7),
    ]


class RecordingRetriever:
    """Retriever wrapper that records every chunk it returned."""

    def __init__(self, inner: QueryEngine) -> None:
        self.inner = inner
        self.returned_chunk_ids: set[str] = set()
        self.returned_document_ids: set[str] = set()

    def search(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        results = self.inner.search(query, top_k=top_k)
        self.returned_chunk_ids.update(result.chunk.chunk_id for result in results)
        self.returned_document_ids.update(result.chunk.source_id for result in results)
        return results


def build_agent_with_recorder() -> tuple[FeedbackInsightAgent, RecordingRetriever]:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        make_chunk("fb-1::chunk-0", "fb-1", "Onboarding checklist was unclear and setup slow."),
        make_chunk("fb-2::chunk-0", "fb-2", "Pricing renewal was hard to explain to finance."),
        make_chunk("fb-3::chunk-0", "fb-3", "Support tickets stayed open for two weeks."),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    recorder = RecordingRetriever(QueryEngine(embedding_model=model, vector_store=store))
    return FeedbackInsightAgent(query_engine=recorder, llm=DeterministicLLM()), recorder


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------


def test_build_citations_assigns_sequential_ids_and_dedupes_documents() -> None:
    citations = build_citations(make_results())
    assert [citation.citation_id for citation in citations] == [1, 2]
    assert [citation.document_id for citation in citations] == ["fb-1", "fb-2"]
    # The first (highest-ranked) chunk of each document is the cited evidence.
    assert citations[0].chunk_id == "fb-1::chunk-0"
    assert citations[0].source == "support_ticket"
    assert citations[0].quote == "onboarding setup slow"


def test_build_citations_is_deterministic() -> None:
    assert build_citations(make_results()) == build_citations(make_results())


def test_build_citations_empty_results() -> None:
    assert build_citations([]) == []


def test_summarize_evidence_truncates_long_text() -> None:
    quote = summarize_evidence("word " * 100, max_chars=40)
    assert len(quote) == 40
    assert quote.endswith("…")


def test_render_citations_includes_markers_and_metadata() -> None:
    rendered = render_citations(build_citations(make_results()))
    assert "[1] fb-1 (support_ticket, chunk fb-1::chunk-0" in rendered
    assert "[2] fb-2" in rendered
    assert render_citations([]) == "Citations: none (no evidence retrieved)"


# ---------------------------------------------------------------------------
# Agent-level guarantees
# ---------------------------------------------------------------------------


def test_agent_citation_ids_are_stable_across_runs() -> None:
    agent, _ = build_agent_with_recorder()
    first = agent.answer("Why is onboarding slow?", top_k=2)
    second = agent.answer("Why is onboarding slow?", top_k=2)
    assert first.citations == second.citations
    assert [citation.citation_id for citation in first.citations] == list(
        range(1, len(first.citations) + 1)
    )


def test_agent_citations_refer_to_retrieved_chunks() -> None:
    agent, recorder = build_agent_with_recorder()
    answer = agent.answer("Why is onboarding slow?", top_k=2)
    assert answer.citations
    for citation in answer.citations:
        assert citation.chunk_id in recorder.returned_chunk_ids
        assert citation.document_id in recorder.returned_document_ids


def test_agent_does_not_cite_documents_that_were_not_retrieved() -> None:
    agent, _ = build_agent_with_recorder()
    answer = agent.answer("Why is onboarding slow?", top_k=2)
    indexed_documents = {"fb-1", "fb-2", "fb-3"}
    cited_documents = {citation.document_id for citation in answer.citations}
    assert cited_documents <= indexed_documents
    # Every marker in the answer text resolves to a real citation entry.
    valid_ids = {citation.citation_id for citation in answer.citations}
    markers = {int(match) for match in MARKER_PATTERN.findall(answer.answer)}
    assert markers
    assert markers <= valid_ids


def test_agent_answer_contains_citation_markers() -> None:
    agent, _ = build_agent_with_recorder()
    answer = agent.answer("Why is onboarding slow?", top_k=2)
    assert "[1]" in answer.answer
    assert answer.citations[0].citation_id == 1


# ---------------------------------------------------------------------------
# Deterministic provider
# ---------------------------------------------------------------------------


def test_deterministic_llm_emits_aligned_markers() -> None:
    llm = DeterministicLLM()
    results = make_results()
    response = llm.generate("prompt", question="Why is onboarding slow?", results=results)
    assert response == llm.generate("prompt", question="Why is onboarding slow?", results=results)
    expected = build_citations(results)
    for citation in expected:
        assert f"{citation.document_id} {citation_marker(citation.citation_id)}" in response
    markers = {int(match) for match in MARKER_PATTERN.findall(response)}
    assert markers <= {citation.citation_id for citation in expected}


def test_deterministic_llm_no_results_has_no_markers() -> None:
    response = DeterministicLLM().generate("prompt", question="anything", results=[])
    assert not MARKER_PATTERN.findall(response)
    assert "could not find enough evidence" in response.lower()


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------


def test_query_command_renders_citation_block(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "query",
            "Why are enterprise customers unhappy with onboarding?",
            "--index-path",
            str(tmp_path / "vector_store.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"citation_id": 1' in result.output
    assert '"document_id"' in result.output
    assert "Citations:" in result.output
    assert "[1]" in result.output


# ---------------------------------------------------------------------------
# API response metadata
# ---------------------------------------------------------------------------


def test_api_query_response_includes_citation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_SHOWCASE_INDEX_PATH", str(tmp_path / "vector_store.json"))
    client = TestClient(create_app())
    response = client.post(
        "/query",
        json={"question": "Why are enterprise customers unhappy with onboarding?", "top_k": 4},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert "[1]" in result["answer"]
    citations = result["citations"]
    assert citations
    for index, citation in enumerate(citations, start=1):
        assert citation["citation_id"] == index
        assert set(citation) == {
            "citation_id",
            "document_id",
            "chunk_id",
            "source",
            "quote",
            "score",
        }
