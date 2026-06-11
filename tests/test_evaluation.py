from __future__ import annotations

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.evaluation import evaluate_retrieval, evaluate_system
from ai_engineering_showcase.llm import DeterministicLLM
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.schemas import DocumentChunk, EvaluationCase
from ai_engineering_showcase.vector_store import InMemoryVectorStore


def build_query_engine_and_agent() -> tuple[QueryEngine, FeedbackInsightAgent]:
    model = HashingEmbeddingModel(dim=128)
    chunks = [
        DocumentChunk(
            chunk_id="1", source_id="fb-a", text="export failed during reporting", metadata={}
        ),
        DocumentChunk(
            chunk_id="2", source_id="fb-b", text="onboarding setup checklist", metadata={}
        ),
    ]
    store = InMemoryVectorStore(dim=128)
    store.add(chunks, model.embed([chunk.text for chunk in chunks]))
    query_engine = QueryEngine(embedding_model=model, vector_store=store)
    agent = FeedbackInsightAgent(query_engine=query_engine, llm=DeterministicLLM())
    return query_engine, agent


def test_evaluate_retrieval_returns_metrics() -> None:
    query_engine, _ = build_query_engine_and_agent()
    cases = [EvaluationCase(question="reporting export", relevant_source_ids=["fb-a"])]
    metrics = evaluate_retrieval(query_engine, cases, top_k=1)
    assert metrics.evaluated_cases == 1
    assert metrics.precision_at_k == 1.0


def test_evaluate_system_returns_full_report() -> None:
    query_engine, agent = build_query_engine_and_agent()
    cases = [EvaluationCase(question="onboarding checklist", relevant_source_ids=["fb-b"])]
    report = evaluate_system(query_engine, agent, cases, top_k=1)
    assert report.retrieval.evaluated_cases == 1
    assert report.answer_quality.evaluated_cases == 1
