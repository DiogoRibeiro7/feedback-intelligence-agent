from __future__ import annotations

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.llm import DeterministicLLM
from feedback_intelligence_agent.memory import ConversationTurn, InMemoryConversationStore
from feedback_intelligence_agent.retrieval import QueryEngine
from feedback_intelligence_agent.schemas import DocumentChunk
from feedback_intelligence_agent.vector_store import InMemoryVectorStore


def build_test_agent() -> FeedbackInsightAgent:
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
    return FeedbackInsightAgent(query_engine=query_engine, llm=DeterministicLLM())


def test_agent_routes_onboarding_question() -> None:
    agent = build_test_agent()
    assert agent.route("What is wrong with onboarding?") == "onboarding"


def test_agent_answer_contains_citations() -> None:
    agent = build_test_agent()
    answer = agent.answer("Why is onboarding slow?", top_k=1)
    assert answer.citations
    assert answer.confidence > 0
    assert answer.recommended_actions
    assert answer.diagnostics["reranker"] == "DeterministicJudgeReranker"


def test_agent_single_turn_answer_has_no_memory_diagnostics() -> None:
    agent = build_test_agent()
    answer = agent.answer("Why is onboarding slow?", top_k=1)
    assert "query_rewritten" not in answer.diagnostics
    assert "retrieval_question" not in answer.diagnostics
    assert "rewrite_strategy" not in answer.diagnostics


def test_agent_answer_with_history_rewrites_followup_for_retrieval() -> None:
    agent = build_test_agent()
    history = [
        ConversationTurn(
            user_message="Why is onboarding slow for enterprise customers?",
            assistant_answer="Onboarding friction dominates.",
            retrieved_document_ids=["fb-1"],
        )
    ]
    answer = agent.answer("What about pricing?", top_k=2, history=history)
    assert answer.question == "What about pricing?"
    assert answer.diagnostics["query_rewritten"] is True
    retrieval_question = str(answer.diagnostics["retrieval_question"])
    assert "pricing" in retrieval_question.lower()
    assert "onboarding" in retrieval_question.lower()


def test_agent_answer_with_empty_history_matches_single_turn() -> None:
    agent = build_test_agent()
    single = agent.answer("Why is onboarding slow?", top_k=1)
    with_history = agent.answer("Why is onboarding slow?", top_k=1, history=[])
    assert with_history.model_dump() == single.model_dump()


def test_agent_chat_persists_turns_and_uses_previous_context() -> None:
    agent = build_test_agent()
    store = InMemoryConversationStore()
    first_answer, conversation_id = agent.chat("Why is onboarding slow?", store=store, top_k=1)
    assert first_answer.citations
    second_answer, same_id = agent.chat(
        "What about pricing?", store=store, conversation_id=conversation_id, top_k=1
    )
    assert same_id == conversation_id
    assert second_answer.diagnostics["query_rewritten"] is True

    memory = store.get(conversation_id)
    assert memory is not None
    assert [turn.user_message for turn in memory.turns] == [
        "Why is onboarding slow?",
        "What about pricing?",
    ]
    assert memory.turns[0].retrieved_document_ids == ["fb-1"]
    assert memory.turns[1].metadata["retrieval_question"]


def test_agent_chat_isolates_conversations() -> None:
    agent = build_test_agent()
    store = InMemoryConversationStore()
    _, first_id = agent.chat("Why is onboarding slow?", store=store, top_k=1)
    _, second_id = agent.chat("Why is pricing renewal hard?", store=store, top_k=1)
    assert first_id != second_id
    first = store.get(first_id)
    second = store.get(second_id)
    assert first is not None and len(first.turns) == 1
    assert second is not None and len(second.turns) == 1
    assert first.turns[0].user_message != second.turns[0].user_message
