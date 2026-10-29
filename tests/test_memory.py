from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.memory import (
    ConversationMemory,
    ConversationTurn,
    DeterministicQueryRewriter,
    InMemoryConversationStore,
    JsonConversationStore,
    LLMQueryRewriter,
    extract_entities,
    new_conversation_id,
)
from feedback_intelligence_agent.schemas import SearchResult


def make_turn(
    message: str = "Why are enterprise customers unhappy with onboarding?",
) -> ConversationTurn:
    return ConversationTurn(
        user_message=message,
        assistant_answer="The strongest signal is around onboarding [1].",
        retrieved_document_ids=["fb-001", "fb-007"],
        metadata={"route": "onboarding", "confidence": 0.65},
    )


def test_json_store_round_trip(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path / "conversations")
    memory = ConversationMemory(conversation_id="abc-123", turns=[make_turn()])
    store.save(memory)

    reloaded_store = JsonConversationStore(tmp_path / "conversations")
    loaded = reloaded_store.get("abc-123")
    assert loaded is not None
    assert loaded == memory
    assert loaded.turns[0].retrieved_document_ids == ["fb-001", "fb-007"]
    assert loaded.turns[0].metadata["route"] == "onboarding"
    assert loaded.turns[0].timestamp == memory.turns[0].timestamp
    assert reloaded_store.conversation_ids() == ["abc-123"]


def test_json_store_retains_tenant_scope_and_loads_legacy_history(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path)
    memory = ConversationMemory(conversation_id="scoped", tenant_id=" AcMe ", turns=[make_turn()])
    store.save(memory)
    loaded = JsonConversationStore(tmp_path).get("scoped")
    assert loaded is not None and loaded.tenant_id == "acme"
    (tmp_path / "legacy.json").write_text(
        '{"conversation_id":"legacy","turns":[]}', encoding="utf-8"
    )
    legacy = store.get("legacy")
    assert legacy is not None and legacy.tenant_id is None
    with pytest.raises(ValueError, match="blank"):
        ConversationMemory(conversation_id="blank", tenant_id=" ")


def test_json_store_returns_none_for_unknown_conversation(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path)
    assert store.get("missing") is None


def test_json_store_rejects_unsafe_conversation_ids(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path)
    with pytest.raises(ValueError, match="invalid conversation_id"):
        store.get("../escape")
    with pytest.raises(ValueError, match="invalid conversation_id"):
        store.save(ConversationMemory(conversation_id="bad id"))


def test_conversations_are_isolated_by_id(tmp_path: Path) -> None:
    store = JsonConversationStore(tmp_path)
    store.save(ConversationMemory(conversation_id="conv-a", turns=[make_turn("question a")]))
    store.save(ConversationMemory(conversation_id="conv-b", turns=[make_turn("question b")]))

    conv_a = store.get("conv-a")
    conv_b = store.get("conv-b")
    assert conv_a is not None and conv_b is not None
    assert [turn.user_message for turn in conv_a.turns] == ["question a"]
    assert [turn.user_message for turn in conv_b.turns] == ["question b"]
    assert store.conversation_ids() == ["conv-a", "conv-b"]


def test_in_memory_store_returns_independent_copies() -> None:
    store = InMemoryConversationStore()
    store.save(ConversationMemory(conversation_id="conv-a", turns=[make_turn()]))

    first = store.get("conv-a")
    assert first is not None
    first.add_turn(make_turn("a mutation that must not leak"))

    second = store.get("conv-a")
    assert second is not None
    assert len(second.turns) == 1


def test_new_conversation_ids_are_unique_and_valid() -> None:
    store = InMemoryConversationStore()
    first, second = new_conversation_id(), new_conversation_id()
    assert first != second
    store.save(ConversationMemory(conversation_id=first))  # Valid for any store.


def test_extract_entities_skips_stopwords_and_preserves_order() -> None:
    entities = extract_entities("Why are enterprise customers unhappy with onboarding?")
    assert entities == ["enterprise", "unhappy", "onboarding"]


def test_rewriter_expands_what_about_followup() -> None:
    rewriter = DeterministicQueryRewriter()
    rewrite = rewriter.rewrite("What about pricing?", make_turn())
    assert rewrite.was_rewritten
    assert rewrite.strategy == "ellipsis_expansion"
    assert "pricing" in rewrite.rewritten.lower()
    assert "onboarding" in rewrite.rewritten.lower()
    assert "enterprise" in rewrite.rewritten.lower()


def test_rewriter_resolves_pronoun_references() -> None:
    rewriter = DeterministicQueryRewriter()
    rewrite = rewriter.rewrite("Why do they complain about it?", make_turn())
    assert rewrite.was_rewritten
    assert rewrite.strategy == "pronoun_resolution"
    assert "enterprise" in rewrite.rewritten.lower()
    assert "they" not in rewrite.rewritten.lower().split()


def test_rewriter_keeps_standalone_question_unchanged() -> None:
    rewriter = DeterministicQueryRewriter()
    question = "Which integrations were requested most by finance teams recently?"
    rewrite = rewriter.rewrite(question, make_turn())
    assert not rewrite.was_rewritten
    assert rewrite.rewritten == question
    assert rewrite.strategy == "none"


def test_rewriter_keeps_determiner_pronouns_unchanged() -> None:
    rewriter = DeterministicQueryRewriter()
    question = "Why is that dashboard export failing for finance users today?"
    rewrite = rewriter.rewrite(question, make_turn())
    assert not rewrite.was_rewritten


def test_rewriter_without_previous_turn_is_a_noop() -> None:
    rewriter = DeterministicQueryRewriter()
    rewrite = rewriter.rewrite("What about pricing?", None)
    assert not rewrite.was_rewritten
    assert rewrite.rewritten == "What about pricing?"


class _StubRewriteLLM:
    """Minimal LLM provider stub returning a fixed standalone question."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, question: str, results: list[SearchResult]) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_llm_rewriter_delegates_to_provider() -> None:
    llm = _StubRewriteLLM("Why are enterprise customers unhappy with pricing?")
    rewriter = LLMQueryRewriter(llm)
    rewrite = rewriter.rewrite("What about pricing?", make_turn())
    assert rewrite.was_rewritten
    assert rewrite.strategy == "llm"
    assert rewrite.rewritten == "Why are enterprise customers unhappy with pricing?"
    assert "What about pricing?" in llm.prompts[0]


def test_llm_rewriter_without_previous_turn_is_a_noop() -> None:
    llm = _StubRewriteLLM("unused")
    rewriter = LLMQueryRewriter(llm)
    rewrite = rewriter.rewrite("What about pricing?", None)
    assert not rewrite.was_rewritten
    assert llm.prompts == []
