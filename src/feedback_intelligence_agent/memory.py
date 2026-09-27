"""Conversation memory and follow-up query rewriting.

The module provides three building blocks:

1. Typed conversation state (:class:`ConversationTurn`, :class:`ConversationMemory`)
   that records what the user asked, what the agent answered, and which
   documents grounded the answer.
2. Pluggable persistence (:class:`InMemoryConversationStore` for tests and
   :class:`JsonConversationStore` for local JSON files under ``.artifacts/``).
3. Deterministic follow-up query rewriting (:class:`DeterministicQueryRewriter`)
   that converts elliptical follow-ups ("What about pricing?") and pronoun
   references ("Why do they complain about it?") into standalone questions
   using only entities from the previous turn. Only the rewritten standalone
   question reaches retrieval, so the index is never queried with the full
   conversation history. An optional :class:`LLMQueryRewriter` can delegate
   rewriting to an LLM provider, but the deterministic local rewriter is the
   default and requires no external APIs.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from feedback_intelligence_agent.llm import LLMProvider
from feedback_intelligence_agent.safe_paths import json_record_path

_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_ENTITY_STOPWORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "can",
        "could",
        "customer",
        "customers",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "is",
        "more",
        "our",
        "please",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)

# Standalone pronouns that refer back to the previous turn. ``this/that/these/
# those`` are only treated as references when not used as determiners
# ("that dashboard" keeps its noun and is not rewritten).
_PRONOUN_PATTERN = re.compile(
    r"\b(?:it|they|them)\b|\b(?:this|that|these|those)\b(?!\s+[A-Za-z])",
    re.IGNORECASE,
)

# Elliptical follow-up openers such as "What about pricing?" or "And exports?".
_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:what|how)\s+about\b|^\s*(?:and|also)\b|^\s*what\s+else\b",
    re.IGNORECASE,
)


def new_conversation_id() -> str:
    """Return a fresh URL-safe conversation identifier."""
    return uuid.uuid4().hex


class ConversationTurn(BaseModel):
    """One user/assistant exchange recorded in conversation memory.

    Attributes:
        user_message: The question exactly as the user asked it.
        assistant_answer: The answer text returned by the agent.
        retrieved_document_ids: IDs of the documents cited by the answer.
        timestamp: UTC time at which the turn was recorded.
        metadata: Optional structured attributes (route, confidence, the
            rewritten retrieval question, etc.).
    """

    user_message: str
    assistant_answer: str
    retrieved_document_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemory(BaseModel):
    """Ordered turns of one conversation identified by ``conversation_id``."""

    conversation_id: str = Field(min_length=1)
    turns: list[ConversationTurn] = Field(default_factory=list)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append one completed turn to the conversation."""
        self.turns.append(turn)

    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent turn, or ``None`` for a new conversation."""
        return self.turns[-1] if self.turns else None


class ConversationStore(Protocol):
    """Protocol implemented by conversation persistence backends."""

    def get(self, conversation_id: str) -> ConversationMemory | None:
        """Return the stored conversation, or ``None`` when unknown."""
        ...

    def save(self, memory: ConversationMemory) -> None:
        """Persist the full conversation state."""
        ...

    def conversation_ids(self) -> list[str]:
        """Return the identifiers of all stored conversations, sorted."""
        ...


def _validate_conversation_id(conversation_id: str) -> str:
    """Validate a conversation identifier and return it unchanged.

    IDs are restricted to a filesystem-safe alphabet so the JSON store can use
    them directly as file names without path traversal risks.
    """
    if not _CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError(
            f"invalid conversation_id {conversation_id!r}: expected 1-64 characters "
            "from [A-Za-z0-9._-] starting with a letter or digit"
        )
    return conversation_id


class InMemoryConversationStore:
    """Dict-backed conversation store intended for tests and ephemeral use."""

    def __init__(self) -> None:
        """Initialise the empty store."""
        self._conversations: dict[str, ConversationMemory] = {}

    def get(self, conversation_id: str) -> ConversationMemory | None:
        """Return a deep copy of the stored conversation, or ``None``."""
        _validate_conversation_id(conversation_id)
        memory = self._conversations.get(conversation_id)
        return memory.model_copy(deep=True) if memory is not None else None

    def save(self, memory: ConversationMemory) -> None:
        """Store a deep copy of the conversation keyed by its ID."""
        _validate_conversation_id(memory.conversation_id)
        self._conversations[memory.conversation_id] = memory.model_copy(deep=True)

    def conversation_ids(self) -> list[str]:
        """Return the identifiers of all stored conversations, sorted."""
        return sorted(self._conversations)


class JsonConversationStore:
    """Conversation store that persists one JSON file per conversation.

    Files are written as ``{root}/{conversation_id}.json`` with indented JSON,
    so conversations are easy to inspect, diff, and replay locally.
    """

    def __init__(self, root: str | Path) -> None:
        """Create the store and ensure the root directory exists."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        """Return the JSON file path for a validated conversation ID."""
        return json_record_path(self.root, _validate_conversation_id(conversation_id))

    def get(self, conversation_id: str) -> ConversationMemory | None:
        """Load a conversation from its JSON file, or ``None`` when missing."""
        path = self._path(conversation_id)
        if not path.exists():
            return None
        return ConversationMemory.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, memory: ConversationMemory) -> None:
        """Write the conversation as an indented JSON file."""
        path = self._path(memory.conversation_id)
        path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")

    def conversation_ids(self) -> list[str]:
        """Return the identifiers of all persisted conversations, sorted."""
        return sorted(path.stem for path in self.root.glob("*.json"))


class QueryRewrite(BaseModel):
    """Result of rewriting a follow-up question into a standalone question.

    ``strategy`` documents which deterministic rule fired (``none``,
    ``pronoun_resolution``, ``ellipsis_expansion``, or ``llm``), keeping the
    rewriting step fully transparent and auditable.
    """

    original: str
    rewritten: str
    was_rewritten: bool
    strategy: str = "none"


class QueryRewriter(Protocol):
    """Protocol for converting follow-up questions into standalone questions."""

    def rewrite(self, question: str, previous_turn: ConversationTurn | None) -> QueryRewrite:
        """Rewrite ``question`` using context from the previous turn, if needed."""
        ...


def extract_entities(text: str, *, limit: int = 4) -> list[str]:
    """Extract salient content tokens from text, preserving first-seen order.

    Tokens are lowercased, at least three characters long, and filtered
    against a small stopword list, so the result approximates the entities
    and topics mentioned in the previous turn.
    """
    entities: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
        if token in _ENTITY_STOPWORDS or token in entities:
            continue
        entities.append(token)
        if len(entities) >= limit:
            break
    return entities


class DeterministicQueryRewriter:
    """Local, deterministic follow-up rewriter with two transparent rules.

    1. **Pronoun resolution**: standalone pronouns (``it``, ``they``, ``them``,
       trailing ``this``/``that``/...) are replaced with the salient entities
       of the previous user message.
    2. **Ellipsis expansion**: elliptical openers ("What about X?", "And X?")
       and very short questions get the previous turn's missing entities
       appended ("... regarding enterprise onboarding").

    Standalone questions are returned unchanged, so single-turn behaviour is
    never affected. No model call is involved.
    """

    short_question_tokens: int = 4

    def rewrite(self, question: str, previous_turn: ConversationTurn | None) -> QueryRewrite:
        """Rewrite a follow-up question using entities from the previous turn."""
        unchanged = QueryRewrite(
            original=question, rewritten=question, was_rewritten=False, strategy="none"
        )
        if previous_turn is None:
            return unchanged
        entities = extract_entities(previous_turn.user_message)
        if not entities:
            return unchanged

        if _PRONOUN_PATTERN.search(question):
            replacement = " ".join(entities[:2])
            rewritten = _PRONOUN_PATTERN.sub(replacement, question)
            rewritten = " ".join(rewritten.split())
            return QueryRewrite(
                original=question,
                rewritten=rewritten,
                was_rewritten=True,
                strategy="pronoun_resolution",
            )

        token_count = len(re.findall(r"[A-Za-z0-9_-]+", question))
        if _FOLLOW_UP_PATTERN.search(question) or token_count <= self.short_question_tokens:
            lower_question = question.lower()
            missing = [entity for entity in entities if entity not in lower_question]
            if missing:
                base = question.strip().rstrip("?!. ")
                rewritten = f"{base} regarding {' '.join(missing)}?"
                return QueryRewrite(
                    original=question,
                    rewritten=rewritten,
                    was_rewritten=True,
                    strategy="ellipsis_expansion",
                )
        return unchanged


class LLMQueryRewriter:
    """Optional rewriter that delegates follow-up rewriting to an LLM provider.

    Intended for real LLM providers (e.g. the OpenAI provider); the
    deterministic local provider is not suitable here because it answers from
    retrieval results. The deterministic rewriter remains the project default.
    """

    def __init__(self, llm: LLMProvider) -> None:
        """Bind the rewriter to a text-generation provider."""
        self.llm = llm

    def rewrite(self, question: str, previous_turn: ConversationTurn | None) -> QueryRewrite:
        """Ask the LLM to produce a standalone version of a follow-up question."""
        if previous_turn is None:
            return QueryRewrite(
                original=question, rewritten=question, was_rewritten=False, strategy="none"
            )
        prompt = (
            "Rewrite the follow-up question as one standalone question that can be "
            "understood without the conversation. Reply with the question only.\n\n"
            f"Previous question: {previous_turn.user_message}\n"
            f"Previous answer: {previous_turn.assistant_answer}\n"
            f"Follow-up question: {question}\n"
            "Standalone question:"
        )
        rewritten = self.llm.generate(prompt, question=question, results=[]).strip()
        if not rewritten or rewritten == question:
            return QueryRewrite(
                original=question, rewritten=question, was_rewritten=False, strategy="none"
            )
        return QueryRewrite(
            original=question, rewritten=rewritten, was_rewritten=True, strategy="llm"
        )
