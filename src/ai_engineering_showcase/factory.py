"""Factories for constructing the application components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.ingestion import load_feedback_csv
from ai_engineering_showcase.lexical_search import BM25Retriever
from ai_engineering_showcase.llm import (
    AnthropicLLM,
    DeterministicLLM,
    LLMProvider,
    OllamaLLM,
    OpenAIChatLLM,
)
from ai_engineering_showcase.memory import JsonConversationStore
from ai_engineering_showcase.retrieval import HybridRetriever, QueryEngine, Retriever
from ai_engineering_showcase.schemas import DocumentChunk
from ai_engineering_showcase.telemetry import JsonlTelemetrySink, Telemetry
from ai_engineering_showcase.tools import build_default_tools
from ai_engineering_showcase.vector_store import InMemoryVectorStore, VectorStore

if TYPE_CHECKING:
    from ai_engineering_showcase.jobs import JsonJobStore


def build_telemetry(settings: Settings) -> Telemetry:
    """Construct the telemetry emitter configured by the settings.

    Telemetry is disabled (a no-op emitter) unless ``AI_SHOWCASE_TELEMETRY_ENABLED``
    is set; when enabled, events are appended to the JSONL file configured by
    ``AI_SHOWCASE_TELEMETRY_PATH``.
    """
    if not settings.telemetry_enabled:
        return Telemetry()
    return Telemetry(sink=JsonlTelemetrySink(settings.telemetry_path))


def build_conversation_store(settings: Settings) -> JsonConversationStore:
    """Construct the JSON-backed conversation store configured by the settings.

    Conversations are persisted as one JSON file each under
    ``AI_SHOWCASE_CONVERSATION_STORE_PATH`` (default ``.artifacts/conversations``).
    """
    return JsonConversationStore(settings.conversation_store_path)


def build_job_store(settings: Settings) -> JsonJobStore:
    """Construct the JSON-backed ingestion job store configured by the settings.

    Jobs are persisted as one JSON file each under ``AI_SHOWCASE_JOB_STORE_PATH``
    (default ``.artifacts/jobs``), so submitted ingestion jobs survive restarts
    and are easy to inspect. Imported lazily to avoid a circular import, since
    :mod:`ai_engineering_showcase.jobs` reuses :func:`chunk_to_embedding_text`.
    """
    from ai_engineering_showcase.jobs import JsonJobStore

    return JsonJobStore(settings.job_store_path)


def chunk_to_embedding_text(chunk: DocumentChunk) -> str:
    """Create the text representation used for embedding and retrieval.

    The visible citation keeps the original feedback text, but retrieval benefits
    from structured metadata such as segment, channel, rating, and date.
    """
    metadata = chunk.metadata
    metadata_text = " ".join(
        [
            f"segment {metadata.get('customer_segment', '')}",
            f"channel {metadata.get('channel', '')}",
            f"rating {metadata.get('rating', '')}",
            f"created {metadata.get('created_at', '')}",
        ]
    )
    return f"{metadata_text} {chunk.text}"


def build_index(
    input_path: str | Path,
    index_path: str | Path,
    *,
    embedding_dim: int,
    telemetry: Telemetry | None = None,
) -> InMemoryVectorStore:
    """Build and persist a vector index from feedback CSV data."""
    records = load_feedback_csv(input_path, telemetry=telemetry)
    chunks = feedback_to_chunks(records)
    embedding_model = HashingEmbeddingModel(dim=embedding_dim)
    vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
    vector_store = InMemoryVectorStore(dim=embedding_dim)
    vector_store.add(chunks, vectors)
    vector_store.save(index_path)
    return vector_store


def build_qdrant_index(settings: Settings, *, telemetry: Telemetry | None = None) -> VectorStore:
    """Build (or refresh) a Qdrant-backed index from the configured data.

    The Qdrant store is imported lazily here so the optional ``qdrant-client``
    dependency is only required when ``AI_SHOWCASE_VECTOR_STORE=qdrant``.
    """
    from ai_engineering_showcase.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(
        dim=settings.embedding_dim,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    if store.size == 0:
        records = load_feedback_csv(settings.data_path, telemetry=telemetry)
        chunks = feedback_to_chunks(records)
        embedding_model = HashingEmbeddingModel(dim=settings.embedding_dim)
        vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
        store.add(chunks, vectors)
    return store


def load_or_build_index(settings: Settings, *, telemetry: Telemetry | None = None) -> VectorStore:
    """Load the configured index, building it from data when needed.

    With the default ``json`` store the index is loaded from (or built and
    persisted to) ``AI_SHOWCASE_INDEX_PATH``. With ``qdrant`` the index lives in
    the configured Qdrant collection.
    """
    if settings.vector_store == "qdrant":
        return build_qdrant_index(settings, telemetry=telemetry)
    settings.ensure_artifact_dir()
    if settings.index_path.exists():
        return InMemoryVectorStore.load(settings.index_path)
    return build_index(
        settings.data_path,
        settings.index_path,
        embedding_dim=settings.embedding_dim,
        telemetry=telemetry,
    )


def build_retriever(settings: Settings, vector_store: VectorStore) -> Retriever:
    """Construct the configured retriever over an existing vector store.

    ``dense`` keeps the original vector-similarity behaviour, ``lexical`` uses
    the local BM25 index, and ``hybrid`` combines both with the configured
    ``dense_weight`` and ``lexical_weight``.
    """
    embedding_model = HashingEmbeddingModel(dim=vector_store.dim)
    dense = QueryEngine(embedding_model=embedding_model, vector_store=vector_store)
    if settings.retriever_type == "dense":
        return dense
    chunks = vector_store.chunks
    lexical = BM25Retriever(chunks, texts=[chunk_to_embedding_text(chunk) for chunk in chunks])
    if settings.retriever_type == "lexical":
        return lexical
    return HybridRetriever(
        dense=dense,
        lexical=lexical,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
    )


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the LLM provider selected by ``AI_SHOWCASE_LLM_PROVIDER``.

    ``local`` (the default) needs no API key and stays fully deterministic.
    ``openai`` targets any OpenAI-compatible endpoint (``OPENAI_BASE_URL``),
    ``anthropic`` uses the official SDK (optional ``anthropic`` extra), and
    ``ollama`` talks to a local Ollama server. Missing credentials raise a
    clear configuration error at construction time.
    """
    if settings.llm_provider == "local":
        return DeterministicLLM()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_SHOWCASE_LLM_PROVIDER=openai")
        return OpenAIChatLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when AI_SHOWCASE_LLM_PROVIDER=anthropic"
            )
        return AnthropicLLM(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    if settings.llm_provider == "ollama":
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model)
    raise ValueError(
        f"Unknown LLM provider {settings.llm_provider!r}. "
        "Valid options: local, openai, anthropic, ollama."
    )


def build_agent(settings: Settings, *, telemetry: Telemetry | None = None) -> FeedbackInsightAgent:
    """Construct a fully wired feedback insight agent.

    When no telemetry emitter is supplied, one is built from the settings
    (a no-op unless telemetry is enabled via the environment). The agent is
    equipped with the default local tool registry built over the indexed
    feedback chunks.
    """
    telemetry = telemetry or build_telemetry(settings)
    vector_store = load_or_build_index(settings, telemetry=telemetry)
    retriever = build_retriever(settings, vector_store)
    return FeedbackInsightAgent(
        query_engine=retriever,
        llm=build_llm(settings),
        telemetry=telemetry,
        tools=build_default_tools(vector_store.chunks),
    )
