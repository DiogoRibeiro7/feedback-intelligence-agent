"""Factories for constructing the application components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.chunking import feedback_to_chunks
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.embeddings import HashingEmbeddingModel
from feedback_intelligence_agent.hallucination import (
    EvidenceOverlapHallucinationChecker,
    LLMHallucinationJudge,
)
from feedback_intelligence_agent.ingestion import load_feedback_csv
from feedback_intelligence_agent.lexical_search import BM25Retriever
from feedback_intelligence_agent.llm import (
    AnthropicLLM,
    BedrockConverseLLM,
    DeterministicLLM,
    LLMProvider,
    OllamaLLM,
    OpenAIChatLLM,
    OpenAIResponsesLLM,
)
from feedback_intelligence_agent.memory import JsonConversationStore
from feedback_intelligence_agent.query_expansion import ProductTerminologyExpander
from feedback_intelligence_agent.resilience import ResiliencePolicy, ResilientLLMProvider
from feedback_intelligence_agent.retrieval import HybridRetriever, QueryEngine, Retriever
from feedback_intelligence_agent.schemas import DocumentChunk
from feedback_intelligence_agent.telemetry import (
    JsonlTelemetrySink,
    OpenTelemetryTraceSink,
    Telemetry,
)
from feedback_intelligence_agent.tools import build_default_tools
from feedback_intelligence_agent.vector_store import InMemoryVectorStore, VectorStore

if TYPE_CHECKING:
    from feedback_intelligence_agent.active_learning import JsonActiveLearningStateStore
    from feedback_intelligence_agent.human_feedback import JsonHumanFeedbackStore
    from feedback_intelligence_agent.jobs import JsonJobStore
    from feedback_intelligence_agent.reports import JsonInsightReportStore


def build_telemetry(settings: Settings) -> Telemetry:
    """Construct the telemetry emitter configured by the settings.

    Telemetry is disabled (a no-op emitter) unless ``FEEDBACK_AGENT_TELEMETRY_ENABLED``
    is set. The default enabled backend appends events to the JSONL file configured
    by ``FEEDBACK_AGENT_TELEMETRY_PATH``; ``opentelemetry`` emits spans through the
    process-global OpenTelemetry tracer provider.
    """
    if not settings.telemetry_enabled:
        return Telemetry()
    if settings.telemetry_backend == "opentelemetry":
        return Telemetry(sink=OpenTelemetryTraceSink(settings.telemetry_service_name))
    return Telemetry(sink=JsonlTelemetrySink(settings.telemetry_path))


def build_conversation_store(settings: Settings) -> JsonConversationStore:
    """Construct the JSON-backed conversation store configured by the settings.

    Conversations are persisted as one JSON file each under
    ``FEEDBACK_AGENT_CONVERSATION_STORE_PATH`` (default ``.artifacts/conversations``).
    """
    return JsonConversationStore(settings.conversation_store_path)


def build_job_store(settings: Settings) -> JsonJobStore:
    """Construct the JSON-backed ingestion job store configured by the settings.

    Jobs are persisted as one JSON file each under ``FEEDBACK_AGENT_JOB_STORE_PATH``
    (default ``.artifacts/jobs``), so submitted ingestion jobs survive restarts
    and are easy to inspect. Imported lazily to avoid a circular import, since
    :mod:`feedback_intelligence_agent.jobs` reuses :func:`chunk_to_embedding_text`.
    """
    from feedback_intelligence_agent.jobs import JsonJobStore

    return JsonJobStore(settings.job_store_path)


def build_report_store(settings: Settings) -> JsonInsightReportStore:
    """Construct the JSON-backed saved insight report store."""
    from feedback_intelligence_agent.reports import JsonInsightReportStore

    return JsonInsightReportStore(settings.report_store_path)


def build_human_feedback_store(settings: Settings) -> JsonHumanFeedbackStore:
    """Construct the JSON-backed human answer feedback store."""
    from feedback_intelligence_agent.human_feedback import JsonHumanFeedbackStore

    return JsonHumanFeedbackStore(settings.human_feedback_store_path)


def build_active_learning_state_store(settings: Settings) -> JsonActiveLearningStateStore:
    """Construct the JSON-backed active-learning workflow state store."""
    from feedback_intelligence_agent.active_learning import JsonActiveLearningStateStore

    return JsonActiveLearningStateStore(settings.active_learning_state_store_path)


def chunk_to_embedding_text(chunk: DocumentChunk) -> str:
    """Create the text representation used for embedding and retrieval.

    The visible citation keeps the original feedback text, but retrieval benefits
    from structured metadata such as segment, channel, rating, and date.
    """
    metadata = chunk.metadata
    metadata_text = " ".join(
        [
            f"tenant {metadata.get('tenant_id', 'default')}",
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
    telemetry = telemetry or Telemetry()
    correlation_id = telemetry.new_correlation_id()
    with telemetry.span(
        "embedding_started",
        "embedding_finished",
        correlation_id=correlation_id,
        metadata={
            "model": type(embedding_model).__name__,
            "embedding_dim": embedding_dim,
            "chunks": len(chunks),
        },
    ) as span:
        vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
        span["vectors"] = len(vectors)
    vector_store = InMemoryVectorStore(dim=embedding_dim)
    vector_store.add(chunks, vectors)
    vector_store.save(index_path)
    return vector_store


def build_qdrant_index(settings: Settings, *, telemetry: Telemetry | None = None) -> VectorStore:
    """Build (or refresh) a Qdrant-backed index from the configured data.

    The Qdrant store is imported lazily here so the optional ``qdrant-client``
    dependency is only required when ``FEEDBACK_AGENT_VECTOR_STORE=qdrant``.
    """
    from feedback_intelligence_agent.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(
        dim=settings.embedding_dim,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    if store.size == 0:
        records = load_feedback_csv(settings.data_path, telemetry=telemetry)
        chunks = feedback_to_chunks(records)
        embedding_model = HashingEmbeddingModel(dim=settings.embedding_dim)
        telemetry = telemetry or Telemetry()
        correlation_id = telemetry.new_correlation_id()
        with telemetry.span(
            "embedding_started",
            "embedding_finished",
            correlation_id=correlation_id,
            metadata={
                "model": type(embedding_model).__name__,
                "embedding_dim": settings.embedding_dim,
                "chunks": len(chunks),
                "vector_store": "qdrant",
            },
        ) as span:
            vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
            span["vectors"] = len(vectors)
        store.add(chunks, vectors)
    return store


def load_or_build_index(settings: Settings, *, telemetry: Telemetry | None = None) -> VectorStore:
    """Load the configured index, building it from data when needed.

    With the default ``json`` store the index is loaded from (or built and
    persisted to) ``FEEDBACK_AGENT_INDEX_PATH``. With ``qdrant`` the index lives in
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
    query_expander = ProductTerminologyExpander()
    dense = QueryEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        query_expander=query_expander,
    )
    if settings.retriever_type == "dense":
        return dense
    chunks = vector_store.chunks
    lexical = BM25Retriever(
        chunks,
        texts=[chunk_to_embedding_text(chunk) for chunk in chunks],
        query_expander=query_expander,
    )
    if settings.retriever_type == "lexical":
        return lexical
    return HybridRetriever(
        dense=dense,
        lexical=lexical,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
    )


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the LLM provider selected by ``FEEDBACK_AGENT_LLM_PROVIDER``.

    ``local`` (the default) needs no API key and stays fully deterministic.
    ``openai`` targets any OpenAI-compatible Chat Completions endpoint
    (``OPENAI_BASE_URL``), ``openai_responses`` targets OpenAI's Responses API,
    ``anthropic`` uses the official SDK (optional ``anthropic`` extra),
    ``bedrock`` uses AWS Bedrock Runtime (optional ``bedrock`` extra), and
    ``ollama`` talks to a local Ollama server. Missing credentials raise a clear
    configuration error at construction time.
    """
    if settings.llm_provider == "local":
        return DeterministicLLM()
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when FEEDBACK_AGENT_LLM_PROVIDER=openai")
        return _with_resilience(
            OpenAIChatLLM(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
            ),
            settings,
        )
    if settings.llm_provider == "openai_responses":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when FEEDBACK_AGENT_LLM_PROVIDER=openai_responses"
            )
        return _with_resilience(
            OpenAIResponsesLLM(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
            ),
            settings,
        )
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when FEEDBACK_AGENT_LLM_PROVIDER=anthropic"
            )
        return _with_resilience(
            AnthropicLLM(api_key=settings.anthropic_api_key, model=settings.anthropic_model),
            settings,
        )
    if settings.llm_provider == "bedrock":
        return _with_resilience(
            BedrockConverseLLM(
                model=settings.bedrock_model,
                region_name=settings.bedrock_region,
                max_tokens=settings.bedrock_max_tokens,
                temperature=settings.bedrock_temperature,
            ),
            settings,
        )
    if settings.llm_provider == "ollama":
        return _with_resilience(
            OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model),
            settings,
        )
    raise ValueError(
        f"Unknown LLM provider {settings.llm_provider!r}. "
        "Valid options: local, openai, openai_responses, anthropic, bedrock, ollama."
    )


def _with_resilience(provider: LLMProvider, settings: Settings) -> LLMProvider:
    """Wrap remote LLM providers with the configured resilience policy."""
    if not settings.llm_resilience_enabled:
        return provider
    return ResilientLLMProvider(
        provider,
        policy=ResiliencePolicy(
            max_attempts=settings.llm_retry_max_attempts,
            timeout_seconds=settings.llm_timeout_seconds,
            backoff_seconds=settings.llm_retry_backoff_seconds,
            circuit_failure_threshold=settings.llm_circuit_failure_threshold,
            circuit_recovery_seconds=settings.llm_circuit_recovery_seconds,
        ),
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
    llm = build_llm(settings)
    hallucination_checker = (
        EvidenceOverlapHallucinationChecker(judge=LLMHallucinationJudge(llm))
        if settings.hallucination_judge_enabled
        else EvidenceOverlapHallucinationChecker()
    )
    return FeedbackInsightAgent(
        query_engine=retriever,
        llm=llm,
        telemetry=telemetry,
        tools=build_default_tools(vector_store.chunks),
        hallucination_checker=hallucination_checker,
    )
