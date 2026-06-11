"""Factories for constructing the application components."""

from __future__ import annotations

from pathlib import Path

from ai_engineering_showcase.agent import FeedbackInsightAgent
from ai_engineering_showcase.chunking import feedback_to_chunks
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.ingestion import load_feedback_csv
from ai_engineering_showcase.llm import DeterministicLLM, LLMProvider, OpenAIChatLLM
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.schemas import DocumentChunk
from ai_engineering_showcase.vector_store import InMemoryVectorStore


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
    input_path: str | Path, index_path: str | Path, *, embedding_dim: int
) -> InMemoryVectorStore:
    """Build and persist a vector index from feedback CSV data."""
    records = load_feedback_csv(input_path)
    chunks = feedback_to_chunks(records)
    embedding_model = HashingEmbeddingModel(dim=embedding_dim)
    vectors = embedding_model.embed([chunk_to_embedding_text(chunk) for chunk in chunks])
    vector_store = InMemoryVectorStore(dim=embedding_dim)
    vector_store.add(chunks, vectors)
    vector_store.save(index_path)
    return vector_store


def load_or_build_index(settings: Settings) -> InMemoryVectorStore:
    """Load an index from disk or build it from configured data."""
    settings.ensure_artifact_dir()
    if settings.index_path.exists():
        return InMemoryVectorStore.load(settings.index_path)
    return build_index(
        settings.data_path,
        settings.index_path,
        embedding_dim=settings.embedding_dim,
    )


def build_llm(settings: Settings) -> LLMProvider:
    """Construct the configured LLM provider."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_SHOWCASE_LLM_PROVIDER=openai")
        return OpenAIChatLLM(api_key=settings.openai_api_key, model=settings.openai_model)
    return DeterministicLLM()


def build_agent(settings: Settings) -> FeedbackInsightAgent:
    """Construct a fully wired feedback insight agent."""
    vector_store = load_or_build_index(settings)
    embedding_model = HashingEmbeddingModel(dim=vector_store.dim)
    query_engine = QueryEngine(embedding_model=embedding_model, vector_store=vector_store)
    return FeedbackInsightAgent(query_engine=query_engine, llm=build_llm(settings))
