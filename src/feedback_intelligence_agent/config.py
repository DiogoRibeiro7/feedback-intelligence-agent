"""Application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrieverType = Literal["dense", "lexical", "hybrid"]
VectorStoreType = Literal["json", "qdrant"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FEEDBACK_AGENT_",
        extra="ignore",
    )

    data_path: Path = Field(default=Path("data/sample_feedback.csv"))
    index_path: Path = Field(default=Path(".artifacts/vector_store.json"))
    embedding_dim: int = Field(default=512, ge=64, le=8192)
    vector_store: VectorStoreType = "json"
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="feedback_intelligence")
    retriever_type: RetrieverType = "dense"
    dense_weight: float = Field(default=0.6, ge=0.0)
    lexical_weight: float = Field(default=0.4, ge=0.0)
    llm_provider: Literal["local", "openai", "anthropic", "ollama"] = "local"
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str = Field(
        default="https://api.openai.com", validation_alias="OPENAI_BASE_URL"
    )
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-opus-4-8", validation_alias="ANTHROPIC_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    telemetry_enabled: bool = Field(default=False)
    telemetry_path: Path = Field(default=Path(".artifacts/telemetry.jsonl"))
    conversation_store_path: Path = Field(default=Path(".artifacts/conversations"))
    job_store_path: Path = Field(default=Path(".artifacts/jobs"))
    cors_allow_origins: str = Field(
        default=(
            "http://localhost:5173,http://localhost:4173,"
            "http://127.0.0.1:5173,http://127.0.0.1:4173"
        )
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse the comma-separated CORS origins into a list.

        A single ``*`` allows any origin (convenient for local demos); an empty
        value disables cross-origin requests entirely.
        """
        value = self.cors_allow_origins.strip()
        if not value:
            return []
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    def ensure_artifact_dir(self) -> None:
        """Create the parent folder used by local artifacts."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
