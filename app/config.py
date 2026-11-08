"""Application configuration settings for docqa-rag."""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration management via environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM & Embedding Settings
    llm_provider: Literal["openai", "mock"] = Field(
        default="openai",
        description="Active LLM provider: 'openai' (compatible with OpenAI, Groq, Ollama) or 'mock'.",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM service. Optional if using mock provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible API endpoint.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name used for question answering synthesis.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name used for document and query vectorization.",
    )

    # Vector Store & Retrieval Settings
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Local persistent directory for ChromaDB vector storage.",
    )
    similarity_threshold: float = Field(
        default=0.53,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity required to accept a retrieved chunk as grounded context.",
    )
    chunk_size: int = Field(
        default=800,
        gt=0,
        description="Target maximum character count per text chunk.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        description="Character overlap between consecutive text chunks.",
    )
    default_top_k: int = Field(
        default=4,
        gt=0,
        le=50,
        description="Default number of top similar chunks to retrieve per query.",
    )

    # Server & UI Settings
    host: str = Field(
        default="0.0.0.0",
        description="Host interface for the FastAPI backend server.",
    )
    port: int = Field(
        default=8000,
        description="Port for the FastAPI backend server.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    api_url: str = Field(
        default="http://localhost:8000",
        description="Backend API URL used by the Streamlit frontend and evaluation script.",
    )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve and cache application settings instance."""
    return Settings()

