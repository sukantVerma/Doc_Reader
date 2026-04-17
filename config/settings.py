"""Application settings loaded from environment variables.

This module is the single place where runtime configuration should live.
It loads values from a local `.env` file when present, then exposes a typed
`settings` object for the rest of the app to use.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


@dataclass(frozen=True)
class Settings:
    """Typed application settings for app startup and knowledge indexing."""

    app_title: str
    app_description: str
    app_version: str
    app_root_path: str
    app_server_url: str
    app_server_description: str
    log_level: str
    upload_dir: str
    knowledge_storage_dir: str
    openai_api_key: str | None
    openai_embedding_model: str
    openai_generation_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int


settings = Settings(
    app_title=os.getenv("APP_TITLE", "Basic Sample App"),
    app_description=os.getenv("APP_DESCRIPTION", "Starting FastAPI and Rag from scratch"),
    app_version=os.getenv("APP_VERSION", "1.0.0"),
    app_root_path=os.getenv("APP_ROOT_PATH", "/icg"),
    app_server_url=os.getenv("APP_SERVER_URL", "http://127.0.0.1:8000"),
    app_server_description=os.getenv("APP_SERVER_DESCRIPTION", "Local development server"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
    knowledge_storage_dir=os.getenv("KNOWLEDGE_STORAGE_DIR", "storage/knowledge_index"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_generation_model=os.getenv("OPENAI_GENERATION_MODEL", "gpt-5.4-mini"),
    embedding_dim=_get_int("OPENAI_EMBEDDING_DIM", 1536),
    chunk_size=_get_int("KNOWLEDGE_CHUNK_SIZE", 800),
    chunk_overlap=_get_int("KNOWLEDGE_CHUNK_OVERLAP", 120),
)
