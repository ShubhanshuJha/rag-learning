"""
Centralized application configuration.

Loads settings from environment variables (populated via .env / docker-compose
env_file). Every other module should import `settings` from here rather than
reading os.environ directly, so configuration stays in exactly one place.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Service endpoints — service names here match docker-compose.yml
    weaviate_url: str = "http://weaviate:8080"
    ollama_api_endpoint: str = "http://ollama:11434"

    # Models
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "llama3.2"

    # RAG tuning
    similarity_threshold: float = 0.75
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_chunks_per_ingest_call: int = 150

    # Upload limits
    max_file_size_mb: int = 50

    # Ops
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
