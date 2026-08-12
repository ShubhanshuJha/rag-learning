"""
Embedding service.

Calls Ollama's /api/embed endpoint directly via httpx, with a timeout we
control. This replaces relying on Weaviate's internal text2vec-ollama
module, whose own HTTP call to Ollama has a fixed (~51-60s), non-
configurable timeout that was the root cause of every ingest failure so
far on CPU-constrained hardware. Weaviate now only stores and indexes
vectors we hand it — it never calls Ollama itself.
"""

import httpx
from weaviate.classes.config import Configure

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_vectorizer_config():
    """Collection is configured as 'self-provided' — we supply vectors
    ourselves on insert, rather than Weaviate computing them internally.
    """
    return Configure.Vectors.self_provided()


def embed_texts(texts: list[str], timeout_seconds: float = 300.0) -> list[list[float]]:
    """Embed a batch of texts via a direct call to Ollama. Generous
    timeout is safe here because it's entirely under our control, unlike
    Weaviate's internal call.
    """
    url = f"{settings.ollama_api_endpoint}/api/embed"
    response = httpx.post(
        url,
        json={"model": settings.embedding_model, "input": texts},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["embeddings"]
