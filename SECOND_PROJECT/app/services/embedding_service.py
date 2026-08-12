"""
Embedding configuration service.

Weaviate's `text2vec-ollama` module calls Ollama internally to embed text,
both on insert and on query — you never call Ollama's embedding endpoint
directly from this codebase. This module centralizes *which* embedding
model and endpoint that Weaviate module points at, so it's configured in
exactly one place instead of being repeated wherever a collection is created.
"""

from weaviate.classes.config import Configure

from app.config import settings


def get_vectorizer_config():
    """Vector-config object passed to collection creation, telling Weaviate
    to embed stored chunks (and incoming queries) using our local Ollama
    embedding model rather than a cloud provider.
    """
    return Configure.Vectors.text2vec_ollama(
        api_endpoint=settings.ollama_api_endpoint,
        model=settings.embedding_model,
    )
