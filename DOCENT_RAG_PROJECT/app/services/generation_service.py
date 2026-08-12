"""
Generation service.

Retrieval uses Weaviate's near_vector (against a query embedding we
compute ourselves via embedding_service). Generation calls Ollama's
/api/generate directly via httpx, with a timeout we control — this
replaces Weaviate's generative-ollama module, which has the same fixed,
non-configurable internal timeout that caused every ingest failure.
Weaviate is now purely a vector index in this service; it never talks
to Ollama itself.
"""

from dataclasses import dataclass

import httpx
from weaviate.classes.query import MetadataQuery

from app.config import settings
from app.services import embedding_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = "Not covered in the ingested documentation."

GROUNDING_INSTRUCTION = (
    "Answer the question using ONLY the information in the provided context. "
    "Do not use any outside knowledge, even if you are confident about the answer. "
    f"If the context does not contain enough information to answer, respond exactly "
    f"with: '{NOT_FOUND_MESSAGE}'"
)


@dataclass
class GroundedAnswer:
    answer: str | None
    sources: list[dict]
    reason: str | None


def _call_ollama_generate(prompt: str, timeout_seconds: float = 300.0) -> str:
    url = f"{settings.ollama_api_endpoint}/api/generate"
    response = httpx.post(
        url,
        json={"model": settings.generation_model, "prompt": prompt, "stream": False},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_grounded_answer(collection, question: str, top_k: int) -> GroundedAnswer:
    query_vector = embedding_service.embed_texts([question])[0]

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(certainty=True),
    )

    if not response.objects:
        logger.info("No chunks retrieved for question: %r", question)
        return GroundedAnswer(answer=None, sources=[], reason=NOT_FOUND_MESSAGE)

    best_certainty = response.objects[0].metadata.certainty or 0.0
    logger.info(
        "question=%r top_k=%d best_certainty=%.3f threshold=%.3f",
        question, top_k, best_certainty, settings.similarity_threshold,
    )

    if best_certainty < settings.similarity_threshold:
        return GroundedAnswer(answer=None, sources=[], reason=NOT_FOUND_MESSAGE)

    context_text = "\n\n".join(obj.properties["text"] for obj in response.objects)
    prompt = (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    answer_text = _call_ollama_generate(prompt)

    sources = [
        {
            "doc": obj.properties.get("doc_title", "unknown"),
            "page": obj.properties.get("page_number"),
            "chunk_id": f"{obj.properties.get('doc_id', 'unknown')}_chunk_{obj.properties.get('chunk_index')}",
        }
        for obj in response.objects
    ]

    return GroundedAnswer(answer=answer_text, sources=sources, reason=None)
