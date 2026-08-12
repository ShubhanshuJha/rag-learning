"""
Generation service.

Wraps Weaviate's `generate.near_text` call (which itself calls Ollama for
generation) with two safeguards:

1. A strict grounding instruction, so the model doesn't fall back on its
   own general AWS knowledge when the ingested docs don't cover something.
2. A similarity-threshold check on the best retrieved match — if nothing
   in the corpus is actually relevant, we skip the generation call
   entirely rather than let the model guess.
"""

from dataclasses import dataclass

from weaviate.classes.generate import GenerativeConfig
from weaviate.classes.query import MetadataQuery

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = "Not covered in the ingested documentation."

GROUNDING_INSTRUCTION = (
    "Answer the question using ONLY the information in the provided context. "
    "Do not use any outside knowledge, even if you are confident about the answer. "
    f"If the context does not contain enough information to answer, respond exactly "
    f"with: '{NOT_FOUND_MESSAGE}'"
)


def get_generative_config():
    return GenerativeConfig.ollama(
        api_endpoint=settings.ollama_api_endpoint,
        model=settings.generation_model,
    )


@dataclass
class GroundedAnswer:
    answer: str | None
    sources: list[dict]
    reason: str | None


def generate_grounded_answer(collection, question: str, top_k: int) -> GroundedAnswer:
    """Retrieve the top_k most relevant chunks for `question`, and only call
    the LLM if the best match clears settings.similarity_threshold.
    """
    response = collection.generate.near_text(
        query=question,
        limit=top_k,
        grouped_task=GROUNDING_INSTRUCTION,
        generative_provider=get_generative_config(),
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

    sources = [
        {
            "doc": obj.properties.get("doc_title", "unknown"),
            "page": obj.properties.get("page_number"),
            "chunk_id": f"{obj.properties.get('doc_id', 'unknown')}_chunk_{obj.properties.get('chunk_index')}",
        }
        for obj in response.objects
    ]

    return GroundedAnswer(answer=response.generative.text, sources=sources, reason=None)
