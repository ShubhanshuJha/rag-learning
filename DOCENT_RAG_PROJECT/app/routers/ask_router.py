"""
POST /ask — accepts a natural-language question, retrieves relevant chunks
from the vector store, and returns a grounded answer with source citations
(or an explicit "not found" reason if nothing relevant was ingested).
"""

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import AskRequest, AskResponse
from app.services import generation_service, vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    with vector_store.get_client() as client:
        collection = vector_store.get_collection(client)

        result = generation_service.generate_grounded_answer(
            collection=collection,
            question=request.question,
            top_k=request.top_k,
        )

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        model=settings.generation_model if result.answer else None,
        reason=result.reason,
    )
