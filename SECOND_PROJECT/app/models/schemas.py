"""
Pydantic request/response models — the single source of truth for every
API contract in this service. Route handlers import from here rather than
building raw dicts, so response shapes can't silently drift out of sync
with the README's documented contract.
"""

from typing import Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    pages_processed: int
    chunks_created: int
    chunks_skipped_duplicate: int
    status: str
    chunks_remaining: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    doc: str
    page: Optional[int] = None
    chunk_id: str


class AskResponse(BaseModel):
    answer: Optional[str] = None
    sources: list[SourceChunk] = []
    model: Optional[str] = None
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    weaviate: str
    ollama: str
