"""
Chunking service.

Splits page text into overlapping chunks, packing whole sentences up to
chunk_size rather than cutting at a raw character count — this is the
fix for the mid-word split problem observed in FIRST_PROJECT's naive
fixed-size chunker (e.g. "information" getting split into "informa" / "tion").
"""

import re
from dataclasses import dataclass

from app.config import settings
from app.services.pdf_parser import PageText
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _chunk_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into chunks up to chunk_size characters,
    carrying the tail of each finished chunk forward as overlap so ideas
    spanning a chunk boundary aren't lost.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = (current[-overlap:] + " " + sentence).strip() if overlap else sentence
        else:
            # Single sentence longer than chunk_size on its own — hard-split
            # as a last resort (rare with real prose, common with tables/code blocks).
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(sentence), step):
                chunks.append(sentence[i:i + chunk_size])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    """Chunk every page's text, tracking a global chunk_index and each
    chunk's originating page number for later citation.
    """
    chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        if not page.text:
            continue

        for text in _chunk_page_text(page.text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(Chunk(text=text, page_number=page.page_number, chunk_index=global_index))
            global_index += 1

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks
