"""
POST /ingest — accepts a PDF upload, parses it, chunks it, deduplicates
against previously ingested content, and stores new chunks in the vector store.
"""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import IngestResponse
from app.services import chunker, pdf_parser, vector_store
from app.utils.hashing import hash_text, slugify
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_title: str | None = Form(default=None),
):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f} MB, exceeds the {settings.max_file_size_mb} MB limit.",
        )

    title = doc_title or os.path.splitext(file.filename)[0]
    doc_id = f"{slugify(title)}-v1"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        pages = pdf_parser.extract_pages(tmp.name)
        if not any(p.text for p in pages):
            raise HTTPException(
                status_code=422,
                detail="No extractable text found — this may be a scanned/image-only PDF.",
            )

        chunks = chunker.chunk_pages(pages)

    with vector_store.get_client() as client:
        vector_store.ensure_collection(client)
        existing_hashes = vector_store.existing_hashes_for_doc(client, doc_id)

        new_objects = []
        skipped = 0

        for chunk in chunks:
            content_hash = hash_text(chunk.text)
            if content_hash in existing_hashes:
                skipped += 1
                continue

            new_objects.append({
                "text": chunk.text,
                "doc_id": doc_id,
                "doc_title": title,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content_hash": content_hash,
            })

        to_insert_now = new_objects[: settings.max_chunks_per_ingest_call]
        remaining = len(new_objects) - len(to_insert_now)

        vector_store.insert_chunks(client, to_insert_now)

    logger.info(
        "Ingested doc_id=%s: %d pages, %d chunks inserted this call, %d skipped as duplicates, %d remaining",
        doc_id, len(pages), len(to_insert_now), skipped, remaining,
    )

    return IngestResponse(
        doc_id=doc_id,
        pages_processed=len(pages),
        chunks_created=len(to_insert_now),
        chunks_skipped_duplicate=skipped,
        chunks_remaining=remaining,
        status="success" if remaining == 0 else "partial",
    )
