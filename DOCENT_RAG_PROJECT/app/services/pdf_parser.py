"""
PDF parsing service.

Extracts text from a PDF page-by-page using PyMuPDF, preserving page
numbers so retrieved chunks can later be cited back to a specific page
in AWS_DMS_Documentation.pdf (or any other ingested PDF).
"""

from dataclasses import dataclass

import fitz  # PyMuPDF

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str


def extract_pages(pdf_path: str) -> list[PageText]:
    """Extract text from every page of a PDF.

    Returns one PageText per page, in document order. Pages with no
    extractable text (e.g. scanned/image-only pages) are still included
    with an empty string, so the caller can detect and warn about them
    rather than silently losing page count.
    """
    pages: list[PageText] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        raise ValueError(f"Could not open PDF file: {exc}") from exc

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            pages.append(PageText(page_number=page_index + 1, text=text))
    finally:
        doc.close()

    empty_pages = sum(1 for p in pages if not p.text)
    if empty_pages:
        logger.warning(
            "%d of %d pages had no extractable text (likely scanned/image-only) in %s",
            empty_pages, len(pages), pdf_path,
        )

    logger.info("Extracted text from %d pages of %s", len(pages), pdf_path)
    return pages
