"""
Content-hashing and slug helpers.

hash_text() is used for chunk-level deduplication on ingest — re-ingesting
the same document won't silently duplicate the corpus.
slugify() turns a document title into a URL/ID-safe doc_id.
"""

import hashlib


def hash_text(text: str) -> str:
    """Return a stable sha256 hash for a piece of chunk text."""
    normalized = text.strip().lower()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    """Convert a title into an ID-safe slug.

    e.g. 'AWS DMS User Guide' -> 'aws-dms-user-guide'
    """
    cleaned = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    return "-".join(cleaned.lower().split())
