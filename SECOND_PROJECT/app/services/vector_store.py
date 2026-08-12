"""
Vector store service.

Owns the Weaviate collection lifecycle for this project: creating the
"DocChunks" collection with an explicit schema, checking which content
hashes already exist for a doc_id (dedup), deleting a document's prior
chunks before re-ingestion, and inserting new chunks.

NOTE: weaviate.connect_to_local() expects a bare host, not a full URL —
_host_from_url() strips the scheme from WEAVIATE_URL for that reason.
Ports are hardcoded to match docker-compose.yml (8080 HTTP / 50051 gRPC);
if you ever change those in compose, update them here too.
"""

from contextlib import contextmanager

import weaviate
from weaviate.classes.config import DataType, Property
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

from app.config import settings
from app.services.embedding_service import get_vectorizer_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "DocChunks"

# Chunks are inserted in batches of this size rather than all at once.
# Each object triggers an Ollama embedding call server-side, which is slow
# on CPU — a single gRPC call carrying thousands of objects reliably hits
# the default deadline. Smaller batches keep each call well within timeout
# and let you see progress in `docker compose logs -f api` on large PDFs.
INSERT_BATCH_SIZE = 50


def _host_from_url(url: str) -> str:
    return url.replace("http://", "").replace("https://", "").split(":")[0]


@contextmanager
def get_client():
    """Context-managed Weaviate client connection.

    Usage:
        with get_client() as client:
            ...
    """
    client = weaviate.connect_to_local(
        host=_host_from_url(settings.weaviate_url),
        port=8080,
        grpc_port=50051,
        additional_config=AdditionalConfig(
            # insert=120s covers a 50-object batch comfortably even on a
            # cold/CPU-only Ollama; query=60s covers a single /ask retrieval.
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )
    try:
        yield client
    finally:
        client.close()


def ensure_collection(client) -> None:
    """Create the DocChunks collection if it doesn't already exist."""
    if client.collections.exists(COLLECTION_NAME):
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vector_config=get_vectorizer_config(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="doc_title", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="content_hash", data_type=DataType.TEXT),
        ],
    )
    logger.info("Created '%s' collection", COLLECTION_NAME)


def existing_hashes_for_doc(client, doc_id: str) -> set[str]:
    """Return content_hash values already stored for a given doc_id, so
    ingestion can skip chunks that are already present instead of
    duplicating them.
    """
    collection = client.collections.get(COLLECTION_NAME)
    response = collection.query.fetch_objects(
        filters=Filter.by_property("doc_id").equal(doc_id),
        return_properties=["content_hash"],
        limit=10000,
    )
    return {obj.properties["content_hash"] for obj in response.objects}


def delete_doc(client, doc_id: str) -> None:
    """Delete every chunk belonging to doc_id — call this before
    re-ingesting an updated version of a document you've already ingested.
    """
    collection = client.collections.get(COLLECTION_NAME)
    collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
    logger.info("Deleted existing chunks for doc_id=%s", doc_id)


def insert_chunks(client, objects: list[dict]) -> None:
    """Insert new chunk objects in batches of INSERT_BATCH_SIZE.

    Caller is responsible for having already filtered out duplicates via
    existing_hashes_for_doc() — this function does not check again.
    Inserting in smaller batches (rather than one call for the whole
    document) avoids gRPC DEADLINE_EXCEEDED on large PDFs, since each
    object requires a real embedding call to Ollama before it's stored.
    """
    if not objects:
        return

    collection = client.collections.get(COLLECTION_NAME)
    total = len(objects)

    for start in range(0, total, INSERT_BATCH_SIZE):
        batch = objects[start:start + INSERT_BATCH_SIZE]
        result = collection.data.insert_many(batch)

        if result.has_errors:
            logger.warning(
                "Batch %d-%d: %d of %d objects failed to insert — see result.errors",
                start, start + len(batch), len(result.errors), len(batch),
            )
        else:
            logger.info("Inserted batch %d-%d of %d chunks", start, start + len(batch), total)


def get_collection(client):
    return client.collections.get(COLLECTION_NAME)
