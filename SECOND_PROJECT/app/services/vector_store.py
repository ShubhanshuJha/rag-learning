"""
Vector store service.

Weaviate is used purely as a vector index here — collection creation
uses self_provided vectors (see embedding_service.get_vectorizer_config),
and every insert supplies its own precomputed vector. Weaviate never
calls Ollama internally, which is what removes the fixed, non-
configurable timeout that caused every prior ingest failure.
"""

from contextlib import contextmanager

import time

import weaviate
from weaviate.classes.config import DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

from app.config import settings
from app.services import embedding_service
from app.services.embedding_service import get_vectorizer_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "DocChunks"
INSERT_BATCH_SIZE = 20
MAX_BATCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def _host_from_url(url: str) -> str:
    return url.replace("http://", "").replace("https://", "").split(":")[0]


@contextmanager
def get_client():
    client = weaviate.connect_to_local(
        host=_host_from_url(settings.weaviate_url),
        port=8080,
        grpc_port=50051,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )
    try:
        yield client
    finally:
        client.close()


def ensure_collection(client) -> None:
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
    collection = client.collections.get(COLLECTION_NAME)
    response = collection.query.fetch_objects(
        filters=Filter.by_property("doc_id").equal(doc_id),
        return_properties=["content_hash"],
        limit=10000,
    )
    return {obj.properties["content_hash"] for obj in response.objects}


def delete_doc(client, doc_id: str) -> None:
    collection = client.collections.get(COLLECTION_NAME)
    collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
    logger.info("Deleted existing chunks for doc_id=%s", doc_id)


def _insert_batch_with_retry(collection, batch, batch_label: str) -> None:
    last_error = None
    for attempt in range(1, MAX_BATCH_RETRIES + 1):
        try:
            result = collection.data.insert_many(batch)
            if result.has_errors:
                logger.warning(
                    "%s: %d of %d objects failed — %s",
                    batch_label, len(result.errors), len(batch), result.errors,
                )
            else:
                logger.info("%s: inserted %d chunks", batch_label, len(batch))
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %ds",
                batch_label, attempt, MAX_BATCH_RETRIES, exc, RETRY_BACKOFF_SECONDS,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    logger.error("%s: gave up after %d attempts", batch_label, MAX_BATCH_RETRIES)
    raise last_error


def insert_chunks(client, objects: list[dict]) -> None:
    if not objects:
        return

    collection = client.collections.get(COLLECTION_NAME)
    total = len(objects)

    for start in range(0, total, INSERT_BATCH_SIZE):
        batch = objects[start:start + INSERT_BATCH_SIZE]
        label = f"batch {start}-{start + len(batch)} of {total}"

        texts = [obj["text"] for obj in batch]
        vectors = embedding_service.embed_texts(texts)

        data_objects = [
            DataObject(properties=props, vector=vector)
            for props, vector in zip(batch, vectors)
        ]
        _insert_batch_with_retry(collection, data_objects, label)


def get_collection(client):
    return client.collections.get(COLLECTION_NAME)
