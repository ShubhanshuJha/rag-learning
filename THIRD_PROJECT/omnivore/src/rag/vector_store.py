import os
import numpy as np
import chromadb  # open-source Vector DB
import uuid  # to provide unique identifier to records
from typing import List, Any


class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "../data/vector_store",
        distance_metric: str = "cosine",
    ):
        """
        distance_metric: the space the HNSW index is built in — 'cosine',
            'l2' (squared euclidean), or 'ip' (inner product).

            Chroma defaults to 'l2' when this is not set, which does NOT
            match the `1 - distance` similarity RAGRetriever computes:
            squared-L2 distances routinely exceed 1.0, so every score comes
            out negative and the score_threshold filters out every result.
            Cosine is the correct space for normalized text embeddings.

            The metric is baked into the index at creation time, so changing
            it means deleting the collection and re-ingesting —
            get_or_create_collection silently ignores metadata on a
            collection that already exists.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self.client = None
        self.collection = None
        self.__initialize_store()
        print(f"(*) VectorStore initialized.")

    def __initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Document embeddings for RAG.",
                    "hnsw:space": self.distance_metric,
                },
            )
            print(f"(*) VectorDB connected. Collection: {self.collection_name}")
            print(f"(*) Existing documents in collection: {self.collection.count()}")

            # An existing collection keeps whatever space it was built with,
            # so warn rather than let a config change look like it applied.
            actual = (self.collection.metadata or {}).get("hnsw:space", "l2")
            if actual != self.distance_metric:
                print(
                    f"(!) Collection '{self.collection_name}' was built with "
                    f"'{actual}', not the configured '{self.distance_metric}'. "
                    "Delete the collection and re-ingest to change it."
                )
            print(f"(*) Distance metric: {actual}")
        except Exception as ex:
            print(f"(*) Error while initializing Vector Store -- {ex}")

    def ingest_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")
        print(f"(*) Ingestion {len(documents)} documents to the Vector Store.")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generating unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:10]}_{idx}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = idx
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Ingest to Collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"(*) Successfully ingested {len(documents)} to the Vector Store.")
            print(f"(*) Total documents in collection: {self.collection.count()}")
        except Exception as ex:
            print(f"(*) Error while ingesting documents to the Vector Store -- {ex}")
            raise
