from rag.vector_store import VectorStore
from rag.embedding import EmbeddingManager


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        print(f"(*) RAGRetriever initialized.")
    
    @staticmethod
    def __to_similarity(distance: float, space: str) -> float:
        """
        Map a ChromaDB distance onto a 0-1-ish similarity.

        Chroma reports a distance whose meaning depends on the collection's
        `hnsw:space`, and each needs its own conversion:

            cosine  distance = 1 - cos_sim, range 0..2   -> 1 - d
            ip      distance = -inner_product            -> -d
            l2      squared euclidean, range 0..inf      -> 1 / (1 + d)

        Applying the cosine formula to an l2 collection is what produced
        negative scores that the score_threshold then filtered away.
        """
        if space == "cosine":
            return 1 - distance
        if space == "ip":
            return -distance
        return 1 / (1 + distance)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        print(f"(*) Retrieving documents for query: '{query}'")
        print(f"(*) Top K: {top_k},  Score Threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            result = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if result['documents'] and result['documents'][0]:
                documents = result['documents'][0]
                metadatas = result['metadatas'][0]
                distances = result['distances'][0]
                ids = result['ids'][0]

                # The conversion below depends on which space the index was
                # built in, so read it rather than assuming cosine.
                space = (self.vector_store.collection.metadata or {}).get("hnsw:space", "l2")

                for idx, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = self.__to_similarity(distance, space)

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': idx + 1
                        })
                
                print(f"(*) Retrieved {len(retrieved_docs)} documents (after filtering).")
            else:
                print("(*) No document found.")
            
            return retrieved_docs
        except Exception as ex:
            print(f"(*) Error while retrieving data for query '{query}' -- {ex}")
            return []
