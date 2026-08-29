import time
import numpy as np
from sentence_transformers import SentenceTransformer  # to use open-source Embedding model for sentences
from typing import List, Optional
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama


class EmbeddingManager:
    """
    Loads a text-embedding model and generates embeddings for input text.

    Two backends are supported:
        - SentenceTransformer: downloads and runs a HuggingFace
          sentence-embedding model locally (default).
        - Ollama: calls a locally running Ollama server to generate
          embeddings, avoiding any network call to HuggingFace.

    Attributes:
        model_name: Name or tag of the embedding model to load.
        model: Loaded SentenceTransformer instance when `ollama_enabled`
            is False; otherwise None (Ollama has no local model object).
        embedding_dim: Dimensionality of the embeddings produced by the
            loaded model. Set once the model has loaded successfully.
        ollama_enabled: Whether embeddings are generated via Ollama
            instead of SentenceTransformer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_ollama: bool = False) -> None:
        """
        Initialize the EmbeddingManager and load the underlying model.

        Args:
            model_name: Embedding model to load. When `use_ollama` is
                False, this should be a HuggingFace SentenceTransformer
                model name (e.g. "all-MiniLM-L6-v2"). When `use_ollama`
                is True, this should be an Ollama model tag that has
                already been pulled locally (e.g. "all-minilm").
            use_ollama: If True, generate embeddings via a local Ollama
                server instead of downloading a SentenceTransformer model.

        Raises:
            Exception: Re-raised if the underlying model fails to load.
                See `__load_model` for details.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None
        self.ollama_enabled = use_ollama
        self.__load_model()
        print(f"(*) EmbeddingManager initialized")

    def __load_model(self) -> None:
        """
        Load the embedding backend (Ollama or SentenceTransformer).

        For Ollama, sends a single test embedding request to confirm the
        model is reachable and to determine its embedding dimension. For
        SentenceTransformer, downloads (if needed) and loads the model
        into memory.

        Raises:
            Exception: Any exception raised while loading the model is
                printed and re-raised so the caller knows initialization
                failed rather than silently ending up with a half-built
                object.
        """
        try:
            print(f"(*) Loading embedding model: {self.model_name} model.")
            if self.ollama_enabled:
                test_response = ollama.embed(model=self.model_name, input="test")
                self.embedding_dim = len(test_response["embeddings"][0])
            else:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_embedding_dimension()
            print(f"(*) Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as ex:
            print(f"(*) Error loading {self.model_name} model -- {ex}")
            raise
    
    def split_documents(self, documents, chunk_size=2000, chunk_overlap=200):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=['\n\n', '\n', ' ', '']
        )

        split_doc_chunks = text_splitter.split_documents(documents)
        print(f"(*) Split {len(documents)} into {len(split_doc_chunks)} chunks.")

        # Show sample chunk
        if split_doc_chunks:
            print("(*) Sample chunk:")
            print(f"\tContent: {split_doc_chunks[0].page_content[:200]}...")
            print(f"\tMetadata: {split_doc_chunks[0].metadata}")
        return split_doc_chunks


    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 256,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        batch_delay: float = 0.3,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts, in batches, with retries.

        Ollama runs each model in a local subprocess and proxies requests to
        it. That subprocess can die mid-run (antivirus interference, memory
        pressure, etc.), which surfaces as a "connection refused" error.
        Ollama typically respawns it on the next request, so a short pause
        + retry usually recovers automatically instead of losing the run.

        Args:
            texts: List of text strings (or Documents) to embed.
            batch_size: Max texts sent to Ollama per request.
            max_retries: Attempts per batch before giving up.
            retry_delay: Base seconds to wait before retrying a failed
                batch (multiplied by attempt number, so it backs off).
            batch_delay: Seconds to pause between successful batches, so
                requests aren't fired back-to-back at the runner.

        Returns:
            Array of shape (len(texts), embedding_dim).

        Raises:
            ValueError: If the model has not been loaded successfully.
            RuntimeError: If a batch still fails after max_retries attempts,
                naming which batch/text range failed.
        """
        if self.embedding_dim is None:
            raise ValueError("Model not loaded")
        if texts and isinstance(texts[0], Document):
            texts = [t.page_content if isinstance(t, Document) else t for t in texts]
            print(f"(*) Found List[Document] as input so converted to List[str] -- considering only the page_content.")

        print(f"(*) Generating embedding for {len(texts)} texts in batches of {batch_size}.")

        if self.ollama_enabled:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1

                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"    - batch {batch_num}/{total_batches} ({len(batch)} texts), attempt {attempt}")
                        response = ollama.embed(model=self.model_name, input=batch)
                        all_embeddings.extend(response["embeddings"])
                        break
                    except Exception as ex:
                        if attempt == max_retries:
                            raise RuntimeError(
                                f"Batch {batch_num}/{total_batches} (texts {i}-{i + len(batch) - 1}) "
                                f"failed after {max_retries} attempts: {ex}"
                            ) from ex
                        wait = retry_delay * attempt
                        print(f"      ! batch {batch_num} failed ({ex}); retrying in {wait:.1f}s...")
                        time.sleep(wait)

                time.sleep(batch_delay)

            embeddings = np.array(all_embeddings)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)

        print(f"(*) Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def __repr__(self) -> str:
        """
        Return a concise, informative representation of this instance.

        Returns:
            A string showing the backend in use, model name, and
            embedding dimension, useful for quick inspection in a
            notebook cell (e.g. just typing `embedding_manager`).
        """
        backend = "Ollama" if self.ollama_enabled else "SentenceTransformer"
        return (
            f"EmbeddingManager(backend={backend}, model_name='{self.model_name}', "
            f"embedding_dim={self.embedding_dim})"
        )
