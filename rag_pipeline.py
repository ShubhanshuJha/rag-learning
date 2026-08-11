import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.generate import GenerativeConfig

### To Remove the Deadline Exceeded Error: often happens due to running LLM Generation on smaller machine
from weaviate.classes.init import AdditionalConfig, Timeout


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Naive fixed-size character chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap  # move forward, but re-cover the overlap
    return [c for c in chunks if c]  # drop any empty trailing chunk

delete_existing_collection = False
chunk_embedding_and_indexing = True

if chunk_embedding_and_indexing:
    with open("docs/rag_intro.txt", "r") as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
    print(f"Split document into {len(chunks)} chunks.\n")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} ({len(c)} chars) ---")
        print(c, "\n")


with weaviate.connect_to_local(additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=120, insert=180)
    )) as client:
    if not client.collections.exists("RagDocs") or delete_existing_collection:
        if client.collections.exists("RagDocs"):
            client.collections.delete("RagDocs")
            print("Deleted collection and creating a new one with the data.")
        
        # Stage 1-2: Define collection + embedding config (vectorization happens on ingest)
        docs = client.collections.create(
            name="RagDocs",
            vector_config=Configure.Vectors.text2vec_ollama(
                api_endpoint="http://ollama:11434",
                model="nomic-embed-text",
            ),
        )
        # Ingest — each object gets embedded automatically on write
        if chunk_embedding_and_indexing:
            docs.data.ingest([
                {"text": chunk, "chunk_index": i, "source": "rag_intro.txt"}
                for i, chunk in enumerate(chunks)
            ])
            print(f"Ingested {len(chunks)} chunks into 'RagDocs' collection.")
        else:
            docs.data.ingest([
                {"text": "Llamas are members of the camelid family, related to camels and vicuñas."},
                {"text": "RAG combines retrieval from a vector store with LLM generation to reduce hallucination."},
            ])

            docs.data.ingest([
                {"text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."},
            ])
            docs.data.ingest([
                {"text": "To reduce hallucination in a chatbot, ground every answer in retrieved documents, cite the source, and instruct the model to say it doesn't know when the retrieved context is insufficient."},
            ])
        print("Created collection and ingested data.")
    else:
        print("Collection already exists, skipping ingestion.")

    collection = client.collections.use("RagDocs")

    # Stage 3: Retrieval — semantic search against the query
    # Stage 4-5: Augmentation + generation, done in one call here
    # query = "What is RAG?"
    query = "How can I stop my chatbot from making things up?"
    if chunk_embedding_and_indexing:
        query = "What is chunking and why does it matter?"

    # limit = 1
    limit = 3

    model = "llama3.2"
    # model = "mistral"

    response = collection.generate.near_text(
        query=query,
        limit=limit,
        grouped_task="Answer the question using only the retrieved context.",
        generative_provider=GenerativeConfig.ollama(
            api_endpoint="http://ollama:11434",
            model=model,
        ),
    )
    print("--- Input query ---")
    print(query)

    print("--- Model & Limit ---")
    print(f"{model}  {limit}")

    if chunk_embedding_and_indexing:
        print("--- Retrieved chunk ---")
        for obj in response.objects:
            print(f"[chunk {obj.properties['chunk_index']}] {obj.properties['text']}\n")
    else:
        print("--- Retrieved context ---")
        for obj in response.objects:
            print(obj.properties["text"])

    print("\n--- Generated answer ---")
    print(response.generative.text)

    # client.collections.delete("RagDocs")
