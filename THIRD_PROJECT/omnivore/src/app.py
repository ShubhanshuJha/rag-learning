"""
Flask backend for the Omnivore RAG UI, wired to AdvancedRAGSearch.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import time
from flask import Flask, render_template, request, jsonify

import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama  # Alternate to ChatGroq
from dotenv import load_dotenv

load_dotenv()

from rag.retriever import RAGRetriever
from rag.search import AdvancedRAGSearch
from rag.vector_store import VectorStore
from rag.embedding import EmbeddingManager
from rag.data_loader import load_all_documents

# from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
# from embedding_manager import EmbeddingManager

app = Flask(__name__)

# =========================================================================
# Pipeline setup — fill these in with your real objects.
# =========================================================================
#
# embedding_manager = EmbeddingManager(model_name="all-minilm", use_ollama=True)
# retriever = RAGRetriever(embedding_manager=embedding_manager, ...)   # <-- your real args
# llm_model = ChatOllama(model="gemma2:9b", temperature=0.1, num_predict=1024)

embedding_manager = EmbeddingManager(model_name="all-minilm", use_ollama=True)
vector_store = VectorStore(collection_name="omnivore_docs")
retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)

# Only chunk/embed/ingest if the collection is actually empty — otherwise
# every restart would re-embed everything from scratch.
try:
    existing_count = vector_store.collection.count()
except AttributeError:
    existing_count = 0
    print("(*) Could not check existing collection count — assuming empty.")

ingest_embeddings = True
if existing_count > 0:
    print(f"(*) Collection already has {existing_count} documents — skipping ingestion.")
else:
    all_data = load_all_documents(data_dir='../data')
    print("(*) Collection is empty — loading, chunking, and ingesting documents.")
    chunks = embedding_manager.split_documents(documents=all_data)
    texts = list(map(lambda x: x.page_content, chunks))
    embeddings = embedding_manager.generate_embeddings(texts=texts)
    vector_store.ingest_documents(documents=chunks, embeddings=embeddings)

### Initialize the Groq LLM (set GROQ_API_KEY in environment, if using it)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

use_ollama_llm = True
max_token = 1536  #1024
temperature = 0.1

if use_ollama_llm:
    model_name = "gemma2:9b"
    llm_model = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=max_token,  # Ollama's equivalent of Groq's max_tokens
    )
else:
    model_name = "gemma2-9b-it"
    llm_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_token
    )

rag = AdvancedRAGSearch(retriever=retriever, llm_model=llm_model) if retriever and llm_model else None

# Shown in the UI's status badge. Keep this in sync with whatever backend is
# actually wired in above.
PIPELINE_STATUS = {
    "embedding_model": "all-minilm",
    "llm_backend": "Ollama",
    "llm_model": "gemma2:9b",
}


@app.route("/")
def index():
    """Render the main UI."""
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Report the configured backend, and whether the pipeline is actually wired up."""
    return jsonify({
        **PIPELINE_STATUS,
        "status": "online" if rag is not None else "not configured",
    })


@app.route("/api/query", methods=["POST"])
def query():
    """
    Accept a question (plus optional top_k / min_score / summarize overrides),
    run it through AdvancedRAGSearch, and return the answer + sources.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("query") or "").strip()

    if not question:
        return jsonify({"error": "Query cannot be empty."}), 400

    if rag is None:
        return jsonify({
            "error": "Retriever/LLM not configured yet — fill in retriever and llm_model in app.py."
        }), 500

    # Only forward overrides the caller actually sent, so AdvancedRAGSearch's
    # own defaults (top_k=5, min_score=0.2, summarize=False) still apply otherwise.
    query_kwargs = {}
    if "top_k" in data:
        query_kwargs["top_k"] = int(data["top_k"])
    if "min_score" in data:
        query_kwargs["min_score"] = float(data["min_score"])
    if "summarize" in data:
        query_kwargs["summarize"] = bool(data["summarize"])

    start = time.time()
    try:
        result = rag.query(question, **query_kwargs)
    except Exception as ex:
        return jsonify({"error": f"Pipeline error: {ex}"}), 500
    elapsed_ms = round((time.time() - start) * 1000)

    return jsonify({
        "answer": result.get("answer", ""),
        "summary": result.get("summary"),
        "sources": result.get("sources", []),
        "elapsed_ms": elapsed_ms,
    })


@app.route("/api/history")
def history():
    """Return the full query history AdvancedRAGSearch has accumulated this session."""
    return jsonify({"history": rag.history if rag is not None else []})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
