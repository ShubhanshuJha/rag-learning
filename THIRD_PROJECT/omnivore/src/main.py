"""
Omnivore — config-driven entrypoint.

Everything this module needs comes from omnivore/config.yaml; there are no
tunable constants below. Run it from anywhere:

    python omnivore/src/main.py

Compared with the legacy app.py, which this file leaves untouched:

  * all parameters come from YAML rather than module-level constants
  * paths resolve against the config file, so there is no "cd src/" requirement
  * the ingestion API (/api/ingest*) is registered alongside the query routes,
    sharing one embedding model and one ChromaDB client in a single process
  * startup ingestion is optional (ingestion.ingest_on_startup)

Routes:
    GET   /                     the UI
    GET   /api/status           configured backend + pipeline health
    POST  /api/query            ask a question
    GET   /api/history          this session's queries
    GET   /api/config           the active configuration (secrets redacted)
    GET   /api/ingest/status    what the collection holds
    POST  /api/ingest           ingest a file or directory
    POST  /api/ingest/upload    multipart upload + ingest
"""

import sys
import time
from pathlib import Path

# Allow `python omnivore/src/main.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from config import CONFIG, ConfigError, describe
from rag.data_loader import load_all_documents
from rag.embedding import EmbeddingManager
from rag.retriever import RAGRetriever
from rag.search import AdvancedRAGSearch
from rag.vector_store import VectorStore

import ingest_api


def build_pipeline(cfg):
    """Construct the embedding/vector/retriever/LLM stack from config."""
    embedding_manager = EmbeddingManager(
        model_name=cfg.embedding.model,
        use_ollama=cfg.embedding.use_ollama,
    )
    vector_store = VectorStore(
        collection_name=cfg.vector_store.collection,
        persist_directory=str(cfg.vector_store.persist_directory),
        distance_metric=cfg.vector_store.distance_metric,
    )
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    if cfg.llm.backend == "ollama":
        from langchain_ollama import ChatOllama

        llm_model = ChatOllama(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            num_predict=cfg.llm.max_tokens,   # Ollama's name for max_tokens
        )
    else:
        from langchain_groq import ChatGroq

        llm_model = ChatGroq(
            groq_api_key=cfg.llm.api_key,     # validated present at load time
            model_name=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )

    return embedding_manager, vector_store, retriever, llm_model


def maybe_ingest_on_startup(cfg, embedding_manager, vector_store):
    """
    Ingest the whole data dir, but only when the collection is empty.

    Re-embedding a populated collection would duplicate every chunk, so the
    count check is load-bearing, not an optimization. Prefer POST /api/ingest
    for adding documents to a running server.
    """
    if not cfg.ingestion.ingest_on_startup:
        print("(*) ingest_on_startup is false — skipping. Use POST /api/ingest.")
        return

    try:
        existing = vector_store.collection.count()
    except AttributeError:
        print("(*) Could not read collection count — skipping startup ingestion.")
        return

    if existing > 0:
        print(f"(*) Collection already has {existing} chunks — skipping ingestion.")
        return

    print(f"(*) Collection is empty — ingesting from {cfg.data.dir}")
    documents = load_all_documents(data_dir=str(cfg.data.dir))
    if not documents:
        print("(*) No documents found. Add files and POST /api/ingest.")
        return

    chunks = embedding_manager.split_documents(
        documents=documents,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    embeddings = embedding_manager.generate_embeddings(
        texts=[c.page_content for c in chunks],
        **cfg.embedding.call_kwargs,
    )
    vector_store.ingest_documents(documents=chunks, embeddings=embeddings)


def create_app(cfg=CONFIG) -> Flask:
    """Build the Flask app: query routes plus the ingestion blueprint."""
    print(describe(cfg))

    embedding_manager, vector_store, retriever, llm_model = build_pipeline(cfg)
    maybe_ingest_on_startup(cfg, embedding_manager, vector_store)
    rag = AdvancedRAGSearch(retriever=retriever, llm_model=llm_model)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.ingestion.max_upload_bytes

    # Hand the ingestion blueprint the objects we already built, so it reuses
    # this process's embedding model and ChromaDB client instead of making
    # its own — which is also what keeps SQLite writes in one process.
    ingest_api.configure(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
        data_dir=str(cfg.data.dir),
        config=cfg,
    )
    app.register_blueprint(ingest_api.ingest_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({
            "embedding_model": cfg.embedding.model,
            "llm_backend": "Ollama" if cfg.llm.backend == "ollama" else "Groq",
            "llm_model": cfg.llm.model,
            "status": "online",
        })

    @app.get("/api/config")
    def config_route():
        """The active configuration, for debugging. Never exposes the API key."""
        return jsonify({
            "source": str(cfg.source_path),
            "data_dir": str(cfg.data.dir),
            "vector_store": {
                "collection": cfg.vector_store.collection,
                "persist_directory": str(cfg.vector_store.persist_directory),
                "distance_metric": cfg.vector_store.distance_metric,
            },
            "embedding": {
                "model": cfg.embedding.model,
                "use_ollama": cfg.embedding.use_ollama,
                "batch_size": cfg.embedding.batch_size,
                "max_retries": cfg.embedding.max_retries,
                "retry_delay": cfg.embedding.retry_delay,
                "batch_delay": cfg.embedding.batch_delay,
            },
            "chunking": {
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
            },
            "llm": {
                "backend": cfg.llm.backend,
                "model": cfg.llm.model,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
                "api_key_env": cfg.llm.api_key_env,
                "api_key_present": bool(cfg.llm.api_key),
            },
            "retrieval": {
                "top_k": cfg.retrieval.top_k,
                "min_score": cfg.retrieval.min_score,
                "summarize": cfg.retrieval.summarize,
            },
            "ingestion": {
                "ingest_on_startup": cfg.ingestion.ingest_on_startup,
                "default_mode": cfg.ingestion.default_mode,
                "max_upload_mb": cfg.ingestion.max_upload_mb,
                "upload_subdir": cfg.ingestion.upload_subdir,
            },
        })

    @app.post("/api/query")
    def query():
        data = request.get_json(silent=True) or {}
        question = (data.get("query") or "").strip()
        if not question:
            return jsonify({"error": "Query cannot be empty."}), 400

        # Config supplies the defaults; a request may still override per call.
        try:
            top_k = int(data.get("top_k", cfg.retrieval.top_k))
            min_score = float(data.get("min_score", cfg.retrieval.min_score))
        except (TypeError, ValueError):
            return jsonify({"error": "top_k must be an integer and min_score a number."}), 400
        summarize = bool(data.get("summarize", cfg.retrieval.summarize))

        started = time.time()
        try:
            result = rag.query(
                question, top_k=top_k, min_score=min_score, summarize=summarize
            )
        except Exception as ex:
            return jsonify({"error": f"Pipeline error: {ex}"}), 500

        return jsonify({
            "answer": result.get("answer", ""),
            "summary": result.get("summary"),
            "sources": result.get("sources", []),
            "elapsed_ms": round((time.time() - started) * 1000),
        })

    @app.get("/api/history")
    def history():
        return jsonify({"history": rag.history})

    return app


if __name__ == "__main__":
    try:
        app = create_app()
    except ConfigError as ex:
        print(f"\n(!) Configuration error:\n{ex}\n", file=sys.stderr)
        sys.exit(1)

    app.run(host=CONFIG.app.host, port=CONFIG.app.port, debug=CONFIG.app.debug)
