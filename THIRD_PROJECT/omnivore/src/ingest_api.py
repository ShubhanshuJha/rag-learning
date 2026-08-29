"""
Ingestion API for Omnivore — document-ingestion routes in a Flask Blueprint,
so nothing in app.py has to change.

All defaults come from omnivore/config.yaml; nothing is hardcoded here.

Two ways to use it
------------------
1. Registered in-process. main.py already does this, handing over the objects
   it built so one embedding model and one ChromaDB client are shared:

       ingest_api.configure(
           embedding_manager=embedding_manager,
           vector_store=vector_store,
           config=cfg,
       )
       app.register_blueprint(ingest_api.ingest_bp)

   This is the recommended way — see "Concurrency" at the bottom of this file.

2. Standalone, one port above the configured app port:

       python omnivore/src/ingest_api.py     # http://127.0.0.1:5001

   Standalone mode builds its own EmbeddingManager/VectorStore from CONFIG,
   pointed at the same ChromaDB directory.

Routes
------
    GET   /api/ingest/status    what the collection holds right now
    POST  /api/ingest           ingest a path (file or directory); defaults to
                                the whole data dir, so this doubles as
                                first-time initialization
    POST  /api/ingest/upload    multipart upload, saved into the data dir and
                                ingested in one step

Ingest modes
------------
The underlying VectorStore appends blindly, so re-running ingestion over
existing data would duplicate every chunk. This module deduplicates on the
`source` metadata field that every loader sets:

    skip     (default) leave files already in the collection alone — makes
             POST /api/ingest safe to call repeatedly, and makes it a no-op
             once initialization has happened
    replace  delete the existing chunks for that source, then re-ingest — the
             correct mode for a document that changed on disk
    append   ingest regardless, duplicates and all — escape hatch only
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, Flask, jsonify, request
from werkzeug.utils import secure_filename

from langchain_core.documents.base import Document
# langchain_community, not langchain_classic: data_loader.py uses the latter
# and emits a deprecation warning per loader on every import.
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from config import CONFIG
from rag.embedding import EmbeddingManager
from rag.vector_store import VectorStore

ingest_bp = Blueprint("ingest", __name__)

# Every default below comes from config.yaml. main.py injects the objects it
# already built via configure(); standalone mode builds its own from CONFIG.
#
# Pipeline objects are built lazily so that importing this module is cheap and
# does not load an embedding model as a side effect.
_config = CONFIG
_embedding_manager: Optional[EmbeddingManager] = None
_vector_store: Optional[VectorStore] = None
_data_dir: str = str(CONFIG.data.dir)

# Ingestion mutates a shared collection and is not reentrant; Flask serves
# requests on threads by default, so serialize the whole operation.
_ingest_lock = threading.Lock()


def configure(
    embedding_manager: Optional[EmbeddingManager] = None,
    vector_store: Optional[VectorStore] = None,
    data_dir: Optional[str] = None,
    config=None,
) -> None:
    """
    Inject already-constructed pipeline objects and the active config.

    Call this before the first request when embedding into an existing app, so
    the embedding model is loaded once rather than twice and both processes'
    writes go through a single ChromaDB client.
    """
    global _embedding_manager, _vector_store, _data_dir, _config
    if config is not None:
        _config = config
        _data_dir = str(config.data.dir)
    if embedding_manager is not None:
        _embedding_manager = embedding_manager
    if vector_store is not None:
        _vector_store = vector_store
    if data_dir is not None:
        _data_dir = data_dir


def _get_pipeline():
    """Return (embedding_manager, vector_store), building them on first use."""
    global _embedding_manager, _vector_store
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(
            model_name=_config.embedding.model,
            use_ollama=_config.embedding.use_ollama,
        )
    if _vector_store is None:
        _vector_store = VectorStore(
            collection_name=_config.vector_store.collection,
            persist_directory=str(_config.vector_store.persist_directory),
            distance_metric=_config.vector_store.distance_metric,
        )
    return _embedding_manager, _vector_store


# =========================================================================
# Loading
# =========================================================================
#
# The extension -> loader mapping duplicates data_loader.load_all_documents,
# which only accepts a directory. Dispatching per file is what lets this API
# ingest one new document, and lets it skip files already in the collection
# before paying to parse them.


def _load_json(path: str) -> List[Document]:
    """
    Load a .json file into a single Document.

    data_loader.py uses langchain's JSONLoader, which requires a `jq_schema`
    it is never given — so JSON silently fails there. It also drags in the `jq`
    package, which has no Windows wheel. Reading it with the stdlib avoids both
    problems and keeps the whole file queryable as pretty-printed text.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [
        Document(
            page_content=json.dumps(payload, indent=2, ensure_ascii=False),
            metadata={"source": path},
        )
    ]


_LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p).load(),
    ".txt": lambda p: TextLoader(p, encoding="utf-8").load(),
    ".csv": lambda p: CSVLoader(p).load(),
    ".xlsx": lambda p: UnstructuredExcelLoader(p).load(),
    ".docx": lambda p: Docx2txtLoader(p).load(),
    ".json": _load_json,
}

SUPPORTED_EXTENSIONS = sorted(_LOADERS)


def _iter_files(target: Path) -> List[Path]:
    """Return every supported file at `target` — one file, or a tree of them."""
    if target.is_file():
        return [target] if target.suffix.lower() in _LOADERS else []
    return sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in _LOADERS
    )


def _existing_ids_for_source(collection, source: str) -> List[str]:
    """IDs of chunks already stored for this source path, empty list if none."""
    try:
        found = collection.get(where={"source": source}, include=[])
        return list(found.get("ids") or [])
    except Exception as ex:  # a missing collection, a Chroma version quirk, …
        print(f"(*) Could not check existing chunks for {source} -- {ex}")
        return []


# =========================================================================
# Routes
# =========================================================================


@ingest_bp.get("/api/ingest/status")
def ingest_status():
    """Report collection size and which source files are already ingested."""
    try:
        _, vector_store = _get_pipeline()
        collection = vector_store.collection
        total = collection.count()

        # Chroma has no "distinct" — pull metadatas and reduce. Fine at this
        # scale; swap for a maintained manifest if the corpus grows large.
        sources = set()
        if total:
            records = collection.get(include=["metadatas"])
            for meta in records.get("metadatas") or []:
                if meta and meta.get("source"):
                    sources.add(meta["source"])

        return jsonify({
            "collection": vector_store.collection_name,
            "chunks": total,
            "source_files": sorted(sources),
            "source_file_count": len(sources),
            "data_dir": str(Path(_data_dir).resolve()),
            "supported_extensions": SUPPORTED_EXTENSIONS,
        })
    except Exception as ex:
        return jsonify({"error": f"Status check failed: {ex}"}), 500


@ingest_bp.post("/api/ingest")
def ingest():
    """
    Ingest a file or a directory.

    Body (all optional):
        path           file or directory; defaults to the configured data dir,
                       which makes a bare POST the "initialize everything" call
        mode           skip (default) | replace | append
        chunk_size     default 1500
        chunk_overlap  default 200

    With no body at all this ingests everything under data/ that is not already
    in the collection — safe to call on a fresh store or on a populated one.
    """
    body = request.get_json(silent=True) or {}

    mode = str(body.get("mode", _config.ingestion.default_mode)).lower()
    if mode not in {"skip", "replace", "append"}:
        return jsonify({"error": "mode must be one of: skip, replace, append"}), 400

    try:
        chunk_size = int(body.get("chunk_size", _config.chunking.chunk_size))
        chunk_overlap = int(body.get("chunk_overlap", _config.chunking.chunk_overlap))
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size and chunk_overlap must be integers"}), 400
    if chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap must be smaller than chunk_size"}), 400

    raw_path = body.get("path") or _data_dir
    target = Path(raw_path).resolve()
    if not target.exists():
        return jsonify({"error": f"Path not found: {target}"}), 404

    return _run_ingestion([target], mode, chunk_size, chunk_overlap)


@ingest_bp.post("/api/ingest/upload")
def upload():
    """
    Accept a multipart upload, save it under <data_dir>/uploads/, ingest it.

    Form fields:
        file           the upload (required, repeatable)
        mode           skip | replace | append   (default: replace, since
                       re-uploading a filename means "this is the new version")
        chunk_size / chunk_overlap
    """
    files = request.files.getlist("file")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "No file provided under form field 'file'."}), 400

    mode = str(request.form.get("mode", "replace")).lower()
    if mode not in {"skip", "replace", "append"}:
        return jsonify({"error": "mode must be one of: skip, replace, append"}), 400

    try:
        chunk_size = int(request.form.get("chunk_size", _config.chunking.chunk_size))
        chunk_overlap = int(request.form.get("chunk_overlap", _config.chunking.chunk_overlap))
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size and chunk_overlap must be integers"}), 400

    upload_dir = Path(_data_dir).resolve() / _config.ingestion.upload_subdir
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    rejected: List[Dict[str, str]] = []
    for storage in files:
        if not storage.filename:
            continue
        # secure_filename strips directory components, so an uploaded name can
        # never escape upload_dir.
        name = secure_filename(storage.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in _LOADERS:
            rejected.append({
                "file": storage.filename,
                "reason": "unsupported extension '{}'".format(suffix or "none"),
            })
            continue
        destination = upload_dir / name
        storage.save(str(destination))
        saved.append(destination)

    if not saved:
        return jsonify({
            "error": "No supported files in upload.",
            "rejected": rejected,
            "supported_extensions": SUPPORTED_EXTENSIONS,
        }), 400

    response, status = _run_ingestion(saved, mode, chunk_size, chunk_overlap, as_tuple=True)
    if rejected:
        response["rejected"] = rejected
    return jsonify(response), status


# =========================================================================
# The ingestion itself
# =========================================================================


def _run_ingestion(targets, mode, chunk_size, chunk_overlap, as_tuple=False):
    """Load -> dedupe -> chunk -> embed -> store, reporting per file."""
    started = time.time()

    try:
        embedding_manager, vector_store = _get_pipeline()
    except Exception as ex:
        payload = {"error": f"Pipeline unavailable — is Ollama running? ({ex})"}
        return (payload, 503) if as_tuple else (jsonify(payload), 503)

    collection = vector_store.collection
    if collection is None:
        payload = {"error": "Vector store failed to initialize; check server logs."}
        return (payload, 503) if as_tuple else (jsonify(payload), 503)

    # One ingestion at a time — concurrent runs over the same collection would
    # interleave their dedup checks and defeat them.
    with _ingest_lock:
        count_before = collection.count()

        candidates: List[Path] = []
        for target in targets:
            candidates.extend(_iter_files(target))

        details: List[Dict[str, Any]] = []
        to_embed: List[Document] = []
        ids_to_drop: List[str] = []

        for file_path in candidates:
            source = str(file_path)
            existing = [] if mode == "append" else _existing_ids_for_source(collection, source)

            if existing and mode == "skip":
                details.append({
                    "file": source,
                    "status": "skipped",
                    "reason": "already ingested",
                    "existing_chunks": len(existing),
                })
                continue

            try:
                loaded = _LOADERS[file_path.suffix.lower()](source)
            except Exception as ex:
                details.append({"file": source, "status": "error", "reason": str(ex)})
                continue

            if not loaded:
                details.append({
                    "file": source,
                    "status": "error",
                    "reason": "loader produced no text (scanned PDF with no text layer?)",
                })
                continue

            # Normalize `source` so dedup on a later run matches what we store:
            # loaders set it from the path string they were handed, but not all
            # of them set it at all.
            for doc in loaded:
                doc.metadata["source"] = source

            if existing and mode == "replace":
                ids_to_drop.extend(existing)

            to_embed.extend(loaded)
            details.append({
                "file": source,
                "status": "ingested",
                "documents": len(loaded),
                "replaced_chunks": len(existing) if mode == "replace" else 0,
            })

        if not to_embed:
            payload = {
                "collection": vector_store.collection_name,
                "mode": mode,
                "files_found": len(candidates),
                "files_ingested": 0,
                "chunks_added": 0,
                "chunks_before": count_before,
                "chunks_after": count_before,
                "elapsed_ms": round((time.time() - started) * 1000),
                "message": "Nothing to ingest — everything found was already present or unreadable.",
                "details": details,
            }
            return (payload, 200) if as_tuple else (jsonify(payload), 200)

        try:
            chunks = embedding_manager.split_documents(
                documents=to_embed, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_manager.generate_embeddings(
                texts=texts, **_config.embedding.call_kwargs
            )

            # Delete only after embedding succeeds, so a failure mid-run cannot
            # leave the collection missing the old chunks and without new ones.
            if ids_to_drop:
                collection.delete(ids=ids_to_drop)

            vector_store.ingest_documents(documents=chunks, embeddings=embeddings)
        except Exception as ex:
            payload = {"error": f"Ingestion failed: {ex}", "details": details}
            return (payload, 500) if as_tuple else (jsonify(payload), 500)

        count_after = collection.count()
        payload = {
            "collection": vector_store.collection_name,
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "files_found": len(candidates),
            "files_ingested": sum(1 for d in details if d["status"] == "ingested"),
            "files_skipped": sum(1 for d in details if d["status"] == "skipped"),
            "files_failed": sum(1 for d in details if d["status"] == "error"),
            "documents_loaded": len(to_embed),
            "chunks_added": len(chunks),
            "chunks_replaced": len(ids_to_drop),
            "chunks_before": count_before,
            "chunks_after": count_after,
            "elapsed_ms": round((time.time() - started) * 1000),
            "details": details,
        }
        return (payload, 200) if as_tuple else (jsonify(payload), 200)


# =========================================================================
# Standalone mode
# =========================================================================


def create_app() -> Flask:
    """Build a minimal Flask app exposing only the ingestion routes."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _config.ingestion.max_upload_bytes
    app.register_blueprint(ingest_bp)
    return app


if __name__ == "__main__":
    # One past the configured app port, so this can run next to main.py.
    #
    # Concurrency: ChromaDB's PersistentClient is backed by SQLite, which does
    # not expect two processes writing the same database. Running this
    # standalone while main.py or app.py is also up is fine for read-mostly
    # querying, but a large ingest can lock the file long enough for a query to
    # fail. main.py registers this blueprint in-process, which avoids the issue
    # entirely and is the better way to run it.
    create_app().run(
        host=_config.app.host,
        port=_config.app.port + 1,
        debug=_config.app.debug,
    )
