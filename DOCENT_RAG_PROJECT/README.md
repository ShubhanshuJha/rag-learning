# Docent

*A docent is a knowledgeable guide — someone who has read the whole exhibit and answers your questions by walking straight to the display that has the answer. That's what this is: it reads an entire technical manual once, then answers your questions grounded strictly in what it read, always pointing back to the exact page.*

A service that turns a technical PDF (e.g., the AWS DMS User Guide) into a queryable knowledge base — populate it once via an ingestion endpoint, then ask natural-language questions against it via a second endpoint, or through the bundled web UI. Built as a follow-up to `FIRST_PROJECT`, moving from a single learning script to a properly separated ingestion service + query service, each behind its own HTTP endpoint.

> For core RAG concepts (embedding, chunking, retrieval, augmentation, generation, hallucination, grounding, etc.), see `../FIRST_PROJECT/README.md` — this document assumes those definitions and focuses on this project's architecture and contract.

---

## The Name

**Docent** was chosen deliberately over something literal like "DMS Doc Assistant," for two reasons:

1. **The product is document-agnostic.** The backend has no idea what a PDF is "about" — it parses, chunks, embeds, and retrieves the same way regardless of subject. The AWS DMS User Guide is just the manual currently loaded into it; feed it a different PDF and it becomes a docent for that instead. Naming it after the AWS-specific use case would have been misleading about what the software actually does.
2. **The metaphor matches the mechanism, not just the vibe.** A real docent doesn't improvise from general knowledge when asked something obscure — a good one says "that's not covered in this exhibit" rather than guessing. That's exactly the grounding behavior this project enforces deliberately (see **Key Design Decisions** below): refuse to answer from the model's own training knowledge, answer only from what was actually ingested, and cite the page every time.

---

## Status

**Core pipeline built and working end to end.** `/health`, `/ingest` (with resumable, bounded batches and content-hash dedup), and `/ask` (with similarity-threshold grounding) are implemented, along with a bundled static web UI. Embedding and generation are handled via direct calls to Ollama from the application itself, rather than through Weaviate's built-in Ollama modules — see **Key Design Decisions** for why.

---

## What This Project Does

Two responsibilities, cleanly split into two endpoints instead of one script — plus a bundled UI on top:

1. **Ingest** — accepts a PDF, extracts text page-by-page, chunks it, embeds each chunk (via a direct call to Ollama), and stores it in Weaviate with metadata (source doc, page number, chunk index).
2. **Ask** — accepts a natural-language question, embeds it, retrieves the most relevant stored chunks from Weaviate, and generates an answer (via a direct call to Ollama) grounded strictly in the ingested documentation — refusing to answer from the model's own general knowledge when the docs don't cover something.

```
        ┌───────────────────────┐
        │   Docent (browser)     │
        │   frontend/index.html  │
        └───────────┬───────────┘
                     │
                    HTTP
                     │
                    ┌─────────────────────────────────────┐
                    │            FastAPI App               │
                    │                                       │
  PDF file  ───────►│  POST /ingest                        │
                    │    → Parse PDF → Chunk → Embed(*)     │──┐
                    │    → Dedup → Store in Weaviate         │  │
                    │                                       │  │
  Question  ───────►│  POST /ask                            │  │
                    │    → Embed(*) query → Retrieve         │  │
                    │    → Generate(*) → Return answer+srcs  │  │
                    │                                       │  │
                    │  GET  /health                          │  │
                    └─────────────────────────────────────┘  │
                                                                │
                    ┌──────────────┐          ┌───────────────┐
                    │   Weaviate    │          │    Ollama      │
                    │ (vector index │          │ (embed + LLM,  │
                    │  only — no    │          │  called        │
                    │  Ollama calls │          │  directly by   │
                    │  of its own)  │◄─(*)─────┤  our app code) │
                    └──────────────┘          └───────────────┘
```
**(\*)** Embedding and generation calls go straight from this app's code to Ollama via `httpx`, not through Weaviate's built-in `text2vec-ollama` / `generative-ollama` modules. See **Key Design Decisions**.

---

## Folder Structure

```
SECOND_PROJECT/               # folder name kept as-is; "Docent" is the product name
├── app/
│   ├── main.py                    # FastAPI app entrypoint, route registration, serves frontend/
│   ├── config.py                  # Settings loaded from environment
│   ├── routers/
│   │   ├── ingest_router.py       # POST /ingest
│   │   └── ask_router.py          # POST /ask
│   ├── services/
│   │   ├── pdf_parser.py          # PDF → text + page numbers
│   │   ├── chunker.py             # Recursive/semantic chunking
│   │   ├── embedding_service.py   # Direct Ollama /api/embed calls
│   │   ├── vector_store.py        # Weaviate client (create/query/dedupe/insert)
│   │   └── generation_service.py  # Direct Ollama /api/generate calls + grounding
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── utils/
│       ├── hashing.py             # content-hash dedup helper
│       └── logger.py              # logging setup
├── frontend/
│   └── index.html                 # Docent web UI — single static file, no build step
├── scripts/
│   └── ingest_large_pdf.py        # loops /ingest until a large PDF is fully stored
├── docs/                          # sample PDFs for local testing
├── tests/
│   ├── test_ingest.py
│   └── test_ask.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

**Why this layout:** `routers/` only handles HTTP concerns (request parsing, response shaping); `services/` holds the actual logic (PDF parsing, chunking, embedding, storage, generation) with zero FastAPI-specific code, so each service is independently testable and reusable outside the API if needed. `models/schemas.py` centralizes every request/response contract in one place rather than scattering `dict` shapes across route handlers.

---

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Python packages/folders | `lowercase_snake_case` | `app`, `routers`, `services` |
| Python module files | `lowercase_snake_case.py`, suffix indicates role | `ingest_router.py`, `pdf_parser.py` |
| Pydantic/schema classes | `PascalCase` | `IngestResponse`, `AskRequest` |
| Functions & variables | `snake_case` | `chunk_text()`, `doc_id` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_FILE_SIZE_MB` |
| Environment variables | `UPPER_SNAKE_CASE` | `WEAVIATE_URL`, `EMBEDDING_MODEL` |
| Test files | `test_<module>.py` (required prefix for pytest auto-discovery) | `test_ingest.py` |
| Docker Compose services | lowercase | `api`, `weaviate`, `ollama` |
| Chunk/doc IDs | `<doc-slug>-v<version>_chunk_<index>` | `dms-user-guide-v1_chunk_112` |

---

## API Contract

### `POST /ingest`
```
Content-Type: multipart/form-data

file: <binary PDF>
doc_title: "AWS DMS User Guide"     # optional, defaults to filename
```
```json
// 200 OK — may take several calls for large PDFs; see chunks_remaining
{
  "doc_id": "dms-user-guide-v1",
  "pages_processed": 142,
  "chunks_created": 150,
  "chunks_skipped_duplicate": 0,
  "chunks_remaining": 437,
  "status": "partial"
}
```

### `POST /ask`
```json
// Request
{
  "question": "How do I enable CDC on a DMS replication task?",
  "top_k": 3
}
```
```json
// 200 OK
{
  "answer": "To enable CDC in a DMS replication task...",
  "sources": [
    {"doc": "AWS DMS User Guide", "page": 47, "chunk_id": "dms-user-guide-v1_chunk_112"}
  ],
  "model": "llama3.2"
}
```
```json
// 200 OK — nothing relevant found (similarity below threshold)
{
  "answer": null,
  "sources": [],
  "reason": "Not covered in the ingested documentation."
}
```

### `GET /health`
```json
// 200 OK
{ "weaviate": "ok", "ollama": "ok" }
```
Returns `503` if either dependency is unreachable.

---

## Key Design Decisions

**Chunk metadata schema** (fixed before writing `/ingest`, since changing it later means re-ingesting everything):
```python
{
    "text": "...",
    "doc_id": "dms-user-guide-v1",
    "doc_title": "AWS DMS User Guide",
    "page_number": 47,
    "chunk_index": 112,
    "content_hash": "sha256:..."
}
```

**Deduplication** — every chunk's text is hashed (`content_hash`) before insert. On re-ingesting the same PDF, chunks with a matching hash under the same `doc_id` are skipped rather than duplicated. Re-ingesting an updated version of a doc deletes all chunks for the old `doc_id` first.

**"Not in the docs" handling** — the generation prompt explicitly instructs the model to answer only from retrieved context and say so when it's insufficient, *and* a similarity-threshold check runs before generation is even called: if the best retrieved match scores below the threshold, `/ask` returns `answer: null` without spending a generation call at all. This matters specifically for AWS documentation — `llama3.2` likely has generic AWS knowledge from training and will confidently answer from memory unless explicitly blocked from doing so.

**Direct Ollama calls instead of Weaviate's built-in modules** — the first working version used Weaviate's `text2vec-ollama` and `generative-ollama` modules, which call Ollama *internally* on Weaviate's behalf. On CPU-only, resource-constrained hardware, this repeatedly failed with `context deadline exceeded` — a fixed, non-configurable ~51–60s timeout inside those modules that no client-side setting can raise. The fix: Weaviate is now used purely as a vector index (`Configure.Vectors.self_provided()`), and `embedding_service.py` / `generation_service.py` call Ollama's `/api/embed` and `/api/generate` endpoints directly via `httpx`, with a timeout this application fully controls. Slower CPU inference just takes longer now instead of failing outright.

---

## Environment Variables (`.env.example`)

```
WEAVIATE_URL=http://weaviate:8080
OLLAMA_API_ENDPOINT=http://ollama:11434
EMBEDDING_MODEL=nomic-embed-text
GENERATION_MODEL=llama3.2
SIMILARITY_THRESHOLD=0.75
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_CHUNKS_PER_INGEST_CALL=150
MAX_FILE_SIZE_MB=50
```

---

## Setup & Running

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
docker compose exec api pip install -r requirements.txt   # or built into the image, see Dockerfile
```

Once running, open Docent itself in the browser — via the Codespace **Ports** tab, **Open in Browser** on port `8000`. Interactive API docs are also available at `http://localhost:8000/docs` if you want to test endpoints directly.

For large PDFs, use the resumable ingest script instead of a single `curl` call:
```bash
python3 scripts/ingest_large_pdf.py docs/AWS_DMS_Documentation.pdf "AWS DMS User Guide"
```
Safe to interrupt and re-run — already-ingested chunks are skipped via content-hash dedup.

---

## Testing

```bash
pytest tests/ -v
```
`test_ingest.py` should cover: successful ingest, duplicate-file re-ingest (expect `chunks_skipped_duplicate > 0`), oversized file rejection, non-PDF file rejection.
`test_ask.py` should cover: relevant question returns grounded answer with sources, irrelevant/out-of-scope question returns `answer: null`.

---

## Failure Modes

| Failure | Symptom | Handling |
|---|---|---|
| Ollama not ready yet | `/ask` hangs or times out | `/health` check + compose `depends_on` ordering |
| Scanned/image-only PDF | `chunks_created: 0` despite pages processed | Detect near-empty extracted text per page; return a warning in the ingest response |
| No relevant chunks retrieved | Model answers from general knowledge instead of docs | Similarity threshold short-circuits before generation |
| Duplicate ingestion | Corpus grows, retrieval quality degrades silently | Content-hash dedup on every chunk |

---

## Build Order

1. `/health` + Compose skeleton (`api`, `weaviate`, `ollama`)
2. `/ingest` — parse → chunk → embed → store (no dedup yet)
3. Content-hash dedup on ingest
4. `/ask` — retrieve → augment → generate → return with sources
5. Similarity-threshold "not found in docs" handling
6. Structured logging (question, retrieved chunk IDs + scores, latency) for every `/ask` call

---

## Next Steps (post-MVP)

- Hybrid search (vector + BM25) for queries with exact AWS resource names/error codes
- Reranking step between retrieval and generation
- Multi-document support with a `doc_id` filter on `/ask` (query only within a specific manual)
- Basic API key auth if ever exposed beyond the Codespace
