# DE-RAG-Learning

A hands-on progression through Retrieval-Augmented Generation — starting from a bare, framework-free RAG loop to understand the mechanics, then building a real two-endpoint service on top of what that taught. Everything runs locally via Docker (Weaviate + Ollama), with no cloud API keys anywhere.

Built and run entirely from a GitHub Codespace — the original plan was a local machine, but a corporate network's SSL-inspecting security agent blocked the model registry outright (see `FIRST_PROJECT/README.md` for the full diagnosis). Codespaces turned out to be the more practical environment anyway: browser-only access, no local Docker install needed, works from a phone.

---

## What's in here

| Project | What it is | Status |
|---|---|---|
| [`FIRST_PROJECT/`](./FIRST_PROJECT/README.md) | RAG fundamentals lab — one script, every RAG stage exposed as plain, inspectable code | Working, complete |
| [`SECOND_PROJECT/`](./DOCENT_RAG_PROJECT/README.md) | **Docent** — a real ingest + ask service with a web UI, built from what `FIRST_PROJECT` taught | Working, actively extended |

Start with `FIRST_PROJECT` if the goal is understanding *how* RAG works. Go straight to `DOCENT_RAG_PROJECT` (SECOND_PROJECT) if the goal is *using* a working RAG tool.

---

## The journey

**`FIRST_PROJECT`** is deliberately minimal — no LangChain, no LlamaIndex, nothing hidden behind a framework. It's Weaviate and Ollama in Docker plus a script that calls `embed → store → retrieve → augment → generate` explicitly, so every stage is something you actually watch happen rather than trust. Its README doubles as a RAG glossary (embedding, chunking, context dilution, grounding, retrieval ceiling, and more) and a running log of every real bug hit while building it — TLS interception, timeout mismatches, naive chunking splitting words mid-token, context dilution from mixed-topic retrieval.

**`SECOND_PROJECT` (Docent)** takes those lessons and turns them into something actually usable: feed it a technical PDF, get back a service you can ask real questions against, grounded strictly in what it read, with page citations — via HTTP endpoints or a bundled UI. It also surfaced a deeper problem `FIRST_PROJECT` didn't: Weaviate's built-in Ollama integration has a fixed, non-configurable internal timeout that fails hard on CPU-constrained hardware. The fix — calling Ollama directly from the application instead of through Weaviate's modules — is documented in its README under **Key Design Decisions**, and is the main architectural difference between the two projects.

---

## Stack (both projects)

- **Weaviate** — vector database
- **Ollama** — local embedding + generation models, no cloud API keys
- **Docker Compose** — everything containerized, nothing installed on the host
- **Python** — `FIRST_PROJECT` is plain scripts; `SECOND_PROJECT` is a FastAPI service

---

## Repo structure

```
rag-learning/
├── FIRST_PROJECT/          # RAG fundamentals — scripts + glossary README
│   ├── rag_pipeline.py
│   ├── chunk_and_ingest.py
│   ├── query_chunks.py
│   ├── docker-compose.yml
│   └── README.md
│
└── DOCENT_RAG_PROJECT/         # Docent — ingest + ask service with a UI
    ├── app/
    ├── frontend/
    ├── scripts/
    ├── docs/
    ├── docker-compose.yml
    └── README.md
```

Each project is self-contained with its own `docker-compose.yml` — they don't share containers or a network, so both can run side by side without conflict (just watch for port collisions if you ever run them at the same time; both default to Weaviate on `8080`/`50051` and Ollama on `11434`).

---

## Getting started

```bash
cd FIRST_PROJECT     # or DOCENT_RAG_PROJECT
cp .env.example .env  # SECOND_PROJECT only
docker compose up -d
```

Then follow the "Setup & Running" section in that project's own README — each one documents its own model-pull commands, endpoints, and how to verify it's actually working.
