# RAG Learning Lab

A hands-on, no-framework implementation of Retrieval-Augmented Generation (RAG), built to *understand* the mechanics rather than just use a black-box tool. Everything runs locally via Docker (Weaviate + Ollama), with zero external API keys and zero orchestration frameworks (no LangChain, no LlamaIndex) — every stage of the pipeline is explicit, inspectable Python.

---

## What This Project Does

This repo ingests text documents, embeds them into a vector database, and answers natural-language questions by retrieving the most relevant chunks and feeding them to a local LLM as grounding context — the core RAG loop, end to end:

```
Documents → Chunk → Embed → Store (Weaviate)
                                   ↓
Query → Embed → Similarity Search → Retrieve Top-K Chunks
                                   ↓
              Retrieved Chunks + Query → Prompt Template
                                   ↓
                        Local LLM (Ollama) → Answer
```

Two components, both containerized:
- **Weaviate** — open-source vector database. Stores document embeddings and performs similarity search.
- **Ollama** — runs local embedding and generation models (`nomic-embed-text` for embeddings, `llama3.2`/`mistral` for generation). No API keys, no cloud calls, no data leaves the machine.

---

## Core RAG Concepts (Glossary)

### RAG Fundamentals

**Retrieval-Augmented Generation (RAG)**
An architecture that combines a retrieval system with a language model. Instead of relying only on what the LLM learned during training, RAG first searches an external knowledge base for relevant information and feeds that as context before generating a response. Reduces hallucination by grounding answers in retrieved data.

**Retriever**
The component responsible for finding relevant information from a knowledge base given a query. In this project, Weaviate's vector similarity search is the retriever.

**Generator**
The component that produces the final natural-language answer from a query plus retrieved context. In this project, the Ollama-served LLM (`llama3.2` / `mistral`) is the generator.

**Knowledge Base / Corpus**
The full set of documents available for retrieval. In this project, the hand-written sentences in `rag_pipeline.py` and `docs/rag_intro.txt` form the corpus. A RAG system can never answer correctly about something outside its corpus — see **Retrieval Ceiling** below.

**Query**
The user's natural-language question or request. It gets embedded the same way documents do, so it can be compared against them in vector space.

**Naive RAG vs. Advanced RAG**
Naive RAG is the simple embed → retrieve → generate loop implemented in this project. Advanced RAG adds extra steps on top — query rewriting, reranking, multi-hop retrieval (retrieve, reason, retrieve again), or agentic decisions about *whether* to retrieve at all for a given query.

**Grounding**
Constraining a model's answer to only what's supported by retrieved context, rather than what it "remembers" from training. A well-grounded system will say "I don't have enough information" rather than guess.

**Hallucination**
When a language model generates plausible-sounding but factually incorrect or invented information — the core problem RAG is designed to reduce.

**Retrieval Ceiling**
The single most important intuition from this project: **generation quality is capped by retrieval quality.** No amount of prompt engineering fixes a knowledge base that doesn't contain the answer. A better-worded prompt cannot substitute for a better document.

### Embedding & Chunking

**Embedding**
A numerical vector representation of text that captures semantic meaning. Two pieces of text with similar meaning produce vectors that are close together in vector space, even if they share no words in common.

**Embedding Model**
The specific model used to convert text into vectors (`nomic-embed-text` in this project). Different embedding models produce differently-shaped vectors with different notions of "similarity" — swapping embedding models generally means re-embedding the entire corpus, since vectors from two different models aren't directly comparable.

**Embedding Dimension**
The length of the vector an embedding model outputs (e.g., a few hundred to a few thousand numbers). Higher dimensions can capture more nuance but cost more storage and compute per comparison.

**Distance Metric / Similarity Metric**
The math used to measure how "close" two vectors are. Common choices: cosine similarity (angle between vectors), dot product, and Euclidean distance. Weaviate defaults to cosine similarity for `near_text` / `near_vector` queries.

**Vector Database**
A database optimized for storing embeddings and performing fast similarity search (e.g., "find the 3 stored vectors closest to this query vector"). Weaviate is the vector database used in this project.

**Vector Index (e.g., HNSW)**
The data structure a vector database builds over stored embeddings so it can find nearest neighbors quickly, without comparing the query against every single stored vector one by one — critical once a corpus grows beyond a handful of documents. Weaviate uses HNSW (Hierarchical Navigable Small World) by default.

**Chunking**
Splitting long documents into smaller pieces before embedding. Necessary because embedding models have limited input length, and because smaller, focused chunks produce more precise retrieval matches than one giant embedding for an entire document.

**Chunk Size / Overlap**
Tunable chunking parameters. Chunk size controls how much text goes into each chunk; overlap controls how much text is duplicated between consecutive chunks so that ideas spanning a chunk boundary aren't lost. Naive character-count chunking (used in this project's first pass) can split words or sentences mid-way — production systems chunk on sentence/paragraph boundaries instead.

**Fixed-Size Chunking**
Splitting by a raw character or token count, regardless of sentence or paragraph structure. Simple to implement but risks cutting a sentence or word in half — demonstrated directly in this project's `chunk_and_ingest.py` output, where "information" was split across chunks 0 and 1.

**Semantic / Recursive Chunking**
Splitting that respects natural text boundaries — trying paragraph breaks first, then sentence breaks, and only falling back to raw character counts as a last resort. Produces more coherent chunks than pure fixed-size splitting, at the cost of more implementation complexity.

**Tokenization**
The process of breaking text into the sub-word units ("tokens") that embedding models and LLMs actually process internally. The hard limits models place on how many tokens they can accept per call is the real underlying reason chunking is necessary at all.

**Chunk Metadata**
Additional fields stored alongside chunk text (e.g., `source`, `chunk_index` in this project) that let you trace a retrieved chunk — and therefore a generated answer — back to exactly where it came from. This is the foundation for citation/source attribution.

### Retrieval & Augmentation

**Semantic Search**
Search based on meaning rather than exact keyword/string matching. This is what makes a query like *"How can I stop my chatbot from making things up?"* correctly retrieve a document about *"reducing hallucination"* despite sharing zero words.

**Keyword Search (BM25)**
Traditional search based on exact term matching and term frequency, not meaning. Would fail entirely on this project's chatbot-hallucination query, since it shares no vocabulary with the relevant document — but excels at exact matches (product codes, names, IDs) that semantic search can sometimes miss.

**Hybrid Search**
Combining semantic (vector) search and keyword (BM25) search, typically with a weighted blend, to get the benefits of both — catches exact-term matches that pure semantic search can miss while still handling meaning-based queries. Not used in this project, but supported natively by Weaviate.

**Retrieval (Top-K / `limit`)**
The step where the query embedding is compared against all stored embeddings, and the `limit` (or "k") most similar chunks are returned.

**Similarity Threshold**
An optional cutoff score below which a retrieved result is discarded even if it's technically the "closest" available match — used to avoid returning an irrelevant chunk just because it happened to be nearest when nothing in the corpus is actually a good fit.

**Context Dilution ("Lost in the Middle")**
Retrieving more chunks isn't automatically better. If retrieved chunks are topically unrelated to each other, a small model can get confused trying to reconcile them into one coherent answer, sometimes producing a worse or refused answer than with fewer, cleaner chunks. Coherent multi-chunk context (same topic) tends to help; incoherent multi-chunk context (mixed topics) tends to hurt — both outcomes were reproduced directly in this project.

**Reranking**
An optional step after initial retrieval where a separate model re-scores the top candidates and filters out chunks that are only superficially similar, keeping just the genuinely relevant ones before they reach the generation step. Not implemented in this project yet — see Next Steps.

**Query Rewriting / Query Expansion**
An advanced-RAG technique where the original query is rewritten or expanded — often by an LLM — before retrieval, to produce a version that matches the corpus's phrasing more closely. Useful when user queries are short, ambiguous, or use different vocabulary than the source documents.

**Prompt Template (Augmentation)**
The structure that combines the retrieved chunks and the original query into the actual text sent to the LLM — the literal "A" in RAG, where retrieval output and the user's question are stitched together. In this project, Weaviate's `grouped_task` parameter plays this role.

**Context Window**
The maximum amount of text (measured in tokens) an LLM can accept in a single call, including instructions, retrieved context, and the question itself. Retrieving too many or too-large chunks risks exceeding this limit or crowding out the actual question — related to, but distinct from, context dilution.

### Generation

**Generation**
The LLM producing a final answer from the augmented prompt.

**Large Language Model (LLM)**
The model that produces natural-language text from a prompt. `llama3.2` and `mistral` in this project are both LLMs, served locally through Ollama.

**System Prompt vs. User Prompt**
Two typical roles in an LLM call: the system prompt sets overall behavior/instructions (e.g., "answer using only the retrieved context"), while the user prompt carries the specific question. This project's `grouped_task` functions like a system-level instruction layered on top of the retrieved context.

**Temperature / Sampling**
A generation parameter controlling how deterministic vs. random an LLM's output is. Lower temperature produces more consistent, conservative answers; higher temperature produces more varied, creative ones. Not explicitly set in this project, so each model's default applies.

**Grounded Generation**
Producing an answer that only asserts what's actually supported by the retrieved context, and explicitly saying "I don't know" when that context is insufficient — demonstrated directly in this project's early "insufficient information" result, where the model correctly declined to invent mitigation advice it hadn't been given.

**Context Under-Utilization**
A generation-side failure, distinct from a retrieval failure: the correct information genuinely was retrieved, but the model only surfaces part of it or summarizes too aggressively. Observed directly in this project when a three-part retrieved instruction (ground answers, cite sources, say "I don't know" when needed) was compressed into just the first part.

**Citation / Source Attribution**
Tracing a generated claim back to the specific chunk(s) it came from — enabled by keeping chunk metadata (source file, chunk index) and having the application surface it alongside the answer. Not yet implemented in this project — see Next Steps.

---

## Project Structure

```
.
├── docker-compose.yml       # Weaviate + Ollama containers
├── docs/
│   └── rag_intro.txt        # Sample multi-paragraph source document for chunking
├── rag_pipeline.py          # End-to-end pipeline: create collection, ingest, query, generate
├── chunk_and_ingest.py      # Chunks docs/rag_intro.txt and ingests into a separate collection
├── query_chunks.py          # Queries the chunked collection
└── README.md
```

---

## Setup & Running

> **Note on environment:** this project is run inside a GitHub Codespace rather than a local machine, because corporate network policy (SSL-inspecting proxy) blocked `registry.ollama.ai` and Docker Hub on the original machine. Codespaces provides a full Docker-enabled Linux VM in the browser, unaffected by that restriction. See **Key Learnings** below for the full diagnosis.

**1. Start the containers**
```bash
docker compose up -d
```

**2. Pull the models**
```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull mistral        # optional, for model comparison
```

**3. Install the Python client**
```bash
pip install -U "weaviate-client[agents]"
```

**4. Run the basic pipeline**
```bash
python rag_pipeline.py
```

**5. Run the chunking experiment**
```bash
python chunk_and_ingest.py
python query_chunks.py
```

---

## Key Learnings from Building This

These are the real issues hit while building this project, and what each one taught. Documented in the order they occurred.

### 1. TLS error: `certificate signed by unknown authority`
**What happened:** `ollama pull` failed inside Docker with an x509 certificate error.
**Root cause:** Corporate SSL-inspecting proxy/endpoint agent intercepting HTTPS traffic — the container's minimal trust store didn't recognize the intercepting CA.
**Lesson:** This error signature (`unknown authority`, not `expired` or `connection refused`) is the fingerprint of TLS interception, not a network outage.

### 2. Deeper diagnosis: `fault filter abort`
**What happened:** Browser navigation to the registry showed a raw `fault filter abort` message instead of a certificate warning.
**Root cause:** This string is emitted by Envoy-based proxy fault filters — meaning the connection wasn't just intercepted, it was actively **blocked by policy** (the domain likely isn't allowlisted). Confirmed further by the block persisting even on home Wi-Fi, indicating an always-on endpoint agent (cloud SASE/Zero Trust proxy) rather than a network-level firewall.
**Fix:** Moved the entire environment to GitHub Codespaces — a cloud-hosted, Docker-enabled dev environment reachable from any browser, entirely outside the corporate network path.
**Lesson:** Not all TLS/network errors have a "trust the certificate" fix. Some are policy-level blocks that require moving the work off the restricted network entirely.

### 3. `class name Docs already exists` (422 error)
**What happened:** Re-running the script after a successful first run crashed on `client.collections.create()`.
**Root cause:** The collection persisted in Weaviate from the previous run; `create()` doesn't overwrite an existing collection.
**Fix:** Check `client.collections.exists("Docs")` before creating, and either skip creation or explicitly `delete()` first depending on whether you want a fresh corpus each run.
**Lesson:** Ingestion and querying are different concerns — production RAG systems keep them as separate steps/processes, not one script that does both every time.

### 4. Retrieval ceiling, demonstrated directly
**What happened:** Asking *"How can I stop my chatbot from making things up?"* against a corpus containing only an abstract RAG definition returned "insufficient information." Adding a document with concrete mitigation advice, re-running the *identical* query and prompt, produced a specific, correct answer.
**Lesson:** The only variable that changed was what existed in the vector store — proof that retrieval quality is the ceiling on generation quality, not prompt wording.

### 5. Context dilution with `limit=3`
**What happened:** Increasing `limit` from 1 to 3 pulled in two irrelevant chunks (about llamas and a generic RAG definition) alongside the one relevant chunk, and the model refused to answer instead of using the relevant one.
**Lesson:** More retrieved chunks isn't inherently better — if they're topically incoherent, they compete for the model's attention and can degrade an otherwise-correct answer. This is why reranking exists in production systems.

### 6. `model 'mistral' not found`
**Root cause:** The model was never pulled — `ollama pull mistral` must run once before it can be referenced in code.
**Lesson:** Ollama only serves models it has locally cached; the API doesn't auto-download on first use.

### 7. Naive chunking splits words mid-token
**What happened:** Character-count chunking with `chunk_size=300` visibly split "information" across a chunk boundary.
**Lesson:** Fixed-size character chunking is simple but naive. Production chunkers split on sentence/paragraph boundaries first, falling back to raw characters only as a last resort.

### 8. Overlap preserves meaning across boundaries
**What happened:** With `overlap=50`, a full definition of "chunking" that spanned a chunk boundary was still fully present in two overlapping chunks, and both were retrieved together — the model got the complete idea.
**Lesson:** Overlap is a hedge against sentence-splitting damage: it recovers some of the concept that fixed-size chunking naively breaks.

### 9. Coherent context (from one document) behaved differently than incoherent context (from unrelated documents)
**What happened:** `limit=3` against a single coherent source document produced a fully correct, well-grounded answer — the opposite outcome from the earlier `limit=3` failure against unrelated documents.
**Lesson:** Context dilution isn't about the *count* of retrieved chunks — it's about whether those chunks are topically coherent with each other and the query.

### 10. gRPC `DEADLINE_EXCEEDED` on generation
**Root cause:** CPU-only inference on a Codespace VM (no GPU) took longer than the client's default 30-second gRPC timeout, especially on a cold model load.
**Fix:** Increase the client's query timeout via `AdditionalConfig(timeout=Timeout(query=120))`, and/or warm the model up with a throwaway `curl` request before running the real script.
**Lesson:** Local CPU inference is genuinely slow compared to cloud APIs — timeouts need to account for that, and smaller models (e.g., `llama3.2:1b`) trade quality for iteration speed during development.

---

## Troubleshooting Quick Reference

| Symptom | Root Cause | Fix |
|---|---|---|
| `x509: certificate signed by unknown authority` | TLS interception by corporate proxy/AV | Trust the intercepting CA in the image, or move off that network |
| `fault filter abort` | Policy-level block by an Envoy-based gateway | No cert fix works — use a different network/environment |
| `class name X already exists` (422) | Collection already created in a prior run | `client.collections.exists()` check before `create()` |
| `model 'X' not found` | Model never pulled into Ollama | `docker compose exec ollama ollama pull <model>` |
| `DEADLINE_EXCEEDED` / gRPC timeout | Slow CPU inference, cold model load | Increase client `Timeout`, warm up the model first |
| Vague/refused answer despite retrieval "working" | Retrieved context doesn't actually contain the answer | Improve the corpus, not the prompt |
| Model gets confused with more retrieved chunks | Context dilution from topically mixed chunks | Reduce `limit`, or add reranking |

---

## Next Steps / Further Experiments

- **Reranking** — add a rerank step after retrieval to filter top-k candidates before they reach generation.
- **Chunk size tuning** — compare `chunk_size=300` vs `chunk_size=80` on the same document; observe the retrieval-precision vs. chunk-coherence tradeoff.
- **Model comparison** — run the identical query/context through `llama3.2` and `mistral`, holding retrieval constant, to isolate generation-model effects.
- **Hybrid search** — combine vector similarity with keyword (BM25) search for queries where exact terms matter.
- **Explicit schema typing** — declare `chunk_index` as `DataType.INT` instead of relying on inferred types.
- **Real document ingestion** — replace the sample `.txt` with a real PDF/markdown corpus and a proper sentence-aware chunker.

---

## References

- [Weaviate Local Quickstart](https://docs.weaviate.io/weaviate/quickstart/local)
- [Weaviate Recipes — Local RAG with Ollama notebook](https://github.com/weaviate/recipes/blob/main/weaviate-features/generative-search/generative_search_ollama/ollama_local_rag.ipynb)
- [Ollama Model Library](https://ollama.com/library)
