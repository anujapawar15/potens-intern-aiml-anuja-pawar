# RAG Document Q&A

A local, end-to-end Retrieval-Augmented Generation (RAG) application that answers
questions strictly from a set of ingested PDF documents, cites its sources, detects
contradictions between documents, supports multilingual queries, and flags low-confidence
answers for human review.

The sample knowledge base is six HR/policy PDFs for a fictional company ("Nimbus Retail
Technologies"): two versions of a remote-work policy that **deliberately contradict each
other** (max 3 remote days/week in 2023 vs. max 1 remote day/week in 2024), plus leave,
code-of-conduct, IT-security, and expense policies. This gives both `/ask` and
`/contradict` real, verifiable content to work with.

## 1. Project Overview

- **Ingestion**: PDFs are parsed page-by-page, split into overlapping chunks, embedded
  locally, and stored in ChromaDB with metadata (source file, page, chunk ID).
- **Retrieval**: a query is embedded, top candidates are pulled from Chroma (vector
  search), then re-scored by a cross-encoder reranker to improve precision (`top_k`
  is configurable per request).
- **Generation**: an LLM (Groq or Gemini, both free-tier) answers using *only* the
  retrieved chunks, with citations and a confidence score derived from retrieval
  relevance (not from the LLM itself).
- **Contradiction check**: retrieves relevant excerpts from two chosen documents and
  asks the LLM to judge, using only those excerpts, whether they conflict.
- **Multilingual**: questions in any language are detected, translated to English for
  retrieval/generation, and the final answer is translated back to the original language.
- **UI**: a Streamlit app exercises both endpoints without needing Postman/curl.

## 2. Architecture

```
PDFs (data/pdfs/)
   │  pypdf: per-page text extraction
   ▼
chunking.py  ──►  overlapping text chunks + metadata (doc_id, source, page, chunk_id)
   │  sentence-transformers (all-MiniLM-L6-v2)
   ▼
embeddings.py ──► vector_store.py (ChromaDB, persisted to chroma_db/)

Query time (FastAPI, app/main.py):
  question ─► translation.py (detect + translate to English)
            ─► embeddings.py (embed query)
            ─► vector_store.py (top_k * RERANK_MULTIPLIER candidates)
            ─► reranker.py (cross-encoder re-scores → true top_k)
            ─► llm.py (Groq/Gemini answers ONLY from retrieved context)
            ─► translation.py (translate answer back)
            ─► AskResponse (answer, citations, confidence, retrieved_context, timing)

  doc_id_1, doc_id_2, topic? ─► contradiction.py
            ─► fetch relevant chunks per document (topic-filtered or capped sample)
            ─► llm.py (JSON-mode verdict: contradiction / no_contradiction / insufficient_evidence)
            ─► ContradictResponse (verdict, reasoning, evidence per document)

Streamlit UI (ui/streamlit_app.py) ──► calls the FastAPI backend over HTTP
```

**Module layout:**
```
rag_app/
├── app/
│   ├── config.py        # all tunables, loaded from .env
│   ├── chunking.py       # PDF text extraction + overlapping chunk splitter
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── vector_store.py   # ChromaDB persistent client wrapper
│   ├── reranker.py       # cross-encoder reranking
│   ├── llm.py            # Groq/Gemini abstraction
│   ├── translation.py    # language detection + translation
│   ├── rag_pipeline.py   # /ask business logic
│   ├── contradiction.py  # /contradict business logic
│   ├── schemas.py        # Pydantic request/response models
│   └── main.py           # FastAPI app + routes
├── scripts/
│   ├── generate_sample_pdfs.py  # builds the 6 sample PDFs
│   └── ingest.py                # PDF → chunks → embeddings → Chroma
├── ui/
│   └── streamlit_app.py
├── data/pdfs/             # sample PDFs live here (or drop in your own)
├── chroma_db/             # persisted vector store (created on first ingest)
├── requirements.txt
└── .env.example
```

## 3. Setup and Installation

**Requirements**: Python 3.10+.

```bash
cd rag_app
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

cp .env.example .env             # then edit .env and add your API key
```

Get a free API key for whichever provider you set as `LLM_PROVIDER` in `.env`:
- Gemini (default): https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys

The embedding model and reranker run locally (no key needed) and download automatically
on first use (~100-300MB combined).

**Generate the sample PDFs and ingest them:**
```bash
python scripts/generate_sample_pdfs.py
python scripts/ingest.py
```
To use your own PDFs instead, drop at least 5 files into `data/pdfs/` and run
`python scripts/ingest.py` (it re-ingests everything found there, resetting the
collection so re-runs are idempotent).

**Run the API:**
```bash
uvicorn app.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

**Run the UI** (in a second terminal, same venv):
```bash
streamlit run ui/streamlit_app.py
```

## 4. Chunking Strategy

Text is extracted **per PDF page** (via `pypdf`) so every chunk can be traced back to an
exact source page for citations. Each page's text is then split into overlapping,
whitespace-aware windows:

- **Chunk size: 800 characters** (~120-150 words). Large enough to keep a policy clause
  or a few related sentences intact (most single rules in the sample documents are
  100-400 characters), small enough that a retrieved chunk stays focused on one topic
  instead of pulling in unrelated sections — which keeps the LLM's context precise and
  reduces the chance it answers from the wrong part of a chunk.
- **Overlap: 150 characters** (~19%). Prevents a sentence or clause that falls near a
  chunk boundary from being truncated in both neighboring chunks — without overlap, a
  rule split exactly at the boundary could lose the clause that gives it meaning (e.g.
  "...maximum of one (1) day per week" separated from "Remote work is limited to a...").
- The splitter snaps to the nearest whitespace before the cut point so words are never
  broken mid-token.
- Both values are configurable via `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env` — smaller
  chunk sizes trade context-per-chunk for retrieval precision; larger ones do the
  opposite. 800/150 was chosen empirically for short policy-style documents; longer,
  denser documents (e.g. research papers) would likely benefit from a larger chunk size
  (1200-1500 chars) to keep multi-sentence arguments intact.

Each chunk is stored with metadata: `doc_id` (filename stem), `source` (filename), `page`
(1-indexed), `chunk_index` (global sequence number), and a unique `chunk_id` of the form
`{doc_id}_p{page}_c{index_on_page}`.

## 5. Retrieval Pipeline

1. **Embed** the (English) query with a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`, 384-dim, cosine similarity).
2. **Vector search**: fetch a *wide candidate pool* from ChromaDB —
   `top_k * RERANK_MULTIPLIER` chunks (default `5 * 4 = 20`) — because bi-encoder vector
   similarity is fast but only approximate.
3. **Rerank**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each
   `(query, chunk)` pair jointly, which is significantly more accurate than vector
   similarity alone since it lets the model directly compare the full query against the
   full chunk text. Candidates are sorted by this score and trimmed to the true `top_k`.
4. **Confidence score**: the average of the reranker's sigmoid-normalized scores
   (0-1) across the final `top_k` chunks. This is computed independently of the LLM, so
   it reflects retrieval quality even if the model still produces a fluent-sounding
   answer. If it falls below `CONFIDENCE_THRESHOLD` (default `0.45`), the API sets
   `low_confidence_warning: true`. The API always returns the generated answer (so
   programmatic clients decide for themselves), but the Streamlit UI enforces an actual
   **human-in-the-loop gate**: it shows the warning and the supporting citations, and
   withholds the answer text behind an "I've reviewed the citations, show the answer
   anyway" checkbox — a person must explicitly act before a low-confidence answer is
   revealed, rather than it being displayed automatically alongside a passive warning.

`top_k` is a parameter on both `/ask` and `/contradict` (falls back to `DEFAULT_TOP_K`
from `.env` if omitted).

## 6. API Endpoints

### `POST /ask`
```json
// request
{ "question": "How many remote work days are employees allowed?", "top_k": 5 }

// response
{
  "answer": "As of the 2024 policy, employees may work remotely a maximum of one day per week [2]... (2023 policy allowed up to three days per week [1], since superseded).",
  "citations": [
    {"source": "remote_work_policy_2024.pdf", "page": 1, "chunk_id": "remote_work_policy_2024_p1_c0", "snippet": "...", "relevance_score": 0.87}
  ],
  "confidence": 0.81,
  "low_confidence_warning": false,
  "detected_language": "en",
  "retrieved_context": ["..."],
  "response_time_seconds": 1.42
}
```
If the documents don't contain the answer, `answer` is exactly:
`"The provided documents do not contain enough information to answer this question."`

### `POST /contradict`
```json
// request
{ "doc_id_1": "remote_work_policy_2023", "doc_id_2": "remote_work_policy_2024", "topic": "remote work days per week" }

// response
{
  "verdict": "contradiction",
  "reasoning": "Document A [A1] states employees may work remotely up to three days per week, while Document B [B1] limits remote work to one day per week and requires four in-office days. These are directly conflicting limits on the same policy.",
  "evidence_doc_1": [...],
  "evidence_doc_2": [...],
  "response_time_seconds": 2.05
}
```
`verdict` is one of `contradiction`, `no_contradiction`, or `insufficient_evidence` — the
last is returned explicitly whenever the retrieved excerpts don't give the model enough
overlapping information, rather than letting it guess.

### `GET /documents`
Lists ingested `doc_id`s, source filenames, page counts, and chunk counts — used by the
Streamlit UI to populate the document-comparison dropdowns.

### `GET /health`
Liveness check; returns the number of chunks currently in the vector store.

## 7. Multilingual Workflow

1. `langdetect` (local, offline) detects the question's language.
2. If it isn't English, the LLM translates the question to English before retrieval —
   embeddings and the cross-encoder reranker were trained primarily on English text, so
   retrieval quality is meaningfully better on an English query against the English
   source documents.
3. The LLM generates the answer in English, grounded in the (English) retrieved context.
4. The LLM translates the final answer back into the question's original language before
   it's returned. Citation markers (`[1]`, `[2]`, ...) are preserved through translation.

This is a translation-based approach rather than multilingual embeddings/documents,
chosen because it works with any source-document language pair with no extra models —
document ingestion doesn't need to change at all.

## 8. Hallucination Prevention

Two independent layers:

1. **Prompt-level constraint**: the system prompt explicitly forbids using knowledge
   outside the provided context and requires citing every claim with a bracket number
   matching a retrieved chunk. It also mandates one exact fallback sentence
   ("The provided documents do not contain enough information to answer this question.")
   when the context is insufficient — an exact string the API/UI can rely on rather than
   parsing free-form refusals.
2. **Confidence score independent of the LLM**: because a model can still produce a
   fluent, wrong-sounding-right answer even with weak context, the confidence score is
   computed purely from cross-encoder relevance of the retrieved chunks, not from the
   LLM's own self-assessment. Low-relevance retrievals surface a warning regardless of
   how confident the generated text sounds.

The `/contradict` endpoint applies the same idea: it returns `insufficient_evidence`
explicitly rather than forcing a contradiction/no-contradiction call when the two
documents' retrieved excerpts don't actually overlap on the requested topic.

## 9. Design Decisions / Notes

- **Local embeddings + reranker, API-based generation only**: keeps ingestion and
  retrieval free and fast, and means the vector store doesn't need to be rebuilt if you
  switch `LLM_PROVIDER` between Groq and Gemini.
- **ChromaDB** over FAISS: built-in metadata filtering (used to scope `/contradict`
  retrieval to a single `doc_id`) and persistence with minimal setup.
- **Confidence threshold (0.45)**: chosen empirically for `ms-marco-MiniLM-L-6-v2` sigmoid
  scores, where clearly relevant pairs typically score >0.6 and unrelated pairs <0.3.
  Treat it as a starting point — tune `CONFIDENCE_THRESHOLD` in `.env` against your own
  documents and query patterns.
- **Re-ingestion is destructive by design** (`scripts/ingest.py` resets the collection
  first) so repeated runs never accumulate duplicate chunks from the same source files.

## 10. What's Incomplete

- **No automated test suite.** Correctness was verified manually against a running API
  during development (documented request/response examples above), not via a repeatable
  `pytest` suite.
- **The confidence gate is UI-only.** `/ask` always returns the full answer text over the
  API regardless of confidence; only the Streamlit UI withholds it behind a manual review
  checkbox. A programmatic client gets the raw (possibly low-confidence) answer.
- **No retry/failover on LLM provider errors.** A rate-limit (429) or access (403) error
  from the configured provider surfaces directly as a 502 to the caller — there's no
  automatic retry with backoff, and no automatic failover to the other provider.
- **Chunking is fixed-size, not semantic.** Chunks are character-count windows snapped to
  whitespace, not split along sentence/section boundaries, so a chunk can still start or
  end mid-thought if a clause happens to fall near the size limit.
- **No auth on the API.** Any client that can reach the host can call every endpoint.
- **Single global collection.** No multi-tenancy — all ingested documents share one
  ChromaDB collection with no per-user/workspace isolation.
- **PDF-only ingestion.** No support yet for `.docx`, `.html`, or `.txt` sources.
- **Stateless Q&A.** Each `/ask` call is independent; there's no conversation memory
  across turns.
- **No CI or containerization.** No GitHub Actions workflow and no Dockerfile yet.
- **Translation quality is unverified.** Hindi/Marathi (and other non-English) translation
  is a single one-shot LLM pass with no back-translation or human review step to confirm
  accuracy.

## 11. What I'd Improve With More Time

- Add a `pytest` suite (chunking edge cases, reranker scoring, citation formatting, and
  the two endpoints against a mocked LLM) so future changes have a regression safety net.
- Enforce the confidence gate server-side too, not just in the UI (e.g. require an explicit
  "confirm" flag on the request before the full low-confidence answer is returned).
- Add retry-with-backoff and automatic failover between Groq and Gemini on quota/rate-limit
  errors, instead of surfacing them directly to the caller.
- Move from fixed-size chunking to semantic chunking (split on headings/sentences first,
  then pack into size-bounded windows).
- Build a small labeled evaluation set (question → expected source chunk) to measure
  retrieval precision/recall and tune `CONFIDENCE_THRESHOLD` data-drivenly instead of
  empirically.
- Add API-key/token auth, a Dockerfile + docker-compose for one-command setup, and a
  GitHub Actions workflow to run lint/tests on every push.
- Support additional document formats (`.docx`, `.html`, `.txt`) in ingestion.
- Stream the LLM's answer token-by-token in the UI instead of waiting for the full
  completion, for better perceived latency.

---

## AI Tools Used

This project was built with the assistance of:
- **Claude Code (Anthropic, Claude Sonnet 5)** — used for the majority of the
  implementation: architecture and endpoint design, the ingestion/retrieval/generation
  pipeline, the reranker and confidence-scoring logic, the multilingual translation flow,
  the Streamlit UI (including the human-in-the-loop confidence gate), refactoring
  (e.g. extracting shared citation formatting), dependency-conflict debugging, and this
  README.
- **ChatGPT (OpenAI)** — used at points during development.
