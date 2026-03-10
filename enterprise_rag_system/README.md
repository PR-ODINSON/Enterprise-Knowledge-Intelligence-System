# Enterprise Knowledge Intelligence System

A production-grade **Retrieval Augmented Generation (RAG)** platform that lets you upload corporate documents and ask natural-language questions powered by a locally-hosted LLM — no external API keys required.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [RAG Architecture](#rag-architecture)
3. [System Architecture Diagram](#system-architecture-diagram)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Running the API](#running-the-api)
7. [API Reference](#api-reference)
8. [Example Queries](#example-queries)
9. [Configuration](#configuration)
10. [Hardware Requirements](#hardware-requirements)
11. [Running Tests](#running-tests)

---

## Project Overview

The **Enterprise Knowledge Intelligence System** ingests PDF and TXT documents, indexes them in a local FAISS vector database, retrieves the most semantically relevant context for any question, and generates answers using **Mistral-7B-Instruct** — all without sending data to any external service.

**Core capabilities:**

| Capability | Technology |
|---|---|
| Document parsing | pdfplumber + pypdf |
| Text chunking | Sentence-boundary aware, 500-token chunks with 50-token overlap |
| Embedding generation | sentence-transformers `BAAI/bge-small-en` (384-dim) |
| Vector indexing & search | FAISS `IndexFlatIP` (cosine similarity) |
| Answer generation | HuggingFace Transformers `mistralai/Mistral-7B-Instruct-v0.2` |
| REST API | FastAPI |

---

## RAG Architecture

RAG (Retrieval Augmented Generation) grounds an LLM's answers in your own documents, reducing hallucinations and ensuring responses are traceable to source material.

```
                         ┌─────────────────────────────────┐
  INDEXING PIPELINE      │  PDF / TXT Document              │
  (run once per doc)     └──────────────┬──────────────────┘
                                        │
                               DocumentLoader
                                        │ extracts raw text
                               TextPreprocessor
                                        │ cleans & normalises
                               TextChunker
                                        │ 500-token chunks, 50-token overlap
                               Embedder (BAAI/bge-small-en)
                                        │ 384-dim float32 vectors
                               FAISSVectorStore
                                        │ IndexFlatIP + JSON metadata
                                        ▼
                              ┌──────────────────┐
                              │  Persisted Index  │
                              │  data/faiss_index │
                              └──────────────────┘

                         ┌─────────────────────────────────┐
  QUERY PIPELINE         │  User Question (REST API)        │
  (per request)          └──────────────┬──────────────────┘
                                        │
                               Embedder (same model)
                                        │ query vector
                               FAISSVectorStore.search()
                                        │ top-5 similar chunks
                               PromptTemplate
                                        │ context + question → prompt
                               LocalLLM (Mistral-7B-Instruct)
                                        │ generated answer
                                        ▼
                              ┌──────────────────────────┐
                              │  JSON Response            │
                              │  { answer, chunks, score }│
                              └──────────────────────────┘
```

---

## System Architecture Diagram

```
enterprise_rag_system/
│
├── app/
│   ├── main.py          ← FastAPI app, middleware, lifecycle hooks
│   └── config.py        ← All tuneable parameters in one place
│
├── ingestion/
│   ├── document_loader.py  ← PDF + TXT extraction (pdfplumber / pypdf)
│   ├── text_chunker.py     ← Sentence-boundary aware chunker
│   └── preprocessing.py    ← Unicode normalisation, boilerplate removal
│
├── embedding/
│   └── embedder.py      ← BAAI/bge-small-en wrapper (lazy-loaded)
│
├── vector_store/
│   └── faiss_store.py   ← FAISS IndexFlatIP + JSON metadata sidecar
│
├── retrieval/
│   └── retriever.py     ← Query → embedding → top-k chunks
│
├── llm/
│   ├── local_llm.py        ← Mistral-7B pipeline (lazy-loaded, optional 4-bit)
│   └── prompt_templates.py ← [INST] instruction-format prompt builders
│
├── api/
│   └── routes.py        ← POST /upload, POST /query, GET /documents, …
│
├── utils/
│   └── logger.py        ← Rotating file + console logger factory
│
├── data/
│   ├── documents/       ← Uploaded source files live here
│   ├── faiss_index.bin  ← Persisted FAISS binary index (auto-created)
│   └── metadata.json    ← Parallel chunk metadata (auto-created)
│
├── models/              ← HuggingFace model cache (auto-populated)
├── logs/                ← Rotating log files (auto-created)
├── tests/               ← Pytest test suite
├── requirements.txt
└── README.md
```

---

## Project Structure

```
enterprise_rag_system/
├── app/                    # Application configuration and entry point
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── ingestion/              # Document loading, cleaning, and chunking
│   ├── __init__.py
│   ├── document_loader.py
│   ├── preprocessing.py
│   └── text_chunker.py
├── embedding/              # Sentence-transformer embedding generation
│   ├── __init__.py
│   └── embedder.py
├── vector_store/           # FAISS index management and persistence
│   ├── __init__.py
│   └── faiss_store.py
├── retrieval/              # Semantic search orchestration
│   ├── __init__.py
│   └── retriever.py
├── llm/                    # Local LLM inference and prompt construction
│   ├── __init__.py
│   ├── local_llm.py
│   └── prompt_templates.py
├── api/                    # FastAPI route definitions
│   ├── __init__.py
│   └── routes.py
├── utils/                  # Shared utilities
│   ├── __init__.py
│   └── logger.py
├── data/
│   └── documents/          # Place documents here (or upload via API)
├── models/                 # HuggingFace model weights cache
├── tests/                  # Pytest tests (offline-safe, mocked)
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_embedder.py
│   └── test_retriever.py
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip / virtualenv
- *(Optional)* CUDA-capable GPU for faster LLM inference

### Steps

```bash
# 1 — Clone / navigate to the project
cd "enterprise_rag_system"

# 2 — Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3 — Install dependencies
pip install -r requirements.txt

# 4 — (GPU users) Replace faiss-cpu with faiss-gpu
pip uninstall faiss-cpu -y
pip install faiss-gpu

# 5 — (Optional) Enable 4-bit quantisation for lower VRAM usage
pip install bitsandbytes
```

> **First run:** The embedding model (`BAAI/bge-small-en`, ~130 MB) and LLM
> (`Mistral-7B-Instruct-v0.2`, ~14 GB in float16) are downloaded from
> HuggingFace on first use. Subsequent runs use the local cache.

---

## Running the API

```bash
# From inside enterprise_rag_system/
python -m app.main

# Or with uvicorn directly (with auto-reload for development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger UI is available at **http://127.0.0.1:8000/docs**

---

## API Reference

### `POST /api/v1/upload`

Upload a PDF or TXT document and index it into the vector store.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | PDF or TXT document |

**Response:**
```json
{
  "filename": "annual_report_2025.pdf",
  "chunks_indexed": 142,
  "message": "Successfully indexed 142 chunks in 8.31 seconds."
}
```

---

### `POST /api/v1/query`

Ask a question and receive a RAG-generated answer.

**Request body:**
```json
{
  "question": "What was the company's revenue in Q3 2025?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What was the company's revenue in Q3 2025?",
  "answer": "According to the report, the company's Q3 2025 revenue was $4.2 billion, representing a 12% increase year-over-year.",
  "retrieved_chunks": [
    {
      "text": "Q3 2025 financial highlights: revenue reached $4.2B...",
      "score": 0.9231,
      "filename": "annual_report_2025.pdf",
      "chunk_id": 47
    }
  ],
  "processing_time_seconds": 3.84
}
```

---

### `GET /api/v1/documents`

List all documents currently in the knowledge base.

```json
{
  "documents": [
    { "filename": "policy.pdf", "size_bytes": 204800, "file_type": ".pdf" }
  ],
  "total_documents": 1,
  "total_vectors_indexed": 89
}
```

---

### `DELETE /api/v1/documents/{filename}`

Remove a document from disk. Call `POST /api/v1/reindex` afterwards to update the vector store.

---

### `POST /api/v1/reindex`

Drop and rebuild the FAISS index from all documents in `data/documents/`.

```json
{
  "message": "Re-indexed 3 document(s) in 22.14s.",
  "documents_processed": 3,
  "chunks_indexed": 381
}
```

---

## Example Queries

### Using curl

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/your/document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'

# List indexed documents
curl http://localhost:8000/api/v1/documents
```

### Using Python requests

```python
import requests

BASE = "http://localhost:8000/api/v1"

# Upload
with open("report.pdf", "rb") as f:
    resp = requests.post(f"{BASE}/upload", files={"file": f})
print(resp.json())

# Query
resp = requests.post(
    f"{BASE}/query",
    json={"question": "Summarise the executive overview.", "top_k": 5},
)
print(resp.json()["answer"])
```

---

## Configuration

All parameters are in [app/config.py](app/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en` | Sentence-transformer model |
| `EMBEDDING_DIMENSION` | `384` | Vector size (must match model) |
| `CHUNK_SIZE` | `500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | `5` | Default chunks retrieved per query |
| `LLM_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | Generation model |
| `LLM_MAX_NEW_TOKENS` | `512` | Max tokens generated per answer |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (0 = greedy) |
| `LLM_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

---

## Hardware Requirements

| Configuration | RAM | VRAM | Speed |
|---|---|---|---|
| CPU only | 16 GB+ | N/A | ~2–5 min / query |
| GPU (float16) | 16 GB | 14 GB+ | ~5–15 s / query |
| GPU (4-bit, bitsandbytes) | 12 GB | 5–6 GB | ~10–20 s / query |

> For CPU-only environments, consider swapping `Mistral-7B` for a smaller model
> such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in `config.py`.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_ingestion.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

All tests are **offline-safe** — HuggingFace models and FAISS disk I/O are mocked so no internet connection or GPU is required.

---

## License

MIT — free to use, modify, and distribute.
