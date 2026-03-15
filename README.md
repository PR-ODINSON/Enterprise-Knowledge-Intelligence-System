# 🧠 Enterprise Knowledge Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react" />
  <img src="https://img.shields.io/badge/PyTorch-2.10+cu128-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Mistral--7B-Instruct-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-4CAF50?style=for-the-badge" />
</p>

A **production-grade, fully local Retrieval-Augmented Generation (RAG) platform** that lets you upload your own documents and ask natural-language questions answered by a local **Mistral-7B-Instruct** model — with zero data leaving your machine. 

This system has been upgraded with **Enterprise capabilities** to support massive document scaled workloads, intelligent conversational memory, and blazing-fast inference optimisations.

---

## ✨ Enterprise Features

- 🏗️ **Multi-Collection Support** — Group documents into independent collections, each with their own vector index and lexical store.
- 🚀 **Hybrid Retrieval (Dense + Lexical)** — Combines traditional keyword search (`rank_bm25`) with deep semantic search (`BAAI/bge-small-en` + FAISS) for massively improved accuracy.
- 🔭 **Cross-Encoder Reranking** — A second-pass `ms-marco-MiniLM-L-6-v2` cross-encoder accurately reranks top candidates before feeding them to the LLM.
- ⚡ **Real-Time Streaming** — Token-by-token generation streaming via Server-Sent Events (SSE) for a fluid and responsive chat experience.
- 🧠 **Conversational Memory** — Follow-up questions are natively supported with in-memory UUID-based session histories.
- 🏎️ **Deterministic Query Caching** — Identical queries hit a Redis-backed (or fast in-memory fallback) cache to prevent wasting expensive GPU compute.
- 📊 **Offline RAG Evaluation** — Built-in `GET /api/v1/evaluate` endpoint computes Faithfulness, Context Recall, and Answer Relevancy metrics fully offline (no OpenAI keys needed).
- 🏷️ **Advanced Citation Highlighting** — The UI breaks down source chunks showing exact page numbers, relevance score bars, and dense/BM25 sub-scores.

### Hardware & Inference Optimisations

This pipeline pushes Local LLM hardware to the absolute limit:

- **Mistral-7B-Instruct** running in **4-bit NF4 quantisation** via `bitsandbytes` (fits easily in 8 GB VRAM).
- Native support for **NVIDIA Blackwell Architecture** (sm_120 / RTX 5050+) via `torch 2.10.0+cu128`.
- Hardcoded pipeline optimisations: 
  - **TF32 (TensorFloat-32)** enabled globally.
  - Generates natively in **`bfloat16`** precision (supported exceptionally well by Blackwell matrix math cores).
  - Explicit **Scaled Dot-Product Attention (`sdpa`)** implementation for lightning-fast, memory-efficient prefill and decoding phases.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                   │
│                      localhost:3000                         │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / REST / SSE Streaming
┌────────────────────────▼────────────────────────────────────┐
│              FastAPI Backend  (Uvicorn)                     │
│                   localhost:8000                            │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   │
│  │  Ingestion  │   │  Retrieval   │   │   Generation    │   │
│  │             │   │              │   │                 │   │
│  │ pdfplumber  │   │ TF-IDF/BM25  │   │ Mistral-7B-v0.2 │   │
│  │ TextChunker │──▶│ FAISSStore   │──▶│ 4-bit NF4 quant │   │
│  │ Page Mapping│   │ CrossEncoder │   │ TF32 + bfloat16 │   │
│  └─────────────┘   │ Redis Cache  │   │ SDPA Attention  │   │
│                    └──────────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │ CUDA 12.8
┌────────────────────────▼────────────────────────────────────┐
│            NVIDIA GPU  (RTX 5050 / any CUDA GPU)            │
│                 PyTorch 2.10.0+cu128                        │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline — Step by Step

1. **Upload** — File is saved, pages tracked.
2. **Chunk** — Overlapping 500-token chunks (50-token overlap) mapped strictly to document page numbers.
3. **Embed & Index** — Chunks embedded via `bge-small-en` to FAISS. The collection's BM25 index is simultaneously updated.
4. **Retrieve (Hybrid)** — User query is run concurrently against FAISS (dense) and BM25 (lexical). Scores are min-max normalised and explicitly fused.
5. **Rerank** — Top 20 candidates scored pairwise against the original query using `ms-marco-MiniLM-L-6-v2`. Top 5 chunks survive.
6. **Generate (Streamed)** — Mistral-7B streams the answer. Conversation history and Context are injected directly into the `[INST]` template structure.

---

## 🔌 API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload?collection=...` | Upload a PDF or TXT to a collection |
| `POST` | `/query` | Ask a question; returns full answer + chunks |
| `POST` | `/query/stream` | Stream tokens using Server-Sent Events (SSE) |
| `POST` | `/conversations/start` | Start a new conversation UUID |
| `GET` | `/collections` | List all active collections |
| `GET` | `/evaluate` | Get Offline RAG Metrics (faithfulness, recall) |

---

## ⚙️ Configuration

Tune the system via `enterprise_rag_system/app/config.py`.

**Retrieval Limits & Weights:**
```python
HYBRID_ENABLED = True
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5
RERANK_ENABLED = True
RERANK_TOP_K = 20
```

**Generation Constraints:**
```python
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.1
LLM_USE_4BIT = True
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.8 driver (recommended).
- 8 GB+ VRAM.

### 1. Clone & Python Environment

```bash
git clone https://github.com/your-username/Enterprise-Knowledge-Intelligence-System.git
cd Enterprise-Knowledge-Intelligence-System
python -m venv venv

# Windows
.\venv\Scripts\activate
```

### 2. Install PyTorch (GPU — CUDA 12.8 for Blackwell Support)

> ⚠️ Do NOT use plain `pip install torch`. RTX 50-series (Blackwell) requires `cu128` wheels.

```bash
pip install torch==2.10.0+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Requirements

```bash
pip install -r enterprise_rag_system/requirements.txt
```

### 4. Set up the frontend

```bash
cd frontend
npm install
```

---

## ▶️ Running the System

Start the backend:
```bash
cd enterprise_rag_system
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger docs: `http://localhost:8000/docs`

Start the frontend:
```bash
cd frontend
npm run dev
```
UI open at `http://localhost:3000`.

---

## 📄 License & Security

This project is fully local. No embeddings, queries, or generation occurs over external APIs, ensuring maximum intellectual property protection.

Licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

<p align="center">Built with ❤️ for on-premise enterprise AI</p>
