# 🧠 Enterprise Knowledge Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react" />
  <img src="https://img.shields.io/badge/PyTorch-2.10+cu128-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Mistral--7B-Instruct-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-4CAF50?style=for-the-badge" />
  <img src="https://img.shields.io/badge/BM25-Lexical%20Search-FF9800?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cross--Encoder-Reranking-9C27B0?style=for-the-badge" />
</p>

A **production-grade, fully local Retrieval-Augmented Generation (RAG) platform** that handles massive-scale enterprise documents. Upload your PDFs and text files and ask natural-language questions answered by a local **Mistral-7B-Instruct** model — with absolute privacy (zero data leaves your machine).

This system has been supercharged with **Enterprise capabilities** to support complex domains, conversational retention, and bleeding-edge hardware inference limits on NVIDIA Blackwell GPUs.

---

## 🎨 System Architecture

<p align="center">
  <img src="docs/architecture.png" alt="RAG Architecture Diagram" width="800"/>
</p>

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
│            NVIDIA GPU  (RTX 5050 / sm_120 Blackwell)        │
│                 PyTorch 2.10.0+cu128                        │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Enterprise Features

### 1. Hybrid Retrieval (Dense + Lexical)
Relies on a dual-search pipeline. We simultaneously query **FAISS** (`BAAI/bge-small-en` dense semantic vectors) and **BM25** (tf-idf lexical exact keyword search). Scores are min-max normalised and dynamically fused. This guarantees that deep semantic similarities _and_ highly-specific part numbers/acronyms are both retrieved correctly.

### 2. Cross-Encoder Reranking
Retrieves a broad candidate net (Top 20), and mathematically reprioritizes them using `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`. The Cross-Encoder evaluates the absolute textual correlation between the query and each chunk, trimming the results down to the ultra-relevant Top 5.

### 3. Native Streaming & Memory
- **Real-Time Streaming**: Token-by-token generation streaming via robust Server-Sent Events (SSE). 
- **Conversational Memory**: Follow-up questions are natively supported. A UUID-based Conversation State Manager holds the last 10 chat turns in memory, injecting context into the Mistral `[INST]` formatting cleanly.

### 4. Advanced Citation Highlighting
No more black-box hallucinations. The system tracks PDF page numbers natively. The UI renders an expandable source-viewer that details the exact originating document, the exact page number, and dynamically renders green/yellow/grey progress bars indicating exactly how relevant the chunk was scored (including its discrete `dense`, `BM25`, and `rerank` sub-scores).

### 5. Deterministic Cache & Offline RAG Eval
- Identical multi-turn queries hit a deterministic hash cache `SHA256(question + collection + top_k)`, resolving in 0.05s instead of 10s of GPU compute.
- An offline, dependency-free endpoint at `/api/v1/evaluate` statically calculates RAG `Faithfulness`, `Context Recall`, and `Answer Relevancy` against internal interaction buffers.

### 6. Hardware Optimisations for NVIDIA Blackwell
Pushed the hardware pipeline to absolute local limits:
- Runs in **4-bit NF4 quantized** precision to crush VRAM overhead.
- Engineered with strict PyTorch **`torch.bfloat16`** tensor compute and globally-enabled **`TensorFloat-32 (TF32)`**, explicitly tuned for NVIDIA Ampere, Ada, and Blackwell cores.
- Forces **`sdpa` (Scaled Dot-Product Attention)**, enabling native Flash Attention 2 implementations deep inside the pipeline.

---

## 🗂️ Complete Codebase Index

Below is a sweeping elaboration of the entire repository, mapping out every file's distinct responsibilities within the architecture.

### 🐍 Backend (`/enterprise_rag_system/`)

#### Application Core
- **`app/main.py`** — The FastAPI absolute entry point. Mounts CORS middleware, instantiates lifecycles (startup/shutdown logging), and aggressively routes the `/api/v1` traffic.
- **`app/config.py`** — Centralized environment mappings. Governs LLM quantization limits (`LLM_USE_4BIT`), fusion weights (`BM25_WEIGHT`, `VECTOR_WEIGHT`), and device mappings (`cuda` vs `cpu`).
- **`api/routes.py`** — Wiring interface detailing endpoints for `/query`, `/query/stream`, `/upload`, `/evaluate`, and collection/session managers. It's the circulatory system bridging HTTP and Python.

#### Hardware LLM & Prompting
- **`llm/local_llm.py`** — The massive local tensor engine. Wraps `transformer` pipelines and spins up the Local LLM models (Mistral-7B). Houses the hardware-level optimizations (TF32, `sdpa`, `bfloat16`, memory-mapping). Generates tokens natively and exposes a `generate_stream()` generator for the SSE pipeline.
- **`llm/prompt_templates.py`** — Formats chunks and memories correctly into Mistral's rigid `<s>[INST] ... [/INST]` format strings. 

#### Retrieval & Storage Pipelines
- **`vector_store/faiss_store.py`** — Memory-mapped flat indices (`IndexFlatIP`). Handles persistent inner-product L2 normalization of vectors locally, with accompanying JSON metadata persistence on disk.
- **`vector_store/collection_manager.py`** — A multi-tenant namespace broker ensuring FAISS stores and hybrid retrievers are correctly scoped to unique `Collection ID` instances (e.g. `hr-docs` vs `engineering-blueprints`).
- **`retrieval/hybrid_retriever.py`** — Calculates `BM25` frequency weights alongside `FAISS` vectors for combined text/semantic matching arrays.
- **`retrieval/reranker.py`** — Wraps a HuggingFace CrossEncoder model to mathematically refine and rank the top-K hybrid results.
- **`retrieval/conversation_memory.py`** — Simple time-to-live mapping queues bounding continuous user query strings against a UUID state.

#### Data Ingestion & Transformation 
- **`ingestion/document_loader.py`** — Employs `pdfplumber` (and heavily optimized `pypdf` fallbacks) to forcefully extract raw byte text natively out of binary PDF constraints, while rigorously tracking the active document `page_number`.
- **`ingestion/preprocessing.py`** — Cleans noise, invalidates bad documents, drops HTML wrappers, normalises extensive line breaks, and manages Unicode stripping.
- **`ingestion/text_chunker.py`** — Intelligently slices vast 400-page strings into digestible, overlapping window arrays (500 tokens logic, 50 token overlap), embedding the `page_number` onto each slice block.
- **`embedding/embedder.py`** — Houses `BAAI/bge-small-en` embeddings loaded off `sentence-transformers` to mathematically cast 500 tokens of string data into 384-dimensional floating point maps.

#### Evaluation & Caching Utilities
- **`evaluation/rag_evaluator.py`** — Internal heuristics calculating token overlap and cosine similarities between Generated Answers and Retrieved Context strings to simulate `Faithfulness` and `Relevance` metrics.
- **`evaluation/dataset_builder.py`** — Maintains a ring buffer logging live end-user queries and answers, feeding them seamlessly into the Evaluator offline.
- **`utils/cache.py`** — Redis caching layer (falling back silently to memory dictionaries) hashing identical payloads to conserve precious GPU prefill cycles.
- **`utils/logger.py`** — System-wide colour-coded structured logging config.

---

### ⚛️ Frontend (`/frontend/`)

- **`src/App.jsx`** — Component root wrapper. Loads styles and mounts `Dashboard.jsx`.
- **`src/api/apiClient.js`** — Deeply integrated Axios instance. Connects directly to backend endpoints. Critically manages internal Fetch mechanisms for resolving raw `ReadableStream` chunks off the backend SSE endpoint parsing delta tokens.
- **`src/pages/Dashboard.jsx`** — The global orchestration frame. Connects the API layer states (`collections`, `conversation_id`, `hasDocuments`) strictly tracking down child views. Renders header stats pill.
- **`src/components/UploadDocuments.jsx`** — Highly robust async Uploader with drag-and-drop mechanics. Streams documents linearly, mapping backend progress and visually tracking file injection logic per Collection ID.
- **`src/components/CollectionSelector.jsx`** — Handles collection abstraction and creation, forcing scoped interactions for the Chat module.
- **`src/components/ChatInterface.jsx`** — Render engine for queries. Traps SSE Token packets iteratively mapping onto conversational bubble components, achieving beautiful simulated "typing" interactions smoothly.
- **`src/components/AnswerViewer.jsx`** — Beautiful expanding Accordion Component explicitly designed to visualize the exact context strings retrieved by the RAG. Dynamically paints color-coded percentage progress bars mapped against sub-scores.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.8 driver.
- 8 GB+ VRAM.

### 1. Python Environment Setup

```bash
git clone https://github.com/PR-ODINSON/Enterprise-Knowledge-Intelligence-System.git
cd Enterprise-Knowledge-Intelligence-System
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install High-Performance PyTorch (CUDA 12.8)

> ⚠️ Do NOT use plain `pip install torch`. RTX 50-series (Blackwell) hardware requires specific `cu128` wheels to bypass kernel execution assertions.

```bash
pip install torch==2.10.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Core Python Dependencies

```bash
pip install -r enterprise_rag_system/requirements.txt
```

### 4. Setup React Frontend

```bash
cd frontend
npm install
```

---

## ▶️ Running the Pipeline

**1. Launch the Backend Server:**
```bash
cd enterprise_rag_system
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger interactive docs are instantly accessible at: `http://localhost:8000/docs`

**2. Launch the Frontend UI:**
```bash
cd frontend
npm run dev
```
The application will launch on `http://localhost:3000`.

---

## 📄 Licensing & Data Privacy

This project strictly provisions enterprise-grade confidentiality. No vector embeddings, text chunks, search queries, or generated LLM packets ever traverse an external API or third-party cloud. All interactions happen 100% locally.

Licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

<p align="center">Built with ❤️ for on-premise enterprise AI</p>
