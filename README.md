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

---

## ✨ Features

- 📄 **Document Ingestion** — Upload PDF and TXT files via a drag-and-drop UI
- ✂️ **Intelligent Chunking** — Overlapping token-aware text chunking for context preservation
- 🔍 **Semantic Search** — Dense vector retrieval with BAAI/bge-small-en embeddings + FAISS
- 🤖 **Local LLM** — Mistral-7B-Instruct with 4-bit NF4 quantisation (runs in ~5 GB VRAM)
- ⚡ **GPU Accelerated** — Full CUDA support including NVIDIA Blackwell (RTX 5050, sm_120) via PyTorch 2.10+cu128
- 🔄 **Persistent Index** — FAISS index saved to disk; survives server restarts
- 🗑️ **Document Management** — List, delete, and re-index documents via the API
- 📊 **Source Attribution** — Every answer cites the source chunks it was grounded on
- 🌐 **React Frontend** — Clean, responsive chat-style UI built with Vite + TailwindCSS
- 📖 **Interactive API Docs** — Auto-generated Swagger UI at `/docs`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                   │
│                      localhost:3000                         │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / REST
┌────────────────────────▼────────────────────────────────────┐
│              FastAPI Backend  (Uvicorn)                     │
│                   localhost:8000                            │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  Ingestion  │   │  Retrieval   │   │   Generation    │  │
│  │             │   │              │   │                 │  │
│  │ DocumentLo- │   │ Embedder     │   │ LocalLLM        │  │
│  │ ader        │──▶│ (bge-small)  │   │ (Mistral-7B)    │  │
│  │ Preprocess  │   │              │   │ 4-bit NF4 quant │  │
│  │ TextChunker │   │ FAISSStore   │   │ device_map=auto │  │
│  └─────────────┘   └──────────────┘   └─────────────────┘  │
│                           │                    ▲            │
│                    Vector Search               │            │
│                           └────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                         │ CUDA
┌────────────────────────▼────────────────────────────────────┐
│            NVIDIA GPU  (RTX 5050 / any CUDA GPU)            │
│         PyTorch 2.10.0+cu128  •  CUDA 12.8                  │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline — Step by Step

1. **Upload** — File is saved, text extracted (pdfplumber → pypdf fallback for PDFs)
2. **Preprocess** — Whitespace normalisation, deduplication, content validation
3. **Chunk** — Overlapping 500-token chunks (50-token overlap) with metadata tagging
4. **Embed** — Each chunk embedded with `BAAI/bge-small-en` (384-dim, L2-normalised)
5. **Index** — Embeddings added to FAISS `IndexFlatIP` (inner-product ≡ cosine similarity)
6. **Query** — Question embedded → top-k chunks retrieved → context injected into Mistral-7B prompt
7. **Generate** — Mistral-7B-Instruct produces a grounded answer; source chunks returned alongside

---

## 🗂️ Project Structure

```
Enterprise-Knowledge-Intelligence-System/
├── enterprise_rag_system/          # Python backend
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, lifecycle hooks
│   │   └── config.py               # All tuneable parameters
│   ├── api/
│   │   └── routes.py               # REST endpoints (upload, query, list, delete, reindex)
│   ├── ingestion/
│   │   ├── document_loader.py      # PDF (pdfplumber + pypdf) & TXT loader
│   │   ├── preprocessing.py        # Text cleaning & validation
│   │   └── text_chunker.py         # Overlapping token-aware chunker
│   ├── embedding/
│   │   └── embedder.py             # SentenceTransformer wrapper (BAAI/bge-small-en)
│   ├── vector_store/
│   │   └── faiss_store.py          # FAISS IndexFlatIP with disk persistence
│   ├── retrieval/
│   │   └── retriever.py            # Top-k semantic retrieval
│   ├── llm/
│   │   ├── local_llm.py            # Mistral-7B loader with 4-bit quantisation
│   │   └── prompt_templates.py     # RAG prompt builder (Mistral instruction format)
│   ├── utils/
│   │   └── logger.py               # Structured logging
│   ├── tests/                      # pytest unit tests (43 tests)
│   ├── data/
│   │   ├── documents/              # Uploaded documents stored here
│   │   ├── faiss_index.bin         # Persisted FAISS index
│   │   └── metadata.json           # Chunk metadata
│   └── requirements.txt
├── frontend/                       # React + Vite frontend
│   ├── src/
│   │   ├── components/             # Reusable UI components
│   │   ├── pages/                  # Page-level views
│   │   ├── api/                    # Axios API client
│   │   └── App.jsx
│   ├── index.html
│   └── package.json
└── venv/                           # Python virtual environment
```

---

## 🔌 API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF or TXT document, chunk and index it |
| `POST` | `/query` | Ask a question; returns answer + source chunks |
| `GET` | `/documents` | List all indexed documents and total vector count |
| `DELETE` | `/documents/{filename}` | Delete a document from disk |
| `POST` | `/reindex` | Drop and rebuild index from all stored documents |
| `GET` | `/health` | Health probe for load balancers / k8s |
| `GET` | `/docs` | Interactive Swagger UI |

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@your_document.pdf"
```
```json
{
  "filename": "your_document.pdf",
  "chunks_indexed": 47,
  "message": "Successfully indexed 47 chunks in 3.21 seconds."
}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "top_k": 5}'
```
```json
{
  "question": "What is the refund policy?",
  "answer": "According to the document, refunds are processed within 30 days...",
  "retrieved_chunks": [
    { "text": "...", "score": 0.91, "filename": "policy.pdf", "chunk_id": 12 }
  ],
  "processing_time_seconds": 4.37
}
```

---

## ⚙️ Configuration

All settings live in `enterprise_rag_system/app/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en` | Sentence-transformers embedding model |
| `EMBEDDING_DIMENSION` | `384` | Output vector dimension |
| `EMBEDDING_DEVICE` | `cuda` | Device for embedder (`cuda` / `cpu`) |
| `CHUNK_SIZE` | `500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlapping tokens between chunks |
| `TOP_K_RESULTS` | `5` | Default retrieval count per query |
| `LLM_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model identifier |
| `LLM_MAX_NEW_TOKENS` | `512` | Max tokens generated per response |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (lower = more factual) |
| `LLM_DEVICE` | `cuda` | Device for LLM (`cuda` / `cpu`) |
| `LLM_USE_4BIT` | `True` | Enable bitsandbytes 4-bit NF4 quantisation |
| `API_HOST` | `0.0.0.0` | Uvicorn bind address |
| `API_PORT` | `8000` | Uvicorn port |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.8 driver (recommended) — see GPU notes below
- 8 GB+ VRAM for Mistral-7B with 4-bit quantisation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Enterprise-Knowledge-Intelligence-System.git
cd Enterprise-Knowledge-Intelligence-System
```

### 2. Set up the Python environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch (GPU — CUDA 12.8)

> ⚠️ **Do not use plain `pip install torch`** — it installs the CPU-only wheel.

```bash
pip install torch==2.10.0+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

For **CPU-only** machines:
```bash
pip install torch torchvision torchaudio
```

### 4. Install remaining dependencies

```bash
pip install -r enterprise_rag_system/requirements.txt
```

### 5. Set up the frontend

```bash
cd frontend
npm install
```

---

## ▶️ Running the System

### Start the backend

```bash
cd enterprise_rag_system
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Swagger docs: `http://localhost:8000/docs`

### Start the frontend

```bash
cd frontend
npm run dev
```

The UI will open at `http://localhost:3000`.

---

## 🖥️ GPU Support

### Tested hardware
| GPU | Architecture | Compute | Status |
|-----|-------------|---------|--------|
| NVIDIA GeForce RTX 5050 Laptop | Blackwell | sm_120 | ✅ Fully supported |
| Any RTX 30xx / 40xx | Ampere / Ada | sm_80–sm_90 | ✅ Fully supported |
| Any GTX 16xx / RTX 20xx | Turing | sm_75 | ✅ Fully supported |

### GPU notes

- **Blackwell GPUs (RTX 5050, 5060, 5070, 5080, 5090)** require `torch 2.10.0+cu128` or newer. Older builds (`cu124` and earlier) do not ship compiled sm_120 kernels and will crash with `no kernel image is available for execution on the device`.
- **4-bit quantisation** via `bitsandbytes` reduces Mistral-7B's VRAM footprint from ~14 GB (float16) to ~5 GB — making it fit in 8 GB VRAM cards.
- **faiss-gpu** is not published on PyPI for Windows. `faiss-cpu` is used instead (FAISS vector search is not the throughput bottleneck — the LLM is).

### CPU-only fallback

The system works fully on CPU. Set in `config.py`:
```python
LLM_DEVICE = "cpu"
LLM_USE_4BIT = False    # bitsandbytes requires CUDA
EMBEDDING_DEVICE = "cpu"
```
> CPU inference for Mistral-7B is slow (~1–3 min per response). Consider switching to a smaller model like `microsoft/phi-2` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for CPU-only setups.

---

## 🧪 Running Tests

```bash
cd enterprise_rag_system
pytest tests/ -v
```

43 unit tests covering:
- Embedding pipeline (shape, dtype, lazy loading, batch processing)
- Document ingestion (loaders, text extraction, chunking)
- FAISS vector store (add, search, persist, reset)

All tests use mocks — no model downloads or GPU required to run them.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend framework** | FastAPI 0.110+ |
| **ASGI server** | Uvicorn |
| **LLM** | Mistral-7B-Instruct-v0.2 (HuggingFace Transformers) |
| **Quantisation** | bitsandbytes 4-bit NF4 |
| **Model distribution** | HuggingFace Accelerate (`device_map=auto`) |
| **Embeddings** | `BAAI/bge-small-en` via sentence-transformers 5.3+ |
| **Vector store** | FAISS `IndexFlatIP` (inner-product / cosine similarity) |
| **PDF parsing** | pdfplumber (primary) + pypdf (fallback) |
| **ML framework** | PyTorch 2.10.0+cu128 |
| **Frontend** | React 18 + Vite 5 + TailwindCSS |
| **HTTP client** | Axios |
| **Data validation** | Pydantic v2 |
| **Testing** | pytest + pytest-asyncio + httpx |
| **Python** | 3.11 |

---

## 📋 Requirements

### Python dependencies (`requirements.txt`)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
pydantic>=2.6.0
pdfplumber>=0.11.0
pypdf>=4.1.0
sentence-transformers>=5.3.0
faiss-cpu>=1.8.0
torch>=2.10.0          # install via cu128 index for GPU support
transformers>=4.40.0
accelerate>=0.29.0
bitsandbytes>=0.43.0
numpy>=1.26.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

---

## 🔒 Security Notes

- Uploaded filenames are **sanitised** (`Path(filename).name`) to prevent path-traversal attacks
- CORS is currently set to `allow_origins=["*"]` for development — **tighten this in production**
- The system is designed for internal / on-premise deployment; no data is sent to external APIs

---

## 🗺️ Roadmap

- [ ] OCR support for scanned/image-based PDFs (Tesseract integration)
- [ ] Multi-document collection management (namespaced indexes)
- [ ] Streaming responses via Server-Sent Events
- [ ] Authentication & API key support
- [ ] Docker / docker-compose deployment
- [ ] Support for DOCX, Markdown, and HTML ingestion
- [ ] Conversation history / multi-turn chat

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ for on-premise enterprise AI</p>
