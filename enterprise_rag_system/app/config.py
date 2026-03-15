"""
Configuration Module
Central configuration for the Enterprise Knowledge Intelligence System.
All tuneable parameters are defined here for easy modification.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
COLLECTIONS_DIR = DATA_DIR / "collections"

# Ensure all required directories exist at startup
for _dir in (DATA_DIR, DOCUMENTS_DIR, MODELS_DIR, LOGS_DIR, COLLECTIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------

# BAAI/bge-small-en produces 384-dimensional embeddings; fast and efficient
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en"
EMBEDDING_DIMENSION: int = 384
EMBEDDING_DEVICE: str = "cuda"      # CUDA: confirmed working on RTX 5050 (sm_120) with torch 2.10.0+cu128

# ---------------------------------------------------------------------------
# Text Chunking Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500       # Target chunk size in approximate tokens
CHUNK_OVERLAP: int = 50     # Overlapping tokens between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------

TOP_K_RESULTS: int = 5      # Default number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# Hybrid Retrieval Configuration
# ---------------------------------------------------------------------------

HYBRID_ENABLED: bool = True     # Enable BM25 + FAISS hybrid retrieval
BM25_WEIGHT: float = 0.5        # Weight for BM25 (lexical) scores
VECTOR_WEIGHT: float = 0.5      # Weight for FAISS (dense) scores

# ---------------------------------------------------------------------------
# Cross-Encoder Reranking Configuration
# ---------------------------------------------------------------------------

RERANK_ENABLED: bool = True
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K: int = 20          # Retrieve this many candidates before reranking
RERANK_DEVICE: str = "cpu"      # Cross-encoder is light — CPU is fine

# ---------------------------------------------------------------------------
# Query Caching Configuration
# ---------------------------------------------------------------------------

CACHE_ENABLED: bool = True
CACHE_TTL: int = 600            # Cache TTL in seconds (10 minutes)
REDIS_URL: str = "redis://localhost:6379/0"

# ---------------------------------------------------------------------------
# Conversation Memory Configuration
# ---------------------------------------------------------------------------

MEMORY_MAX_TURNS: int = 10      # Maximum turns retained per conversation session

# ---------------------------------------------------------------------------
# Streaming Configuration
# ---------------------------------------------------------------------------

STREAMING_ENABLED: bool = True

# ---------------------------------------------------------------------------
# Multi-Collection Configuration
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION: str = "default"

# ---------------------------------------------------------------------------
# Local LLM Configuration
# ---------------------------------------------------------------------------

LLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MAX_NEW_TOKENS: int = 512       # Maximum tokens to generate per response
LLM_TEMPERATURE: float = 0.1        # Low temperature for factual, deterministic answers
LLM_DEVICE: str = "cuda"            # CUDA: torch 2.10.0+cu128 supports RTX 5050 (Blackwell/sm_120)
LLM_USE_4BIT: bool = True           # 4-bit NF4 quantisation via bitsandbytes (~5 GB VRAM vs ~14 GB float16)

# ---------------------------------------------------------------------------
# FAISS Persistence Paths (legacy default collection paths)
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# API Server Configuration
# ---------------------------------------------------------------------------

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
