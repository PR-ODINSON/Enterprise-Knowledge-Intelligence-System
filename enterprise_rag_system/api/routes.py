"""
API Routes Module
Defines all FastAPI endpoints exposed by the Enterprise Knowledge Intelligence System.

Endpoints:
  POST /upload                — Upload and index a document
  POST /query                 — Ask a question via the RAG pipeline
  GET  /documents             — List indexed documents
  DELETE /documents/{filename}— Remove a document from disk
  POST /reindex               — Re-index all documents from scratch

Service instances (Embedder, FAISSVectorStore, Retriever, LocalLLM) are
created once at module load time and shared across requests.  The LLM is
initialised lazily on the first /query call to avoid slowing down startup.
"""

import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import DOCUMENTS_DIR
from embedding.embedder import Embedder
from ingestion.document_loader import DocumentLoader
from ingestion.preprocessing import TextPreprocessor
from ingestion.text_chunker import TextChunker
from llm.local_llm import LocalLLM
from retrieval.retriever import Retriever
from utils.logger import get_logger
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared singleton services
# ---------------------------------------------------------------------------

_loader = DocumentLoader()
_chunker = TextChunker()
_preprocessor = TextPreprocessor()
_embedder = Embedder()
_vector_store = FAISSVectorStore()
_retriever = Retriever(embedder=_embedder, vector_store=_vector_store)

# LLM is lazy-loaded on the first /query call (model download is expensive)
_llm: Optional[LocalLLM] = None


def _get_llm() -> LocalLLM:
    """Return the shared LLM instance, loading it on first call."""
    global _llm
    if _llm is None:
        logger.info("Initialising local LLM for the first time…")
        _llm = LocalLLM()
    return _llm


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question to answer",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve from the vector store",
    )


class RetrievedChunk(BaseModel):
    text: str
    score: float
    filename: str
    chunk_id: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    processing_time_seconds: float


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    tags=["Ingestion"],
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document, extract its text, chunk it, generate
    embeddings, and persist everything to the FAISS vector store.

    - Accepted formats: `.pdf`, `.txt`
    - The file is saved to `data/documents/` for future re-indexing.
    - The FAISS index is saved to disk after each successful upload.
    """
    # --- Validate file type ---
    allowed = {".pdf", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    try:
        t0 = time.perf_counter()

        # --- Read and persist the uploaded bytes ---
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path = _loader.save_uploaded_file(file.filename, content)

        # --- Extract and preprocess text ---
        doc = _loader.load_document(file_path)
        doc["text"] = _preprocessor.preprocess(doc["text"])

        if not _preprocessor.is_meaningful(doc["text"]):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Document contains insufficient readable text. "
                    "Scanned / image-based PDFs are not supported without OCR."
                ),
            )

        # --- Chunk ---
        chunks = _chunker.chunk_text(
            doc["text"],
            metadata={
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "file_path": doc["file_path"],
            },
        )
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No valid chunks could be extracted from the document.",
            )

        # --- Embed and index ---
        embeddings, chunks = _embedder.embed_chunks(chunks)
        _vector_store.add_embeddings(embeddings, chunks)
        _vector_store.save()

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Indexed '{file.filename}' — {len(chunks)} chunks in {elapsed:.2f}s"
        )

        return UploadResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            message=(
                f"Successfully indexed {len(chunks)} chunks "
                f"in {elapsed:.2f} seconds."
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Upload failed for '{file.filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {exc}",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question using the RAG pipeline",
    tags=["Retrieval"],
)
async def query_documents(request: QueryRequest):
    """
    Answer a question using the full RAG pipeline:

    1. Embed the question with the sentence-transformers model.
    2. Retrieve the top-k most relevant chunks from FAISS.
    3. Inject the context + question into a Mistral-7B-Instruct prompt.
    4. Return the generated answer together with the source chunks.
    """
    if _vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been indexed yet. Upload documents first.",
        )

    try:
        t0 = time.perf_counter()

        # --- Retrieve ---
        results = _retriever.retrieve(request.question, top_k=request.top_k)
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Could not find relevant context for the given question.",
            )

        # --- Build context string ---
        context_parts = [
            f"[Source {i} — {r.get('filename', 'unknown')}]\n{r['text']}"
            for i, r in enumerate(results, start=1)
        ]
        context = "\n\n---\n\n".join(context_parts)

        # --- Generate ---
        llm = _get_llm()
        answer = llm.answer_question(context, request.question)

        elapsed = time.perf_counter() - t0

        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=[
                RetrievedChunk(
                    text=r["text"],
                    score=r["score"],
                    filename=r.get("filename", "unknown"),
                    chunk_id=r.get("chunk_id", -1),
                )
                for r in results
            ],
            processing_time_seconds=elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )


@router.get(
    "/documents",
    summary="List all documents in the knowledge base",
    tags=["Ingestion"],
)
async def list_documents():
    """
    Return metadata about every document currently stored in
    ``data/documents/``, along with the total number of indexed vectors.
    """
    docs = []
    for fp in DOCUMENTS_DIR.iterdir():
        if fp.is_file() and fp.suffix.lower() in {".pdf", ".txt"}:
            stat = fp.stat()
            docs.append(
                {
                    "filename": fp.name,
                    "size_bytes": stat.st_size,
                    "file_type": fp.suffix.lower(),
                }
            )

    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_vectors_indexed": _vector_store.total_vectors,
    }


@router.delete(
    "/documents/{filename}",
    summary="Delete a document from disk",
    tags=["Ingestion"],
)
async def delete_document(filename: str):
    """
    Remove a document from the local filesystem.

    Note: this does **not** update the FAISS index. Call ``POST /reindex``
    afterwards to rebuild the index without the deleted document.
    """
    # Sanitise to prevent path-traversal
    safe_name = Path(filename).name
    file_path = DOCUMENTS_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{safe_name}' not found.",
        )

    file_path.unlink()
    logger.info(f"Deleted document: {safe_name}")

    return {
        "message": (
            f"Document '{safe_name}' deleted. "
            "Call POST /reindex to update the vector store."
        )
    }


@router.post(
    "/reindex",
    summary="Re-index all documents from scratch",
    tags=["Ingestion"],
)
async def reindex_all():
    """
    Drop the current FAISS index and rebuild it from all documents in
    ``data/documents/``.  Useful after deleting documents or changing
    chunking / embedding settings.
    """
    try:
        t0 = time.perf_counter()

        _vector_store.reset()
        documents = _loader.load_all_documents()

        if not documents:
            return {
                "message": "No documents found in the documents directory.",
                "chunks_indexed": 0,
            }

        # Preprocess and filter
        valid_docs = []
        for doc in documents:
            doc["text"] = _preprocessor.preprocess(doc["text"])
            if _preprocessor.is_meaningful(doc["text"]):
                valid_docs.append(doc)
            else:
                logger.warning(
                    f"Skipped '{doc['filename']}': insufficient readable text"
                )

        all_chunks = _chunker.chunk_documents(valid_docs)

        if all_chunks:
            embeddings, all_chunks = _embedder.embed_chunks(all_chunks)
            _vector_store.add_embeddings(embeddings, all_chunks)
            _vector_store.save()

        elapsed = time.perf_counter() - t0

        return {
            "message": (
                f"Re-indexed {len(valid_docs)} document(s) in {elapsed:.2f}s."
            ),
            "documents_processed": len(valid_docs),
            "chunks_indexed": len(all_chunks),
        }

    except Exception as exc:
        logger.error(f"Re-indexing failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Re-indexing failed: {exc}",
        )
