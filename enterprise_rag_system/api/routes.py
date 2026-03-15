"""
API Routes Module
Defines all FastAPI endpoints exposed by the Enterprise Knowledge Intelligence System.

Endpoints:
  POST /upload                    — Upload and index a document (optional ?collection=)
  POST /query                     — Ask a question via the RAG pipeline
  POST /query/stream              — Streaming SSE query endpoint
  GET  /documents                 — List indexed documents (optional ?collection=)
  DELETE /documents/{filename}    — Remove a document from disk
  POST /reindex                   — Re-index all documents from scratch
  GET  /collections               — List all collections with stats
  GET  /evaluate                  — Return offline RAG quality metrics

Service instances are created once at module load time and shared across requests.
The LLM is initialised lazily on the first /query call.
"""

import json
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import DEFAULT_COLLECTION, RERANK_TOP_K
from embedding.embedder import Embedder
from evaluation.dataset_builder import DatasetBuilder
from evaluation.rag_evaluator import RagEvaluator
from ingestion.document_loader import DocumentLoader
from ingestion.preprocessing import TextPreprocessor
from ingestion.text_chunker import TextChunker
from llm.local_llm import LocalLLM
from retrieval.conversation_memory import ConversationMemory
from retrieval.reranker import Reranker
from utils.cache import get_cache
from utils.logger import get_logger
from vector_store.collection_manager import CollectionManager

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared singleton services
# ---------------------------------------------------------------------------

_embedder = Embedder()
_loader = DocumentLoader()
_chunker = TextChunker()
_preprocessor = TextPreprocessor()
_collection_manager = CollectionManager(embedder=_embedder)
_reranker = Reranker()
_memory = ConversationMemory()
_cache = get_cache()

# Evaluation
_dataset_builder = DatasetBuilder()
_evaluator = RagEvaluator(_dataset_builder)

# LLM is lazy-loaded on the first /query call
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
        description="Number of context chunks to retrieve",
    )
    collection: str = Field(
        default=DEFAULT_COLLECTION,
        description="Document collection to query",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation session UUID for multi-turn memory",
    )


class RetrievedChunk(BaseModel):
    text: str
    score: float
    filename: str
    chunk_id: int
    page_number: Optional[int] = None
    rerank_score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    processing_time_seconds: float
    conversation_id: Optional[str] = None
    cached: bool = False
    collection: str = DEFAULT_COLLECTION


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str
    collection: str = DEFAULT_COLLECTION


class StartConversationResponse(BaseModel):
    conversation_id: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context(results: list) -> str:
    """Format retrieved chunks into an LLM-ready context string."""
    parts = [
        f"[Source {i} — {r.get('filename', 'unknown')}]\\n{r['text']}"
        for i, r in enumerate(results, start=1)
    ]
    return "\\n\\n---\\n\\n".join(parts)


def _make_chunk_response(r: dict) -> RetrievedChunk:
    """Map a raw retrieval result dict to the API response schema."""
    return RetrievedChunk(
        text=r["text"],
        score=r.get("score", 0.0),
        filename=r.get("filename", "unknown"),
        chunk_id=r.get("chunk_id", -1),
        page_number=r.get("page_number"),
        rerank_score=r.get("rerank_score"),
        dense_score=r.get("dense_score"),
        bm25_score=r.get("bm25_score"),
    )


def _run_rag_pipeline(
    question: str,
    collection: str,
    top_k: int,
    conversation_id: Optional[str],
) -> tuple:
    """
    Core RAG pipeline: retrieve → rerank → generate.

    Returns:
        (answer, results, cached) tuple.
    """
    # --- Cache check ---
    cache_key = _cache.make_key(question, collection, top_k)
    cached_data = _cache.get(cache_key)
    if cached_data:
        logger.info(f"Cache HIT for '{question[:60]}…'")
        return cached_data["answer"], cached_data["results"], True

    # --- Retrieve ---
    store, retriever = _collection_manager.get_or_create(collection)
    if store.total_vectors == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection}' has no indexed documents. Upload documents first.",
        )

    candidate_k = max(top_k, RERANK_TOP_K)
    results = retriever.retrieve(question, top_k=candidate_k)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Could not find relevant context for the given question.",
        )

    # --- Rerank ---
    results = _reranker.rerank(question, results, top_k=top_k)

    # --- Conversation history ---
    history = _memory.get_history_text(conversation_id)

    # --- Generate ---
    context = _build_context(results)
    llm = _get_llm()
    answer = llm.answer_question(context, question, history=history)

    # --- Update conversation memory ---
    if conversation_id:
        _memory.add_turn(conversation_id, "user", question)
        _memory.add_turn(conversation_id, "assistant", answer)

    # --- Record for evaluation ---
    _dataset_builder.record(question, answer, results, collection)

    # --- Cache result ---
    _cache.set(cache_key, {"answer": answer, "results": results})

    return answer, results, False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/conversations/start",
    response_model=StartConversationResponse,
    summary="Start a new conversation session",
    tags=["Conversation"],
)
async def start_conversation():
    """Create a new conversation session UUID for multi-turn chat."""
    conv_id = _memory.start_conversation()
    return StartConversationResponse(conversation_id=conv_id)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    tags=["Ingestion"],
)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Query(default=DEFAULT_COLLECTION, description="Target collection"),
):
    """
    Upload a PDF or TXT document, extract its text, chunk it, generate
    embeddings, and persist everything to the specified collection's FAISS store.

    - Accepted formats: `.pdf`, `.txt`
    - Defaults to the ``default`` collection if none specified.
    """
    allowed = {".pdf", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
        )

    try:
        t0 = time.perf_counter()

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save to the collection's documents dir
        docs_dir = _collection_manager.get_documents_dir(collection)
        _loader.documents_dir = docs_dir
        file_path = _loader.save_uploaded_file(file.filename, content)

        # Extract and preprocess
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

        # Chunk — inject page_number metadata from page_map
        page_map = doc.get("page_map", [(1, doc["text"])])
        chunks = _chunker.chunk_text(
            doc["text"],
            metadata={
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "file_path": doc["file_path"],
                "page_number": page_map[0][0] if page_map else None,
            },
        )

        # Propagate page numbers more accurately by matching chunk char positions
        _assign_page_numbers(chunks, page_map)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No valid chunks could be extracted from the document.",
            )

        # Embed and index in the collection's store
        store, retriever = _collection_manager.get_or_create(collection)
        embeddings, chunks = _embedder.embed_chunks(chunks)
        store.add_embeddings(embeddings, chunks)
        store.save()

        # Rebuild BM25
        retriever.rebuild_bm25(store._metadata)

        # Invalidate cache for this collection
        _cache.invalidate_collection(collection)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Indexed '{file.filename}' → '{collection}' — {len(chunks)} chunks in {elapsed:.2f}s"
        )

        return UploadResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks in {elapsed:.2f} seconds.",
            collection=collection,
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

    1. Cache check (Redis or in-memory TTL).
    2. Hybrid retrieval (BM25 + FAISS score fusion).
    3. Cross-encoder reranking.
    4. Inject conversation history + context into Mistral-7B-Instruct prompt.
    5. Return generated answer with source chunks.
    """
    try:
        t0 = time.perf_counter()

        answer, results, cached = _run_rag_pipeline(
            request.question,
            request.collection,
            request.top_k,
            request.conversation_id,
        )

        elapsed = time.perf_counter() - t0
        if cached:
            elapsed = 0.0  # Cache hits are near-instant

        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=[_make_chunk_response(r) for r in results],
            processing_time_seconds=elapsed,
            conversation_id=request.conversation_id,
            cached=cached,
            collection=request.collection,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )


@router.post(
    "/query/stream",
    summary="Stream a RAG answer token-by-token (SSE)",
    tags=["Retrieval"],
)
async def query_stream(request: QueryRequest):
    """
    Streaming variant of /query. Emits Server-Sent Events (SSE):

    - ``data: <token>`` for each generated token
    - ``data: [DONE]`` when generation completes
    - ``data: [ERROR] <message>`` on failure

    Retrieve and rerank run synchronously before streaming begins.
    """
    # Retrieve + rerank first (fast), then stream generation
    store, retriever = _collection_manager.get_or_create(request.collection)
    if store.total_vectors == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection}' has no indexed documents.",
        )

    candidate_k = max(request.top_k, RERANK_TOP_K)
    results = retriever.retrieve(request.question, top_k=candidate_k)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Could not find relevant context for the given question.",
        )

    results = _reranker.rerank(request.question, results, top_k=request.top_k)
    history = _memory.get_history_text(request.conversation_id)
    context = _build_context(results)
    llm = _get_llm()

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer = []
        try:
            for token in llm.answer_question_stream(context, request.question, history):
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Update memory and record after streaming completes
            answer = "".join(full_answer)
            if request.conversation_id:
                _memory.add_turn(request.conversation_id, "user", request.question)
                _memory.add_turn(request.conversation_id, "assistant", answer)
            _dataset_builder.record(
                request.question, answer, results, request.collection
            )

            # Send metadata in final event
            meta = {
                "done": True,
                "retrieved_chunks": [
                    {
                        "text": r["text"][:200],
                        "score": r.get("score", 0.0),
                        "filename": r.get("filename", "unknown"),
                        "chunk_id": r.get("chunk_id", -1),
                        "page_number": r.get("page_number"),
                    }
                    for r in results
                ],
            }
            yield f"data: {json.dumps(meta)}\n\n"

        except Exception as exc:
            logger.error(f"Streaming failed: {exc}", exc_info=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/documents",
    summary="List all documents in a collection",
    tags=["Ingestion"],
)
async def list_documents(
    collection: str = Query(default=DEFAULT_COLLECTION),
):
    """Return metadata about every document in the specified collection."""
    docs_dir = _collection_manager.get_documents_dir(collection)
    store, _ = _collection_manager.get_or_create(collection)

    docs = []
    if docs_dir.exists():
        for fp in docs_dir.iterdir():
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
        "total_vectors_indexed": store.total_vectors,
        "collection": collection,
    }


@router.delete(
    "/documents/{filename}",
    summary="Delete a document from disk",
    tags=["Ingestion"],
)
async def delete_document(
    filename: str,
    collection: str = Query(default=DEFAULT_COLLECTION),
):
    """
    Remove a document from the specified collection.
    Call ``POST /reindex`` afterwards to rebuild the vector index.
    """
    docs_dir = _collection_manager.get_documents_dir(collection)
    safe_name = Path(filename).name
    file_path = docs_dir / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{safe_name}' not found in collection '{collection}'.",
        )

    file_path.unlink()
    _cache.invalidate_collection(collection)
    logger.info(f"Deleted '{safe_name}' from '{collection}'")

    return {
        "message": (
            f"Document '{safe_name}' deleted from '{collection}'. "
            "Call POST /reindex to update the vector store."
        )
    }


@router.post(
    "/reindex",
    summary="Re-index all documents from scratch",
    tags=["Ingestion"],
)
async def reindex_all(
    collection: str = Query(default=DEFAULT_COLLECTION),
):
    """Drop and rebuild the FAISS index for a collection from stored documents."""
    try:
        t0 = time.perf_counter()
        store, retriever = _collection_manager.get_or_create(collection)
        docs_dir = _collection_manager.get_documents_dir(collection)

        store.reset()
        _loader.documents_dir = docs_dir
        documents = _loader.load_all_documents()

        if not documents:
            return {
                "message": f"No documents found in collection '{collection}'.",
                "chunks_indexed": 0,
            }

        valid_docs = []
        for doc in documents:
            doc["text"] = _preprocessor.preprocess(doc["text"])
            if _preprocessor.is_meaningful(doc["text"]):
                valid_docs.append(doc)

        all_chunks = _chunker.chunk_documents(valid_docs)

        if all_chunks:
            embeddings, all_chunks = _embedder.embed_chunks(all_chunks)
            store.add_embeddings(embeddings, all_chunks)
            store.save()
            retriever.rebuild_bm25(store._metadata)

        _cache.invalidate_collection(collection)
        elapsed = time.perf_counter() - t0

        return {
            "message": f"Re-indexed {len(valid_docs)} document(s) in {elapsed:.2f}s.",
            "documents_processed": len(valid_docs),
            "chunks_indexed": len(all_chunks),
            "collection": collection,
        }

    except Exception as exc:
        logger.error(f"Re-indexing failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {exc}")


@router.get(
    "/collections",
    summary="List all document collections",
    tags=["Collections"],
)
async def list_collections():
    """Return all collections with their document and vector counts."""
    collections = _collection_manager.list_collections()
    return {
        "collections": collections,
        "total_collections": len(collections),
        "cache_backend": _cache.backend,
        "active_conversations": _memory.active_sessions,
    }


@router.get(
    "/evaluate",
    summary="Get offline RAG quality metrics",
    tags=["Evaluation"],
)
async def evaluate():
    """
    Compute and return offline RAG quality metrics over recent interactions.

    Metrics (all in [0, 1], higher is better):
    - **faithfulness**: answer tokens grounded in retrieved context
    - **context_recall**: query terms found in retrieved passages
    - **answer_relevancy**: cosine similarity between question and answer embeddings
    """
    return _evaluator.evaluate()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_page_numbers(
    chunks: list,
    page_map: list,
) -> None:
    """
    Assign page_number to each chunk based on text position within page_map.

    Modifies chunks in place.
    """
    if not page_map or len(page_map) == 1:
        for c in chunks:
            c["page_number"] = page_map[0][0] if page_map else 1
        return

    # Build a flat string of all page content to find char offsets
    page_boundaries = []
    offset = 0
    for page_num, page_text in page_map:
        page_boundaries.append((offset, offset + len(page_text), page_num))
        offset += len(page_text) + 1

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        # Find first page containing chunk text
        assigned = page_map[0][0]
        for start, end, pnum in page_boundaries:
            if start <= offset < end:
                assigned = pnum
                break
        chunk["page_number"] = assigned
