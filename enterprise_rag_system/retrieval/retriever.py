"""
Retriever Module
Orchestrates query embedding and FAISS similarity search to surface the
most semantically relevant document chunks for a given user query.

Typical usage:
    retriever = Retriever()
    chunks    = retriever.retrieve("What is the refund policy?")
    context   = retriever.retrieve_context("What is the refund policy?")
"""

from typing import Any, Dict, List, Optional

from embedding.embedder import Embedder
from vector_store.faiss_store import FAISSVectorStore
from app.config import TOP_K_RESULTS
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Converts a natural-language query into an embedding and retrieves the
    top-k most similar chunks from the FAISS vector store.

    Both the Embedder and FAISSVectorStore are injected for testability;
    default instances are created automatically when not provided.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        """
        Args:
            embedder:     Embedder instance for query encoding.
            vector_store: FAISSVectorStore instance to search against.
            top_k:        Default number of chunks to return per query.
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or FAISSVectorStore()
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed *query* and return the most relevant document chunks.

        Args:
            query: Natural-language question or search string.
            top_k: Override the default number of results to return.

        Returns:
            List of chunk dicts sorted by descending similarity score.
            Each dict contains at minimum:
              - ``score``     — cosine similarity (0–1, higher is better)
              - ``text``      — chunk text
              - ``filename``  — source document name
              - ``chunk_id``  — position within that document
        """
        k = top_k if top_k is not None else self.top_k
        display_query = query[:80] + "…" if len(query) > 80 else query
        logger.info(f"Retrieving top-{k} chunks for: '{display_query}'")

        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=k)

        logger.info(f"Retrieved {len(results)} chunk(s)")
        if results:
            logger.debug(f"Best match score: {results[0]['score']:.4f}")

        return results

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant chunks and format them as a single context string
        ready to be injected into an LLM prompt.

        Each chunk is prefixed with its source filename and index number for
        attribution. Chunks are separated by a horizontal rule.

        Args:
            query: Natural-language question.
            top_k: Override the default number of chunks to include.

        Returns:
            Multi-line string with labelled source sections, or a
            "No relevant context found." message when the store is empty.
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant context found."

        parts: List[str] = []
        for i, chunk in enumerate(results, start=1):
            source = chunk.get("filename", "unknown")
            parts.append(f"[Source {i} — {source}]\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)
