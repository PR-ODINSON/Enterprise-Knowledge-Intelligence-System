"""
Hybrid Retriever Module
Combines BM25 lexical retrieval with FAISS dense semantic retrieval
using score fusion for higher-quality, more robust results.

Score fusion strategy:
    final_score = VECTOR_WEIGHT * normalised_dense + BM25_WEIGHT * normalised_bm25

The BM25 index is rebuilt automatically whenever documents are added or deleted.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import BM25_WEIGHT, HYBRID_ENABLED, TOP_K_RESULTS, VECTOR_WEIGHT
from embedding.embedder import Embedder
from utils.logger import get_logger
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Performs hybrid retrieval by combining BM25 and FAISS scores.

    When HYBRID_ENABLED is False, falls back to pure FAISS dense retrieval.

    Args:
        embedder:     Embedder instance for query/document dense encoding.
        vector_store: FAISSVectorStore instance to search against.
        top_k:        Default number of final results to return.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or FAISSVectorStore()
        self.top_k = top_k
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_chunks: List[Dict[str, Any]] = []

        # Bootstrap BM25 from any pre-loaded vector store metadata
        if self.vector_store._metadata:
            self.rebuild_bm25(self.vector_store._metadata)

    # ------------------------------------------------------------------
    # BM25 management
    # ------------------------------------------------------------------

    def rebuild_bm25(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Rebuild the BM25 index from the given chunk list.

        Must be called after any add or delete operation to keep the
        BM25 index in sync with the FAISS vector store.

        Args:
            chunks: List of chunk dicts each containing a ``"text"`` key.
        """
        if not chunks:
            self._bm25 = None
            self._corpus_chunks = []
            logger.info("BM25 index cleared — no chunks available")
            return

        self._corpus_chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index rebuilt — {len(chunks)} documents")

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks using hybrid scoring.

        Steps:
            1. Dense retrieval: embed query → FAISS search (top RERANK_TOP_K candidates).
            2. Lexical retrieval: BM25 on the same candidate pool.
            3. Normalise both score vectors to [0, 1].
            4. Fuse: final = VECTOR_WEIGHT * dense + BM25_WEIGHT * bm25.
            5. Sort by fused score and return top_k.

        Args:
            query: Natural-language question or search string.
            top_k: Override the instance default.

        Returns:
            List of chunk dicts enriched with ``"score"`` (fused) and
            ``"dense_score"`` / ``"bm25_score"`` for transparency.
        """
        k = top_k if top_k is not None else self.top_k

        if not HYBRID_ENABLED or self._bm25 is None:
            return self._dense_only(query, k)

        # --- 1. Dense candidates (retrieve more to give BM25 room to reorder) ---
        candidate_k = min(max(k * 4, 20), self.vector_store.total_vectors)
        query_embedding = self.embedder.embed_text(query)
        dense_results = self.vector_store.search(query_embedding, top_k=candidate_k)

        if not dense_results:
            return []

        # --- 2. BM25 scores on the candidate pool ---
        query_tokens = _tokenize(query)
        bm25_scores_all = self._bm25.get_scores(query_tokens)

        # Match dense results back to their BM25 corpus positions via chunk_id + filename
        chunk_key = lambda c: (c.get("filename", ""), c.get("chunk_id", -1))
        bm25_key_map = {
            chunk_key(c): bm25_scores_all[i]
            for i, c in enumerate(self._corpus_chunks)
        }

        # --- 3. Build fused score list ---
        dense_scores = np.array([r["score"] for r in dense_results], dtype=np.float32)
        bm25_raw = np.array(
            [bm25_key_map.get(chunk_key(r), 0.0) for r in dense_results],
            dtype=np.float32,
        )

        dense_norm = _minmax(dense_scores)
        bm25_norm = _minmax(bm25_raw)

        fused = VECTOR_WEIGHT * dense_norm + BM25_WEIGHT * bm25_norm

        # --- 4. Sort and return top-k ---
        order = np.argsort(fused)[::-1]
        results = []
        for idx in order[:k]:
            chunk = dict(dense_results[idx])
            chunk["score"] = float(fused[idx])
            chunk["dense_score"] = float(dense_norm[idx])
            chunk["bm25_score"] = float(bm25_norm[idx])
            results.append(chunk)

        logger.info(
            f"Hybrid retrieval: top-{k} from {len(dense_results)} candidates "
            f"(VECTOR={VECTOR_WEIGHT}, BM25={BM25_WEIGHT})"
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to pure FAISS dense retrieval."""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info(f"Dense-only retrieval: top-{top_k} results")
        return results


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Normalise array values to [0, 1]. Returns zeros if range is 0."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
