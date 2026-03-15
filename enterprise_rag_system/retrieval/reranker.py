"""
Cross-Encoder Reranker Module
Applies a cross-encoder model to re-score candidate chunks against the query,
producing more accurate relevance rankings than bi-encoder similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6-layer MiniLM fine-tuned on MS-MARCO passage ranking
  - ~22 MB — fast CPU inference, fits comfortably alongside the 7B LLM
  - Input: (query, passage) pairs → scalar relevance score

Pipeline:
    initial_results (top RERANK_TOP_K)
    → score all (query, chunk) pairs with cross-encoder
    → sort by cross-encoder score
    → return top_k
"""

from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.config import RERANK_DEVICE, RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Lazy-loaded cross-encoder reranker.

    The model is downloaded and initialised on the first call to ``rerank()``
    to avoid paying the loading cost at startup.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
        device:     Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str = RERANK_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[CrossEncoder] = None
        logger.info(
            f"Reranker configured — model: {model_name}, device: {device}"
        )

    # ------------------------------------------------------------------
    # Model access (lazy)
    # ------------------------------------------------------------------

    @property
    def model(self) -> CrossEncoder:
        """Load and cache the cross-encoder model on first access."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name} …")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Cross-encoder model ready")
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank *chunks* by cross-encoder relevance to *query*.

        If reranking is disabled via config or the chunk list is empty,
        the original order is preserved and top_k is applied.

        Args:
            query:  Natural-language query string.
            chunks: Candidate chunks from initial retrieval.
            top_k:  Number of chunks to return after reranking.

        Returns:
            Top-k chunks sorted by descending cross-encoder score.
            Each chunk dict gains a ``"rerank_score"`` field.
        """
        if not RERANK_ENABLED or not chunks:
            return chunks[:top_k]

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in ranked[:top_k]:
            c = dict(chunk)
            c["rerank_score"] = float(score)
            results.append(c)

        logger.info(
            f"Reranked {len(chunks)} candidates → top-{top_k} "
            f"(best rerank score: {results[0]['rerank_score']:.4f})"
        )
        return results
