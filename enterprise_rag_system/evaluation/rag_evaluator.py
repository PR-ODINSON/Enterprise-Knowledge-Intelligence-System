"""
RAG Evaluator Module
Computes offline quality metrics over collected RAG interactions.

All metrics are computed locally — no external API or LLM call required.

Metrics:
    faithfulness       — fraction of answer tokens present in the context
    context_recall     — fraction of query tokens found in retrieved context
    answer_relevancy   — cosine similarity between question and answer embeddings
                         (uses the same sentence-transformer model as the retriever)

Each metric is in [0, 1]; higher is better.
"""

from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
from evaluation.dataset_builder import DatasetBuilder, EvalSample
from utils.logger import get_logger

logger = get_logger(__name__)


def _token_set(text: str) -> set:
    """Lowercase word-token set for overlap computation."""
    return set(text.lower().split())


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Faithfulness: fraction of unique answer tokens present in the combined context.

    A high score means the answer is grounded in the retrieved passages.
    """
    if not answer.strip() or not contexts:
        return 0.0

    answer_tokens = _token_set(answer)
    context_tokens = _token_set(" ".join(contexts))

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def compute_context_recall(question: str, contexts: List[str]) -> float:
    """
    Context recall: fraction of question tokens found in the retrieved context.

    A high score means the retriever surfaced relevant passages.
    """
    if not question.strip() or not contexts:
        return 0.0

    q_tokens = _token_set(question)
    context_tokens = _token_set(" ".join(contexts))

    # Remove common stopwords to avoid inflating the score
    stopwords = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "or",
                 "how", "why", "when", "where", "who", "which", "are", "was"}
    q_tokens -= stopwords

    if not q_tokens:
        return 1.0  # Question was entirely stopwords

    overlap = q_tokens & context_tokens
    return len(overlap) / len(q_tokens)


class RagEvaluator:
    """
    Computes quality metrics over a dataset of RAG interactions.

    The sentence-transformer model is loaded lazily on first call to
    ``evaluate()`` and reused for all subsequent evaluations.

    Args:
        dataset_builder: DatasetBuilder instance holding evaluation samples.
        model_name:      Model used for embedding-based answer_relevancy.
        device:          Torch device for the embedding model.
    """

    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.model_name = model_name
        self.device = device
        self._embed_model: Optional[SentenceTransformer] = None

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embed_model is None:
            logger.info(f"Loading embedding model for evaluation: {self.model_name}")
            self._embed_model = SentenceTransformer(
                self.model_name, device=self.device
            )
        return self._embed_model

    def _answer_relevancy(self, question: str, answer: str) -> float:
        """Cosine similarity between question and answer embeddings."""
        if not answer.strip():
            return 0.0
        embs = self.embed_model.encode(
            [question, answer], normalize_embeddings=True
        )
        return float(np.dot(embs[0], embs[1]))

    def evaluate(self) -> Dict:
        """
        Compute metrics over all buffered evaluation samples.

        Returns:
            Dict with:
                ``sample_count``            — number of samples evaluated
                ``avg_faithfulness``        — mean faithfulness score
                ``avg_context_recall``      — mean context recall score
                ``avg_answer_relevancy``    — mean cosine-similarity relevancy
                ``samples``                 — per-sample breakdown list
        """
        samples = self.dataset_builder.get_samples()

        if not samples:
            logger.warning("No evaluation samples available")
            return {
                "sample_count": 0,
                "avg_faithfulness": None,
                "avg_context_recall": None,
                "avg_answer_relevancy": None,
                "samples": [],
                "message": "No queries have been made yet. Ask some questions first.",
            }

        per_sample = []
        faithfulness_scores = []
        recall_scores = []
        relevancy_scores = []

        for s in samples:
            f = compute_faithfulness(s.answer, s.contexts)
            r = compute_context_recall(s.question, s.contexts)
            rel = self._answer_relevancy(s.question, s.answer)

            faithfulness_scores.append(f)
            recall_scores.append(r)
            relevancy_scores.append(rel)

            per_sample.append({
                "question": s.question[:120],
                "faithfulness": round(f, 4),
                "context_recall": round(r, 4),
                "answer_relevancy": round(rel, 4),
                "collection": s.collection,
            })

        result = {
            "sample_count": len(samples),
            "avg_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
            "avg_context_recall": round(float(np.mean(recall_scores)), 4),
            "avg_answer_relevancy": round(float(np.mean(relevancy_scores)), 4),
            "samples": per_sample,
        }

        logger.info(
            f"Evaluation complete — {len(samples)} samples | "
            f"faithfulness={result['avg_faithfulness']:.3f} "
            f"recall={result['avg_context_recall']:.3f} "
            f"relevancy={result['avg_answer_relevancy']:.3f}"
        )
        return result
