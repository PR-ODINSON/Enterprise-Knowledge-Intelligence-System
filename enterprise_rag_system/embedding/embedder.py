"""
Embedding Module
Generates dense vector representations of text using a sentence-transformers
model (BAAI/bge-small-en by default).

Key characteristics of BAAI/bge-small-en:
  - Output dimension: 384
  - Optimised for semantic similarity / retrieval tasks
  - Lightweight enough to run efficiently on CPU

Embeddings are L2-normalised so that inner-product search is equivalent
to cosine similarity, which matches the FAISS IndexFlatIP configuration.
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Wraps a sentence-transformers model to produce fixed-size text embeddings.

    The model is loaded lazily on the first call to avoid paying the
    initialisation cost at import time.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE) -> None:
        """
        Args:
            model_name: HuggingFace / sentence-transformers model identifier.
            device:     Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        """
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        logger.info(f"Embedder configured — model: {model_name}, device: {device}")

    # ------------------------------------------------------------------
    # Model access (lazy loading)
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        """Load and cache the sentence-transformers model on first access."""
        if self._model is None:
            logger.info(
                f"Loading sentence-transformers model: {self.model_name} "
                f"on device: {self.device}"
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"Embedder ready — dimension: "
                f"{self._model.get_sentence_embedding_dimension()}, "
                f"device: {self.device}"
            )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Dimensionality of the model's output vectors."""
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an L2-normalised embedding for a single string.

        Args:
            text: Input text.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate L2-normalised embeddings for a list of strings.

        Texts are processed in batches to balance memory usage and throughput.

        Args:
            texts:      List of input strings.
            batch_size: Number of strings to encode per forward pass.

        Returns:
            Float32 numpy array of shape ``(len(texts), embedding_dim)``.
            Returns an empty array of shape ``(0, embedding_dim)`` when
            *texts* is empty.
        """
        if not texts:
            logger.debug("embed_texts called with empty list — returning empty array")
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        logger.info(
            f"Embedding {len(texts)} texts "
            f"(batch_size={batch_size}, show_progress={len(texts) > 10})"
        )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )

        result = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings generated — shape: {result.shape}")
        return result

    def embed_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Convenience method that extracts text from chunk dicts, embeds them,
        and returns both the embedding matrix and the original chunk list.

        Args:
            chunks:     List of chunk dictionaries each containing a ``"text"`` key.
            batch_size: Batch size forwarded to ``embed_texts``.

        Returns:
            A ``(embeddings, chunks)`` tuple where *embeddings* has shape
            ``(len(chunks), embedding_dim)`` and *chunks* is the unchanged
            input list — preserving index alignment between the two.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        return embeddings, chunks
