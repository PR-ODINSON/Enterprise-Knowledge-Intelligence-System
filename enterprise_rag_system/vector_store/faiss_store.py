"""
FAISS Vector Store Module
Manages a persistent FAISS index for storing and retrieving document embeddings.

Index type — IndexFlatIP (Inner Product):
  Combined with L2-normalised embeddings (produced by the Embedder), inner-
  product search is mathematically equivalent to cosine similarity search.
  This avoids the overhead of IndexFlatL2 + manual cosine conversion.

Persistence:
  - The FAISS binary index is saved/loaded via ``faiss.write_index`` /
    ``faiss.read_index``.
  - Per-vector metadata (chunk text, filename, chunk_id, …) is stored as a
    parallel JSON array so that search results carry full context.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.config import EMBEDDING_DIMENSION, FAISS_INDEX_PATH, METADATA_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    Wraps a FAISS ``IndexFlatIP`` with a JSON metadata sidecar.

    Each call to ``add_embeddings`` appends vectors and their metadata to the
    store. Index position ``i`` corresponds directly to ``self._metadata[i]``,
    so retrieval results are always paired with the right chunk.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []

        # Attempt to load an existing persisted index on startup
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> faiss.IndexFlatIP:
        """Return the FAISS index, creating a new one if it doesn't exist."""
        if self._index is None:
            logger.info(
                f"Creating new FAISS IndexFlatIP (dimension={self.dimension})"
            )
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently stored in the index."""
        return self._index.ntotal if self._index is not None else 0

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add a batch of embeddings and their metadata to the vector store.

        The embeddings are re-normalised with ``faiss.normalize_L2`` to ensure
        cosine similarity semantics even if the Embedder's normalisation was
        skipped upstream.

        Args:
            embeddings: Float32 array of shape ``(n, dimension)``.
            metadata:   List of n metadata dicts (one per embedding).

        Raises:
            ValueError: If the number of embeddings and metadata items differ.
        """
        n = len(embeddings)
        if n != len(metadata):
            raise ValueError(
                f"Embedding count ({n}) must equal metadata count ({len(metadata)})."
            )

        # Ensure correct dtype
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalise so inner-product == cosine similarity
        embeddings_copy = np.copy(embeddings)  # normalize_L2 operates in-place
        faiss.normalize_L2(embeddings_copy)

        self.index.add(embeddings_copy)
        self._metadata.extend(metadata)

        logger.info(
            f"Added {n} vectors — store total: {self.total_vectors:,}"
        )

    def reset(self) -> None:
        """Clear all vectors and metadata from the store (in-memory only)."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = []
        logger.info("Vector store reset — all data cleared")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the *top_k* most similar vectors to *query_embedding*.

        Args:
            query_embedding: Shape ``(dimension,)`` or ``(1, dimension)``.
            top_k:           Maximum number of results to return.

        Returns:
            List of result dicts sorted by descending similarity score.
            Each dict merges ``{"score": float}`` with the chunk's metadata.
            Returns an empty list when the store has no vectors.
        """
        if self.total_vectors == 0:
            logger.warning("Search called on empty vector store — returning []")
            return []

        # Ensure shape is (1, dimension) for FAISS
        query = (
            query_embedding.reshape(1, -1)
            if query_embedding.ndim == 1
            else query_embedding
        )
        query = query.astype(np.float32)
        query_copy = np.copy(query)
        faiss.normalize_L2(query_copy)

        # Cap top_k to available vectors
        k = min(top_k, self.total_vectors)
        scores, indices = self.index.search(query_copy, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS sentinel for unfilled slots
                continue
            results.append({"score": float(score), **self._metadata[idx]})

        logger.debug(f"Search returned {len(results)} results (top_k={top_k})")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and metadata JSON to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved FAISS index ({self.total_vectors:,} vectors) "
            f"→ {self.index_path}"
        )

    def load(self) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.

        Returns:
            ``True`` if both files were found and loaded successfully.
            ``False`` if either file is missing (a fresh index will be used).
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("No persisted index found — starting with an empty store")
            return False

        self._index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info(
            f"Loaded FAISS index: {self.total_vectors:,} vectors "
            f"from {self.index_path}"
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_if_exists(self) -> None:
        """Called once during __init__ to restore any persisted state."""
        self.load()
