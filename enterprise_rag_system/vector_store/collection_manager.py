"""
Collection Manager Module
Manages multiple independent document collections, each with its own
FAISS vector store, BM25 index, and documents directory.

Each collection lives at:
    data/collections/{collection_id}/
        documents/          — uploaded source files
        faiss_index.bin     — FAISS index for this collection
        metadata.json       — chunk metadata sidecar

The "default" collection maps to the existing data/ layout for
full backward compatibility with the original single-collection setup.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from app.config import (
    COLLECTIONS_DIR,
    DEFAULT_COLLECTION,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    DOCUMENTS_DIR,
)
from embedding.embedder import Embedder
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import get_logger
from vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class CollectionManager:
    """
    Registry of document collections.

    Each collection entry is a (FAISSVectorStore, HybridRetriever) pair.
    Collections are created on first access.

    The ``default`` collection re-uses the legacy paths for backward
    compatibility (data/faiss_index.bin, data/metadata.json,
    data/documents/).
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self._embedder = embedder or Embedder()
        self._registry: Dict[str, Tuple[FAISSVectorStore, HybridRetriever]] = {}
        # Pre-load the default collection
        self.get_or_create(DEFAULT_COLLECTION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(
        self, collection_id: str
    ) -> Tuple[FAISSVectorStore, HybridRetriever]:
        """
        Return the (FAISSVectorStore, HybridRetriever) for a collection,
        creating it if it does not exist yet.

        Args:
            collection_id: Slug-style identifier (e.g. ``"research"``).

        Returns:
            Tuple of (vector_store, hybrid_retriever).
        """
        cid = _sanitise(collection_id)
        if cid in self._registry:
            return self._registry[cid]

        index_path, meta_path, docs_dir = self._paths_for(cid)
        docs_dir.mkdir(parents=True, exist_ok=True)

        store = FAISSVectorStore(
            index_path=index_path,
            metadata_path=meta_path,
            dimension=EMBEDDING_DIMENSION,
        )
        retriever = HybridRetriever(
            embedder=self._embedder,
            vector_store=store,
        )

        self._registry[cid] = (store, retriever)
        logger.info(f"Collection ready: '{cid}' ({store.total_vectors} vectors)")
        return store, retriever

    def get_documents_dir(self, collection_id: str) -> Path:
        """Return the documents directory for a collection."""
        _, _, docs_dir = self._paths_for(_sanitise(collection_id))
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    def list_collections(self) -> list:
        """
        Enumerate all known collections with their stats.

        Returns:
            List of dicts with ``collection_id``, ``total_vectors``,
            ``total_documents`` fields.
        """
        result = []
        # Include collections on disk not yet loaded into the registry
        known_ids = set(self._registry.keys())

        # Scan collections directory for persisted collections
        if COLLECTIONS_DIR.exists():
            for entry in COLLECTIONS_DIR.iterdir():
                if entry.is_dir():
                    known_ids.add(entry.name)

        # Always include default
        known_ids.add(DEFAULT_COLLECTION)

        for cid in sorted(known_ids):
            store, _ = self.get_or_create(cid)
            docs_dir = self.get_documents_dir(cid)
            doc_count = sum(
                1 for f in docs_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}
            ) if docs_dir.exists() else 0

            result.append({
                "collection_id": cid,
                "total_vectors": store.total_vectors,
                "total_documents": doc_count,
            })

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _paths_for(collection_id: str):
        """
        Return (index_path, metadata_path, documents_dir) for a collection.

        The ``default`` collection uses legacy root-level paths.
        """
        if collection_id == DEFAULT_COLLECTION:
            return FAISS_INDEX_PATH, METADATA_PATH, DOCUMENTS_DIR

        coll_dir = COLLECTIONS_DIR / collection_id
        return (
            coll_dir / "faiss_index.bin",
            coll_dir / "metadata.json",
            coll_dir / "documents",
        )


def _sanitise(collection_id: str) -> str:
    """Strip whitespace; fallback to default on empty input."""
    cid = collection_id.strip().lower() if collection_id else ""
    return cid if cid else DEFAULT_COLLECTION
