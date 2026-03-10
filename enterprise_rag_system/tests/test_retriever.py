"""
Tests — Retrieval & Vector Store
Unit tests for FAISSVectorStore and Retriever.
Run with:  pytest tests/test_retriever.py -v

All heavy dependencies (sentence-transformers, FAISS disk I/O) are mocked
so these tests execute offline and without GPU.
"""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding.embedder import Embedder
from retrieval.retriever import Retriever
from vector_store.faiss_store import FAISSVectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_vecs(n: int, dim: int = DIM) -> np.ndarray:
    """Generate n random unit-normalised float32 vectors."""
    vecs = np.random.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _make_metadata(n: int) -> List[dict]:
    return [
        {
            "text": f"Chunk {i} content about topic {i}.",
            "chunk_id": i,
            "filename": f"doc_{i % 3}.txt",
            "file_type": ".txt",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path: Path) -> FAISSVectorStore:
    """Return a FAISSVectorStore backed by a temporary directory."""
    return FAISSVectorStore(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        dimension=DIM,
    )


@pytest.fixture()
def populated_store(tmp_store: FAISSVectorStore) -> FAISSVectorStore:
    """A store pre-loaded with 10 vectors."""
    vecs = _rand_vecs(10)
    meta = _make_metadata(10)
    tmp_store.add_embeddings(vecs, meta)
    return tmp_store


@pytest.fixture()
def mock_embedder() -> Embedder:
    """Embedder whose encode calls return deterministic random vectors."""
    emb = MagicMock(spec=Embedder)
    emb.embed_text.side_effect = lambda text: _rand_vecs(1)[0]
    emb.embed_texts.side_effect = lambda texts, **_: _rand_vecs(len(texts))
    emb.embed_chunks.side_effect = lambda chunks, **_: (
        _rand_vecs(len(chunks)),
        chunks,
    )
    return emb


# ---------------------------------------------------------------------------
# FAISSVectorStore tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def test_initial_total_vectors_zero(self, tmp_store: FAISSVectorStore):
        assert tmp_store.total_vectors == 0

    def test_add_embeddings_increments_count(self, tmp_store: FAISSVectorStore):
        vecs = _rand_vecs(5)
        meta = _make_metadata(5)
        tmp_store.add_embeddings(vecs, meta)
        assert tmp_store.total_vectors == 5

    def test_add_mismatched_sizes_raises(self, tmp_store: FAISSVectorStore):
        vecs = _rand_vecs(3)
        meta = _make_metadata(5)   # wrong length
        with pytest.raises(ValueError):
            tmp_store.add_embeddings(vecs, meta)

    def test_search_returns_top_k(self, populated_store: FAISSVectorStore):
        query = _rand_vecs(1)[0]
        results = populated_store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_result_has_score_and_metadata(
        self, populated_store: FAISSVectorStore
    ):
        query = _rand_vecs(1)[0]
        results = populated_store.search(query, top_k=1)
        r = results[0]
        assert "score" in r
        assert "text" in r
        assert "chunk_id" in r
        assert "filename" in r

    def test_search_scores_descending(self, populated_store: FAISSVectorStore):
        query = _rand_vecs(1)[0]
        results = populated_store.search(query, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty(self, tmp_store: FAISSVectorStore):
        query = _rand_vecs(1)[0]
        results = tmp_store.search(query, top_k=5)
        assert results == []

    def test_search_top_k_capped_at_available(self, tmp_store: FAISSVectorStore):
        vecs = _rand_vecs(3)
        meta = _make_metadata(3)
        tmp_store.add_embeddings(vecs, meta)
        results = tmp_store.search(_rand_vecs(1)[0], top_k=10)
        assert len(results) == 3   # only 3 vectors exist

    def test_save_and_load(self, tmp_store: FAISSVectorStore, tmp_path: Path):
        vecs = _rand_vecs(5)
        meta = _make_metadata(5)
        tmp_store.add_embeddings(vecs, meta)
        tmp_store.save()

        # Load into a fresh instance pointing at the same files
        loaded = FAISSVectorStore(
            index_path=tmp_path / "index.bin",
            metadata_path=tmp_path / "meta.json",
            dimension=DIM,
        )
        assert loaded.total_vectors == 5

    def test_reset_clears_store(self, populated_store: FAISSVectorStore):
        populated_store.reset()
        assert populated_store.total_vectors == 0


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_retrieve_returns_results(
        self, populated_store: FAISSVectorStore, mock_embedder: Embedder
    ):
        retriever = Retriever(
            embedder=mock_embedder, vector_store=populated_store, top_k=3
        )
        results = retriever.retrieve("What is this about?")
        assert len(results) > 0

    def test_retrieve_respects_top_k(
        self, populated_store: FAISSVectorStore, mock_embedder: Embedder
    ):
        retriever = Retriever(
            embedder=mock_embedder, vector_store=populated_store, top_k=5
        )
        results = retriever.retrieve("Question", top_k=2)
        assert len(results) == 2

    def test_retrieve_context_returns_string(
        self, populated_store: FAISSVectorStore, mock_embedder: Embedder
    ):
        retriever = Retriever(
            embedder=mock_embedder, vector_store=populated_store
        )
        context = retriever.retrieve_context("Some question")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_retrieve_context_empty_store(self, mock_embedder: Embedder, tmp_path: Path):
        empty_store = FAISSVectorStore(
            index_path=tmp_path / "idx.bin",
            metadata_path=tmp_path / "meta.json",
            dimension=DIM,
        )
        retriever = Retriever(embedder=mock_embedder, vector_store=empty_store)
        context = retriever.retrieve_context("Any question")
        assert "No relevant context found" in context

    def test_retrieve_calls_embedder(
        self, populated_store: FAISSVectorStore, mock_embedder: Embedder
    ):
        """embed_text must be called exactly once per retrieve() call."""
        retriever = Retriever(embedder=mock_embedder, vector_store=populated_store)
        retriever.retrieve("Test query")
        mock_embedder.embed_text.assert_called_once_with("Test query")

    def test_retrieve_context_labels_sources(
        self, populated_store: FAISSVectorStore, mock_embedder: Embedder
    ):
        """Context string should contain '[Source N' markers."""
        retriever = Retriever(embedder=mock_embedder, vector_store=populated_store, top_k=3)
        context = retriever.retrieve_context("What is the topic?")
        assert "[Source 1" in context
