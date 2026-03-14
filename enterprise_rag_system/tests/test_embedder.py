"""
Tests — Embedding Pipeline
Unit tests for the Embedder class.
Run with:  pytest tests/test_embedder.py -v

The sentence-transformers model is mocked so these tests run offline
without downloading weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedding.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

MOCK_DIM = 384


def _make_mock_model(dim: int = MOCK_DIM) -> MagicMock:
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim

    def fake_encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.random.rand(dim).astype(np.float32)
        n = len(texts)
        vecs = np.random.rand(n, dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        return vecs

    mock.encode.side_effect = fake_encode
    return mock


@pytest.fixture()
def embedder() -> Embedder:
    """Embedder with its internal model replaced by a mock."""
    emb = Embedder(model_name="mock-model")
    emb._model = _make_mock_model()
    return emb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_embed_text_shape(self, embedder: Embedder):
        """Single text → 1-D array of length EMBEDDING_DIM."""
        vec = embedder.embed_text("What is machine learning?")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == MOCK_DIM

    def test_embed_text_dtype(self, embedder: Embedder):
        """Output array must be float32."""
        vec = embedder.embed_text("Some text")
        assert vec.dtype == np.float32

    def test_embed_texts_shape(self, embedder: Embedder):
        """List of N texts → 2-D array (N, DIM)."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vecs = embedder.embed_texts(texts)
        assert vecs.shape == (3, MOCK_DIM)

    def test_embed_texts_empty(self, embedder: Embedder):
        """Empty list → empty array with correct second dimension."""
        vecs = embedder.embed_texts([])
        assert vecs.shape[0] == 0
        assert vecs.shape[1] == MOCK_DIM

    def test_embed_chunks_alignment(self, embedder: Embedder):
        """embed_chunks must return embeddings aligned with input chunks."""
        chunks = [
            {"text": "Chunk A", "chunk_id": 0},
            {"text": "Chunk B", "chunk_id": 1},
            {"text": "Chunk C", "chunk_id": 2},
        ]
        embeddings, returned_chunks = embedder.embed_chunks(chunks)

        assert embeddings.shape == (3, MOCK_DIM)
        assert returned_chunks is chunks   # same object, not a copy

    def test_embedding_dimension_property(self, embedder: Embedder):
        """embedding_dimension should delegate to the model."""
        assert embedder.embedding_dimension == MOCK_DIM

    def test_model_lazy_loading(self):
        """Model should not be loaded until the first encode call."""
        emb = Embedder(model_name="mock-model", device="cpu")
        assert emb._model is None  # not loaded yet

        with patch("embedding.embedder.SentenceTransformer") as MockST:
            MockST.return_value = _make_mock_model()
            _ = emb.model  # trigger lazy load
            MockST.assert_called_once_with("mock-model", device="cpu")

    def test_embed_texts_dtype_float32(self, embedder: Embedder):
        """All outputs should be float32 regardless of mock."""
        texts = ["Hello", "World"]
        vecs = embedder.embed_texts(texts)
        assert vecs.dtype == np.float32
