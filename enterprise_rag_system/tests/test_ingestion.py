"""
Tests — Ingestion Pipeline
Unit tests for DocumentLoader, TextChunker, and TextPreprocessor.
Run with:  pytest tests/test_ingestion.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.document_loader import DocumentLoader
from ingestion.preprocessing import TextPreprocessor
from ingestion.text_chunker import TextChunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for document storage."""
    docs = tmp_path / "documents"
    docs.mkdir()
    return docs


@pytest.fixture()
def loader(tmp_docs_dir: Path) -> DocumentLoader:
    return DocumentLoader(documents_dir=tmp_docs_dir)


@pytest.fixture()
def chunker() -> TextChunker:
    return TextChunker(chunk_size=200, chunk_overlap=20)


@pytest.fixture()
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def test_load_txt_file(self, loader: DocumentLoader, tmp_docs_dir: Path):
        """Should extract full text from a plain-text file."""
        txt_file = tmp_docs_dir / "sample.txt"
        txt_file.write_text("Hello world. This is a test document.", encoding="utf-8")

        doc = loader.load_document(txt_file)

        assert doc["filename"] == "sample.txt"
        assert doc["file_type"] == ".txt"
        assert "Hello world" in doc["text"]
        assert doc["char_count"] > 0

    def test_unsupported_extension_raises(self, loader: DocumentLoader, tmp_docs_dir: Path):
        """Should raise ValueError for unsupported file types."""
        bad_file = tmp_docs_dir / "file.docx"
        bad_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_document(bad_file)

    def test_save_uploaded_file(self, loader: DocumentLoader, tmp_docs_dir: Path):
        """Should save binary content and return the correct path."""
        content = b"This is uploaded content."
        saved_path = loader.save_uploaded_file("upload.txt", content)

        assert saved_path.exists()
        assert saved_path.read_bytes() == content

    def test_save_uploaded_file_sanitises_path(
        self, loader: DocumentLoader, tmp_docs_dir: Path
    ):
        """Path traversal in filename must be stripped."""
        saved = loader.save_uploaded_file("../../malicious.txt", b"data")
        # The file should land inside tmp_docs_dir, not outside it
        assert saved.parent == tmp_docs_dir

    def test_load_all_documents_empty_dir(self, loader: DocumentLoader):
        """Should return an empty list when no documents exist."""
        docs = loader.load_all_documents()
        assert docs == []

    def test_load_all_documents_multiple(
        self, loader: DocumentLoader, tmp_docs_dir: Path
    ):
        """Should load every .txt file in the directory."""
        (tmp_docs_dir / "a.txt").write_text("Document A content here.", encoding="utf-8")
        (tmp_docs_dir / "b.txt").write_text("Document B content here.", encoding="utf-8")

        docs = loader.load_all_documents()
        filenames = {d["filename"] for d in docs}

        assert len(docs) == 2
        assert "a.txt" in filenames
        assert "b.txt" in filenames


# ---------------------------------------------------------------------------
# TextChunker tests
# ---------------------------------------------------------------------------

class TestTextChunker:
    def test_basic_chunking(self, chunker: TextChunker):
        """Short text should produce at least one chunk."""
        text = "This is a simple sentence. " * 20
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "chunk_id" in chunk
            assert len(chunk["text"]) > 0

    def test_empty_text_returns_no_chunks(self, chunker: TextChunker):
        """Empty / whitespace-only text must return an empty list."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_metadata_propagated(self, chunker: TextChunker):
        """Metadata dict should appear in every chunk."""
        text = "Sentence one. Sentence two. Sentence three. " * 10
        meta = {"filename": "test.pdf", "file_type": ".pdf"}
        chunks = chunker.chunk_text(text, metadata=meta)

        for chunk in chunks:
            assert chunk["filename"] == "test.pdf"
            assert chunk["file_type"] == ".pdf"

    def test_chunk_ids_sequential(self, chunker: TextChunker):
        """chunk_id values should start at 0 and be sequential."""
        text = "Word. " * 500   # long enough to produce multiple chunks
        chunks = chunker.chunk_text(text)

        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    def test_overlap_gt_size_raises(self):
        """Constructor should reject overlap >= chunk_size."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_documents(self, chunker: TextChunker):
        """chunk_documents should flatten chunks from multiple docs."""
        docs = [
            {"text": "Doc one. " * 50, "filename": "one.txt", "file_type": ".txt", "file_path": ""},
            {"text": "Doc two. " * 50, "filename": "two.txt", "file_type": ".txt", "file_path": ""},
        ]
        all_chunks = chunker.chunk_documents(docs)

        assert len(all_chunks) >= 2
        filenames = {c["filename"] for c in all_chunks}
        assert "one.txt" in filenames
        assert "two.txt" in filenames


# ---------------------------------------------------------------------------
# TextPreprocessor tests
# ---------------------------------------------------------------------------

class TestTextPreprocessor:
    def test_removes_control_chars(self, preprocessor: TextPreprocessor):
        text = "Hello\x00\x01\x1f World"
        result = preprocessor.preprocess(text)
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

    def test_normalises_whitespace(self, preprocessor: TextPreprocessor):
        text = "Word1    Word2\t\tWord3\n\n\n\nWord4"
        result = preprocessor.preprocess(text)
        assert "  " not in result          # no double spaces
        assert "\t" not in result          # no tabs

    def test_removes_page_numbers(self, preprocessor: TextPreprocessor):
        text = "Introduction text.\nPage 1 of 10\nMore content here."
        result = preprocessor.preprocess(text)
        assert "Page 1 of 10" not in result

    def test_is_meaningful_true(self, preprocessor: TextPreprocessor):
        text = "This document describes the quarterly financial results for our company."
        assert preprocessor.is_meaningful(text) is True

    def test_is_meaningful_too_short(self, preprocessor: TextPreprocessor):
        assert preprocessor.is_meaningful("Hi.") is False

    def test_is_meaningful_no_words(self, preprocessor: TextPreprocessor):
        assert preprocessor.is_meaningful("123 456 789 000 111 222") is False

    def test_preprocess_batch(self, preprocessor: TextPreprocessor):
        texts = ["Hello\x00 World", "  extra   spaces  "]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 2
        assert "\x00" not in results[0]
        assert "extra   spaces" not in results[1]
