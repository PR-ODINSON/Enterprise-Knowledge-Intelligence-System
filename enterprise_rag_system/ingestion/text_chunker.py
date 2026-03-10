"""
Text Chunker Module
Splits a document's raw text into overlapping, token-aware chunks.

Design decisions:
  - Splits on natural sentence boundaries first for more coherent chunks.
  - Very long sentences are further split at clause boundaries.
  - Overlap is implemented by retaining the last N characters worth of
    sentences from the preceding chunk as a sliding window.
  - 1 token ≈ 4 characters (standard English approximation).
"""

import re
from typing import Any, Dict, List

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)

# Approximate number of characters per token for English text
_CHARS_PER_TOKEN: int = 4


class TextChunker:
    """
    Splits text into overlapping chunks suitable for dense embedding models.

    Attributes:
        chunk_size:         Target chunk size expressed in *tokens*.
        chunk_overlap:      Number of *token*-equivalent characters to overlap
                            between consecutive chunks.
        chunk_size_chars:   Derived character-count limit per chunk.
        chunk_overlap_chars: Derived character-count for the overlap window.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_size_chars = chunk_size * _CHARS_PER_TOKEN
        self.chunk_overlap_chars = chunk_overlap * _CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Split a single text string into overlapping chunks.

        Args:
            text:     Raw text to be chunked.
            metadata: Optional key/value metadata merged into every chunk dict.

        Returns:
            List of chunk dictionaries. Each dict contains:
            ``text``, ``chunk_id``, ``char_count``, ``approx_tokens``,
            plus any fields from *metadata*.
        """
        if not text or not text.strip():
            logger.warning("chunk_text received empty text; returning no chunks")
            return []

        metadata = metadata or {}
        cleaned = _normalize_whitespace(text)
        sentences = _split_sentences(cleaned, self.chunk_size_chars)

        chunks: List[Dict[str, Any]] = []
        current_sentences: List[str] = []
        current_len: int = 0
        chunk_id: int = 0

        for sentence in sentences:
            slen = len(sentence)

            # Flush current buffer when adding this sentence would overflow
            if current_len + slen > self.chunk_size_chars and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(self._make_chunk(chunk_text, chunk_id, metadata))
                    chunk_id += 1

                # Build overlap window from the tail of the flushed buffer
                overlap_sents: List[str] = []
                overlap_len = 0
                for sent in reversed(current_sentences):
                    if overlap_len + len(sent) <= self.chunk_overlap_chars:
                        overlap_sents.insert(0, sent)
                        overlap_len += len(sent)
                    else:
                        break

                current_sentences = overlap_sents
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += slen

        # Emit the final (possibly partial) chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, chunk_id, metadata))

        logger.debug(
            f"chunk_text → {len(chunks)} chunks from {len(text):,} chars"
        )
        return chunks

    def chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of document dicts and return all chunks.

        Each document dict must contain at least a ``"text"`` key.
        Metadata fields (``filename``, ``file_type``, ``file_path``) are
        automatically forwarded to every chunk produced from that document.

        Args:
            documents: List of document dictionaries from DocumentLoader.

        Returns:
            Flat list of chunk dictionaries from all documents combined.
        """
        all_chunks: List[Dict[str, Any]] = []

        for doc in documents:
            doc_meta = {
                "filename": doc.get("filename", "unknown"),
                "file_type": doc.get("file_type", "unknown"),
                "file_path": doc.get("file_path", ""),
            }
            doc_chunks = self.chunk_text(doc["text"], metadata=doc_meta)
            all_chunks.extend(doc_chunks)
            logger.info(
                f"Chunked '{doc_meta['filename']}' → {len(doc_chunks)} chunks"
            )

        logger.info(f"Total chunks created from {len(documents)} document(s): {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk(
        text: str, chunk_id: int, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble a chunk dictionary."""
        return {
            "text": text,
            "chunk_id": chunk_id,
            "char_count": len(text),
            "approx_tokens": len(text) // _CHARS_PER_TOKEN,
            **metadata,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str, max_sentence_chars: int) -> List[str]:
    """
    Split text into sentence-like units using punctuation heuristics.

    Sentences longer than *max_sentence_chars* are further split at clause
    boundaries (semicolons or commas) to prevent single sentences from
    exceeding a whole chunk.
    """
    # Split on ". ", "! ", "? " when followed by a capital letter or digit
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    result: List[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_sentence_chars:
            # Split on clause boundaries for very long sentences
            sub = [s.strip() for s in re.split(r"[;,]\s+", sentence) if s.strip()]
            result.extend(sub)
        else:
            result.append(sentence)

    return result
