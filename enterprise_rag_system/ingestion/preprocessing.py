"""
Text Preprocessing Module
Cleans and normalises raw extracted text before it is chunked and embedded.

Pipeline steps applied by ``TextPreprocessor.preprocess()``:
  1. Unicode normalisation (NFC)
  2. Removal of non-printable / control characters
  3. Whitespace normalisation (collapse runs, limit consecutive newlines)
  4. Removal of common PDF boilerplate (page numbers, watermarks, etc.)
"""

import re
import unicodedata
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Stateless text-cleaning helper used between document loading and chunking.

    All public methods are idempotent and side-effect free — safe to call
    multiple times on the same text.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to *text*.

        Args:
            text: Raw extracted text from a document.

        Returns:
            Cleaned, normalised string ready for chunking.
        """
        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._normalize_whitespace(text)
        text = self._remove_boilerplate(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Apply the preprocessing pipeline to every string in *texts*.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned strings in the same order.
        """
        return [self.preprocess(t) for t in texts]

    def is_meaningful(self, text: str, min_length: int = 50) -> bool:
        """
        Determine whether *text* contains enough meaningful content to index.

        Returns ``False`` for texts that are too short or consist mainly of
        numbers and punctuation without real words.

        Args:
            text:       Text to evaluate (should already be preprocessed).
            min_length: Minimum character count required (default: 50).

        Returns:
            ``True`` if the text is worth indexing, ``False`` otherwise.
        """
        stripped = text.strip()

        if len(stripped) < min_length:
            logger.debug(
                f"Text rejected (too short): {len(stripped)} < {min_length} chars"
            )
            return False

        # Require at least 5 words of 2+ letters
        word_count = len(re.findall(r"\b[a-zA-Z]{2,}\b", stripped))
        if word_count < 5:
            logger.debug(
                f"Text rejected (too few words): {word_count} words found"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalise unicode to NFC (canonical composition)."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """
        Strip non-printable control characters.
        Preserves tab (\\t), newline (\\n), and carriage return (\\r).
        """
        # Allow: 0x09 tab, 0x0A newline, 0x0D carriage return
        # Remove: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalise whitespace:
          - Collapse horizontal whitespace (spaces/tabs) to a single space.
          - Limit consecutive blank lines to two newlines.
          - Strip leading/trailing whitespace.
        """
        text = re.sub(r"[ \t]+", " ", text)                 # collapse horizontal WS
        text = re.sub(r"\n{3,}", "\n\n", text)               # max 2 consecutive newlines
        return text.strip()

    @staticmethod
    def _remove_boilerplate(text: str) -> str:
        """
        Remove common PDF extraction artefacts:
          - "Page N of M" strings
          - Lines that contain only a page number
          - Null bytes and other PDF-specific junk characters
        """
        # "Page 3 of 15", "PAGE 3 OF 15", etc.
        text = re.sub(
            r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE
        )
        # Lines containing only a digit (standalone page numbers)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Residual null bytes from binary PDF streams
        text = text.replace("\x00", "")
        return text.strip()
