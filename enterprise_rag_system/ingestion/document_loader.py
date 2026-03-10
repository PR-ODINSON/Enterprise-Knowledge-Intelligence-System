"""
Document Loader Module
Responsible for reading PDF and TXT files from disk and extracting their
raw text content. Provides both single-file and bulk loading, plus an
upload helper for saving files received through the API.

Supported formats:
  - PDF  — extracted via pdfplumber (preferred) with pypdf as fallback
  - TXT  — decoded using UTF-8, with latin-1 / cp1252 fallbacks
"""

from pathlib import Path
from typing import Dict, List

import pdfplumber
from pypdf import PdfReader

from app.config import DOCUMENTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt"})


class DocumentLoader:
    """
    Loads documents from the filesystem and returns structured dictionaries
    containing the extracted text and associated metadata.
    """

    def __init__(self, documents_dir: Path = DOCUMENTS_DIR) -> None:
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_document(self, file_path: Path) -> Dict[str, str]:
        """
        Load a single document and extract its text.

        Args:
            file_path: Absolute path to the target file.

        Returns:
            Dictionary with keys: ``text``, ``filename``, ``file_path``,
            ``file_type``, and ``char_count``.

        Raises:
            ValueError: If the file extension is not supported.
            RuntimeError: If text extraction fails for both parsers (PDF).
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        logger.info(f"Loading document: {file_path.name}")

        text = (
            self._extract_pdf_text(file_path)
            if ext == ".pdf"
            else self._extract_txt_text(file_path)
        )

        return {
            "text": text,
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_type": ext,
            "char_count": len(text),
        }

    def load_all_documents(self) -> List[Dict[str, str]]:
        """
        Scan the documents directory and load every supported file.

        Returns:
            List of document dictionaries (same structure as ``load_document``).
            Files that fail to parse are logged and skipped.
        """
        documents: List[Dict[str, str]] = []

        candidate_files = [
            p
            for p in self.documents_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not candidate_files:
            logger.warning(f"No supported documents found in {self.documents_dir}")
            return documents

        for file_path in candidate_files:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                logger.info(f"Loaded: {file_path.name} ({doc['char_count']:,} chars)")
            except Exception as exc:
                logger.error(f"Failed to load '{file_path.name}': {exc}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def save_uploaded_file(self, filename: str, content: bytes) -> Path:
        """
        Persist an uploaded file to the documents directory.

        The filename is sanitised (``Path(filename).name``) to prevent
        path-traversal attacks.

        Args:
            filename: Original filename supplied by the uploader.
            content:  Raw binary content of the file.

        Returns:
            Path to the saved file.
        """
        # Strip any directory components to prevent path traversal
        safe_name = Path(filename).name
        dest = self.documents_dir / safe_name

        with open(dest, "wb") as fh:
            fh.write(content)

        logger.info(f"Saved uploaded file: {safe_name} ({len(content):,} bytes)")
        return dest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pdf_text(self, file_path: Path) -> str:
        """
        Extract text from a PDF using pdfplumber.
        Falls back to pypdf if pdfplumber fails or returns no text.
        """
        text = self._try_pdfplumber(file_path)

        if not text.strip():
            logger.debug(f"pdfplumber returned empty text for {file_path.name}; trying pypdf")
            text = self._try_pypdf(file_path)

        if not text.strip():
            logger.warning(
                f"No text extracted from '{file_path.name}'. "
                "The PDF may be scanned/image-based."
            )

        return text

    def _try_pdfplumber(self, file_path: Path) -> str:
        """Attempt text extraction with pdfplumber."""
        try:
            pages_text: List[str] = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)

            extracted = "\n".join(pages_text)
            logger.debug(
                f"pdfplumber extracted {len(extracted):,} chars from {file_path.name}"
            )
            return extracted
        except Exception as exc:
            logger.warning(f"pdfplumber error on '{file_path.name}': {exc}")
            return ""

    def _try_pypdf(self, file_path: Path) -> str:
        """Attempt text extraction with pypdf as a fallback."""
        try:
            reader = PdfReader(str(file_path))
            pages_text = [
                page.extract_text() or "" for page in reader.pages
            ]
            extracted = "\n".join(pages_text)
            logger.debug(
                f"pypdf extracted {len(extracted):,} chars from {file_path.name}"
            )
            return extracted
        except Exception as exc:
            logger.error(f"pypdf error on '{file_path.name}': {exc}")
            raise RuntimeError(
                f"Unable to extract text from '{file_path.name}' "
                f"using pdfplumber or pypdf."
            ) from exc

    def _extract_txt_text(self, file_path: Path) -> str:
        """
        Read a plain-text file, trying multiple encodings in order.
        Encodings tried: utf-8, latin-1, cp1252.
        """
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text = file_path.read_text(encoding=encoding)
                logger.debug(
                    f"Read {len(text):,} chars from '{file_path.name}' "
                    f"(encoding: {encoding})"
                )
                return text
            except UnicodeDecodeError:
                continue

        raise ValueError(
            f"Could not decode '{file_path.name}' with utf-8, latin-1, or cp1252."
        )
