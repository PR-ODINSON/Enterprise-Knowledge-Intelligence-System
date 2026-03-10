"""
Logger Utility Module
Provides a consistent, reusable logging configuration across all application modules.
Each module calls get_logger(__name__) to obtain a named logger.

Logs are written to both the console (stdout) and a rotating file at logs/app.log.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

# Rotate after 10 MB, keep 5 backup files
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger configured with console and rotating file handlers.

    If a logger with the given name already has handlers attached (e.g., on
    module re-import), the existing logger is returned unchanged to avoid
    duplicate log entries.

    Args:
        name:  Logger name — pass __name__ from the calling module.
        level: Minimum logging level (default: INFO).

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Guard against duplicate handlers on repeated imports
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console handler ------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # --- Rotating file handler ------------------------------------------
    try:
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)   # Capture verbose detail in file
        logger.addHandler(file_handler)
    except OSError as exc:
        # Non-fatal: file logging may be unavailable in certain environments
        logger.warning(f"Could not create file log handler: {exc}")

    # Prevent propagation to the root logger (avoids duplicate entries)
    logger.propagate = False

    return logger
