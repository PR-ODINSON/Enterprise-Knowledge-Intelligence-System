"""
RAG Evaluation Dataset Builder
Collects recent (query, answer, chunks) triples from live traffic
and makes them available to the evaluator.

Operates as an in-memory ring buffer of the last MAX_SAMPLES interactions.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

MAX_SAMPLES: int = 50


@dataclass
class EvalSample:
    """One question-answer interaction with its retrieved context."""
    question: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    collection: str = "default"


class DatasetBuilder:
    """
    Collects evaluation samples from live RAG interactions.

    Thread-safety: not guaranteed; acceptable for single-user local use.
    """

    def __init__(self, max_samples: int = MAX_SAMPLES) -> None:
        self._buffer: Deque[EvalSample] = deque(maxlen=max_samples)

    def record(
        self,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        collection: str = "default",
    ) -> None:
        """
        Record a completed RAG interaction.

        Args:
            question:   The user's original question.
            answer:     The LLM-generated answer.
            chunks:     Retrieved chunk dicts (must contain ``"text"``).
            collection: The collection queried.
        """
        contexts = [c["text"] for c in chunks if c.get("text")]
        self._buffer.append(
            EvalSample(
                question=question,
                answer=answer,
                contexts=contexts,
                collection=collection,
            )
        )
        logger.debug(f"Eval sample recorded (buffer size: {len(self._buffer)})")

    def get_samples(self) -> List[EvalSample]:
        """Return all buffered evaluation samples."""
        return list(self._buffer)

    def clear(self) -> None:
        """Flush the buffer."""
        self._buffer.clear()

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the buffer."""
        return len(self._buffer)
