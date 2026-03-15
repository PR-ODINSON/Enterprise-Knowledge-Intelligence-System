"""
Conversation Memory Module
Maintains multi-turn conversation histories keyed by a session UUID.

Each conversation stores a capped list of (role, text) turns.
Older turns beyond MEMORY_MAX_TURNS are dropped to prevent prompt bloat.

Usage:
    memory = ConversationMemory()
    conv_id = memory.start_conversation()
    memory.add_turn(conv_id, "user", "What is RAG?")
    memory.add_turn(conv_id, "assistant", "RAG stands for ...")
    history = memory.get_history_text(conv_id)
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from app.config import MEMORY_MAX_TURNS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    """A single conversation turn."""
    role: str       # "user" or "assistant"
    text: str


class ConversationMemory:
    """
    In-memory store for multi-turn conversation histories.

    Thread-safety: not guaranteed for concurrent requests sharing the same
    conversation_id; acceptable for single-user local deployments.
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._sessions: Dict[str, Deque[Turn]] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_conversation(self) -> str:
        """
        Create a new conversation session.

        Returns:
            A new UUID string to use as the conversation_id.
        """
        conv_id = str(uuid.uuid4())
        self._sessions[conv_id] = deque(maxlen=self.max_turns)
        logger.debug(f"New conversation started: {conv_id}")
        return conv_id

    def end_conversation(self, conversation_id: str) -> None:
        """Remove a conversation from memory."""
        self._sessions.pop(conversation_id, None)
        logger.debug(f"Conversation ended: {conversation_id}")

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(self, conversation_id: str, role: str, text: str) -> None:
        """
        Append a turn to the specified conversation.

        If the conversation_id is unknown, a new session is created
        automatically (graceful handling of stale client IDs).

        Args:
            conversation_id: Session identifier.
            role:            ``"user"`` or ``"assistant"``.
            text:            Turn content.
        """
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = deque(maxlen=self.max_turns)

        self._sessions[conversation_id].append(Turn(role=role, text=text))
        logger.debug(
            f"Turn added [{conversation_id}] role={role} "
            f"({len(self._sessions[conversation_id])} turns total)"
        )

    def get_history(self, conversation_id: str) -> list:
        """Return the list of Turn objects for a conversation (may be empty)."""
        return list(self._sessions.get(conversation_id, []))

    def get_history_text(self, conversation_id: Optional[str]) -> str:
        """
        Format conversation history as a string for prompt injection.

        Args:
            conversation_id: Session identifier, or None for no history.

        Returns:
            Multi-line string of previous turns, or empty string if none.
        """
        if not conversation_id:
            return ""

        turns = self.get_history(conversation_id)
        if not turns:
            return ""

        lines = []
        for turn in turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.text}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def active_sessions(self) -> int:
        """Number of active conversation sessions."""
        return len(self._sessions)
