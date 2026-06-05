"""
app/memory/conversation.py — Conversation Memory Helpers

Thin wrapper around sessions_db for use in the workflow.
Keeps workflow.py clean by centralising session logic here.
"""

from typing import Optional
from app.db.sessions_db import (
    create_session,
    get_session,
    get_session_messages,
    append_message,
)


def get_or_create_session(user_id: str, session_id: Optional[str]) -> str:
    """
    Return an existing session_id or create a new one.

    If session_id is provided and belongs to a different user, a new
    session is created instead (security: prevent session hijacking).
    """
    if session_id:
        session = get_session(session_id)
        if session and session["user_id"] == user_id:
            return session_id
    # Create new session
    return create_session(user_id)


def load_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Load the last N messages from the session as LangChain-compatible dicts.

    Returns list of {"role": "user"|"assistant", "content": str}
    """
    return get_session_messages(session_id, limit=limit)


def save_exchange(session_id: str, user_message: str, assistant_message: str) -> None:
    """Persist both sides of an exchange to the session."""
    append_message(session_id, "user", user_message)
    append_message(session_id, "assistant", assistant_message)
