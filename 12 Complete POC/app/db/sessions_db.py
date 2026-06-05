"""
app/db/sessions_db.py — Conversation Sessions SQLite Database

Stores: chat sessions and messages (ChatGPT-style history).

Future: Replace with PostgreSQL or Redis by swapping get_sessions_db_connection()
        and the SQL in init_sessions_db().

Database file: data/sessions.db
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from app.config import SESSIONS_DB_PATH


@contextmanager
def get_sessions_db_connection():
    conn = sqlite3.connect(SESSIONS_DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_sessions_db() -> None:
    with get_sessions_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,            -- UUID string
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT    NOT NULL,            -- 'user' | 'assistant'
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON sessions(user_id, updated_at);
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(user_id: str, title: str = "New Chat") -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    now = _now()
    with get_sessions_db_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_id, user_id, title, now, now)
        )
    return session_id


def get_session(session_id: str) -> dict | None:
    with get_sessions_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def get_session_messages(session_id: str, limit: int = 10) -> list[dict]:
    """Return the last `limit` messages ordered oldest-first."""
    with get_sessions_db_connection() as conn:
        rows = conn.execute("""
            SELECT role, content, created_at FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]


def append_message(session_id: str, role: str, content: str) -> None:
    now = _now()
    with get_sessions_db_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )


def list_user_sessions(user_id: str) -> list[dict]:
    with get_sessions_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
