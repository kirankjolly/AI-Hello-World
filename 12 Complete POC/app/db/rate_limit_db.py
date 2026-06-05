"""
app/db/rate_limit_db.py — Rate Limit SQLite Database

Stores: per-user request timestamps for sliding window rate limiting.

Future: Replace this module with Redis (ZADD/ZREMRANGEBYSCORE/ZCARD).
        The public API (init, connection) is the seam to swap.

Database file: data/rate_limit.db
"""

import sqlite3
from contextlib import contextmanager
from app.config import RATE_LIMIT_DB_PATH


@contextmanager
def get_rate_limit_db_connection():
    """WAL mode is essential here — this DB is hit on every single request."""
    conn = sqlite3.connect(RATE_LIMIT_DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_rate_limit_db() -> None:
    """Create the rate_limit_log table if it doesn't exist."""
    with get_rate_limit_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT    NOT NULL,
                timestamp REAL    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_user_ts "
            "ON rate_limit_log(user_id, timestamp)"
        )
