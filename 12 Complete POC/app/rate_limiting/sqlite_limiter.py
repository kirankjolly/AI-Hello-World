"""
app/rate_limiting/sqlite_limiter.py — SQLite-backed Sliding Window Rate Limiter

Algorithm: Sliding Window Counter (same as the previous in-memory version)
  1. Delete all log entries older than the window
  2. Count remaining entries for this user
  3. If count >= limit → block
  4. Otherwise → insert a new entry and allow

─────────────────────────────────────────────────────────────────
PRODUCTION NOTE: Replace this module with Redis.

Redis equivalent (using Sorted Sets):
    r.zremrangebyscore(key, 0, now - window)
    count = r.zcard(key)
    if count < limit:
        r.zadd(key, {str(now): now})
        r.expire(key, window)
        return allowed
    return blocked

This module's public API is identical, making the swap a one-file change.
─────────────────────────────────────────────────────────────────
"""

import time
from typing import Tuple

from app.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
from app.db.rate_limit_db import get_rate_limit_db_connection
from app.observability.logger import log_rate_limit_hit


def check_rate_limit(user_id: str) -> Tuple[bool, int, int]:
    """
    Check whether a user is within their rate limit.

    Returns:
        (is_allowed, current_count, limit)
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    with get_rate_limit_db_connection() as conn:
        # Remove expired entries (sliding window cleanup)
        conn.execute(
            "DELETE FROM rate_limit_log WHERE user_id = ? AND timestamp < ?",
            (user_id, window_start)
        )

        # Count remaining requests in current window
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM rate_limit_log WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        current_count = row["cnt"]

        if current_count >= RATE_LIMIT_REQUESTS:
            log_rate_limit_hit(user_id, current_count, RATE_LIMIT_REQUESTS)
            return False, current_count, RATE_LIMIT_REQUESTS

        # Record this request
        conn.execute(
            "INSERT INTO rate_limit_log (user_id, timestamp) VALUES (?, ?)",
            (user_id, now)
        )

    return True, current_count + 1, RATE_LIMIT_REQUESTS


def get_remaining_requests(user_id: str) -> int:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    with get_rate_limit_db_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM rate_limit_log "
            "WHERE user_id = ? AND timestamp >= ?",
            (user_id, window_start)
        ).fetchone()
        return max(0, RATE_LIMIT_REQUESTS - row["cnt"])


def reset_user_limit(user_id: str) -> None:
    """Reset a user's rate limit counter (useful for testing)."""
    with get_rate_limit_db_connection() as conn:
        conn.execute("DELETE FROM rate_limit_log WHERE user_id = ?", (user_id,))
