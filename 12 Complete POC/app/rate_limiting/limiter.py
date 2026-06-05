"""
app/rate_limiting/limiter.py — Rate Limiter (SQLite-backed)

This module delegates to sqlite_limiter.py.
To upgrade to Redis in production, replace sqlite_limiter.py with
a Redis implementation exposing the same three functions.
All imports of this module remain unchanged.
"""

# Re-export the SQLite implementation under the original names.
# Callers (routes.py etc.) import from here and are unaffected by the swap.
from app.rate_limiting.sqlite_limiter import (   # noqa: F401
    check_rate_limit,
    get_remaining_requests,
    reset_user_limit,
)
