"""
app/rate_limiting/limiter.py — In-Memory Rate Limiter

[Concept: API Rate Limiting]

────────────────────────────────────────────────────────────────
WHY RATE LIMITING IS CRITICAL FOR LLM APIS
────────────────────────────────────────────────────────────────
LLM APIs are expensive. A single GPT-4 or Claude API call can
cost $0.01–$0.06. Without rate limiting:

  - One user could accidentally (or maliciously) make thousands
    of requests, causing huge bills.
  - A bug in client code with a retry loop could bankrupt you.
  - Denial-of-service attacks become trivially cheap for attackers.

Production systems use Redis for distributed rate limiting
(works across multiple server instances). For this reference
project we use an in-memory dictionary — same logic, no Redis
dependency needed to run locally.

Algorithm used: Sliding Window Counter
  - Track timestamps of recent requests per user
  - Remove timestamps older than the window
  - If remaining count < limit → allow request
  - Otherwise → block request
────────────────────────────────────────────────────────────────
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple
from app.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
from app.observability.logger import log_rate_limit_hit


# ──────────────────────────────────────────────
# In-Memory Store
# ──────────────────────────────────────────────

# Maps user_id → list of request timestamps (Unix epoch seconds)
# In production this would be a Redis sorted set
_request_log: Dict[str, List[float]] = defaultdict(list)


# ──────────────────────────────────────────────
# Rate Limiter Logic
# ──────────────────────────────────────────────

def check_rate_limit(user_id: str) -> Tuple[bool, int, int]:
    """
    Check whether a user is within their rate limit.

    Algorithm:
      1. Get the current timestamp
      2. Remove all timestamps older than the window
      3. Count remaining (recent) requests
      4. If count < limit → allowed, record this request
      5. Otherwise → blocked

    Returns:
        (is_allowed, current_count, limit)
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Keep only timestamps within the current window (sliding window)
    recent_requests = [
        ts for ts in _request_log[user_id]
        if ts > window_start
    ]
    _request_log[user_id] = recent_requests

    current_count = len(recent_requests)

    if current_count >= RATE_LIMIT_REQUESTS:
        # User has exceeded the limit — log and block
        log_rate_limit_hit(user_id, current_count, RATE_LIMIT_REQUESTS)
        return False, current_count, RATE_LIMIT_REQUESTS

    # Record this request and allow it
    _request_log[user_id].append(now)
    return True, current_count + 1, RATE_LIMIT_REQUESTS


def get_remaining_requests(user_id: str) -> int:
    """Return how many more requests the user can make in this window."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    recent = [ts for ts in _request_log[user_id] if ts > window_start]
    return max(0, RATE_LIMIT_REQUESTS - len(recent))


def reset_user_limit(user_id: str) -> None:
    """Reset a user's rate limit counter (useful for testing)."""
    _request_log[user_id] = []
