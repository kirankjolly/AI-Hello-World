"""
app/observability/logger.py — Structured Logging and Observability

[Concept: Observability]

────────────────────────────────────────────────────────────────
WHY OBSERVABILITY MATTERS IN AI SYSTEMS
────────────────────────────────────────────────────────────────
Traditional software: if a bug occurs, look at a stack trace.

AI systems have a different failure mode:
  - The code runs fine but the AI gives a WRONG ANSWER.
  - The LLM might hallucinate or retrieve irrelevant documents.
  - A guardrail might incorrectly block a valid question.

Without observability you cannot debug these problems.

What we log:
  1. Every incoming query
  2. What documents were retrieved (and their scores)
  3. The exact prompt sent to the LLM
  4. The LLM's raw response
  5. Which tools the agent used
  6. Rate limit hits
  7. Guardrail triggers

This creates a full "audit trail" for every AI interaction.
────────────────────────────────────────────────────────────────
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime


# ──────────────────────────────────────────────
# Logger Setup
# ──────────────────────────────────────────────

def setup_logger(name: str = "ai_assistant") -> logging.Logger:
    """Configure a structured logger that outputs JSON-formatted lines."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # Simple readable format — in production use JSON formatter + send to ELK/Datadog
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Single shared logger instance
logger = setup_logger("ai_assistant")


# ──────────────────────────────────────────────
# Observability Helper Functions
# ──────────────────────────────────────────────

def log_query(user_id: str, query: str, user_role: str) -> None:
    """Log an incoming user query."""
    logger.info(
        f"[QUERY] user={user_id} role={user_role} | query=\"{query[:100]}\""
    )


def log_retrieval(user_id: str, query: str, num_docs: int, scores: List[float]) -> None:
    """
    Log retrieval results.

    The scores tell us how relevant the retrieved documents are.
    Low scores (< 0.3) might mean the query has no matching documents.
    This helps debug cases where the LLM says 'I don't know'.
    """
    score_summary = [f"{s:.3f}" for s in scores[:5]]
    logger.info(
        f"[RETRIEVAL] user={user_id} | retrieved={num_docs} docs | "
        f"scores={score_summary}"
    )


def log_llm_call(user_id: str, model: str, prompt_tokens: int, context_length: int) -> None:
    """
    Log an LLM API call.

    Tracking token counts helps with:
      - Cost monitoring (you pay per token)
      - Debugging context length issues
      - Identifying when context is being truncated
    """
    logger.info(
        f"[LLM_CALL] user={user_id} | model={model} | "
        f"prompt_tokens≈{prompt_tokens} | context_chars={context_length}"
    )


def log_tool_use(user_id: str, tool_name: str, tool_input: str, tool_result: str) -> None:
    """
    Log agent tool usage.

    When an agent uses a tool, we log:
      - Which tool was chosen (tells us the agent's reasoning)
      - The input passed to the tool
      - The tool's output
    This is critical for debugging agent behavior.
    """
    logger.info(
        f"[TOOL_USE] user={user_id} | tool={tool_name} | "
        f"input=\"{tool_input[:80]}\" | result=\"{tool_result[:80]}\""
    )


def log_guardrail_trigger(user_id: str, reason: str, query: str) -> None:
    """
    Log when a guardrail blocks a query.

    This helps distinguish:
      - Legitimate safety blocks (prompt injection attempts)
      - False positives (valid queries blocked by mistake)
    High false-positive rates indicate guardrails need tuning.
    """
    logger.warning(
        f"[GUARDRAIL] user={user_id} | blocked_reason={reason} | "
        f"query=\"{query[:100]}\""
    )


def log_rate_limit_hit(user_id: str, request_count: int, limit: int) -> None:
    """Log when a user hits the rate limit."""
    logger.warning(
        f"[RATE_LIMIT] user={user_id} | count={request_count}/{limit} | BLOCKED"
    )


def log_permission_denied(user_id: str, user_role: str, doc_access_level: str) -> None:
    """Log when a user attempts to access a document above their permission level."""
    logger.warning(
        f"[PERMISSION] user={user_id} role={user_role} | "
        f"attempted_access={doc_access_level} | DENIED"
    )


def log_workflow_step(step_name: str, user_id: str, details: Optional[str] = None) -> None:
    """Log each step of the LangGraph workflow for tracing."""
    msg = f"[WORKFLOW] step={step_name} | user={user_id}"
    if details:
        msg += f" | {details}"
    logger.debug(msg)


def log_error(user_id: str, error: str, context: Optional[str] = None) -> None:
    """Log an error with context for debugging."""
    msg = f"[ERROR] user={user_id} | error={error}"
    if context:
        msg += f" | context={context}"
    logger.error(msg)
