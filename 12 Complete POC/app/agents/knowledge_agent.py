"""
app/agents/knowledge_agent.py — AI Agent

[Concept: Agent Architecture]

────────────────────────────────────────────────────────────────
WHAT IS AN AI AGENT?
────────────────────────────────────────────────────────────────
An AI Agent is an LLM that can decide WHAT TO DO next, not just
generate text. It has access to tools and decides which (if any)
to use based on the user's request.

AGENT vs. SIMPLE LLM:
  Simple LLM: Takes input → generates text → done
  Agent:       Takes input → thinks → decides action → executes
               → observes result → thinks again → answers

This project uses a CLASSIFIER AGENT pattern:
  1. Look at the query
  2. Classify: "Does this need a calculation, a summary, or a document search?"
  3. Route to the right handler

Why not just always use RAG?
  - "Calculate my bonus at 10%"  → RAG won't help; need a calculator
  - "What is the vacation policy?" → RAG is perfect
  - "Summarize the IT handbook"   → A summary tool is more efficient than RAG

THE REACT PATTERN (Reasoning + Acting):
  Thought:     Deliberate about the query
  Action:      Choose a tool or approach
  Observation: Receive the result
  Repeat until confident enough to give a final answer

This creates an auditable chain of reasoning — you can see exactly
why the agent chose a particular tool.
────────────────────────────────────────────────────────────────
"""

from typing import Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_MODEL, LLM_PROVIDER
from app.models.schemas import UserRole
from app.security.permissions import get_allowed_access_levels
from app.tools.company_tools import (
    ALL_TOOLS, TOOL_NAMES, POLICY_DATABASE, DOCUMENT_SUMMARIES,
    calculate_bonus, lookup_employee_policy, summarize_document, list_available_documents,
)
from app.observability.logger import log_tool_use, log_workflow_step, logger


# ──────────────────────────────────────────────
# Query Classification
# ──────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """You are a query router for a company knowledge assistant.

Given a user's question, decide which approach is best:

1. "rag"         — Search company documents (use for policy questions, procedures, general HR/IT/Finance questions)
2. "calculate"   — Use the bonus calculator (use when user explicitly asks to calculate a bonus amount with specific numbers)
3. "policy"      — Quick policy lookup (use for quick summaries of named policies like vacation, sick_leave, remote_work)
4. "summarize"   — Document summarizer (use when user says "summarize the [document name]")
5. "list"        — List available documents (use when user asks what documents/policies exist, or asks to summarize "everything" or "all documents")

Respond with ONLY ONE of these five words: rag, calculate, policy, summarize, list
No explanation needed."""


def _get_llm(max_tokens: int = 10):
    """Return LLM instance based on configured provider."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=max_tokens,
        )


def classify_query(query: str, user_id: str) -> str:
    """
    Use the LLM to classify what approach should handle this query.

    This is the "Thought" step of the ReAct loop.
    Returns one of: "rag", "calculate", "policy", "summarize"
    """
    log_workflow_step("classify_query", user_id, f"query='{query[:60]}'")

    llm = _get_llm(max_tokens=10)

    messages = [
        SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        classification = response.content.strip().lower()

        # Validate the response is one of our expected values
        valid_classes = {"rag", "calculate", "policy", "summarize", "list"}
        if classification not in valid_classes:
            logger.warning(f"[AGENT] Unexpected classification '{classification}', defaulting to 'rag'")
            classification = "rag"

        log_workflow_step("classify_query", user_id, f"classified_as={classification}")
        return classification

    except Exception as e:
        logger.error(f"[AGENT] Classification error: {e}")
        return "rag"  # Safe default: use document search


# ──────────────────────────────────────────────
# Tool Executors
# ──────────────────────────────────────────────

def execute_calculate_tool(query: str, user_id: str) -> Tuple[str, str]:
    """
    Handle calculation queries by extracting parameters and calling the tool.

    The agent uses the LLM to extract structured parameters from natural
    language, then calls the actual Python function.

    Returns: (tool_result, tool_name)
    """
    log_workflow_step("execute_tool", user_id, "tool=calculate_bonus")

    # Use LLM to extract salary and rate from natural language
    extractor_llm = _get_llm(max_tokens=50)

    extraction_prompt = """Extract the salary and bonus rate from this query.
Return ONLY JSON in this format: {"salary": 50000, "bonus_rate": 0.10}
If values are missing, use defaults: salary=50000, bonus_rate=0.10
Query: """ + query

    try:
        import json
        response = extractor_llm.invoke([HumanMessage(content=extraction_prompt)])
        # Extract JSON from response
        text = response.content.strip()
        # Find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            params = json.loads(text[start:end])
            salary = float(params.get("salary", 50000))
            bonus_rate = float(params.get("bonus_rate", 0.10))
        else:
            salary, bonus_rate = 50000.0, 0.10
    except Exception:
        salary, bonus_rate = 50000.0, 0.10

    # Execute the tool (this is the "Action" step)
    result = calculate_bonus.invoke({"salary": salary, "bonus_rate": bonus_rate})
    log_tool_use(user_id, "calculate_bonus", f"salary={salary}, rate={bonus_rate}", str(result)[:80])

    return str(result), "calculate_bonus"


def execute_policy_tool(query: str, user_id: str, user_role: Optional[UserRole] = None) -> Tuple[str, str]:
    """
    Extract the policy name from the query and look it up.

    Permission check: before calling the tool, verify the matched policy's
    access_level is within the user's allowed levels. This prevents a user
    with no/low role from receiving restricted policy content via the tool
    path (which bypasses the ChromaDB permission filter used in RAG).

    Returns: (tool_result, tool_name)
    """
    log_workflow_step("execute_tool", user_id, "tool=lookup_employee_policy")

    # Map common query keywords to policy names
    query_lower = query.lower()
    if "vacation" in query_lower or "holiday" in query_lower or "time off" in query_lower:
        policy_name = "vacation"
    elif "sick" in query_lower or "illness" in query_lower:
        policy_name = "sick_leave"
    elif "remote" in query_lower or "work from home" in query_lower or "wfh" in query_lower:
        policy_name = "remote_work"
    elif "parental" in query_lower or "maternity" in query_lower or "paternity" in query_lower:
        policy_name = "maternity_paternity"
    elif "performance" in query_lower or "review" in query_lower or "evaluation" in query_lower:
        policy_name = "performance_review"
    elif "compensation" in query_lower or "salary band" in query_lower or "pay band" in query_lower:
        policy_name = "compensation_bands"
    else:
        policy_name = "vacation"

    # ── Permission Check ──────────────────────────────────────────────
    # Resolve allowed access levels for this user's role.
    # If user_role is None (unauthenticated), treat as empty — no access.
    allowed_levels = get_allowed_access_levels(user_role) if user_role else []
    entry = POLICY_DATABASE.get(policy_name)
    if entry and entry["access_level"] not in allowed_levels:
        log_workflow_step(
            "permission_denied", user_id,
            f"policy='{policy_name}' requires '{entry['access_level']}', user has {allowed_levels}"
        )
        return (
            "You don't have permission to access this policy. "
            "Please contact your manager or HR for assistance."
        ), "lookup_employee_policy"
    # ─────────────────────────────────────────────────────────────────

    result = lookup_employee_policy.invoke({"policy_name": policy_name})
    log_tool_use(user_id, "lookup_employee_policy", policy_name, str(result)[:80])

    return str(result), "lookup_employee_policy"


def execute_summarize_tool(query: str, user_id: str, user_role: Optional[UserRole] = None) -> Tuple[str, str]:
    """
    Extract the document name from the query and return its summary.

    Permission check: verify the document's access_level against the user's
    role before invoking the tool. Prevents the tool path from leaking
    manager/confidential document summaries to lower-privileged users.

    Returns: (tool_result, tool_name)
    """
    log_workflow_step("execute_tool", user_id, "tool=summarize_document")

    query_lower = query.lower()
    if "hr" in query_lower or "handbook" in query_lower or "human resources" in query_lower:
        doc_name = "hr_handbook"
    elif "it" in query_lower or "security" in query_lower or "tech" in query_lower:
        doc_name = "it_security"
    elif "finance" in query_lower or "financial" in query_lower or "expense" in query_lower:
        doc_name = "finance_policy"
    elif "exec" in query_lower or "executive" in query_lower or "compensation" in query_lower:
        doc_name = "exec_compensation"
    else:
        doc_name = "hr_handbook"  # Default

    # ── Permission Check ──────────────────────────────────────────────
    allowed_levels = get_allowed_access_levels(user_role) if user_role else []
    entry = DOCUMENT_SUMMARIES.get(doc_name)
    if entry and entry["access_level"] not in allowed_levels:
        log_workflow_step(
            "permission_denied", user_id,
            f"document='{doc_name}' requires '{entry['access_level']}', user has {allowed_levels}"
        )
        return (
            "You don't have permission to access this document. "
            "Please contact your manager or HR for assistance."
        ), "summarize_document"
    # ─────────────────────────────────────────────────────────────────

    result = summarize_document.invoke({"document_name": doc_name})
    log_tool_use(user_id, "summarize_document", doc_name, str(result)[:80])

    return str(result), "summarize_document"


def execute_list_tool(user_id: str, user_role: Optional[UserRole] = None) -> Tuple[str, str]:
    """
    Call list_available_documents filtered to only entries the user can see.

    Permission filtering is applied HERE before invoking the tool — we pass
    allowed_levels so the tool only lists documents/policies within the user's
    access level. This prevents existence-leaking: an employee should not even
    know that confidential documents exist.

    Returns: (tool_result, tool_name)
    """
    log_workflow_step("execute_tool", user_id, "tool=list_available_documents")

    allowed_levels = get_allowed_access_levels(user_role) if user_role else []

    result = list_available_documents.invoke({"allowed_levels": allowed_levels})
    log_tool_use(user_id, "list_available_documents", "n/a", str(result)[:80])
    return str(result), "list_available_documents"


# ──────────────────────────────────────────────
# Main Agent Entry Point
# ──────────────────────────────────────────────

def run_agent(
    query: str,
    user_id: str,
    user_role: Optional[UserRole] = None,
) -> Tuple[str, Optional[str], bool]:
    """
    Main agent function: classify the query and route to the right handler.

    The agent decides:
      - Is this a document search question? → return "rag" signal
      - Is this a calculation? → execute tool, return result
      - Is this a policy summary? → execute tool, return result
      - Is this a document summary? → execute tool, return result

    Returns:
        (result_or_signal, tool_used_name, is_tool_answer)
        - If classification is "rag": returns ("rag", None, False)
          The workflow will handle RAG separately
        - If a tool was used: returns (tool_result, tool_name, True)
    """
    log_workflow_step("agent_start", user_id, f"query='{query[:60]}'")

    # Step 1: THOUGHT — classify what this query needs
    classification = classify_query(query, user_id)

    # Step 2: ACTION — execute based on classification
    if classification == "rag":
        # Signal to the workflow: handle this with RAG
        log_workflow_step("agent_decision", user_id, "routing to RAG pipeline")
        return "rag", None, False

    elif classification == "calculate":
        # ACTION: use calculator tool
        # OBSERVATION: get the calculation result
        result, tool_name = execute_calculate_tool(query, user_id)
        return result, tool_name, True

    elif classification == "policy":
        result, tool_name = execute_policy_tool(query, user_id, user_role)
        return result, tool_name, True

    elif classification == "summarize":
        result, tool_name = execute_summarize_tool(query, user_id, user_role)
        return result, tool_name, True

    elif classification == "list":
        result, tool_name = execute_list_tool(user_id, user_role)
        return result, tool_name, True

    else:
        # Unknown classification — fall back to RAG
        return "rag", None, False
