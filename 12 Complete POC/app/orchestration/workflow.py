"""
app/orchestration/workflow.py — LangGraph Workflow Orchestration

[Concept: Workflow Orchestration using LangGraph]

────────────────────────────────────────────────────────────────
WHY ORCHESTRATION IMPROVES RELIABILITY
────────────────────────────────────────────────────────────────
A "black-box agent" approach: give the LLM one big prompt and hope
it does everything correctly. Problems:
  - Non-deterministic (sometimes follows instructions, sometimes not)
  - Hard to debug (which step failed?)
  - No clear audit trail
  - Can't easily add middleware (rate limits, guardrails, logging)

LangGraph approach: define an EXPLICIT GRAPH of steps:
  - Each node is a clear Python function
  - Edges define allowed transitions
  - State is typed and explicit
  - Each step is logged separately
  - You can add conditional routing at any edge

This is like the difference between:
  "AI, do everything"  (black box)
  vs.
  "Step 1: validate → Step 2: retrieve → Step 3: generate" (explicit graph)

GRAPH STRUCTURE:
  START
    ↓
  validate_user         — Check user exists and get their role
    ↓
  apply_guardrails      — Check query safety (no injections, in domain)
    ↓
  classify_and_route    — Agent decides: RAG or Tool?
    ↓ (RAG path)          ↓ (Tool path)
  retrieve_documents    execute_tool
    ↓                     ↓
  build_context         ─────┘
    ↓
  generate_answer       — Call LLM with retrieved context
    ↓
  format_response       — Add citations, finalize
    ↓
  END
────────────────────────────────────────────────────────────────
"""

from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END, START

from app.models.schemas import UserRole, RetrievedChunk, Citation, QueryResponse
from app.security.permissions import validate_user, get_user_role
from app.security.guardrails import run_all_guardrails
from app.rag.retriever import retrieve_documents, generate_rag_answer, build_context
from app.agents.knowledge_agent import run_agent
from app.observability.logger import log_workflow_step, log_error, logger


# ──────────────────────────────────────────────
# Workflow State Definition
#
# TypedDict defines the shape of state that flows between nodes.
# LangGraph passes this state object to each node function.
# Each node can read from and write to the state.
# ──────────────────────────────────────────────

class WorkflowState(TypedDict):
    # Input
    query:          str
    user_id:        str

    # Permission Resolution
    user_role:      Optional[UserRole]
    is_valid_user:  bool

    # Guardrail Results
    passed_guardrails: bool
    guardrail_error:   str

    # Agent Decision
    use_tool:       bool
    tool_name:      Optional[str]
    tool_result:    Optional[str]

    # RAG Results
    retrieved_chunks:  List[Any]  # List[RetrievedChunk]
    context:           str

    # Final Answer
    answer:         str
    citations:      List[Any]  # List[Citation]
    error:          Optional[str]


# ──────────────────────────────────────────────
# Node Functions
# Each node receives the current state and returns updated fields
# ──────────────────────────────────────────────

def node_validate_user(state: WorkflowState) -> dict:
    """
    Node 1: Validate that the user exists and retrieve their role.

    Why this is a separate node:
      - Separates authentication concerns from business logic
      - If user validation fails, we short-circuit immediately
      - Easy to swap with real auth system (JWT validation, LDAP lookup...)
    """
    log_workflow_step("validate_user", state["user_id"])

    is_valid, role, error_msg = validate_user(state["user_id"])

    if not is_valid:
        log_error(state["user_id"], error_msg, "user_validation")
        return {
            "is_valid_user": False,
            "error": error_msg,
            "user_role": None,
        }

    return {
        "is_valid_user": True,
        "user_role": role,
        "error": None,
    }


def node_apply_guardrails(state: WorkflowState) -> dict:
    """
    Node 2: Run all safety guardrail checks on the query.

    Why guardrails are a separate node:
      - They run BEFORE any expensive LLM calls
      - Failed guardrail = immediate response, no tokens wasted
      - Easy to add/remove/tune guardrails independently
      - Clear log of why a query was blocked
    """
    log_workflow_step("apply_guardrails", state["user_id"])

    passed, error_msg = run_all_guardrails(state["query"], state["user_id"])

    return {
        "passed_guardrails": passed,
        "guardrail_error":   error_msg if not passed else "",
    }


def node_classify_and_route(state: WorkflowState) -> dict:
    """
    Node 3: Agent classifies the query and decides the approach.

    This is the "brain" of the system. The agent decides:
      - Should we search documents (RAG)?
      - Should we use a tool (calculator, policy lookup, summary)?

    Returns a signal to the router on which path to take.
    """
    log_workflow_step("classify_and_route", state["user_id"])

    result, tool_name, is_tool = run_agent(state["query"], state["user_id"])

    if is_tool:
        # Agent used a tool — we have a result already
        return {
            "use_tool":    True,
            "tool_name":   tool_name,
            "tool_result": result,
        }
    else:
        # Agent says: use RAG
        return {
            "use_tool":    False,
            "tool_name":   None,
            "tool_result": None,
        }


def node_retrieve_documents(state: WorkflowState) -> dict:
    """
    Node 4a (RAG path): Retrieve relevant document chunks from ChromaDB.

    Uses permission-aware retrieval — only returns documents
    the user is authorized to see.
    """
    log_workflow_step("retrieve_documents", state["user_id"])

    chunks = retrieve_documents(
        query=state["query"],
        user_role=state["user_role"],
        user_id=state["user_id"],
    )

    return {"retrieved_chunks": chunks}


def node_build_context(state: WorkflowState) -> dict:
    """
    Node 5 (RAG path): Format retrieved chunks into a context string.

    Why a separate node: Context building can be customized.
    Future enhancements: summarize long contexts, rerank chunks,
    filter by recency, etc.
    """
    log_workflow_step("build_context", state["user_id"])

    chunks = state.get("retrieved_chunks", [])
    context = build_context(chunks)

    return {"context": context}


def node_generate_answer(state: WorkflowState) -> dict:
    """
    Node 6: Generate the final answer.

    Two paths converge here:
      - RAG path: use retrieved documents as context
      - Tool path: tool result IS the answer (just needs formatting)
    """
    log_workflow_step("generate_answer", state["user_id"])

    if state.get("use_tool") and state.get("tool_result"):
        # Tool path: the tool already computed the answer
        # We still use the LLM to format the tool output as natural language
        answer = state["tool_result"]
        citations = []

    else:
        # RAG path: generate answer from retrieved documents
        chunks = state.get("retrieved_chunks", [])
        answer, citations = generate_rag_answer(
            query=state["query"],
            chunks=chunks,
            user_id=state["user_id"],
        )

    return {
        "answer":    answer,
        "citations": citations,
    }


def node_format_response(state: WorkflowState) -> dict:
    """
    Node 7: Final formatting and response assembly.

    This node is a good place to:
      - Add standard disclaimers
      - Format citations as footnotes
      - Check answer quality (post-generation guardrails)
      - Log the final response
    """
    log_workflow_step("format_response", state["user_id"])

    answer = state.get("answer", "")
    tool_name = state.get("tool_name")

    # If no documents were found and answer seems empty
    if not answer or answer.strip() == "":
        answer = "I don't have enough information in the available documents to answer this question."

    # Add citation footer if we have citations
    citations = state.get("citations", [])
    if citations and not state.get("use_tool"):
        doc_refs = ", ".join([f"'{c.title}'" for c in citations])
        answer += f"\n\n📎 Sources: {doc_refs}"

    logger.info(
        f"[WORKFLOW] Complete for user={state['user_id']} | "
        f"tool_used={tool_name or 'none'} | "
        f"citations={len(citations)} | "
        f"answer_length={len(answer)}"
    )

    return {"answer": answer}


# ──────────────────────────────────────────────
# Conditional Edge Routers
# These functions decide which node to go to next
# ──────────────────────────────────────────────

def route_after_validation(state: WorkflowState) -> str:
    """After user validation: proceed or end with error."""
    if not state.get("is_valid_user", False):
        return "end_with_error"
    return "apply_guardrails"


def route_after_guardrails(state: WorkflowState) -> str:
    """After guardrail checks: proceed or end with blocked message."""
    if not state.get("passed_guardrails", False):
        return "end_with_guardrail_block"
    return "classify_and_route"


def route_after_classification(state: WorkflowState) -> str:
    """After agent classification: go to tool path or RAG path."""
    if state.get("use_tool", False):
        return "generate_answer"    # Tool already ran, skip retrieval
    return "retrieve_documents"     # Use RAG pipeline


# ──────────────────────────────────────────────
# Terminal Error Nodes
# ──────────────────────────────────────────────

def node_end_with_error(state: WorkflowState) -> dict:
    """Terminal node for user validation failures."""
    return {
        "answer":    f"Error: {state.get('error', 'Unknown error')}",
        "citations": [],
    }


def node_end_with_guardrail_block(state: WorkflowState) -> dict:
    """Terminal node for guardrail blocks."""
    return {
        "answer":    state.get("guardrail_error", "Query blocked by safety guardrails."),
        "citations": [],
        "error":     "guardrail_block",
    }


# ──────────────────────────────────────────────
# Graph Assembly
# ──────────────────────────────────────────────

def build_workflow() -> Any:
    """
    Assemble the LangGraph StateGraph.

    LangGraph represents the workflow as a directed graph:
      - Nodes are Python functions that transform state
      - Edges define allowed transitions
      - Conditional edges use functions to decide which path to take
      - START and END are special built-in nodes

    Why StateGraph?
      - State is typed and explicit (TypedDict)
      - Each node only gets/sets what it needs
      - Makes the data flow transparent and debuggable
    """
    # Create the graph with our state type
    graph = StateGraph(WorkflowState)

    # ── Add all nodes ──
    graph.add_node("validate_user",           node_validate_user)
    graph.add_node("apply_guardrails",         node_apply_guardrails)
    graph.add_node("classify_and_route",       node_classify_and_route)
    graph.add_node("retrieve_documents",       node_retrieve_documents)
    graph.add_node("build_context",            node_build_context)
    graph.add_node("generate_answer",          node_generate_answer)
    graph.add_node("format_response",          node_format_response)
    graph.add_node("end_with_error",           node_end_with_error)
    graph.add_node("end_with_guardrail_block", node_end_with_guardrail_block)

    # ── Add edges ──

    # Entry point: START → validate user
    graph.add_edge(START, "validate_user")

    # Conditional: after validation → proceed or error
    graph.add_conditional_edges(
        "validate_user",
        route_after_validation,
        {
            "apply_guardrails": "apply_guardrails",
            "end_with_error":   "end_with_error",
        }
    )

    # Conditional: after guardrails → proceed or block
    graph.add_conditional_edges(
        "apply_guardrails",
        route_after_guardrails,
        {
            "classify_and_route":        "classify_and_route",
            "end_with_guardrail_block":  "end_with_guardrail_block",
        }
    )

    # Conditional: after classification → RAG path or Tool path
    graph.add_conditional_edges(
        "classify_and_route",
        route_after_classification,
        {
            "retrieve_documents": "retrieve_documents",   # RAG path
            "generate_answer":    "generate_answer",      # Tool path (skip retrieval)
        }
    )

    # RAG path: retrieve → build context → generate
    graph.add_edge("retrieve_documents", "build_context")
    graph.add_edge("build_context",      "generate_answer")

    # Both paths converge at generate_answer → format → END
    graph.add_edge("generate_answer",   "format_response")
    graph.add_edge("format_response",   END)

    # Error terminals → END
    graph.add_edge("end_with_error",           END)
    graph.add_edge("end_with_guardrail_block", END)

    # Compile the graph into a runnable
    return graph.compile()


# ──────────────────────────────────────────────
# Compiled Workflow (singleton)
# ──────────────────────────────────────────────

# Build the graph once at startup
workflow = build_workflow()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_workflow(query: str, user_id: str) -> WorkflowState:
    """
    Execute the complete workflow for a user query.

    This is the main entry point called by the FastAPI routes.
    It runs the full LangGraph: validation → guardrails → agent
    → retrieval → generation → formatting.

    Args:
        query:   The user's question
        user_id: The authenticated user's ID

    Returns:
        Final WorkflowState with answer, citations, and metadata
    """
    initial_state: WorkflowState = {
        "query":              query,
        "user_id":            user_id,
        "user_role":          None,
        "is_valid_user":      False,
        "passed_guardrails":  False,
        "guardrail_error":    "",
        "use_tool":           False,
        "tool_name":          None,
        "tool_result":        None,
        "retrieved_chunks":   [],
        "context":            "",
        "answer":             "",
        "citations":          [],
        "error":              None,
    }

    logger.info(f"[WORKFLOW] Starting for user={user_id} | query='{query[:60]}'")

    try:
        final_state = workflow.invoke(initial_state)
        return final_state
    except Exception as e:
        log_error(user_id, str(e), "workflow_execution")
        initial_state["answer"] = f"An internal error occurred: {str(e)}"
        initial_state["error"]  = str(e)
        return initial_state
