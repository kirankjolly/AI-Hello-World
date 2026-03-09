"""
app/api/routes.py — FastAPI Route Handlers

[Concept: Production-Style Backend]

────────────────────────────────────────────────────────────────
HOW THE API CONNECTS TO THE ORCHESTRATION LAYER
────────────────────────────────────────────────────────────────
The FastAPI routes are thin — they do three things:
  1. Validate the request (Pydantic does this automatically)
  2. Check rate limits
  3. Call the LangGraph workflow
  4. Format and return the response

ALL the business logic lives in the workflow and its nodes.
The API layer is intentionally simple.

Why separate API from workflow?
  - You can call the workflow from tests, CLI scripts, or other
    services without going through HTTP.
  - Easier to add new protocols (gRPC, WebSocket) without
    changing the core logic.
  - Clear separation of concerns.
────────────────────────────────────────────────────────────────
"""

from fastapi import APIRouter, HTTPException, status, Header
from typing import Optional

from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    DocumentListResponse, DocumentSummary,
    Citation,
)
from app.rag.ingestion import ingest_text_content
from app.orchestration.workflow import run_workflow
from app.rate_limiting.limiter import check_rate_limit, get_remaining_requests
from app.vector_store.chroma_store import vector_store
from app.security.permissions import get_user
from app.observability.logger import log_query, log_error, logger


# ──────────────────────────────────────────────
# Router Setup
# ──────────────────────────────────────────────

router = APIRouter()


# ──────────────────────────────────────────────
# POST /ingest_document
# ──────────────────────────────────────────────

@router.post(
    "/ingest_document",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into the knowledge base",
    description=(
        "Upload a document (as raw text) with metadata. "
        "The document will be split into chunks, embedded, and stored "
        "in ChromaDB for later retrieval."
    )
)
async def ingest_document(request: IngestRequest):
    """
    [Section: RAG - Document Ingestion]

    Accepts a document as text and ingests it into the vector database.

    The pipeline:
      1. Receive content + metadata via HTTP POST
      2. Split into chunks
      3. Generate embeddings via OpenAI
      4. Store in ChromaDB with metadata (access_level, department, etc.)

    Access levels:
      - public:       all employees can see this
      - manager:      managers and admins only
      - confidential: admins only
    """
    try:
        result = ingest_text_content(
            content=request.content,
            title=request.title,
            department=request.department,
            access_level=request.access_level,
        )

        logger.info(f"[API] Document ingested: {result['doc_id']} — '{request.title}'")

        return IngestResponse(
            doc_id=result["doc_id"],
            title=request.title,
            chunks_created=result["chunks_created"],
            message=result["message"],
        )

    except Exception as e:
        log_error("system", str(e), "ingest_document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


# ──────────────────────────────────────────────
# POST /ask
# ──────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question to the AI Knowledge Assistant",
    description=(
        "Submit a question and user ID. The system will: "
        "1) validate the user, 2) apply guardrails, 3) retrieve "
        "relevant documents (based on permissions), 4) generate a "
        "grounded answer with citations."
    )
)
async def ask_question(request: QueryRequest):
    """
    [Section: Full Pipeline — RAG + Agent + Guardrails + Permissions]

    Main query endpoint. Routes through the full LangGraph workflow:
      validate_user → guardrails → classify → retrieve/tool → answer

    Rate limiting is applied per user_id.
    """
    user_id = request.user_id
    query   = request.query

    # ── Rate Limiting ──
    # Check before doing any expensive operations
    is_allowed, current_count, limit = check_rate_limit(user_id)
    if not is_allowed:
        remaining_requests = get_remaining_requests(user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error":     "Rate limit exceeded",
                "message":   f"You have exceeded {limit} requests per minute. Please wait.",
                "remaining": remaining_requests,
            },
            headers={"Retry-After": "60"},
        )

    # ── Log incoming query ──
    user = get_user(user_id)
    user_role_str = user["role"].value if user else "unknown"
    log_query(user_id, query, user_role_str)

    # ── Run the LangGraph Workflow ──
    try:
        final_state = run_workflow(query=query, user_id=user_id)
    except Exception as e:
        log_error(user_id, str(e), "ask_question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow error: {str(e)}"
        )

    # ── Build the response ──
    # Citations come from the workflow state
    citations = []
    for cit in final_state.get("citations", []):
        if isinstance(cit, Citation):
            citations.append(cit)
        elif isinstance(cit, dict):
            citations.append(Citation(**cit))

    return QueryResponse(
        query=query,
        answer=final_state.get("answer", "No answer generated."),
        citations=citations,
        used_tool=final_state.get("tool_name"),
        is_from_docs=not final_state.get("use_tool", False),
        user_id=user_id,
        error=final_state.get("error"),
    )


# ──────────────────────────────────────────────
# GET /documents
# ──────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    description="Returns a list of all documents stored in the vector database with their metadata."
)
async def list_documents():
    """
    [Section: Vector Database]

    Returns a summary of all documents currently in ChromaDB.
    Groups chunks by document ID to show one entry per document.
    """
    try:
        raw_docs = vector_store.list_documents()

        documents = [
            DocumentSummary(
                doc_id=d["doc_id"],
                title=d["title"],
                department=d["department"],
                access_level=d["access_level"],
                chunk_count=d["chunk_count"],
            )
            for d in raw_docs
        ]

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    except Exception as e:
        log_error("system", str(e), "list_documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@router.get(
    "/health",
    summary="Health check endpoint",
)
async def health_check():
    """Simple health check — confirms the service is running."""
    return {
        "status":  "healthy",
        "service": "AI Knowledge Assistant",
        "version": "1.0.0",
    }


# ──────────────────────────────────────────────
# GET /users
# ──────────────────────────────────────────────

@router.get(
    "/users",
    summary="List available test users",
    description="Returns the list of predefined users for testing. In production, this endpoint would not exist."
)
async def list_users():
    """
    Convenience endpoint for the demo — shows available test user IDs.
    In a production system, user management would be in a separate auth service.
    """
    from app.security.permissions import USERS
    return {
        "users": [
            {
                "user_id": uid,
                "name":    info["name"],
                "role":    info["role"].value,
            }
            for uid, info in USERS.items()
        ]
    }
