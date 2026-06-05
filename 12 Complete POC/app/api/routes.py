"""
app/api/routes.py — FastAPI Route Handlers

Routes:
  POST /login                          — Authenticate, get JWT token
  POST /ask                            — Query the AI (JWT required)
  POST /ingest_document                — Ingest text document (JWT required)
  POST /upload_document                — Upload PDF/TXT file (JWT required)
  GET  /documents                      — List all documents (JWT required)
  GET  /employees                      — List employees (hr/admin only)
  GET  /projects                       — List projects (JWT required)
  GET  /sessions                       — List user's chat sessions (JWT required)
  GET  /sessions/{session_id}/messages — Get session history (JWT required)
  GET  /health                         — Health check (public)
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Depends
from typing import Optional, List

from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    DocumentListResponse, DocumentSummary,
    Citation,
    LoginRequest, LoginResponse,
    EmployeeResponse, ProjectResponse,
    SessionResponse, MessageResponse,
    AccessLevel,
)
from app.rag.ingestion import ingest_text_content
from app.orchestration.workflow import run_workflow
from app.rate_limiting.limiter import check_rate_limit, get_remaining_requests
from app.vector_store.chroma_store import vector_store
from app.observability.logger import log_query, log_error, logger
from app.auth.jwt_handler import create_access_token
from app.auth.dependencies import get_current_user
from app.db.app_db import get_employee_by_email, list_employees, list_projects
from app.db.sessions_db import list_user_sessions, get_session_messages, get_session
from app.file_processing.processor import process_upload

router = APIRouter()


# ──────────────────────────────────────────────
# POST /login
# ──────────────────────────────────────────────

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate and receive a JWT token",
)
async def login(request: LoginRequest):
    """
    Verify credentials against the employees table and return a JWT.

    The returned token must be sent as:
        Authorization: Bearer <token>
    on all protected endpoints.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    employee = get_employee_by_email(request.email)
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not pwd_context.verify(request.password, employee["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not employee.get("is_active", 1):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    token = create_access_token({
        "user_id":     employee["employee_id"],
        "employee_id": employee["employee_id"],
        "email":       employee["email"],
        "name":        employee["name"],
        "role":        employee["role"],
        "department":  employee["department"],
    })

    return LoginResponse(
        access_token=token,
        user_id=employee["employee_id"],
        name=employee["name"],
        role=employee["role"],
        department=employee["department"],
    )


# ──────────────────────────────────────────────
# POST /ask
# ──────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question to the AI Knowledge Assistant",
    description=(
        "Submit a question. JWT required in Authorization header. "
        "Optionally pass session_id to continue a conversation. "
        "A new session_id is returned if not provided."
    )
)
async def ask_question(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    query   = request.query

    # ── Rate Limiting ──
    is_allowed, current_count, limit = check_rate_limit(user_id)
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error":     "Rate limit exceeded",
                "message":   f"You have exceeded {limit} requests per minute. Please wait.",
                "remaining": get_remaining_requests(user_id),
            },
            headers={"Retry-After": "60"},
        )

    log_query(user_id, query, current_user.get("role", "unknown"))

    try:
        final_state = run_workflow(
            query=query,
            user_id=user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        log_error(user_id, str(e), "ask_question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow error: {str(e)}"
        )

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
        session_id=final_state.get("session_id"),
        error=final_state.get("error"),
    )


# ──────────────────────────────────────────────
# POST /ingest_document
# ──────────────────────────────────────────────

@router.post(
    "/ingest_document",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document (raw text) into the knowledge base",
)
async def ingest_document(
    request: IngestRequest,
    current_user: dict = Depends(get_current_user),
):
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
        log_error(current_user["user_id"], str(e), "ingest_document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


# ──────────────────────────────────────────────
# POST /upload_document
# ──────────────────────────────────────────────

@router.post(
    "/upload_document",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF or TXT file into the knowledge base",
)
async def upload_document(
    file:         UploadFile = File(...),
    title:        str        = Form(...),
    department:   str        = Form(...),
    access_level: str        = Form(default="public"),
    current_user: dict       = Depends(get_current_user),
):
    """
    Upload a file (PDF or TXT). The content is extracted and ingested
    into ChromaDB exactly like /ingest_document.

    Set MOCK_FILE_PROCESSING=false in .env to enable real PDF parsing.
    """
    content = await process_upload(file)
    try:
        level = AccessLevel(access_level)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid access_level '{access_level}'. Choose: public, manager, confidential"
        )

    try:
        result = ingest_text_content(
            content=content,
            title=title,
            department=department,
            access_level=level,
        )
        logger.info(f"[API] File uploaded and ingested: {result['doc_id']} — '{title}'")
        return IngestResponse(
            doc_id=result["doc_id"],
            title=title,
            chunks_created=result["chunks_created"],
            message=result["message"],
        )
    except Exception as e:
        log_error(current_user["user_id"], str(e), "upload_document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest uploaded file: {str(e)}"
        )


# ──────────────────────────────────────────────
# GET /documents
# ──────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(current_user: dict = Depends(get_current_user)):
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
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        log_error("system", str(e), "list_documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


# ──────────────────────────────────────────────
# GET /employees  (HR and Admin only)
# ──────────────────────────────────────────────

@router.get(
    "/employees",
    response_model=List[EmployeeResponse],
    summary="List all employees (HR and Admin only)",
)
async def get_employees(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in ("admin", "hr"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Only HR and Admin roles can view the employee list.",
        )
    return [EmployeeResponse(**e) for e in list_employees()]


# ──────────────────────────────────────────────
# GET /projects
# ──────────────────────────────────────────────

@router.get(
    "/projects",
    response_model=List[ProjectResponse],
    summary="List all company projects",
)
async def get_projects(current_user: dict = Depends(get_current_user)):
    return [ProjectResponse(**p) for p in list_projects()]


# ──────────────────────────────────────────────
# GET /sessions
# ──────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=List[SessionResponse],
    summary="List user's chat sessions",
)
async def get_sessions(current_user: dict = Depends(get_current_user)):
    sessions = list_user_sessions(current_user["user_id"])
    return [
        SessionResponse(
            session_id=s["id"],
            title=s["title"],
            created_at=s["created_at"],
            updated_at=s["updated_at"],
        )
        for s in sessions
    ]


# ──────────────────────────────────────────────
# GET /sessions/{session_id}/messages
# ──────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/messages",
    response_model=List[MessageResponse],
    summary="Get messages for a specific session",
)
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    # Ownership check — return 404 to avoid leaking session existence
    session = get_session(session_id)
    if not session or session["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = get_session_messages(session_id, limit=100)
    return [MessageResponse(**m) for m in messages]


# ──────────────────────────────────────────────
# GET /health  (public)
# ──────────────────────────────────────────────

@router.get("/health", summary="Health check")
async def health_check():
    return {"status": "healthy", "service": "AI Knowledge Assistant", "version": "2.0.0"}
