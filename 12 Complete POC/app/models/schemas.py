"""
app/models/schemas.py — Pydantic Data Models

[Concept: API Design + Data Validation]

Pydantic models define the shape of data flowing through the API.
They provide:
  - Automatic request/response validation
  - Type checking at runtime
  - Self-documenting API (FastAPI generates OpenAPI from these)
  - Clear contracts between layers of the system
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# ──────────────────────────────────────────────
# Permission Model
# ──────────────────────────────────────────────

class UserRole(str, Enum):
    """User roles that control document access."""
    EMPLOYEE  = "employee"   # Access: public documents only
    MANAGER   = "manager"    # Access: public + manager documents
    ADMIN     = "admin"      # Access: all documents (including confidential)
    HR        = "hr"         # Access: public + manager documents (HR staff)


class AccessLevel(str, Enum):
    """Document access levels — matches against user roles."""
    PUBLIC       = "public"       # All employees can read
    MANAGER      = "manager"      # Managers and above
    CONFIDENTIAL = "confidential" # Admins only


# ──────────────────────────────────────────────
# Document Models
# ──────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata stored alongside each document chunk in the vector DB."""
    doc_id:       str
    title:        str
    department:   str
    access_level: AccessLevel
    source:       Optional[str] = None   # file path or URL


class IngestRequest(BaseModel):
    """Request body for POST /ingest_document"""
    title:        str         = Field(..., description="Document title")
    department:   str         = Field(..., description="Owning department (HR, Finance, IT...)")
    access_level: AccessLevel = Field(default=AccessLevel.PUBLIC, description="Who can read this doc")
    content:      str         = Field(..., description="Raw text content of the document")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Vacation Policy 2024",
                "department": "HR",
                "access_level": "public",
                "content": "All full-time employees receive 15 vacation days per year..."
            }
        }


class IngestResponse(BaseModel):
    """Response body for POST /ingest_document"""
    doc_id:      str
    title:       str
    chunks_created: int
    message:     str


# ──────────────────────────────────────────────
# Query / Answer Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /ask"""
    query:      str            = Field(..., description="The user's question")
    user_id:    Optional[str]  = Field(None, description="Deprecated — user identity comes from JWT token")
    session_id: Optional[str]  = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many vacation days do employees get?",
                "session_id": None
            }
        }


class Citation(BaseModel):
    """A single document citation included in the answer."""
    doc_id:     str
    title:      str
    department: str
    snippet:    str   # The relevant text chunk used


class QueryResponse(BaseModel):
    """Response body for POST /ask"""
    query:         str
    answer:        str
    citations:     List[Citation]
    used_tool:     Optional[str]  = None   # name of tool if agent used one
    is_from_docs:  bool           = True   # False if answer came from a tool
    user_id:       str
    session_id:    Optional[str]  = None
    error:         Optional[str]  = None


# ──────────────────────────────────────────────
# Document Listing Model
# ──────────────────────────────────────────────

class DocumentSummary(BaseModel):
    """Summary of an ingested document returned by GET /documents"""
    doc_id:       str
    title:        str
    department:   str
    access_level: AccessLevel
    chunk_count:  int


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
    total:     int


# ──────────────────────────────────────────────
# Internal Workflow State
# ──────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single document chunk retrieved from the vector DB."""
    doc_id:       str
    title:        str
    department:   str
    access_level: str
    content:      str
    score:        float = 0.0   # Cosine similarity score (higher = more relevant)


# ──────────────────────────────────────────────
# Auth Models
# ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str = Field(..., description="Employee email address")
    password: str = Field(..., description="Employee password")

    class Config:
        json_schema_extra = {
            "example": {"email": "alice@company.com", "password": "password123"}
        }


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    name:         str
    role:         str
    department:   str


# ──────────────────────────────────────────────
# Employee & Project Models
# ──────────────────────────────────────────────

class EmployeeResponse(BaseModel):
    id:           int
    employee_id:  str
    name:         str
    email:        str
    department:   str
    role:         str
    joining_date: str
    is_active:    int


class ProjectResponse(BaseModel):
    id:                    int
    name:                  str
    start_date:            str
    end_date:              Optional[str]
    status:                str
    assignee_name:         Optional[str]
    assignee_employee_id:  Optional[str]


# ──────────────────────────────────────────────
# Session & Message Models
# ──────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    title:      str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    role:       str
    content:    str
    created_at: str
