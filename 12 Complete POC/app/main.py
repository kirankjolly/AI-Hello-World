"""
app/main.py — FastAPI Application Entry Point

[Concept: Production-Style Backend]

────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW
────────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────┐
  │                       CLIENT                             │
  │              (curl, browser, Postman)                    │
  └─────────────────────┬────────────────────────────────────┘
                        │ HTTP Request
                        ▼
  ┌──────────────────────────────────────────────────────────┐
  │              FastAPI API Layer                           │
  │         app/api/routes.py                                │
  │  - Input validation (Pydantic)                           │
  │  - Rate limiting                                         │
  │  - Route to workflow                                     │
  └─────────────────────┬────────────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────────────┐
  │         LangGraph Workflow Orchestration                 │
  │         app/orchestration/workflow.py                    │
  │                                                          │
  │  validate_user → guardrails → classify_query             │
  │       ↓ (RAG)                    ↓ (Tool)               │
  │  retrieve_docs              execute_tool                 │
  │  build_context                   ↓                       │
  │       ↓                          ↓                       │
  │  generate_answer ────────────────┘                       │
  │  format_response                                         │
  └──────┬──────────────────┬───────────────────────────────┘
         │                  │
         ▼                  ▼
  ┌─────────────┐   ┌──────────────────┐
  │ RAG Pipeline│   │ Agent Tools      │
  │ retriever.py│   │ company_tools.py │
  └──────┬──────┘   └──────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │              ChromaDB Vector Database                    │
  │         app/vector_store/chroma_store.py                 │
  │  - Stores document chunks + embeddings                   │
  │  - Permission-filtered similarity search                 │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                 LLM (Claude)                             │
  │  - Generates grounded answers from retrieved context     │
  │  - Enforced to answer ONLY from provided documents       │
  └──────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_TITLE, APP_VERSION
from app.api.routes import router
from app.observability.logger import logger


# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "A complete reference implementation of an enterprise AI Knowledge Assistant. "
        "Demonstrates RAG, LangGraph orchestration, AI agents, tool use, "
        "permission-based retrieval, guardrails, rate limiting, and observability."
    ),
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
)


# ──────────────────────────────────────────────
# CORS Middleware
# Allows browser-based clients to call the API
# ──────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production: list specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Include Routes
# ──────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")


# ──────────────────────────────────────────────
# Startup Event
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"  {APP_TITLE} v{APP_VERSION} starting up...")
    logger.info("  Architecture: FastAPI → LangGraph → RAG → ChromaDB → Claude")
    logger.info("  Endpoints available at: http://localhost:8000/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{APP_TITLE} shutting down.")


# ──────────────────────────────────────────────
# Root Redirect
# ──────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": f"Welcome to {APP_TITLE}",
        "docs":    "http://localhost:8000/docs",
        "api":     "http://localhost:8000/api/v1",
    }


# ──────────────────────────────────────────────
# Run directly (python app/main.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="::",
        port=8000,
        reload=True,   # Auto-reload on code changes (development only)
        log_level="info",
    )
