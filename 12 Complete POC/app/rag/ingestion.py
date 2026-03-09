"""
app/rag/ingestion.py — Document Ingestion Pipeline

[Concept: RAG - Document Ingestion]

────────────────────────────────────────────────────────────────
WHY CHUNKING IS CRITICAL
────────────────────────────────────────────────────────────────
LLMs have a context window limit (e.g., 200K tokens for Claude).
But the REAL reason we chunk is different:

  PROBLEM: If you store entire documents (e.g., a 50-page handbook),
  the similarity search is comparing your 10-word query against
  50 pages of text. The signal gets diluted — the document might
  be very relevant in one section but mostly irrelevant overall.

  SOLUTION: Split documents into small chunks (300-600 tokens each).
  Now similarity search compares your query against focused pieces
  of text. The most relevant chunk rises to the top.

  A good chunk:
    - Contains ONE idea or concept
    - Is large enough to have context (not too small)
    - Has overlap with neighbors (so ideas don't get cut off)

  Example:
    Full HR Handbook → 200 chunks of 500 tokens each
    Query: "vacation days"
    → Chunk 47: "Full-time employees receive 15 vacation days..."  (score: 0.89)
    → Chunk 48: "Vacation days accrue monthly at 1.25 days per..."  (score: 0.82)
    → Chunk 12: "The onboarding process begins on day one..."  (score: 0.21)

  We return the TOP 4 chunks, not the whole 200-page document.

The ingestion pipeline:
  Raw text/PDF → Load → Split into chunks → Embed → Store in ChromaDB
────────────────────────────────────────────────────────────────
"""

import uuid
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models.schemas import AccessLevel
from app.vector_store.chroma_store import vector_store
from app.observability.logger import logger


# ──────────────────────────────────────────────
# Text Splitter
# ──────────────────────────────────────────────

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter tries to split on natural boundaries:
      1. Double newline (paragraph break) — preferred split point
      2. Single newline
      3. Period + space (sentence break)
      4. Space (word break)
      5. Character level (last resort)

    chunk_overlap: the number of characters shared between consecutive
    chunks. This prevents ideas that span a chunk boundary from being
    split in half and losing context.

    Example with overlap=50:
      Chunk 1: "...employees receive 15 vacation days per year. Unused"
      Chunk 2: "Unused vacation days may be carried over up to 5 days..."
      (The word "Unused" appears in both, preserving the continuity)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ──────────────────────────────────────────────
# Ingestion Functions
# ──────────────────────────────────────────────

def ingest_text_content(
    content: str,
    title: str,
    department: str,
    access_level: AccessLevel,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Ingest raw text content into the vector database.

    Pipeline:
      1. Wrap text in a LangChain Document object
      2. Split into chunks using RecursiveCharacterTextSplitter
      3. Attach metadata to each chunk (doc_id, title, etc.)
      4. Embed and store all chunks in ChromaDB

    The metadata stored with each chunk is critical because:
      - It allows us to filter by access_level during search
      - It allows us to show the user which document was cited
      - It allows us to group chunks back to their source document

    Args:
        content:      The raw text to ingest
        title:        Human-readable document title
        department:   Owning department (HR, Finance, IT...)
        access_level: Who can access this document
        doc_id:       Optional — auto-generated if not provided

    Returns:
        dict with doc_id, chunk_count, and status message
    """
    if doc_id is None:
        # Generate a unique ID for this document
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    logger.info(f"[INGESTION] Starting ingestion: title='{title}' doc_id={doc_id}")

    # Step 1: Create a LangChain Document
    source_doc = Document(
        page_content=content,
        metadata={
            "doc_id":       doc_id,
            "title":        title,
            "department":   department,
            "access_level": access_level.value,
            "source":       "text_input",
        }
    )

    # Step 2: Split into chunks
    splitter = get_text_splitter()
    chunks = splitter.split_documents([source_doc])

    logger.info(f"[INGESTION] Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Add chunk index to metadata
    # This helps us display chunks in order and debug retrieval
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    # Step 4: Embed and store in ChromaDB
    # The vector_store handles embedding generation and persistence
    ids = vector_store.add_documents(chunks)

    logger.info(f"[INGESTION] Successfully stored {len(ids)} chunks for doc_id={doc_id}")

    return {
        "doc_id":        doc_id,
        "title":         title,
        "chunks_created": len(chunks),
        "message":       f"Successfully ingested '{title}' as {len(chunks)} chunks."
    }


def ingest_file(
    file_path: str,
    title: str,
    department: str,
    access_level: AccessLevel,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Ingest a file (TXT or PDF) from disk into the vector database.

    Uses LangChain document loaders which handle format-specific
    parsing (PDF page extraction, text encoding, etc.)
    """
    if doc_id is None:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    logger.info(f"[INGESTION] Loading file: {file_path}")

    # Select loader based on file extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    raw_docs = loader.load()

    # Combine pages into a single text for consistent chunking
    full_text = "\n\n".join([doc.page_content for doc in raw_docs])

    return ingest_text_content(
        content=full_text,
        title=title,
        department=department,
        access_level=access_level,
        doc_id=doc_id,
    )
