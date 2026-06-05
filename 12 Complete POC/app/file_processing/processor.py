"""
app/file_processing/processor.py — File Upload Processing

Supports PDF and TXT files.

MOCK_FILE_PROCESSING=true  (default, for development):
    Reads file bytes and decodes as UTF-8. No external libraries needed.
    Fast to set up, works for plain text and readable PDFs.

MOCK_FILE_PROCESSING=false (production):
    PDF  → PyPDFLoader extracts text page-by-page
    TXT  → plain UTF-8 read

To remove mock mode in production:
    1. Set MOCK_FILE_PROCESSING=false in .env
    2. Delete the `if MOCK_FILE_PROCESSING` branch in process_upload()
    3. The rest of the pipeline is unchanged.
"""

import os
import tempfile
from fastapi import UploadFile, HTTPException, status

from app.config import MOCK_FILE_PROCESSING


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


async def process_upload(file: UploadFile) -> str:
    """
    Read an uploaded file and return its text content.

    Args:
        file: FastAPI UploadFile object

    Returns:
        Extracted text content as a string

    Raises:
        HTTPException(400) for unsupported file types
        HTTPException(422) if content cannot be decoded
    """
    ext = _get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    file_bytes = await file.read()

    if MOCK_FILE_PROCESSING:
        # ── MOCK MODE ──────────────────────────────────────────────────────
        # Read bytes as plain text. Suitable for TXT and human-readable PDFs.
        # Remove this branch when MOCK_FILE_PROCESSING=false in production.
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not decode file: {e}"
            )
    else:
        # ── PRODUCTION MODE ────────────────────────────────────────────────
        # Write to a temp file so LangChain loaders can read by path.
        suffix = ext
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                return "\n\n".join(doc.page_content for doc in docs)
            else:  # .txt
                with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
