"""
app/config.py — Centralized Configuration

Loads all settings from environment variables (.env file).
Using a single config module prevents scattered os.getenv() calls
and makes the project easier to configure and test.
"""

import os
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()

# ──────────────────────────────────────────────
# LLM & Embedding Settings
# ──────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")

# LLM_PROVIDER controls which LLM is used for answer generation.
# "openai"    → requires OPENAI_API_KEY only
# "anthropic" → requires ANTHROPIC_API_KEY (+ OPENAI_API_KEY for embeddings)
LLM_PROVIDER: str    = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL: str       = os.getenv("LLM_MODEL", "gpt-4o-mini")   # default: OpenAI
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# ──────────────────────────────────────────────
# Storage Settings
# ──────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
SQLITE_DB_PATH: str     = os.getenv("SQLITE_DB_PATH", "./data/metadata.db")

# ──────────────────────────────────────────────
# RAG Settings
# ──────────────────────────────────────────────
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
# WINDOW_SIZE controls sentence window retrieval: how many neighboring chunks
# to fetch on each side of a matched chunk. WINDOW_SIZE=1 fetches [N-1, N, N+1].
# Set to 0 to disable sentence window retrieval and use plain top-K only.
WINDOW_SIZE: int   = int(os.getenv("WINDOW_SIZE", "1"))

# ──────────────────────────────────────────────
# Rate Limiting Settings
# ──────────────────────────────────────────────
RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW: int   = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# ──────────────────────────────────────────────
# App Settings
# ──────────────────────────────────────────────
APP_TITLE: str       = "AI Knowledge Assistant"
APP_VERSION: str     = "1.0.0"
COLLECTION_NAME: str = "company_documents"
