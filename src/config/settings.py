"""
Centralized configuration for the RAG system.

All environment variables, constants, and defaults live here.
Import from this module instead of hardcoding values in individual files.
"""

import os
from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Load .env (if not already loaded by the runtime) ──────────────
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            if _line.strip() and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.strip().split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))


# ── Database ──────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "6024"))
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
DB_NAME = os.getenv("DB_NAME", "rag_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")

DB_CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
DB_CONNECTION_STRING_PSYCOPG = (
    f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASSWORD} dbname={DB_NAME}"
)

# ── Models ────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
REWRITER_MODEL = os.getenv("REWRITER_MODEL", "openrouter/free")
AGENT_MODEL = os.getenv("AGENT_MODEL", "openrouter/auto")

# ── API ───────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Server ────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))

# ── Retrieval ─────────────────────────────────────────────────────
HYBRID_K = int(os.getenv("HYBRID_K", "8"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))
REWRITE_TIMEOUT_SECONDS = int(os.getenv("REWRITE_TIMEOUT_SECONDS", "10"))

# ── Cache ─────────────────────────────────────────────────────────
CACHE_COLLECTION = "semantic_cache"
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.05"))
