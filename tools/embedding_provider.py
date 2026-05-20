"""tools/embedding_provider.py

Central factory for the embedding model used by all RAG tools.

Controlled entirely by environment variables so no code changes are needed
when switching between providers or models:

  EMBEDDING_PROVIDER   "huggingface" (default) | "ollama"
  EMBEDDING_MODEL      Model name/path for the chosen provider.
                       Defaults depend on provider (see DEFAULTS below).
  OLLAMA_BASE_URL      Base URL of the local Ollama server.
                       Default: http://localhost:11434

Examples
--------
# Use Ollama with mxbai-embed-large (recommended for local inference)
EMBEDDING_PROVIDER=ollama python main.py

# Override model explicitly
EMBEDDING_PROVIDER=ollama EMBEDDING_MODEL=nomic-embed-text python main.py

# Keep HuggingFace but switch model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 python main.py

IMPORTANT — Index / query model parity
---------------------------------------
The embedding model used at *index time* (05_build_chromadb.py) and at
*query time* (cfn_hybrid_rag.py, security_hybrid_rag.py) MUST be the same
model.  Switching EMBEDDING_PROVIDER or EMBEDDING_MODEL without rebuilding
the ChromaDB collection will silently return wrong results.

Refer to scripts/graphrag/README.md → "Step 3.5 Build ChromaDB" for the
rebuild procedure.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Union

from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()

_DEFAULTS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama":      "mxbai-embed-large",
}

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", _DEFAULTS.get(_PROVIDER, _DEFAULTS["huggingface"]))
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the embedding model instance for the configured provider.

    The result is cached for the process lifetime so the model is loaded
    only once regardless of how many RAG tools call this function.
    """
    if _PROVIDER == "ollama":
        return _build_ollama_embeddings()
    return _build_huggingface_embeddings()


def _build_huggingface_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"[Embeddings] Provider: HuggingFace  Model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_ollama_embeddings() -> Embeddings:
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for EMBEDDING_PROVIDER=ollama. "
            "Install it with: pip install langchain-ollama"
        ) from exc

    print(f"[Embeddings] Provider: Ollama  Model: {EMBEDDING_MODEL}  URL: {OLLAMA_BASE_URL}")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Convenience accessors (used by build scripts)
# ---------------------------------------------------------------------------

def get_provider() -> str:
    """Return the active provider name ('huggingface' or 'ollama')."""
    return _PROVIDER


def get_model_name() -> str:
    """Return the active model identifier string."""
    return EMBEDDING_MODEL
