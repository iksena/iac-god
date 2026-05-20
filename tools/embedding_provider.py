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

Distance metric
---------------
All collections are created with hnsw:space="cosine" so scores are
true cosine distances in [0, 2].  Both providers normalise their output
vectors to unit length so L2 and cosine rankings are equivalent, but the
explicit cosine space is set to make threshold semantics unambiguous.

Refer to scripts/graphrag/README.md for rebuild instructions.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
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
# Normalisation wrapper
# ---------------------------------------------------------------------------

class _NormalisedEmbeddings(Embeddings):
    """Wraps any Embeddings instance and L2-normalises every output vector.

    Guarantees unit-length vectors so that cosine distance and L2 distance
    produce identical rankings.  This is especially important for
    OllamaEmbeddings which does not normalise by default, unlike
    HuggingFaceEmbeddings with encode_kwargs={"normalize_embeddings": True}.
    """

    def __init__(self, base: Embeddings) -> None:
        self._base = base

    @staticmethod
    def _normalise(vecs: list[list[float]]) -> list[list[float]]:
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors (degenerate edge case)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._normalise(self._base.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._normalise([self._base.embed_query(text)])[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the (normalised) embedding model for the configured provider.

    The result is cached for the process lifetime so the model is loaded
    only once regardless of how many RAG tools call this function.
    """
    if _PROVIDER == "ollama":
        return _NormalisedEmbeddings(_build_ollama_embeddings())
    return _build_huggingface_embeddings()  # HuggingFace already normalises


def _build_huggingface_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"[Embeddings] Provider: HuggingFace  Model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # unit vectors
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
# Convenience accessors (used by build scripts and tests)
# ---------------------------------------------------------------------------

def get_provider() -> str:
    """Return the active provider name ('huggingface' or 'ollama')."""
    return _PROVIDER


def get_model_name() -> str:
    """Return the active model identifier string."""
    return EMBEDDING_MODEL


# CHROMA_COLLECTION_METADATA must be passed to every Chroma.from_documents()
# and Chroma() constructor so all collections share the same distance metric.
# Cosine distance scores are in [0, 2]; lower = more similar.
# Threshold bands for mxbai-embed-large (cosine, normalised):
#   0.00 – 0.20  near-exact match
#   0.20 – 0.40  topically related  ← default threshold keeps these
#   0.40 – 0.65  loosely related    ← default threshold drops these
#   0.65+        unrelated
# Threshold bands for all-mpnet-base-v2 (cosine, normalised):
#   0.00 – 0.30  near-exact match
#   0.30 – 0.55  topically related  ← same logic
#   0.55 – 0.80  loosely related
#   0.80+        unrelated
CHROMA_COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}

# Default distance threshold (override with CHROMA_DISTANCE_THRESHOLD env var).
# 0.40 is the safe conservative starting point for mxbai-embed-large.
# Raise toward 0.55 if recall is too low; lower toward 0.25 for higher precision.
DEFAULT_DISTANCE_THRESHOLD: float = float(
    os.getenv("CHROMA_DISTANCE_THRESHOLD", "0.40")
)
