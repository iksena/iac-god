#!/usr/bin/env python3
"""
03_build_security_chromadb.py

Stage 1c – Build a ChromaDB vector collection for security remediation context.

This mirrors 05_build_chromadb.py in the CFN pipeline and stores into the
SAME ChromaDB Docker container (HttpClient, localhost:8000) so that both
collections are queryable from a single client in the retrieval stage.

CFN pipeline collection : 'cfn_schema_properties'  (built by 05_build_chromadb.py)
Security collection     : 'security_checks'         (built by this script)

Chunking strategy:
  - ONE document chunk per security check (vs. per-property in CFN pipeline)
  - page_content concatenates all semantic fields for rich matching:
      [check_id] [check_name] (Severity: [severity]): [description]
      AWS Service: [service] | CloudFormation resource prefix: [cfn_prefix]
      Impact: [impact]
      Remediation (CloudFormation): [remediation_cfn joined]
      CloudFormation Good Example:\n[cfn_good_example first 600 chars]
  - metadata carries bridge keys for Neo4j traversal:
      check_id, service, cfn_resource_prefix, severity, avd_url

Embedding model:
  Controlled by EMBEDDING_PROVIDER env var (huggingface | ollama).
  Defaults to sentence-transformers/all-mpnet-base-v2 via HuggingFace,
  matching 05_build_chromadb.py so both collections share the same vector space.

ChromaDB distance metric:
  Collection is always created with hnsw:space=cosine so similarity scores
  are comparable with cfn_schema_properties and threshold semantics are
  unambiguous regardless of provider.

Clean rebuild:
  If the collection already exists it is DELETED before ingestion so stale
  vectors from a previous CSV are never silently left in place.

Usage:
    python scripts/graphrag/security/03_build_security_chromadb.py

Environment variables:
    CHROMA_HOST          (default: localhost)
    CHROMA_PORT          (default: 8000)
    EMBEDDING_PROVIDER   (default: huggingface)  huggingface | ollama
    EMBEDDING_MODEL      (default: provider default)
    OLLAMA_BASE_URL      (default: http://localhost:11434)  Ollama only

Dependencies: langchain-core, langchain-huggingface, langchain-chroma, chromadb, numpy
               (+ langchain-ollama when EMBEDDING_PROVIDER=ollama)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import chromadb

try:
    from langchain_core.documents import Document
except ImportError:
    print("ERROR: pip install langchain-core", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("ERROR: pip install chromadb langchain-chroma", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

COLLECTION_NAME     = "security_checks"
EXAMPLE_PREVIEW_CHARS = 600

# Collection is always created with cosine distance so threshold semantics
# are unambiguous regardless of provider.  Must match the query-time config
# in 05_execute_security_g_retrieval.py (COLLECTION_METADATA).
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# ---------------------------------------------------------------------------
# Embedding provider — inline mirror of 05_build_chromadb.py so this script
# has no dependency on the tools/ package and supports the same env vars.
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()
_DEFAULTS = {
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama":      "mxbai-embed-large",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULTS.get(_PROVIDER, _DEFAULTS["huggingface"]))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class _NormalisedEmbeddings:
    """Thin wrapper that L2-normalises every vector before ingestion.

    Ensures unit-length vectors so cosine distance == angular distance,
    regardless of whether the underlying model normalises by default.
    HuggingFaceEmbeddings with normalize_embeddings=True already does this;
    OllamaEmbeddings does not.
    """
    def __init__(self, base):
        self._base = base

    @staticmethod
    def _norm(vecs):
        arr   = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).tolist()

    def embed_documents(self, texts):
        return self._norm(self._base.embed_documents(texts))

    def embed_query(self, text):
        return self._norm([self._base.embed_query(text)])[0]


def _get_embeddings():
    """Return an embedding object matching the configured EMBEDDING_PROVIDER.

    This is an exact copy of the factory in 05_build_chromadb.py so that the
    security collection is always embedded in the same vector space as
    cfn_schema_properties, enabling cross-collection similarity comparison.
    """
    if _PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            print(
                "ERROR: langchain-ollama not installed. "
                "Run: pip install langchain-ollama",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[Build] Embedding provider: Ollama  model: {EMBEDDING_MODEL}  url: {OLLAMA_BASE_URL}")
        base = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return _NormalisedEmbeddings(base)  # Ollama does not normalise by default

    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[Build] Embedding provider: HuggingFace  model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # already normalised
    )


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_page_content(check: dict) -> str:
    """
    Build the text chunk for a single security check.

    Concatenates all semantic fields so the embedding captures natural-language
    descriptions, impact text, and remediation instructions, enabling matching
    against queries like:
      'S3 bucket public read access'      → AVD-AWS-0173
      'API Gateway missing authorization' → AVD-AWS-0004
    """
    check_id   = check.get("check_id", "")
    check_name = check.get("check_name", "")
    severity   = check.get("severity", "")
    description = check.get("description", "")
    service    = check.get("service", "")
    cfn_prefix = check.get("cfn_resource_prefix", "")
    impact     = check.get("impact", "")

    rem_cfn = check.get("remediation_cfn", [])
    if isinstance(rem_cfn, list):
        remediation = "; ".join(rem_cfn)
    else:
        remediation = str(rem_cfn)

    cfn_example = check.get("cfn_good_example", "")[:EXAMPLE_PREVIEW_CHARS]

    lines = [
        f"[{check_id}] {check_name} (Severity: {severity})",
        f"Description: {description}",
        f"AWS Service: {service} | CloudFormation resource prefix: {cfn_prefix}",
    ]
    if impact:
        lines.append(f"Impact: {impact}")
    if remediation:
        lines.append(f"Remediation (CloudFormation): {remediation}")
    if cfn_example:
        lines.append(f"CloudFormation Good Example:\n{cfn_example}")

    return "\n".join(lines)


def build_documents(checks: dict) -> list[Document]:
    docs = []
    for check_id, check in checks.items():
        docs.append(Document(
            page_content=build_page_content(check),
            metadata={
                "check_id":            check_id,
                "check_name":          check.get("check_name", ""),
                "severity":            check.get("severity", ""),
                "service":             check.get("service", ""),
                "cfn_resource_prefix": check.get("cfn_resource_prefix", ""),
                "avd_url":             check.get("avd_url", ""),
                "source_file_url":     check.get("source_file_url", ""),
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading security checks from: {CHECKS_JSON}")
    if not CHECKS_JSON.exists():
        print("ERROR: Run 01_load_trivy_csv.py first.", file=sys.stderr)
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    print(f"Building documents for {len(checks)} security checks...")
    documents = build_documents(checks)

    embeddings = _get_embeddings()

    print(f"Connecting to ChromaDB Docker at {CHROMA_HOST}:{CHROMA_PORT} ...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Verify the CFN collection is present so the user knows both will co-exist.
    existing = [c.name for c in chroma_client.list_collections()]
    print(f"  Existing collections: {existing}")
    if "cfn_schema_properties" not in existing:
        print(
            "  WARNING: 'cfn_schema_properties' not found.\n"
            "  Run scripts/graphrag/05_build_chromadb.py first so both\n"
            "  collections share the same Docker container."
        )

    # -----------------------------------------------------------------------
    # Clean rebuild: drop stale collection before re-ingestion.
    # This ensures stale vectors from a previous CSV are never silently
    # left in place when the source data changes.
    # -----------------------------------------------------------------------
    if COLLECTION_NAME in existing:
        print(f"  Dropping existing collection '{COLLECTION_NAME}' for clean rebuild...")
        chroma_client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' (hnsw:space=cosine) ...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        collection_metadata=COLLECTION_METADATA,   # hnsw:space=cosine
    )

    # Smoke-test
    print("\nSmoke test – querying: 'S3 bucket public access encryption'")
    results = vectorstore.similarity_search("S3 bucket public access encryption", k=3)
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"  [{i}] {m['check_id']} | {m['check_name']} | {m['severity']}")

    print(f"\n✓ Collection '{COLLECTION_NAME}' built in ChromaDB Docker.")
    print(f"  Provider   : {_PROVIDER}")
    print(f"  Model      : {EMBEDDING_MODEL}")
    print(f"  Distance   : cosine (hnsw:space=cosine)")
    print(f"  Normalised : True")
    print(f"  Total docs : {len(documents)}")
    print(f"  Endpoint   : {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"  Collections: {[c.name for c in chroma_client.list_collections()]}")


if __name__ == "__main__":
    main()
