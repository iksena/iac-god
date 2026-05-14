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

Embedding model: sentence-transformers/all-mpnet-base-v2 (same as CFN pipeline)

Usage:
    python scripts/graphrag/security/03_build_security_chromadb.py

Environment variables:
    CHROMA_HOST   (default: localhost)
    CHROMA_PORT   (default: 8000)

Dependencies: langchain-core, langchain-huggingface, langchain-chroma, chromadb
"""

import json
import os
import sys
from pathlib import Path

import chromadb

try:
    from langchain_core.documents import Document
except ImportError:
    print("ERROR: pip install langchain-core", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("ERROR: pip install langchain-huggingface", file=sys.stderr)
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
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"

# ChromaDB Docker connection – must match the container started by the CFN pipeline
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

COLLECTION_NAME = "security_checks"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EXAMPLE_PREVIEW_CHARS = 600


def build_page_content(check: dict) -> str:
    """
    Build the text chunk for a single security check.

    Concatenates all semantic fields so the embedding captures natural-language
    descriptions, impact text, and remediation instructions, enabling matching
    against queries like:
      'S3 bucket public read access'      → AVD-AWS-0173
      'API Gateway missing authorization' → AVD-AWS-0004
    """
    check_id = check.get("check_id", "")
    check_name = check.get("check_name", "")
    severity = check.get("severity", "")
    description = check.get("description", "")
    service = check.get("service", "")
    cfn_prefix = check.get("cfn_resource_prefix", "")
    impact = check.get("impact", "")

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
                "check_id": check_id,
                "check_name": check.get("check_name", ""),
                "severity": check.get("severity", ""),
                "service": check.get("service", ""),
                "cfn_resource_prefix": check.get("cfn_resource_prefix", ""),
                "avd_url": check.get("avd_url", ""),
                "source_file_url": check.get("source_file_url", ""),
            },
        ))
    return docs


def main():
    print(f"Loading security checks from: {CHECKS_JSON}")
    if not CHECKS_JSON.exists():
        print("ERROR: Run 01_load_trivy_csv.py first.", file=sys.stderr)
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    print(f"Building documents for {len(checks)} security checks...")
    documents = build_documents(checks)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Connecting to ChromaDB Docker at {CHROMA_HOST}:{CHROMA_PORT} ...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Verify the CFN collection is present so the user knows both will co-exist
    existing = [c.name for c in chroma_client.list_collections()]
    print(f"  Existing collections: {existing}")
    if "cfn_schema_properties" not in existing:
        print(
            "  WARNING: 'cfn_schema_properties' not found.\n"
            "  Run scripts/graphrag/05_build_chromadb.py first so both\n"
            "  collections share the same Docker container."
        )

    print(f"Creating collection '{COLLECTION_NAME}' ...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
    )

    # Smoke-test
    print("\nSmoke test – querying: 'S3 bucket public access encryption'")
    results = vectorstore.similarity_search("S3 bucket public access encryption", k=3)
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"  [{i}] {m['check_id']} | {m['check_name']} | {m['severity']}")

    print(f"\n✓ Collection '{COLLECTION_NAME}' built in ChromaDB Docker.")
    print(f"  Total documents : {len(documents)}")
    print(f"  Docker endpoint : {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"  Collections now : {[c.name for c in chroma_client.list_collections()]}")


if __name__ == "__main__":
    main()
