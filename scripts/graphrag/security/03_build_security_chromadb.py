#!/usr/bin/env python3
"""
03_build_security_chromadb.py

Stage 1c – Build a ChromaDB vector collection for security remediation context.

This mirrors 05_build_chromadb.py in the CFN pipeline but uses security_checks.json
as the source and creates a separate 'security_checks' collection.

Chunking strategy:
  - ONE document chunk per security check (vs. per-property in CFN pipeline)
  - page_content concatenates all semantic fields for rich semantic matching:
      [check_id] [check_name] ([severity]): [description]
      Service: [service] | CFN Prefix: [cfn_resource_prefix]
      Impact: [impact]
      Remediation: [remediation_cfn joined]
      Example:\n[cfn_good_example first 500 chars]
  - metadata carries bridge keys for Neo4j traversal:
      check_id, service, cfn_resource_prefix, severity, avd_url

Embedding model: sentence-transformers/all-mpnet-base-v2 (same as CFN pipeline)
ChromaDB collection: 'security_checks'
Persist directory: data/chroma_security_db/

Usage:
    python scripts/graphrag/security/03_build_security_chromadb.py

Dependencies: langchain-community, langchain-huggingface, chromadb
"""

import json
import sys
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    print("ERROR: Install langchain-core: pip install langchain-core", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("ERROR: Install langchain-huggingface: pip install langchain-huggingface", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("ERROR: Install chromadb + langchain-chroma: pip install chromadb langchain-chroma", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"
CHROMA_PERSIST_DIR = str(Path("/Users/iksena/Documents/research/cfn-chroma-docker") / "chroma_data")
COLLECTION_NAME = "security_checks"

# Embedding model – same as CFN pipeline for consistency
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Max chars of cfn_good_example to include in the chunk (keep token count reasonable)
EXAMPLE_PREVIEW_CHARS = 600


def build_page_content(check: dict) -> str:
    """
    Build the text chunk for a single security check.

    Strategy: concatenate all semantic fields so the embedding captures
    natural-language descriptions, impact text, and remediation instructions.
    This allows matching against queries like:
      'S3 bucket public read access' → AVD-AWS-0173
      'API Gateway missing authorization' → AVD-AWS-0004
    """
    check_id = check.get("check_id", "")
    check_name = check.get("check_name", "")
    severity = check.get("severity", "")
    description = check.get("description", "") or check.get("page_description", "")
    service = check.get("service", "")
    cfn_prefix = check.get("cfn_resource_prefix", "")

    # Impact: prefer page-scraped (richer) over CSV column
    impact = check.get("page_impact", "") or check.get("impact", "")
    
    # Remediation: join list items
    rem_cfn = check.get("remediation_cfn", [])
    if isinstance(rem_cfn, list):
        remediation = "; ".join(rem_cfn)
    else:
        remediation = str(rem_cfn)
    if not remediation:
        remediation = check.get("page_remediation", "")

    # CFN example preview
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
    """Convert security_checks dict into LangChain Document objects."""
    docs = []
    for check_id, check in checks.items():
        page_content = build_page_content(check)
        metadata = {
            "check_id": check_id,
            "check_name": check.get("check_name", ""),
            "severity": check.get("severity", ""),
            "service": check.get("service", ""),
            "cfn_resource_prefix": check.get("cfn_resource_prefix", ""),
            "avd_url": check.get("avd_url", ""),
            "source_file_url": check.get("source_file_url", ""),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def main():
    print(f"Loading security checks from: {CHECKS_JSON}")
    if not CHECKS_JSON.exists():
        print("ERROR: Run 01_load_trivy_csv.py (and optionally 02_scrape_avd_docs.py) first.",
              file=sys.stderr)
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    print(f"Building documents for {len(checks)} security checks...")
    documents = build_documents(checks)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~420MB – cached locally afterwards)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Creating ChromaDB collection '{COLLECTION_NAME}' at: {CHROMA_PERSIST_DIR}")
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Smoke-test: semantic search for a known pattern
    print("\nSmoke test – querying: 'S3 bucket public access encryption'")
    results = vectorstore.similarity_search(
        "S3 bucket public access encryption", k=3
    )
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"  [{i}] {meta['check_id']} | {meta['check_name']} | {meta['severity']}")
        print(f"       AVD: {meta['avd_url']}")

    print(f"\n✓ ChromaDB collection '{COLLECTION_NAME}' built successfully.")
    print(f"  Total documents: {len(documents)}")
    print(f"  Persist directory: {CHROMA_PERSIST_DIR}")
    print("\nNext steps:")
    print("  Stage 2: Run scripts/graphrag/security/04_import_security_to_neo4j.py")


if __name__ == "__main__":
    main()
