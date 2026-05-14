#!/usr/bin/env python3
"""
05_execute_security_g_retrieval.py

Stage 3 – Combined G-Retrieval: CFN schema + security remediation context.

Retrieval flow
--------------
  User query
      ├─ ChromaDB 'cfn_resources'    (k=SEMANTIC_K)   → resource_names
      └─ ChromaDB 'security_checks'  (k=SEMANTIC_K_SEC) → check_ids
           │  (+ optional service-hint boost pass if service name in query)
           ▼
      Neo4j pass A: query_cfn_subgraph(resource_names)
      Neo4j pass B: query_security_subgraph(check_ids)
           ▼
      format_combined_context(cfn_result, security_result)
           ▼
      Caller injects into LLM prompt

Environment variables
---------------------
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    CHROMA_PERSIST_DIR   (default: same path used by 03_build_security_chromadb.py
                          and 05_build_chromadb.py in the CFN pipeline)

Usage (standalone test)
-----------------------
    python scripts/graphrag/security/05_execute_security_g_retrieval.py \\
        --query "S3 bucket with encryption and no public access"
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: pip install neo4j", file=sys.stderr)
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
# Config
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Both the CFN pipeline (05_build_chromadb.py) and the security pipeline
# (03_build_security_chromadb.py) must write to the SAME persist directory
# so that retrieve_combined_context() can find both collections in one store.
# Override via env var if your local path differs.
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path("/Users/iksena/Documents/research/cfn-chroma-docker") / "chroma_data"),
)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CFN_COLLECTION = "cfn_resources"
SECURITY_COLLECTION = "security_checks"

# k for CFN schema search – 5 is enough (schema results are per-resource)
SEMANTIC_K = 5
# k for security search – higher because one service can have 10-15 checks
SEMANTIC_K_SEC = 10
# absolute ceiling when service-hint boost is added
SEMANTIC_K_MAX = 15

# Known AWS service name tokens for service-hint boosting.
# When the user query contains one of these words, a second semantic search
# is run with just the service name to surface additional service-specific checks.
_AWS_SERVICE_TOKENS = {
    "s3", "ec2", "rds", "iam", "lambda", "cloudtrail", "cloudfront",
    "apigateway", "api", "dynamodb", "elasticache", "eks", "ecs",
    "kinesis", "sqs", "sns", "kms", "vpc", "elb", "elbv2", "msk",
    "athena", "glue", "redshift", "emr", "sagemaker", "codebuild",
    "codecommit", "secretsmanager", "ssm", "config",
}

# ---------------------------------------------------------------------------
# HTML comment cleaning (AVD scaffold for empty impact fields)
# ---------------------------------------------------------------------------
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)


def clean_impact(raw: str | None) -> str:
    """Strip HTML comment placeholders (<!-- Add Impact here -->) from impact."""
    if not raw:
        return ""
    return _HTML_COMMENT_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Shared embedding model (lazy-initialised once)
# ---------------------------------------------------------------------------
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Collection health-check helper
# ---------------------------------------------------------------------------

def _validate_collection(vectorstore: Chroma, collection_name: str) -> int:
    """
    Return the document count in the collection.
    Raises RuntimeError with an actionable message if the collection is empty.
    """
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0
    return count


# ---------------------------------------------------------------------------
# Stage 1a: Semantic search – CFN collection
# ---------------------------------------------------------------------------

def semantic_search_cfn(query: str, k: int = SEMANTIC_K) -> list[str]:
    """
    Returns a deduplicated list of CFN resource names relevant to the query.

    Raises RuntimeError with an actionable message if the cfn_resources
    collection is missing or empty – previously this returned [] silently,
    masking the fact that the CFN ChromaDB had not been built into this dir.
    """
    print(f"  [CFN ChromaDB] Semantic search: '{query[:60]}'")
    vectorstore = Chroma(
        collection_name=CFN_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    count = _validate_collection(vectorstore, CFN_COLLECTION)
    if count == 0:
        raise RuntimeError(
            f"\nERROR: CFN collection '{CFN_COLLECTION}' is empty or missing in:\n"
            f"  {CHROMA_PERSIST_DIR}\n"
            "Fix: Run scripts/graphrag/05_build_chromadb.py first, ensuring\n"
            "  CHROMA_PERSIST_DIR points to the same directory as this script.\n"
            "  Both collections (cfn_resources + security_checks) must share\n"
            "  a single persist directory."
        )

    results = vectorstore.similarity_search(query, k=k)
    resource_names: list[str] = []
    seen: set[str] = set()
    for doc in results:
        name = doc.metadata.get("resource_name", "")
        if name and name not in seen:
            seen.add(name)
            resource_names.append(name)
    print(f"    → {resource_names}")
    return resource_names


# ---------------------------------------------------------------------------
# Stage 1b: Semantic search – security collection
# ---------------------------------------------------------------------------

def _extract_service_hint(query: str) -> str | None:
    """
    If the query mentions a known AWS service token, return it.
    Used to run a second targeted search to boost service-specific recall.
    """
    tokens = set(re.findall(r'[a-z0-9]+', query.lower()))
    hits = tokens & _AWS_SERVICE_TOKENS
    # Prefer more specific tokens (longer = more specific, e.g. 'apigateway' > 'api')
    return max(hits, key=len) if hits else None


def semantic_search_security(
    query: str,
    k: int = SEMANTIC_K_SEC,
    k_max: int = SEMANTIC_K_MAX,
) -> list[str]:
    """
    Returns a deduplicated list of check_ids relevant to the query.

    Two-pass strategy:
      Pass 1: full query, k=k results
      Pass 2 (if service hint detected): service token as query, k=k//2 results
    Results are merged (pass-1 order preserved, pass-2 appended) up to k_max.
    """
    print(f"  [Security ChromaDB] Semantic search: '{query[:60]}'")
    vectorstore = Chroma(
        collection_name=SECURITY_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )

    count = _validate_collection(vectorstore, SECURITY_COLLECTION)
    if count == 0:
        print(f"  WARNING: Security collection '{SECURITY_COLLECTION}' is empty.")
        print("  Run scripts/graphrag/security/03_build_security_chromadb.py first.")
        return []

    # Pass 1: full query
    results = vectorstore.similarity_search(query, k=k)
    check_ids: list[str] = []
    seen: set[str] = set()
    for doc in results:
        cid = doc.metadata.get("check_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            check_ids.append(cid)

    # Pass 2: service-hint boost
    service_hint = _extract_service_hint(query)
    if service_hint and len(check_ids) < k_max:
        remaining = k_max - len(check_ids)
        boost_results = vectorstore.similarity_search(service_hint, k=remaining)
        for doc in boost_results:
            cid = doc.metadata.get("check_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                check_ids.append(cid)
        if service_hint:
            print(f"    (service hint '{service_hint}' added {len(check_ids) - len(results)} more checks)")

    print(f"    → {check_ids}")
    return check_ids


# ---------------------------------------------------------------------------
# Stage 2a: Neo4j graph traversal – CFN schema subgraph
# ---------------------------------------------------------------------------

CFN_CYPHER = """
    MATCH (r:Resource {name: $resource_name})

    OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(req_prop:Property)
    WHERE req_prop.required = true
    WITH r, collect(DISTINCT {name: req_prop.name, type: req_prop.type}) AS required_properties

    OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(opt_prop:Property)
    WHERE opt_prop.required = false
    WITH r, required_properties,
         collect(DISTINCT {name: opt_prop.name, type: opt_prop.type}) AS optional_properties

    OPTIONAL MATCH (r)-[:HAS_NESTED_TYPE]->(nt:NestedType)
    WITH r, required_properties, optional_properties,
         collect(DISTINCT {name: nt.name, type: nt.type, required: nt.required}) AS nested_types

    OPTIONAL MATCH (r)-[:HAS_EXAMPLE]->(e:Example)
    WHERE e.index = 0

    RETURN r.name        AS resource_name,
           r.description AS resource_description,
           required_properties,
           optional_properties,
           nested_types,
           {code: e.code} AS example
"""


def query_cfn_subgraph(driver, resource_names: list[str]) -> list[dict[str, Any]]:
    results = []
    with driver.session() as session:
        for name in resource_names:
            row = session.run(CFN_CYPHER, resource_name=name).single()
            if row:
                results.append({
                    "name": row["resource_name"],
                    "description": row["resource_description"],
                    "required_properties": row["required_properties"],
                    "optional_properties": row["optional_properties"],
                    "nested_types": row["nested_types"],
                    "example": row["example"],
                })
    return results


# ---------------------------------------------------------------------------
# Stage 2b: Neo4j graph traversal – security subgraph
# ---------------------------------------------------------------------------

SECURITY_CYPHER = """
    MATCH (s:SecurityCheck {check_id: $check_id})

    OPTIONAL MATCH (s)-[:AFFECTS_SERVICE]->(svc:AwsService)
    OPTIONAL MATCH (s)-[:HAS_IMPACT]->(imp:Impact)
    OPTIONAL MATCH (s)-[:HAS_REMEDIATION]->(rem:Remediation)
    OPTIONAL MATCH (s)-[:HAS_GOOD_EXAMPLE]->(ex:GoodExample)
    OPTIONAL MATCH (s)-[:ENFORCED_BY]->(rp:RegoPolicy)
    OPTIONAL MATCH (s)-[:APPLIES_TO_RESOURCE]->(cfn_r:Resource)

    RETURN s.check_id     AS check_id,
           s.check_name   AS check_name,
           s.severity     AS severity,
           s.description  AS description,
           s.avd_url      AS avd_url,
           s.title        AS title,
           svc.name       AS service,
           imp.text       AS impact,
           collect(DISTINCT {framework: rem.framework,
                             instruction: rem.instruction}) AS remediations,
           collect(DISTINCT {framework: ex.framework,
                             code: ex.code})               AS examples,
           rp.source_file_url                              AS rego_source_url,
           collect(DISTINCT cfn_r.name)                    AS cfn_resource_names
"""


def query_security_subgraph(driver, check_ids: list[str]) -> list[dict[str, Any]]:
    results = []
    with driver.session() as session:
        for cid in check_ids:
            row = session.run(SECURITY_CYPHER, check_id=cid).single()
            if row:
                results.append({
                    "check_id": row["check_id"],
                    "check_name": row["check_name"],
                    "severity": row["severity"],
                    "description": row["description"],
                    "avd_url": row["avd_url"],
                    "title": row["title"],
                    "service": row["service"],
                    "impact": row["impact"],
                    "remediations": row["remediations"],
                    "examples": row["examples"],
                    "rego_source_url": row["rego_source_url"],
                    "cfn_resource_names": row["cfn_resource_names"],
                })
    return results


# ---------------------------------------------------------------------------
# Stage 3: Format context blocks
# ---------------------------------------------------------------------------

def format_cfn_context(resources: list[dict]) -> str:
    if not resources:
        return "No CloudFormation schema context found."

    lines = []
    for res in resources:
        lines.append(f"Resource: {res['name']}")
        lines.append(f"Description: {res.get('description', '')}")

        req = [p for p in (res.get("required_properties") or []) if p.get("name")]
        if req:
            lines.append("Required Properties:")
            for p in req:
                lines.append(f"  - {p['name']} ({p['type']})")

        opt = [p for p in (res.get("optional_properties") or []) if p.get("name")]
        if opt:
            lines.append("Optional Properties (selection):")
            for p in opt[:10]:
                lines.append(f"  - {p['name']} ({p['type']})")

        nt = [n for n in (res.get("nested_types") or []) if n.get("name")]
        if nt:
            lines.append("Nested Types:")
            for n in nt:
                lines.append(f"  - {n['name']} ({n['type']})")

        ex = res.get("example") or {}
        if ex.get("code"):
            lines.append("YAML Example:")
            lines.append(f"```yaml\n{ex['code']}\n```")

        lines.append("")

    return "\n".join(lines)


def format_security_context(checks: list[dict]) -> str:
    """
    Produce the security remediation block for the LLM prompt.

    No filtering is applied – all 723 checks in the CSV are public AVD data.
    The only data cleaning is stripping HTML comment placeholders from the
    impact field (AVD scaffold: <!-- Add Impact here -->).
    """
    if not checks:
        return "No security constraints found for this query."

    lines = []
    for chk in checks:
        severity = chk.get("severity") or "UNKNOWN"
        lines.append(
            f"[{chk['check_id']}] {chk['check_name']} (Severity: {severity})"
        )
        if chk.get("description"):
            lines.append(f"Description: {chk['description']}")

        # Clean and emit impact only if non-empty after stripping placeholders
        impact = clean_impact(chk.get("impact"))
        if impact:
            lines.append(f"Impact: {impact}")

        # CFN remediations only
        cfn_rems = [
            r["instruction"] for r in (chk.get("remediations") or [])
            if r.get("framework") == "cfn" and r.get("instruction")
        ]
        if cfn_rems:
            lines.append("CloudFormation Remediation:")
            for rem in cfn_rems:
                lines.append(f"  - {rem}")

        # CFN good example
        cfn_exs = [
            e["code"] for e in (chk.get("examples") or [])
            if e.get("framework") == "cfn" and e.get("code")
        ]
        if cfn_exs:
            lines.append("CloudFormation Good Example:")
            lines.append(f"```yaml\n{cfn_exs[0]}\n```")

        if chk.get("avd_url"):
            lines.append(f"Reference: {chk['avd_url']}")

        lines.append("")

    return "\n".join(lines)


def format_combined_context(cfn_results: list[dict], security_results: list[dict]) -> str:
    return (
        "=== CloudFormation Schema Context ===\n"
        + format_cfn_context(cfn_results)
        + "\n=== Security Constraints & Remediations ===\n"
        + format_security_context(security_results)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_combined_context(query: str) -> str:
    """
    Main entry point for the IaC generation agent.

    Returns a structured context string ready to be injected into an LLM prompt.
    Raises RuntimeError if the CFN ChromaDB collection is missing/empty.
    """
    print(f"\n[G-Retrieval] Query: '{query}'")
    print("Stage 1: Semantic search in ChromaDB...")

    cfn_names = semantic_search_cfn(query)
    check_ids = semantic_search_security(query)

    print("Stage 2: Graph traversal in Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        cfn_results = query_cfn_subgraph(driver, cfn_names)
        security_results = query_security_subgraph(driver, check_ids)
    finally:
        driver.close()

    print(f"  CFN resources retrieved   : {len(cfn_results)}")
    print(f"  Security checks retrieved : {len(security_results)}")

    print("Stage 3: Formatting combined context...")
    return format_combined_context(cfn_results, security_results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run combined CFN + security G-Retrieval for a query."
    )
    parser.add_argument(
        "--query", "-q",
        default="Create a secure S3 bucket with encryption and versioning enabled",
        help="Natural language IaC generation query",
    )
    args = parser.parse_args()

    context = retrieve_combined_context(args.query)

    print("\n" + "=" * 70)
    print("COMBINED CONTEXT (will be injected into LLM prompt)")
    print("=" * 70)
    print(context)


if __name__ == "__main__":
    main()
