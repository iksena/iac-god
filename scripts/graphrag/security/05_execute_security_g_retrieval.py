#!/usr/bin/env python3
"""
05_execute_security_g_retrieval.py

Stage 3 – Combined G-Retrieval: CFN schema + security remediation context.

This is the retrieval layer that feeds the LLM in the IaC generation agent.
It extends execute_g_retrieval.py (CFN-only) with a dual-collection fan-out
that simultaneously queries:
  1. ChromaDB 'cfn_resources'   → CFN schema subgraph from Neo4j
  2. ChromaDB 'security_checks' → security remediation subgraph from Neo4j

The two subgraphs are merged into a single structured prompt context.

Retrieval flow
--------------
  User query
      ├─ ChromaDB 'cfn_resources'   (k=5) → resource_names
      └─ ChromaDB 'security_checks'  (k=5) → check_ids
           │
           ▼
      Neo4j pass A: query_cfn_subgraph(resource_names)
           │  MATCH (r:Resource) with full property + example subgraph
           ▼
      Neo4j pass B: query_security_subgraph(check_ids)
           │  MATCH (s:SecurityCheck) with impact, remediation, examples
           │  + follow APPLIES_TO_RESOURCE to enrich with linked CFN resources
           ▼
      format_combined_context(cfn_result, security_result)
           │  Produces a structured prompt string with two labelled blocks
           ▼
      Caller injects into LLM prompt

Usage (standalone test)
-----------------------
    python scripts/graphrag/security/05_execute_security_g_retrieval.py \\
        --query "S3 bucket with encryption and no public access"

Usage (as module)
-----------------
    from scripts.graphrag.security.execute_security_g_retrieval import (
        retrieve_combined_context
    )
    context = retrieve_combined_context("create a secure RDS instance")
    # Pass context into your LLM prompt builder

Environment variables
---------------------
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    CHROMA_PERSIST_DIR   (default: same as CFN pipeline docker volume path)

Dependencies: neo4j, langchain-huggingface, langchain-chroma, chromadb
"""

import argparse
import os
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
# Config  (mirrors execute_g_retrieval.py + 03_build_security_chromadb.py)
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path("/Users/iksena/Documents/research/cfn-chroma-docker") / "chroma_data"),
)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CFN_COLLECTION = "cfn_resources"
SECURITY_COLLECTION = "security_checks"
SEMANTIC_K = 5  # results from each ChromaDB collection

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
# Stage 1a: Semantic search – CFN collection
# ---------------------------------------------------------------------------

def semantic_search_cfn(query: str, k: int = SEMANTIC_K) -> list[str]:
    """
    Returns a deduplicated list of CFN resource names relevant to the query.
    e.g. ['AWS::S3::Bucket', 'AWS::S3::BucketPolicy']
    """
    print(f"  [CFN ChromaDB] Semantic search: '{query[:60]}'")
    vectorstore = Chroma(
        collection_name=CFN_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
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

def semantic_search_security(query: str, k: int = SEMANTIC_K) -> list[str]:
    """
    Returns a deduplicated list of check_ids relevant to the query.
    e.g. ['AVD-AWS-0173', 'AVD-AWS-0089']
    """
    print(f"  [Security ChromaDB] Semantic search: '{query[:60]}'")
    vectorstore = Chroma(
        collection_name=SECURITY_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    results = vectorstore.similarity_search(query, k=k)
    check_ids: list[str] = []
    seen: set[str] = set()
    for doc in results:
        cid = doc.metadata.get("check_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            check_ids.append(cid)
    print(f"    → {check_ids}")
    return check_ids


# ---------------------------------------------------------------------------
# Stage 2a: Neo4j graph traversal – CFN schema subgraph
#           Mirrors query_knowledge_graph() in neo4j_client.py
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
#           Also follows APPLIES_TO_RESOURCE for cross-graph enrichment
# ---------------------------------------------------------------------------

SECURITY_CYPHER = """
    MATCH (s:SecurityCheck {check_id: $check_id})

    OPTIONAL MATCH (s)-[:AFFECTS_SERVICE]->(svc:AwsService)
    OPTIONAL MATCH (s)-[:HAS_IMPACT]->(imp:Impact)
    OPTIONAL MATCH (s)-[:HAS_REMEDIATION]->(rem:Remediation)
    OPTIONAL MATCH (s)-[:HAS_GOOD_EXAMPLE]->(ex:GoodExample)
    OPTIONAL MATCH (s)-[:ENFORCED_BY]->(rp:RegoPolicy)

    // Cross-graph: CFN resource names this check applies to
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
#           Mirrors format_prompt_from_neo4j_result() in neo4j_client.py
# ---------------------------------------------------------------------------

def format_cfn_context(resources: list[dict]) -> str:
    """Produce the CloudFormation schema block for the LLM prompt."""
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
            for p in opt[:10]:  # cap to keep prompt size manageable
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
    """Produce the security remediation block for the LLM prompt."""
    if not checks:
        return "No security constraints found for this query."

    lines = []
    for chk in checks:
        lines.append(
            f"[{chk['check_id']}] {chk['check_name']} "
            f"(Severity: {chk.get('severity', 'UNKNOWN')})"
        )
        if chk.get("description"):
            lines.append(f"Description: {chk['description']}")
        if chk.get("impact"):
            lines.append(f"Impact: {chk['impact']}")

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
    """Combine both context blocks into the final prompt string."""
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
    Covers both CloudFormation schema and security remediation constraints.

    Example
    -------
    >>> context = retrieve_combined_context("secure S3 bucket with versioning")
    >>> prompt = f"{context}\n\nUSER QUERY: {query}\n\nGenerate CloudFormation YAML:"
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
