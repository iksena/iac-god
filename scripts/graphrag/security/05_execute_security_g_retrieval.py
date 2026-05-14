#!/usr/bin/env python3
"""
05_execute_security_g_retrieval.py

Stage 3 – Combined G-Retrieval: CFN schema + security remediation context.

Both ChromaDB collections live in the same Docker container:
  'cfn_schema_properties'  – built by scripts/graphrag/05_build_chromadb.py
  'security_checks'        – built by scripts/graphrag/security/03_build_security_chromadb.py

Retrieval flow
--------------
  User query
      ├─ ChromaDB 'cfn_schema_properties'  (k=5)            → candidate_resource_names
      └─ ChromaDB 'security_checks'         (k=10 + boost)  → check_ids
           │
           │  Cross-graph re-rank (Stage 2a):
           │  security check_ids → Neo4j APPLIES_TO_RESOURCE → resource_names
           │  Merge with candidate_resource_names, linked resources ranked first.
           │
           ▼
      Neo4j pass A: query_cfn_subgraph(ranked_resource_names)
      Neo4j pass B: query_security_subgraph(check_ids)
           ▼
      format_combined_context(cfn_result, security_result)
           ▼
      Token-budgeted output (security checks sorted HIGH/CRITICAL > MEDIUM > LOW > UNKNOWN)

Design notes
------------
Cross-graph re-rank:
  The CFN vector search embeds per-property, so "S3 bucket" matches any
  resource with "S3" in the name (VectorBucket, DirectoryBucket, TableBucket,
  etc.). The correct resource AWS::S3::Bucket has APPLIES_TO_RESOURCE edges
  from the retrieved security checks; the noise resources do not. We use this
  graph signal to rank linked resources first and cap at CFN_RESOURCE_LIMIT.

Severity ordering:
  Security checks are sorted CRITICAL > HIGH > MEDIUM > LOW > UNKNOWN before
  formatting. UNKNOWN-severity checks (non-standard AVD IDs) are included but
  placed last. A token budget (SECURITY_TOKEN_BUDGET) caps the total security
  context so low-signal UNKNOWN checks are dropped if the budget is exceeded.

Environment variables
---------------------
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    CHROMA_HOST   (default: localhost)
    CHROMA_PORT   (default: 8000)

Usage
-----
    python scripts/graphrag/security/05_execute_security_g_retrieval.py \\
        --query "S3 bucket with encryption and no public access"
"""

import argparse
import os
import re
import sys
from typing import Any

import chromadb

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
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CFN_COLLECTION      = "cfn_schema_properties"
SECURITY_COLLECTION = "security_checks"

SEMANTIC_K     = 5   # CFN candidate resources from vector search
SEMANTIC_K_SEC = 10  # security checks from vector search
SEMANTIC_K_MAX = 15  # ceiling after service-hint boost

# After cross-graph re-rank, include at most this many CFN resources in context.
# Keeps the prompt focused; linked resources always appear first.
CFN_RESOURCE_LIMIT = 2

# Approximate token budget for the security context block.
# avg token ~ 4 chars; 3000 tokens ~ 12 000 chars is a reasonable slice of a
# 16k-token context window, leaving room for the CFN schema + prompt.
SECURITY_TOKEN_BUDGET = 12_000  # characters

# Severity sort order: lower number = higher priority
_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}

_AWS_SERVICE_TOKENS = {
    "s3", "ec2", "rds", "iam", "lambda", "cloudtrail", "cloudfront",
    "apigateway", "api", "dynamodb", "elasticache", "eks", "ecs",
    "kinesis", "sqs", "sns", "kms", "vpc", "elb", "elbv2", "msk",
    "athena", "glue", "redshift", "emr", "sagemaker", "codebuild",
    "codecommit", "secretsmanager", "ssm", "config",
}

_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)


def _clean_impact(raw: str | None) -> str:
    if not raw:
        return ""
    return _HTML_COMMENT_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Shared embedding model (lazy-init)
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


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


# ---------------------------------------------------------------------------
# Stage 1a: CFN semantic search → candidate resource names
# ---------------------------------------------------------------------------

def semantic_search_cfn(query: str, k: int = SEMANTIC_K) -> list[str]:
    """Returns up to k unique CFN resource names ordered by vector similarity."""
    print(f"  [CFN ChromaDB] Semantic search: '{query[:60]}'")
    client = get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    if CFN_COLLECTION not in existing:
        raise RuntimeError(
            f"\nERROR: CFN collection '{CFN_COLLECTION}' not found in ChromaDB Docker "
            f"({CHROMA_HOST}:{CHROMA_PORT}).\n"
            "Fix: Run scripts/graphrag/05_build_chromadb.py first.\n"
            f"     Existing collections: {existing}"
        )

    vectorstore = Chroma(
        collection_name=CFN_COLLECTION,
        embedding_function=get_embeddings(),
        client=client,
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
# Stage 1b: Security semantic search → check_ids (two-pass with service boost)
# ---------------------------------------------------------------------------

def _extract_service_hint(query: str) -> str | None:
    tokens = set(re.findall(r'[a-z0-9]+', query.lower()))
    hits = tokens & _AWS_SERVICE_TOKENS
    return max(hits, key=len) if hits else None


def semantic_search_security(
    query: str,
    k: int = SEMANTIC_K_SEC,
    k_max: int = SEMANTIC_K_MAX,
) -> list[str]:
    """Returns check_ids ordered by relevance (vector similarity + service boost)."""
    print(f"  [Security ChromaDB] Semantic search: '{query[:60]}'")
    client = get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    if SECURITY_COLLECTION not in existing:
        print(f"  WARNING: Security collection '{SECURITY_COLLECTION}' not found.")
        print("  Run scripts/graphrag/security/03_build_security_chromadb.py first.")
        return []

    vectorstore = Chroma(
        collection_name=SECURITY_COLLECTION,
        embedding_function=get_embeddings(),
        client=client,
    )

    results = vectorstore.similarity_search(query, k=k)
    check_ids: list[str] = []
    seen: set[str] = set()
    for doc in results:
        cid = doc.metadata.get("check_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            check_ids.append(cid)

    service_hint = _extract_service_hint(query)
    if service_hint and len(check_ids) < k_max:
        boost_results = vectorstore.similarity_search(
            service_hint, k=(k_max - len(check_ids))
        )
        added = 0
        for doc in boost_results:
            cid = doc.metadata.get("check_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                check_ids.append(cid)
                added += 1
        if added:
            print(f"    (service hint '{service_hint}' added {added} more checks)")

    print(f"    → {check_ids}")
    return check_ids


# ---------------------------------------------------------------------------
# Stage 2a: Cross-graph CFN re-rank
# ---------------------------------------------------------------------------

_CFN_LINKED_RESOURCES_CYPHER = """
UNWIND $check_ids AS cid
MATCH (s:SecurityCheck {check_id: cid})-[:APPLIES_TO_RESOURCE]->(r:Resource)
RETURN DISTINCT r.name AS resource_name
"""


def rerank_cfn_resources(
    driver,
    candidate_names: list[str],
    check_ids: list[str],
    limit: int = CFN_RESOURCE_LIMIT,
) -> list[str]:
    """
    Re-rank CFN resource candidates using the cross-graph signal:
      resources that have APPLIES_TO_RESOURCE edges from the retrieved
      security checks are ranked first (graph-confirmed relevance).
      Remaining candidates fill up to `limit` from the original vector order.

    This solves the false-positive problem where the CFN vector search returns
    AWS::S3Vectors::VectorBucket alongside AWS::S3::Bucket because both share
    the 'S3' token, but only AWS::S3::Bucket has security check edges.
    """
    if not check_ids:
        return candidate_names[:limit]

    with driver.session() as session:
        result = session.run(_CFN_LINKED_RESOURCES_CYPHER, check_ids=check_ids)
        linked = {row["resource_name"] for row in result}

    ranked: list[str] = []
    # Linked resources first, preserving original vector order within the group
    for name in candidate_names:
        if name in linked:
            ranked.append(name)
    # Fill remaining slots from unlinked candidates in vector order
    for name in candidate_names:
        if name not in linked and len(ranked) < limit:
            ranked.append(name)

    final = ranked[:limit]
    dropped = [n for n in candidate_names if n not in final]
    if dropped:
        print(f"  [Re-rank] Dropped {len(dropped)} unlinked CFN resources: {dropped}")
    print(f"  [Re-rank] CFN resources after cross-graph re-rank: {final}")
    return final


# ---------------------------------------------------------------------------
# Stage 2b: Neo4j – CFN schema subgraph
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
# Stage 2c: Neo4j – security subgraph
# ---------------------------------------------------------------------------

SECURITY_CYPHER = """
    MATCH (s:SecurityCheck {check_id: $check_id})

    OPTIONAL MATCH (s)-[:AFFECTS_SERVICE]->(svc:AwsService)
    OPTIONAL MATCH (s)-[:HAS_IMPACT]->(imp:Impact)
    OPTIONAL MATCH (s)-[:HAS_REMEDIATION]->(rem:Remediation)
    OPTIONAL MATCH (s)-[:HAS_GOOD_EXAMPLE]->(ex:GoodExample)
    OPTIONAL MATCH (s)-[:ENFORCED_BY]->(rp:RegoPolicy)
    OPTIONAL MATCH (s)-[:APPLIES_TO_RESOURCE]->(cfn_r:Resource)

    RETURN s.check_id       AS check_id,
           s.check_name     AS check_name,
           s.severity       AS severity,
           s.description    AS description,
           svc.name         AS service,
           imp.text         AS impact,
           collect(DISTINCT {framework: rem.framework,
                             instruction: rem.instruction}) AS remediations,
           collect(DISTINCT {framework: ex.framework,
                             code: ex.code})               AS examples,
           rp.code                                         AS rego_code,
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
                    "check_id":          row["check_id"],
                    "check_name":        row["check_name"],
                    "severity":          row["severity"],
                    "description":       row["description"],
                    "service":           row["service"],
                    "impact":            row["impact"],
                    "remediations":      row["remediations"],
                    "examples":          row["examples"],
                    "rego_code":         row["rego_code"],
                    "rego_source_url":   row["rego_source_url"],
                    "cfn_resource_names": row["cfn_resource_names"],
                })
    return results


# ---------------------------------------------------------------------------
# Stage 3: Format combined context
# ---------------------------------------------------------------------------

def format_cfn_context(resources: list[dict]) -> str:
    if not resources:
        return "No CloudFormation schema context found."
    lines = []
    for res in resources:
        lines.append(f"Resource: {res['name']}")
        desc = (res.get("description") or "").strip()
        if desc:
            lines.append(f"Description: {desc}")
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
            lines.append(f"YAML Example:\n```yaml\n{ex['code']}\n```")
        lines.append("")
    return "\n".join(lines)


def _sort_checks_by_severity(checks: list[dict]) -> list[dict]:
    """Sort security checks CRITICAL > HIGH > MEDIUM > LOW > UNKNOWN."""
    return sorted(
        checks,
        key=lambda c: _SEVERITY_RANK.get((c.get("severity") or "UNKNOWN").upper(), 4),
    )


def _format_one_check(chk: dict) -> str:
    """Format a single security check for the LLM prompt context block."""
    lines = []
    severity = (chk.get("severity") or "UNKNOWN").upper()
    lines.append(f"[{chk['check_id']}] {chk['check_name']} (Severity: {severity})")

    desc = (chk.get("description") or "").strip()
    if desc:
        lines.append(f"Description: {desc}")

    impact = _clean_impact(chk.get("impact"))
    if impact:
        lines.append(f"Impact: {impact}")

    cfn_rems = [
        r["instruction"] for r in (chk.get("remediations") or [])
        if r.get("framework") in ("cfn", "cloudformation") and r.get("instruction")
    ]
    if cfn_rems:
        lines.append("CloudFormation Remediation:")
        for rem in cfn_rems:
            lines.append(f"  - {rem}")

    cfn_exs = [
        e["code"] for e in (chk.get("examples") or [])
        if e.get("framework") in ("cfn", "cloudformation") and e.get("code")
    ]
    if cfn_exs:
        lines.append(f"CloudFormation Good Example:\n```yaml\n{cfn_exs[0]}\n```")

    # Rego policy source URL is more useful than avd_url as a research reference
    # because it points directly to the machine-readable policy that Trivy enforces.
    rego_url = (chk.get("rego_source_url") or "").strip()
    if rego_url:
        lines.append(f"Policy Source: {rego_url}")

    lines.append("")
    return "\n".join(lines)


def format_security_context(
    checks: list[dict],
    token_budget: int = SECURITY_TOKEN_BUDGET,
) -> str:
    """
    Format security remediation block for the LLM prompt.

    Checks are sorted CRITICAL > HIGH > MEDIUM > LOW > UNKNOWN.
    The formatted output is truncated to `token_budget` characters so that
    low-signal UNKNOWN-severity checks (e.g., lifecycle config, transfer
    acceleration) are dropped before they consume prompt context.
    """
    if not checks:
        return "No security constraints found for this query."

    sorted_checks = _sort_checks_by_severity(checks)
    lines = []
    total_chars = 0
    included = 0
    skipped_unknown = 0

    for chk in sorted_checks:
        block = _format_one_check(chk)
        if total_chars + len(block) > token_budget:
            severity = (chk.get("severity") or "UNKNOWN").upper()
            if severity == "UNKNOWN":
                skipped_unknown += 1
                continue  # drop UNKNOWN checks that exceed budget
            # For non-UNKNOWN checks we still include but warn
            print(
                f"  WARNING: token budget exceeded at check {chk['check_id']} "
                f"(severity={severity}). Truncating security context."
            )
            break
        lines.append(block)
        total_chars += len(block)
        included += 1

    if skipped_unknown:
        print(
            f"  [Context] Dropped {skipped_unknown} UNKNOWN-severity checks "
            f"(token budget). Included {included}/{len(checks)} checks."
        )
    else:
        print(f"  [Context] Included {included}/{len(checks)} security checks.")

    return "\n".join(lines) if lines else "No security constraints found for this query."


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
    print(f"\n[G-Retrieval] Query: '{query}'")
    print("Stage 1: Semantic search in ChromaDB...")
    candidate_cfn_names = semantic_search_cfn(query)
    check_ids = semantic_search_security(query)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        print("Stage 2: Graph traversal in Neo4j...")

        # Cross-graph re-rank: keep only CFN resources that have
        # APPLIES_TO_RESOURCE edges from the retrieved security checks.
        # Linked resources are ranked first; unlinked fill remaining slots.
        ranked_cfn_names = rerank_cfn_resources(driver, candidate_cfn_names, check_ids)

        cfn_results      = query_cfn_subgraph(driver, ranked_cfn_names)
        security_results = query_security_subgraph(driver, check_ids)
    finally:
        driver.close()

    print(f"  CFN resources retrieved   : {len(cfn_results)}")
    print(f"  Security checks retrieved : {len(security_results)}")

    print("Stage 3: Formatting combined context...")
    return format_combined_context(cfn_results, security_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run combined CFN + security G-Retrieval."
    )
    parser.add_argument(
        "--query", "-q",
        default="Create a secure S3 bucket with encryption and versioning enabled",
    )
    args = parser.parse_args()
    context = retrieve_combined_context(args.query)
    print("\n" + "=" * 70)
    print("COMBINED CONTEXT (will be injected into LLM prompt)")
    print("=" * 70)
    print(context)


if __name__ == "__main__":
    main()
