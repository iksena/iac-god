"""tools/security_hybrid_rag.py

ChromaDB + Neo4j hybrid retrieval for security remediation context.

Mirrors the structure of tools/cfn_hybrid_rag.py but targets the
'security_checks' ChromaDB collection and the SecurityCheck subgraph in
Neo4j, which were built by scripts/graphrag/security/.

Retrieval runs in two sequential stages:
  Stage 1 — _security_semantic_search():
      ChromaDB similarity search over pre-indexed security check chunks.
      Only chunks at or below SECURITY_DISTANCE_THRESHOLD are kept.
      Results are de-duplicated by check_id.

  Stage 2 — _security_graph_lookup():
      Neo4j traversal per check_id. Returns description, impact,
      CloudFormation remediation text, CFN good examples, and the Rego
      policy source code. URLs (avd_url, rego_source_url, links) are
      intentionally excluded — the context is designed for LLM
      consumption, not human browsing.

  Final — execute_security_retrieval():
      Merges exact-match and semantic results into the single security
      context block consumed by the retriever / remediator.

Dependency direction (strictly unidirectional, no cycles):
  remediator_agent  →  security_hybrid_rag  (no agent imports)
  security_hybrid_rag does NOT import from agents/

Input contract
--------------
The caller passes LLM-generated retrieval queries. If a query explicitly
contains an AVD/Trivy check ID, the function routes it to the deterministic
Neo4j lookup path first. Otherwise it performs semantic search against the
security ChromaDB collection and re-ranks the matching graph rows.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

SECURITY_COLLECTION = "security_checks"

# Raw cosine-distance floor for the security_checks collection.
# Lower = more similar. 0.55 matches the CFN collection baseline.
# Tune DOWNWARD for higher precision, UPWARD for higher recall.
SECURITY_DISTANCE_THRESHOLD: float = float(
    os.getenv("SECURITY_DISTANCE_THRESHOLD", "0.55")
)

# Sort order: lower = higher priority in the formatted context block.
_SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4
}

# Approximate character budget for the full security context block.
# ~3 000 tokens at 4 chars/token; leaves headroom in a 16k context window
# for the CFN schema block and the prompt frame.
_SECURITY_CHAR_BUDGET: int = int(os.getenv("SECURITY_CHAR_BUDGET", "12000"))

# Patterns used to extract Trivy check IDs from raw finding text
_AVD_ID_RE   = re.compile(r"\b(?:AVD-)?AWS-\d{4}\b", re.IGNORECASE)
_BRACKET_RE  = re.compile(r"\[([A-Z0-9_-]+)\]")
_HTML_CMNT   = re.compile(r"<!--.*?-->", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_html_comments(text: str | None) -> str:
    if not text:
        return ""
    return _HTML_CMNT.sub("", text).strip()


def extract_trivy_check_ids(finding_texts: list[str]) -> list[str]:
    """Extract Trivy/AVD check-IDs from raw finding text strings.

    Accepts the same variety of formats that validators.py produces:
      - '[AVD-AWS-0088] ...'   (bracket-wrapped, AVD prefix)
      - 'AVD-AWS-0132: ...'    (plain AVD-AWS-####)
      - 'AWS-0090 ...'         (short form without AVD prefix)

    Returns a de-duplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for text in finding_texts:
        for match in _BRACKET_RE.findall(text):
            cid = match.strip().upper()
            if re.fullmatch(r"(?:AVD-)?AWS-\d{4}", cid) and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        for match in _AVD_ID_RE.findall(text):
            cid = match.strip().upper()
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _id_variants(check_id: str) -> list[str]:
    """Return both AVD-AWS-XXXX and AWS-XXXX forms for a given check ID."""
    base = check_id.strip().upper()
    if re.fullmatch(r"AWS-\d{4}", base):
        return [base, f"AVD-{base}"]
    if re.fullmatch(r"AVD-AWS-\d{4}", base):
        return [base, base.removeprefix("AVD-")]
    return [base]


# ---------------------------------------------------------------------------
# Shared embedding model (lazy, process-scoped)
# ---------------------------------------------------------------------------
_embeddings: HuggingFaceEmbeddings | None = None


@lru_cache(maxsize=1)
def _get_global_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@contextmanager
def _neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Stage 1 — Query routing: explicit IDs → exact lookup, free text → semantic search
# ---------------------------------------------------------------------------

def _security_semantic_search(
    finding_texts: list[str],
    k_per_query: int = 5,
) -> list[str]:
    """Run ChromaDB similarity search using free-text retrieval queries.

    Each query string is used as its own search query — LLM-generated
    prompts such as 'S3 bucket encryption' or 'AWS::S3::Bucket public access'
    embed well without pre-processing.

    Returns a de-duplicated list of check_ids ordered by first-seen
    (similarity-descending within each query).
    """
    seen: OrderedDict[str, None] = OrderedDict()

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        existing = [c.name for c in client.list_collections()]
        if SECURITY_COLLECTION not in existing:
            print(
                f"[SecurityRAG] Stage 1: collection '{SECURITY_COLLECTION}' not found. "
                "Falling back to CSV lookup."
            )
            return []

        vectorstore = Chroma(
            collection_name=SECURITY_COLLECTION,
            embedding_function=_get_global_embeddings(),
            client=client,
        )

        kept = dropped = 0
        for query in finding_texts:
            scored = vectorstore.similarity_search_with_score(query, k=k_per_query)
            for doc, score in scored:
                if score > SECURITY_DISTANCE_THRESHOLD:
                    dropped += 1
                    continue
                cid = (doc.metadata.get("check_id") or "").strip().upper()
                if cid and cid not in seen:
                    seen[cid] = None
                    kept += 1

        print(
            f"[SecurityRAG] Stage 1: {kept + dropped} chunks retrieved, "
            f"{dropped} dropped (distance > {SECURITY_DISTANCE_THRESHOLD}), "
            f"{kept} kept ({len(seen)} unique check_ids)."
        )

    except Exception as exc:
        print(f"[SecurityRAG] Stage 1 warning: ChromaDB unavailable. {exc}")

    return list(seen.keys())


# ---------------------------------------------------------------------------
# Stage 2 — Graph lookup: check_ids → full security subgraph
# ---------------------------------------------------------------------------

_SECURITY_CYPHER = """
MATCH (s:SecurityCheck {check_id: $check_id})

OPTIONAL MATCH (s)-[:HAS_IMPACT]->(imp:Impact)
OPTIONAL MATCH (s)-[:HAS_REMEDIATION]->(rem:Remediation)
    WHERE rem.framework IN ['cfn', 'cloudformation']
OPTIONAL MATCH (s)-[:HAS_GOOD_EXAMPLE]->(ex:GoodExample)
    WHERE ex.framework  IN ['cfn', 'cloudformation']
OPTIONAL MATCH (s)-[:ENFORCED_BY]->(rp:RegoPolicy)

RETURN s.check_id   AS check_id,
       s.check_name AS check_name,
       s.severity   AS severity,
       s.description AS description,
       imp.text     AS impact,
       collect(DISTINCT rem.instruction) AS cfn_remediations,
       collect(DISTINCT ex.code)         AS cfn_examples,
       rp.code                           AS rego_code
"""


def _query_security_check(driver, check_id: str) -> dict[str, Any] | None:
    """Fetch one SecurityCheck subgraph row from Neo4j.

    Tries all ID variants (AVD-AWS-XXXX and AWS-XXXX) so callers never need
    to normalise the ID form before calling.
    """
    with driver.session() as session:
        for variant in _id_variants(check_id):
            row = session.run(_SECURITY_CYPHER, check_id=variant).single()
            if row:
                return {
                    "check_id":        row["check_id"],
                    "check_name":      row["check_name"],
                    "severity":        row["severity"],
                    "description":     row["description"],
                    "impact":          row["impact"],
                    "cfn_remediations": [r for r in (row["cfn_remediations"] or []) if r],
                    "cfn_examples":     [e for e in (row["cfn_examples"] or []) if e],
                    "rego_code":        row["rego_code"],
                }
    return None


def _security_graph_lookup(check_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch full subgraph rows for each check_id from Neo4j."""
    results: list[dict] = []
    if not check_ids:
        return results
    print(f"[SecurityRAG] Stage 2: querying Neo4j for {len(check_ids)} checks...")
    try:
        with _neo4j_driver() as driver:
            seen: set[str] = set()
            for cid in check_ids:
                norm = cid.strip().upper()
                if norm in seen:
                    continue
                seen.add(norm)
                row = _query_security_check(driver, norm)
                if row:
                    results.append(row)
    except Exception as exc:
        print(f"[SecurityRAG] Stage 2 warning: Neo4j unavailable. {exc}")
    print(f"[SecurityRAG] Stage 2: {len(results)}/{len(check_ids)} checks retrieved from graph.")
    return results


# ---------------------------------------------------------------------------
# Format — build LLM-ready context block (no URLs)
# ---------------------------------------------------------------------------

def _format_one_check(chk: dict[str, Any]) -> str:
    """Format a single SecurityCheck graph row into a prompt-ready block.

    Design rules:
      - No URLs of any kind (avd_url, rego_source_url, links) — this context
        goes directly to the LLM and URLs add noise without semantic value.
      - Fields: check_id, check_name, severity, description, impact,
        CFN remediation instructions, CFN good examples, Rego source code.
      - Rego code is included because it contains the exact condition the
        policy enforces, which guides the LLM to produce a compliant fix.
    """
    lines: list[str] = []
    severity = (chk.get("severity") or "UNKNOWN").upper()
    name     = chk.get("check_name") or chk.get("check_id", "")
    lines.append(f"### [{chk['check_id']}] {name} (Severity: {severity})")

    desc = (chk.get("description") or "").strip()
    if desc:
        lines.append(f"Description: {desc}")

    impact = _clean_html_comments(chk.get("impact"))
    if impact:
        lines.append(f"Impact: {impact}")

    for rem in chk.get("cfn_remediations") or []:
        rem = rem.strip()
        if rem:
            lines.append(f"Remediation (CloudFormation): {rem}")

    for code in chk.get("cfn_examples") or []:
        code = code.strip()
        if code:
            lines.append(f"CloudFormation Good Example:\n```yaml\n{code}\n```")
            break  # one example is enough to guide the LLM

    rego = (chk.get("rego_code") or "").strip()
    if rego:
        lines.append(f"Policy (Rego):\n```rego\n{rego}\n```")

    return "\n".join(lines)


def _assemble_security_context(
    checks: list[dict[str, Any]],
    char_budget: int = _SECURITY_CHAR_BUDGET,
) -> str:
    """Sort by severity, apply char budget, return formatted context block.

    CRITICAL/HIGH/MEDIUM/LOW checks are always included if budget allows.
    UNKNOWN-severity checks (non-standard AVD IDs) are appended last and
    silently dropped when the budget is exceeded.
    """
    if not checks:
        return ""

    sorted_checks = sorted(
        checks,
        key=lambda c: _SEVERITY_RANK.get((c.get("severity") or "UNKNOWN").upper(), 4),
    )

    included_blocks: list[str] = []
    total_chars = 0
    skipped_unknown = 0

    for chk in sorted_checks:
        block = _format_one_check(chk)
        severity = (chk.get("severity") or "UNKNOWN").upper()
        if total_chars + len(block) > char_budget:
            if severity == "UNKNOWN":
                skipped_unknown += 1
                continue
            print(
                f"[SecurityRAG] Budget exceeded at {chk['check_id']} "
                f"(severity={severity}). Truncating."
            )
            break
        included_blocks.append(block)
        total_chars += len(block)

    if skipped_unknown:
        print(
            f"[SecurityRAG] Dropped {skipped_unknown} UNKNOWN-severity checks "
            f"(char budget). Included {len(included_blocks)}/{len(checks)} checks."
        )
    else:
        print(f"[SecurityRAG] Included {len(included_blocks)}/{len(checks)} checks.")

    return "\n\n".join(included_blocks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_security_retrieval(retrieval_queries: list[str]) -> str:
    """Execute full security G-Retrieval: ChromaDB → Neo4j → formatted context.

    Args:
        retrieval_queries: LLM-generated semantic queries, possibly including
            explicit AVD / Trivy IDs such as AVD-AWS-0086.

    Returns:
        A formatted context string suitable for injection into the retriever
        or remediator prompt.
    """
    if not retrieval_queries:
        return ""

    print(f"[SecurityRAG] Retrieval for {len(retrieval_queries)} query(ies).")

    # Stage 1A: deterministic exact-match path for queries with explicit IDs.
    exact_match_queries = [
        query for query in retrieval_queries
        if extract_trivy_check_ids([query])
    ]
    exact_ids = extract_trivy_check_ids(exact_match_queries)

    # Stage 1B: semantic search for the remaining free-text queries.
    semantic_queries = [
        query for query in retrieval_queries
        if not extract_trivy_check_ids([query])
    ]
    semantic_ids = _security_semantic_search(semantic_queries) if semantic_queries else []

    # Merge: exact IDs first (highest confidence), then semantic matches.
    merged: OrderedDict[str, None] = OrderedDict()
    for cid in exact_ids + semantic_ids:
        norm = cid.strip().upper()
        if norm:
            merged[norm] = None

    if not merged:
        print("[SecurityRAG] No check_ids identified. Returning empty context.")
        return ""

    print(
        f"[SecurityRAG] check_ids: {len(exact_ids)} exact, "
        f"{len(semantic_ids)} from semantic search, "
        f"{len(merged)} total unique."
    )

    # Stage 2: graph traversal per check_id.
    checks = _security_graph_lookup(list(merged.keys()))
    if not checks:
        return ""

    # Format and return.
    context = _assemble_security_context(checks)
    print(f"[SecurityRAG] Context assembled: {len(context)} chars.")
    return context
