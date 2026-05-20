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
      Merges both stages into the single context block consumed by
      remediator_agent via _build_policy_source_context.

Dependency direction (strictly unidirectional, no cycles):
  remediator_agent  →  security_hybrid_rag  (no agent imports)
  security_hybrid_rag does NOT import from agents/

Input contract
--------------
The caller passes raw Trivy finding text (the full JSON-serialised error
strings that validators.py emits, or the formatted strings from
format_trivy_errors / _build_validation_errors_text). No pre-parsing is
required — check_ids are extracted here via regex so the caller can pass
the finding text as-is.

Fallback
--------
If ChromaDB or Neo4j are unavailable the function degrades gracefully to
the CSV-backed get_trivy_policy_context() already used by remediator.py.
This preserves the existing behaviour for deployments that have not yet
built the security graph.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import chromadb
from langchain_chroma import Chroma
from neo4j import GraphDatabase

from tools.embedding_provider import get_embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    """Extract Trivy/AVD check-IDs from raw finding text strings."""
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


@contextmanager
def _neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Stage 1 — Semantic search: finding text → check_ids
# ---------------------------------------------------------------------------

def _security_semantic_search(
    finding_texts: list[str],
    k_per_query: int = 5,
) -> list[str]:
    """Run ChromaDB similarity search using raw finding texts as queries."""
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
            embedding_function=get_embeddings(),
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
    """Fetch one SecurityCheck subgraph row from Neo4j."""
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
    """Format a single SecurityCheck graph row into a prompt-ready block."""
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
            break

    rego = (chk.get("rego_code") or "").strip()
    if rego:
        lines.append(f"Policy (Rego):\n```rego\n{rego}\n```")

    return "\n".join(lines)


def _assemble_security_context(
    checks: list[dict[str, Any]],
    char_budget: int = _SECURITY_CHAR_BUDGET,
) -> str:
    """Sort by severity, apply char budget, return formatted context block."""
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

def execute_security_retrieval(finding_texts: list[str]) -> str:
    """Execute full security G-Retrieval: ChromaDB → Neo4j → formatted context."""
    if not finding_texts:
        return ""

    print(f"[SecurityRAG] Retrieval for {len(finding_texts)} finding(s).")

    seed_ids = extract_trivy_check_ids(finding_texts)
    semantic_ids = _security_semantic_search(finding_texts)

    merged: OrderedDict[str, None] = OrderedDict()
    for cid in seed_ids + semantic_ids:
        norm = cid.strip().upper()
        if norm:
            merged[norm] = None

    if not merged:
        print("[SecurityRAG] No check_ids identified. Returning empty context.")
        return ""

    print(
        f"[SecurityRAG] check_ids: {len(seed_ids)} seeded, "
        f"{len(semantic_ids)} from semantic search, "
        f"{len(merged)} total unique."
    )

    checks = _security_graph_lookup(list(merged.keys()))
    if not checks:
        return ""

    context = _assemble_security_context(checks)
    print(f"[SecurityRAG] Context assembled: {len(context)} chars.")
    return context
