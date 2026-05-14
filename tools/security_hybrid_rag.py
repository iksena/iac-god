"""tools/security_hybrid_rag.py

Hybrid GraphRAG retrieval for Trivy security-remediation context.

Design decisions vs. cfn_hybrid_rag.py
---------------------------------------
- NO retriever agent / no LLM query generation step.
  Trivy findings are already precise (check_id + message + severity) so
  the raw finding strings are used directly as embedding queries.  Injecting
  an extra LLM step to rewrite them would add latency without improving recall.

- Separate ChromaDB collection (``security_checks``) and separate Neo4j node
  labels (SecurityCheck, AwsService, Impact, Remediation, GoodExample,
  RegoPolicy).  The same Docker / bolt instances are reused.

- All URLs are stripped from the assembled context before it is returned.
  The remediator prompt must not contain AVD / docs links because they are
  noise that inflates token cost and can confuse the LLM into hallucinating
  non-existent pages.

- Graceful CSV fallback: when the GraphRAG stores are unavailable (first
  bootstrap, unit tests, CI) the function falls back to
  ``get_trivy_policy_context()`` from the existing trivy_context.py.

Retrieval flow
--------------
  findings  →  _build_queries_from_findings()
             →  _semantic_search()          (ChromaDB ``security_checks``)
             →  _graph_remediation_lookup() (Neo4j SecurityCheck subgraph)
             →  _assemble_security_context()
             →  caller (remediator_agent)

Dependency direction (no cycles):
  remediator_agent  →  security_hybrid_rag
  security_hybrid_rag  does NOT import from agents/
  security_hybrid_rag  falls back to tools/trivy_context.py
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

SECURITY_COLLECTION = "security_checks"

# Distance threshold — security embeddings are typically longer and more
# specific than CFN property chunks, so a slightly more permissive threshold
# is appropriate. Tune via SECURITY_CHROMA_DISTANCE_THRESHOLD env var.
CHROMA_DISTANCE_THRESHOLD: float = float(
    os.getenv("SECURITY_CHROMA_DISTANCE_THRESHOLD", "0.65")
)

# ---------------------------------------------------------------------------
# URL stripper — keep context clean, no external links in remediator prompt
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Embedding model (shared process-lifetime singleton with cfn_hybrid_rag)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_global_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# Stage 0 — derive embedding queries from raw Trivy findings
# ---------------------------------------------------------------------------

def _extract_finding_text(finding: Any) -> str:
    """Extract a human-readable string from a finding (dict, str, or object)."""
    if isinstance(finding, dict):
        parts: list[str] = []
        for key in ("check_id", "Title", "Message", "Description", "Severity"):
            val = str(finding.get(key) or "").strip()
            if val:
                parts.append(val)
        return " — ".join(parts)
    return str(finding).strip()


def _build_queries_from_findings(findings: list[Any]) -> list[str]:
    """Convert raw Trivy findings directly into embedding query strings.

    No LLM call is made here — the finding text is already information-dense
    enough to drive semantic search over security_checks embeddings.
    Each query is the full finding text with URLs stripped.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        text = _strip_urls(_extract_finding_text(finding))
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries


# ---------------------------------------------------------------------------
# Stage 1 — Semantic search (ChromaDB ``security_checks`` collection)
# ---------------------------------------------------------------------------

def _semantic_search(
    queries: list[str],
) -> tuple[dict[str, list[str]], set[str]]:
    """Run ChromaDB similarity search against the security_checks collection.

    Returns:
        chunks_by_check_id: check_id → list of cleaned text chunks.
        found_check_ids:    set of matched check_ids for Neo4j traversal.
    """
    chunks_by_check_id: dict[str, list[str]] = defaultdict(list)
    found_check_ids: set[str] = set()

    if not queries:
        return dict(chunks_by_check_id), found_check_ids

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        # Check collection exists before attempting search.
        existing = [c.name for c in chroma_client.list_collections()]
        if SECURITY_COLLECTION not in existing:
            print(
                f"[Security RAG] Collection '{SECURITY_COLLECTION}' not found. "
                "Run 03_build_security_chromadb.py first. Falling back to CSV."
            )
            return dict(chunks_by_check_id), found_check_ids

        vectorstore = Chroma(
            client=chroma_client,
            collection_name=SECURITY_COLLECTION,
            embedding_function=_get_global_embeddings(),
        )

        seen_chunk_keys: set[str] = set()
        kept = dropped = 0

        for query in queries:
            scored_chunks = vectorstore.similarity_search_with_score(query, k=3)
            for chunk, score in scored_chunks:
                if score > CHROMA_DISTANCE_THRESHOLD:
                    dropped += 1
                    continue

                meta = chunk.metadata
                check_id = str(meta.get("check_id") or "").strip().upper()
                chunk_key = (
                    f"{check_id}::{hash(chunk.page_content)}"
                    if check_id
                    else f"_unknown::{hash(chunk.page_content)}"
                )
                if chunk_key in seen_chunk_keys:
                    continue
                seen_chunk_keys.add(chunk_key)

                cleaned = _strip_urls(chunk.page_content)
                if cleaned:
                    chunks_by_check_id[check_id or "_unknown"].append(cleaned)
                    kept += 1
                if check_id:
                    found_check_ids.add(check_id)

        print(
            f"[Security RAG] Stage 1: {kept + dropped} chunks retrieved, "
            f"{dropped} dropped (distance > {CHROMA_DISTANCE_THRESHOLD}), "
            f"{kept} kept."
        )

    except Exception as exc:
        print(f"[Security RAG] Warning: ChromaDB semantic search failed — {exc}")

    return dict(chunks_by_check_id), found_check_ids


# ---------------------------------------------------------------------------
# Stage 2 — Graph traversal (Neo4j SecurityCheck subgraph)
# ---------------------------------------------------------------------------

_SECURITY_CYPHER = """
MATCH (sc:SecurityCheck {check_id: $check_id})

OPTIONAL MATCH (sc)-[:HAS_IMPACT]->(i:Impact)
OPTIONAL MATCH (sc)-[:HAS_REMEDIATION]->(rem:Remediation)
OPTIONAL MATCH (sc)-[:HAS_GOOD_EXAMPLE]->(ge:GoodExample)
OPTIONAL MATCH (sc)-[:ENFORCED_BY]->(rp:RegoPolicy)

RETURN
    sc.check_id        AS check_id,
    sc.check_name      AS check_name,
    sc.severity        AS severity,
    sc.description     AS description,
    collect(DISTINCT i.text)           AS impacts,
    collect(DISTINCT {framework: rem.framework, instruction: rem.instruction})
                                       AS remediations,
    collect(DISTINCT {framework: ge.framework, code: ge.code})
                                       AS good_examples,
    collect(DISTINCT rp.code)          AS rego_policies
"""


@contextmanager
def _neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


def _query_security_graph(driver, check_id: str) -> dict | None:
    with driver.session() as session:
        record = session.run(_SECURITY_CYPHER, check_id=check_id.upper()).single()
        if not record:
            return None
        return dict(record)


def _format_security_node(data: dict) -> str:
    """Render a SecurityCheck Neo4j record as a Markdown block.

    URLs are intentionally excluded — all link fields are omitted.
    Only description, impact, remediation instructions, CFN good examples,
    and the Rego policy code are included.
    """
    lines: list[str] = [
        f"### [{data['check_id']}] {data.get('check_name', '')} "
        f"({data.get('severity', 'UNKNOWN')})",
    ]

    desc = _strip_urls(str(data.get("description") or "").strip())
    if desc:
        lines.append(f"**Description:** {desc}")

    impacts = [_strip_urls(str(i).strip()) for i in (data.get("impacts") or []) if i]
    if impacts:
        lines.append(f"**Impact:** {'; '.join(impacts)}")

    for rem in (data.get("remediations") or []):
        fw   = str(rem.get("framework") or "").strip()
        inst = _strip_urls(str(rem.get("instruction") or "").strip())
        if inst:
            label = f"Remediation ({fw})" if fw else "Remediation"
            lines.append(f"**{label}:** {inst}")

    for ex in (data.get("good_examples") or []):
        fw   = str(ex.get("framework") or "").strip()
        code = str(ex.get("code") or "").strip()
        if code and fw.lower() == "cloudformation":
            lines.append(f"**CloudFormation Good Example:**\n```yaml\n{code}\n```")

    for policy_code in (data.get("rego_policies") or []):
        code = str(policy_code or "").strip()
        if code:
            lines.append(f"**Policy (Rego):**\n```rego\n{code}\n```")

    return "\n".join(lines)


def _graph_remediation_lookup(check_ids: set[str]) -> list[str]:
    """Fetch SecurityCheck subgraphs from Neo4j for all matched check_ids."""
    blocks: list[str] = []
    if not check_ids:
        return blocks

    print(f"[Security RAG] Stage 2: Querying Neo4j for {len(check_ids)} check(s)...")
    try:
        with _neo4j_driver() as driver:
            seen: set[str] = set()
            for check_id in sorted(check_ids):
                if check_id in seen:
                    continue
                seen.add(check_id)
                data = _query_security_graph(driver, check_id)
                if data:
                    blocks.append(_format_security_node(data))
    except Exception as exc:
        print(f"[Security RAG] Warning: Neo4j traversal failed — {exc}")

    return blocks


# ---------------------------------------------------------------------------
# Final — context assembly
# ---------------------------------------------------------------------------

def _assemble_security_context(
    chunks_by_check_id: dict[str, list[str]],
    graph_blocks: list[str],
) -> str:
    final_blocks: list[str] = ["## Security Remediation Context\n"]

    if graph_blocks:
        final_blocks.append(
            "### Security Check Details (GraphRAG)\n" + "\n\n".join(graph_blocks)
        )
    elif chunks_by_check_id:
        # Graph lookup returned nothing but Chroma had hits — use raw chunks.
        sections: list[str] = []
        for check_id, chunks in sorted(chunks_by_check_id.items()):
            header = f"#### [{check_id}]" if check_id != "_unknown" else "#### (unknown check)"
            sections.append(header + "\n" + "\n---\n".join(chunks))
        final_blocks.append(
            "### Semantically Matched Security Checks\n" + "\n\n".join(sections)
        )

    return "\n\n".join(final_blocks)


# ---------------------------------------------------------------------------
# CSV fallback (wraps existing trivy_context.get_trivy_policy_context)
# ---------------------------------------------------------------------------

def _csv_fallback(findings: list[Any]) -> str:
    """Fall back to CSV-based context when GraphRAG stores are unavailable."""
    try:
        from tools.trivy_context import get_trivy_policy_context  # noqa: PLC0415

        # trivy_context expects list[{"check_id": ...}] dicts.
        normalised = [
            {"check_id": _extract_finding_text(f).split(" — ")[0]}
            if not isinstance(f, dict)
            else f
            for f in findings
        ]
        raw = get_trivy_policy_context(normalised)
        return _strip_urls(raw) if raw else ""
    except Exception as exc:
        print(f"[Security RAG] CSV fallback failed — {exc}")
        return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_security_retrieval(findings: list[Any]) -> str:
    """Retrieve security remediation context for a list of Trivy findings.

    The findings list can contain either:
      - Plain strings  (e.g. the ``errors`` list from ValidationResult)
      - Dicts with keys such as ``check_id``, ``Title``, ``Message``, ``Severity``
        (raw Trivy JSON misconfig objects)

    Returns:
        A multi-section Markdown context string ready for injection into the
        remediator prompt, with all URLs stripped.  Falls back to CSV lookup
        when GraphRAG stores are unavailable.
    """
    if not findings:
        return ""

    queries = _build_queries_from_findings(findings)
    chunks_by_check_id, found_check_ids = _semantic_search(queries)

    # If Chroma found nothing, go straight to CSV fallback.
    if not found_check_ids and not chunks_by_check_id:
        print("[Security RAG] No Chroma hits — using CSV fallback.")
        return _csv_fallback(findings)

    graph_blocks = _graph_remediation_lookup(found_check_ids)

    # If Neo4j also returned nothing, fall back to Chroma chunks + CSV.
    if not graph_blocks:
        print("[Security RAG] No Neo4j results — using Chroma chunks + CSV fallback.")
        chroma_context = _assemble_security_context(chunks_by_check_id, [])
        csv_context    = _csv_fallback(findings)
        parts = [p for p in (chroma_context, csv_context) if p.strip()]
        return "\n\n".join(parts) if parts else ""

    return _assemble_security_context(chunks_by_check_id, graph_blocks)
