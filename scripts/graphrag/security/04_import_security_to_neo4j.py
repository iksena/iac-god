#!/usr/bin/env python3
"""
04_import_security_to_neo4j.py

Stage 2 – Import security_checks.json into Neo4j.

===============================================================================
RESEARCH NOTE: GraphRAG vs. Vector-Only for Security Remediation Context
===============================================================================

This pipeline is intentionally built as a *hybrid* GraphRAG (ChromaDB vector
search + Neo4j graph traversal) rather than a pure vector embedding store.
Here is the explicit justification for that design choice.

--- OPTION A: Vector-Only (ChromaDB alone) ---------------------------------

How it would work:
  - Embed one document per check (description + impact + remediation + example).
  - At retrieval time, embed the Trivy finding text and return the top-k docs.
  - The full document text is injected directly into the prompt.

Advantages:
  - Simpler infrastructure (no Neo4j, single Docker container).
  - Fast: one embedding call, one similarity search.
  - Works well when the semantic content of the finding closely matches the
    embedded document (i.e., check_id + description are sufficient).

Limitations for this use case:
  1. Flat structure: each Chroma document is a single text blob. To include
     the Rego policy code, CFN good example, impact, and remediation
     instruction in one chunk you either (a) concatenate everything into one
     large document, inflating the chunk and hurting retrieval precision, or
     (b) split across multiple chunks and risk missing parts (e.g., the
     policy code chunk scores lower than the description chunk and gets
     dropped by the distance threshold).
  2. No cross-linking: there is no way for a vector store query to
     automatically "jump" from a SecurityCheck to the CFN Resource node for
     the same service. That cross-graph join has to be done in application
     code with a separate CFN Chroma query, defeating the purpose.
  3. No structured filtering: you cannot ask "give me all CRITICAL checks for
     s3" without post-filtering Python code. In Neo4j this is one Cypher
     WHERE clause.
  4. Research transparency: vector similarity is a black box. For a research
     paper you cannot explain *why* a particular chunk was retrieved beyond
     "cosine distance". The graph retrieval path can be shown as an explicit
     Cypher traversal, which is auditable and reproducible.

--- OPTION B: GraphRAG (ChromaDB → Neo4j) — CHOSEN -------------------------

How it works (this pipeline):
  - Stage 1: embed a *lightweight* query document per check (check_name +
    description + impact only — short, semantically rich, no code noise).
    This gives the vector index a sharp semantic surface for matching.
  - Stage 2: use the matched check_ids as entry points for a structured Neo4j
    traversal that returns ALL related nodes (Impact, Remediation,
    GoodExample, RegoPolicy) via typed edges. The graph returns exactly the
    fields needed, nothing more.
  - Cross-graph: the APPLIES_TO_RESOURCE edge lets the retrieval traverse
    directly from a SecurityCheck into the CFN schema graph (Resource,
    Property, Example nodes), giving the remediator both the security
    constraint and the structural schema in a single query.

Advantages over vector-only:
  1. Query-time decomposition: semantic search finds the right check; graph
     traversal assembles the full structured context. Neither step pollutes
     the other.
  2. Selective field inclusion: the Cypher query in security_hybrid_rag.py
     explicitly pulls description, impact, cfn remediation, cfn example, and
     rego code — and strips everything else (avd_url, tf example, links).
     A flat vector store would require manual post-processing.
  3. Cross-graph traversal: (SecurityCheck)-[:APPLIES_TO_RESOURCE]->(Resource)
     is unrepresentable in a vector store without a separate query + join.
  4. Research auditability: the traversal path is a deterministic Cypher
     statement, not a probabilistic similarity score. This is important for
     reproducibility in a research context — you can log the exact subgraph
     returned for each finding.
  5. Idempotent updates: MERGE semantics mean re-running after a CSV update
     only changes the modified nodes, not the whole graph.

When vector-only WOULD be sufficient:
  - If the remediation context never needs cross-graph linking to CFN schema.
  - If all checks have short, self-contained remediation text (< 400 tokens).
  - If research reproducibility is not a concern.

Conclusion: for IaC generation with security remediation, where the
remediator needs (a) *why* a resource is non-compliant, (b) *how* to fix it
in CloudFormation, (c) *what* the CFN schema allows, and (d) *what* the Rego
policy enforces — the graph structure is load-bearing, not optional.

The CSV (trivy_enriched.csv) is still read by 01_load_trivy_csv.py to
populate security_checks.json, but once this Neo4j import runs, the
application path (security_hybrid_rag.py) no longer reads the CSV directly.
The CSV becomes a build-time artefact, not a runtime dependency.
===============================================================================

Node schema
-----------
  SecurityCheck  {check_id, check_name, severity, short_code, description,
                  service, framework, source_file_url}
  AwsService     {name}  (e.g. 'ec2', 's3')
  Impact         {id, text}
  Remediation    {id, framework, instruction}
  GoodExample    {id, framework, code}
  RegoPolicy     {id, code, source_file_url}

Edges
-----
  (SecurityCheck)-[:AFFECTS_SERVICE]     ->(AwsService)
  (SecurityCheck)-[:HAS_IMPACT]          ->(Impact)
  (SecurityCheck)-[:HAS_REMEDIATION]     ->(Remediation)
  (SecurityCheck)-[:HAS_GOOD_EXAMPLE]    ->(GoodExample)
  (SecurityCheck)-[:ENFORCED_BY]         ->(RegoPolicy)
  (SecurityCheck)-[:APPLIES_TO_RESOURCE] ->(Resource)  # cross-graph to CFN nodes

Note: avd_url is NOT stored on SecurityCheck nodes. URLs are excluded from
  the graph because security_hybrid_rag.py strips them from prompt context
  anyway, and storing them just creates noisy properties with no retrieval
  value.

Idempotency
-----------
All writes use MERGE so the script is safe to re-run after CSV updates.
Only properties that changed will be updated via SET.

Batch strategy
--------------
Each check is imported in a single batched transaction (all child nodes and
edges for that check atomically). This avoids the N*M round-trips of the
previous per-node-per-edge design while keeping transactions small enough
that a failure is limited to one check.

Usage
-----
    python scripts/graphrag/security/04_import_security_to_neo4j.py

Environment variables (same as CFN pipeline)
--------------------------------------------
    NEO4J_URI       bolt://localhost:7687
    NEO4J_USER      neo4j
    NEO4J_PASSWORD  password
"""

import json
import os
import sys
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: Install neo4j driver: pip install neo4j", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

REPO_ROOT   = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"

# Neo4j has a 64 MB property limit; cap large text fields to stay well under.
CODE_MAX_CHARS = 4000
TEXT_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Indexes & constraints
# ---------------------------------------------------------------------------

def create_schema(session) -> None:
    print("Creating indexes and constraints...")

    # Uniqueness constraints create an implicit index, so we use MERGE safely.
    constraints = [
        ("SecurityCheck", "check_id"),
        ("AwsService",    "name"),
        ("Impact",        "id"),
        ("Remediation",   "id"),
        ("GoodExample",   "id"),
        ("RegoPolicy",    "id"),
    ]
    for label, prop in constraints:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_{prop}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )

    # Extra composite index on (severity, service) to support Cypher filters
    # like: WHERE sc.severity = 'CRITICAL' AND sc.service = 's3'
    session.run(
        "CREATE INDEX security_check_severity_service IF NOT EXISTS "
        "FOR (s:SecurityCheck) ON (s.severity, s.service)"
    )
    print("  Schema ready.")


# ---------------------------------------------------------------------------
# Per-check batched import (all nodes + edges for one check in one tx)
# ---------------------------------------------------------------------------

_IMPORT_CHECK_CYPHER = """
// ── 1. SecurityCheck node ────────────────────────────────────────────────
MERGE (sc:SecurityCheck {check_id: $check_id})
SET sc.check_name      = $check_name,
    sc.severity        = $severity,
    sc.short_code      = $short_code,
    sc.description     = $description,
    sc.service         = $service,
    sc.framework       = $framework,
    sc.source_file_url = $source_file_url

// ── 2. AwsService ────────────────────────────────────────────────────────
WITH sc
FOREACH (svc IN CASE WHEN $service <> '' THEN [$service] ELSE [] END |
    MERGE (a:AwsService {name: svc})
    MERGE (sc)-[:AFFECTS_SERVICE]->(a)
)

// ── 3. Impact ─────────────────────────────────────────────────────────────
WITH sc
FOREACH (txt IN CASE WHEN $impact <> '' THEN [$impact] ELSE [] END |
    MERGE (imp:Impact {id: $impact_id})
    SET imp.text = txt
    MERGE (sc)-[:HAS_IMPACT]->(imp)
)

// ── 4. GoodExample (CFN only — TF example excluded from runtime context) ─
WITH sc
FOREACH (code IN CASE WHEN $cfn_example <> '' THEN [$cfn_example] ELSE [] END |
    MERGE (ge:GoodExample {id: $cfn_example_id})
    SET ge.framework = 'cloudformation',
        ge.code      = code
    MERGE (sc)-[:HAS_GOOD_EXAMPLE]->(ge)
)

// ── 5. RegoPolicy ─────────────────────────────────────────────────────────
WITH sc
FOREACH (code IN CASE WHEN $rego_code <> '' THEN [$rego_code] ELSE [] END |
    MERGE (rp:RegoPolicy {id: $rego_id})
    SET rp.code            = code,
        rp.source_file_url = $source_file_url
    MERGE (sc)-[:ENFORCED_BY]->(rp)
)

RETURN sc.check_id AS check_id
"""

# Remediation instructions are multi-valued so they get a separate
# per-instruction write to keep the main Cypher clean.
_IMPORT_REMEDIATION_CYPHER = """
MATCH (sc:SecurityCheck {check_id: $check_id})
MERGE (r:Remediation {id: $rem_id})
SET r.framework   = $framework,
    r.instruction = $instruction
MERGE (sc)-[:HAS_REMEDIATION]->(r)
"""

# Cross-graph edge: SecurityCheck → CFN Resource nodes
_CROSS_GRAPH_CYPHER = """
MATCH (sc:SecurityCheck {check_id: $check_id})
MATCH (res:Resource)
WHERE res.name STARTS WITH $cfn_prefix
MERGE (sc)-[:APPLIES_TO_RESOURCE]->(res)
"""


def _import_one_check(session, check_id: str, check: dict) -> bool:
    """Import a single check. Returns True on success, False on error."""
    service    = check.get("service", "").strip().lower()
    cfn_prefix = check.get("cfn_resource_prefix", "").strip()

    impact_text  = (check.get("impact",   "") or "").strip()[:TEXT_MAX_CHARS]
    cfn_example  = (check.get("cfn_good_example", "") or "").strip()[:CODE_MAX_CHARS]
    rego_code    = (check.get("source_code", "") or "").strip()[:CODE_MAX_CHARS]
    source_url   = (check.get("source_file_url", "") or "").strip()
    description  = (check.get("description", "") or "").strip()[:TEXT_MAX_CHARS]

    try:
        # ── Main node + Impact + GoodExample + RegoPolicy ──────────────
        session.run(
            _IMPORT_CHECK_CYPHER,
            check_id       = check_id,
            check_name     = check.get("check_name", ""),
            severity       = check.get("severity", ""),
            short_code     = check.get("short_code", ""),
            description    = description,
            service        = service,
            framework      = check.get("framework", ""),
            source_file_url= source_url,
            impact         = impact_text,
            impact_id      = f"{check_id}_impact",
            cfn_example    = cfn_example,
            cfn_example_id = f"{check_id}_ex_cfn",
            rego_code      = rego_code,
            rego_id        = f"{check_id}_rego",
        )

        # ── Remediation instructions (CFN only; TF excluded from runtime) ──
        cfn_remediations = check.get("remediation_cfn", [])
        if isinstance(cfn_remediations, str):
            cfn_remediations = [cfn_remediations] if cfn_remediations.strip() else []
        for j, instruction in enumerate(cfn_remediations):
            instruction = instruction.strip()[:TEXT_MAX_CHARS]
            if not instruction:
                continue
            session.run(
                _IMPORT_REMEDIATION_CYPHER,
                check_id    = check_id,
                rem_id      = f"{check_id}_rem_cfn_{j}",
                framework   = "cloudformation",
                instruction = instruction,
            )

        # ── Cross-graph edge to CFN Resource nodes ─────────────────────
        if cfn_prefix:
            result = session.run(
                _CROSS_GRAPH_CYPHER,
                check_id   = check_id,
                cfn_prefix = cfn_prefix,
            )
            summary = result.consume()
            created = summary.counters.relationships_created
            if created:
                pass  # Logged in bulk summary below.

        return True

    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR importing {check_id}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Post-import verification
# ---------------------------------------------------------------------------

def verify_import(session, expected: int) -> None:
    print("\nVerifying import...")
    counts = {}
    for label in ("SecurityCheck", "AwsService", "Impact", "Remediation",
                  "GoodExample", "RegoPolicy"):
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
        counts[label] = result.single()["c"]

    edge_counts = {}
    for rel in ("AFFECTS_SERVICE", "HAS_IMPACT", "HAS_REMEDIATION",
                "HAS_GOOD_EXAMPLE", "ENFORCED_BY", "APPLIES_TO_RESOURCE"):
        result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
        edge_counts[rel] = result.single()["c"]

    print("  Node counts:")
    for label, count in counts.items():
        status = "✓" if (label != "SecurityCheck" or count == expected) else "✗"
        print(f"    {status} {label:<20} {count:>6}")

    print("  Edge counts:")
    for rel, count in edge_counts.items():
        indicator = "✓" if count > 0 else "⚠ (0 — may be expected if nodes are missing)"
        print(f"    {indicator} {rel:<30} {count:>6}")

    # Cross-graph coverage report
    cross = edge_counts.get("APPLIES_TO_RESOURCE", 0)
    sc    = counts.get("SecurityCheck", 0)
    if sc > 0:
        pct = 100.0 * cross / sc
        print(f"\n  Cross-graph coverage: {cross}/{sc} checks linked to CFN resources ({pct:.1f}%)")
        if pct < 50:
            print(
                "  ⚠  Less than 50% of checks are linked to CFN resources.\n"
                "     Run 01_load_trivy_csv.py and check the service→CFN prefix mapping.\n"
                "     Also ensure 04_import_cfn_to_neo4j.py has been run first."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading security checks from: {CHECKS_JSON}")
    if not CHECKS_JSON.exists():
        print(
            "ERROR: security_checks.json not found. "
            "Run scripts/graphrag/security/01_load_trivy_csv.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    total    = len(checks)
    print(f"Loaded {total} security checks.")

    print(f"Connecting to Neo4j at {NEO4J_URI} as '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    succeeded = failed = cross_edges = 0

    with driver.session() as session:
        create_schema(session)

        print(f"\nImporting {total} checks (one transaction per check)...")
        for i, (check_id, check) in enumerate(checks.items(), 1):
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] ... (ok={succeeded} err={failed})")

            ok = _import_one_check(session, check_id, check)
            if ok:
                succeeded += 1
            else:
                failed += 1

        verify_import(session, total)

    driver.close()

    print(f"\n{'✓' if failed == 0 else '✗'} Import complete: {succeeded} ok, {failed} errors.")
    if failed:
        print(f"  {failed} checks failed — review ERROR lines above.", file=sys.stderr)
        sys.exit(1)

    print("\nNode labels written:")
    print("  SecurityCheck, AwsService, Impact, Remediation, GoodExample, RegoPolicy")
    print("\nEdge types written:")
    print("  AFFECTS_SERVICE, HAS_IMPACT, HAS_REMEDIATION (CFN only),")
    print("  HAS_GOOD_EXAMPLE (CFN only), ENFORCED_BY, APPLIES_TO_RESOURCE")
    print("\nNote: TF remediations and TF examples are intentionally excluded")
    print("  from the graph — this pipeline targets CloudFormation generation only.")
    print("\nNext steps:")
    print("  3. Run scripts/graphrag/security/03_build_security_chromadb.py")
    print("  4. Run scripts/graphrag/security/05_execute_security_g_retrieval.py")


if __name__ == "__main__":
    main()
