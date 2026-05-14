#!/usr/bin/env python3
"""
04_import_security_to_neo4j.py

Stage 2 – Import security_checks.json into Neo4j.

Node schema
-----------
  SecurityCheck  {check_id, check_name, severity, short_code, description,
                  avd_url, framework, source_file_url}
  AwsService     {name}  (e.g. 'ec2', 's3')
  Impact         {id, text}
  Remediation    {id, framework, instruction}  ('cfn' | 'tf' | 'console')
  GoodExample    {id, framework, code}         ('cfn' | 'tf')
  RegoPolicy     {id, code, source_file_url}

Edges
-----
  (SecurityCheck)-[:AFFECTS_SERVICE]     ->(AwsService)
  (SecurityCheck)-[:HAS_IMPACT]          ->(Impact)
  (SecurityCheck)-[:HAS_REMEDIATION]     ->(Remediation)
  (SecurityCheck)-[:HAS_GOOD_EXAMPLE]    ->(GoodExample)
  (SecurityCheck)-[:ENFORCED_BY]         ->(RegoPolicy)
  (SecurityCheck)-[:APPLIES_TO_RESOURCE] ->(Resource)   # cross-graph to CFN nodes

The APPLIES_TO_RESOURCE edge is created by matching existing Resource nodes
whose .name starts with the check's cfn_resource_prefix (e.g. 'AWS::S3::').
This cross-links the two graphs so a retrieval can traverse from a security
check directly into the CFN schema subgraph.

Idempotency
-----------
All writes use MERGE so the script is safe to re-run after CSV updates.

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
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"

# Cap example/code stored in Neo4j to avoid property size limits
CODE_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

def create_indexes(session) -> None:
    print("Creating indexes...")
    session.run(
        "CREATE INDEX security_check_id IF NOT EXISTS "
        "FOR (s:SecurityCheck) ON (s.check_id)"
    )
    session.run(
        "CREATE INDEX aws_service_name IF NOT EXISTS "
        "FOR (a:AwsService) ON (a.name)"
    )
    session.run(
        "CREATE INDEX remediation_id IF NOT EXISTS "
        "FOR (r:Remediation) ON (r.id)"
    )
    session.run(
        "CREATE INDEX good_example_id IF NOT EXISTS "
        "FOR (g:GoodExample) ON (g.id)"
    )


# ---------------------------------------------------------------------------
# Core import
# ---------------------------------------------------------------------------

def import_security_graph(session, checks: dict) -> None:
    print(f"Importing {len(checks)} security checks into Neo4j...")
    total = len(checks)

    for i, (check_id, check) in enumerate(checks.items(), 1):
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {check_id}")

        service = check.get("service", "").strip().lower()
        cfn_prefix = check.get("cfn_resource_prefix", "").strip()

        # ------------------------------------------------------------------
        # 1. SecurityCheck node
        # ------------------------------------------------------------------
        session.run(
            """
            MERGE (s:SecurityCheck {check_id: $check_id})
            SET s.check_name      = $check_name,
                s.severity        = $severity,
                s.short_code      = $short_code,
                s.description     = $description,
                s.avd_url         = $avd_url,
                s.framework       = $framework,
                s.source_file_url = $source_file_url,
                s.title           = $title
            """,
            check_id=check_id,
            check_name=check.get("check_name", ""),
            severity=check.get("severity", ""),
            short_code=check.get("short_code", ""),
            description=check.get("description", ""),
            avd_url=check.get("avd_url", ""),
            framework=check.get("framework", ""),
            source_file_url=check.get("source_file_url", ""),
            title=check.get("title", ""),
        )

        # ------------------------------------------------------------------
        # 2. AwsService node + AFFECTS_SERVICE edge
        # ------------------------------------------------------------------
        if service:
            session.run(
                """
                MERGE (a:AwsService {name: $service})
                WITH a
                MATCH (s:SecurityCheck {check_id: $check_id})
                MERGE (s)-[:AFFECTS_SERVICE]->(a)
                """,
                service=service,
                check_id=check_id,
            )

        # ------------------------------------------------------------------
        # 3. Impact node + HAS_IMPACT edge
        # ------------------------------------------------------------------
        impact_text = check.get("impact", "").strip()
        if impact_text:
            session.run(
                """
                MERGE (imp:Impact {id: $impact_id})
                SET imp.text = $text
                WITH imp
                MATCH (s:SecurityCheck {check_id: $check_id})
                MERGE (s)-[:HAS_IMPACT]->(imp)
                """,
                impact_id=f"{check_id}_impact",
                text=impact_text,
                check_id=check_id,
            )

        # ------------------------------------------------------------------
        # 4. Remediation nodes + HAS_REMEDIATION edges
        # ------------------------------------------------------------------
        remediation_sources = [
            ("cfn",     check.get("remediation_cfn", [])),
            ("tf",      check.get("remediation_tf",  [])),
            ("console", check.get("remediation_console", "")),
        ]
        for framework, instructions in remediation_sources:
            # Normalise to list
            if isinstance(instructions, str):
                instructions = [instructions] if instructions.strip() else []
            for j, instruction in enumerate(instructions):
                instruction = instruction.strip()
                if not instruction:
                    continue
                rem_id = f"{check_id}_rem_{framework}_{j}"
                session.run(
                    """
                    MERGE (r:Remediation {id: $rem_id})
                    SET r.framework   = $framework,
                        r.instruction = $instruction
                    WITH r
                    MATCH (s:SecurityCheck {check_id: $check_id})
                    MERGE (s)-[:HAS_REMEDIATION]->(r)
                    """,
                    rem_id=rem_id,
                    framework=framework,
                    instruction=instruction,
                    check_id=check_id,
                )

        # ------------------------------------------------------------------
        # 5. GoodExample nodes + HAS_GOOD_EXAMPLE edges
        # ------------------------------------------------------------------
        example_sources = [
            ("cfn", check.get("cfn_good_example", "")),
            ("tf",  check.get("tf_good_example",  "")),
        ]
        for framework, code in example_sources:
            code = (code or "").strip()[:CODE_MAX_CHARS]
            if not code:
                continue
            ex_id = f"{check_id}_ex_{framework}"
            session.run(
                """
                MERGE (g:GoodExample {id: $ex_id})
                SET g.framework = $framework,
                    g.code      = $code
                WITH g
                MATCH (s:SecurityCheck {check_id: $check_id})
                MERGE (s)-[:HAS_GOOD_EXAMPLE]->(g)
                """,
                ex_id=ex_id,
                framework=framework,
                code=code,
                check_id=check_id,
            )

        # ------------------------------------------------------------------
        # 6. RegoPolicy node + ENFORCED_BY edge
        # ------------------------------------------------------------------
        rego_code = (check.get("source_code", "") or "").strip()[:CODE_MAX_CHARS]
        source_file_url = check.get("source_file_url", "").strip()
        if rego_code or source_file_url:
            session.run(
                """
                MERGE (rp:RegoPolicy {id: $policy_id})
                SET rp.code            = $code,
                    rp.source_file_url = $source_file_url
                WITH rp
                MATCH (s:SecurityCheck {check_id: $check_id})
                MERGE (s)-[:ENFORCED_BY]->(rp)
                """,
                policy_id=f"{check_id}_rego",
                code=rego_code,
                source_file_url=source_file_url,
                check_id=check_id,
            )

        # ------------------------------------------------------------------
        # 7. Cross-graph: APPLIES_TO_RESOURCE edges to CFN Resource nodes
        #    Match all Resource nodes whose .name starts with cfn_prefix.
        #    These nodes were created by 04_import_cfn_to_neo4j.py.
        # ------------------------------------------------------------------
        if cfn_prefix:
            session.run(
                """
                MATCH (s:SecurityCheck {check_id: $check_id})
                MATCH (r:Resource)
                WHERE r.name STARTS WITH $cfn_prefix
                MERGE (s)-[:APPLIES_TO_RESOURCE]->(r)
                """,
                check_id=check_id,
                cfn_prefix=cfn_prefix,
            )

    print(f"  Import complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading security checks from: {CHECKS_JSON}")
    if not CHECKS_JSON.exists():
        print(
            "ERROR: security_checks.json not found. "
            "Run 01_load_trivy_csv.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    print(f"Connecting to Neo4j at {NEO4J_URI} as '{NEO4J_USER}'...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        create_indexes(session)
        import_security_graph(session, checks)

    driver.close()

    print("\n\u2713 Security graph import complete.")
    print("\nNode types created:")
    print("  SecurityCheck, AwsService, Impact, Remediation, GoodExample, RegoPolicy")
    print("\nEdge types created (within security graph):")
    print("  AFFECTS_SERVICE, HAS_IMPACT, HAS_REMEDIATION, HAS_GOOD_EXAMPLE, ENFORCED_BY")
    print("\nCross-graph edges created:")
    print("  (SecurityCheck)-[:APPLIES_TO_RESOURCE]->(Resource)  [to CFN graph]")
    print("\nNext step: Run scripts/graphrag/security/05_execute_security_g_retrieval.py")


if __name__ == "__main__":
    main()
