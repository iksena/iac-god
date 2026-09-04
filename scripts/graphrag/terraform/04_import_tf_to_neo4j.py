"""04_import_tf_to_neo4j.py

Ingest tf_knowledge_graph.json into the shared Neo4j instance using
Terraform-specific node labels (TFResource, TFAttribute, TFBlock, TFExample).

CRITICAL: the clear step is LABEL-SCOPED.  It deletes only TF* nodes,
preventing any collision with the existing CloudFormation graph
(Resource, Property, NestedType, Example) loaded by 04_import_cfn_to_neo4j.py.

Node / relationship schema
--------------------------
(:TFResource)   -[:HAS_ATTRIBUTE]->  (:TFAttribute)
(:TFResource)   -[:HAS_BLOCK]->      (:TFBlock)
(:TFBlock)      -[:HAS_ATTRIBUTE]->  (:TFAttribute)
(:TFBlock)      -[:HAS_BLOCK]->      (:TFBlock)      # nested blocks
(:TFResource)   -[:HAS_EXAMPLE]->    (:TFExample)

Security cross-link (written by 04_import_security_to_neo4j.py):
(:SecurityCheck)-[:APPLIES_TO_TF_RESOURCE]-> (:TFResource)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

KG_FILE = Path("tf_knowledge_graph.json")

# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

_INDEX_QUERIES = [
    "CREATE INDEX tf_resource_name   IF NOT EXISTS FOR (r:TFResource)  ON (r.name)",
    "CREATE INDEX tf_attribute_id    IF NOT EXISTS FOR (a:TFAttribute) ON (a.id)",
    "CREATE INDEX tf_block_id        IF NOT EXISTS FOR (b:TFBlock)     ON (b.id)",
]


def create_indexes(session) -> None:
    print("[04] Creating indexes ...")
    for q in _INDEX_QUERIES:
        session.run(q)


# ---------------------------------------------------------------------------
# Scoped clear (TF labels only — CFN graph untouched)
# ---------------------------------------------------------------------------

def clear_tf_data(session) -> None:
    """Delete all TF* nodes and their relationships.

    Uses label-scoped MATCH so the CloudFormation graph (Resource, Property,
    NestedType, Example) is never touched.
    """
    print("[04] Clearing existing Terraform nodes (TFResource, TFAttribute, TFBlock, TFExample) ...")
    for label in ("TFResource", "TFAttribute", "TFBlock", "TFExample"):
        result = session.run(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS deleted")
        deleted = result.single()["deleted"]
        print(f"     Deleted {deleted:,} :{label} nodes.")


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _import_attributes(
    session,
    parent_id: str,
    parent_label: str,
    attributes: dict,
) -> None:
    """Create TFAttribute nodes and connect them to a parent node."""
    for attr_name, attr_data in attributes.items():
        attr_id = f"{parent_id}.{attr_name}"
        session.run(
            f"""
            MATCH (parent:{parent_label} {{id: $parent_id}})
            MERGE (a:TFAttribute {{id: $attr_id}})
            SET a.name        = $name,
                a.type        = $type,
                a.optional    = $optional,
                a.required    = $required,
                a.computed    = $computed,
                a.sensitive   = $sensitive,
                a.description = $description
            MERGE (parent)-[:HAS_ATTRIBUTE]->(a)
            """,
            parent_id   = parent_id,
            attr_id     = attr_id,
            name        = attr_name,
            type        = attr_data.get("type", "unknown"),
            optional    = attr_data.get("optional", False),
            required    = attr_data.get("required", False),
            computed    = attr_data.get("computed", False),
            sensitive   = attr_data.get("sensitive", False),
            description = attr_data.get("description", ""),
        )


def _import_block_types(
    session,
    parent_id: str,
    parent_label: str,
    block_types: dict,
    depth: int = 0,
) -> None:
    """Recursively create TFBlock nodes and their attributes."""
    if depth > 6:  # guard against pathological nesting
        return

    for block_name, block_data in block_types.items():
        block_id = f"{parent_id}.{block_name}"
        session.run(
            f"""
            MATCH (parent:{parent_label} {{id: $parent_id}})
            MERGE (b:TFBlock {{id: $block_id}})
            SET b.name         = $name,
                b.nesting_mode = $nesting_mode,
                b.min_items    = $min_items,
                b.max_items    = $max_items
            MERGE (parent)-[:HAS_BLOCK]->(b)
            """,
            parent_id    = parent_id,
            block_id     = block_id,
            name         = block_name,
            nesting_mode = block_data.get("nesting_mode", "single"),
            min_items    = block_data.get("min_items", 0),
            max_items    = block_data.get("max_items", 0),
        )
        _import_attributes(session, block_id, "TFBlock", block_data.get("attributes", {}))
        _import_block_types(
            session, block_id, "TFBlock",
            block_data.get("block_types", {}),
            depth=depth + 1,
        )


def import_tf_graph(session, kg: dict) -> None:
    total = len(kg)
    print(f"[04] Importing {total:,} Terraform resources ...")

    for idx, (resource_name, resource_data) in enumerate(kg.items(), start=1):
        if idx % 100 == 0 or idx == total:
            print(f"     {idx:,}/{total:,} — {resource_name}")

        # --- TFResource node ---
        session.run(
            """
            MERGE (r:TFResource {name: $name})
            SET r.id          = $name,
                r.description = $description,
                r.subcategory = $subcategory
            """,
            name        = resource_name,
            description = resource_data.get("description", ""),
            subcategory = resource_data.get("subcategory", ""),
        )

        # --- Top-level attributes ---
        _import_attributes(
            session,
            parent_id    = resource_name,
            parent_label = "TFResource",
            attributes   = resource_data.get("attributes", {}),
        )

        # --- Top-level block types (recursive) ---
        _import_block_types(
            session,
            parent_id    = resource_name,
            parent_label = "TFResource",
            block_types  = resource_data.get("block_types", {}),
        )

        # --- HCL examples ---
        for i, example_code in enumerate(resource_data.get("examples", [])):
            session.run(
                """
                MATCH (r:TFResource {name: $resource_name})
                MERGE (e:TFExample {id: $ex_id})
                SET e.code  = $code,
                    e.index = $index
                MERGE (r)-[:HAS_EXAMPLE]->(e)
                """,
                resource_name = resource_name,
                ex_id         = f"{resource_name}_ex_{i}",
                code          = example_code,
                index         = i,
            )

    print(f"[04] Import complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[04] Connecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    kg = json.loads(KG_FILE.read_text(encoding="utf-8"))

    with driver.session() as session:
        create_indexes(session)
        clear_tf_data(session)
        import_tf_graph(session, kg)

    driver.close()
    print("[04] Done.")
