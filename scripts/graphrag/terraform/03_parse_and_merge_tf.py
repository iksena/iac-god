"""03_parse_and_merge_tf.py

Merge the raw provider schema (tf_schema_raw.json) with human-readable
descriptions and HCL examples from the Registry docs (tf_registry_docs.json)
into a single knowledge graph artefact (tf_knowledge_graph.json).

Mirrors the structure of scripts/graphrag/03_parse_and_merge.py (CFN) but
handles Terraform's attribute / block_types nesting instead of CFN's flat
Properties dict.

Key alignment
-------------
tf_schema_raw.json    → keyed by full resource name: 'aws_s3_bucket'
tf_registry_docs.json → also keyed by full resource name: 'aws_s3_bucket'
                         (_slug_to_tf_name in 02_fetch adds 'aws_' prefix)

The merge is therefore a direct identity lookup: registry_docs[resource_name].
No slug conversion is needed here.

Schema dictionaries processed
------------------------------
  resource_schemas      — managed resources  (resource "aws_s3_bucket" ...)
  data_source_schemas   — data lookup blocks (data "aws_ami" ...)

Both are merged into a single tf_knowledge_graph.json.  Each node carries an
`is_data_source` boolean so downstream consumers (Neo4j importer, ChromaDB
builder) can distinguish them if needed.  The Remediator benefits from having
both: managed resources to generate infrastructure, and data sources to look
up static provider metadata (AMIs, availability zones, Beanstalk solution
stacks, etc.) using correct syntax.

Output shape (tf_knowledge_graph.json)
--------------------------------------
{
  "aws_s3_bucket": {
    "name":           "aws_s3_bucket",
    "is_data_source": false,
    "description":    "...",
    "subcategory":    "S3 (Simple Storage)",
    "attributes":     { ... },
    "block_types":    { ... },
    "examples":       ["..."]
  },
  "aws_ami": {
    "name":           "aws_ami",
    "is_data_source": true,
    ...
  },
  ...
}
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_FILE   = Path("tf_schema_raw.json")
DOCS_FILE     = Path("tf_registry_docs.json")
OUTPUT_FILE   = Path("tf_knowledge_graph.json")

_PROVIDER_KEY = "registry.terraform.io/hashicorp/aws"


# ---------------------------------------------------------------------------
# Schema parsing helpers
# ---------------------------------------------------------------------------

def _type_str(type_expr) -> str:
    """Convert a Terraform type expression to a readable string.

    Type expressions are either a bare string ("string") or a nested list
    like ["list", "string"] or ["map", ["object", {...}]].
    """
    if isinstance(type_expr, str):
        return type_expr
    if isinstance(type_expr, list) and type_expr:
        outer = type_expr[0]
        if len(type_expr) == 1:
            return outer
        inner = _type_str(type_expr[1])
        return f"{outer}({inner})"
    return str(type_expr)


def _parse_attributes(attrs: dict) -> dict:
    """Recursively parse a block's attributes dict."""
    result: dict[str, dict] = {}
    for attr_name, attr_data in attrs.items():
        result[attr_name] = {
            "type":      _type_str(attr_data.get("type", "unknown")),
            "optional":  bool(attr_data.get("optional", False)),
            "required":  bool(attr_data.get("required", False)),
            "computed":  bool(attr_data.get("computed", False)),
            "sensitive": bool(attr_data.get("sensitive", False)),
            "description": attr_data.get("description", ""),
        }
    return result


def _parse_block_types(block_types: dict) -> dict:
    """Recursively parse nested block_types."""
    result: dict[str, dict] = {}
    for block_name, block_data in block_types.items():
        nesting = block_data.get("nesting_mode", "single")
        inner   = block_data.get("block", {})
        result[block_name] = {
            "nesting_mode": nesting,
            "min_items":    block_data.get("min_items", 0),
            "max_items":    block_data.get("max_items", 0),  # 0 = unlimited
            "attributes":   _parse_attributes(inner.get("attributes", {})),
            # Recurse into nested block_types (e.g. rule inside encryption config)
            "block_types":  _parse_block_types(inner.get("block_types", {})),
        }
    return result


# ---------------------------------------------------------------------------
# Core processing helper — handles both managed resources and data sources
# ---------------------------------------------------------------------------

def _process_schema_dict(
    schema_dict: dict,
    registry_docs: dict,
    kg: dict,
    is_data_source: bool,
) -> tuple[int, int, list[str]]:
    """Merge one schema dictionary (resource_schemas or data_source_schemas)
    with Registry docs into the shared knowledge graph dict.

    Returns:
        (enriched_count, examples_count, missing_docs)
    """
    enriched_count = 0
    examples_count = 0
    missing_docs:  list[str] = []

    for resource_name, resource_schema in schema_dict.items():
        # Both files are keyed by the full TF name ('aws_s3_bucket', 'aws_ami').
        # Direct lookup — no slug conversion needed here.
        doc = registry_docs.get(resource_name, {})

        if doc:
            enriched_count += 1
        else:
            missing_docs.append(resource_name)

        block       = resource_schema.get("block", {})
        attributes  = _parse_attributes(block.get("attributes", {}))
        block_types = _parse_block_types(block.get("block_types", {}))

        # Enrich top-level attribute descriptions from Registry docs when the
        # schema's own description field is empty (common for older resources).
        doc_content = doc.get("content", "")
        for attr_name, attr_data in attributes.items():
            if not attr_data["description"] and doc_content:
                # Registry markdown format: "* `{attr_name}` - description"
                marker = f"* `{attr_name}` -"
                idx    = doc_content.find(marker)
                if idx != -1:
                    end     = doc_content.find("\n", idx)
                    snippet = doc_content[idx + len(marker): end].strip()
                    if snippet:
                        attr_data["description"] = snippet

        examples = doc.get("hcl_examples", [])
        examples_count += len(examples)

        kg[resource_name] = {
            "name":           resource_name,
            "is_data_source": is_data_source,
            "description":    doc.get("description", ""),
            "subcategory":    doc.get("subcategory", ""),
            "attributes":     attributes,
            "block_types":    block_types,
            "examples":       examples,
        }

    return enriched_count, examples_count, missing_docs


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def build_knowledge_graph() -> None:
    print("[03] Loading tf_schema_raw.json ...")
    raw_schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

    provider_block = (
        raw_schema
        .get("provider_schemas", {})
        .get(_PROVIDER_KEY, {})
    )

    # Extract BOTH schema dictionaries from the provider block.
    # resource_schemas    → managed resources  (resource "aws_s3_bucket" ...)
    # data_source_schemas → data lookup blocks (data "aws_ami" ...)
    resource_schemas:     dict = provider_block.get("resource_schemas", {})
    data_source_schemas:  dict = provider_block.get("data_source_schemas", {})
    print(f"[03] Found {len(resource_schemas):,} managed resource types in schema.")
    print(f"[03] Found {len(data_source_schemas):,} data source types in schema.")

    print("[03] Loading tf_registry_docs.json ...")
    registry_docs: dict = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    print(f"[03] Found {len(registry_docs):,} registry doc entries.")

    kg: dict[str, dict] = {}

    # --- Managed resources ---
    r_enriched, r_examples, r_missing = _process_schema_dict(
        resource_schemas, registry_docs, kg, is_data_source=False
    )

    # --- Data sources ---
    d_enriched, d_examples, d_missing = _process_schema_dict(
        data_source_schemas, registry_docs, kg, is_data_source=True
    )

    OUTPUT_FILE.write_text(json.dumps(kg, indent=2), encoding="utf-8")

    total_enriched = r_enriched + d_enriched
    total_missing  = r_missing  + d_missing
    total_examples = r_examples + d_examples

    print(f"\n[03] Knowledge Graph built:")
    print(f"     Managed resources          : {len(resource_schemas):,}  "
          f"({r_enriched:,} with docs, {len(r_missing):,} schema-only)")
    print(f"     Data sources               : {len(data_source_schemas):,}  "
          f"({d_enriched:,} with docs, {len(d_missing):,} schema-only)")
    print(f"     Total KG nodes             : {len(kg):,}")
    print(f"     Total with Registry docs   : {total_enriched:,}")
    print(f"     Total schema-only (no docs): {len(total_missing):,}")
    print(f"     Total HCL examples         : {total_examples:,}")
    print(f"     Saved to                   : {OUTPUT_FILE}")

    if total_missing:
        print(f"\n[03] Nodes with no Registry doc ({len(total_missing)}):")
        for r in total_missing[:20]:
            print(f"       - {r}")
        if len(total_missing) > 20:
            print(f"       ... and {len(total_missing) - 20} more.")


if __name__ == "__main__":
    build_knowledge_graph()
