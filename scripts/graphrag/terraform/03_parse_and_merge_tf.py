"""03_parse_and_merge_tf.py

Merge the raw provider schema (tf_schema_raw.json) with human-readable
descriptions and HCL examples from the Registry docs (tf_registry_docs.json)
into a single knowledge graph artefact (tf_knowledge_graph.json).

Mirrors the structure of scripts/graphrag/03_parse_and_merge.py (CFN) but
handles Terraform's attribute / block_types nesting instead of CFN's flat
Properties dict.

Output shape (tf_knowledge_graph.json)
--------------------------------------
{
  "aws_s3_bucket": {
    "name":        "aws_s3_bucket",
    "description": "...",
    "subcategory": "S3 (Simple Storage)",
    "attributes": {
      "bucket": {
        "type":        "string",
        "optional":    true,
        "computed":    true,
        "sensitive":   false,
        "description": "..."
      },
      ...
    },
    "block_types": {
      "server_side_encryption_configuration": {
        "nesting_mode": "list",
        "min_items":    0,
        "max_items":    1,
        "attributes":   { ... },
        "block_types":  { ... }   # recursively nested
      },
      ...
    },
    "examples": ["..."]
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

# Primitive Terraform types that map directly (no recursion needed).
_PRIMITIVE_TYPES = {"string", "number", "bool", "dynamic", "any"}


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
# Slug helpers
# ---------------------------------------------------------------------------

def _resource_to_slug(resource_name: str) -> str:
    """Convert 'aws_s3_bucket' to the Registry slug 'r/s3_bucket'."""
    # Registry slugs strip the 'aws_' prefix and prefix with 'r/'.
    without_prefix = resource_name.removeprefix("aws_")
    return f"r/{without_prefix}"


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
    resource_schemas: dict = provider_block.get("resource_schemas", {})
    print(f"[03] Found {len(resource_schemas):,} resource types in schema.")

    print("[03] Loading tf_registry_docs.json ...")
    registry_docs: dict = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    print(f"[03] Found {len(registry_docs):,} registry doc entries.")

    kg: dict[str, dict] = {}
    enriched_count    = 0
    examples_count    = 0
    missing_docs      = []

    for resource_name, resource_schema in resource_schemas.items():
        slug  = _resource_to_slug(resource_name)
        doc   = registry_docs.get(slug, {})

        if doc:
            enriched_count += 1
        else:
            missing_docs.append(resource_name)

        block = resource_schema.get("block", {})
        attributes  = _parse_attributes(block.get("attributes", {}))
        block_types = _parse_block_types(block.get("block_types", {}))

        # Enrich top-level attribute descriptions from Registry docs when the
        # schema's own description field is empty (common for older resources).
        doc_content = doc.get("content", "")
        for attr_name, attr_data in attributes.items():
            if not attr_data["description"] and doc_content:
                # Heuristic: look for "* `{attr_name}` -" in the markdown,
                # which is the standard Registry attribute-description format.
                marker = f"* `{attr_name}` -"
                idx = doc_content.find(marker)
                if idx != -1:
                    end = doc_content.find("\n", idx)
                    snippet = doc_content[idx + len(marker): end].strip()
                    if snippet:
                        attr_data["description"] = snippet

        examples = doc.get("hcl_examples", [])
        examples_count += len(examples)

        kg[resource_name] = {
            "name":        resource_name,
            "description": doc.get("description", ""),
            "subcategory": doc.get("subcategory", ""),
            "attributes":  attributes,
            "block_types": block_types,
            "examples":    examples,
        }

    OUTPUT_FILE.write_text(json.dumps(kg, indent=2), encoding="utf-8")

    print(f"\n[03] Knowledge Graph built:")
    print(f"     Total resources          : {len(kg):,}")
    print(f"     With Registry docs       : {enriched_count:,}")
    print(f"     Schema-only (no docs)    : {len(missing_docs):,}")
    print(f"     Total HCL examples       : {examples_count:,}")
    print(f"     Saved to                 : {OUTPUT_FILE}")

    if missing_docs:
        print(f"\n[03] Resources with no Registry doc ({len(missing_docs)}):")
        for r in missing_docs[:20]:
            print(f"       - {r}")
        if len(missing_docs) > 20:
            print(f"       ... and {len(missing_docs) - 20} more.")


if __name__ == "__main__":
    build_knowledge_graph()
