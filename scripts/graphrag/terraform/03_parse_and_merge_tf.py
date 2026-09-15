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
                         (_slug_to_tf_name in 02_fetch adds the provider's
                         resource_prefix, e.g. 'aws_' or 'docker_')

The merge is therefore a direct identity lookup: registry_docs[resource_name].
No slug conversion is needed here.

Multi-provider coverage
------------------------
This script does NOT hardcode a single provider key. It iterates over every
key present under tf_schema_raw.json's "provider_schemas" dict — i.e.
whatever 01_download_tf_schema.py's PROVIDERS list actually produced (AWS,
Docker, or any future addition) — and merges each provider's
resource_schemas / data_source_schemas into the same flat knowledge graph.
Resource type names are unique across providers in practice (aws_* vs.
docker_* prefixes), so no additional namespacing beyond the existing
"data." prefix is needed.

Schema dictionaries processed
------------------------------
  resource_schemas      — managed resources  (resource "aws_s3_bucket" ...)
  data_source_schemas   — data lookup blocks (data "aws_ami" ...)

Namespace collision prevention
-------------------------------
Terraform reuses the same provider-type name for both a managed resource and
its corresponding data source (e.g. aws_vpc exists in both dictionaries).
To prevent the data source node from silently overwriting the managed resource
node in the knowledge graph, data source entries are keyed with an explicit
"data." prefix:

  managed resource  →  kg["aws_vpc"]           (resource "aws_vpc" ...)
  data source       →  kg["data.aws_vpc"]       (data "aws_vpc" ...)

This prefix propagates through to ChromaDB metadata and Neo4j node names so
the Retriever can query the correct schema when the error points to a data
block versus a managed resource block.

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
  "data.aws_ami": {
    "name":           "data.aws_ami",
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

    Namespace collision prevention
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Terraform frequently defines identically-named types for both a managed
    resource and its data-source counterpart (e.g. ``aws_vpc`` exists in
    *both* ``resource_schemas`` and ``data_source_schemas``).  Processing
    both dictionaries into the same ``kg`` dict using the bare ``res_name``
    as the key would cause the data source entry to silently overwrite the
    managed resource entry (or vice versa, depending on iteration order).

    To prevent this, data source entries receive an explicit ``"data."``
    prefix on their graph key::

        managed resource  →  kg["aws_vpc"]        (no prefix)
        data source       →  kg["data.aws_vpc"]   ("data." prefix)

    The ``name`` field inside the node is set to the same prefixed key so
    that Neo4j node labels and ChromaDB document IDs are consistent with the
    dictionary key that the Retriever uses for lookups.

    Returns:
        (enriched_count, examples_count, missing_docs)
    """
    enriched_count = 0
    examples_count = 0
    missing_docs:  list[str] = []

    for resource_name, resource_schema in schema_dict.items():
        # -----------------------------------------------------------------
        # Namespace: data sources get a "data." prefix; resources stay bare.
        # -----------------------------------------------------------------
        graph_key = f"data.{resource_name}" if is_data_source else resource_name

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

        # Store using graph_key so managed resource and data source variants
        # of the same type coexist without collision.
        kg[graph_key] = {
            "name":           graph_key,   # namespaced key, not bare res_name
            "is_data_source": is_data_source,
            # `or ""` (not just a .get default) guards against docs where the
            # Registry API returned an explicit JSON null for these fields
            # (seen on kreuzwerker/docker entries) — a plain .get(key, "")
            # only substitutes when the key is absent, not when it's present
            # with value None, and a stray None here crashes any downstream
            # .strip() call (e.g. in 05_build_tf_chromadb.py).
            "description":    doc.get("description") or "",
            "subcategory":    doc.get("subcategory") or "",
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

    provider_schemas: dict = raw_schema.get("provider_schemas", {})
    provider_keys = sorted(provider_schemas.keys())
    print(f"[03] Found {len(provider_keys)} provider(s) in schema: {provider_keys}")

    print("[03] Loading tf_registry_docs.json ...")
    registry_docs: dict = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    print(f"[03] Found {len(registry_docs):,} registry doc entries.")

    kg: dict[str, dict] = {}
    total_resource_schemas = 0
    total_data_source_schemas = 0
    total_enriched = 0
    total_missing: list[str] = []
    total_examples = 0

    for provider_key in provider_keys:
        provider_block = provider_schemas.get(provider_key, {})
        # Extract BOTH schema dictionaries from the provider block.
        # resource_schemas    → managed resources  (resource "aws_s3_bucket" ...)
        # data_source_schemas → data lookup blocks (data "aws_ami" ...)
        resource_schemas:    dict = provider_block.get("resource_schemas", {})
        data_source_schemas: dict = provider_block.get("data_source_schemas", {})
        print(f"\n[03] --- Provider: {provider_key} ---")
        print(f"[03] Found {len(resource_schemas):,} managed resource types in schema.")
        print(f"[03] Found {len(data_source_schemas):,} data source types in schema.")

        total_resource_schemas    += len(resource_schemas)
        total_data_source_schemas += len(data_source_schemas)

        # --- Managed resources (keyed by bare resource name, e.g. "aws_vpc") ---
        r_enriched, r_examples, r_missing = _process_schema_dict(
            resource_schemas, registry_docs, kg, is_data_source=False
        )

        # --- Data sources (keyed as "data.{name}", e.g. "data.aws_vpc") ---
        d_enriched, d_examples, d_missing = _process_schema_dict(
            data_source_schemas, registry_docs, kg, is_data_source=True
        )

        total_enriched += r_enriched + d_enriched
        total_missing  += r_missing + d_missing
        total_examples += r_examples + d_examples

    OUTPUT_FILE.write_text(json.dumps(kg, indent=2), encoding="utf-8")

    print(f"\n[03] Knowledge Graph built (across {len(provider_keys)} provider(s)):")
    print(f"     Total managed resources    : {total_resource_schemas:,}")
    print(f"     Total data sources         : {total_data_source_schemas:,}")
    print(f"     Total KG nodes             : {len(kg):,}")
    print(f"     Total with Registry docs   : {total_enriched:,}")
    print(f"     Total schema-only (no docs): {len(total_missing):,}")
    print(f"     Total HCL examples         : {total_examples:,}")
    print(f"     Saved to                   : {OUTPUT_FILE}")

    # Collision check: confirm no bare key from one provider's resources was
    # overwritten by another provider's data. key, or vice versa (would
    # indicate a logic error in _process_schema_dict, or a genuine cross-
    # provider name collision that needs its own namespacing).
    all_resource_keys: set[str] = set()
    all_data_keys: set[str] = set()
    for provider_key in provider_keys:
        provider_block = provider_schemas.get(provider_key, {})
        all_resource_keys |= set(provider_block.get("resource_schemas", {}).keys())
        all_data_keys     |= {f"data.{n}" for n in provider_block.get("data_source_schemas", {}).keys()}
    overlap = all_resource_keys & all_data_keys
    if overlap:
        print(f"\n[03] WARNING: unexpected key overlap ({len(overlap)}): {sorted(overlap)[:5]}")
    else:
        print("\n[03] Collision check passed — no managed resource keys overlap with data. keys.")

    if total_missing:
        print(f"\n[03] Nodes with no Registry doc ({len(total_missing)}):")
        for r in total_missing[:20]:
            print(f"       - {r}")
        if len(total_missing) > 20:
            print(f"       ... and {len(total_missing) - 20} more.")

    docker_present = "docker_container" in kg and "docker_image" in kg
    print(f"\n[03] Sanity check docker_container & docker_image present in KG: {docker_present}")


if __name__ == "__main__":
    build_knowledge_graph()
