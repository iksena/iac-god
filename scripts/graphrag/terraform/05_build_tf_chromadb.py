"""05_build_tf_chromadb.py

Embed Terraform resource / attribute / block descriptions into ChromaDB
collection `tf_schema_properties`.

Design decisions
----------------
* Same ChromaDB instance as the CFN pipeline — logical isolation via a
  separate collection name (tf_schema_properties vs cfn_schema_properties).
* Every document gets `iac_type: "terraform"` metadata for future
  cross-IaC queries.
* Chunking strategy mirrors 05_build_chromadb.py (CFN):
    - One document per resource  (resource-level description + subcategory)
    - One document per attribute (name + type + description)
    - One document per block     (flattened block path + description)
  This keeps chunk granularity consistent with the CFN embedding space.

Output
------
ChromaDB collection `tf_schema_properties` in the configured persist directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KG_FILE        = Path("tf_knowledge_graph.json")
COLLECTION    = "tf_schema_properties"
PERSIST_DIR    = os.getenv("CHROMADB_PATH", "./chroma_db")
EMBED_MODEL    = os.getenv("EMBED_MODEL",   "all-MiniLM-L6-v2")
# Batch size for ChromaDB upsert calls — keeps memory usage bounded.
_UPSERT_BATCH  = 500


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

def _flatten_block(
    resource_name: str,
    block_path: str,
    block_data: dict,
    depth: int = 0,
) -> Generator[tuple[str, str, dict], None, None]:
    """Yield (doc_id, text, metadata) tuples for a block and its children."""
    if depth > 6:
        return

    block_text = (
        f"{resource_name} block {block_path} "
        f"[{block_data.get('nesting_mode', 'single')}]. "
    )
    # Inline first-level attribute names as context.
    attr_names = list(block_data.get("attributes", {}).keys())
    if attr_names:
        block_text += "Attributes: " + ", ".join(attr_names[:20]) + "."

    yield (
        f"{resource_name}.{block_path}",
        block_text,
        {"resource": resource_name, "type": "block", "block_path": block_path, "iac_type": "terraform"},
    )

    for attr_name, attr_data in block_data.get("attributes", {}).items():
        attr_path = f"{block_path}.{attr_name}"
        desc = attr_data.get("description", "").strip()
        req  = "required" if attr_data.get("required") else "optional"
        text = (
            f"{resource_name} {attr_path}: {desc} "
            f"Type: {attr_data.get('type', 'unknown')}. {req.capitalize()}."
        ).strip()
        yield (
            f"{resource_name}.{attr_path}",
            text,
            {"resource": resource_name, "type": "block_attribute", "block_path": attr_path, "iac_type": "terraform"},
        )

    for child_name, child_data in block_data.get("block_types", {}).items():
        child_path = f"{block_path}.{child_name}"
        yield from _flatten_block(resource_name, child_path, child_data, depth + 1)


def generate_documents(
    kg: dict,
) -> Generator[tuple[str, str, dict], None, None]:
    """Yield (doc_id, text, metadata) for every embeddable chunk."""
    for resource_name, resource_data in kg.items():
        desc        = resource_data.get("description", "").strip()
        subcategory = resource_data.get("subcategory", "").strip()

        # Resource-level document
        resource_text = f"{resource_name}: {desc}"
        if subcategory:
            resource_text += f" Category: {subcategory}."
        yield (
            resource_name,
            resource_text,
            {"resource": resource_name, "type": "resource", "iac_type": "terraform"},
        )

        # Per-attribute documents (top-level)
        for attr_name, attr_data in resource_data.get("attributes", {}).items():
            attr_desc = attr_data.get("description", "").strip()
            req       = "required" if attr_data.get("required") else "optional"
            text = (
                f"{resource_name} {attr_name}: {attr_desc} "
                f"Type: {attr_data.get('type', 'unknown')}. {req.capitalize()}."
            ).strip()
            yield (
                f"{resource_name}.{attr_name}",
                text,
                {"resource": resource_name, "type": "attribute", "attribute": attr_name, "iac_type": "terraform"},
            )

        # Per-block documents (recursive)
        for block_name, block_data in resource_data.get("block_types", {}).items():
            yield from _flatten_block(resource_name, block_name, block_data)


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

def _upsert_batch(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[05] Loading tf_knowledge_graph.json ...")
    kg = json.loads(KG_FILE.read_text(encoding="utf-8"))
    print(f"[05] {len(kg):,} resources loaded.")

    print(f"[05] Connecting to ChromaDB at {PERSIST_DIR} ...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Delete and recreate for a clean rebuild (collection-scoped — CFN
    # collection cfn_schema_properties is not touched).
    try:
        client.delete_collection(COLLECTION)
        print(f"[05] Dropped existing collection '{COLLECTION}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[05] Created collection '{COLLECTION}'.")

    batch_ids:   list[str]  = []
    batch_docs:  list[str]  = []
    batch_metas: list[dict] = []
    total_docs = 0
    seen_ids: set[str] = set()

    for doc_id, text, meta in generate_documents(kg):
        if not text.strip():
            continue
        # ChromaDB requires unique IDs; deduplicate silently.
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        batch_ids.append(doc_id)
        batch_docs.append(text)
        batch_metas.append(meta)
        total_docs += 1

        if len(batch_ids) >= _UPSERT_BATCH:
            _upsert_batch(collection, batch_ids, batch_docs, batch_metas)
            print(f"[05] Upserted {total_docs:,} documents so far ...")
            batch_ids, batch_docs, batch_metas = [], [], []

    if batch_ids:
        _upsert_batch(collection, batch_ids, batch_docs, batch_metas)

    print(f"\n[05] Done. {total_docs:,} documents embedded into '{COLLECTION}'.")
    print(f"[05] ChromaDB persist dir: {PERSIST_DIR}")
