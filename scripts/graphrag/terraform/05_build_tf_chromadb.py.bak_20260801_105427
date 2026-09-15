"""05_build_tf_chromadb.py

Embed Terraform resource / attribute / block descriptions into ChromaDB
collection `tf_schema_properties`.

Design decisions
----------------
* Mirrors 05_build_chromadb.py (CFN) exactly in embedding stack:
    - EMBEDDING_PROVIDER env var selects 'ollama' (default) or 'huggingface'.
    - _NormalisedEmbeddings wraps every provider so vectors are L2-unit-length
      before ingestion.  This makes cosine distance == angular distance and
      keeps query-time threshold semantics identical between CFN and TF.
    - ChromaDB HttpClient (not PersistentClient) so both pipelines share the
      same running ChromaDB server instance.
* Same ChromaDB instance as the CFN pipeline — logical isolation via a
  separate collection name (tf_schema_properties vs cfn_schema_properties).
* Every document gets `iac_type: "terraform"` metadata for future
  cross-IaC queries.
* Chunking strategy:
    - One document per resource  (resource-level description + subcategory)
    - One document per attribute (name + type + description)
    - One document per block     (flattened block path + description)
    - One document per HCL example
  Granularity consistent with CFN embedding space.

Environment variables
---------------------
EMBEDDING_PROVIDER   'ollama' | 'huggingface'   (default: 'ollama')
EMBEDDING_MODEL      model name                  (default per provider below)
OLLAMA_BASE_URL      Ollama server URL            (default: http://localhost:11434)
CHROMA_HOST          ChromaDB host                (default: localhost)
CHROMA_PORT          ChromaDB port                (default: 8000)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Generator

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Embedding provider — mirrors tools/embedding_provider.py so this build
# script has no dependency on the tools/ package.
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower().strip()
_DEFAULTS = {
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama":      "mxbai-embed-large",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULTS.get(_PROVIDER, _DEFAULTS["ollama"]))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Collection is always created with cosine distance so threshold semantics
# are unambiguous regardless of provider.  Must match the query-time config
# in tools/embedding_provider.py (CHROMA_COLLECTION_METADATA).
COLLECTION_METADATA = {"hnsw:space": "cosine"}


class _NormalisedEmbeddings:
    """Thin wrapper that L2-normalises every vector before ingestion.

    Ensures unit-length vectors so cosine distance == angular distance,
    regardless of whether the underlying model normalises by default.
    HuggingFaceEmbeddings with normalize_embeddings=True already does this;
    OllamaEmbeddings does not.
    """
    def __init__(self, base):
        self._base = base

    @staticmethod
    def _norm(vecs):
        arr   = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).tolist()

    def embed_documents(self, texts):
        return self._norm(self._base.embed_documents(texts))

    def embed_query(self, text):
        return self._norm([self._base.embed_query(text)])[0]


def _get_embeddings():
    if _PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            print("ERROR: langchain-ollama not installed. Run: pip install langchain-ollama")
            sys.exit(1)
        print(f"[05] Embedding provider : Ollama")
        print(f"[05] Model             : {EMBEDDING_MODEL}")
        print(f"[05] Ollama URL        : {OLLAMA_BASE_URL}")
        base = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return _NormalisedEmbeddings(base)  # Ollama does not normalise by default

    # HuggingFace fallback
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[05] Embedding provider : HuggingFace")
    print(f"[05] Model             : {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # already normalised
    )


# ---------------------------------------------------------------------------
# ChromaDB connection
# ---------------------------------------------------------------------------

CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "tf_schema_properties"

# ---------------------------------------------------------------------------
# Knowledge graph path
# ---------------------------------------------------------------------------

KG_FILE = Path("tf_knowledge_graph.json")


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

def _flatten_block(
    resource_name: str,
    block_path: str,
    block_data: dict,
    depth: int = 0,
) -> Generator[Document, None, None]:
    """Yield LangChain Documents for a block and all its nested children."""
    if depth > 6:
        return

    attr_names = list(block_data.get("attributes", {}).keys())
    block_text = (
        f"Resource: {resource_name}\n"
        f"Block: {block_path}\n"
        f"Nesting: {block_data.get('nesting_mode', 'single')}\n"
        + (f"Attributes: {', '.join(attr_names[:20])}\n" if attr_names else "")
    )
    yield Document(
        page_content=block_text,
        metadata={
            "resource_name": resource_name,
            "block_path":    block_path,
            "chunk_type":    "block",
            "iac_type":      "terraform",
        },
    )

    for attr_name, attr_data in block_data.get("attributes", {}).items():
        attr_path = f"{block_path}.{attr_name}"
        desc = attr_data.get("description", "").strip()
        req  = "required" if attr_data.get("required") else "optional"
        text = (
            f"Resource: {resource_name}\n"
            f"Block attribute: {attr_path}\n"
            f"Type: {attr_data.get('type', 'unknown')}\n"
            f"Required: {req}\n"
            + (f"Description: {desc}\n" if desc else "")
        )
        yield Document(
            page_content=text,
            metadata={
                "resource_name": resource_name,
                "block_path":    attr_path,
                "chunk_type":    "block_attribute",
                "iac_type":      "terraform",
            },
        )

    for child_name, child_data in block_data.get("block_types", {}).items():
        yield from _flatten_block(
            resource_name, f"{block_path}.{child_name}", child_data, depth + 1
        )


def generate_documents(kg: dict) -> list[Document]:
    """Convert the full knowledge graph into LangChain Document chunks."""
    docs: list[Document] = []

    for resource_name, resource_data in kg.items():
        desc        = resource_data.get("description", "").strip()
        subcategory = resource_data.get("subcategory", "").strip()

        # --- Resource-level document ---
        resource_text = (
            f"Resource: {resource_name}\n"
            + (f"Description: {desc}\n" if desc else "")
            + (f"Category: {subcategory}\n" if subcategory else "")
        )
        docs.append(Document(
            page_content=resource_text,
            metadata={
                "resource_name": resource_name,
                "chunk_type":    "resource",
                "iac_type":      "terraform",
            },
        ))

        # --- Top-level attribute documents ---
        for attr_name, attr_data in resource_data.get("attributes", {}).items():
            attr_desc = attr_data.get("description", "").strip()
            req       = "required" if attr_data.get("required") else "optional"
            text = (
                f"Resource: {resource_name}\n"
                f"Attribute: {attr_name}\n"
                f"Type: {attr_data.get('type', 'unknown')}\n"
                f"Required: {req}\n"
                + (f"Description: {attr_desc}\n" if attr_desc else "")
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "resource_name": resource_name,
                    "attribute":     attr_name,
                    "chunk_type":    "attribute",
                    "iac_type":      "terraform",
                },
            ))

        # --- Block documents (recursive) ---
        for block_name, block_data in resource_data.get("block_types", {}).items():
            docs.extend(_flatten_block(resource_name, block_name, block_data))

        # --- HCL example documents ---
        for i, example_code in enumerate(resource_data.get("examples", [])):
            docs.append(Document(
                page_content=(
                    f"Terraform HCL example for {resource_name}:\n{example_code}"
                ),
                metadata={
                    "resource_name": resource_name,
                    "chunk_type":    "example",
                    "example_index": i,
                    "iac_type":      "terraform",
                },
            ))

    return docs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_vector_db() -> None:
    embeddings = _get_embeddings()

    print(f"[05] Loading {KG_FILE} ...")
    kg = json.loads(KG_FILE.read_text(encoding="utf-8"))
    print(f"[05] {len(kg):,} resources loaded.")

    print(f"[05] Generating document chunks ...")
    documents = generate_documents(kg)
    print(f"[05] {len(documents):,} chunks ready.")

    print(f"[05] Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} ...")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Drop and recreate for a clean rebuild — scoped to tf collection only.
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"[05] Dropping existing collection '{COLLECTION_NAME}' for clean rebuild ...")
        chroma_client.delete_collection(COLLECTION_NAME)

    print(f"[05] Embedding and ingesting into '{COLLECTION_NAME}' ...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        collection_metadata=COLLECTION_METADATA,  # hnsw:space=cosine
    )

    print(f"\n[05] Done.")
    print(f"     Provider   : {_PROVIDER}")
    print(f"     Model      : {EMBEDDING_MODEL}")
    print(f"     Chunks     : {len(documents):,}")
    print(f"     Collection : {COLLECTION_NAME} @ {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"     Distance   : cosine (hnsw:space=cosine)")
    print(f"     Normalised : True")


if __name__ == "__main__":
    build_vector_db()
