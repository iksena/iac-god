# scripts/graphrag/05_build_chromadb.py
import json
import os
import sys

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np

# ---------------------------------------------------------------------------
# Embedding provider — inline mirror of tools/embedding_provider.py so the
# build script has no dependency on the tools/ package.
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()
_DEFAULTS = {
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama":      "mxbai-embed-large",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULTS.get(_PROVIDER, _DEFAULTS["huggingface"]))
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
        arr = np.array(vecs, dtype=np.float32)
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
        print(f"[Build] Embedding provider: Ollama  model: {EMBEDDING_MODEL}  url: {OLLAMA_BASE_URL}")
        base = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return _NormalisedEmbeddings(base)  # Ollama does not normalise by default

    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[Build] Embedding provider: HuggingFace  model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # already normalised
    )


# ---------------------------------------------------------------------------
# ChromaDB connection
# ---------------------------------------------------------------------------

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "cfn_schema_properties"


def build_vector_db():
    embeddings = _get_embeddings()

    with open("cfn_knowledge_graph.json", "r") as f:
        kg_data = json.load(f)

    documents = []
    print("Chunking CloudFormation data...")

    for res_name, res_data in kg_data.items():
        res_description = res_data.get("description", "")

        for prop_name, prop_details in res_data.get("properties", {}).items():
            prop_type   = prop_details.get("Type", prop_details.get("PrimitiveType", "Unknown"))
            required    = prop_details.get("Required", False)
            update_type = prop_details.get("UpdateType", "Unknown")
            doc_url     = prop_details.get("Documentation", "")
            # Description is now populated by 03_parse_and_merge.py for resources
            # whose HTML doc was scraped. Falls back to empty string otherwise.
            description = prop_details.get("Description", "")

            text_content = (
                f"Resource: {res_name}\n"
                f"Resource Description: {res_description}\n"
                f"Property: {prop_name}\n"
                f"Property ID: {res_name}.{prop_name}\n"
                f"Type: {prop_type}\n"
                f"Required: {required}\n"
                f"Update Type: {update_type}\n"
                + (f"Description: {description}\n" if description else "")
                + f"Documentation: {doc_url}\n"
            )

            documents.append(Document(
                page_content=text_content,
                metadata={
                    "resource_name": res_name,
                    "property_name": prop_name,
                    "property_id":   f"{res_name}.{prop_name}",
                    "required":      required,
                    "update_type":   update_type,
                    "type":          prop_type,
                },
            ))

        for i, example_code in enumerate(res_data.get("examples", [])):
            documents.append(Document(
                page_content=f"CloudFormation example for {res_name}:\n{example_code}",
                metadata={
                    "resource_name": res_name,
                    "chunk_type":    "example",
                    "example_index": i,
                    "property_id":   None,
                },
            ))

    print(f"Created {len(documents)} document chunks. Ingesting into ChromaDB...")

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Delete existing collection if present so the distance metric and
    # vector space are always reset cleanly on rebuild.
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Dropping existing collection '{COLLECTION_NAME}' for clean rebuild...")
        chroma_client.delete_collection(COLLECTION_NAME)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        collection_metadata=COLLECTION_METADATA,   # hnsw:space=cosine
    )
    print("Vector database successfully built!")
    print(f"  Provider   : {_PROVIDER}")
    print(f"  Model      : {EMBEDDING_MODEL}")
    print(f"  Chunks     : {len(documents)}")
    print(f"  Collection : {COLLECTION_NAME} @ {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"  Distance   : cosine (hnsw:space=cosine)")
    print(f"  Normalised : True")


if __name__ == "__main__":
    build_vector_db()
