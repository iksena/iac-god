# scripts/graphrag/05_build_chromadb.py
import json
import os
import sys
import time

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

# Ollama's local model runner tokenizes/embeds an entire request in one shot.
# langchain_chroma hands whole batches (up to ChromaDB's internal batch cap —
# often several thousand documents) to embed_documents() in a single call,
# which can crash the Ollama runner subprocess under memory/context pressure.
# The client then sees a bare connection reset, surfaced as
# `ollama._types.ResponseError: Post ".../tokenize": EOF (status code: 400)` —
# nothing is actually wrong with any individual chunk of text.
# Sub-batching here (independent of whatever batch size Chroma passes in)
# keeps each Ollama request small enough not to trigger this, and retries
# with backoff absorb transient runner hiccups instead of killing the whole
# 13k-document build.
OLLAMA_EMBED_BATCH_SIZE = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "16"))
OLLAMA_EMBED_MAX_RETRIES = int(os.getenv("OLLAMA_EMBED_MAX_RETRIES", "4"))
OLLAMA_EMBED_RETRY_BASE_S = float(os.getenv("OLLAMA_EMBED_RETRY_BASE_S", "2.0"))

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

    For the Ollama provider, embed_documents() also sub-batches and retries
    internally (see OLLAMA_EMBED_* above) to work around the runner crashing
    on oversized single requests.
    """
    def __init__(self, base, sub_batch_size=None):
        self._base = base
        # None disables sub-batching (e.g. HuggingFace runs locally in-process
        # and doesn't hit this failure mode).
        self._sub_batch_size = sub_batch_size

    @staticmethod
    def _norm(vecs):
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).tolist()

    def _embed_batch_with_retry(self, texts):
        """Embed a single (small) batch, retrying with backoff on transient
        Ollama runner failures. On persistent failure, bisects the batch to
        isolate a poison document rather than failing the whole build."""
        last_exc = None
        for attempt in range(OLLAMA_EMBED_MAX_RETRIES):
            try:
                return self._base.embed_documents(texts)
            except Exception as exc:  # ollama.ResponseError, ConnectionError, etc.
                last_exc = exc
                wait_s = OLLAMA_EMBED_RETRY_BASE_S * (2 ** attempt)
                print(f"    [embed] batch of {len(texts)} failed "
                      f"(attempt {attempt + 1}/{OLLAMA_EMBED_MAX_RETRIES}): "
                      f"{type(exc).__name__}: {exc}. Retrying in {wait_s:.0f}s...")
                time.sleep(wait_s)

        if len(texts) == 1:
            # Can't bisect further — this single document is the problem.
            preview = texts[0][:200].replace("\n", " ")
            print(f"    [embed] GIVING UP on 1 document after "
                  f"{OLLAMA_EMBED_MAX_RETRIES} retries: {preview!r}")
            raise last_exc

        mid = len(texts) // 2
        print(f"    [embed] bisecting batch of {len(texts)} into "
              f"{mid} + {len(texts) - mid} to isolate the failure...")
        left = self._embed_batch_with_retry(texts[:mid])
        right = self._embed_batch_with_retry(texts[mid:])
        return left + right

    def embed_documents(self, texts):
        if not self._sub_batch_size or len(texts) <= self._sub_batch_size:
            vecs = self._embed_batch_with_retry(texts) if self._sub_batch_size else self._base.embed_documents(texts)
            return self._norm(vecs)

        all_vecs = []
        n = len(texts)
        n_batches = (n + self._sub_batch_size - 1) // self._sub_batch_size
        for i in range(0, n, self._sub_batch_size):
            batch_idx = i // self._sub_batch_size + 1
            batch = texts[i:i + self._sub_batch_size]
            print(f"    [embed] batch {batch_idx}/{n_batches} ({len(batch)} docs)...")
            all_vecs.extend(self._embed_batch_with_retry(batch))
        return self._norm(all_vecs)

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
        print(f"[Build]   sub-batch size: {OLLAMA_EMBED_BATCH_SIZE} "
              f"(override with OLLAMA_EMBED_BATCH_SIZE env var)")
        base = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return _NormalisedEmbeddings(base, sub_batch_size=OLLAMA_EMBED_BATCH_SIZE)

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
