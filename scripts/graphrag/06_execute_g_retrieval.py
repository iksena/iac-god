# scripts/graphrag/06_execute_g_retrieval.py
#
# Standalone test harness for the CFN G-Retrieval pipeline.
# Mirrors the two-stage retrieval used by tools/cfn_hybrid_rag.py so
# you can validate the full ChromaDB → Neo4j flow before running the agent.
#
# Embedding provider is controlled by the same env vars as the build script
# and the RAG tools, so the model is always consistent:
#
#   EMBEDDING_PROVIDER=ollama python 06_execute_g_retrieval.py
#   EMBEDDING_PROVIDER=huggingface python 06_execute_g_retrieval.py
#
# See tools/embedding_provider.py and scripts/graphrag/README.md for details.

import os
import sys
import chromadb
from langchain_chroma import Chroma

from neo4j_client import query_knowledge_graph, format_prompt_from_neo4j_result

# ---------------------------------------------------------------------------
# Embedding provider — inline mirror of tools/embedding_provider.py so this
# standalone script has no dependency on the tools/ package.
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower().strip()
_DEFAULTS = {
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama":      "mxbai-embed-large",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULTS.get(_PROVIDER, _DEFAULTS["huggingface"]))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_embeddings():
    if _PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            print("ERROR: langchain-ollama not installed. Run: pip install langchain-ollama")
            sys.exit(1)
        print(f"[Embeddings] Provider: Ollama  model: {EMBEDDING_MODEL}  url: {OLLAMA_BASE_URL}")
        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[Embeddings] Provider: HuggingFace  model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# ChromaDB / Neo4j connection config
# ---------------------------------------------------------------------------

CHROMA_HOST       = os.getenv("CHROMA_HOST",  "localhost")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME   = "cfn_schema_properties"   # must match 05_build_chromadb.py
DISTANCE_THRESHOLD = float(os.getenv("CHROMA_DISTANCE_THRESHOLD", "0.55"))
TOP_K             = int(os.getenv("CHROMA_TOP_K", "5"))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def execute_g_retrieval(user_query: str) -> str | None:
    print(f"\n--- G-Retrieval for query: '{user_query}' ---\n")

    # ======================================================================
    # STAGE 1: Semantic search (ChromaDB)
    # ======================================================================
    print("Stage 1: Performing Semantic Search in ChromaDB...")

    embeddings = _get_embeddings()
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Verify the collection exists before querying so the error message is
    # actionable rather than a raw ChromaDB exception.
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME not in existing_collections:
        print(
            f"ERROR: Collection '{COLLECTION_NAME}' not found in ChromaDB.\n"
            f"       Run '05_build_chromadb.py' first to build the index.\n"
            f"       Available collections: {existing_collections or ['(none)']}"
        )
        return None

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Use similarity_search_with_score to apply the distance threshold.
    # Lower score = more similar. Chunks above the threshold are discarded.
    scored_chunks = vectorstore.similarity_search_with_score(user_query, k=TOP_K)

    if not scored_chunks:
        print("WARNING: Vector search returned no results. Check ChromaDB connection.")
        return None

    kept_chunks = [
        (doc, score) for doc, score in scored_chunks
        if score <= DISTANCE_THRESHOLD
    ]
    dropped = len(scored_chunks) - len(kept_chunks)
    print(
        f"  Retrieved {len(scored_chunks)} chunks, "
        f"kept {len(kept_chunks)} (distance ≤ {DISTANCE_THRESHOLD}), "
        f"dropped {dropped}."
    )

    if not kept_chunks:
        print(
            f"WARNING: All {len(scored_chunks)} chunks exceeded the distance "
            f"threshold ({DISTANCE_THRESHOLD}). Try raising CHROMA_DISTANCE_THRESHOLD "
            f"or rephrasing the query."
        )
        return None

    # Collect unique resource names from chunk metadata.
    identified_resources: set[str] = set()
    for doc, score in kept_chunks:
        res = doc.metadata.get("resource_name", "").strip()
        if res:
            identified_resources.add(res)
            print(f"  ✓ {res} (distance={score:.4f})")

    if not identified_resources:
        print("WARNING: No resource_name metadata found in matched chunks.")
        return None

    print(f"\n  Entry points identified: {sorted(identified_resources)}\n")

    # ======================================================================
    # STAGE 2: Graph traversal (Neo4j)
    # ======================================================================
    print("Stage 2: Performing Graph Traversal in Neo4j...")

    final_llm_context = ""
    for resource in sorted(identified_resources):
        print(f"  Extracting subgraph for {resource}...")
        resource_data = query_knowledge_graph(resource_name=resource)
        formatted = format_prompt_from_neo4j_result(resource_data)
        final_llm_context += formatted + "\n\n"

    # ======================================================================
    # FINAL: Assemble prompt
    # ======================================================================
    final_prompt = f"""You are an expert AWS CloudFormation architect.
Use the following official documentation context to answer the user query.

CONTEXT:
{final_llm_context.strip()}

USER QUERY: {user_query}

Generate the appropriate CloudFormation YAML code."""

    return final_prompt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CFN G-Retrieval pipeline.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Create an Amazon S3 bucket that has versioning enabled.",
        help="Natural-language query to retrieve CFN schema context for.",
    )
    args = parser.parse_args()

    result = execute_g_retrieval(args.query)

    if result:
        print("\n--- Final prompt to send to LLM ---")
        print(result)
    else:
        print("\nRetrieval failed. See warnings above.")
        sys.exit(1)
