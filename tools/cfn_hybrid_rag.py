"""tools/cfn_hybrid_rag.py

ChromaDB + Neo4j hybrid retrieval for CloudFormation schema context.

Retrieval runs in two sequential stages:
  Stage 1 — _semantic_search():        ChromaDB similarity search over pre-indexed
                                        CFN property chunks. Returns chunks GROUPED
                                        by resource name and any resource names
                                        surfaced by chunk metadata.
  Stage 2 — _graph_schema_lookup():    Neo4j Cypher traversal for each identified
                                        resource. Returns structured schema blocks
                                        (required properties, capped optional
                                        properties, nested types, examples).
  Final   — _assemble_retrieval_context(): merges both sets into the single context
                                        string consumed by the remediator prompt.

Dependency direction (strictly unidirectional, no cycles):
  retriever_agent  →  cfn_hybrid_rag       →  template_annotator (type hints only)
  retriever_agent  →  retriever_helpers    (no DB deps)
  cfn_hybrid_rag   does NOT import from agents/
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# Re-export so existing callers of
#   from tools.cfn_hybrid_rag import QUERY_GEN_SYSTEM
# continue to work without modification.
from prompts.retriever_prompt import QUERY_GEN_SYSTEM  # noqa: F401

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Maximum number of optional properties shown per resource in the Neo4j block.
# Keeping this low prevents the context from being dominated by rarely-used
# optional fields that are irrelevant to the current errors.
_MAX_OPTIONAL_PROPS = 10


@lru_cache(maxsize=1)
def _get_global_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load the embedding model once and cache it for the process lifetime."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# Chroma chunk cleaner
# ---------------------------------------------------------------------------
_DOC_LINK_RE    = re.compile(r"https?://docs\.aws\.amazon\.com\S*", re.IGNORECASE)
_DESCRIPTION_RE = re.compile(r"(?:^|\n)Description:\s*.+?(?=\n[A-Z]|\Z)", re.DOTALL)


def _clean_chroma_chunk(content: str) -> str:
    """Remove AWS doc links and Description fields from a Chroma property chunk."""
    content = _DOC_LINK_RE.sub("", content)
    content = _DESCRIPTION_RE.sub("", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def format_prompt_from_neo4j_result(resource_data: dict) -> str:
    """Format Neo4j schema data into a concise single-block prompt section.

    Optional properties are capped at _MAX_OPTIONAL_PROPS (10) to keep the
    context block focused on the properties most likely to appear in errors.
    """
    if "error" in resource_data:
        return f"# {resource_data['error']}"

    lines = [f"### {resource_data['name']}"]

    req = resource_data.get("required_properties") or []
    opt = resource_data.get("optional_properties") or []

    if req:
        lines.append("Required: " + ", ".join(
            f"{p['name']}({p['type']})" for p in req
        ))
    if opt:
        shown = opt[:_MAX_OPTIONAL_PROPS]
        lines.append("Optional: " + ", ".join(
            f"{p['name']}({p['type']})" for p in shown
        ))
        if len(opt) > _MAX_OPTIONAL_PROPS:
            lines.append(f"  ... and {len(opt) - _MAX_OPTIONAL_PROPS} more optional properties")

    if resource_data.get("nested_types"):
        nt_list = ", ".join(
            f"{nt['name']}({'req' if nt['required'] else 'opt'})"
            for nt in resource_data["nested_types"]
        )
        lines.append(f"NestedTypes: {nt_list}")

    if resource_data.get("example"):
        lines.append(f"Example:\n```yaml\n{resource_data['example']['code']}\n```")

    return "\n".join(lines)


def query_knowledge_graph(driver, resource_name: str) -> dict:
    """Execute the Cypher traversal to pull the schema structure for a resource."""
    CYPHER_QUERY = """
        MATCH (r:Resource {name: $resource_name})

        OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(prop:Property)
        WHERE prop.required = true
        WITH r, collect(DISTINCT {name: prop.name, type: prop.type}) AS required_properties

        OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(opt_prop:Property)
        WHERE opt_prop.required = false
        WITH r, required_properties,
             collect(DISTINCT {name: opt_prop.name, type: opt_prop.type}) AS optional_properties

        OPTIONAL MATCH (r)-[:HAS_NESTED_TYPE]->(nt:NestedType)
        WITH r, required_properties, optional_properties,
             collect(DISTINCT {name: nt.name, type: nt.type, required: nt.required}) AS nested_types

        OPTIONAL MATCH (r)-[:HAS_EXAMPLE]->(e:Example)
        WHERE e.index = 0

        RETURN r.name AS resource_name,
               r.description AS resource_description,
               required_properties,
               optional_properties,
               nested_types,
               { code: e.code } AS example
    """
    with driver.session() as session:
        result = session.run(CYPHER_QUERY, resource_name=resource_name).single()
        if not result:
            return {"error": f"Resource '{resource_name}' not found in Knowledge Graph."}
        return {
            "name":                result["resource_name"],
            "description":        result["resource_description"],
            "required_properties": result["required_properties"],
            "optional_properties": result["optional_properties"],
            "nested_types":        result["nested_types"],
            "example":             result["example"],
        }


@contextmanager
def _neo4j_driver():
    """Context manager that opens a Neo4j driver and ensures it is closed."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Stage 1 — Semantic search (ChromaDB), chunks grouped by resource
# ---------------------------------------------------------------------------

def _semantic_search(
    retrieval_queries: list[str],
) -> tuple[dict[str, list[str]], set[str]]:
    """Run ChromaDB similarity search for all retrieval queries.

    Returns:
        chunks_by_resource: Dict mapping resource_name → list of cleaned property
                            text chunks for that resource. Chunks from different
                            queries that share the same resource are grouped together
                            so _assemble_retrieval_context can render one sub-section
                            per resource instead of a flat interleaved list.
        found_resources:    Set of AWS resource type names surfaced from metadata.
    """
    chunks_by_resource: dict[str, list[str]] = defaultdict(list)
    found_resources: set[str] = set()

    if not retrieval_queries:
        return chunks_by_resource, found_resources

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="cfn_schema_properties",
            embedding_function=_get_global_embeddings(),
        )

        seen_prop_keys: set[str] = set()
        for query in retrieval_queries:
            for chunk in vectorstore.similarity_search(query, k=3):
                meta = chunk.metadata
                res  = meta.get("resource_name", "") or "_unknown"
                prop = meta.get("property_name", "") or meta.get("property_path", "")
                prop_key = (
                    f"{res}.{prop}" if prop
                    else f"{res}::content::{hash(chunk.page_content)}"
                )
                if prop_key in seen_prop_keys:
                    continue
                seen_prop_keys.add(prop_key)
                cleaned = _clean_chroma_chunk(chunk.page_content)
                if cleaned:
                    chunks_by_resource[res].append(cleaned)
                if res and res != "_unknown":
                    found_resources.add(res)
    except Exception as exc:
        print(f"[RAG Tool] Warning: ChromaDB Semantic Search failed. {exc}")

    return dict(chunks_by_resource), found_resources


# ---------------------------------------------------------------------------
# Stage 2 — Knowledge-graph schema lookup (Neo4j)
# ---------------------------------------------------------------------------

def _graph_schema_lookup(resources: set[str]) -> list[str]:
    """Fetch full schema blocks for each resource from Neo4j.

    Returns a list of formatted schema block strings (one per resource).
    Returns an empty list on connection failure.
    """
    schema_blocks: list[str] = []

    if not resources:
        return schema_blocks

    print(f"[RAG Tool] Stage 2: Querying Neo4j for {len(resources)} resources...")
    try:
        with _neo4j_driver() as driver:
            seen: set[str] = set()
            for resource in sorted(resources):
                if resource in seen:
                    continue
                seen.add(resource)
                res_data = query_knowledge_graph(driver, resource)
                if "error" not in res_data:
                    schema_blocks.append(format_prompt_from_neo4j_result(res_data))
    except Exception as exc:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {exc}")

    return schema_blocks


# ---------------------------------------------------------------------------
# Final — Context assembly
# ---------------------------------------------------------------------------

def _assemble_retrieval_context(
    chunks_by_resource: dict[str, list[str]],
    schema_blocks: list[str],
) -> str:
    """Merge grouped Chroma chunks and Neo4j schema blocks into a single
    context string for the remediator prompt.

    Chroma chunks are rendered one sub-section per resource so the remediator
    can immediately associate each property snippet with its resource type,
    rather than scanning an interleaved flat list.
    """
    final_blocks: list[str] = ["## Official AWS CloudFormation Schema Context\n"]

    if chunks_by_resource:
        resource_sections: list[str] = []
        for resource_name, chunks in sorted(chunks_by_resource.items()):
            header = (
                f"#### {resource_name}" if resource_name != "_unknown"
                else "#### (resource type unknown)"
            )
            resource_sections.append(header + "\n" + "\n---\n".join(chunks))
        final_blocks.append(
            "### Semantically Matched Properties\n"
            + "\n\n".join(resource_sections)
        )

    final_blocks.extend(schema_blocks)
    return "\n\n".join(final_blocks)


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------

def execute_hybrid_retrieval(
    retrieval_queries: list[str],
    seed_resources: set[str],
) -> str:
    """Execute full hybrid retrieval: ChromaDB semantic search → Neo4j schema lookup.

    Args:
        retrieval_queries: HyDE queries generated upstream by the retriever agent.
        seed_resources:    AWS resource type names pre-extracted from the template
                           annotation by the caller (via extract_resource_types()).
                           Chroma results may augment this set further.

    Returns:
        A multi-section context string for the remediator, or a short fallback
        message when no resources could be identified.
    """
    chunks_by_resource, chroma_resources = _semantic_search(retrieval_queries)
    identified_resources = seed_resources | chroma_resources

    if not identified_resources:
        return "No specific AWS resources identified in template or retrieval context."

    schema_blocks = _graph_schema_lookup(identified_resources)
    if not schema_blocks and not chunks_by_resource:
        return "Failed to connect to Knowledge Graph."

    return _assemble_retrieval_context(chunks_by_resource, schema_blocks)
