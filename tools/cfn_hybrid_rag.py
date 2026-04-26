"""tools/cfn_hybrid_rag.py

ChromaDB + Neo4j hybrid retrieval for CloudFormation schema context.

Dependency direction (strictly unidirectional, no cycles):
  retriever_agent  →  cfn_hybrid_rag  →  template_annotator (type hints only)
  retriever_agent  →  retriever_helpers  (no DB deps)
  cfn_hybrid_rag   does NOT import from agents/
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from neo4j import GraphDatabase

from tools.template_annotator import TemplateAnnotation

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ---------------------------------------------------------------------------
# Query-generation system prompt
# (consumed by retriever_agent; defined here to keep all RAG concerns together)
# ---------------------------------------------------------------------------
QUERY_GEN_SYSTEM = """\
You are an AWS CloudFormation schema expert. You are given:
1. A list of validation errors from a CFN template.
2. An annotated CloudFormation template where each resource block has inline
   # ERROR comments identifying which errors apply to that specific resource.
3. (Optionally) a structured annotation of the template's resources: their logical IDs,
   AWS resource types, source line numbers, and the property keys actually present.

Use the inline error comments as the primary signal for which Resource.Property
pairs need schema retrieval. Errors without a resource annotation are template-level
and should inform general structural queries. Use the annotation summary as
supplementary context when the annotated template is not available.

Using this context, generate a list of precise retrieval queries. Each query must target a
specific Resource.Property combination that is EITHER referenced in an error OR present in
the template annotation and relevant to the errors.

Output ONLY a JSON object with a single key "queries" whose value is an array of strings.
Example:
{
  "queries": [
    "What are the required properties for AWS::S3::Bucket BucketEncryption?",
    "What valid values exist for AWS::RDS::DBInstance DBInstanceClass?"
  ]
}

Prioritise resources that appear in the errors. Limit to at most 8 queries.
"""


@lru_cache(maxsize=1)
def _get_global_embeddings():
    """Lazy-load the embedding model once and cache it for the process lifetime.

    Deferred import intentional: HuggingFaceEmbeddings loads a ~400 MB model
    on first call. Importing at module level would slow startup even when
    embeddings are not needed (e.g. unit tests, dry-run mode).
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# Chroma chunk cleaner
# ---------------------------------------------------------------------------
_DOC_LINK_RE = re.compile(r"https?://docs\.aws\.amazon\.com\S*", re.IGNORECASE)
_DESCRIPTION_RE = re.compile(r"(?:^|\n)Description:\s*.+?(?=\n[A-Z]|\Z)", re.DOTALL)


def _clean_chroma_chunk(content: str) -> str:
    """Remove AWS doc links and Description fields from a Chroma property chunk.

    These fields bloat the Semantically Matched Properties section without
    providing schema information the remediator can act on.
    """
    content = _DOC_LINK_RE.sub("", content)
    content = _DESCRIPTION_RE.sub("", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------
def format_prompt_from_neo4j_result(resource_data: dict) -> str:
    """Format Neo4j schema data into a concise single-block prompt section."""
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
        shown = opt[:20]
        lines.append("Optional: " + ", ".join(
            f"{p['name']}({p['type']})" for p in shown
        ))
        if len(opt) > 20:
            lines.append(f"  ... and {len(opt) - 20} more optional properties")

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
        WITH r, required_properties, collect(DISTINCT {name: opt_prop.name, type: opt_prop.type}) AS optional_properties

        OPTIONAL MATCH (r)-[:HAS_NESTED_TYPE]->(nt:NestedType)
        WITH r, required_properties, optional_properties, collect(DISTINCT {name: nt.name, type: nt.type, required: nt.required}) AS nested_types

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
            "name": result["resource_name"],
            "description": result["resource_description"],
            "required_properties": result["required_properties"],
            "optional_properties": result["optional_properties"],
            "nested_types": result["nested_types"],
            "example": result["example"],
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
# Public retrieval entry point
# ---------------------------------------------------------------------------
def _execute_hybrid_retrieval(
    retrieval_queries: list[str],
    annotation: TemplateAnnotation | None,
) -> str:
    """Execute retrieval: ChromaDB semantic search followed by Neo4j schema lookup.

    The annotation is built upstream in retriever_agent before this call.
    If annotation is None (parse failure), retrieval proceeds with an empty
    resource seed — Chroma results may still populate identified_resources.
    """
    identified_resources: set[str] = (
        {r.resource_type for r in annotation.resources if r.resource_type}
        if annotation and annotation.resources
        else set()
    )

    property_chunks: list[str] = []
    if retrieval_queries:
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            vectorstore = Chroma(
                client=chroma_client,
                collection_name="cfn_schema_properties",
                embedding_function=_get_global_embeddings(),
            )

            seen_resources: set[str] = set()
            for query in retrieval_queries:
                for chunk in vectorstore.similarity_search(query, k=3):
                    meta = chunk.metadata
                    res = meta.get("resource_name", "")
                    prop = meta.get("property_name", "") or meta.get("property_path", "")
                    prop_key = (
                        f"{res}.{prop}" if prop
                        else f"{res}::content::{hash(chunk.page_content)}"
                    )
                    if prop_key in seen_resources:
                        continue
                    seen_resources.add(prop_key)
                    cleaned = _clean_chroma_chunk(chunk.page_content)
                    if cleaned:
                        property_chunks.append(cleaned)
                    if res:
                        identified_resources.add(res)
        except Exception as e:
            print(f"[RAG Tool] Warning: ChromaDB Semantic Search failed. {e}")

    if not identified_resources:
        return "No specific AWS resources identified in template or retrieval context."

    print(f"[RAG Tool] Stage 2: Querying Neo4j for {len(identified_resources)} resources...")
    final_context_blocks: list[str] = ["## Official AWS CloudFormation Schema Context\n"]

    if property_chunks:
        final_context_blocks.append(
            "### Semantically Matched Properties\n" + "\n---\n".join(property_chunks)
        )

    try:
        with _neo4j_driver() as driver:
            seen_neo4j: set[str] = set()
            for resource in sorted(identified_resources):
                if resource in seen_neo4j:
                    print(f"[RAG Tool] Skipping duplicate schema: {resource}")
                    continue
                seen_neo4j.add(resource)
                res_data = query_knowledge_graph(driver, resource)
                if "error" not in res_data:
                    final_context_blocks.append(format_prompt_from_neo4j_result(res_data))
    except Exception as e:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {e}")
        return "Failed to connect to Knowledge Graph."

    return "\n\n".join(final_context_blocks)