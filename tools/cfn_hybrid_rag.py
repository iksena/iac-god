"""tools/cfn_hybrid_rag.py

ChromaDB + Neo4j hybrid retrieval for CloudFormation schema context.

Retrieval runs in two sequential stages:
  Stage 1 — _semantic_search():        ChromaDB similarity search over pre-indexed
                                        CFN property chunks. Only chunks whose
                                        raw distance score is AT OR BELOW
                                        CHROMA_DISTANCE_THRESHOLD are kept.
                                        Results are filtered to seed_resources so
                                        that unrelated resource types (e.g. Lambda,
                                        Redshift) cannot leak in via coincidental
                                        semantic similarity.
                                        Results are grouped by resource name;
                                        the resource description is printed once
                                        and properties are deduplicated.
  Stage 2 — _graph_schema_lookup():    Neo4j Cypher traversal for each identified
                                        resource. When error_resources is supplied,
                                        only those resources (plus any chroma-covered
                                        ones) are fetched — avoiding full-template
                                        schema dumps when only 1-2 resources have
                                        active errors.
  Final   — _assemble_retrieval_context(): merges both sets into the single context
                                        string consumed by the remediator prompt.

Context verbosity is controlled by CHROMA_CONTEXT_MODE:
  compact (default) — ChromaDB chunks grouped per resource; description printed
                      once; each property rendered as a single summary line.
                      Neo4j example YAML omitted for resources already covered
                      by ChromaDB semantic hits (saves ~45% tokens).
  raw               — Original behaviour: full chunk text per property, no
                      deduplication of resource descriptions. Useful for
                      debugging retrieval quality.

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
from dataclasses import dataclass, field

import chromadb
from langchain_chroma import Chroma
from neo4j import GraphDatabase

from tools.embedding_provider import (
    get_embeddings,
    CHROMA_COLLECTION_METADATA,
    DEFAULT_DISTANCE_THRESHOLD,
)

# Re-export for back-compat: existing callers of
#   from tools.cfn_hybrid_rag import QUERY_GEN_SYSTEM
# continue to work without modification.
from prompts.retriever_prompt import QUERY_GEN_SYSTEM  # noqa: F401

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Raw cosine distance floor for Chroma property chunks.
#
# The collection is created with hnsw:space="cosine" and all vectors are
# unit-normalised, so scores are true cosine distances in [0, 2]:
#   0 = identical direction, 1 = orthogonal, 2 = opposite direction.
# LOWER score = MORE similar.  Chunks ABOVE the threshold are discarded.
#
# Model-specific calibration (cosine distance, normalised vectors):
#
#   mxbai-embed-large (1024-dim, Ollama):
#     0.00 – 0.20  near-exact match
#     0.20 – 0.40  topically related   ← default threshold keeps these
#     0.40 – 0.65  loosely related     ← default threshold drops these
#     0.65+        unrelated
#
#   all-mpnet-base-v2 (768-dim, HuggingFace):
#     0.00 – 0.30  near-exact match
#     0.30 – 0.55  topically related
#     0.55 – 0.80  loosely related
#     0.80+        unrelated
#
# Default is 0.40 (conservative for mxbai-embed-large).
# Raise toward 0.55 for higher recall; lower toward 0.25 for higher precision.
# Override at runtime: CHROMA_DISTANCE_THRESHOLD=0.50 python main.py
CHROMA_DISTANCE_THRESHOLD: float = DEFAULT_DISTANCE_THRESHOLD

# Context verbosity mode.
# "compact" (default): group chunks per resource, deduplicate description,
#                      render each property as a single summary line.
# "raw":               legacy full-chunk text, one block per property.
# Override: CHROMA_CONTEXT_MODE=raw python main.py
_CONTEXT_MODE: str = os.getenv("CHROMA_CONTEXT_MODE", "compact").lower().strip()

# Maximum number of optional properties shown per resource in the Neo4j block.
_MAX_OPTIONAL_PROPS = 10


# ---------------------------------------------------------------------------
# Chunk parsing — structured property data extracted from raw chunk text
# ---------------------------------------------------------------------------

@dataclass
class _PropertyChunk:
    """Parsed representation of one CFN property chunk from ChromaDB."""
    resource_name: str = ""
    resource_description: str = ""
    property_name: str = ""
    property_type: str = ""
    required: str = ""
    update_type: str = ""
    is_example: bool = False
    raw_text: str = ""  # fallback for unparseable chunks


_FIELD_RE = re.compile(
    r"^(?P<key>Resource|Resource Description|Property|Type|Required|Update Type)"
    r":\s*(?P<value>.+)$",
    re.MULTILINE,
)
_DOC_LINK_RE = re.compile(r"https?://docs\.aws\.amazon\.com\S*", re.IGNORECASE)


def _parse_chunk(content: str, meta: dict) -> _PropertyChunk:
    """Parse a raw ChromaDB chunk text into a structured _PropertyChunk.

    Falls back to storing raw_text when the chunk doesn't match the
    standard property format (e.g. example code chunks).
    """
    # Example chunks have a distinct prefix
    if content.lstrip().startswith("CloudFormation example for"):
        resource = meta.get("resource_name", "") or ""
        return _PropertyChunk(resource_name=resource, is_example=True, raw_text=content)

    fields: dict[str, str] = {}
    for m in _FIELD_RE.finditer(content):
        fields[m.group("key")] = m.group("value").strip()

    if not fields.get("Property"):
        # Not a structured property chunk — keep as raw
        return _PropertyChunk(
            resource_name=meta.get("resource_name", ""),
            raw_text=_DOC_LINK_RE.sub("", content).strip(),
        )

    return _PropertyChunk(
        resource_name=fields.get("Resource", meta.get("resource_name", "")),
        resource_description=fields.get("Resource Description", ""),
        property_name=fields.get("Property", ""),
        property_type=fields.get("Type", ""),
        required=fields.get("Required", ""),
        update_type=fields.get("Update Type", ""),
    )


# ---------------------------------------------------------------------------
# Grouped resource data — accumulates all property chunks for one resource
# ---------------------------------------------------------------------------

@dataclass
class _ResourceChunks:
    name: str
    description: str = ""
    properties: list[_PropertyChunk] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    raw_chunks: list[str] = field(default_factory=list)  # unparseable fallbacks


def _format_resource_block_compact(rc: _ResourceChunks) -> str:
    """Render one resource's ChromaDB hits as a compact grouped block.

    The resource description is printed once (deduplicated across all chunks
    for the same resource). Each property is a single summary line.

    Format:
        #### AWS::EC2::SecurityGroup
        > Specifies a security group.

        - IpProtocol (String, Immutable)  [required]
        - FromPort (Integer, Immutable)
        - ToPort (Integer, Immutable)
    """
    lines: list[str] = []
    header = f"#### {rc.name}" if rc.name and rc.name != "_unknown" \
        else "#### (resource type unknown)"
    lines.append(header)

    # Resource description — printed once, stripped of embedded links and
    # boilerplate cross-references.
    desc = _DOC_LINK_RE.sub("", rc.description).strip()
    desc = re.sub(r"\s+see[^.]+\.", ".", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s{2,}", " ", desc).strip()
    if desc:
        lines.append(f"> {desc}")

    # Property summary lines — one line per unique property
    for p in rc.properties:
        parts = [p.property_name]
        type_update = ""
        if p.property_type:
            type_update = p.property_type
            if p.update_type and p.update_type.lower() not in ("unknown", ""):
                type_update += f", {p.update_type}"
        if type_update:
            parts.append(f"({type_update})")
        req_flag = "  [required]" if str(p.required).lower() == "true" else ""
        lines.append(f"- {'  '.join(parts)}{req_flag}")

    # Fallback raw lines (unparseable chunks)
    for raw in rc.raw_chunks:
        lines.append(raw)

    return "\n".join(lines)


def _format_resource_block_raw(chunks: list[str]) -> str:
    """Legacy: join raw cleaned chunks with separators (original behaviour)."""
    return "\n---\n".join(chunks)


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def format_prompt_from_neo4j_result(
    resource_data: dict,
    include_example: bool = True,
) -> str:
    """Format Neo4j schema data into a concise single-block prompt section.

    Args:
        resource_data:   Dict returned by query_knowledge_graph().
        include_example: When False the YAML example block is omitted.
                         Callers set this to False when ChromaDB already
                         provided semantic context for this resource,
                         avoiding ~50–80 lines of redundant YAML skeleton.
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

    # Only include the full YAML example when the caller requests it
    # (i.e. this resource has no ChromaDB semantic context).
    if include_example and resource_data.get("example"):
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
    resource_filter: set[str] | None = None,
) -> tuple[dict[str, _ResourceChunks], set[str]]:
    """Run ChromaDB similarity search for all retrieval queries.

    Chunks are filtered to resource_filter (when provided) so that only
    resources present in the template can appear in the output. This prevents
    unrelated resource types from leaking in via coincidental semantic
    similarity (e.g. Lambda or Redshift appearing for an EC2-only template).

    In compact mode, parses each chunk into structured _PropertyChunk data
    and accumulates them into per-resource _ResourceChunks groups so the
    resource description is deduplicated at render time.

    In raw mode, stores cleaned full chunk text as before.

    Args:
        retrieval_queries: HyDE queries from the retriever agent.
        resource_filter:   When provided, only chunks whose resource_name is
                           in this set are kept. Pass seed_resources here.

    Returns:
        resource_chunks:  Dict mapping resource_name → _ResourceChunks.
        found_resources:  Set of AWS resource type names from passing chunks.
    """
    resource_chunks: dict[str, _ResourceChunks] = defaultdict(
        lambda: _ResourceChunks(name="")
    )
    found_resources: set[str] = set()

    if not retrieval_queries:
        return dict(resource_chunks), found_resources

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="cfn_schema_properties",
            embedding_function=get_embeddings(),
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )

        seen_prop_keys: set[str] = set()
        kept = 0
        dropped = 0
        filtered_out = 0

        for query in retrieval_queries:
            scored_chunks = vectorstore.similarity_search_with_score(query, k=3)
            for chunk, score in scored_chunks:
                if score > CHROMA_DISTANCE_THRESHOLD:
                    dropped += 1
                    continue

                meta = chunk.metadata
                res  = meta.get("resource_name", "") or "_unknown"

                # Drop chunks for resources not present in the template.
                # This prevents unrelated resources from polluting the context
                # via coincidental semantic similarity.
                if resource_filter and res not in resource_filter:
                    filtered_out += 1
                    continue

                prop = meta.get("property_name", "") or meta.get("property_path", "")
                prop_key = (
                    f"{res}.{prop}" if prop
                    else f"{res}::content::{hash(chunk.page_content)}"
                )
                if prop_key in seen_prop_keys:
                    continue
                seen_prop_keys.add(prop_key)

                rc = resource_chunks[res]
                rc.name = rc.name or res

                if _CONTEXT_MODE == "compact":
                    parsed = _parse_chunk(chunk.page_content, meta)
                    if parsed.is_example:
                        rc.examples.append(parsed.raw_text)
                    elif parsed.property_name:
                        # Capture description from first chunk seen for this resource;
                        # subsequent chunks for the same resource are deduplicated here.
                        if not rc.description and parsed.resource_description:
                            rc.description = parsed.resource_description
                        rc.properties.append(parsed)
                    else:
                        rc.raw_chunks.append(parsed.raw_text)
                else:
                    # raw mode: store cleaned text as before
                    cleaned = _clean_chunk_raw(chunk.page_content)
                    if cleaned:
                        rc.raw_chunks.append(cleaned)

                kept += 1
                if res and res != "_unknown":
                    found_resources.add(res)

        print(
            f"[RAG Tool] Stage 1: {kept + dropped + filtered_out} chunks retrieved, "
            f"{dropped} dropped (distance > {CHROMA_DISTANCE_THRESHOLD}), "
            f"{filtered_out} filtered (not in template), "
            f"{kept} kept  [mode={_CONTEXT_MODE}]."
        )

    except Exception as exc:
        print(f"[RAG Tool] Warning: ChromaDB Semantic Search failed. {exc}")

    return dict(resource_chunks), found_resources


def _clean_chunk_raw(content: str) -> str:
    """Legacy raw-mode cleaner: strip doc links and Description fields."""
    content = _DOC_LINK_RE.sub("", content)
    content = re.sub(r"(?:^|\n)Description:\s*.+?(?=\n[A-Z]|\Z)", "", content, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


# ---------------------------------------------------------------------------
# Stage 2 — Knowledge-graph schema lookup (Neo4j)
# ---------------------------------------------------------------------------

def _graph_schema_lookup(
    resources: set[str],
    chroma_covered: set[str],
    error_resources: set[str] | None = None,
) -> list[str]:
    """Fetch full schema blocks for each resource from Neo4j.

    When error_resources is provided, only those resources are fetched from
    Neo4j (plus any chroma-covered resources that already have semantic hits).
    This avoids dumping full schema blocks for every resource in a large
    template when only 1-2 resources have active validation errors.

    When error_resources is None (no error signal), falls back to fetching
    all resources — preserving the original behaviour for initial generation.

    Args:
        resources:       All identified AWS resource type names (seed + chroma).
        chroma_covered:  Resources that already have ChromaDB semantic context.
                         For these, the Neo4j YAML example is omitted in
                         compact mode to avoid redundant tokens.
        error_resources: Resource types extracted from active cfn-lint / deploy
                         errors. When supplied, Neo4j lookups are scoped to
                         these plus chroma_covered resources only.
    """
    schema_blocks: list[str] = []

    if not resources:
        return schema_blocks

    # Determine the target set for Neo4j lookups.
    # - If error_resources is given: fetch only erroring resources + those
    #   that have chroma semantic hits (they may have relevant property context).
    # - Otherwise: fetch all identified resources (original behaviour).
    if error_resources:
        target_resources = error_resources | (resources & chroma_covered)
        print(
            f"[RAG Tool] Stage 2: Scoped to {len(target_resources)} error/chroma resources "
            f"(skipping {len(resources) - len(target_resources)} non-erroring resources)."
        )
    else:
        target_resources = resources
        print(f"[RAG Tool] Stage 2: Querying Neo4j for {len(target_resources)} resources...")

    try:
        with _neo4j_driver() as driver:
            seen: set[str] = set()
            for resource in sorted(target_resources):
                if resource in seen:
                    continue
                seen.add(resource)
                res_data = query_knowledge_graph(driver, resource)
                if "error" not in res_data:
                    # Omit the YAML example when compact mode is active AND
                    # ChromaDB already returned semantic hits for this resource.
                    include_ex = not (
                        _CONTEXT_MODE == "compact" and resource in chroma_covered
                    )
                    schema_blocks.append(
                        format_prompt_from_neo4j_result(res_data, include_example=include_ex)
                    )
    except Exception as exc:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {exc}")

    return schema_blocks


# ---------------------------------------------------------------------------
# Final — Context assembly
# ---------------------------------------------------------------------------

def _assemble_retrieval_context(
    resource_chunks: dict[str, _ResourceChunks],
    schema_blocks: list[str],
) -> str:
    """Merge grouped ChromaDB resource blocks and Neo4j schema blocks into a
    single context string for the remediator prompt.
    """
    final_blocks: list[str] = ["## Official AWS CloudFormation Schema Context\n"]

    if resource_chunks:
        resource_sections: list[str] = []
        for resource_name, rc in sorted(resource_chunks.items()):
            if _CONTEXT_MODE == "compact":
                block = _format_resource_block_compact(rc)
            else:
                # raw mode: rc.raw_chunks holds the cleaned text strings
                header = (
                    f"#### {resource_name}" if resource_name != "_unknown"
                    else "#### (resource type unknown)"
                )
                block = header + "\n" + _format_resource_block_raw(rc.raw_chunks)
            resource_sections.append(block)

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
    error_resources: set[str] | None = None,
) -> str:
    """Execute full hybrid retrieval: ChromaDB semantic search → Neo4j schema lookup.

    Args:
        retrieval_queries: HyDE queries generated upstream by the retriever agent.
        seed_resources:    AWS resource type names pre-extracted from the template
                           annotation by the caller (via extract_resource_types()).
                           Used as an allowlist for ChromaDB results — only chunks
                           belonging to these resource types are kept, preventing
                           unrelated resources from polluting the context.
        error_resources:   AWS resource type names extracted from active cfn-lint
                           or deployment validation errors (e.g. {"AWS::EC2::SecurityGroup",
                           "AWS::EC2::Instance"}). When provided, Neo4j schema lookups
                           are scoped to these resources only (plus any chroma-covered
                           ones), avoiding full-template schema dumps when only a
                           subset of resources have active errors.
                           Pass None (default) for initial generation where no prior
                           error signal exists.

    Returns:
        A multi-section context string for the remediator, or a short fallback
        message when no resources could be identified.
    """
    # Stage 1: semantic search scoped to template resources only
    resource_chunks, chroma_resources = _semantic_search(
        retrieval_queries,
        resource_filter=seed_resources,
    )

    identified_resources = seed_resources | chroma_resources

    if not identified_resources:
        return "No specific AWS resources identified in template or retrieval context."

    # Stage 2: Neo4j lookup scoped to error resources (when provided)
    schema_blocks = _graph_schema_lookup(
        identified_resources,
        chroma_covered=chroma_resources,
        error_resources=error_resources,
    )
    if not schema_blocks and not resource_chunks:
        return "Failed to connect to Knowledge Graph."

    return _assemble_retrieval_context(resource_chunks, schema_blocks)
