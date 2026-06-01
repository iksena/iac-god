"""tools/tf_hybrid_rag.py

ChromaDB + Neo4j hybrid retrieval for Terraform provider schema context.

Mirrors cfn_hybrid_rag.py exactly in structure and public API.
All internal names map the TF graph schema built by the ingestion pipeline
under scripts/graphrag/terraform/:

  Neo4j graph shape (from 06_test_rag_queries.py):
    (:TFResource)  -[:HAS_ATTRIBUTE]->  (:TFAttribute  {name, type, required, description})
    (:TFResource)  -[:HAS_BLOCK]->      (:TFBlock      {name, nesting_mode, min_items, max_items})
    (:TFBlock)     -[:HAS_ATTRIBUTE]->  (:TFAttribute)
    (:TFResource)  -[:HAS_EXAMPLE]->    (:TFExample    {code})

  ChromaDB collection: tf_schema_properties
    Chunk metadata keys: resource_name, attribute_name (or attribute_path)

Retrieval runs in two sequential stages:
  Stage 1 — _semantic_search():      ChromaDB similarity search over pre-indexed
                                      TF attribute chunks from tf_schema_properties.
                                      Filtered to seed_resources to prevent
                                      cross-resource pollution.
  Stage 2 — _graph_schema_lookup():  Neo4j Cypher traversal per :TFResource.
                                      Scoped to error_resources when provided.
  Final   — _assemble_retrieval_context(): merges both into the single context
                                      string consumed by the remediator prompt.

Context verbosity (CHROMA_CONTEXT_MODE env var):
  compact (default) — grouped per resource; attribute as one summary line.
  raw               — full chunk text; no deduplication.

Dependency direction (strictly unidirectional, mirrors cfn_hybrid_rag.py):
  retriever_agent  ->  tf_hybrid_rag  ->  template_annotator (type hints only)
  tf_hybrid_rag does NOT import from agents/
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

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Same threshold constant as cfn_hybrid_rag — enables direct benchmark comparison.
# Cosine distance in [0, 2]; lower = more similar.
# LOWER score = MORE similar. Chunks ABOVE the threshold are discarded.
CHROMA_DISTANCE_THRESHOLD: float = DEFAULT_DISTANCE_THRESHOLD

# Context verbosity mode (mirrors cfn_hybrid_rag._CONTEXT_MODE).
# "compact" (default): group chunks per resource, one line per attribute.
# "raw":               full chunk text, no deduplication.
_CONTEXT_MODE: str = os.getenv("CHROMA_CONTEXT_MODE", "compact").lower().strip()

# Maximum optional attributes shown per resource in the Neo4j block.
# Mirrors cfn_hybrid_rag._MAX_OPTIONAL_PROPS — same cap keeps token counts
# directly comparable across CFN and TF benchmark runs.
_MAX_OPTIONAL_ATTRS = 10

# ChromaDB collection populated by scripts/graphrag/terraform/ ingestion pipeline.
_TF_COLLECTION_NAME = "tf_schema_properties"

# Doc-link cleaner for Terraform Registry URLs.
_DOC_LINK_RE = re.compile(r"https?://registry\.terraform\.io\S*", re.IGNORECASE)

# Strip "See also / see <Xxx>." cross-reference boilerplate from resource
# descriptions — mirrors the equivalent re.sub in cfn_hybrid_rag._format_resource_block_compact().
# TF registry descriptions commonly end with "See the Foo resource for details."
_SEE_ALSO_RE = re.compile(r"\s+see[^.]+\.", re.IGNORECASE)

# Raw-mode description noise pattern.
# TF registry chunks label argument/attribute descriptions as
# "Argument Description:" or "Attribute Description:" (not bare "Description:").
# The pattern below covers all three forms so _clean_chunk_raw() strips them
# in raw mode, matching cfn_hybrid_rag's behaviour for the CFN "Description:" field.
_DESC_FIELD_RE = re.compile(
    r"(?:^|\n)(?:Argument |Attribute )?Description:\s*.+?(?=\n[A-Z]|\Z)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Chunk parsing — structured attribute data from raw TF doc chunks
# ---------------------------------------------------------------------------

@dataclass
class _AttributeChunk:
    """Parsed representation of one TF attribute chunk from ChromaDB.

    Maps to the chunk format produced by the TF ingestion pipeline:
        Resource: aws_vpc
        Resource Description: Provides a VPC resource.
        Attribute: cidr_block
        Type: String
        Required: True
        Force New: True        <- equivalent of CFN UpdateType=Immutable
    """
    resource_name: str = ""
    resource_description: str = ""
    attribute_name: str = ""
    attribute_type: str = ""
    required: str = ""
    force_new: str = ""
    is_example: bool = False
    raw_text: str = ""  # fallback for unparseable chunks


_FIELD_RE = re.compile(
    r"^(?P<key>Resource|Resource Description|Attribute|Type|Required|Force New)"
    r":\s*(?P<value>.+)$",
    re.MULTILINE,
)


def _parse_chunk(content: str, meta: dict) -> _AttributeChunk:
    """Parse a raw ChromaDB chunk into a structured _AttributeChunk.

    Mirrors cfn_hybrid_rag._parse_chunk() but uses 'Attribute' key
    instead of 'Property' and 'Force New' instead of 'Update Type'.
    Falls back to raw_text for example blocks or unstructured chunks.
    """
    if content.lstrip().startswith("Terraform example for"):
        resource = meta.get("resource_name", "") or ""
        return _AttributeChunk(resource_name=resource, is_example=True, raw_text=content)

    fields: dict[str, str] = {}
    for m in _FIELD_RE.finditer(content):
        fields[m.group("key")] = m.group("value").strip()

    if not fields.get("Attribute"):
        return _AttributeChunk(
            resource_name=meta.get("resource_name", ""),
            raw_text=_DOC_LINK_RE.sub("", content).strip(),
        )

    return _AttributeChunk(
        resource_name=fields.get("Resource", meta.get("resource_name", "")),
        resource_description=fields.get("Resource Description", ""),
        attribute_name=fields.get("Attribute", ""),
        attribute_type=fields.get("Type", ""),
        required=fields.get("Required", ""),
        force_new=fields.get("Force New", ""),
    )


# ---------------------------------------------------------------------------
# Grouped resource data — accumulates all attribute chunks for one resource
# ---------------------------------------------------------------------------

@dataclass
class _ResourceChunks:
    name: str
    description: str = ""
    attributes: list[_AttributeChunk] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    raw_chunks: list[str] = field(default_factory=list)  # unparseable fallbacks


def _format_resource_block_compact(rc: _ResourceChunks) -> str:
    """Render one TF resource's ChromaDB hits as a compact grouped block.

    Mirrors cfn_hybrid_rag._format_resource_block_compact() exactly,
    replacing 'property' vocabulary with 'attribute'.

    The resource description is printed once per resource (deduplicated
    across all attribute chunks). "See also" cross-reference boilerplate
    is stripped from descriptions — mirrors cfn_hybrid_rag behaviour.

    Format:
        #### aws_vpc
        > Provides a VPC resource.

        - cidr_block (String, Force New)  [required]
        - enable_dns_support (Bool)
    """
    lines: list[str] = []
    header = f"#### {rc.name}" if rc.name and rc.name != "_unknown" \
        else "#### (resource type unknown)"
    lines.append(header)

    # Strip registry links, "See also" boilerplate, and excess whitespace —
    # mirrors the three-step cleanup in cfn_hybrid_rag._format_resource_block_compact().
    desc = _DOC_LINK_RE.sub("", rc.description).strip()
    desc = _SEE_ALSO_RE.sub(".", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip()
    if desc:
        lines.append(f"> {desc}")

    for a in rc.attributes:
        parts = [a.attribute_name]
        type_info = ""
        if a.attribute_type:
            type_info = a.attribute_type
            if a.force_new and a.force_new.lower() not in ("false", ""):
                type_info += ", Force New"
        if type_info:
            parts.append(f"({type_info})")
        req_flag = "  [required]" if str(a.required).lower() == "true" else ""
        lines.append(f"- {'  '.join(parts)}{req_flag}")

    for raw in rc.raw_chunks:
        lines.append(raw)

    return "\n".join(lines)


def _format_resource_block_raw(chunks: list[str]) -> str:
    """Legacy: join raw cleaned chunks with separators (mirrors cfn_hybrid_rag)."""
    return "\n---\n".join(chunks)


# ---------------------------------------------------------------------------
# Neo4j helpers — TFResource / TFAttribute / TFBlock / TFExample schema
# ---------------------------------------------------------------------------

def query_knowledge_graph(driver, resource_name: str) -> dict:
    """Execute the Cypher traversal to pull the schema structure for a TF resource.

    Graph schema (from scripts/graphrag/terraform/06_test_rag_queries.py):
        (:TFResource)  -[:HAS_ATTRIBUTE]-> (:TFAttribute {name, type, required, description})
        (:TFResource)  -[:HAS_BLOCK]->     (:TFBlock     {name, nesting_mode, min_items, max_items})
        (:TFBlock)     -[:HAS_ATTRIBUTE]-> (:TFAttribute)
        (:TFResource)  -[:HAS_EXAMPLE]->   (:TFExample   {code})

    Returns a dict mirroring cfn_hybrid_rag.query_knowledge_graph() in key names
    (required_properties / optional_properties / nested_types / example) so that
    format_prompt_from_neo4j_result() is a direct parallel.
    """
    CYPHER_QUERY = """
        MATCH (r:TFResource {name: $resource_name})

        OPTIONAL MATCH (r)-[:HAS_ATTRIBUTE]->(req_attr:TFAttribute)
        WHERE req_attr.required = true
        WITH r, collect(DISTINCT {
            name:        req_attr.name,
            type:        req_attr.type,
            description: req_attr.description
        }) AS required_properties

        OPTIONAL MATCH (r)-[:HAS_ATTRIBUTE]->(opt_attr:TFAttribute)
        WHERE opt_attr.required = false
        WITH r, required_properties,
             collect(DISTINCT {
                 name:        opt_attr.name,
                 type:        opt_attr.type,
                 description: opt_attr.description
             }) AS optional_properties

        OPTIONAL MATCH (r)-[:HAS_BLOCK]->(b:TFBlock)
        WITH r, required_properties, optional_properties,
             collect(DISTINCT {
                 name:         b.name,
                 nesting_mode: b.nesting_mode,
                 min_items:    b.min_items,
                 required:     (b.min_items IS NOT NULL AND b.min_items > 0)
             }) AS nested_types

        OPTIONAL MATCH (r)-[:HAS_EXAMPLE]->(e:TFExample)
        WITH r, required_properties, optional_properties, nested_types,
             collect(e.code)[0] AS example_code

        RETURN r.name         AS resource_name,
               r.description  AS resource_description,
               required_properties,
               optional_properties,
               nested_types,
               { code: example_code } AS example
    """
    with driver.session() as session:
        result = session.run(CYPHER_QUERY, resource_name=resource_name).single()
        if not result:
            return {"error": f"TFResource '{resource_name}' not found in Knowledge Graph."}
        return {
            "name":                result["resource_name"],
            "description":        result["resource_description"],
            "required_properties": result["required_properties"],
            "optional_properties": result["optional_properties"],
            "nested_types":        result["nested_types"],
            "example":             result["example"],
        }


def format_prompt_from_neo4j_result(
    resource_data: dict,
    include_example: bool = True,
) -> str:
    """Format Neo4j TFResource data into a concise single-block prompt section.

    Mirrors cfn_hybrid_rag.format_prompt_from_neo4j_result() exactly.
    Uses identical key names (required_properties, optional_properties,
    nested_types) so the rendering logic is structurally identical to CFN.
    Returns empty string for error/stub nodes (same contract as CFN version).

    Args:
        resource_data:   Dict returned by query_knowledge_graph().
        include_example: When False the HCL example block is omitted.
                         Callers set this to False when ChromaDB already
                         provided semantic context for this resource,
                         avoiding redundant tokens (same logic as CFN).
    """
    if "error" in resource_data:
        print(f"[TF RAG] ⚠ Neo4j: {resource_data['error']}")
        return ""

    req    = resource_data.get("required_properties") or []
    opt    = resource_data.get("optional_properties") or []
    nested = resource_data.get("nested_types") or []

    if not req and not opt and not nested:
        print(
            f"[TF RAG] ⚠ {resource_data['name']}: node exists in Neo4j but has "
            f"no attributes/blocks. Index may be stale — rerun TF ingestion pipeline."
        )
        return ""

    lines = [f"### {resource_data['name']}"]

    if resource_data.get("description"):
        desc = _DOC_LINK_RE.sub("", resource_data["description"]).strip()
        if desc:
            lines.append(f"> {desc}")

    def _fmt_attr(a: dict) -> str:
        base = f"{a['name']}({a.get('type', '?')})"
        desc = (a.get("description") or "").strip()
        return f"{base} — {desc}" if desc else base

    if req:
        lines.append("Required: " + ", ".join(_fmt_attr(a) for a in req))
    if opt:
        shown = opt[:_MAX_OPTIONAL_ATTRS]
        lines.append("Optional: " + ", ".join(_fmt_attr(a) for a in shown))
        if len(opt) > _MAX_OPTIONAL_ATTRS:
            lines.append(f"  ... and {len(opt) - _MAX_OPTIONAL_ATTRS} more optional attributes")

    if nested:
        # Use .get() for both keys — the Cypher computed boolean
        # (b.min_items IS NOT NULL AND b.min_items > 0) can return None
        # when min_items is absent, so nt['required'] would KeyError.
        nt_list = ", ".join(
            f"{nt['name']}({'req' if nt.get('required') else 'opt'})"
            for nt in nested
            if nt.get("name")
        )
        if nt_list:
            # Label matches CFN ("NestedTypes:") so benchmark log parsing
            # tools treat both outputs identically.
            lines.append(f"NestedTypes: {nt_list}")

    if include_example and resource_data.get("example") and resource_data["example"].get("code"):
        lines.append(f"Example:\n```hcl\n{resource_data['example']['code']}\n```")

    return "\n".join(lines)


@contextmanager
def _neo4j_driver():
    """Context manager that opens a Neo4j driver and ensures it is closed."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Stage 1 — Semantic search (ChromaDB tf_schema_properties)
# ---------------------------------------------------------------------------

def _semantic_search(
    retrieval_queries: list[str],
    resource_filter: set[str] | None = None,
) -> tuple[dict[str, _ResourceChunks], set[str]]:
    """Run ChromaDB similarity search over tf_schema_properties.

    Mirrors cfn_hybrid_rag._semantic_search() exactly.
    Only the collection name and chunk parser differ.

    Args:
        retrieval_queries: HyDE queries from the retriever agent.
        resource_filter:   When provided, only chunks whose resource_name
                           is in this set are kept. Pass seed_resources here.

    Returns:
        resource_chunks:  Dict mapping resource_name -> _ResourceChunks.
        found_resources:  Set of TF resource type names from passing chunks.
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
            collection_name=_TF_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )

        seen_attr_keys: set[str] = set()
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

                if resource_filter and res not in resource_filter:
                    filtered_out += 1
                    continue

                # Deduplicate by (resource, attribute) pair — mirrors CFN prop dedup.
                attr = meta.get("attribute_name", "") or meta.get("attribute_path", "")
                attr_key = (
                    f"{res}.{attr}" if attr
                    else f"{res}::content::{hash(chunk.page_content)}"
                )
                if attr_key in seen_attr_keys:
                    continue
                seen_attr_keys.add(attr_key)

                rc = resource_chunks[res]
                rc.name = rc.name or res

                if _CONTEXT_MODE == "compact":
                    parsed = _parse_chunk(chunk.page_content, meta)
                    if parsed.is_example:
                        rc.examples.append(parsed.raw_text)
                    elif parsed.attribute_name:
                        if not rc.description and parsed.resource_description:
                            rc.description = parsed.resource_description
                        rc.attributes.append(parsed)
                    else:
                        rc.raw_chunks.append(parsed.raw_text)
                else:
                    cleaned = _clean_chunk_raw(chunk.page_content)
                    if cleaned:
                        rc.raw_chunks.append(cleaned)

                kept += 1
                if res and res != "_unknown":
                    found_resources.add(res)

        print(
            f"[TF RAG] Stage 1: {kept + dropped + filtered_out} chunks retrieved, "
            f"{dropped} dropped (distance > {CHROMA_DISTANCE_THRESHOLD}), "
            f"{filtered_out} filtered (not in template), "
            f"{kept} kept  [mode={_CONTEXT_MODE}]."
        )

    except Exception as exc:
        print(f"[TF RAG] Warning: ChromaDB Semantic Search failed. {exc}")

    return dict(resource_chunks), found_resources


def _clean_chunk_raw(content: str) -> str:
    """Legacy raw-mode cleaner: strip registry links and description fields.

    Mirrors cfn_hybrid_rag._clean_chunk_raw() but broadens the description
    pattern to cover all three TF registry variants:
      - "Description:"            (bare, sometimes present)
      - "Argument Description:"   (most common in resource argument docs)
      - "Attribute Description:"  (exported/computed attribute docs)
    The cfn_hybrid_rag version only strips bare "Description:" which would
    miss the TF-specific prefixed forms, leaving full description paragraphs
    in raw output and doubling token cost for no retrieval benefit.
    """
    content = _DOC_LINK_RE.sub("", content)
    content = _DESC_FIELD_RE.sub("", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


# ---------------------------------------------------------------------------
# Stage 2 — Knowledge-graph schema lookup (Neo4j :TFResource)
# ---------------------------------------------------------------------------

def _graph_schema_lookup(
    resources: set[str],
    chroma_covered: set[str],
    error_resources: set[str] | None = None,
) -> list[str]:
    """Fetch full schema blocks for each TF resource from Neo4j.

    Mirrors cfn_hybrid_rag._graph_schema_lookup() exactly in scoping logic:
    - When error_resources is provided: only erroring resources + chroma-covered
      resources are fetched — avoids full-template schema dumps.
    - When error_resources is None: fetch all identified resources.

    Args:
        resources:       All identified TF resource type names (seed + chroma).
        chroma_covered:  Resources that already have ChromaDB semantic context.
                         For these, the HCL example is omitted in compact mode.
        error_resources: Resource types from active tflint / terraform-validate /
                         deploy errors. Scopes Neo4j lookups when provided.
    """
    schema_blocks: list[str] = []

    if not resources:
        return schema_blocks

    if error_resources:
        target_resources = error_resources | (resources & chroma_covered)
        print(
            f"[TF RAG] Stage 2: Scoped to {len(target_resources)} error/chroma resources "
            f"(skipping {len(resources) - len(target_resources)} non-erroring resources)."
        )
    else:
        target_resources = resources
        print(f"[TF RAG] Stage 2: Querying Neo4j for {len(target_resources)} TF resources...")

    try:
        with _neo4j_driver() as driver:
            seen: set[str] = set()
            for resource in sorted(target_resources):
                if resource in seen:
                    continue
                seen.add(resource)
                res_data = query_knowledge_graph(driver, resource)
                include_ex = not (
                    _CONTEXT_MODE == "compact" and resource in chroma_covered
                )
                block = format_prompt_from_neo4j_result(res_data, include_example=include_ex)
                if block:
                    schema_blocks.append(block)
    except Exception as exc:
        print(f"[TF RAG] Warning: Neo4j retrieval failed. {exc}")

    return schema_blocks


# ---------------------------------------------------------------------------
# Final — Context assembly
# ---------------------------------------------------------------------------

def _assemble_retrieval_context(
    resource_chunks: dict[str, _ResourceChunks],
    schema_blocks: list[str],
) -> str:
    """Merge grouped ChromaDB attribute blocks and Neo4j schema blocks.

    Header uses 'Terraform Provider Schema Context' to distinguish from
    CFN context when both are visible in debug logs.
    Mirrors cfn_hybrid_rag._assemble_retrieval_context() structure exactly.
    """
    final_blocks: list[str] = ["## Official Terraform Provider Schema Context\n"]

    if resource_chunks:
        resource_sections: list[str] = []
        for resource_name, rc in sorted(resource_chunks.items()):
            if _CONTEXT_MODE == "compact":
                block = _format_resource_block_compact(rc)
            else:
                header = (
                    f"#### {resource_name}" if resource_name != "_unknown"
                    else "#### (resource type unknown)"
                )
                block = header + "\n" + _format_resource_block_raw(rc.raw_chunks)
            resource_sections.append(block)

        final_blocks.append(
            "### Semantically Matched Attributes\n"
            + "\n\n".join(resource_sections)
        )

    final_blocks.extend(schema_blocks)
    return "\n\n".join(final_blocks)


# ---------------------------------------------------------------------------
# Public retrieval entry point — called by retriever.py
# ---------------------------------------------------------------------------

def execute_terraform_retrieval(
    retrieval_queries: list[str],
    seed_resources: set[str],
    error_resources: set[str] | None = None,
) -> str:
    """Execute full hybrid retrieval for Terraform: ChromaDB -> Neo4j.

    Drop-in parallel to cfn_hybrid_rag.execute_hybrid_retrieval().
    Called by retriever_agent() when iac_type == "terraform".

    Args:
        retrieval_queries: HyDE queries generated by the retriever agent.
        seed_resources:    TF resource type names pre-extracted from the
                           template annotation (e.g. {"aws_vpc", "aws_s3_bucket"}).
                           Used as an allowlist for ChromaDB results.
        error_resources:   TF resource type names extracted from active tflint /
                           terraform-validate / deploy errors. Scopes Neo4j
                           lookups to avoid full-template schema dumps.
                           Pass None for initial generation with no error signal.

    Returns:
        A multi-section context string for the remediator, or a short fallback
        message when no resources could be identified.
    """
    resource_chunks, chroma_resources = _semantic_search(
        retrieval_queries,
        resource_filter=seed_resources,
    )

    identified_resources = seed_resources | chroma_resources

    if not identified_resources:
        return "No specific Terraform resources identified in template or retrieval context."

    schema_blocks = _graph_schema_lookup(
        identified_resources,
        chroma_covered=chroma_resources,
        error_resources=error_resources,
    )

    if not schema_blocks and not resource_chunks:
        return "Failed to connect to Terraform Knowledge Graph."

    return _assemble_retrieval_context(resource_chunks, schema_blocks)
