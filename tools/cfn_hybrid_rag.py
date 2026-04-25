import os
import re
import json
from functools import lru_cache
from typing import Dict, Any
from neo4j import GraphDatabase
from tools.template_annotator import annotate_template, TemplateAnnotation

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_global_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ---------------------------------------------------------------------------
# Change 1: Security stage filter constant
# ---------------------------------------------------------------------------
# Validation stages that emit security policy violations (checkov, trivy).
# These do not benefit from CFN schema retrieval — the remediator already knows
# the fix (e.g. enable encryption) without needing property-level schema context.
# Import this constant in retriever.py if you want to log skipped stages.
SECURITY_STAGES = {"checkov", "trivy"}

# ---------------------------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------------------------
def _extract_errors(validation_results: list[dict], deploy_validation_result: dict | None) -> list[str]:
    """Extracts a flat list of error strings from the validation state.

    Security stages (checkov, trivy) are excluded — their findings are policy
    violations, not schema errors, and do not require CFN schema retrieval.
    """
    errors = []

    # Extract static validation errors, skipping security scanners
    for result in validation_results:
        stage = str(result.get("stage") or "").strip().lower()
        if stage in SECURITY_STAGES:  # Change 1: skip checkov/trivy
            continue
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip():
                    errors.append(str(err))

    # Extract live deployment errors — these are never security errors
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("error_message"):
            errors.append(deploy_validation_result["error_message"])

        for fr in deploy_validation_result.get("failed_resources", []):
            name = fr.get("logical_name") or fr.get("resource") or ""
            reason = fr.get("status_reason") or fr.get("reason") or ""
            if name or reason:
                errors.append(f"{name} {reason}")

    return errors


# ---------------------------------------------------------------------------
# Change 4: Chroma chunk cleaner
# ---------------------------------------------------------------------------
_DOC_LINK_RE = re.compile(
    r"https?://docs\.aws\.amazon\.com\S*", re.IGNORECASE
)
_DESCRIPTION_RE = re.compile(
    r"(?:^|\n)Description:\s*.+?(?=\n[A-Z]|\Z)", re.DOTALL
)


def _clean_chroma_chunk(content: str) -> str:
    """Remove AWS doc links and Description fields from a Chroma property chunk.

    These fields bloat the ### Semantically Matched Properties section without
    providing schema information the remediator can act on.
    """
    content = _DOC_LINK_RE.sub("", content)
    content = _DESCRIPTION_RE.sub("", content)
    # Collapse excess blank lines left behind by removals
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


# ---------------------------------------------------------------------------
# Change 3: Compact Neo4j formatter
# ---------------------------------------------------------------------------
def format_prompt_from_neo4j_result(resource_data: dict) -> str:
    """Formats Neo4j schema data into a concise single-block prompt section.

    Changes from original:
    - Resource description removed (not actionable for the remediator)
    - Required and optional properties rendered as single comma-separated lines
    - Optional properties capped at 20 to keep token budget lean
    - Error sentinel uses '#' prefix to match YAML comment style
    """
    if "error" in resource_data:
        return f"# {resource_data['error']}"

    lines = [f"### {resource_data['name']}"]
    # Description intentionally omitted — not needed by the remediator

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

def query_knowledge_graph(driver, resource_name: str) -> Dict[str, Any]:
    """Executes the Cypher traversal to pull the schema structure for a resource."""

    CYPHER_QUERY = """
        MATCH (r:Resource {name: $resource_name})
        
        // Get required properties
        OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(prop:Property)
        WHERE prop.required = true
        WITH r, collect(DISTINCT {name: prop.name, type: prop.type}) AS required_properties

        // Get optional properties
        OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(opt_prop:Property)
        WHERE opt_prop.required = false
        WITH r, required_properties, collect(DISTINCT {name: opt_prop.name, type: opt_prop.type}) AS optional_properties

        // Get nested types (complex objects)
        OPTIONAL MATCH (r)-[:HAS_NESTED_TYPE]->(nt:NestedType)
        WITH r, required_properties, optional_properties, collect(DISTINCT {name: nt.name, type: nt.type, required: nt.required}) AS nested_types

        // Get the first example
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
            "example": result["example"]
        }

# ---------------------------------------------------------------------------
# Main Orchestration Method for the Remediator Agent
# ---------------------------------------------------------------------------

# Change 2 + suggestion: updated system prompt instructs LLM to use inline
# error comments as the primary signal for resource+property targeting.
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

def _build_annotation_summary(annotation: TemplateAnnotation) -> str:
    """
    Serialize a TemplateAnnotation into a richer text block for query-gen.
    Includes actual property keys from the template so the LLM can target
    specific Resource.Property combinations rather than just resource types.
    """
    lines = [f"Template: {annotation.file_path} ({annotation.template_type})"]
    for r in annotation.resources:
        smell_ids = ", ".join(s.get("rule_id", "?") for s in r.smells) or "none"
        # Extract property keys actually present in the template
        props = sorted(r.raw.get("Properties", {}).keys()) if r.raw else []
        props_str = ", ".join(props) if props else "none"
        lines.append(
            f"  - [{r.resource_id}] type={r.resource_type} "
            f"line={r.start_line} smells=[{smell_ids}]\n"
            f"    properties_present=[{props_str}]"
        )
    return "\n".join(lines)

def _parse_query_response(raw: str, max_queries: int = 8) -> list[str]:
    """
    Dedicated parser for the LLM's query-generation response.
    Accepts both {"queries": [...]} object form and bare [...] array form.
    Strips markdown fences before parsing.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[RAG Tool] Query parse error (JSONDecodeError): {e}. Raw: {cleaned[:200]}")
        return []

    # Accept both {"queries": [...]} and bare [...] 
    if isinstance(parsed, dict):
        queries = parsed.get("queries") or parsed.get("query") or []
    elif isinstance(parsed, list):
        queries = parsed
    else:
        print(f"[RAG Tool] Unexpected query response type: {type(parsed)}")
        return []

    if not isinstance(queries, list):
        print(f"[RAG Tool] 'queries' field is not a list: {queries}")
        return []

    result = [str(q).strip() for q in queries if str(q).strip()][:max_queries]
    print(f"[RAG Tool] Parsed {len(result)} retrieval queries from LLM response.")
    return result


def _execute_hybrid_retrieval(
    retrieval_queries: list[str],
    annotation: TemplateAnnotation | None,
    template_yaml: str | None,
) -> str:
    """
    Execute retrieval only: ChromaDB semantic search followed by Neo4j schema lookup.

    The annotation is used to seed exact resource identification. If none is
    provided, a best-effort template parse is attempted locally.
    """
    if annotation is None and template_yaml:
        try:
            annotation = annotate_template(
                file_path="<in-memory>",
                content=template_yaml,
            )
        except Exception as exc:
            print(f"[RAG Tool] Annotation failed during retrieval bootstrap: {exc}")
            annotation = None

    if annotation and annotation.resources:
        identified_resources: set[str] = {r.resource_type for r in annotation.resources if r.resource_type}
    else:
        identified_resources = set()

    property_chunks: list[str] = []
    if retrieval_queries:
        try:
            import chromadb
            from langchain_chroma import Chroma

            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            vectorstore = Chroma(
                client=chroma_client,
                collection_name="cfn_schema_properties",
                embedding_function=_get_global_embeddings(),
            )

            seen_resources: set[str] = set()
            for query in retrieval_queries:
                chunks = vectorstore.similarity_search(query, k=3)
                for chunk in chunks:
                    meta = chunk.metadata
                    res = meta.get("resource_name", "")
                    prop = meta.get("property_name", "") or meta.get("property_path", "")
                    prop_key = f"{res}.{prop}" if prop else f"{res}::content::{hash(chunk.page_content)}"
                    if prop_key in seen_resources:
                        continue
                    seen_resources.add(prop_key)
                    # Change 4: strip doc links and description noise
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
    final_context_blocks = ["## Official AWS CloudFormation Schema Context\n"]

    if property_chunks:
        final_context_blocks.append("### Semantically Matched Properties\n" + "\n---\n".join(property_chunks))

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        seen_neo4j: set[str] = set()
        for resource in sorted(identified_resources):
            if resource in seen_neo4j:
                print(f"[RAG Tool] Skipping duplicate schema: {resource}")
                continue
            seen_neo4j.add(resource)
            res_data = query_knowledge_graph(driver, resource)
            if "error" not in res_data:
                final_context_blocks.append(format_prompt_from_neo4j_result(res_data))
        driver.close()
    except Exception as e:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {e}")
        return "Failed to connect to Knowledge Graph."

    return "\n\n".join(final_context_blocks)

def get_cfn_graph_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    template_yaml: str | None,
    smell_report: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """
    Backward-compatible wrapper that returns retrieval context without LLM query generation.

    The dedicated retriever agent now owns HyDE query generation. This wrapper
    is kept only for non-agent callers that want a pure retrieval context.
    """
    print("[RAG Tool] Assembling G-Retrieval Context (pure retrieval mode)...")
    annotation: TemplateAnnotation | None = None
    if template_yaml:
        try:
            annotation = annotate_template(
                file_path="<in-memory>",
                content=template_yaml,
            )
            print(f"[RAG Tool] Annotation: {len(annotation.resources)} resources parsed.")
        except Exception as exc:
            print(f"[RAG Tool] Annotation failed (non-fatal): {exc}")
            annotation = None

    context = _execute_hybrid_retrieval([], annotation, template_yaml)
    return context, []
