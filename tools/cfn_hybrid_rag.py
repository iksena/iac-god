import os
import yaml
import json
import chromadb
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from agents.engineer import _build_client
from tools.template_annotator import annotate_template, attach_smells, TemplateAnnotation

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

# ---------------------------------------------------------------------------
# Safe YAML Loading for CFN tags (!Ref, !Sub, etc.)
# ---------------------------------------------------------------------------
def _cfn_tag_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)

yaml.SafeLoader.add_multi_constructor("!", _cfn_tag_constructor)

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------------------------
def _extract_errors(validation_results: list[dict], deploy_validation_result: dict | None) -> list[str]:
    """Extracts a flat list of error strings from the validation state."""
    errors = []
    
    # Extract static validation errors
    for result in validation_results:
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip(): 
                    errors.append(str(err))
                    
    # Extract live deployment errors
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("error_message"): 
            errors.append(deploy_validation_result["error_message"])
            
        for fr in deploy_validation_result.get("failed_resources", []):
            name = fr.get("logical_name") or fr.get("resource") or ""
            reason = fr.get("status_reason") or fr.get("reason") or ""
            if name or reason: 
                errors.append(f"{name} {reason}")

    return errors

def _get_active_resources(template_yaml: str) -> set[str]:
    """Parses the YAML template to find exactly which AWS resources are being used."""
    resources = set()
    if not template_yaml:
        return resources
    try:
        parsed = yaml.safe_load(template_yaml)
        if isinstance(parsed, dict) and "Resources" in parsed:
            for _, res_val in parsed["Resources"].items():
                if isinstance(res_val, dict) and "Type" in res_val:
                    resources.add(res_val["Type"])
    except Exception as e:
        print(f"YAML parsing error during resource extraction: {e}")
    return resources

def format_prompt_from_neo4j_result(resource_data: dict) -> str:
    """Formats the Neo4j schema response into a concise prompt for the LLM."""
    if "error" in resource_data:
        return f"Error: {resource_data['error']}"

    lines = [f"Resource: {resource_data['name']}"]
    if resource_data.get('description'):
        lines.append(f"Description: {resource_data['description']}\n")

    if resource_data.get("required_properties"):
        lines.append("Required Properties:")
        for prop in resource_data["required_properties"]:
            lines.append(f"- {prop['name']} ({prop['type']})")
        lines.append("")

    if resource_data.get("optional_properties"):
        lines.append("Optional Properties:")
        for prop in resource_data["optional_properties"]:
            lines.append(f"- {prop['name']} ({prop['type']})")
        lines.append("")

    if resource_data.get("nested_types"):
        lines.append("Nested Complex Types:")
        for nt in resource_data["nested_types"]:
            lines.append(f"- {nt['name']} ({nt['type']}) - Required: {nt['required']}")
        lines.append("")

    if resource_data.get("example"):
        lines.append("YAML Example:")
        lines.append(f"```yaml\n{resource_data['example']['code']}\n```")

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

QUERY_GEN_SYSTEM = """\
You are an AWS CloudFormation schema expert. You are given:
1. A list of validation errors from a CFN template.
2. A structured annotation of the template's resources: their logical IDs, AWS resource types, \
   source line numbers, and any detected security smells.

Using this context, generate a list of precise retrieval queries. Each query must target a specific \
Resource.Property combination to look up its schema, required constraints, or correct usage.

Output a JSON array of strings only. Each string should be a natural-language question like:
"What are the required properties for AWS::S3::Bucket BucketEncryption ServerSideEncryptionConfiguration?"
"What type is AWS::RDS::DBInstance DBSubnetGroupName and is it required?"
"What values are valid for AWS::IAM::Role AssumeRolePolicyDocument Version?"

Prioritise resources that appear in the errors. Limit to at most 8 queries.\
"""

def _build_annotation_summary(annotation: TemplateAnnotation) -> str:
    """
    Serialize a TemplateAnnotation into a compact text block for the query-gen prompt.
    Keeps token cost low while giving the LLM full resource-type + smell signal.
    """
    lines = [f"Template: {annotation.file_path} ({annotation.template_type})"]
    for r in annotation.resources:
        smell_ids = ", ".join(s.get("rule_id", "?") for s in r.smells) or "none"
        lines.append(
            f"  - [{r.resource_id}] type={r.resource_type} "
            f"line={r.start_line} smells=[{smell_ids}]"
        )
    return "\n".join(lines)


def generate_retrieval_queries(
    errors: list[str],
    template_yaml: str | None,
    annotation: TemplateAnnotation | None = None,   # NEW optional param
) -> list[str]:
    """HyDE-style: LLM reformulates errors + annotation into schema-targeted retrieval queries."""
    if not errors:
        return []

    client, model = _build_client()

    # Build user content: errors first, then annotation (compact), then raw YAML fallback
    user_parts = ["## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)]

    if annotation and not annotation.parse_error:
        user_parts.append(
            "## Template Resource Annotation\n"
            "(Logical IDs, resource types, source lines, detected smells)\n"
            + _build_annotation_summary(annotation)
        )
    elif template_yaml:
        # Fallback: raw YAML snippet as before
        user_parts.append(
            f"## Template Snippet (for resource type context)\n"
            f"```yaml\n{template_yaml[:2000]}\n```"
        )

    user_content = "\n\n".join(user_parts)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_GEN_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [str(q) for q in queries[:8]]
    except Exception as e:
        print(f"[RAG Tool] Query generation failed, falling back to raw errors: {e}")

    return errors

def get_cfn_graph_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    template_yaml: str | None,
    smell_report: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """
    Executes the Hybrid RAG workflow with annotation-enriched HyDE query reformulation.
    Returns (context_text, retrieval_queries).
    """
    print("[RAG Tool] Assembling G-Retrieval Context (HyDE + Annotation mode)...")

    errors = _extract_errors(validation_results, deploy_validation_result)

    annotation: TemplateAnnotation | None = None
    if template_yaml:
        try:
            annotation = annotate_template(
                file_path="<in-memory>",
                content=template_yaml,
            )
            if smell_report:
                annotation = attach_smells(annotation, smell_report)
            print(
                f"[RAG Tool] Annotation: {len(annotation.resources)} resources parsed, "
                f"{sum(len(r.smells) for r in annotation.resources)} smells attached."
            )
        except Exception as exc:
            print(f"[RAG Tool] Annotation failed (non-fatal): {exc}")
            annotation = None

    # Derive identified_resources from annotation (more precise than regex scan)
    if annotation and annotation.resources:
        identified_resources: set[str] = {r.resource_type for r in annotation.resources}
    else:
        identified_resources = _get_active_resources(template_yaml)  # fallback

    # ── HyDE STAGE 0: Query Reformulation (annotation-enriched) ───────────
    retrieval_queries = generate_retrieval_queries(errors, template_yaml, annotation)
    print(f"[RAG Tool] Stage 0: Generated {len(retrieval_queries)} retrieval queries.")

    # ── STAGE 1: Semantic Search (ChromaDB) — multi-query ──────────────────
    if retrieval_queries:
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            vectorstore = Chroma(
                client=chroma_client,
                collection_name="cfn_schema_properties",
                embedding_function=GLOBAL_EMBEDDINGS,
            )

            seen_resources: set[str] = set()
            property_chunks: list[str] = []

            for query in retrieval_queries:
                chunks = vectorstore.similarity_search(query, k=3)
                for chunk in chunks:
                    meta = chunk.metadata
                    res = meta.get("resource_name", "")
                    prop = meta.get("property_name", "")
                    if res:
                        identified_resources.add(res)
                    # Capture property-level text for direct context injection
                    prop_key = f"{res}.{prop}"
                    if prop_key not in seen_resources:
                        seen_resources.add(prop_key)
                        property_chunks.append(chunk.page_content)

        except Exception as e:
            print(f"[RAG Tool] Warning: ChromaDB Semantic Search failed. {e}")
            property_chunks = []
    else:
        property_chunks = []

    # ── STAGE 2: Exact Schema Traversal (Neo4j) ────────────────────────────
    if not identified_resources:
        return "No specific AWS resources identified in template or errors.", retrieval_queries

    print(f"[RAG Tool] Stage 2: Querying Neo4j for {len(identified_resources)} resources...")
    final_context_blocks = ["## Official AWS CloudFormation Schema Context\n"]

    # Inject property-level chunks from semantic stage (highest relevance first)
    if property_chunks:
        final_context_blocks.append("### Semantically Matched Properties\n" + "\n---\n".join(property_chunks))

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        for resource in identified_resources:
            res_data = query_knowledge_graph(driver, resource)
            if "error" not in res_data:
                final_context_blocks.append(format_prompt_from_neo4j_result(res_data))
        driver.close()
    except Exception as e:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {e}")
        return "Failed to connect to Knowledge Graph.", retrieval_queries

    return "\n\n".join(final_context_blocks), retrieval_queries