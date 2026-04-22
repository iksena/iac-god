import os
import yaml
import chromadb
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

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
def get_cfn_graph_context_for_state(
    validation_results: list[dict], 
    deploy_validation_result: dict | None, 
    template_yaml: str | None
) -> str:
    """
    Executes the Hybrid RAG workflow for the remediator agent.
    Combines template resources with semantically matched resources from validation errors.
    """
    print("[RAG Tool] Assembling G-Retrieval Context...")

    # Step 1: Base resources explicitly defined in the template
    identified_resources = _get_active_resources(template_yaml)
    
    # Step 2: Extract errors from state
    errors = _extract_errors(validation_results, deploy_validation_result)

    # STAGE 1: Semantic Search (ChromaDB)
    if errors:
        print(f"[RAG Tool] Stage 1: Embedding {len(errors)} error messages for semantic search...")
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            
            vectorstore = Chroma(
                client=chroma_client,
                collection_name="cloudformation_docs",
                embedding_function=GLOBAL_EMBEDDINGS
            )

            # Combine errors into a single query to retrieve relevant documentation chunks
            combined_query = " ".join(errors)
            top_chunks = vectorstore.similarity_search(combined_query, k=5)
            
            # Extract unique resource metadata from semantic matches
            for chunk in top_chunks:
                if "resource_name" in chunk.metadata:
                    identified_resources.add(chunk.metadata["resource_name"])
                    
        except Exception as e:
            print(f"[RAG Tool] Warning: ChromaDB Semantic Search failed. {e}")

    # STAGE 2: Exact Schema Traversal (Neo4j)
    if not identified_resources:
        return "No specific AWS resources identified in template or errors."

    print(f"[RAG Tool] Stage 2: Querying Neo4j for {len(identified_resources)} identified resources...")
    final_context_blocks = ["## Official AWS CloudFormation Schema Context\n"]
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        for resource in identified_resources:
            res_data = query_knowledge_graph(driver, resource)
            if "error" not in res_data:
                final_context_blocks.append(format_prompt_from_neo4j_result(res_data))
        driver.close()
    except Exception as e:
        print(f"[RAG Tool] Warning: Neo4j retrieval failed. {e}")
        return "Failed to connect to Knowledge Graph."

    return "\n\n".join(final_context_blocks)