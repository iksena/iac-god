"""
Neo4j client for the Terraform Agent
"""

import os
from typing import Dict, Any
from neo4j import GraphDatabase

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize the Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def format_prompt_from_neo4j_result(resource_data):
    if "error" in resource_data:
        return f"Error: {resource_data['error']}"

    lines = []
    lines.append(f"Resource: {resource_data['name']}")
    lines.append(f"Description: {resource_data['description']}\n")

    if resource_data["required_properties"]:
        lines.append("Required Properties:")
        for prop in resource_data["required_properties"]:
            lines.append(f"- {prop['name']} ({prop['type']})")
        lines.append("")

    if resource_data["optional_properties"]:
        lines.append("Optional Properties:")
        for prop in resource_data["optional_properties"]:
            lines.append(f"- {prop['name']} ({prop['type']})")
        lines.append("")

    if resource_data["nested_types"]:
        lines.append("Nested Complex Types:")
        for nt in resource_data["nested_types"]:
            lines.append(f"- {nt['name']} ({nt['type']}) - Required: {nt['required']}")
        lines.append("")

    if resource_data["example"]:
        lines.append("YAML Example:")
        lines.append(f"```yaml\n{resource_data['example']['code']}\n```")

    return "\n".join(lines)

def query_knowledge_graph(resource_name: str) -> Dict[str, Any]:
    # Removed 'example_title' from the signature
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
    
    with neo4j_driver.session() as session:
        # Only passing resource_name
        result = session.run(CYPHER_QUERY, resource_name=resource_name).single()

        if not result:
            return {"error": f"Resource '{resource_name}' not found"}

        return {
            "name": result["resource_name"],
            "description": result["resource_description"],
            "required_properties": result["required_properties"],
            "optional_properties": result["optional_properties"],
            "nested_types": result["nested_types"],
            "example": result["example"]
        }

def get_resource_information(resource_name: str) -> str:
    """
    Retrieve resource information from the Neo4j knowledge graph.
    """
    # Removed 'example_title' from the call
    resource_data = query_knowledge_graph(resource_name)
    formatted_prompt = format_prompt_from_neo4j_result(resource_data)
    return formatted_prompt


def get_remediation_context(resource_name: str, property_name: str):
    CYPHER_QUERY = """
        // Find the specific resource
        MATCH (r:Resource {name: $resource_name})
        
        // Find the specific property or nested type causing the error
        OPTIONAL MATCH (r)-[:HAS_PROPERTY|HAS_NESTED_TYPE]->(target)
        WHERE target.name = $property_name
        
        // If it's a nested type, get its sub-properties so the LLM knows how to build it
        OPTIONAL MATCH (target)-[:HAS_PROPERTY]->(sub_prop)
        
        // Get a code example to show correct syntax
        OPTIONAL MATCH (r)-[:HAS_EXAMPLE]->(e:Example)
        WHERE e.code CONTAINS $property_name
        
        RETURN r.name AS Resource,
               target.name AS Property,
               target.type AS Type,
               target.required AS Required,
               collect(DISTINCT {name: sub_prop.name, type: sub_prop.type, required: sub_prop.required}) AS SubProperties,
               collect(DISTINCT e.code)[0] AS RelevantExample
    """
    
    with neo4j_driver.session() as session:
        result = session.run(CYPHER_QUERY, resource_name=resource_name, property_name=property_name).single()

        if not result:
            return {"error": f"Resource '{resource_name}' or Property '{property_name}' not found"}

        # Fixed the return dict to map to the Cypher RETURN statement
        return {
            "resource": result["Resource"],
            "property": result["Property"],
            "type": result["Type"],
            "required": result["Required"],
            "sub_properties": result["SubProperties"],
            "example": result["RelevantExample"]
        }