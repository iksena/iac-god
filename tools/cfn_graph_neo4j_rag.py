"""
Neo4j GraphRAG context tool — retrieves exact CFN schema context via deterministic graph traversal.
Dynamically anchors to resources in the current template and fetches deeply nested PropertyTypes.
"""
from __future__ import annotations

import os
import re
import textwrap

try:
    import yaml as _yaml
    
    # Register a generic multi-constructor to safely ignore CFN tags (!Ref, !Sub, etc.)
    # This prevents yaml.safe_load from crashing on valid CloudFormation templates.
    def _cfn_tag_constructor(loader, tag_suffix, node):
        if isinstance(node, _yaml.ScalarNode):
            return loader.construct_scalar(node)
        elif isinstance(node, _yaml.SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, _yaml.MappingNode):
            return loader.construct_mapping(node)
            
    _yaml.SafeLoader.add_multi_constructor("!", _cfn_tag_constructor)
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError as exc:
    GraphDatabase = None
    neo4j_import_error = exc


class CFNGraphRAGNeo4j:
    def __init__(self, uri, user, password):
        if GraphDatabase is None:
            raise ImportError(f"neo4j library is not installed: {neo4j_import_error}")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _get_active_resources(self, template_yaml: str) -> set:
        """Parse the YAML and collect AWS resource types used in the template."""
        if not template_yaml:
            return set()
        try:
            tpl = _yaml.safe_load(template_yaml) or {}
            return {v.get("Type") for v in tpl.get("Resources", {}).values() if isinstance(v, dict) and "Type" in v}
        except Exception:
            return set()

    def _extract_flagged_properties(self, errors: list) -> set:
        """Extract property names enclosed in single quotes from validation errors."""
        flagged = set()
        for error in errors:
            matches = re.findall(r"'([A-Za-z0-9]+)'", error)
            flagged.update(matches)
        return flagged

    def _fetch_property_type_properties(self, session, type_name: str) -> list:
        """Fetch properties defined on a nested PropertyType node."""
        result = session.run("""
            MATCH (pt:PropertyType {name: $type_name})-[:HAS_PROPERTY]->(p:Property)
            OPTIONAL MATCH (p)-[:USES_TYPE]->(npt:PropertyType)
            RETURN p.name AS prop_name, p.type AS type, p.required AS required, npt.name AS nested_type
        """, type_name=type_name)
        return result.data()

    def retrieve_schema_context(self, template_yaml: str, errors: list) -> str:
        """Queries the Neo4j Graph to construct a highly specific schema document for the LLM."""
        active_resources = self._get_active_resources(template_yaml)
        flagged_props = self._extract_flagged_properties(errors)

        if not active_resources:
            return ""

        context_blocks = []
        with self.driver.session() as session:
            for resource in sorted(active_resources):
                # 1. Fetch direct properties for the Resource
                direct_result = session.run("""
                    MATCH (r:Resource {name: $res_name})-[:HAS_PROPERTY]->(p:Property)
                    OPTIONAL MATCH (p)-[:USES_TYPE]->(pt:PropertyType)
                    RETURN p.name AS prop_name, p.type AS type, p.required AS required, pt.name AS nested_type
                """, res_name=resource)

                direct_props = direct_result.data()
                if not direct_props:
                    continue

                # 2. Fetch nested properties for each PropertyType linked to the Resource
                nested_type_names = sorted({row["nested_type"] for row in direct_props if row.get("nested_type")})
                nested_props_by_type = {
                    type_name: self._fetch_property_type_properties(session, type_name)
                    for type_name in nested_type_names
                }

                # 3. Correlate flagged properties with graph structure
                direct_required = [row["prop_name"] for row in direct_props if row.get("required")]
                direct_flagged = [row for row in direct_props if row["prop_name"] in flagged_props]

                nested_flagged = []
                for type_name, props in nested_props_by_type.items():
                    for prop in props:
                        if prop["prop_name"] in flagged_props:
                            nested_flagged.append((type_name, prop))

                # 4. Construct the Markdown Block
                block_lines = [f"### {resource}"]
                
                if direct_flagged or nested_flagged:
                    block_lines.append("**Properties flagged in errors:**")
                    for row in direct_flagged:
                        req = "required" if row["required"] else "optional"
                        block_lines.append(f"- `{row['prop_name']}` ({row['type']}, {req})")
                    for type_name, prop in nested_flagged:
                        req = "required" if prop["required"] else "optional"
                        block_lines.append(f"- `{prop['prop_name']}` ({prop['type']}, {req}) under `{type_name}`")

                req_str = ", ".join(direct_required) if direct_required else "(none)"
                block_lines.append(f"\n**Required properties:** {req_str}")
                
                if nested_type_names:
                    block_lines.append(f"**Nested property types:** {', '.join(nested_type_names)}")
                
                for type_name in nested_type_names:
                    nested_props = nested_props_by_type.get(type_name, [])
                    if not nested_props:
                        continue
                    block_lines.append(f"\n**{type_name} properties:**")
                    for prop in nested_props:
                        req = "required" if prop["required"] else "optional"
                        marker = " *flagged*" if prop["prop_name"] in flagged_props else ""
                        block_lines.append(f"- `{prop['prop_name']}` ({prop['type']}, {req}){marker}")

                context_blocks.append("\n".join(block_lines))

        return "\n\n".join(context_blocks) if context_blocks else ""


# =============================================================================
# Adapter Interface for Remediator Agent
# =============================================================================

def get_cfn_graph_context_for_state(
    validation_results: list[dict], 
    deploy_validation_result: dict | None, 
    template_yaml: str | None
) -> str:
    """Parses LangGraph state dictionary into clean query arrays for the Neo4j RAG pipeline."""
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

    if not errors:
        return "No errors provided."

    # Load Neo4j credentials from environment variables or use local defaults
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    try:
        rag_neo4j = CFNGraphRAGNeo4j(neo4j_uri, neo4j_user, neo4j_password)
        context = rag_neo4j.retrieve_schema_context(template_yaml, errors)
        rag_neo4j.close()
        return context
    except Exception as exc:
        return f"Neo4j retrieval failed: {exc}"