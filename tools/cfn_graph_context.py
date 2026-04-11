# tools/cfn_graph_context.py
"""
Deterministic Graph Context Tool (No RAG/FAISS).

Parses the current YAML template to find actual AWS resource types, then 
cross-references the validation/deploy errors to highlight problematic properties.
Pulls exact schemas from the pre-built NetworkX cfn_graph.pkl.
"""
from __future__ import annotations

import pickle
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import networkx as nx

try:
    import yaml as _yaml
    
    # Register a generic multi-constructor to safely ignore CFN tags (!Ref, !Sub, etc.)
    def _cfn_tag_constructor(loader, tag_suffix, node):
        if isinstance(node, _yaml.ScalarNode):
            return loader.construct_scalar(node)
        elif isinstance(node, _yaml.SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, _yaml.MappingNode):
            return loader.construct_mapping(node)
            
    _yaml.SafeLoader.add_multi_constructor("!", _cfn_tag_constructor)
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "cfn_graph.pkl"
_AWS_TYPE_RE = re.compile(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+")

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph | None:
    if not _GRAPH_PATH.exists():
        return None
    with _GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj

def _parse_template_resource_map(template_yaml: str | None) -> dict[str, str]:
    """Extracts a map of {LogicalId: AWS::Service::Type} from the current template."""
    if not template_yaml or not _YAML_OK: 
        return {}
    try:
        tpl = _yaml.safe_load(template_yaml)
        if not isinstance(tpl, dict): 
            return {}
        return {
            name: body["Type"]
            for name, body in tpl.get("Resources", {}).items()
            if isinstance(body, dict) and "Type" in body
        }
    except Exception:
        return {}

@lru_cache(maxsize=1)
def _build_prop_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """Maps lowercase property name → parent resource types that own it."""
    index: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("ntype") == "Property":
            name = (data.get("name") or "").lower()
            if not name: continue
            for parent, _ in G.in_edges(node_id):
                # Using ("ResourceType", "Resource", None) to handle graph metadata variations
                if G.nodes[parent].get("ntype") in ("ResourceType", "Resource", None) and parent.startswith("AWS::"):
                    index.setdefault(name, []).append(parent)
    return index

def _extract_errors(validation_results: list[dict], deploy_validation_result: dict | None) -> list[str]:
    queries = []
    for result in validation_results:
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip(): queries.append(str(err))
    
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("error_message"): 
            queries.append(deploy_validation_result["error_message"])
        for fr in deploy_validation_result.get("failed_resources", []):
            name = fr.get("logical_name") or fr.get("resource") or ""
            reason = fr.get("status_reason") or fr.get("reason") or ""
            if name or reason: queries.append(f"{name} {reason}")
    return queries

def _build_resource_block(
    G: nx.DiGraph,
    rtype: str,
    pinned_props: set[str],
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    props, ptypes = [], []
    for _, neighbour in G.out_edges(rtype):
        nd = G.nodes[neighbour]
        if nd.get("ntype") == "Property": props.append(nd)
        elif nd.get("ntype") == "PropertyType": ptypes.append(nd)

    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]

    lines = [f"### {rtype}"]

    if pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in sorted(pinned_props):
            match = next((p for p in props if p.get("name") == name), None)
            if match:
                prim = match.get("primitive_type") or match.get("type") or "Any"
                req  = "**required**" if match.get("required") else "optional"
                lines.append(f"  - `{name}` ({prim}, {req})")
            else:
                lines.append(f"  - `{name}` *(invalid property name for this resource)*")

    req_rest = [p for p in required if p.get("name") not in pinned_props]
    if req_rest:
        parts = [f"`{p.get('name','?')}` ({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)" for p in req_rest]
        lines.append("**Required properties:** " + ", ".join(parts))

    opt_rest = [p for p in optional if p.get("name") not in pinned_props]
    if opt_rest:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_rest[:max_optional]:
            lines.append(f"  - `{p.get('name', '?')}` ({p.get('primitive_type') or p.get('type') or 'Any'})")
        if len(opt_rest) > max_optional:
            lines.append(f"  - … and {len(opt_rest) - max_optional} more")

    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)

def get_cfn_graph_context(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    template_yaml: str | None,
) -> str:
    """Deterministic exact-match context builder."""
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    errors = _extract_errors(validation_results, deploy_validation_result)
    if not errors:
        return "No errors provided."

    # 1. Identify types in the current template + explicit AWS::X::Y mentions in errors
    logical_to_type = _parse_template_resource_map(template_yaml)
    active_types = set(logical_to_type.values())
    
    for error in errors:
        for match in _AWS_TYPE_RE.findall(error):
            active_types.add(match)

    if not active_types:
        return "No AWS resource types identified in the template or errors."

    # 2. Map properties mentioned in errors strictly to active resource types
    prop_index = _build_prop_index(G)
    property_hints: dict[str, set[str]] = {rtype: set() for rtype in active_types}
    
    for error in errors:
        # Tokenize on punctuation/whitespace
        for token in error.split():
            token_clean = token.strip("'\"[]().,:")
            if not token_clean: continue
            
            # If the token is a known property name
            for rtype in prop_index.get(token_clean.lower(), []):
                # ONLY attach it if the resource type is actually in the user's template
                if rtype in active_types:
                    property_hints[rtype].add(token_clean)

    # 3. Render blocks only for active types
    blocks = []
    for rtype in active_types:
        if rtype not in G:
            blocks.append(f"### {rtype}\n*(Not found in local CloudFormation specification)*")
            continue
            
        block = _build_resource_block(G, rtype, pinned_props=property_hints[rtype])
        if block:
            blocks.append(block)

    if not blocks:
        return "No renderable schema context found."

    header = textwrap.dedent("""\
        ## Relevant AWS CloudFormation Resource Schemas
        The following schemas are for resources currently detected in your template.
        Properties marked *(invalid property name for this resource)* are causing validation failures and must be corrected.

    """)
    return header + "\n\n".join(blocks)