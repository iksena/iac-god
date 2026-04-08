# tools/cfn_graph_context.py
"""
GraphRAG context tool — graph-only (no BM25/FAISS) fallback path.

Given pre-extracted error query strings (produced by remediator._extract_error_queries),
looks up AWS resource types and properties directly in cfn_graph.pkl and returns
structured schema context for injection into the remediator prompt.

Used when the RAG index is not available. When the RAG index IS available,
cfn_graph_context_rag.get_cfn_schema_context() supersedes this.

Mirrors the interface of checkov_context.py / trivy_context.py.
"""
from __future__ import annotations

import pickle
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import networkx as nx

_GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "cfn_graph.pkl"

# Matches any AWS::Service::Resource token (also catches inner types in
# SSM parameter strings like AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>)
_AWS_TYPE_RE = re.compile(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+")

# Matches "AWS::X::Y/PropName" — cfn-lint property path format
_CFN_PROP_PATH_RE = re.compile(
    r"(AWS::[A-Za-z0-9]+::[A-Za-z0-9]+)/([A-Za-z0-9]+)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph loader
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph | None:
    if not _GRAPH_PATH.exists():
        return None
    with _GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj


# ─────────────────────────────────────────────────────────────────────────────
# Property-name reverse index
# Maps lowercase property name → parent resource types that own it.
# Fixes errors like E3021 where only property names appear (VpcId,
# SecurityGroupEgress) with no AWS:: type token in the error string.
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_prop_index(G: nx.DiGraph) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("ntype") != "Property":
            continue
        name = (data.get("name") or "").lower()
        if not name:
            continue
        for parent, _ in G.in_edges(node_id):
            pdata = G.nodes[parent]
            if pdata.get("ntype") in ("Resource", None) and parent.startswith("AWS::"):
                index.setdefault(name, []).append(parent)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Extraction — operates on pre-cleaned query strings from the remediator
# ─────────────────────────────────────────────────────────────────────────────

def _extract_resource_info(
    queries: list[str],
    G: nx.DiGraph,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    From a list of pre-extracted error query strings, identify:
      - resource_types: ordered deduplicated AWS::X::Y strings
      - property_hints: {resource_type: [prop_name, ...]} surfaced first in render

    Three lookup paths (in order of specificity):
      1. Explicit "AWS::X::Y/PropName" paths in the query string
      2. Bare "AWS::X::Y" token matches
      3. Property-name reverse index (handles E3021-style errors where only
         property names like "VpcId" or "SecurityGroupEgress" appear)
    """
    resource_types: list[str] = []
    property_hints: dict[str, list[str]] = {}
    seen_types: set[str] = set()
    prop_index = _build_prop_index(G)

    def _add_type(rtype: str) -> None:
        if rtype not in seen_types:
            seen_types.add(rtype)
            resource_types.append(rtype)

    def _add_hint(rtype: str, prop: str) -> None:
        hints = property_hints.setdefault(rtype, [])
        if prop not in hints:
            hints.append(prop)

    for query in queries:
        # Path 1: explicit property paths "AWS::X::Y/PropName"
        for rtype, prop in _CFN_PROP_PATH_RE.findall(query):
            _add_type(rtype)
            _add_hint(rtype, prop)

        # Path 2: bare AWS::X::Y tokens (also catches inner types in SSM strings)
        for rtype in _AWS_TYPE_RE.findall(query):
            _add_type(rtype)

        # Path 3: property-name reverse lookup
        # Tokenise on whitespace + strip punctuation — catches "VpcId",
        # "'SecurityGroupEgress'", "[KeyName]" etc.
        for token in query.split():
            token_clean = token.strip("'\"[]().,:")
            for rtype in prop_index.get(token_clean.lower(), []):
                _add_type(rtype)
                _add_hint(rtype, token_clean)

    return resource_types, property_hints


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

def _build_resource_block(
    G: nx.DiGraph,
    rtype: str,
    pinned_props: list[str],
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    props: list[dict] = []
    ptypes: list[dict] = []

    for _, neighbour in G.out_edges(rtype):
        nd = G.nodes[neighbour]
        if nd.get("ntype") == "Property":
            props.append(nd)
        elif nd.get("ntype") == "PropertyType":
            ptypes.append(nd)

    required  = [p for p in props if p.get("required")]
    optional  = [p for p in props if not p.get("required")]
    pinned_set = set(pinned_props)

    lines: list[str] = [f"### {rtype}"]

    if pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in pinned_props:
            match = next((p for p in props if p.get("name") == name), None)
            if match:
                prim = match.get("primitive_type") or match.get("type") or "Any"
                req  = "**required**" if match.get("required") else "optional"
                upd  = match.get("update_type", "")
                lines.append(
                    f"  - `{name}` ({prim}, {req}"
                    + (f" — UpdateType: {upd}" if upd else "") + ")"
                )
            else:
                lines.append(f"  - `{name}` *(invalid property name)*")

    req_rest = [p for p in required if p.get("name") not in pinned_set]
    if req_rest:
        parts = [
            f"`{p.get('name','?')}` ({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)"
            for p in req_rest
        ]
        lines.append("**Other required properties:** " + ", ".join(parts))
    elif not pinned_props:
        lines.append("**Required properties:** *(none)*")

    opt_rest = [p for p in optional if p.get("name") not in pinned_set]
    if opt_rest:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_rest[:max_optional]:
            name = p.get("name", "?")
            prim = p.get("primitive_type") or p.get("type") or "Any"
            upd  = p.get("update_type", "")
            lines.append(
                f"  - `{name}` ({prim}" + (f" — UpdateType: {upd}" if upd else "") + ")"
            )
        if len(opt_rest) > max_optional:
            lines.append(f"  - … and {len(opt_rest) - max_optional} more")

    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def get_cfn_graph_context(
    queries: list[str],
) -> str:
    """
    Build schema context from pre-extracted error query strings.

    Args:
        queries: Human-readable error strings produced by
                 remediator._extract_error_queries(state). No parsing of
                 validation_results or deploy logs is done here.

    Returns a Markdown string ready for injection into the remediator prompt.
    """
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    if not queries:
        return "No error queries provided."

    resource_types, property_hints = _extract_resource_info(queries, G)

    # Filter: only valid AWS::X::Y types that are Resource nodes (or unknown)
    # Drops ghost PropertyType nodes and malformed corpus entries
    valid_types = [
        rtype for rtype in resource_types
        if rtype.startswith("AWS::")
        and G.nodes.get(rtype, {}).get("ntype") in ("Resource", None, "")
    ]

    if not valid_types:
        return (
            "No AWS resource types identified in errors.\n"
            "(Errors may reference only property names, SSM parameter types, "
            "or non-resource constructs.)"
        )

    blocks: list[str] = []
    for rtype in valid_types:
        if rtype not in G:
            # Type mentioned in error but not in spec — still worth flagging
            blocks.append(
                f"### {rtype}\n"
                f"*(not in CFN spec — may be invalid, region-specific, or "
                f"an SSM parameter type rather than a deployable resource)*"
            )
            continue
        block = _build_resource_block(
            G, rtype, pinned_props=property_hints.get(rtype, [])
        )
        if block:
            blocks.append(block)

    if not blocks:
        return "No renderable schema context found."

    header = textwrap.dedent("""\
        Schema context from AWS CloudFormation Resource Specification v243.
        Only resources referenced in current errors are shown.
        Properties marked *(invalid property name)* must be removed or
        replaced with the correct name from the spec below.

    """)
    return header + "\n\n".join(blocks)