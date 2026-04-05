# tools/cfn_graph_context.py
"""
GraphRAG context tool: given cfn-lint or deploy validation errors, extract
the AWS resource types and properties that are failing, look them up in the
pre-built cfn_graph.pkl, and return structured schema context for injection
into the remediator prompt.

Only resources mentioned in errors are included — no template-wide scanning.
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

# "AWS::Service::Resource/PropertyName" — cfn-lint property path format
_CFN_LINT_PROP_RE = re.compile(
    r"(AWS::[A-Za-z0-9]+::[A-Za-z0-9]+)/([A-Za-z0-9]+)"
)
# Bare "AWS::Service::Resource" — fallback for both cfn-lint and deploy errors
_RESOURCE_TYPE_RE = re.compile(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph | None:
    """Load cfn_graph.pkl once and cache it for the process lifetime."""
    if not _GRAPH_PATH.exists():
        return None
    with _GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_cfnlint_resource_info(
    validation_results: list[dict],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Parse cfn-lint validation results.

    Returns:
        resource_types:  ordered deduplicated list of AWS::X::Y strings
        property_hints:  {resource_type: [prop_name, ...]} from error paths
    """
    resource_types: list[str] = []
    property_hints: dict[str, list[str]] = {}
    seen_types: set[str] = set()

    for result in validation_results:
        if result.get("stage") != "cfn-lint":
            continue

        for error in result.get("errors", []):
            error_str = str(error)

            # First: typed property paths "AWS::X::Y/PropName" (most specific)
            for rtype, prop in _CFN_LINT_PROP_RE.findall(error_str):
                if rtype not in seen_types:
                    seen_types.add(rtype)
                    resource_types.append(rtype)
                hints = property_hints.setdefault(rtype, [])
                if prop not in hints:
                    hints.append(prop)

            # Second: bare resource type mentions not caught above
            for rtype in _RESOURCE_TYPE_RE.findall(error_str):
                if rtype not in seen_types:
                    seen_types.add(rtype)
                    resource_types.append(rtype)

    return resource_types, property_hints


def _extract_deploy_resource_info(
    deploy_validation_result: dict | None,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Parse the deploy validation result for resource type mentions in
    error_message and failed_resources entries.

    Returns the same shape as _extract_cfnlint_resource_info so callers
    can merge them uniformly.
    """
    resource_types: list[str] = []
    property_hints: dict[str, list[str]] = {}
    seen_types: set[str] = set()

    if not deploy_validation_result:
        return resource_types, property_hints
    if deploy_validation_result.get("passed"):
        return resource_types, property_hints
    if deploy_validation_result.get("target") == "skipped":
        return resource_types, property_hints

    # Collect all free-form error strings from the deploy result
    error_strings: list[str] = []
    if deploy_validation_result.get("error_message"):
        error_strings.append(str(deploy_validation_result["error_message"]))
    for failed in deploy_validation_result.get("failed_resources", []):
        for val in failed.values():
            if val:
                error_strings.append(str(val))
    for log_line in deploy_validation_result.get("deployment_logs", []):
        error_strings.append(str(log_line))

    for error_str in error_strings:
        # Deploy errors rarely carry "AWS::X::Y/PropName" paths, but handle it
        for rtype, prop in _CFN_LINT_PROP_RE.findall(error_str):
            if rtype not in seen_types:
                seen_types.add(rtype)
                resource_types.append(rtype)
            hints = property_hints.setdefault(rtype, [])
            if prop not in hints:
                hints.append(prop)

        for rtype in _RESOURCE_TYPE_RE.findall(error_str):
            if rtype not in seen_types:
                seen_types.add(rtype)
                resource_types.append(rtype)

    return resource_types, property_hints


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_resource_block(
    G: nx.DiGraph,
    rtype: str,
    pinned_props: list[str],
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    """
    Build a Markdown schema block for a single resource type.
    Pinned props (from error paths) are surfaced first with full detail.
    """
    props: list[dict] = []
    ptypes: list[dict] = []

    for _, neighbour in G.out_edges(rtype):
        nd = G.nodes[neighbour]
        ntype = nd.get("ntype", "")
        if ntype == "Property":
            props.append(nd)
        elif ntype == "PropertyType":
            ptypes.append(nd)

    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]
    pinned_set = set(pinned_props)

    lines: list[str] = [f"### {rtype}"]

    # ── Pinned: properties explicitly named in errors ──────────────────────
    if pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in pinned_props:
            match = next((p for p in props if p.get("name") == name), None)
            if match:
                prim = match.get("primitive_type") or match.get("type") or "Any"
                req  = "**required**" if match.get("required") else "optional"
                upd  = match.get("update_type", "")
                upd_note = f" — UpdateType: {upd}" if upd else ""
                lines.append(f"  - `{name}` ({prim}, {req}{upd_note})")
            else:
                lines.append(
                    f"  - `{name}` *(not found in spec — invalid property name)*"
                )

    # ── Required properties not already pinned ────────────────────────────
    req_remainder = [p for p in required if p.get("name") not in pinned_set]
    if req_remainder:
        req_parts = [
            f"`{p.get('name','?')}` "
            f"({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)"
            for p in req_remainder
        ]
        lines.append("**Other required properties:** " + ", ".join(req_parts))
    elif not pinned_props:
        lines.append("**Required properties:** *(none)*")

    # ── Optional sample (excluding pinned) ────────────────────────────────
    opt_remainder = [p for p in optional if p.get("name") not in pinned_set]
    if opt_remainder:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_remainder[:max_optional]:
            name = p.get("name", "?")
            prim = p.get("primitive_type") or p.get("type") or "Any"
            upd  = p.get("update_type", "")
            upd_note = f" — UpdateType: {upd}" if upd else ""
            lines.append(f"  - `{name}` ({prim}{upd_note})")
        remainder = len(opt_remainder) - max_optional
        if remainder > 0:
            lines.append(f"  - … and {remainder} more optional properties")

    # ── Nested PropertyTypes ───────────────────────────────────────────────
    if ptypes:
        nested = [
            nd.get("name", "?").rsplit(".", 1)[-1]
            for nd in ptypes[:max_nested]
        ]
        lines.append(
            "**Nested property types:** " + ", ".join(f"`{n}`" for n in nested)
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public interface — used directly by remediator agent
# ---------------------------------------------------------------------------

def get_cfn_graph_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None = None,
) -> str:
    """
    Build schema context from cfn-lint errors and/or deploy validation errors
    only. No template-wide scanning — only broken resources get context.

    Args:
        validation_results:       state["validation_results"]
        deploy_validation_result: state["deploy_validation_result"]

    Returns a Markdown string ready for injection into the remediator prompt,
    or a short fallback message if nothing is actionable.
    """
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    # --- Extract from cfn-lint errors ---
    cfnlint_types, cfnlint_hints = _extract_cfnlint_resource_info(
        validation_results
    )

    # --- Extract from deploy errors ---
    deploy_types, deploy_hints = _extract_deploy_resource_info(
        deploy_validation_result
    )

    # --- Merge: cfn-lint first, then deploy-only types ---
    all_types = list(dict.fromkeys(cfnlint_types + deploy_types))

    if not all_types:
        return "No AWS resource types identified in cfn-lint or deploy errors."

    # Merge property hints from both sources
    merged_hints: dict[str, list[str]] = {}
    for rtype in all_types:
        hints: list[str] = []
        for h in cfnlint_hints.get(rtype, []) + deploy_hints.get(rtype, []):
            if h not in hints:
                hints.append(h)
        if hints:
            merged_hints[rtype] = hints

    # --- Build per-resource blocks ---
    blocks: list[str] = []
    for rtype in all_types:
        if rtype not in G:
            # Still worth noting — type might be entirely invalid
            blocks.append(
                f"### {rtype}\n"
                f"*(not found in CFN spec — resource type may be invalid or unsupported)*"
            )
            continue
        block = _build_resource_block(
            G, rtype, pinned_props=merged_hints.get(rtype, [])
        )
        blocks.append(block)

    # --- Label which source each block came from ---
    cfnlint_set = set(cfnlint_types)
    deploy_set  = set(deploy_types)

    labeled_blocks: list[str] = []
    for rtype, block in zip(all_types, blocks):
        sources: list[str] = []
        if rtype in cfnlint_set:
            sources.append("cfn-lint")
        if rtype in deploy_set:
            sources.append("deploy")
        label = f"*Source: {', '.join(sources)}*" if sources else ""
        labeled_blocks.append(f"{block}\n{label}" if label else block)

    header = textwrap.dedent("""\
        Schema context from AWS CloudFormation Resource Specification v243.
        Only resources referenced in current errors are shown.
        Properties marked *(not found in spec)* are invalid and must be
        removed or replaced with the correct property name.

    """)
    return header + "\n\n".join(labeled_blocks)