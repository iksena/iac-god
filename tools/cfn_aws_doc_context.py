"""
AWS Documentation MCP context tool — alternative to cfn_graph_context_rag.py.

Instead of querying a locally-built FAISS/BM25/GraphRAG index, this module
calls the AWS Documentation MCP Server (awslabs.aws-documentation-mcp-server)
at runtime to fetch live, authoritative CloudFormation documentation for the
resource types present in current validation errors.

Retrieval strategy mirrors the three-route RAG design but replaces the
offline index with two MCP tool calls per resource type:

  Step 1 — Resource extraction
      Parse CloudFormation YAML template to resolve logical names → AWS::X::Y
      types (Route A, same as cfn_graph_context_rag).  Extract resource types
      from cfn-lint / Trivy / deploy errors as fallback (Route B text parse).

  Step 2 — search_documentation (MCP)
      For each unique resource type, call search_documentation with a focused
      query like "AWS::S3::Bucket CloudFormation resource properties".
      Collect the top result URLs.

  Step 3 — read_documentation (MCP)
      For each URL returned, call read_documentation to retrieve the full
      Markdown page.  Extract the Properties section to keep context size
      manageable.

  Step 4 — Render
      Format the fetched documentation into Markdown blocks identical in
      structure to cfn_graph_context_rag output so the remediator prompt
      template needs no changes.

Advantages over the RAG approach:
  - Always up-to-date (live AWS docs, no stale offline index).
  - No offline build step (no FAISS/BM25/sentence-transformers dependency).
  - Covers new/preview resource types not yet in the spec snapshot.

Trade-offs:
  - Requires network access and the MCP server to be running.
  - Higher latency per call (HTTP round-trips vs in-process index lookup).
  - Subject to AWS docs site availability.

Usage:
    from tools.cfn_aws_doc_context import get_cfn_aws_doc_context_for_state

    context_str = get_cfn_aws_doc_context_for_state(
        validation_results=state["validation_results"],
        deploy_validation_result=state.get("deploy_validation_result"),
        template_yaml=state["iac_template"],
    )
"""
from __future__ import annotations

import json
import textwrap
from functools import lru_cache
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

# ── MCP Server base URL ───────────────────────────────────────────────────────
# The awslabs.aws-documentation-mcp-server exposes an HTTP/SSE transport when
# launched with `uvx awslabs.aws-documentation-mcp-server@latest --transport http`.
# Override via environment variable AWS_DOC_MCP_BASE_URL if your deployment
# differs (e.g. Docker container, different port).
import os

_MCP_BASE_URL = os.environ.get(
    "AWS_DOC_MCP_BASE_URL", "http://localhost:8080"
).rstrip("/")

# Maximum characters to keep from a documentation page (prevents token overflow)
_MAX_DOC_CHARS = 8_000

# Maximum number of resource types to fetch docs for in one remediator call
_MAX_RESOURCE_TYPES = 6

# Stages whose errors carry no resource-schema information
_SYNTACTIC_STAGES = {"yaml", "json", "comments"}

# CloudFormation User Guide base URL for direct resource reference pages
_CFN_UG_BASE = "https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide"


# ─────────────────────────────────────────────────────────────────────────────
# MCP client helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mcp_search_documentation(search_phrase: str, limit: int = 3) -> list[dict]:
    """
    Call the MCP server's search_documentation tool via HTTP JSON-RPC.
    Returns a list of result dicts with keys: title, url, excerpt.

    Falls back to an empty list on any network / parse error so the caller
    degrades gracefully.
    """
    if not _HTTPX_OK:
        return []

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_documentation",
            "arguments": {
                "search_phrase": search_phrase,
                "limit": limit,
            },
        },
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{_MCP_BASE_URL}/mcp", json=payload)
            resp.raise_for_status()
            data = resp.json()
        result = data.get("result", {})
        # MCP tool result is in result["content"][0]["text"] as JSON
        raw_text = ""
        for part in result.get("content", []):
            if part.get("type") == "text":
                raw_text = part["text"]
                break
        if not raw_text:
            return []
        parsed = json.loads(raw_text)
        return parsed.get("results", []) if isinstance(parsed, dict) else []
    except Exception:
        return []


def _mcp_read_documentation(url: str) -> str:
    """
    Call the MCP server's read_documentation tool.
    Returns the page as Markdown text, truncated to _MAX_DOC_CHARS.
    Returns empty string on failure.
    """
    if not _HTTPX_OK:
        return ""

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "read_documentation",
            "arguments": {"url": url},
        },
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(f"{_MCP_BASE_URL}/mcp", json=payload)
            resp.raise_for_status()
            data = resp.json()
        result = data.get("result", {})
        for part in result.get("content", []):
            if part.get("type") == "text":
                return part["text"][:_MAX_DOC_CHARS]
        return ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Direct URL construction (no search needed for well-known types)
# ─────────────────────────────────────────────────────────────────────────────

def _cfn_resource_doc_url(resource_type: str) -> str:
    """
    Build the canonical CFN User Guide URL for a resource type.

    AWS::S3::Bucket  →  https://docs.aws.amazon.com/AWSCloudFormation/
                         latest/UserGuide/aws-resource-s3-bucket.html
    AWS::EC2::SecurityGroup  →  aws-resource-ec2-securitygroup.html

    The pattern is: "aws-resource-" + lower(service) + "-" + lower(resourcename).
    """
    parts = resource_type.split("::")  # ["AWS", "S3", "Bucket"]
    if len(parts) != 3:
        return ""
    _, service, resource = parts
    slug = f"aws-resource-{service.lower()}-{resource.lower()}"
    return f"{_CFN_UG_BASE}/{slug}.html"


# ─────────────────────────────────────────────────────────────────────────────
# Resource type extraction (mirrors Route A + B in cfn_graph_context_rag)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_template_resource_map(template_yaml: str | None) -> dict[str, str]:
    """Return {LogicalName: 'AWS::X::Y'} from template YAML. Empty on failure."""
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


def _extract_resource_types_from_errors(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    logical_to_type: dict[str, str],
) -> list[str]:
    """
    Collect AWS resource types referenced in validation errors.
    Preserves insertion order; deduplicates.  Failed/deploy resources first.
    """
    seen: dict[str, int] = {}   # type → insertion_order
    counter = 0

    def _add(rtype: str) -> None:
        nonlocal counter
        if rtype and rtype not in seen:
            seen[rtype] = counter
            counter += 1

    # ── Deploy failures (highest signal — most likely cause of remediation need) ─
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("target") != "skipped":
            for fr in deploy_validation_result.get("failed_resources", []):
                logical = fr.get("logical_name") or fr.get("resource") or ""
                if logical and logical in logical_to_type:
                    _add(logical_to_type[logical])
            for line in deploy_validation_result.get("deployment_logs", []):
                line_str = str(line)
                if ": " not in line_str:
                    continue
                candidate = line_str.split(": ", 1)[0].strip()
                if candidate in logical_to_type:
                    _add(logical_to_type[candidate])

    # ── cfn-lint / checkov / trivy structured errors ──────────────────────────
    for result in validation_results:
        if result.get("stage") in _SYNTACTIC_STAGES:
            continue
        for error in result.get("errors", []):
            if isinstance(error, dict):
                logical = error.get("resource") or error.get("logical_id") or ""
                if logical and logical in logical_to_type:
                    _add(logical_to_type[logical])

    # ── Fallback: all template resource types (for stack-level deploy errors) ─
    if not seen:
        for rtype in logical_to_type.values():
            _add(rtype)

    return list(seen.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Properties section extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_properties_section(markdown: str) -> str:
    """
    Heuristically extract the 'Properties' section from CFN doc markdown.
    Returns the section text, or the full markdown if the section isn't found.
    Keeps the output under _MAX_DOC_CHARS.
    """
    lines = markdown.splitlines()
    start_idx: int | None = None
    end_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower().lstrip("#").strip()
        if start_idx is None and stripped in ("properties", "resource properties"):
            start_idx = i
        elif start_idx is not None and line.startswith("#"):
            # Next heading at same or higher level → end of section
            section_level = len(line) - len(line.lstrip("#"))
            start_level  = len(lines[start_idx]) - len(lines[start_idx].lstrip("#"))
            if section_level <= start_level:
                end_idx = i
                break

    if start_idx is not None:
        section_lines = lines[start_idx:end_idx]
        return "\n".join(section_lines)[:_MAX_DOC_CHARS]

    # Properties section not found — return the full doc (already truncated upstream)
    return markdown[:_MAX_DOC_CHARS]


# ─────────────────────────────────────────────────────────────────────────────
# Per-resource fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_resource_context(resource_type: str) -> str:
    """
    Fetch documentation for a single resource type.  Strategy:

    1. Try the canonical CFN User Guide URL directly (read_documentation).
    2. On empty/error, fall back to search_documentation then read first result.

    Returns a formatted Markdown block, or an error placeholder.
    """
    # ── Step 1: direct URL ────────────────────────────────────────────────────
    direct_url = _cfn_resource_doc_url(resource_type)
    doc_text = ""
    source_url = ""

    if direct_url:
        doc_text = _mcp_read_documentation(direct_url)
        if doc_text:
            source_url = direct_url

    # ── Step 2: search fallback ───────────────────────────────────────────────
    if not doc_text:
        query = f"{resource_type} CloudFormation resource properties"
        results = _mcp_search_documentation(query, limit=2)
        for r in results:
            url = r.get("url", "")
            if url:
                doc_text = _mcp_read_documentation(url)
                if doc_text:
                    source_url = url
                    break

    if not doc_text:
        return (
            f"### {resource_type}\n"
            f"*(documentation not available — check network / MCP server)*\n"
        )

    properties_section = _extract_properties_section(doc_text)
    return textwrap.dedent(f"""\
        ### {resource_type}
        *Source: [{source_url}]({source_url})*

        {properties_section}
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def get_cfn_aws_doc_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None = None,
    template_yaml: str | None = None,
    *,
    max_resource_types: int = _MAX_RESOURCE_TYPES,
    max_workers: int = 4,
) -> str:
    """
    AWS Documentation MCP alternative to get_cfn_graph_context_for_state.

    Fetches live CloudFormation resource documentation via the
    awslabs.aws-documentation-mcp-server for every resource type found in
    the current validation errors.

    Args:
        validation_results:       state["validation_results"]
        deploy_validation_result: state.get("deploy_validation_result")
        template_yaml:            state["iac_template"]
        max_resource_types:       cap on how many resource types to fetch
        max_workers:              parallel HTTP workers (default 4)

    Returns:
        Markdown string ready for injection into REMEDIATOR_USER prompt
        under the {cfn_graph_context} placeholder.
    """
    logical_to_type = _parse_template_resource_map(template_yaml)
    resource_types  = _extract_resource_types_from_errors(
        validation_results, deploy_validation_result, logical_to_type
    )

    if not resource_types:
        return (
            "No AWS resource types identified in errors.\n"
            "(Errors may be syntactic — YAML formatting, indentation, etc.)\n"
        )

    # Cap to avoid excessive latency / token usage
    resource_types = resource_types[:max_resource_types]
    skipped = len(resource_types) - len(resource_types[:max_resource_types])

    # Fetch in parallel — each resource is an independent HTTP round-trip
    blocks: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_rtype = {
            pool.submit(_fetch_resource_context, rtype): rtype
            for rtype in resource_types
        }
        for future in as_completed(future_to_rtype):
            rtype = future_to_rtype[future]
            try:
                blocks[rtype] = future.result()
            except Exception as exc:
                blocks[rtype] = (
                    f"### {rtype}\n"
                    f"*(fetch error: {exc})*\n"
                )

    # Re-order to match original priority (failed resources first)
    ordered_blocks = [blocks[rt] for rt in resource_types if rt in blocks]

    header = textwrap.dedent(f"""\
        Schema context from AWS CloudFormation Documentation (live, via AWS Documentation MCP Server).
        Resource types: {", ".join(resource_types)}.
        {f"(+ {skipped} additional types skipped due to limit)" if skipped else ""}

    """)

    return header + "\n\n---\n\n".join(ordered_blocks)
