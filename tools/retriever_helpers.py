"""tools/retriever_helpers.py

State-extraction and LLM-response parsing utilities shared by the retriever
agent and the remediator agent.

Extracted from cfn_hybrid_rag.py so that:
- cfn_hybrid_rag.py owns only DB connectivity and retrieval logic.
- retriever.py owns only LLM orchestration and prompt assembly.
- These helpers, which depend only on state shape and stdlib, have no
  heavyweight dependencies and can be unit-tested without DB fixtures.

format_deploy_errors(), format_cfn_lint_errors(), and format_tflint_errors()
are defined here (not in remediator.py) so both the retriever and the
remediator render failures with the same level of detail.
"""
from __future__ import annotations

import json
import re

# Keywords that flag a CFN deployment event log line as actionable.
# NOTE: this list is used ONLY for the CloudFormation path.
# Terraform uses Error: block extraction instead (see _extract_tf_error_context).
_CFN_DEPLOY_ERROR_KEYWORDS = (
    "FAILED",
    "ERROR",
    "timed out",
    "does not exist",
    "InvalidAMI",
)

# How many lines of surrounding context to include around each Terraform
# Error: block when building the LLM-facing error summary.
_TF_CONTEXT_LINES = 5

# How many tail lines of the full Terraform log to always include.
_TF_TAIL_LINES = 5


def get_latest_stage_result(validation_results: list[dict], stage: str) -> dict | None:
    """Return the most recent ValidationResult for *stage*, or None if absent."""
    for result in reversed(validation_results):
        if result.get("stage") == stage:
            return result
    return None


# ---------------------------------------------------------------------------
# Shared error formatters
# ---------------------------------------------------------------------------

def format_cfn_lint_errors(errors: list[str]) -> str:
    """Format cfn-lint error strings as a bullet list.

    cfn-lint errors already contain the rule ID, resource path, and line
    reference — emitted as-is with consistent bullet style.
    """
    return "\n".join(f"  - {e.strip()}" for e in errors)


def format_tflint_errors(tflint_errors: list[str], tf_validate_errors: list[str]) -> str:
    """Format tflint and terraform-validate errors as a structured bullet list.

    Mirrors format_cfn_lint_errors() in style but handles two Terraform
    structural stages in a single call:
      - tflint (Stage 1): HCL style/best-practice rule violations
      - terraform-validate (Stage 2): provider type/attribute errors

    Both stages are rendered with a per-stage heading so the remediator
    LLM can distinguish rule-based linting issues (tflint) from hard type
    errors (terraform-validate) and apply the correct fix strategy.

    Either list may be empty — empty lists produce no heading.
    """
    parts: list[str] = []

    if tflint_errors:
        parts.append("**tflint (HCL lint):**")
        parts.extend(f"  - {e.strip()}" for e in tflint_errors)

    if tf_validate_errors:
        parts.append("**terraform-validate (provider schema):**")
        parts.extend(f"  - {e.strip()}" for e in tf_validate_errors)

    return "\n".join(parts) if parts else "No Terraform structural errors."


def _extract_tf_error_context(deploy_logs: list[str]) -> list[str]:
    """Extract context windows around every Terraform Error: block.

    Terraform apply writes structured error blocks to stdout:

        Error: <summary>

          on main.tf line N, in resource "type" "name":
          N: <hcl line>

        <detail paragraph>

    A pure keyword filter (FAILED, ERROR) misses most of these because:
      - "Error:" starts with a capital E followed by a colon, not the bare
        word ERROR that would match the CFN keyword list.
      - Progress lines like
            + parameter_group_name = (known after apply)
        contain "parameter" and trip a false-positive on the old CFN list.

    This function:
      1. Finds every line index where a Terraform error block starts
         (line starts with 'Error:' or '│ Error:' after stripping).
      2. For each hit, emits _TF_CONTEXT_LINES lines before it, the error
         line itself, and _TF_CONTEXT_LINES lines after it.
      3. Deduplicates overlapping windows (sequential errors).
      4. Always appends the last _TF_TAIL_LINES lines of the full log as
         a tail summary so the LLM sees where execution ended.

    Returns a list of formatted strings ready to bullet-point.
    """
    if not deploy_logs:
        return []

    # Identify error-block start indices
    error_line_re = re.compile(r'^(\u2502\s*)?Error:', re.IGNORECASE)
    error_indices: list[int] = [
        i for i, line in enumerate(deploy_logs)
        if error_line_re.match(line.strip())
    ]

    result_lines: list[str] = []
    covered: set[int] = set()

    if error_indices:
        result_lines.append("**Terraform error context (±5 lines per error block):**")
        for idx in error_indices:
            lo = max(0, idx - _TF_CONTEXT_LINES)
            hi = min(len(deploy_logs), idx + _TF_CONTEXT_LINES + 1)
            window = range(lo, hi)
            # Only emit a separator if this window doesn't directly follow the previous
            if covered and lo > max(covered) + 1:
                result_lines.append("  ---")
            for line_idx in window:
                if line_idx not in covered:
                    result_lines.append(f"  {deploy_logs[line_idx]}")
                    covered.add(line_idx)

    # Always include a tail summary
    tail = deploy_logs[-_TF_TAIL_LINES:]
    tail_start = len(deploy_logs) - len(tail)
    uncovered_tail = [
        line for i, line in enumerate(tail, start=tail_start)
        if i not in covered
    ]
    if uncovered_tail:
        if result_lines:
            result_lines.append("  ---")
        result_lines.append(f"**Last {_TF_TAIL_LINES} log lines:**")
        result_lines.extend(f"  {line}" for line in uncovered_tail)

    return result_lines


def format_deploy_errors(deploy_result: dict) -> str:
    """Format a deploy_validation_result dict into a structured error block.

    Each entry in failed_resources uses the canonical FailedResource shape:
      {"logical_name": str, "status_reason": str}
    as emitted by deploy_validator.py in every code path.

    Renders (in order):
      1. Deployment target (e.g. LOCALSTACK, AWS).
      2. Failed resources with logical ID + status reason.
      3. General error message when no per-resource breakdown is available.
      4. Resources that completed successfully (context for the LLM).
      5a. Terraform path: context windows around Error: blocks extracted by
          _extract_tf_error_context(), plus a tail summary.
      5b. CFN path: event log lines filtered by _CFN_DEPLOY_ERROR_KEYWORDS,
          or the last 5 log lines when no keyword matches.

    The iac_type is inferred from the target field — if target contains no
    CFN-specific signals and failed_resources use Terraform-style addresses
    (e.g. "aws_db_instance.mysql") it uses the Terraform extraction path.
    Callers may also pass iac_type explicitly via deploy_result["iac_type"].

    This function is the single source of truth for deploy error rendering.
    Both the retriever prompt and the remediator prompt call this function so
    both agents see the same level of detail.
    """
    target = deploy_result.get("target", "unknown").upper()
    lines: list[str] = [f"**Target:** {target}"]

    failed = deploy_result.get("failed_resources", [])
    if failed:
        lines.append("**Failed resources:**")
        for fr in failed:
            name   = fr.get("logical_name") or "unknown"
            reason = fr.get("status_reason") or "no reason provided"
            lines.append(f"  - `{name}`: {reason}")
    elif deploy_result.get("error_message"):
        lines.append(f"**Error:** {deploy_result['error_message']}")

    completed = deploy_result.get("completed_resources", [])
    if completed:
        lines.append(
            "**Completed successfully:** "
            + ", ".join(f"`{r}`" for r in completed)
        )

    deploy_logs = deploy_result.get("deployment_logs", [])

    # -----------------------------------------------------------------------
    # Choose log extraction strategy based on iac_type.
    # Prefer an explicit iac_type key; fall back to heuristic detection
    # (Terraform resource addresses contain a dot and no "::").
    # -----------------------------------------------------------------------
    explicit_iac_type: str = deploy_result.get("iac_type", "")
    if explicit_iac_type == "terraform":
        is_terraform = True
    elif explicit_iac_type == "cloudformation":
        is_terraform = False
    else:
        # Heuristic: if any failed resource uses a Terraform address style
        # (e.g. "aws_db_instance.mysql") rather than CFN ("AWS::RDS::DBInstance")
        is_terraform = any(
            "." in (fr.get("logical_name") or "")
            and "::" not in (fr.get("logical_name") or "")
            for fr in failed
        )

    if is_terraform:
        tf_log_lines = _extract_tf_error_context(deploy_logs)
        lines.extend(tf_log_lines)
    else:
        # Original CFN keyword-filter path (unchanged)
        actionable = [
            log_line for log_line in deploy_logs
            if any(kw in str(log_line) for kw in _CFN_DEPLOY_ERROR_KEYWORDS)
        ]
        if actionable:
            lines.append("**Deployment event log (errors only):**")
            for log_line in actionable:
                lines.append(f"  - {log_line}")
        elif not failed and deploy_logs:
            lines.append("**Last deployment events:**")
            for log_line in deploy_logs[-5:]:
                lines.append(f"  - {log_line}")

    if len(lines) == 1:
        lines.append("Deployment failed with no structured error details.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error extraction (used by retriever_agent)
# ---------------------------------------------------------------------------

def extract_errors(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
) -> list[str]:
    """Extract a list of error strings from the validation state.

    Deploy failures are rendered with format_deploy_errors() so the retriever
    LLM receives the same structured breakdown (target, failed resources,
    filtered event log) that the remediator prompt shows.
    """
    errors: list[str] = []

    for result in validation_results:
        stage = str(result.get("stage") or "").strip().lower()
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip():
                    errors.append(str(err))

    if (
        deploy_validation_result
        and not deploy_validation_result.get("passed")
        and deploy_validation_result.get("target") != "skipped"
    ):
        errors.append(format_deploy_errors(deploy_validation_result))

    return errors


def parse_query_response(raw: str, max_queries: int = 8) -> dict[str, list[str]]:
    """Parse the LLM's query-generation response.

    Accepts {"schema_queries": [...], "security_queries": [...]} object form.
    Strips markdown fences before parsing.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[RAG Tool] Query parse error (JSONDecodeError): {e}. Raw: {cleaned[:200]}")
        return {"schema_queries": [], "security_queries": []}

    schema_queries = []
    security_queries = []

    if isinstance(parsed, dict):
        schema_queries = parsed.get("schema_queries", [])
        security_queries = parsed.get("security_queries", [])
        # Fallback for old {"queries": [...]} format
        if not schema_queries and not security_queries:
             queries = parsed.get("queries") or parsed.get("query") or []
             schema_queries = queries
    elif isinstance(parsed, list):
        schema_queries = parsed
    else:
        print(f"[RAG Tool] Unexpected query response type: {type(parsed)}")
        return {"schema_queries": [], "security_queries": []}

    if not isinstance(schema_queries, list):
        schema_queries = []
    if not isinstance(security_queries, list):
        security_queries = []

    schema_result = [str(q).strip() for q in schema_queries if str(q).strip()][:max_queries]
    security_result = [str(q).strip() for q in security_queries if str(q).strip()][:max_queries]

    print(f"[RAG Tool] Parsed {len(schema_result)} schema queries and {len(security_result)} security queries from LLM response.")
    return {"schema_queries": schema_result, "security_queries": security_result}
