from __future__ import annotations

import re
import json
from datetime import datetime, timezone

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from agents.history_context import _build_remediation_history_context
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context
from tools.retriever_helpers import get_latest_stage_result
from tracking.recorder import ResearchRecorder


# ---------------------------------------------------------------------------
# Internal helpers — error extraction
# ---------------------------------------------------------------------------

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _extract_check_ids_from_errors(errors: list[object]) -> set[str]:
    check_ids: set[str] = set()
    for error in errors:
        text = str(error or "")
        for match in re.findall(r"\[([A-Z0-9_-]+)\]", text):
            check_ids.add(match.strip().upper())
        for match in re.findall(r"\b(?:AVD-)?AWS-\d{4}\b", text, flags=re.IGNORECASE):
            check_ids.add(match.strip().upper())
        for match in re.findall(r"\bCKV2?_[A-Z0-9_]+\b", text, flags=re.IGNORECASE):
            check_ids.add(match.strip().upper())
    return check_ids


# ---------------------------------------------------------------------------
# Policy source context builders (Checkov / Trivy)
# ---------------------------------------------------------------------------

def _extract_security_findings(
    validation_results: list[dict],
    stage: str,
    results_key: str,
    items_path: list[str],
) -> list[dict[str, str]]:
    """Extract check-ID findings from a security tool's raw JSON output.

    Replaces the near-identical _extract_checkov_findings and
    _extract_trivy_findings functions.  The two callers differ only in which
    JSON keys to traverse to reach the individual finding objects:

        Checkov: results_key="results", items_path=["failed_checks"]
                 finding id field: "check_id"
        Trivy:   results_key="Results", items_path=["Misconfigurations"]
                 finding id field: "ID"

    Args:
        validation_results: Full list of ValidationResult dicts from state.
        stage:              Stage name to look up ("checkov" or "trivy").
        results_key:        Top-level JSON key for the results object.
        items_path:         List of nested keys to reach the findings list.
                            Supports one level of nesting (e.g. ["failed_checks"]
                            inside data[results_key]) or two levels
                            (e.g. data[results_key][*][items_path[0]]).

    Returns:
        List of {"check_id": str} dicts, deduplicated by insertion order.
    """
    findings: list[dict[str, str]] = []
    result = get_latest_stage_result(validation_results, stage)
    if not result:
        return findings

    raw = result.get("raw_output", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings

    top = data.get(results_key, {})

    # Checkov: data["results"]["failed_checks"] — top is a dict
    if isinstance(top, dict):
        for item in top.get(items_path[0], []):
            check_id = str(item.get("check_id") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})

    # Trivy: data["Results"] is a list; each element has data["Results"][*][items_path[0]]
    elif isinstance(top, list):
        for entry in top:
            for item in entry.get(items_path[0], []):
                check_id = str(item.get("ID") or "").strip()
                if check_id:
                    findings.append({"check_id": check_id})

    return findings


def _filter_findings_by_check_ids(
    findings: list[dict[str, str]],
    allowed_check_ids: set[str],
) -> list[dict[str, str]]:
    if not allowed_check_ids:
        return findings

    allowed = {cid.strip().upper() for cid in allowed_check_ids if cid}
    seen: set[str] = set()
    filtered: list[dict[str, str]] = []

    for finding in findings:
        check_id = str(finding.get("check_id") or finding.get("rule_id") or "").strip()
        if not check_id:
            continue
        normalized = check_id.upper()
        if normalized not in allowed or normalized in seen:
            continue
        seen.add(normalized)
        filtered.append({"check_id": check_id})

    return filtered


def _build_policy_source_context(validation_results: list[dict]) -> str:
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    allowed_check_ids: set[str] = set()
    for result in latest_by_stage.values():
        allowed_check_ids.update(_extract_check_ids_from_errors(result.get("errors", [])))

    if not allowed_check_ids:
        return ""

    checkov_findings = _extract_security_findings(
        validation_results,
        stage="checkov",
        results_key="results",
        items_path=["failed_checks"],
    )
    trivy_findings = _extract_security_findings(
        validation_results,
        stage="trivy",
        results_key="Results",
        items_path=["Misconfigurations"],
    )

    checkov_context = get_checkov_policy_context(
        _filter_findings_by_check_ids(checkov_findings, allowed_check_ids)
    )
    trivy_context = get_trivy_policy_context(
        _filter_findings_by_check_ids(trivy_findings, allowed_check_ids)
    )

    sections: list[str] = []
    if checkov_context:
        sections.append(f"### Checkov Policy Source\n{checkov_context}")
    if trivy_context:
        sections.append(f"### Trivy Policy Source\n{trivy_context}")
    if not sections:
        return ""

    return "## Relevant Policy Source Context (Checkov/Trivy)\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Validation error formatter
# ---------------------------------------------------------------------------

def _build_validation_errors_text(state: GraphState) -> str:
    error_blocks: list[str] = []

    validation_results = state.get("validation_results", [])
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    for result in latest_by_stage.values():
        if result["passed"]:
            continue
        deduped_errors = _dedupe_preserve_order(
            [str(e) for e in result.get("errors", []) if str(e).strip()]
        )
        if not deduped_errors:
            continue
        errors_text = "\n".join(f"  - {e}" for e in deduped_errors)
        error_blocks.append(f"### {result['stage'].upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if deploy_result and not deploy_result["passed"] and deploy_result["target"] != "skipped":
        target = deploy_result["target"].upper()
        lines: list[str] = []

        failed = deploy_result.get("failed_resources", [])
        if failed:
            lines.append("**Failed resources:**")
            for fr in failed:
                name = fr.get("logical_name") or fr.get("resource") or "unknown"
                reason = fr.get("status_reason") or fr.get("reason") or "no reason provided"
                lines.append(f"  - `{name}`: {reason}")
        elif deploy_result.get("error_message"):
            lines.append(f"**Error:** {deploy_result['error_message']}")

        completed = deploy_result.get("completed_resources", [])
        if completed:
            lines.append(
                "**Resources that completed successfully:** "
                + ", ".join(f"`{r}`" for r in completed)
            )

        deploy_logs = deploy_result.get("deployment_logs", [])
        actionable = [
            line for line in deploy_logs
            if any(kw in str(line) for kw in (
                "FAILED", "ERROR", "timed out", "does not exist", "InvalidAMI", "parameter"
            ))
        ]
        if actionable:
            lines.append("**Deployment event log (errors only):**")
            for log_line in actionable:
                lines.append(f"  - {log_line}")
        elif not failed and deploy_logs:
            lines.append("**Last deployment events:**")
            for log_line in deploy_logs[-5:]:
                lines.append(f"  - {log_line}")

        if not lines:
            lines.append("Deployment failed with no structured error details.")

        error_blocks.append(
            f"### DEPLOYABILITY Errors ({target})\n" + "\n".join(lines)
        )

    return "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."


# ---------------------------------------------------------------------------
# Context inclusion guards
# ---------------------------------------------------------------------------

def should_include_remediation_context(state: GraphState) -> bool:
    """Return True only for YAML, cfn-lint, or deploy failures.

    CFN schema retrieval context is not useful for pure security policy
    violations (checkov, trivy) — those require policy source context instead.
    """
    validation_results = state.get("validation_results", [])
    yaml_result     = get_latest_stage_result(validation_results, "yaml")
    cfn_lint_result = get_latest_stage_result(validation_results, "cfn-lint")

    if yaml_result and not yaml_result.get("passed", True):
        return True
    if cfn_lint_result and not cfn_lint_result.get("passed", True):
        return True

    deploy_result = state.get("deploy_validation_result")
    if (
        deploy_result
        and not deploy_result.get("passed", True)
        and deploy_result.get("target") != "skipped"
    ):
        return True

    return False


def _should_include_policy_source_context(state: GraphState) -> bool:
    """Return True for Trivy/Checkov failures."""
    validation_results = state.get("validation_results", [])
    for stage in ("trivy", "checkov"):
        result = get_latest_stage_result(validation_results, stage)
        if result and not result.get("passed", True):
            return True
    return False


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])),
    )

    # Pull retriever outputs from state — written by retriever_agent in the
    # previous graph node. Keys match exactly what retriever_agent returns.
    cfn_graph_context  = state.get("retriever_context", "")
    retrieval_queries  = state.get("retriever_queries", [])

    # Guard: only inject heavy context when it is actionable for the error type.
    include_remediation = should_include_remediation_context(state)
    include_policy      = _should_include_policy_source_context(state)

    if include_remediation:
        print(
            f"[Remediator] CFN schema context from retriever: {len(cfn_graph_context)} chars, "
            f"{len(retrieval_queries)} retrieval queries used."
        )
    else:
        cfn_graph_context = ""
        print("[Remediator] CFN schema context skipped (non-YAML/cfn-lint/deploy failure).")

    policy_source_context = (
        _build_policy_source_context(state["validation_results"])
        if include_policy
        else ""
    )

    # Build structured history context from past RemediationHistory entries.
    # The model receives a compact document rather than a verbatim transcript,
    # avoiding the exploitative incremental-edit behaviour documented in
    # AgentCoder / FAIR multi-turn findings.
    remediation_history_context = _build_remediation_history_context(
        state.get("remediation_history", [])
    )

    formatted_errors = _build_validation_errors_text(state)
    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=formatted_errors,
        policy_source_context=policy_source_context,
        cfn_graph_context=cfn_graph_context,
        remediation_history_context=remediation_history_context,
    )
    user_msg: Message = {"role": "user", "content": user_content}

    # Single-turn call: no conversation history passed — full context is in
    # the prompt. remediator_history is retained in state for debugging only.
    client, model = _build_client()
    content, usage = _call_llm_with_history(client, model, system, [user_msg])
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state,
        agent="remediator",
        model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content,
        token_usage=usage,
    )

    new_history_entry: RemediationHistory = {
        "iteration":        iteration,
        "errors":           state["validation_results"],
        "formatted_errors": formatted_errors,
        "suggestion":       content,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "cfn_context":      cfn_graph_context,
        "retrieval_queries": retrieval_queries,
    }

    print("[Remediator] Suggestions generated. Routing back to Engineer.")
    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "current_iteration":   iteration + 1,
        "llm_call_log":        state["llm_call_log"] + [llm_record],
        # Kept for debugging/recording only — not used as LLM context.
        "remediator_history":  append_and_cap(
            state["remediator_history"], user_msg, assistant_msg
        ),
    }
