from __future__ import annotations

import re
import json
from datetime import datetime, timezone

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context
from tools.security_hybrid_rag import execute_security_retrieval
from tools.retriever_helpers import (
    get_latest_stage_result,
    format_cfn_lint_errors,
    format_deploy_errors,
    extract_errors,
)
from tools.template_annotator import render_annotated_template
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
# Policy source context builders
# ---------------------------------------------------------------------------

def _extract_security_findings(
    validation_results: list[dict],
    stage: str,
    results_key: str,
    items_path: list[str],
) -> list[dict[str, str]]:
    """Extract check-ID findings from a security tool's raw JSON output."""
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

    if isinstance(top, dict):
        for item in top.get(items_path[0], []):
            check_id = str(item.get("check_id") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})

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


def _build_trivy_findings_for_rag(
    validation_results: list[dict],
) -> list[dict]:
    """Extract Trivy misconfig objects from raw JSON output for RAG retrieval.

    Returns full misconfig dicts (with ID, Severity, Title, Message) so that
    security_hybrid_rag can embed the complete finding text rather than just
    the check ID — richer context leads to better semantic matching.
    """
    result = get_latest_stage_result(validation_results, "trivy")
    if not result:
        return []
    raw = result.get("raw_output", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    misconfigs: list[dict] = []
    for entry in data.get("Results", []):
        for m in entry.get("Misconfigurations", []):
            if m.get("Severity", "").lower() in ("high", "critical"):
                misconfigs.append({
                    "check_id":    str(m.get("ID") or "").strip(),
                    "Title":       str(m.get("Title") or "").strip(),
                    "Message":     str(m.get("Message") or "").strip(),
                    "Description": str(m.get("Description") or "").strip(),
                    "Severity":    str(m.get("Severity") or "").strip(),
                })
    return misconfigs


def _build_checkov_policy_source_context(validation_results: list[dict]) -> str:
    """Build Checkov-only policy source context (unchanged from original)."""
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

    checkov_context = get_checkov_policy_context(
        _filter_findings_by_check_ids(checkov_findings, allowed_check_ids)
    )
    if not checkov_context:
        return ""
    return "## Relevant Policy Source Context (Checkov)\n\n### Checkov Policy Source\n" + checkov_context


# ---------------------------------------------------------------------------
# Validation error section builder
# ---------------------------------------------------------------------------

def _build_validation_errors_text(state: GraphState) -> str:
    """Build the full validation error section for the remediator user prompt."""
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
        deduped = _dedupe_preserve_order(
            [str(e) for e in result.get("errors", []) if str(e).strip()]
        )
        if not deduped:
            continue

        stage = result["stage"]
        if stage == "cfn-lint":
            errors_text = format_cfn_lint_errors(deduped)
        else:
            errors_text = "\n".join(f"  - {e}" for e in deduped)

        error_blocks.append(f"### {stage.upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if (
        deploy_result
        and not deploy_result["passed"]
        and deploy_result["target"] != "skipped"
    ):
        error_blocks.append(
            f"### DEPLOYABILITY Errors\n{format_deploy_errors(deploy_result)}"
        )

    return "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."


# ---------------------------------------------------------------------------
# Context inclusion guards
# ---------------------------------------------------------------------------

def should_include_remediation_context(state: GraphState) -> bool:
    """Return True only when cfn-lint or deployment failures are present."""
    validation_results = state.get("validation_results", [])
    cfn_lint_result = get_latest_stage_result(validation_results, "cfn-lint")

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


def _should_include_security_context(state: GraphState) -> bool:
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
    # current_iteration was already incremented by validator_agent; use it
    # as-is for history labelling (reflects the iteration just completed).
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])),
    )

    cfn_graph_context = state.get("retriever_context", "")
    retrieval_queries = state.get("retriever_queries", [])

    include_cfn_schema  = should_include_remediation_context(state)
    include_security    = _should_include_security_context(state)

    if include_cfn_schema:
        print(
            f"[Remediator] CFN schema context: {len(cfn_graph_context)} chars, "
            f"{len(retrieval_queries)} retrieval queries."
        )
    else:
        cfn_graph_context = ""
        print("[Remediator] CFN schema context skipped (YAML/security-only failure).")

    # ------------------------------------------------------------------
    # Security remediation context — GraphRAG path (no retriever agent).
    # The Trivy findings are passed directly as embedding queries; the RAG
    # tool handles Chroma search → Neo4j traversal → CSV fallback.
    # Checkov context is kept on its existing CSV path.
    # ------------------------------------------------------------------
    security_rag_context    = ""
    checkov_policy_context  = ""

    if include_security:
        trivy_findings = _build_trivy_findings_for_rag(state["validation_results"])
        if trivy_findings:
            print(f"[Remediator] Running Security GraphRAG for {len(trivy_findings)} Trivy finding(s).")
            security_rag_context = execute_security_retrieval(trivy_findings)
            print(f"[Remediator] Security RAG context: {len(security_rag_context)} chars.")

        checkov_policy_context = _build_checkov_policy_source_context(state["validation_results"])

    # Merge security contexts: GraphRAG result (Trivy) + Checkov CSV.
    policy_source_context = "\n\n".join(
        p for p in (security_rag_context, checkov_policy_context) if p.strip()
    )

    formatted_errors = _build_validation_errors_text(state)

    flat_errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )
    annotated_template = render_annotated_template(
        template_yaml=state.get("cloudformation_template", ""),
        errors=flat_errors,
    )

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        annotated_template=annotated_template,
        validation_errors=formatted_errors,
        policy_source_context=policy_source_context,
        cfn_graph_context=cfn_graph_context,
        remediation_history_context="",
    )
    user_msg: Message = {"role": "user", "content": user_content}

    client, model = _build_client()
    content, usage = _call_llm_with_history(
        client,
        model,
        system,
        state.get("remediator_history", []) + [user_msg],
    )
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
        "iteration":         iteration,
        "errors":            state["validation_results"],
        "flat_errors":       flat_errors,
        "formatted_errors":  formatted_errors,
        "suggestion":        content,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "cfn_context":       cfn_graph_context,
        "retrieval_queries": retrieval_queries,
    }

    print("[Remediator] Suggestions generated. Routing back to Engineer.")
    # NOTE: current_iteration is NOT incremented here — validator_agent owns
    # the counter and already advanced it before this node ran.
    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "llm_call_log":        state["llm_call_log"] + [llm_record],
        "remediator_history":  append_and_cap(
            state.get("remediator_history", []), user_msg, assistant_msg
        ),
    }
