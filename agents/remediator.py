from datetime import datetime, timezone
import json
import re
from state import GraphState, RemediationHistory, Message, append_and_cap
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tracking.recorder import ResearchRecorder
from agents.engineer import _build_client, _call_llm_with_history
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


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


def _get_latest_stage_result(validation_results: list[dict], stage: str) -> dict | None:
    for result in reversed(validation_results):
        if result.get("stage") == stage:
            return result
    return None


def _extract_checkov_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    result = _get_latest_stage_result(validation_results, "checkov")
    if not result:
        return findings

    raw = result.get("raw_output", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings

    failed = data.get("results", {}).get("failed_checks", [])
    for item in failed:
        check_id = str(item.get("check_id") or "").strip()
        if check_id:
            findings.append({"check_id": check_id})
    return findings


def _extract_trivy_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    result = _get_latest_stage_result(validation_results, "trivy")
    if not result:
        return findings

    raw = result.get("raw_output", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings

    for r in data.get("Results", []):
        for misconfig in r.get("Misconfigurations", []):
            check_id = str(misconfig.get("ID") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})
    return findings


def _filter_findings_by_check_ids(
    findings: list[dict[str, str]],
    allowed_check_ids: set[str],
) -> list[dict[str, str]]:
    if not allowed_check_ids:
        return findings

    allowed = {check_id.strip().upper() for check_id in allowed_check_ids if check_id}
    filtered: list[dict[str, str]] = []
    seen: set[str] = set()

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
        if not stage:
            continue
        latest_by_stage[stage] = result

    allowed_check_ids: set[str] = set()
    for result in latest_by_stage.values():
        allowed_check_ids.update(_extract_check_ids_from_errors(result.get("errors", [])))

    if not allowed_check_ids:
        return ""

    checkov_context = get_checkov_policy_context(
        _filter_findings_by_check_ids(
            _extract_checkov_findings(validation_results),
            allowed_check_ids,
        )
    )
    trivy_context = get_trivy_policy_context(
        _filter_findings_by_check_ids(
            _extract_trivy_findings(validation_results),
            allowed_check_ids,
        )
    )

    sections: list[str] = []
    if checkov_context:
        sections.append(f"### Checkov Policy Source\n{checkov_context}")
    if trivy_context:
        sections.append(f"### Trivy Policy Source\n{trivy_context}")
    if not sections:
        return ""

    policy_context = "## Relevant Policy Source Context (Checkov/Trivy)\n\n" + "\n\n".join(sections)
    return policy_context


def _build_validation_errors_text(state: GraphState) -> str:
    error_blocks: list[str] = []

    validation_results = state.get("validation_results", [])
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if not stage:
            continue
        latest_by_stage[stage] = result

    for result in latest_by_stage.values():
        if result["passed"]:
            continue
        deduped_errors = _dedupe_preserve_order(
            [str(error) for error in result.get("errors", []) if str(error).strip()]
        )
        if not deduped_errors:
            continue
        errors_text = "\n".join(f"  - {error}" for error in deduped_errors)
        error_blocks.append(f"### {result['stage'].upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if deploy_result and not deploy_result["passed"] and deploy_result["target"] != "skipped":
        target = deploy_result["target"].upper()
        lines: list[str] = []

        failed = deploy_result.get("failed_resources", [])
        if failed:
            lines.append("**Failed resources:**")
            for fr in failed:
                name   = fr.get("logical_name") or fr.get("resource") or "unknown"
                reason = fr.get("status_reason") or fr.get("reason") or "no reason provided"
                lines.append(f"  - `{name}`: {reason}")
        else:
            if deploy_result.get("error_message"):
                lines.append(f"**Error:** {deploy_result['error_message']}")

        completed = deploy_result.get("completed_resources", [])
        if completed:
            lines.append(f"**Resources that completed successfully:** {', '.join(f'`{r}`' for r in completed)}")

        deploy_logs = deploy_result.get("deployment_logs", [])
        actionable_log_lines = [
            line for line in deploy_logs
            if any(kw in str(line) for kw in ("FAILED", "ERROR", "timed out", "does not exist", "InvalidAMI", "parameter"))
        ]
        if actionable_log_lines:
            lines.append("**Deployment event log (errors only):**")
            for log_line in actionable_log_lines:
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

    if not error_blocks:
        return "No validation errors reported."
    return "\n\n".join(error_blocks)


def _should_include_remediation_context(state: GraphState) -> bool:
    """Include heavy context only for YAML, cfn-lint, or deploy failures."""
    validation_results = state.get("validation_results", [])
    yaml_result = _get_latest_stage_result(validation_results, "yaml")
    cfn_lint_result = _get_latest_stage_result(validation_results, "cfn-lint")

    if yaml_result and not yaml_result.get("passed", True):
        return True
    if cfn_lint_result and not cfn_lint_result.get("passed", True):
        return True

    deploy_result = state.get("deploy_validation_result")
    if deploy_result and not deploy_result.get("passed", True) and deploy_result.get("target") != "skipped":
        return True

    return False


def _should_include_policy_source_context(state: GraphState) -> bool:
    """Include security policy context for Trivy/Checkov failures."""
    validation_results = state.get("validation_results", [])
    trivy_result = _get_latest_stage_result(validation_results, "trivy")
    checkov_result = _get_latest_stage_result(validation_results, "checkov")

    if trivy_result and not trivy_result.get("passed", True):
        return True
    if checkov_result and not checkov_result.get("passed", True):
        return True
    return False


def _build_remediation_history_context(remediation_history: list[RemediationHistory]) -> str:
    """Build a structured, read-only history block from past RemediationHistory entries.

    This replaces passing remediator_history conversation turns to the LLM.
    The history is injected as a compact document so both the Remediator and
    Engineer agents can avoid repeating failed strategies without the token
    overhead of verbatim conversation transcripts.
    """
    if not remediation_history:
        return ""

    lines: list[str] = [
        "## Prior Remediation Attempts",
        "The following fix strategies were already attempted. Do NOT repeat them.",
        "Use this history to choose a different approach if a strategy failed or was insufficient.",
        "",
    ]

    for entry in remediation_history:
        iteration = entry["iteration"]
        lines.append(f"### Attempt {iteration} ({entry['timestamp'][:10]})")

        # Summarise the errors that were present at the time
        if entry.get("formatted_errors"):
            # Truncate to avoid bloat — first 800 chars is sufficient for pattern recognition
            error_summary = entry["formatted_errors"][:800]
            if len(entry["formatted_errors"]) > 800:
                error_summary += "\n  ... (truncated)"
            lines.append(f"**Errors present:**\n{error_summary}")

        # Summarise the fix objectives suggested (first 600 chars)
        if entry.get("suggestion"):
            suggestion_summary = entry["suggestion"][:600]
            if len(entry["suggestion"]) > 600:
                suggestion_summary += "\n  ... (truncated)"
            lines.append(f"**Fix objectives suggested:**\n{suggestion_summary}")

        # Note which retrieval queries were used so the retriever can diversify
        if entry.get("retrieval_queries"):
            queries_str = ", ".join(f'"{q}"' for q in entry["retrieval_queries"])
            lines.append(f"**Retrieval queries used:** {queries_str}")

        lines.append("")

    return "\n".join(lines).strip()


def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    policy_source_context = ""
    cfn_graph_context = state.get("retriever_context", "")
    retrieval_queries = state.get("retriever_queries", [])

    if _should_include_policy_source_context(state):
        policy_source_context = _build_policy_source_context(state["validation_results"])

    if _should_include_remediation_context(state):
        print(
            f"[Remediator] CFN schema context from retriever: {len(cfn_graph_context)} chars, "
            f"{len(retrieval_queries)} retrieval queries used."
        )
    else:
        print("[Remediator] Context injection skipped for non-YAML/cfn-lint/deploy failures.")

    # Build structured history context from past RemediationHistory entries.
    # This replaces passing remediator_history conversation turns to the LLM —
    # the model gets a compact document instead of a verbatim transcript.
    remediation_history_context = _build_remediation_history_context(
        state.get("remediation_history", [])
    )

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=_build_validation_errors_text(state),
        policy_source_context=policy_source_context,
        cfn_graph_context=cfn_graph_context,
        remediation_history_context=remediation_history_context,
    )
    user_msg: Message = {"role": "user", "content": user_content}

    # Single-turn call: no conversation history passed — full context is in the prompt.
    # remediator_history is kept in state for debugging/recording only.
    client, model = _build_client()
    content, usage = _call_llm_with_history(client, model, system, [user_msg])
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state, agent="remediator", model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content, token_usage=usage,
    )

    formatted_errors = _build_validation_errors_text(state)
    new_history_entry: RemediationHistory = {
        "iteration": iteration,
        "errors": state["validation_results"],
        "formatted_errors": formatted_errors,
        "suggestion": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cfn_context": cfn_graph_context,
        "retrieval_queries": retrieval_queries,
    }

    print(f"[Remediator] Suggestions generated. Routing back to Engineer.")
    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "current_iteration": iteration + 1,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        # Keep history for recording/debugging — no longer used as LLM conversation context
        "remediator_history": append_and_cap(state["remediator_history"], user_msg, assistant_msg),
    }
