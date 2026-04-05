# agents/remediator.py
from datetime import datetime, timezone
import json
from state import GraphState, RemediationHistory, Message, compact_message_history
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tracking.recorder import ResearchRecorder
from agents.engineer import _build_client, _call_llm_with_history
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context


def _extract_checkov_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for result in validation_results:
        if result.get("stage") != "checkov":
            continue
        raw = result.get("raw_output", "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        failed = data.get("results", {}).get("failed_checks", [])
        for item in failed:
            check_id = str(item.get("check_id") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})
    return findings


def _extract_trivy_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for result in validation_results:
        if result.get("stage") != "trivy":
            continue
        raw = result.get("raw_output", "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for r in data.get("Results", []):
            for misconfig in r.get("Misconfigurations", []):
                check_id = str(misconfig.get("ID") or "").strip()
                if check_id:
                    findings.append({"check_id": check_id})
    return findings


def _build_policy_source_context(validation_results: list[dict]) -> str:
    checkov_context = get_checkov_policy_context(
        _extract_checkov_findings(validation_results)
    )
    trivy_context = get_trivy_policy_context(
        _extract_trivy_findings(validation_results)
    )

    sections: list[str] = []
    if checkov_context:
        sections.append(f"### Checkov Policy Source\n{checkov_context}")
    if trivy_context:
        sections.append(f"### Trivy Policy Source\n{trivy_context}")
    if not sections:
        return "No policy source context found for current findings."
    return "\n\n".join(sections)


def _build_validation_errors_text(state: GraphState) -> str:
    error_blocks: list[str] = []

    for result in state["validation_results"]:
        if result["passed"]:
            continue
        errors_text = "\n".join(f"  - {error}" for error in result["errors"])
        error_blocks.append(f"### {result['stage'].upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if deploy_result and not deploy_result["passed"] and deploy_result["target"] != "skipped":
        deploy_errors: list[str] = []

        if deploy_result.get("error_message"):
            deploy_errors.append(str(deploy_result["error_message"]))

        for failed in deploy_result.get("failed_resources", []):
            resource = str(failed.get("resource") or "unknown")
            reason = str(failed.get("reason") or "no reason provided")
            deploy_errors.append(f"{resource}: {reason}")

        if not deploy_errors:
            deploy_errors.append("Deployment failed with no structured error details.")

        errors_text = "\n".join(f"  - {error}" for error in deploy_errors)
        error_blocks.append(
            f"### DEPLOYABILITY Errors ({deploy_result['target'].upper()})\n{errors_text}"
        )

    if not error_blocks:
        return "No validation errors reported."
    return "\n\n".join(error_blocks)

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    # Objectives injected ONCE into system prompt
    system = REMEDIATOR_SYSTEM.format(
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    # Only NEW information goes in user turn:
    # - current template (not in remediator's own history — cross-agent boundary)
    # - current validation errors (brand new this iteration)
    # - policy context (brand new this iteration)
    validation_errors_text = _build_validation_errors_text(state)
    policy_source_context = _build_policy_source_context(state["validation_results"])

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=validation_errors_text,
        policy_source_context=policy_source_context,
    )
    user_msg: Message = {"role": "user", "content": user_content}

    # Full accumulated history + new user turn
    messages = compact_message_history(state["remediator_history"]) + [user_msg]

    client, model = _build_client()
    content, usage = _call_llm_with_history(client, model, system, messages)
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state, agent="remediator", model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content, token_usage=usage,
    )

    new_history_entry: RemediationHistory = {
        "iteration": iteration,
        "errors": state["validation_results"],
        "suggestion": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[Remediator] Suggestions generated. Routing back to Engineer.")
    return {
        **state,
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "current_iteration": iteration + 1,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "remediator_history": [user_msg, assistant_msg],
    }