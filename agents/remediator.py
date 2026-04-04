# agents/remediator.py
from datetime import datetime, timezone
import json
from state import GraphState, RemediationHistory
from config import DEFAULT_CONFIG, LLMProvider
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tracking.recorder import ResearchRecorder
from agents.engineer import _build_client, _call_llm
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

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """
    Analyzes validation errors against the Grounded Objectives Document
    and produces remediation suggestions for the Engineer's next iteration.
    """
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors and generating fix suggestions (iteration {iteration})...")

    # Format all validation errors for the prompt
    error_blocks = []
    for r in state["validation_results"]:
        if not r["passed"]:
            errors_text = "\n".join(f"  - {e}" for e in r["errors"])
            error_blocks.append(f"### {r['stage'].upper()} Errors\n{errors_text}")
    validation_errors_text = "\n\n".join(error_blocks)

    # Format remediation history
    history_text = "No previous remediations." if not state["remediation_history"] else \
        "\n".join(
            # f"Iteration {h['iteration']}: {h['suggestion'][:200]}..."
            f"Iteration {h['iteration']}: {h['suggestion']}"
            for h in state["remediation_history"]
        )

    objectives_text = "\n".join(
        f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])
    )
    policy_source_context = _build_policy_source_context(state["validation_results"])
    prompt = REMEDIATOR_USER.format(
        objectives=objectives_text,
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=validation_errors_text,
        policy_source_context=policy_source_context,
        remediation_history=history_text,
    )

    client, model = _build_client()
    content, usage = _call_llm(client, model, REMEDIATOR_SYSTEM, prompt)

    llm_record = recorder.record_llm_call(
        state=state,
        agent="remediator",
        model=model,
        prompt=f"SYSTEM:\n{REMEDIATOR_SYSTEM}\n\nUSER:\n{prompt}",
        response=content,
        token_usage=usage,
    )

    # Append to remediation history (immutable log)
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
        "current_iteration": iteration + 1,   # Increment iteration counter
        "llm_call_log": state["llm_call_log"] + [llm_record],
    }