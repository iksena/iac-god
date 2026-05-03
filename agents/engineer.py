# agents/engineer.py
from __future__ import annotations

from state import GraphState, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from agents.history_context import _build_remediation_history_context
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM,
    ENGINEER_USER_INITIAL,
    ENGINEER_USER_SIMPLE_FIX,
    ENGINEER_USER_REMEDIATION,
)
from tools.template_annotator import render_annotated_template
from tools.retriever_helpers import (
    extract_errors,
    format_cfn_lint_errors,
    format_deploy_errors,
    get_latest_stage_result,
)
from tracking.recorder import ResearchRecorder


def _build_simple_fix_errors(state: GraphState) -> tuple[str, list[str]]:
    """Build the formatted errors block and flat error list for simple-fix mode.

    Produces the same rich format as the remediator prompt so the engineer
    sees [RuleId] line N | Resource: X | message | description for each
    cfn-lint finding, and a structured block for deploy failures.

    Returns (formatted_errors_text, flat_errors_list).
    """
    validation_results = state.get("validation_results", [])
    deploy_result = state.get("deploy_validation_result")

    flat_errors = extract_errors(validation_results, deploy_result)

    error_blocks: list[str] = []

    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    for result in latest_by_stage.values():
        if result.get("passed"):
            continue
        errors = [str(e) for e in result.get("errors", []) if str(e).strip()]
        if not errors:
            continue
        stage = result["stage"]
        if stage == "cfn-lint":
            errors_text = format_cfn_lint_errors(errors)
        else:
            errors_text = "\n".join(f"  - {e}" for e in errors)
        error_blocks.append(f"### {stage.upper()} Errors\n{errors_text}")

    if (
        deploy_result
        and not deploy_result.get("passed")
        and deploy_result.get("target") != "skipped"
    ):
        error_blocks.append(
            f"### DEPLOYABILITY Errors\n{format_deploy_errors(deploy_result)}"
        )

    formatted = "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."
    return formatted, flat_errors


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    system = ENGINEER_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    has_remediation_history = bool(state.get("remediation_history"))
    has_validation_errors = (
        not state.get("validation_passed", True)
        and bool(state.get("validation_results"))
    )

    if not has_remediation_history and not has_validation_errors:
        # ----------------------------------------------------------------
        # Path A — Iteration 1: clean generation, no prior context at all.
        # ----------------------------------------------------------------
        print("[Engineer] Path A: initial generation.")
        user_content = ENGINEER_USER_INITIAL
        history_to_pass: list[Message] = []

    elif has_validation_errors and not has_remediation_history:
        # ----------------------------------------------------------------
        # Path B — Simple mode: validator failed but remediator has not run
        # yet for this error cycle.  The engineer self-corrects using the
        # rich cfn-lint error format (rule ID + line + resource + description)
        # and the annotated template.  No schema context needed.
        # ----------------------------------------------------------------
        print("[Engineer] Path B: simple self-correction from validation errors.")
        formatted_errors, flat_errors = _build_simple_fix_errors(state)
        annotated_template = render_annotated_template(
            template_yaml=state.get("cloudformation_template", ""),
            errors=flat_errors,
        )
        user_content = ENGINEER_USER_SIMPLE_FIX.format(
            iteration=iteration,
            annotated_template=annotated_template,
            validation_errors=formatted_errors,
        )
        history_to_pass = state.get("engineer_history", [])

    else:
        # ----------------------------------------------------------------
        # Path C — Moderate mode: remediator has produced a suggestion;
        # use full context prompt with schema context.
        # ----------------------------------------------------------------
        print("[Engineer] Path C: moderate remediation with schema context.")
        latest = state["remediation_history"][-1]
        remediation_history_context = _build_remediation_history_context(
            state["remediation_history"][:-1]
        )
        annotated_template = render_annotated_template(
            template_yaml=state.get("cloudformation_template", ""),
            errors=latest.get("flat_errors", []),
        )
        user_content = ENGINEER_USER_REMEDIATION.format(
            iteration=latest["iteration"],
            annotated_template=annotated_template,
            error_context=latest["formatted_errors"],
            remediation_suggestion=latest["suggestion"],
            cfn_context=latest.get("cfn_context", ""),
            remediation_history_context=remediation_history_context,
        )
        history_to_pass = state.get("engineer_history", [])

    user_msg: Message = {"role": "user", "content": user_content}

    content, usage = _call_llm_with_history(
        client, model, system,
        history_to_pass + [user_msg],
    )
    template = _strip_yaml_fences(content)
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state,
        agent="engineer",
        model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content,
        token_usage=usage,
    )

    print(f"[Engineer] Template generated ({len(template.splitlines())} lines).")
    return {
        "cloudformation_template": template,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "engineer_history": append_and_cap(
            state.get("engineer_history", []), user_msg, assistant_msg
        ),
    }


def _strip_yaml_fences(text: str) -> str:
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
