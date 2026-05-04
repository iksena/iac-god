# agents/engineer.py
from __future__ import annotations

from state import GraphState, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from config import SIMPLE_MODE_THRESHOLD
from state import any_stage_in_moderate_mode, classify_failing_stages
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM,
    ENGINEER_USER_INITIAL,
    ENGINEER_USER_SIMPLE_FIX,
    ENGINEER_USER_REMEDIATION,
)
from tools.retriever_helpers import format_cfn_lint_errors, format_deploy_errors
from tracking.recorder import ResearchRecorder


def _build_validation_errors_text(state: GraphState) -> str:
    """Render a compact validation error block for Path B (simple self-correction).

    Uses the same per-stage formatting as the remediator so the engineer
    receives identically structured error descriptions on both paths.
    """
    error_blocks: list[str] = []

    validation_results = state.get("validation_results", [])
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    for result in latest_by_stage.values():
        if result.get("passed", True):
            continue
        errors_raw = [str(e) for e in result.get("errors", []) if str(e).strip()]
        if not errors_raw:
            continue
        stage = result["stage"]
        if stage == "cfn-lint":
            errors_text = format_cfn_lint_errors(errors_raw)
        else:
            errors_text = "\n".join(f"  - {e}" for e in errors_raw)
        error_blocks.append(f"### {stage.upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if (
        deploy_result
        and not deploy_result.get("passed", True)
        and deploy_result.get("target") != "skipped"
    ):
        error_blocks.append(
            f"### DEPLOYABILITY Errors\n{format_deploy_errors(deploy_result)}"
        )

    return "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Generate or repair the CloudFormation template.

    Three paths:
      Path A — iteration 1 (no remediation history).
               Clean generation from objectives.

      Path B — simple self-correction.
               All currently-failing stage groups are below SIMPLE_MODE_THRESHOLD.
               The engineer receives ONLY the rich validation errors and fixes
               them directly, without remediator involvement.
               Conversation history holds the prior template as an assistant turn,
               so we do NOT resend the annotated template — that would duplicate
               the template in the context window.

      Path C — moderate remediation.
               At least one failing stage group has reached SIMPLE_MODE_THRESHOLD.
               The remediator has already produced RCA + Fix Objectives.
               The engineer receives the formatted errors + remediation suggestion.
               Schema context (cfn_context) is intentionally NOT forwarded — it
               was consumed by the remediator to produce the suggestion, and
               forwarding raw schema chunks adds noise without improving quality.
    """
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    system = ENGINEER_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    remediation_history = state.get("remediation_history", [])

    if not remediation_history:
        # ------------------------------------------------------------------ #
        # Path A — initial generation                                         #
        # ------------------------------------------------------------------ #
        print("[Engineer] Path A — initial generation.")
        user_content = ENGINEER_USER_INITIAL
        history_for_call: list[Message] = []

    else:
        failing_stages = classify_failing_stages(
            state.get("validation_results", []),
            state.get("deploy_validation_result"),
        )
        in_moderate = any_stage_in_moderate_mode(
            state.get("stage_error_counts", {}),
            failing_stages,
            SIMPLE_MODE_THRESHOLD,
        )

        if not in_moderate:
            # -------------------------------------------------------------- #
            # Path B — simple self-correction                                 #
            # -------------------------------------------------------------- #
            print("[Engineer] Path B — simple self-correction (below threshold).")
            validation_errors = _build_validation_errors_text(state)
            user_content = ENGINEER_USER_SIMPLE_FIX.format(
                iteration=iteration,
                validation_errors=validation_errors,
            )
            # Pass engineer_history so the LLM sees its own last template as
            # an assistant turn — avoids re-sending the full YAML in the prompt.
            history_for_call = list(state.get("engineer_history", []))

        else:
            # -------------------------------------------------------------- #
            # Path C — moderate remediation (remediator output available)     #
            # -------------------------------------------------------------- #
            print("[Engineer] Path C — moderate remediation (remediator suggestions).")
            latest = remediation_history[-1]
            user_content = ENGINEER_USER_REMEDIATION.format(
                iteration=latest["iteration"],
                formatted_errors=latest["formatted_errors"],
                remediation_suggestion=latest["suggestion"],
            )
            # Single-turn: full context is in the prompt — no history needed.
            history_for_call = []

    user_msg: Message = {"role": "user", "content": user_content}
    messages_for_call = history_for_call + [user_msg]

    content, usage = _call_llm_with_history(client, model, system, messages_for_call)
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
        # Keep history for Path B continuity and debugging.
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
