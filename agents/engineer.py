# agents/engineer.py
from __future__ import annotations

from state import GraphState, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from agents.history_context import _build_remediation_history_context
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM,
    ENGINEER_USER_INITIAL,
    ENGINEER_USER_REMEDIATION,
)
from tools.template_annotator import render_annotated_template
from tracking.recorder import ResearchRecorder


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    system = ENGINEER_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    is_remediation = bool(state["remediation_history"])

    if not is_remediation:
        # Iteration 1: no history yet — plain single-turn generation.
        user_content = ENGINEER_USER_INITIAL
        history_to_pass: list[Message] = []
    else:
        # Iteration 2+: build full context prompt and pass rolling history
        # so the model can track its own prior generations and the fixes
        # it was asked to apply.
        latest = state["remediation_history"][-1]
        remediation_history_context = _build_remediation_history_context(
            state["remediation_history"][:-1]
        )

        # flat_errors was computed and stored by the remediator in the same
        # history entry — no need to re-derive it here.
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
        # Pass the rolling conversation history so the LLM has prior
        # generation context. The current user turn is appended below.
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
