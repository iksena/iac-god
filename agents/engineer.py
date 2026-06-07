# agents/engineer.py
from __future__ import annotations

from state import GraphState, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.engineer_prompt import (
    get_engineer_system_prompt,
    get_engineer_user_initial,
    get_engineer_user_simple_fix,
    get_engineer_user_remediation,
    get_engineer_user_no_remediator,
)
from tools.retriever_helpers import (
    extract_errors,
    format_cfn_lint_errors,
    format_deploy_errors,
)
from tracking.recorder import ResearchRecorder

# ---------------------------------------------------------------------------
# Ablation tuning
# ---------------------------------------------------------------------------

# Maximum characters of retriever_context injected into the no-remediator
# prompt.  Keeping this below the full context size (~20-30k chars) ensures
# the ablation tests the Engineer's *diagnostic* capability rather than its
# ability to process an arbitrarily large schema dump.
#
# If the Engineer fails even with 8k of directly relevant schema, the
# Remediator's *synthesis* value is proven — not just its token-reduction value.
# Raise toward 16_000 to give the Engineer more rope; lower to 4_000 for a
# stricter test.  Override at runtime: ABLATION_CONTEXT_LIMIT=12000 python main.py
import os as _os
_ABLATION_CONTEXT_LIMIT: int = int(_os.getenv("ABLATION_CONTEXT_LIMIT", "8000"))


def _build_simple_fix_errors(state: GraphState) -> str:
    """Format validation errors for Path B (simple self-correction).

    Produces the same rich format used by the remediator:
      [RuleId] line N | Resource: LogicalId | message | description | See: <url>

    The engineer's conversation history already contains the template, so only
    the error text is needed in the user turn.
    """
    validation_results = state.get("validation_results", [])
    deploy_result = state.get("deploy_validation_result")

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

    return "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    iac_type = state.get("iac_type", "cloudformation")
    lang_label = "HCL" if iac_type == "terraform" else "CFN"
    print(f"\n[Engineer] Generating {lang_label} template (iteration {iteration})...")

    client, model = _build_client()

    system = get_engineer_system_prompt(iac_type).format(
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
        # Path A — Iteration 1: no history, no errors; clean generation.
        # No conversation history to pass yet.
        # ----------------------------------------------------------------
        print("[Engineer] Path A: initial generation.")
        user_content = get_engineer_user_initial(iac_type)
        history_to_pass: list[Message] = []

    elif has_validation_errors and not has_remediation_history:
        # ----------------------------------------------------------------
        # Path B — Simple mode: validator failed but remediator has not run.
        # User turn contains ONLY the rich validation errors.
        # The template is already in engineer_history (conv context).
        # ----------------------------------------------------------------
        print("[Engineer] Path B: simple self-correction from validation errors.")
        validation_errors = _build_simple_fix_errors(state)
        user_content = get_engineer_user_simple_fix(iac_type).format(
            iteration=iteration,
            validation_errors=validation_errors,
        )
        history_to_pass = state.get("engineer_history", [])

    else:
        # ----------------------------------------------------------------
        # Path C (ABLATION: no-remediator) — Engineer ingests errors +
        # RAG context directly. No Remediator RCA is available.
        #
        # Three clearly labelled sections are passed to the LLM:
        #   1. Validation Errors  — live errors from current validator output,
        #                           never stale Remediator history.
        #   2. Schema & Remediation Reference — raw retriever_context, capped
        #                           at _ABLATION_CONTEXT_LIMIT chars so the
        #                           test isolates diagnostic capability from
        #                           context-window size effects.
        #   3. Output instruction — produce a corrected template only.
        # ----------------------------------------------------------------
        print("[Engineer] Path C (ABLATION): Ingesting errors + RAG context directly.")

        # Always read live errors — never rely on remediation_history["formatted_errors"]
        # which is written by the Remediator (absent in this ablation branch).
        validation_errors = _build_simple_fix_errors(state)

        retriever_context = state.get("retriever_context", "") or "No schema context available."
        if len(retriever_context) > _ABLATION_CONTEXT_LIMIT:
            retriever_context = (
                retriever_context[:_ABLATION_CONTEXT_LIMIT]
                + f"\n\n... [context truncated at {_ABLATION_CONTEXT_LIMIT} chars for ablation] ..."
            )
            print(
                f"[Engineer] RAG context truncated to {_ABLATION_CONTEXT_LIMIT} chars "
                f"(set ABLATION_CONTEXT_LIMIT env var to adjust)."
            )

        user_content = get_engineer_user_no_remediator(iac_type).format(
            iteration=iteration,
            validation_errors=validation_errors,
            retriever_context=retriever_context,
        )
        history_to_pass = state.get("engineer_history", [])

    user_msg: Message = {"role": "user", "content": user_content}

    content, usage = _call_llm_with_history(
        client, model, system,
        history_to_pass + [user_msg],
    )
    template = _strip_code_fences(content)
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
        "iac_template": template,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "engineer_history": append_and_cap(
            state.get("engineer_history", []), user_msg, assistant_msg
        ),
    }


def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```yaml, ```hcl, ``` etc.)."""
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
