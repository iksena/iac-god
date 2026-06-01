# agents/remediator.py
from __future__ import annotations

from datetime import datetime, timezone

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.remediator_prompt import get_remediator_system_prompt, REMEDIATOR_USER
from tools.retriever_helpers import (
    format_cfn_lint_errors,
    format_tflint_errors,
    format_deploy_errors,
    extract_errors,
)
from tools.template_annotator import render_annotated_template
from tracking.recorder import ResearchRecorder


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


# ---------------------------------------------------------------------------
# Validation error section builder
# ---------------------------------------------------------------------------

def _build_validation_errors_text(state: GraphState) -> str:
    """Build the full validation error section for the remediator user prompt.

    Error formatting is iac_type-aware and symmetric across both pipelines:
      CloudFormation: cfn-lint errors  -> format_cfn_lint_errors()
      Terraform:      tflint errors    -> format_tflint_errors()  (Stage 1)
                      terraform-validate errors -> format_tflint_errors()  (Stage 2)
      Both pipelines: all other stages -> generic bullet formatter

    tflint and terraform-validate are collapsed into a single formatted block
    via format_tflint_errors() so the remediator prompt mirrors the cfn-lint
    section structure: one heading, two sub-sections (rule violations vs. type
    errors), rather than two separate top-level error blocks.
    """
    error_blocks: list[str] = []

    validation_results = state.get("validation_results", [])
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    # Collect Terraform structural stage errors upfront so they can be
    # rendered together in a single format_tflint_errors() block, mirroring
    # the single cfn-lint block on the CFN side.
    tflint_errors: list[str] = []
    tf_validate_errors: list[str] = []
    tf_structural_emitted = False

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
            error_blocks.append(
                f"### {stage.upper()} Errors\n{format_cfn_lint_errors(deduped)}"
            )

        elif stage == "tflint":
            # Accumulate — rendered together with terraform-validate below
            tflint_errors = deduped

        elif stage == "terraform-validate":
            # Accumulate — rendered together with tflint below
            tf_validate_errors = deduped

        else:
            # Trivy, checkov, yaml, and any future stages use generic bullets
            errors_text = "\n".join(f"  - {e}" for e in deduped)
            error_blocks.append(f"### {stage.upper()} Errors\n{errors_text}")

    # Emit the combined Terraform structural block once, after all stages
    # have been visited, so tflint and terraform-validate always appear
    # together regardless of iteration order.
    if tflint_errors or tf_validate_errors:
        error_blocks.append(
            f"### TERRAFORM STRUCTURAL Errors\n"
            f"{format_tflint_errors(tflint_errors, tf_validate_errors)}"
        )

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
# Agent entry point
# ---------------------------------------------------------------------------

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    # current_iteration was already incremented by validator_agent; use it
    # as-is for history labelling (reflects the iteration just completed).
    iteration = state["current_iteration"]
    iac_type = state.get("iac_type", "cloudformation")
    print(f"\n[Remediator] Analyzing errors (iteration {iteration}, iac_type={iac_type})...")

    system = get_remediator_system_prompt(iac_type).format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])),
    )

    knowledge_base_context = state.get("retriever_context", "")
    retrieval_queries = state.get("retriever_queries", [])

    print(
        f"[Remediator] Knowledge base context: {len(knowledge_base_context)} chars, "
        f"{len(retrieval_queries)} retrieval queries."
    )

    formatted_errors = _build_validation_errors_text(state)

    flat_errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )
    annotated_template = render_annotated_template(
        template_yaml=state.get("iac_template", ""),
        errors=flat_errors,
    )

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        annotated_template=annotated_template,
        validation_errors=formatted_errors,
        knowledge_base_context=knowledge_base_context,
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
        "retriever_context": knowledge_base_context,
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
