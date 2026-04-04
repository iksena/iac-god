# agents/remediator.py
from datetime import datetime, timezone
from state import GraphState, RemediationHistory
from config import DEFAULT_CONFIG, LLMProvider
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tracking.recorder import ResearchRecorder
from agents.engineer import _build_client, _call_llm

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
            f"Iteration {h['iteration']}: {h['suggestion'][:200]}..."
            for h in state["remediation_history"]
        )

    objectives_text = "\n".join(
        f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])
    )
    prompt = REMEDIATOR_USER.format(
        objectives=objectives_text,
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=validation_errors_text,
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