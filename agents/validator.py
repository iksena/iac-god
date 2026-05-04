# agents/validator.py
from state import GraphState, classify_failing_stages
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig


def validator_agent(
    state: GraphState,
    recorder: ResearchRecorder,
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> GraphState:
    """
    Runs the static validation pipeline (YAML -> cfn-lint) then optionally a
    live deployability check against LocalStack or AWS.

    Responsibilities:
      - Run all validators and collect results.
      - Increment stage_error_counts for every failing stage group.
      - Increment current_iteration unconditionally so the counter advances
        on BOTH the simple path (validator -> engineer) and the
        moderate path (validator -> remediator -> engineer).
    """
    iteration = state["current_iteration"]
    print(f"\n[Validator] Running validation pipeline (iteration {iteration})...")

    results, all_passed, deploy_result = run_all_validators(
        state["cloudformation_template"],
        deploy_config=deploy_config,
    )

    for r in results:
        status = "\u2705 PASS" if r["passed"] else f"\u274c FAIL ({len(r['errors'])} errors)"
        print(f"  [{r['stage']:10s}] {status}")

    if deploy_result["target"] != "skipped":
        deploy_status = (
            "\u2705 PASS" if deploy_result["passed"]
            else f"\u274c FAIL: {deploy_result['error_message']}"
        )
        print(f"  [{'deploy':10s}] {deploy_status} ({deploy_result['duration_seconds']}s)")

    # ------------------------------------------------------------------
    # Increment per-stage error counters.
    # ------------------------------------------------------------------
    failing_stages = classify_failing_stages(results, deploy_result)
    prev_counts: dict[str, int] = dict(state.get("stage_error_counts") or {})
    updated_counts = {
        **prev_counts,
        **{stage: prev_counts.get(stage, 0) + 1 for stage in failing_stages},
    }

    if failing_stages:
        counts_str = ", ".join(
            f"{s}={updated_counts[s]}" for s in sorted(failing_stages)
        )
        print(f"  [stage counts ] {counts_str}")

    # ------------------------------------------------------------------
    # Advance iteration counter here so it is always incremented exactly
    # once per engineer -> validator cycle, regardless of which path the
    # router chooses next (simple or moderate).
    # ------------------------------------------------------------------
    next_iteration = iteration + 1
    print(f"  [iteration    ] {iteration} -> {next_iteration}")

    updated_state = {
        "validation_results": results,
        "validation_passed": all_passed,
        "deploy_validation_result": deploy_result,
        "stage_error_counts": updated_counts,
        "current_iteration": next_iteration,
    }

    recorder.save_iteration_snapshot({**state, **updated_state})
    return updated_state
