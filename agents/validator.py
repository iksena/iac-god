# agents/validator.py
from state import GraphState
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig

def validator_agent(
    state: GraphState,
    recorder: ResearchRecorder,
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> GraphState:
    """
    Runs static validation pipeline (YAML → cfn-lint → Trivy) then optionally
    a live deployability check against LocalStack or AWS.
    """
    iteration = state["current_iteration"]
    print(f"\n[Validator] Running validation pipeline (iteration {iteration})...")

    results, all_passed, deploy_result = run_all_validators(
        state["cloudformation_template"],
        deploy_config=deploy_config,
    )

    for r in results:
        status = "✅ PASS" if r["passed"] else f"❌ FAIL ({len(r['errors'])} errors)"
        print(f"  [{r['stage']:10s}] {status}")

    if deploy_result["target"] != "skipped":
        deploy_status = "✅ PASS" if deploy_result["passed"] else f"❌ FAIL: {deploy_result['error_message']}"
        print(f"  [{'deploy':10s}] {deploy_status} ({deploy_result['duration_seconds']}s)")

    updated_state = {
        "validation_results": results,
        "validation_passed": all_passed,
        "deploy_validation_result": deploy_result,
    }

    recorder.save_iteration_snapshot({**state, **updated_state})
    return updated_state