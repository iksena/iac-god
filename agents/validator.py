# agents/validator.py
from state import GraphState
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder

def validator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """
    Runs all 4 validation stages: YAML → cfn-lint → Checkov → Trivy.
    All errors are stored in the shared state for Remediator consumption.
    """
    iteration = state["current_iteration"]
    print(f"\n[Validator] Running validation pipeline (iteration {iteration})...")

    results, all_passed = run_all_validators(state["cloudformation_template"])

    for r in results:
        status = "✅ PASS" if r["passed"] else f"❌ FAIL ({len(r['errors'])} errors)"
        print(f"  [{r['stage']:10s}] {status}")

    # Save iteration snapshot for research
    recorder.save_iteration_snapshot({
        **state,
        "validation_results": results,
        "validation_passed": all_passed,
    })

    return {
        **state,
        "validation_results": results,
        "validation_passed": all_passed,
    }