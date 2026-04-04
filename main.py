# main.py
import uuid
import argparse
from graph import build_graph
from state import GraphState
from tracking.recorder import ResearchRecorder
from config import DEFAULT_CONFIG, DEFAULT_DEPLOY_CONFIG, LLMProvider, DeployTarget, DeployConfig
from functools import partial

def run_pipeline(
    user_request: str,
    max_iterations: int = 5,
    provider: str = "openrouter",
    model: str | None = None,
    deploy_target: str = "localstack",
    localstack_endpoint: str | None = None,
) -> GraphState:

    # Configure provider
    if provider == "claude":
        DEFAULT_CONFIG.provider = LLMProvider.CLAUDE
        DEFAULT_CONFIG.model = model or "claude-3-5-sonnet-20241022"
    else:
        DEFAULT_CONFIG.provider = LLMProvider.OPENROUTER
        DEFAULT_CONFIG.model = model or "arcee-ai/trinity-large-preview:free"
    
    # Configure deploy target
    deploy_config = DeployConfig(
        target=DeployTarget(deploy_target),
        localstack_endpoint=localstack_endpoint or DEFAULT_DEPLOY_CONFIG.localstack_endpoint,
    )

    run_id = str(uuid.uuid4())[:8]
    recorder = ResearchRecorder(run_id=run_id)
    graph = build_graph(recorder, deploy_config=deploy_config)

    # Initialize the Grounded Objectives Document
    initial_state: GraphState = {
        "user_request": user_request,
        "objectives": [],
        "cloudformation_template": "",
        "validation_results": [],
        "validation_passed": False,
        "remediation_history": [],
        "current_iteration": 1,
        "max_iterations": max_iterations,
        "llm_call_log": [],
        "final_template": None,
        "run_id": run_id,
        "deploy_validation_result": None,
    }

    print(f"\n{'='*60}")
    print(f"IaC Multi-Agent System | Run ID: {run_id}")
    print(f"Model: {DEFAULT_CONFIG.model}")
    print(f"Max iterations: {max_iterations}")
    print(f"Deploy target: {deploy_target.upper()}")
    print(f"{'='*60}")

    # Execute the graph
    final_state = graph.invoke(initial_state)

    # Persist final template
    final_state["final_template"] = final_state["cloudformation_template"]
    recorder.save_final_report(final_state)

    print(f"\n{'='*60}")
    print(f"Total LLM calls recorded: {len(final_state['llm_call_log'])}")
    print(f"Total iterations: {final_state['current_iteration']}")
    print(f"Final validation: {'PASSED' if final_state['validation_passed'] else 'FAILED'}")
    print(f"{'='*60}")

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IaC Multi-Agent System")
    parser.add_argument("--request", type=str, required=True,
                        help="Infrastructure request in natural language")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--provider", choices=["openrouter", "claude"],
                        default="openrouter")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--deploy-target",
        choices=["none", "localstack", "aws"],
        default="localstack",
        help="Deploy each iteration's template to validate actual deployability",
    )
    parser.add_argument(
        "--localstack-endpoint",
        type=str,
        default=None,
        help="Override LocalStack endpoint (default: http://localhost:4566)",
    )
    args = parser.parse_args()

    result = run_pipeline(
        user_request=args.request,
        max_iterations=args.max_iterations,
        provider=args.provider,
        model=args.model,
        deploy_target=args.deploy_target,
        localstack_endpoint=args.localstack_endpoint,
    )