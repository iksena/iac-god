# main.py
from datetime import datetime
import os
import uuid
import argparse
import traceback
from graph import build_graph
from state import GraphState
from tracking.recorder import ResearchRecorder
from config import DEFAULT_CONFIG, DEFAULT_DEPLOY_CONFIG, LLMProvider, DeployTarget, DeployConfig


def _parse_csv_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


def run_pipeline(
    user_request: str,
    max_iterations: int = 5,
    provider: str = "openrouter",
    model: str | None = None,
    deploy_target: str = "localstack",
    localstack_endpoint: str | None = None,
    openrouter_provider_only: str | None = None,
    openrouter_min_quantization: str | None = None,
) -> GraphState:

    # ------------------------------------------------------------------
    # Configure LLM provider
    # ------------------------------------------------------------------
    if provider == "claude":
        DEFAULT_CONFIG.provider = LLMProvider.CLAUDE
        DEFAULT_CONFIG.model = model or "claude-3-5-sonnet-20241022"

    elif provider == "openai":
        DEFAULT_CONFIG.provider = LLMProvider.OPENAI
        # Fallback order: explicit --model arg → OPENAI_MODEL env var → o3-mini
        DEFAULT_CONFIG.model = model or os.getenv("OPENAI_MODEL", "o3-mini")
        # api_key and base_url are already populated from .env by LLMConfig,
        # but allow callers to override them via the existing DEFAULT_CONFIG fields.
        if not DEFAULT_CONFIG.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set the environment variable directly."
            )

    else:  # openrouter (default)
        DEFAULT_CONFIG.provider = LLMProvider.OPENROUTER
        DEFAULT_CONFIG.model = model or "arcee-ai/trinity-large-preview:free"

        if openrouter_provider_only is not None:
            DEFAULT_CONFIG.openrouter_provider_only = _parse_csv_arg(openrouter_provider_only)
        if openrouter_min_quantization is not None:
            DEFAULT_CONFIG.openrouter_min_quantization = openrouter_min_quantization.strip().lower()

    # ------------------------------------------------------------------
    # Configure deploy target
    # ------------------------------------------------------------------
    deploy_config = DeployConfig(
        target=DeployTarget(deploy_target),
        localstack_endpoint=localstack_endpoint or DEFAULT_DEPLOY_CONFIG.localstack_endpoint,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = ts + "_" + str(uuid.uuid4())[:8]
    recorder = ResearchRecorder(run_id=run_id)
    graph = build_graph(recorder, deploy_config=deploy_config)

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
        "planner_history":    [],
        "engineer_history":   [],
        "remediator_history": [],
        "retriever_history":  [],
        "retriever_context": "",
        "retriever_queries": [],
        "final_template": None,
        "run_id": run_id,
        "deploy_validation_result": None,
    }

    print(f"\n{'='*60}")
    print(f"IaC Multi-Agent System | Run ID: {run_id}")
    print(f"Provider: {DEFAULT_CONFIG.provider.value}")
    print(f"Model: {DEFAULT_CONFIG.model}")
    print(f"Max iterations: {max_iterations}")
    print(f"Deploy target: {deploy_target.upper()}")
    print(f"{'='*60}")

    try:
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": run_id}},
        )
        if final_state is None:
            raise RuntimeError("Pipeline graph returned None final_state")
    except Exception:
        print("\n[Pipeline] Unhandled exception while running graph:")
        print(traceback.format_exc())
        raise

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
    parser.add_argument(
        "--provider",
        choices=["openrouter", "claude", "openai"],
        default="openrouter",
        help="LLM provider to use. 'openai' reads OPENAI_API_KEY / OPENAI_MODEL from .env",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Examples: "
            "openrouter → 'x-ai/grok-4.1-fast', "
            "claude → 'claude-3-5-sonnet-20241022', "
            "openai → 'o3-mini', 'o3', 'o4-mini', 'gpt-4o', 'codex-mini-latest'"
        ),
    )
    parser.add_argument(
        "--openrouter-provider-only",
        type=str,
        default=None,
        help="Comma-separated OpenRouter provider slugs to allow (maps to provider.only)",
    )
    parser.add_argument(
        "--openrouter-min-quantization",
        type=str,
        choices=["int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"],
        default=None,
        help="Minimum quantization level for OpenRouter provider filtering",
    )
    parser.add_argument(
        "--deploy-target",
        choices=["none", "localstack", "aws"],
        default="localstack",
    )
    parser.add_argument(
        "--localstack-endpoint",
        type=str,
        default=None,
        help="Override LocalStack endpoint (default: http://localhost:4566)",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            user_request=args.request,
            max_iterations=args.max_iterations,
            provider=args.provider,
            model=args.model,
            deploy_target=args.deploy_target,
            localstack_endpoint=args.localstack_endpoint,
            openrouter_provider_only=args.openrouter_provider_only,
            openrouter_min_quantization=args.openrouter_min_quantization,
        )
    except Exception:
        print("\n[Main] Pipeline execution failed:")
        print(traceback.format_exc())
        raise SystemExit(1)
