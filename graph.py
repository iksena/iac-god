# graph.py
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig, SIMPLE_MODE_THRESHOLD
from state import GraphState, classify_failing_stages, any_stage_in_moderate_mode
from agents.planner import planner_agent
from agents.engineer import engineer_agent
from agents.validator import validator_agent
from agents.remediator import remediator_agent
from tracking.recorder import ResearchRecorder


def route_after_validator(state: GraphState) -> str:
    """
    Routing logic after every validator run.

    Decision tree:
      1. Passed or max iterations reached → END.
      2. At least one stage group has reached SIMPLE_MODE_THRESHOLD failures
         → 'remediator' (moderate path: Remediator calls RAG tool, produces
            RCA + Fix Objectives, then Engineer applies them).
      3. All failing stages are still below the threshold
         → 'engineer' (simple path: Engineer self-corrects directly from
            the rich validation error block — no remediator round-trip).

    SIMPLE_MODE_THRESHOLD = 0 (default) means every failure goes to the
    remediator immediately, preserving the original behaviour.
    Set it to 1 or higher to give the engineer N self-correction attempts
    before escalating to the remediator.
    """
    if state["validation_passed"]:
        print(f"\n\u2705 All validations passed at iteration {state['current_iteration']}!")
        return "end"

    if state["current_iteration"] >= state["max_iterations"]:
        print(f"\n\u26a0\ufe0f  Max iterations ({state['max_iterations']}) reached. Stopping.")
        return "end"

    failing_stages = classify_failing_stages(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    if any_stage_in_moderate_mode(
        state.get("stage_error_counts", {}),
        failing_stages,
        SIMPLE_MODE_THRESHOLD,
    ):
        print(
            f"[Router] Moderate mode — routing to remediator "
            f"(threshold={SIMPLE_MODE_THRESHOLD}, failing={sorted(failing_stages)})."
        )
        return "remediator"

    print(
        f"[Router] Simple mode — routing direct to engineer "
        f"(threshold={SIMPLE_MODE_THRESHOLD}, failing={sorted(failing_stages)})."
    )
    return "engineer"


def build_graph(
    recorder: ResearchRecorder,
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> StateGraph:
    graph = StateGraph(GraphState)

    # Bind recorder (and deploy_config for validator) via partial application.
    graph.add_node("planner",    partial(planner_agent,    recorder=recorder))
    graph.add_node("engineer",   partial(engineer_agent,   recorder=recorder))
    graph.add_node("validator",  partial(validator_agent,  recorder=recorder, deploy_config=deploy_config))
    graph.add_node("remediator", partial(remediator_agent, recorder=recorder))

    # Fixed edges.
    graph.set_entry_point("planner")
    graph.add_edge("planner",    "engineer")
    graph.add_edge("engineer",   "validator")
    graph.add_edge("remediator", "engineer")

    # Validator → END | remediator | engineer (simple self-correction).
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "end":        END,
            "remediator": "remediator",
            "engineer":   "engineer",
        },
    )

    return graph.compile(checkpointer=MemorySaver())
