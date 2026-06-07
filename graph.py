# graph.py
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig, SIMPLE_MODE_THRESHOLD
from state import GraphState, any_stage_in_moderate_mode, classify_failing_stages
from agents.planner import planner_agent
from agents.engineer import engineer_agent
from agents.validator import validator_agent
from agents.retriever import retriever_agent
from agents.remediator import remediator_agent
# from agents.remediator import should_include_remediation_context
from tracking.recorder import ResearchRecorder


def route_after_validator(state: GraphState) -> str:
    """Route to the correct next node after validation.

    Decision tree:
      1. Passed → end.
      2. Max iterations reached → end.
      3. Any failing stage has reached SIMPLE_MODE_THRESHOLD iterations
         (moderate mode):
           - cfn-lint / deploy errors present → retriever (needs schema context)
           - security-only errors → remediator (policy context, no schema)
      4. All failing stages still below threshold (simple mode):
           → engineer_simple (direct self-correction, no retriever/remediator)
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

    moderate = any_stage_in_moderate_mode(
        stage_error_counts=state.get("stage_error_counts", {}),
        failing_stages=failing_stages,
        threshold=SIMPLE_MODE_THRESHOLD,
    )

    if moderate:
        print(
            f"[Router] Moderate mode — stage counts: "
            + ", ".join(
                f"{s}={state.get('stage_error_counts', {}).get(s, 0)}"
                for s in sorted(failing_stages)
            )
        )
        # if should_include_remediation_context(state):
        if True:  # for now, always include remediation context if in moderate mode
            return "retriever"
        return "remediator"

    print(
        f"[Router] Simple mode — stage counts: "
        + ", ".join(
            f"{s}={state.get('stage_error_counts', {}).get(s, 0)}"
            for s in sorted(failing_stages)
        )
    )
    return "engineer_simple"


def build_graph(
    recorder: ResearchRecorder,
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> StateGraph:
    graph = StateGraph(GraphState)

    # graph.add_node("planner",         partial(planner_agent,    recorder=recorder))
    graph.add_node("engineer",        partial(engineer_agent,   recorder=recorder))
    # engineer_simple is the same agent function — mode is detected from state
    graph.add_node("engineer_simple", partial(engineer_agent,   recorder=recorder))
    graph.add_node("validator",       partial(validator_agent,  recorder=recorder, deploy_config=deploy_config))
    graph.add_node("retriever",       partial(retriever_agent,  recorder=recorder))
    graph.add_node("remediator",      partial(remediator_agent, recorder=recorder))

    graph.set_entry_point("engineer")
    # graph.add_edge("planner",         "engineer")
    graph.add_edge("engineer",        "validator")
    graph.add_edge("engineer_simple", "validator")

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "end":             END,
            "retriever":       "retriever",
            "remediator":      "remediator",
            "engineer_simple": "engineer_simple",
        },
    )

    graph.add_edge("retriever",  "remediator")
    graph.add_edge("remediator", "engineer")

    return graph.compile(checkpointer=MemorySaver())
