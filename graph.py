# graph.py
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig
from state import GraphState
from agents.planner import planner_agent
from agents.engineer import engineer_agent
from agents.validator import validator_agent
from agents.retriever import retriever_agent
from agents.remediator import remediator_agent
from tracking.recorder import ResearchRecorder


def route_after_validator(state: GraphState) -> str:
    """
    After validation:
      - passed or max iterations → end
      - otherwise              → remediator (always; Remediator decides internally
                                 whether to invoke the build_retrieval_queries tool)
    """
    if state["validation_passed"]:
        print(f"\n\u2705 All validations passed at iteration {state['current_iteration']}!")
        return "end"
    if state["current_iteration"] >= state["max_iterations"]:
        print(f"\n\u26a0\ufe0f  Max iterations ({state['max_iterations']}) reached. Stopping.")
        return "end"
    return "remediator"


def route_after_remediator(state: GraphState) -> str:
    """
    After the Remediator runs:
      - awaiting_retriever=True  → the tool produced queries; run Retriever next
                                   so it can fetch schema context, then return
                                   to Remediator for the synthesis pass.
      - awaiting_retriever=False → Remediator already has context (or doesn't
                                   need it); go straight to Engineer.
    """
    if state.get("awaiting_retriever", False):
        return "retriever"
    return "engineer"


def build_graph(recorder: ResearchRecorder, deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG) -> StateGraph:
    graph = StateGraph(GraphState)

    # Bind recorder to each agent via partial application
    graph.add_node("planner",    partial(planner_agent,    recorder=recorder))
    graph.add_node("engineer",   partial(engineer_agent,   recorder=recorder))
    graph.add_node("validator",  partial(validator_agent,  recorder=recorder, deploy_config=deploy_config))
    graph.add_node("retriever",  partial(retriever_agent,  recorder=recorder))
    graph.add_node("remediator", partial(remediator_agent, recorder=recorder))

    # Define the flow
    graph.set_entry_point("planner")
    graph.add_edge("planner",  "engineer")
    graph.add_edge("engineer", "validator")

    # Validator → Remediator (always on failure) or END
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "end":       END,
            "remediator": "remediator",
        },
    )

    # Remediator → Retriever (tool produced queries) or → Engineer (synthesis done)
    graph.add_conditional_edges(
        "remediator",
        route_after_remediator,
        {
            "retriever": "retriever",
            "engineer":  "engineer",
        },
    )

    # Retriever always returns to Remediator for the synthesis pass
    graph.add_edge("retriever", "remediator")

    # MemorySaver stores per-thread execution history for short-term memory.
    return graph.compile(checkpointer=MemorySaver())
