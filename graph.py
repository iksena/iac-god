# graph.py
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from config import DEFAULT_DEPLOY_CONFIG, DeployConfig
from state import GraphState
from agents.planner import planner_agent
from agents.engineer import engineer_agent
from agents.validator import validator_agent
from agents.remediator import remediator_agent
from tracking.recorder import ResearchRecorder


def route_after_validator(state: GraphState) -> str:
    """
    After validation:
      - passed or max iterations → end
      - otherwise               → remediator

    The Remediator now calls `retrieve_schema_context` as a tool directly,
    so there is no separate retriever node in the graph.
    """
    if state["validation_passed"]:
        print(f"\n\u2705 All validations passed at iteration {state['current_iteration']}!")
        return "end"
    if state["current_iteration"] >= state["max_iterations"]:
        print(f"\n\u26a0\ufe0f  Max iterations ({state['max_iterations']}) reached. Stopping.")
        return "end"
    return "remediator"


def build_graph(recorder: ResearchRecorder, deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG) -> StateGraph:
    graph = StateGraph(GraphState)

    # Bind recorder to each agent via partial application
    graph.add_node("planner",    partial(planner_agent,    recorder=recorder))
    graph.add_node("engineer",   partial(engineer_agent,   recorder=recorder))
    graph.add_node("validator",  partial(validator_agent,  recorder=recorder, deploy_config=deploy_config))
    graph.add_node("remediator", partial(remediator_agent, recorder=recorder))

    # Define the flow
    graph.set_entry_point("planner")
    graph.add_edge("planner",    "engineer")
    graph.add_edge("engineer",   "validator")
    graph.add_edge("remediator", "engineer")

    # Validator → Remediator (on failure) or END
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "end":        END,
            "remediator": "remediator",
        },
    )

    return graph.compile(checkpointer=MemorySaver())
