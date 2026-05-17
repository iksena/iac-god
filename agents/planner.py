# agents/planner.py
from state import GraphState, Message, compact_message_history, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.planner_prompt import PLANNER_SYSTEM, PLANNER_USER
from tracking.recorder import ResearchRecorder


def planner_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """CGO Stage 1: Objective Generation with conversation history."""
    iteration = state["current_iteration"]
    print(f"\n[Planner] Generating objectives (iteration {iteration})...")

    client, model = _build_client()

    user_msg: Message = {"role": "user", "content": PLANNER_USER.format(
        user_request=state["user_request"]
    )}

    messages = compact_message_history(list(state["planner_history"])) + [user_msg]

    content, usage = _call_llm_with_history(client, model, PLANNER_SYSTEM, messages)

    assistant_msg: Message = {"role": "assistant", "content": content}

    objectives = [
        line.strip().lstrip("0123456789. ").strip()
        for line in content.strip().split("\n")
        if line.strip() and line[0].isdigit()
    ]

    llm_record = recorder.record_llm_call(
        state=state,
        agent="planner",
        model=model,
        prompt=f"SYSTEM:\n{PLANNER_SYSTEM}\n\nUSER:\n{user_msg['content']}",
        response=content,
        token_usage=usage,
    )

    print(f"[Planner] Generated {len(objectives)} objectives.")
    return {
        "objectives": objectives,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "planner_history": append_and_cap(state["planner_history"], user_msg, assistant_msg),
    }
