# agents/planner.py
from state import GraphState, Message, compact_message_history, append_and_cap
from config import DEFAULT_CONFIG, LLMProvider
from prompts.planner_prompt import PLANNER_SYSTEM, PLANNER_USER
from tracking.recorder import ResearchRecorder

def build_llm_client(config=DEFAULT_CONFIG):
    if config.provider == LLMProvider.OPENROUTER:
        from openai import OpenAI
        return OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        ), config.model
    else:
        import anthropic
        return anthropic.Anthropic(api_key=config.anthropic_api_key), config.model


def planner_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """CGO Stage 1: Objective Generation with conversation history."""
    iteration = state["current_iteration"]
    print(f"\n[Planner] Generating objectives (iteration {iteration})...")

    client, model = build_llm_client()
    user_msg: Message = {"role": "user", "content": PLANNER_USER.format(
        user_request=state["user_request"]
    )}

    messages = compact_message_history(list(state["planner_history"])) + [user_msg]

    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PLANNER_SYSTEM}] + messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    else:
        import anthropic
        response = client.messages.create(
            model=model,
            system=PLANNER_SYSTEM,
            messages=messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

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