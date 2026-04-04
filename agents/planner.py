# agents/planner.py
import uuid
from datetime import datetime, timezone
from state import GraphState
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
    """
    CGO Stage 1: Objective Generation.
    Generates grounded functional objectives from the user request.
    """
    print(f"\n[Planner] Generating objectives for iteration {state['current_iteration']}...")

    client, model = build_llm_client()
    prompt = PLANNER_USER.format(user_request=state["user_request"])

    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
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
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    # Parse objectives from numbered list
    objectives = [
        line.strip().lstrip("0123456789. ").strip()
        for line in content.strip().split("\n")
        if line.strip() and line[0].isdigit()
    ]

    # Record LLM call for research tracking
    llm_record = recorder.record_llm_call(
        state=state,
        agent="planner",
        model=model,
        prompt=f"SYSTEM:\n{PLANNER_SYSTEM}\n\nUSER:\n{prompt}",
        response=content,
        token_usage=usage,
    )

    print(f"[Planner] Generated {len(objectives)} objectives.")
    return {
        **state,
        "objectives": objectives,
        "llm_call_log": state["llm_call_log"] + [llm_record],
    }