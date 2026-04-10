# agents/planner.py
import json

from state import GraphState, Message, compact_message_history, append_and_cap
from config import DEFAULT_CONFIG, LLMProvider, build_openrouter_provider_preferences
from prompts.planner_prompt import PLANNER_SYSTEM, PLANNER_USER
from tracking.recorder import ResearchRecorder


def _to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _response_debug_blob(response: object) -> str:
    if response is None:
        return "response=None"

    try:
        if hasattr(response, "model_dump_json"):
            return response.model_dump_json(indent=2)
    except Exception:
        pass

    try:
        return json.dumps(response, default=str, indent=2)
    except Exception:
        return repr(response)

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
        request_kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": PLANNER_SYSTEM}] + messages,
            "temperature": DEFAULT_CONFIG.temperature,
            "max_tokens": DEFAULT_CONFIG.max_tokens,
        }
        provider_preferences = build_openrouter_provider_preferences(DEFAULT_CONFIG)
        if provider_preferences:
            request_kwargs["extra_body"] = {"provider": provider_preferences}
        if DEFAULT_CONFIG.reasoning_enabled:
            request_kwargs["extra_body"] = request_kwargs.get("extra_body", {})
            request_kwargs["extra_body"]["reasoning"] = { "enabled": True }

        response = client.chat.completions.create(
            **request_kwargs,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "OpenRouter returned no choices in planner call. "
                f"model={model} response={_response_debug_blob(response)}"
            )

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None

        if isinstance(content, list):
            text_parts = [
                part.get("text", "") if isinstance(part, dict) else ""
                for part in content
            ]
            content = "".join(text_parts).strip()

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenRouter returned empty/non-text content in planner call. "
                f"model={model} response={_response_debug_blob(response)}"
            )

        usage = {
            "prompt_tokens": _to_int(getattr(response.usage, "prompt_tokens", 0)),
            "completion_tokens": _to_int(getattr(response.usage, "completion_tokens", 0)),
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