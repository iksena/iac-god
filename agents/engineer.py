# agents/engineer.py
import json

from state import GraphState, Message, append_and_cap
from agents.remediator import _build_remediation_history_context
from config import DEFAULT_CONFIG, LLMProvider, build_openrouter_provider_preferences
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM, ENGINEER_USER_INITIAL,
    ENGINEER_USER_REMEDIATION,
)
from tracking.recorder import ResearchRecorder


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    system = ENGINEER_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    is_remediation = bool(state["remediation_history"])

    if not is_remediation:
        # Iteration 1: simple generation request — no history yet
        user_content = ENGINEER_USER_INITIAL
    else:
        # Iteration 2+: full context in prompt — no conversation history passed to LLM.
        # The current template, latest fix directive, and structured history are all
        # included here. No prior engineer_history turns are sent.
        latest = state["remediation_history"][-1]
        remediation_history_context = _build_remediation_history_context(
            state["remediation_history"][:-1]  # all entries except the latest (already shown above)
        )
        user_content = ENGINEER_USER_REMEDIATION.format(
            iteration=latest["iteration"],
            current_template=state["cloudformation_template"],
            error_context=latest["formatted_errors"],
            remediation_suggestion=latest["suggestion"],
            cfn_context=latest.get("cfn_context", ""),
            remediation_history_context=remediation_history_context,
        )

    user_msg: Message = {"role": "user", "content": user_content}

    # Single-turn call: no conversation history passed — full context is in the prompt.
    # engineer_history is kept in state for debugging/recording only.
    content, usage = _call_llm_with_history(client, model, system, [user_msg])
    template = _strip_yaml_fences(content)
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state,
        agent="engineer",
        model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content,
        token_usage=usage,
    )

    print(f"[Engineer] Template generated ({len(template.splitlines())} lines).")
    return {
        "cloudformation_template": template,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        # Keep history for recording/debugging — no longer used as LLM conversation context
        "engineer_history": append_and_cap(state["engineer_history"], user_msg, assistant_msg),
    }


def _strip_yaml_fences(text: str) -> str:
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _build_client():
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        from openai import OpenAI
        return OpenAI(
            api_key=DEFAULT_CONFIG.openrouter_api_key,
            base_url=DEFAULT_CONFIG.openrouter_base_url,
        ), DEFAULT_CONFIG.model
    else:
        import anthropic
        return anthropic.Anthropic(
            api_key=DEFAULT_CONFIG.anthropic_api_key
        ), DEFAULT_CONFIG.model


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


def _call_llm_with_history(client, model, system, messages):
    """Call LLM with a messages list. In the new stateless design, this is always
    a single [user_msg] — full context is embedded in the prompt, not in history."""
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        request_kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": DEFAULT_CONFIG.temperature,
            "max_tokens": DEFAULT_CONFIG.max_tokens,
        }
        provider_preferences = build_openrouter_provider_preferences(DEFAULT_CONFIG)
        if provider_preferences:
            request_kwargs["extra_body"] = {"provider": provider_preferences}
        if DEFAULT_CONFIG.reasoning_enabled:
            request_kwargs["extra_body"] = request_kwargs.get("extra_body", {})
            request_kwargs["extra_body"]["reasoning"] = { "enabled": True }

        r = client.chat.completions.create(**request_kwargs)
        choices = getattr(r, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "OpenRouter returned no choices in engineer call. "
                f"model={model} response={_response_debug_blob(r)}"
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
                "OpenRouter returned empty/non-text content in engineer call. "
                f"model={model} response={_response_debug_blob(r)}"
            )

        usage_obj = getattr(r, "usage", None)
        usage = {
            "prompt_tokens": _to_int(getattr(usage_obj, "prompt_tokens", 0)),
            "completion_tokens": _to_int(getattr(usage_obj, "completion_tokens", 0)),
        }
        return content, usage
    else:
        import anthropic as ant
        r = client.messages.create(
            model=model, system=system,
            messages=messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        return r.content[0].text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
        }
