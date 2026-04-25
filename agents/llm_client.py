"""Shared LLM client factory and call helper.

Extracted from agents/engineer.py so both engineer and remediator can
import these utilities without creating a circular dependency.
"""
from __future__ import annotations

import json

from config import DEFAULT_CONFIG, LLMProvider, build_openrouter_provider_preferences


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


def _call_llm_with_history(client, model: str, system: str, messages: list) -> tuple[str, dict]:
    """Call LLM with a messages list.

    In the stateless prompt design this is always a single [user_msg] —
    full context is embedded in the prompt, not in conversation history.
    """
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
            request_kwargs["extra_body"]["reasoning"] = {"enabled": True}

        r = client.chat.completions.create(**request_kwargs)
        choices = getattr(r, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "OpenRouter returned no choices in LLM call. "
                f"model={model} response={_response_debug_blob(r)}"
            )

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else ""
                for part in content
            ).strip()

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenRouter returned empty/non-text content in LLM call. "
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
            model=model,
            system=system,
            messages=messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        return r.content[0].text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
        }
