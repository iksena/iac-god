"""Shared LLM client factory and call helper.

Extracted from agents/engineer.py so both engineer and remediator can
import these utilities without creating a circular dependency.

Supported providers
-------------------
LLMProvider.OPENROUTER  - OpenRouter proxy (any model via openai-compat API)
LLMProvider.CLAUDE      - Anthropic direct (claude-* models)
LLMProvider.OPENAI      - OpenAI direct (gpt-4o, o3-mini, codex, etc.)

Configuring OpenAI
------------------
Add to .env::

    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=o3-mini          # or gpt-4o, o3, o4-mini, codex-mini-latest
    # Optional - Azure OpenAI or a local proxy:
    # OPENAI_BASE_URL=https://your-resource.openai.azure.com/

Then in your entry-point (main.py / run.py / evaluate.py) set DEFAULT_CONFIG
before any agent is imported::

    import os, config
    config.DEFAULT_CONFIG = config.LLMConfig(
        provider=config.LLMProvider.OPENAI,
        model=os.getenv("OPENAI_MODEL", "o3-mini"),
    )

o-series / reasoning model handling
-------------------------------------
Models matched by is_openai_reasoning_model() (o1, o3, o4, codex-*) require:
  - max_completion_tokens  instead of  max_tokens
  - NO temperature parameter (fixed at 1 by the API)
This is handled automatically; no extra flag is needed.
"""
from __future__ import annotations

import json

from config import (
    DEFAULT_CONFIG,
    LLMProvider,
    build_openrouter_provider_preferences,
    is_openai_reasoning_model,
)


def _build_client():
    """Return (client, model_name) for the configured provider."""
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        from openai import OpenAI
        return OpenAI(
            api_key=DEFAULT_CONFIG.openrouter_api_key,
            base_url=DEFAULT_CONFIG.openrouter_base_url,
        ), DEFAULT_CONFIG.model

    if DEFAULT_CONFIG.provider == LLMProvider.OPENAI:
        from openai import OpenAI
        kwargs: dict = {"api_key": DEFAULT_CONFIG.openai_api_key}
        if DEFAULT_CONFIG.openai_base_url:
            kwargs["base_url"] = DEFAULT_CONFIG.openai_base_url
        return OpenAI(**kwargs), DEFAULT_CONFIG.model

    # Default: Anthropic direct
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


def _call_openai_compat(
    client,
    model: str,
    system: str,
    messages: list,
    *,
    is_reasoning: bool,
    extra_body: dict | None = None,
) -> tuple[str, dict]:
    """Shared call path for OpenRouter and OpenAI direct (both use openai SDK).

    o-series / reasoning models require:
      - max_completion_tokens  (not max_tokens)
      - temperature omitted    (API rejects it)
    Standard chat models use the normal max_tokens + temperature params.
    """
    request_kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
    }

    if is_reasoning:
        request_kwargs["max_completion_tokens"] = DEFAULT_CONFIG.max_tokens
    else:
        request_kwargs["temperature"] = DEFAULT_CONFIG.temperature
        request_kwargs["max_tokens"] = DEFAULT_CONFIG.max_tokens

    if extra_body:
        request_kwargs["extra_body"] = extra_body

    r = client.chat.completions.create(**request_kwargs)

    choices = getattr(r, "choices", None) or []
    if not choices:
        raise RuntimeError(
            f"OpenAI-compat API returned no choices. "
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
            f"OpenAI-compat API returned empty/non-text content. "
            f"model={model} response={_response_debug_blob(r)}"
        )

    usage_obj = getattr(r, "usage", None)
    usage = {
        "prompt_tokens": _to_int(getattr(usage_obj, "prompt_tokens", 0)),
        "completion_tokens": _to_int(getattr(usage_obj, "completion_tokens", 0)),
    }
    return content, usage


def _call_llm_with_history(client, model: str, system: str, messages: list) -> tuple[str, dict]:
    """Call LLM with a messages list.

    In the stateless prompt design this is always a single [user_msg] -
    full context is embedded in the prompt, not in conversation history.
    """
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        extra_body: dict = {}
        provider_preferences = build_openrouter_provider_preferences(DEFAULT_CONFIG)
        if provider_preferences:
            extra_body["provider"] = provider_preferences
        if DEFAULT_CONFIG.reasoning_enabled:
            extra_body["reasoning"] = {"enabled": True}
        return _call_openai_compat(
            client, model, system, messages,
            is_reasoning=False,  # OpenRouter handles reasoning server-side
            extra_body=extra_body or None,
        )

    if DEFAULT_CONFIG.provider == LLMProvider.OPENAI:
        return _call_openai_compat(
            client, model, system, messages,
            is_reasoning=is_openai_reasoning_model(model),
        )

    # Anthropic direct
    import anthropic as ant  # noqa: F401
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
