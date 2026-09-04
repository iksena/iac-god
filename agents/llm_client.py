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
import time

import anthropic
import openai

from config import (
    DEFAULT_CONFIG,
    LLMProvider,
    build_openrouter_provider_preferences,
    is_openai_reasoning_model,
)


class EmptyCompletionError(RuntimeError):
    """Raised when the LLM API returns no usable completion content.

    Carries any token usage the failed attempt still reported (e.g. reasoning
    tokens burned before the model emitted a blank message), so a retry
    wrapper can account for it even though the attempt failed.
    """

    def __init__(self, message: str, usage: dict | None = None):
        super().__init__(message)
        self.usage = usage or {}


_CONNECTION_ERRORS = (openai.APIConnectionError, anthropic.APIConnectionError)
# openai.APITimeoutError / anthropic.APITimeoutError both subclass
# APIConnectionError, so they're covered automatically. Auth/bad-request/
# rate-limit errors are NOT in this tuple and propagate immediately, unretried.


def _merge_usage(usage: dict, wasted: dict) -> dict:
    merged = dict(usage)
    for key, value in wasted.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _call_with_retry(fn, *, label: str, max_attempts: int, backoff_seconds: float):
    """Call fn() (a zero-arg closure issuing the exact same request every time)
    up to max_attempts times, retrying only on connection/timeout errors and
    EmptyCompletionError. On eventual success, any token usage burned by prior
    failed attempts is folded into the returned usage dict. Re-raises the last
    exception unchanged if every attempt fails.
    """
    wasted_usage: dict = {}
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            content, usage = fn()
            if wasted_usage:
                usage = _merge_usage(usage, wasted_usage)
            return content, usage
        except EmptyCompletionError as exc:
            last_exc = exc
            for key, value in exc.usage.items():
                wasted_usage[key] = wasted_usage.get(key, 0) + value
            reason = "empty completion"
        except _CONNECTION_ERRORS as exc:
            last_exc = exc
            reason = "connection error"

        if attempt >= max_attempts:
            print(f"[LLMClient] {label}: {reason} on attempt {attempt}/{max_attempts}, giving up: {last_exc}")
            raise last_exc

        delay = backoff_seconds * (2 ** (attempt - 1))
        print(
            f"[LLMClient] {label}: {reason} on attempt {attempt}/{max_attempts}, "
            f"retrying in {delay:.0f}s: {last_exc}"
        )
        time.sleep(delay)

    raise last_exc  # unreachable


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

    def _do_call() -> tuple[str, dict]:
        r = client.chat.completions.create(**request_kwargs)

        usage_obj = getattr(r, "usage", None)
        usage = {
            "prompt_tokens": _to_int(getattr(usage_obj, "prompt_tokens", 0)),
            "completion_tokens": _to_int(getattr(usage_obj, "completion_tokens", 0)),
        }

        choices = getattr(r, "choices", None) or []
        if not choices:
            raise EmptyCompletionError(
                f"OpenAI-compat API returned no choices. "
                f"model={model} response={_response_debug_blob(r)}",
                usage=usage,
            )

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else ""
                for part in content
            ).strip()

        if not isinstance(content, str) or not content.strip():
            raise EmptyCompletionError(
                f"OpenAI-compat API returned empty/non-text content. "
                f"model={model} response={_response_debug_blob(r)}",
                usage=usage,
            )

        return content, usage

    return _call_with_retry(
        _do_call,
        label=f"openai-compat({model})",
        max_attempts=DEFAULT_CONFIG.llm_retry_max_attempts,
        backoff_seconds=DEFAULT_CONFIG.llm_retry_backoff_seconds,
    )


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
            reasoning_opts: dict = {"enabled": True}
            if DEFAULT_CONFIG.openrouter_reasoning_effort:
                reasoning_opts["effort"] = DEFAULT_CONFIG.openrouter_reasoning_effort
            extra_body["reasoning"] = reasoning_opts
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
    def _do_call() -> tuple[str, dict]:
        r = client.messages.create(
            model=model,
            system=system,
            messages=messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        usage = {
            "input_tokens": _to_int(getattr(r.usage, "input_tokens", 0)),
            "output_tokens": _to_int(getattr(r.usage, "output_tokens", 0)),
        }
        blocks = getattr(r, "content", None) or []
        text = "".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
        ).strip()
        if not text:
            raise EmptyCompletionError(
                f"Anthropic API returned empty/non-text content. "
                f"model={model} response={_response_debug_blob(r)}",
                usage=usage,
            )
        return text, usage

    return _call_with_retry(
        _do_call,
        label=f"anthropic({model})",
        max_attempts=DEFAULT_CONFIG.llm_retry_max_attempts,
        backoff_seconds=DEFAULT_CONFIG.llm_retry_backoff_seconds,
    )
