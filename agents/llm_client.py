"""Shared LLM client factory and call helper.

Extracted from agents/engineer.py so both engineer and remediator can
import these utilities without creating a circular dependency.

Supported providers
-------------------
LLMProvider.OPENROUTER  - OpenRouter proxy (any model via openai-compat API)
LLMProvider.CLAUDE      - Anthropic direct (claude-* models)
LLMProvider.OPENAI      - OpenAI direct (gpt-4o, o3-mini, codex, etc.)
LLMProvider.DEEPSEEK    - DeepSeek direct (deepseek-chat, deepseek-reasoner)

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

    if DEFAULT_CONFIG.provider == LLMProvider.DEEPSEEK:
        from openai import OpenAI
        return OpenAI(
            api_key=DEFAULT_CONFIG.deepseek_api_key,
            base_url=DEFAULT_CONFIG.deepseek_base_url,
        ), DEFAULT_CONFIG.model

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


# ---------------------------------------------------------------------------
# Prompt caching helpers
# ---------------------------------------------------------------------------
# System prompts are byte-identical across every call within a scenario (the
# planner sets user_request/objectives once), and engineer's/remediator's
# conversation history repeats every prior turn's content verbatim on each
# new call -- none of it was cached before, so it was billed at full price
# every single time. These helpers wrap that stable content in a
# cache_control breakpoint. The JSON shape is identical between the
# Anthropic SDK's native content-block format and OpenRouter's
# Anthropic-compatible cache_control passthrough (confirmed in OpenRouter's
# prompt-caching docs), so one helper covers both call paths.

def _cache_block(text: str) -> list[dict]:
    """Wrap plain text as a single cache-marked content block."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _mark_last_message_cacheable(messages: list) -> list:
    """Return a copy of `messages` with the last message's content wrapped in
    a cache breakpoint, so a growing multi-turn conversation (engineer's/
    remediator's turn-by-turn history) gets a cache hit on everything up to
    this point on the NEXT call, once this call's messages become that next
    call's prefix. No-op on an empty list or non-string content
    (already-structured content is left untouched)."""
    if not messages or not isinstance(messages[-1].get("content"), str):
        return messages
    marked = dict(messages[-1])
    marked["content"] = _cache_block(marked["content"])
    return messages[:-1] + [marked]


def _openrouter_model_needs_explicit_cache_marker(model: str) -> bool:
    """OpenRouter documents Anthropic, Google Gemini, and Alibaba Qwen as the
    only families requiring an explicit cache_control breakpoint to get any
    caching benefit -- DeepSeek, Z.AI/GLM, OpenAI, Grok, Moonshot, and Groq
    all cache automatically with zero request changes. OpenRouter's docs
    don't say what happens if the marker is sent to a model that doesn't
    need it, so it's only sent to the three confirmed-required families."""
    m = model.lower()
    return m.startswith("anthropic/") or m.startswith("google/gemini") or "qwen" in m


def build_session_id(state: dict, agent_name: str) -> str | None:
    """Stable OpenRouter sticky-routing key.

    OpenRouter routes a model across multiple backend replicas; a cache is
    only warm on whichever replica served the previous call. Without a
    stable routing hint, a follow-up call can land on a different replica
    and miss the cache even though caching itself is "working." A
    session_id pins follow-up calls to the same warm replica from the
    first call. One run_id per benchmark scenario (already assigned once
    per pipeline run, see main.py), suffixed per agent so each agent's own
    repeated calls (sharing a system prompt, and for engineer/remediator a
    growing history) pin together across the scenario's iterations. Returns
    None when run_id is unavailable so callers simply omit the parameter.
    """
    run_id = state.get("run_id")
    return f"{run_id}:{agent_name}" if run_id else None


# ---------------------------------------------------------------------------
# Reasoning-token budget protection (OpenRouter)
# ---------------------------------------------------------------------------
# OpenRouter: "the request's max_tokens limit ... applies to reasoning and
# visible output combined" -- for every reasoning configuration, with no
# server-side protection ("the caller must manually set a larger max_tokens
# themselves"). This codebase already hit this: config.py's own comment on
# openrouter_reasoning_effort notes some models (e.g. GLM 5.3 Flash) "can
# burn the entire budget on reasoning and return empty content otherwise,"
# and the existing workaround was turning effort DOWN rather than protecting
# the content budget. The openrouter_reasoning_max_tokens branch already
# expands the ceiling correctly; this does the same for effort-only mode,
# using OpenRouter's own documented effort-to-budget-share ratio table
# (exact for OpenAI o-series/GPT-5/Grok; used here as a best-effort safety
# margin for other reasoning models that don't publish their own ratio,
# since no correction at all is the confirmed-worse status quo).
_OPENROUTER_REASONING_EFFORT_SHARE = {
    "minimal": 0.10, "low": 0.20, "medium": 0.50, "high": 0.80,
    "xhigh": 0.95, "max": 0.95,
}
_MAX_TOKENS_OVERRIDE_CAP = 128_000  # matches OpenRouter's documented Anthropic reasoning-budget cap


def _reasoning_expanded_max_tokens(base_max_tokens: int | None, effort: str) -> int | None:
    """Expand the completion-token ceiling so `base_max_tokens` of visible
    content budget survives even when the model spends `effort`'s documented
    share of the request on reasoning. Returns None (no override) for an
    unrecognized/empty effort string, since guessing a ratio for it would be
    worse than sending no correction at all — also None (already uncapped,
    nothing to expand) when base_max_tokens itself is None."""
    if base_max_tokens is None:
        return None
    share = _OPENROUTER_REASONING_EFFORT_SHARE.get(effort)
    if share is None:
        return None
    return min(int(base_max_tokens / (1 - share)), _MAX_TOKENS_OVERRIDE_CAP)


def _call_openai_compat(
    client,
    model: str,
    system: str,
    messages: list,
    *,
    is_reasoning: bool,
    extra_body: dict | None = None,
    max_tokens_override: int | None = None,
    use_cache_control: bool = False,
    session_id: str | None = None,
) -> tuple[str, dict]:
    """Shared call path for OpenRouter and OpenAI direct (both use openai SDK).

    o-series / reasoning models require:
      - max_completion_tokens  (not max_tokens)
      - temperature omitted    (API rejects it)
    Standard chat models use the normal max_tokens + temperature params.

    max_tokens_override, when given, replaces DEFAULT_CONFIG.max_tokens for
    this call only — used by the OpenRouter path to add a separate reasoning
    token allowance on top of the configured content budget without mutating
    global config.

    use_cache_control, when True, wraps the system prompt and the last
    message in a cache_control breakpoint (see _cache_block /
    _mark_last_message_cacheable). Callers only set this for the three
    OpenRouter model families documented as requiring it (see
    _openrouter_model_needs_explicit_cache_marker) — every other model's
    request shape here is byte-identical to before this parameter existed.

    session_id, when given, is forwarded as OpenRouter's sticky-routing key
    (see build_session_id) via extra_body (it isn't a parameter the openai
    SDK's own create() signature recognizes) — safe and beneficial for every
    OpenRouter model regardless of use_cache_control.
    """
    if use_cache_control:
        chat_messages = (
            [{"role": "system", "content": _cache_block(system)}]
            + _mark_last_message_cacheable(messages)
        )
    else:
        chat_messages = [{"role": "system", "content": system}] + messages

    request_kwargs: dict = {
        "model": model,
        "messages": chat_messages,
    }

    # A resolved value of None means "uncapped": omit the key entirely rather
    # than sending it as JSON null, so the provider applies its own (usually
    # much larger) default ceiling instead of any fixed number we pick.
    max_tokens = max_tokens_override if max_tokens_override is not None else DEFAULT_CONFIG.max_tokens

    if is_reasoning:
        if max_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_tokens
    else:
        request_kwargs["temperature"] = DEFAULT_CONFIG.temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

    # session_id is an OpenRouter-specific field with no place in the openai
    # SDK's own typed create() signature (confirmed: TypeError "unexpected
    # keyword argument" if passed directly, same as any other non-standard
    # field) -- it has to travel inside extra_body, exactly like `reasoning`
    # and `provider` already do, so the SDK injects it into the raw JSON
    # body instead of validating it against its own parameter list.
    merged_extra_body: dict = dict(extra_body) if extra_body else {}
    if session_id:
        merged_extra_body["session_id"] = session_id
    if merged_extra_body:
        request_kwargs["extra_body"] = merged_extra_body

    def _do_call() -> tuple[str, dict]:
        r = client.chat.completions.create(**request_kwargs)

        usage_obj = getattr(r, "usage", None)
        # prompt_tokens_details carries cache/reasoning breakdowns reported
        # by OpenRouter (and passed through from providers that populate it,
        # including ones that need no cache_control marker at all -- DeepSeek
        # and Z.AI/GLM report cached_tokens automatically). Read unconditionally
        # so those savings are visible regardless of use_cache_control.
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        usage = {
            "prompt_tokens": _to_int(getattr(usage_obj, "prompt_tokens", 0)),
            "completion_tokens": _to_int(getattr(usage_obj, "completion_tokens", 0)),
            "cache_creation_input_tokens": _to_int(getattr(prompt_details, "cache_write_tokens", 0)),
            "cache_read_input_tokens": _to_int(getattr(prompt_details, "cached_tokens", 0)),
            "reasoning_tokens": _to_int(getattr(completion_details, "reasoning_tokens", 0)),
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


def _call_llm_with_history(
    client,
    model: str,
    system: str,
    messages: list,
    *,
    session_id: str | None = None,
) -> tuple[str, dict]:
    """Call LLM with a messages list.

    In the stateless prompt design this is always a single [user_msg] -
    full context is embedded in the prompt, not in conversation history.

    session_id (OpenRouter only — see build_session_id) pins repeated calls
    from the same logical conversation to the same warm backend replica, so
    cache hits (automatic or cache_control-marked) actually land instead of
    depending on incidental routing luck.
    """
    if DEFAULT_CONFIG.provider == LLMProvider.CLAUDE and DEFAULT_CONFIG.max_tokens is None:
        raise ValueError(
            "max_tokens=None (uncapped) is only supported for OpenRouter/OpenAI "
            "direct — Anthropic's API requires an explicit max_tokens and errors "
            "out without one. Set --max-tokens to a specific value for Claude."
        )
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        extra_body: dict = {}
        provider_preferences = build_openrouter_provider_preferences(DEFAULT_CONFIG)
        if provider_preferences:
            extra_body["provider"] = provider_preferences
        max_tokens_override = None
        if DEFAULT_CONFIG.reasoning_enabled:
            reasoning_opts: dict = {"enabled": True}
            # OpenRouter rejects requests that set both reasoning.effort and
            # reasoning.max_tokens ("Only one of ... can be specified"), so
            # these are mutually exclusive here. max_tokens is the more
            # precise lever — it directly caps the shared budget rather than
            # just hinting at it — so it takes precedence when both are
            # configured (e.g. a stray effort default left over alongside a
            # newly-added max_tokens override).
            if DEFAULT_CONFIG.openrouter_reasoning_max_tokens:
                reasoning_opts["max_tokens"] = DEFAULT_CONFIG.openrouter_reasoning_max_tokens
                # Reasoning tokens share the same completion budget as content
                # on OpenRouter, so add the reasoning allowance on top of the
                # configured max_tokens instead of letting it eat into the
                # content budget — max_tokens keeps meaning "guaranteed
                # content budget" even when it's a fixed, research-controlled
                # parameter that can't itself be changed. Base is already
                # uncapped (None) when max_tokens is unset entirely — adding a
                # finite reasoning allowance on top of "uncapped" is still
                # uncapped, so there's nothing to add.
                if DEFAULT_CONFIG.max_tokens is not None:
                    max_tokens_override = DEFAULT_CONFIG.max_tokens + DEFAULT_CONFIG.openrouter_reasoning_max_tokens
            elif DEFAULT_CONFIG.openrouter_reasoning_effort:
                reasoning_opts["effort"] = DEFAULT_CONFIG.openrouter_reasoning_effort
                # Effort-only mode has no explicit token budget to add on top,
                # so without this the same "reasoning shares max_tokens"
                # problem above applies here too -- and does, in practice:
                # see config.py's note on GLM 5.3 Flash burning the whole
                # budget on reasoning and returning empty content. Expand the
                # ceiling using OpenRouter's documented effort/budget-share
                # ratio as a best-effort margin instead of leaving it
                # unprotected.
                max_tokens_override = _reasoning_expanded_max_tokens(
                    DEFAULT_CONFIG.max_tokens, DEFAULT_CONFIG.openrouter_reasoning_effort
                )
            extra_body["reasoning"] = reasoning_opts
        return _call_openai_compat(
            client, model, system, messages,
            is_reasoning=False,  # OpenRouter handles reasoning server-side
            extra_body=extra_body or None,
            max_tokens_override=max_tokens_override,
            use_cache_control=_openrouter_model_needs_explicit_cache_marker(model),
            session_id=session_id,
        )

    if DEFAULT_CONFIG.provider == LLMProvider.OPENAI:
        return _call_openai_compat(
            client, model, system, messages,
            is_reasoning=is_openai_reasoning_model(model),
        )

    if DEFAULT_CONFIG.provider == LLMProvider.DEEPSEEK:
        # deepseek-reasoner returns its reasoning as an extra response field
        # (reasoning_content) rather than needing max_completion_tokens/no-
        # temperature like OpenAI's o-series, so is_reasoning is always False
        # here -- plain max_tokens + temperature both apply normally.
        return _call_openai_compat(
            client, model, system, messages,
            is_reasoning=False,
        )

    # Anthropic direct
    def _do_call() -> tuple[str, dict]:
        r = client.messages.create(
            model=model,
            system=_cache_block(system),
            messages=_mark_last_message_cacheable(messages),
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        usage = {
            "input_tokens": _to_int(getattr(r.usage, "input_tokens", 0)),
            "output_tokens": _to_int(getattr(r.usage, "output_tokens", 0)),
            "cache_creation_input_tokens": _to_int(getattr(r.usage, "cache_creation_input_tokens", 0)),
            "cache_read_input_tokens": _to_int(getattr(r.usage, "cache_read_input_tokens", 0)),
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
