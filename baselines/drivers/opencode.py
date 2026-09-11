"""OpenCodeDriver — second harness, chosen specifically to be
model-agnostic in a way Claude Code is not.

Verified directly (real API calls, not just documentation) before writing
any of this, in the same conversation that led to this file:

  - Real DeepSeek V4 Flash call via OpenRouter's native OpenAI-compatible
    endpoint, through OpenCode: clean structured response, no shim
    translation involved (OpenCode never touches the Anthropic Messages API
    shape at all).
  - A full 9-round-trip MCP tool-calling session (write -> validate_iac
    (fail) -> edit -> validate_iac (fail) -> read -> edit -> edit ->
    validate_iac (pass) -> done) against baselines/mcp_server.py,
    unmodified: zero malformed tool calls, zero DSML-style leaks, zero
    empty completions — a result Claude Code never produced on this model.
  - A 20-call raw burst against OpenRouter's /v1/chat/completions endpoint
    for this model: 0/20 empty completions (vs. the confirmed ~7% (1/15)
    rate on the Anthropic-compat /v1/messages endpoint Claude Code uses).
    Not proof the failure mode is absent here (0/20 is statistically
    consistent with a true rate as high as ~10-15%), but a real, positive
    signal.
  - Bash is genuinely unreachable under the deny-all + allowlist agent
    permission model (provoked the model into trying; OpenCode returned an
    explicit "unavailable tool" error naming the real, restricted toolset).
  - XDG_DATA_HOME/XDG_CONFIG_HOME/XDG_STATE_HOME isolation: confirmed all
    session/project state, including the wire-level debug log, is written
    under those directories and nothing leaks to the real
    ~/.local/share/opencode or ~/.config/opencode.
  - small_model (used for auxiliary tasks like session-title generation)
    defaults to a DIFFERENT vendor's model (google/gemini-3.8-flash was
    observed) unless pinned — pinned here to the same target model so every
    LLM call in a session is attributable to one model, mirroring why
    ClaudeCodeDriver maps all three Claude model aliases to the same target.

What is NOT yet verified, and should be treated as open until it is:
  - Whether OpenCode has any equivalent to Claude Code's one-shot
    "no visible output, please continue" nudge, or any retry behaviour of
    its own on an empty completion. `stalled` below is therefore a
    structural definition (no text or successful tool_use appeared
    anywhere in the whole session) rather than a copy of a confirmed
    failure signature — there is no confirmed OpenCode failure signature
    yet, because none has been observed.
  - Whether the title-generation call (now pinned to the same model, so it
    no longer pollutes model attribution) should be excluded from
    llm_calls_total/token accounting entirely. It is currently counted —
    one extra, cheap call per scenario — documented here rather than
    filtered via fragile correlation against the internal log file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark_common import _safe_int, _token_totals

from baselines.drivers.base import HarnessTelemetry

if TYPE_CHECKING:
    from baselines.evaluation import ScenarioConfig
    from baselines.harness_baseline import BaselineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

AGENT_NAME = "iacgod"
MCP_SERVER_NAME = "iacgod"


def _empty_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }


class OpenCodeDriver:
    name = "opencode"

    # -----------------------------------------------------------------
    # build_command
    # -----------------------------------------------------------------

    def _model_arg(self, config: "BaselineConfig") -> str:
        if config.native_auth:
            # Whatever alias the user's own opencode auth/config resolves —
            # mirrors ClaudeCodeDriver's native_auth path, which likewise
            # leaves model selection to the harness's own configuration.
            return config.harness_model
        return f"openrouter/{config.model}"

    def _write_config(
        self,
        *,
        workdir: Path,
        scenario: "ScenarioConfig",
        config: "BaselineConfig",
        system_prompt_path: Path,
    ) -> None:
        model_id = config.model or config.harness_model
        base = (
            config.effective_base_url or config.base_url or "https://openrouter.ai/api"
        ).rstrip("/")
        # OpenRouter's OpenAI-compatible endpoint lives at /v1, not the bare
        # /api Claude Code's Anthropic-compat routing uses — this is a
        # genuinely different endpoint, not a URL-shape coincidence (see
        # module docstring: no Anthropic-shim translation happens at all on
        # this path).
        base_url = base if base.endswith("/v1") else f"{base}/v1"

        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {
                MCP_SERVER_NAME: {
                    "type": "local",
                    "command": [
                        os.environ.get("PYTHON_EXECUTABLE") or "python3",
                        str(REPO_ROOT / "baselines" / "mcp_server.py"),
                    ],
                    "environment": {**scenario.to_env(), "PYTHONPATH": str(REPO_ROOT)},
                }
            },
            "agent": {
                AGENT_NAME: {
                    "description": "IaCGOD harness-baseline scenario agent",
                    "mode": "primary",
                    "prompt": f"{{file:{system_prompt_path}}}",
                    # Deny-all catch-all, then an explicit allowlist —
                    # verified: the model cannot reach bash under this
                    # (tried; got an explicit "unavailable tool" error
                    # naming exactly this set). The model must go through
                    # validate_iac/deploy_iac, not a shelled-out
                    # cfn-lint/aws-cli call that would bypass the iteration
                    # counter and break "one validate_iac call == one
                    # iteration" comparability — the same requirement
                    # ClaudeCodeDriver enforces via --tools.
                    "permission": {
                        "*": "deny",
                        "read": "allow",
                        "write": "allow",
                        "edit": "allow",
                        f"{MCP_SERVER_NAME}_validate_iac": "allow",
                        f"{MCP_SERVER_NAME}_deploy_iac": "allow",
                        f"{MCP_SERVER_NAME}_submit_template": "allow",
                    },
                }
            },
        }

        if not config.native_auth:
            cfg["small_model"] = f"openrouter/{model_id}"
            cfg["provider"] = {
                "openrouter": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenRouter",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": f"{{env:{config.api_key_env}}}",
                    },
                    "models": {model_id: {"name": model_id}},
                }
            }

        (workdir / "opencode.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def build_command(
        self,
        *,
        config: "BaselineConfig",
        scenario: "ScenarioConfig",
        prompt: str,
        workdir: Path,
        system_prompt: str,
        state_dir: Path,
        debug_log_path: Path,
    ) -> list[str]:
        # opencode run has no --append-system-prompt equivalent (confirmed:
        # not present in `opencode run --help`'s flag list) — the agent's
        # own "prompt" field (a {file:...} reference, written here) is the
        # documented mechanism instead.
        system_prompt_path = (workdir / "SYSTEM_PROMPT.md").resolve()
        system_prompt_path.write_text(system_prompt, encoding="utf-8")
        self._write_config(
            workdir=workdir, scenario=scenario, config=config,
            system_prompt_path=system_prompt_path,
        )

        return [
            "opencode",
            "run",
            prompt,
            "--model",
            self._model_arg(config),
            "--agent",
            AGENT_NAME,
            "--format",
            "json",
            "--auto",
        ]

    # -----------------------------------------------------------------
    # build_env
    # -----------------------------------------------------------------

    def build_env(self, config: "BaselineConfig", state_dir: Path) -> dict[str, str]:
        """Isolation via XDG base directories — verified empirically (see
        module docstring): with these set, OpenCode's entire session/
        project database, config, locks and its own wire-level log file are
        written under state_dir, and nothing touches the real
        ~/.local/share/opencode or ~/.config/opencode. No equivalent of
        Claude Code's HOME-preservation caveat is needed here for the same
        reason it wasn't needed there: HOME itself is never touched, only
        the XDG_* variables, so the MCP server's validator toolchain
        (boto3, terraform, cfn-lint) still resolves ~/.aws/credentials and
        its own caches normally.
        """
        data_dir = state_dir / "data"
        config_dir = state_dir / "config"
        xdg_state = state_dir / "state"
        for d in (data_dir, config_dir, xdg_state):
            d.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(
            {
                "XDG_DATA_HOME": str(data_dir),
                "XDG_CONFIG_HOME": str(config_dir),
                "XDG_STATE_HOME": str(xdg_state),
            }
        )

        if config.native_auth:
            return env

        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{config.api_key_env} is not set. Export it, add it to .env, or "
                "pass --native-auth to use opencode's own authentication."
            )
        if not config.model:
            raise RuntimeError("--model is required unless --native-auth is used.")
        # No further translation needed: opencode.json's apiKey field
        # interpolates "{env:<config.api_key_env>}" directly (see
        # _write_config) — the env var just needs to be present under its
        # own name, unlike Claude Code's ANTHROPIC_* remapping.
        return env

    # -----------------------------------------------------------------
    # parse_stream
    # -----------------------------------------------------------------

    def parse_stream(
        self, stdout: str, *, returncode: int, timed_out: bool, duration: float
    ) -> HarnessTelemetry:
        """Extract per-call token usage from opencode's `run --format json` stream.

        Event shapes (as actually observed, not merely documented):
          {"type":"step_start", "part":{"type":"step-start", "messageID":...}}
          {"type":"text", "part":{"type":"text","text":"...", "messageID":...}}
          {"type":"tool_use", "part":{"type":"tool","tool":"<name>","callID":...,
              "state":{"status":"completed"|"error","input":{...},"output":"...",
                       "error":"..."(if failed)}, "messageID":...}}
          {"type":"step_finish", "part":{"type":"step-finish","reason":"stop"|
              "tool-calls", "tokens":{"total":N,"input":N,"output":N,
              "reasoning":N,"cache":{"read":N,"write":N}}, "cost":N,
              "messageID":...}}

        One step_finish == one LLM round trip (mirrors ClaudeCodeDriver's
        per-message dedup, just keyed off step_finish directly since
        opencode's stream doesn't split one response across several
        top-level events the way Claude Code's assistant/content-block
        events do).

        No per-call `model` field is present on step_finish events in the
        stream (confirmed by inspection — it only appears in opencode's own
        internal log file, which this driver does not parse). model_usage
        is therefore keyed by the model we explicitly requested, which is
        safe for how this runner always invokes opencode (a single
        --model, no fallback list) but would need revisiting if that ever
        changes.

        `stalled` here is a structural definition, not a copy of a
        confirmed failure signature — see the module docstring: no
        OpenCode-specific empty-completion pattern has been observed yet.
        A session counts as stalled when no step anywhere in it produced
        either non-empty text or a completed tool_use.
        """
        llm_calls: list[dict[str, Any]] = []
        thinking_tokens_total = 0
        total_input = total_output = 0
        model_id: str | None = None
        any_real_content = False
        last_reason: str | None = None
        seen_error = False

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            part = event.get("part") or {}

            if etype == "text" and (part.get("text") or "").strip():
                any_real_content = True
            elif etype == "tool_use":
                state = part.get("state") or {}
                if state.get("status") == "completed":
                    any_real_content = True
                elif state.get("status") == "error":
                    seen_error = True
            elif etype == "step_finish":
                tokens = part.get("tokens") or {}
                last_reason = part.get("reason")
                total_input += _safe_int(tokens.get("input"))
                total_output += _safe_int(tokens.get("output"))
                thinking_tokens_total += _safe_int(tokens.get("reasoning"))
                message_id = part.get("messageID") or f"anon-{len(llm_calls)}"
                cache = tokens.get("cache") or {}
                llm_calls.append(
                    {
                        "agent": "harness",
                        "iteration": None,
                        "model": None,  # not present per-call — see docstring
                        "message_id": message_id,
                        "prompt": "",
                        "response": "",
                        "timestamp": datetime.now().isoformat(),
                        "token_usage": {
                            "input_tokens": _safe_int(tokens.get("input")),
                            "output_tokens": _safe_int(tokens.get("output")),
                            "cache_read_input_tokens": _safe_int(cache.get("read")),
                            "cache_creation_input_tokens": _safe_int(cache.get("write")),
                        },
                    }
                )
            elif isinstance(etype, str) and "error" in etype.lower():
                seen_error = True

        token_usage = _empty_tokens()
        token_usage["input_tokens"] = total_input
        token_usage["output_tokens"] = total_output
        token_usage["all_tokens"] = total_input + total_output

        model_usage: dict[str, Any] = {}
        if llm_calls:
            model_usage = {
                "__requested__": {
                    "inputTokens": total_input,
                    "outputTokens": total_output,
                    "thinkingTokens": thinking_tokens_total,
                }
            }

        stalled = len(llm_calls) > 0 and not any_real_content

        return {
            "llm_call_log": llm_calls,
            "token_usage": token_usage,
            "token_usage_source": "per_call",
            "num_turns": len(llm_calls),
            "session_id": None,
            "model_usage": model_usage,
            "thinking_tokens_total": thinking_tokens_total,
            "is_error": (returncode != 0) or (seen_error and not any_real_content),
            "stalled": stalled,
            "stop_reason": last_reason,
            "subtype": "success" if returncode == 0 else "error",
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "harness_reported_cost_usd": None,  # summed via compute_cost() instead
        }

    # -----------------------------------------------------------------
    # write_debug_transcripts
    # -----------------------------------------------------------------

    def write_debug_transcripts(
        self,
        stdout: str,
        *,
        run_id: str,
        system_prompt: str,
        user_prompt: str,
        harness_model_alias: str,
        routed_model: str | None,
        state_dir: Path,
        debug_log_path: Path,
        transcript_path: Path,
        tool_calls_path: Path,
    ) -> None:
        header = (
            f"{'=' * 78}\n"
            f"Harness Transcript\n"
            f"Run ID        : {run_id}\n"
            f"Harness model : {harness_model_alias}"
            + (f"  (routed to: {routed_model})" if routed_model else "")
            + f"\n{'=' * 78}\n\n"
            f"[SYSTEM PROMPT]\n{system_prompt.strip()}\n\n"
            f"[USER] (initial prompt)\n{user_prompt.strip()}\n\n"
        )
        transcript_lines: list[str] = [header]
        tool_call_lines: list[str] = [
            f"{'=' * 78}\nTool Calls — Run {run_id}\n{'=' * 78}\n\n"
        ]
        tool_call_index = 0
        last_result_text = ""

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            part = event.get("part") or {}

            if etype == "text":
                text = (part.get("text") or "").strip()
                if text:
                    transcript_lines.append(f"[ASSISTANT — text]\n{text}\n\n")
                    last_result_text = text
            elif etype == "tool_use":
                tool_call_index += 1
                name = part.get("tool", "?")
                state = part.get("state") or {}
                lines = [f"    {k}: {v}" for k, v in (state.get("input") or {}).items()]
                entry = (
                    f"[ASSISTANT — tool_use #{tool_call_index}: {name}]\n"
                    + ("\n".join(lines) + "\n" if lines else "  (no arguments)\n")
                    + "\n"
                )
                transcript_lines.append(entry)
                tool_call_lines.append(entry)

                result_text = state.get("error") or state.get("output") or ""
                if result_text:
                    result_entry = f"[TOOL RESULT]\n{str(result_text).strip()}\n\n"
                    transcript_lines.append(result_entry)
                    tool_call_lines.append(result_entry)
            elif etype == "step_finish":
                tokens = part.get("tokens") or {}
                summary = (
                    f"  [step finished: reason={part.get('reason')} "
                    f"tokens={tokens} cost={part.get('cost')}]\n\n"
                )
                transcript_lines.append(summary)

        final_summary = (
            f"{'=' * 78}\n"
            f"Session Result\n"
            f"  final text  : {last_result_text!r}\n"
            f"{'=' * 78}\n"
        )
        transcript_lines.append(final_summary)
        tool_call_lines.append(f"\n{final_summary}")

        transcript_path.write_text("".join(transcript_lines), encoding="utf-8")
        if tool_call_index == 0:
            tool_call_lines.insert(1, "(no tool calls were made in this session)\n\n")
        tool_calls_path.write_text("".join(tool_call_lines), encoding="utf-8")

        # opencode has no --debug-file equivalent to write its wire-level
        # log to an arbitrary path directly (see module docstring) — its
        # log lands at XDG_DATA_HOME/opencode/log/opencode.log, which
        # build_env pointed at state_dir/data. Copy it into the standard
        # harness_debug.log artifact name so both drivers produce the same
        # filename for anyone inspecting runs/<run_id>/ without needing to
        # know which harness produced it.
        source_log = state_dir / "data" / "opencode" / "log" / "opencode.log"
        if source_log.exists():
            debug_log_path.write_text(
                source_log.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
            )
