"""ClaudeCodeDriver — the original, battle-tested harness driver.

This is a straight extraction of what was previously free functions in
baselines/harness_baseline.py (build_harness_env, write_mcp_config,
run_harness's cmd-construction, write_debug_transcripts,
tokens_from_model_usage, parse_harness_stream) into the HarnessDriver
interface. The logic itself is unchanged from the verified, extensively
debugged version — see the inline comments for the evidence behind each
non-obvious decision (isolation via CLAUDE_CONFIG_DIR, --tools vs
--allowedTools, --bare being tried and dropped, --debug api, the stalled
detection heuristic, thinking_tokens_total).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark_common import _safe_int, _token_totals

from baselines.drivers.base import HarnessTelemetry

if TYPE_CHECKING:
    from baselines.evaluation import ScenarioConfig
    from baselines.harness_baseline import BaselineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _empty_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }


class ClaudeCodeDriver:
    name = "claude_code"

    # -----------------------------------------------------------------
    # build_command
    # -----------------------------------------------------------------

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
        mcp_path = self._write_mcp_config(workdir, scenario)

        return [
            "claude",
            "-p",
            prompt,
            "--append-system-prompt",
            system_prompt,
            "--model",
            config.harness_model,
            *(["--effort", config.harness_effort] if config.harness_effort else []),
            "--mcp-config",
            str(mcp_path.resolve()),
            "--strict-mcp-config",
            # --tools is what actually restricts the built-in tool set
            # (verified: --allowedTools alone does NOT — a model given only
            # --allowedTools still sees, and freely uses, Bash). No Bash here
            # deliberately: the model must go through validate_iac/
            # deploy_iac, not a shelled-out cfn-lint/aws-cli call that would
            # bypass the iteration counter and break "one validate_iac call
            # == one iteration" comparability.
            #
            # --bare was tried and dropped: its own baseline tool set has no
            # Write tool at all (confirmed: `--tools Write` under --bare is
            # rejected as unrecognized — Write only exists outside --bare),
            # which would force file creation through Bash, reopening
            # exactly the gap above. --setting-sources "" alone was verified
            # sufficient to block AGENTS.md/CLAUDE.md auto-discovery (a
            # probe run reported no awareness of this repo's multi-agent
            # architecture), so --bare's other guarantees were not worth
            # trading Write away for.
            "--tools",
            "Read,Write,Edit",
            "--allowedTools",
            "mcp__iacgod__validate_iac",
            "mcp__iacgod__deploy_iac",
            "mcp__iacgod__submit_template",
            "Read",
            "Write",
            "Edit",
            "--permission-mode",
            "acceptEdits",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--output-format",
            "stream-json",
            "--verbose",
            # api-category debug log. This is the one place that shows what
            # happens BELOW the content layer: exact request dispatch time,
            # which routing path was used (firstParty vs a fallback),
            # latency to first byte, and whether a response stream ever
            # started at all — none of which the stream or transcript files
            # can show, since they only ever see what Claude Code decided to
            # hand back as message content. A session with no "Stream
            # started - received first chunk" line never got a real response
            # from the shim at all (infra-layer failure); one with that line
            # but still empty content got a response the model/shim produced
            # as empty (content-layer failure) — the two are
            # indistinguishable from the transcript alone, but not from this
            # log.
            "--debug",
            "api",
            "--debug-file",
            str(debug_log_path),
        ]

    def _write_mcp_config(self, workdir: Path, scenario: "ScenarioConfig") -> Path:
        path = (workdir / ".mcp.json").resolve()
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "iacgod": {
                            "type": "stdio",
                            "command": sys.executable,
                            "args": [str(REPO_ROOT / "baselines" / "mcp_server.py")],
                            "env": {**scenario.to_env(), "PYTHONPATH": str(REPO_ROOT)},
                        }
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    # -----------------------------------------------------------------
    # build_env
    # -----------------------------------------------------------------

    def build_env(self, config: "BaselineConfig", state_dir: Path) -> dict[str, str]:
        """Environment for the harness subprocess — isolated, but not stripped.

        Isolation here means: nothing the researcher's own Claude Code
        installation carries (OAuth session, hooks, custom agents/MCP
        servers, memory, plugins, this repo's own AGENTS.md) can reach the
        scenario under test, and every scenario gets a clean slate rather
        than accumulating state run to run. Two mechanisms, both empirically
        verified:
          - CLAUDE_CONFIG_DIR (a real, undocumented-in---help but
            confirmed-real env var) redirects Claude Code's entire
            session/project state to a scenario-scoped temp directory —
            verified nothing touches the real ~/.claude.
          - --setting-sources "" (passed in build_command) loads no
            user/project/local settings.json at all, which is where hooks,
            custom agents and output styles are configured — and was
            separately verified to block AGENTS.md/CLAUDE.md
            auto-discovery too: a probe run asked whether it had been told
            about any multi-agent architecture reported no awareness of
            this repo's own AGENTS.md.
        --bare was tried and dropped — see build_command — because its own
        baseline tool set has no Write tool at all.

        What this deliberately does NOT do: strip or redirect the rest of
        the environment (HOME included). The MCP server Claude Code spawns
        runs the real validator toolchain — boto3 resolves AWS credentials
        via ~/.aws/credentials (HOME-relative) for real deploys,
        terraform/cfn-lint/trivy may have their own HOME-relative caches —
        and that subprocess inherits this same environment. Isolating
        Claude Code's own state via CLAUDE_CONFIG_DIR + --setting-sources ""
        achieves the actual goal (a clean, reproducible harness environment
        with a reliably-injected model) without risking a silent, confusing
        deploy failure from a missing credentials file.

        All three model aliases are mapped to the same target so that
        background, subagent and main-loop calls all route to the model
        under test — otherwise the run is not attributable to a single
        model.

        MAX_THINKING_TOKENS controls extended thinking. When
        config.max_thinking_tokens == 0, the harness ASKS to disable
        thinking outright. This is a real, non-trivial request — a probe
        call with it set returned a plain `text` block, no `thinking` block
        at all, thinking_tokens reported as None rather than a spent
        budget — but it is NOT reliably honoured. This exists because of a
        specific, repeatedly-observed failure: deepseek-v4-flash's own
        native tool-call syntax (`<｜DSML｜tool_calls>...`) has been seen
        leaking out as literal text INSIDE a thinking block instead of a
        structured tool_use block — the OpenRouter shim does not parse it
        there — which is exactly the stalled-session pattern (see
        baselines/README.md). Verified split, not a clean fix: reliably
        honoured on short, fresh sessions (0 cache reads) but overridden by
        the model on long, cache-heavy mid-repair-loop turns (one confirmed
        case burned 10,935 thinking tokens with this set to 0). Treat as a
        partial mitigation to combine with --max-stall-retries, not a
        substitute for it — watch thinking_tokens_total in the CSV rather
        than assuming 0 means disabled.
        """
        env = dict(os.environ)
        env.update(
            {
                "CLAUDE_CONFIG_DIR": str(state_dir),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_ERROR_REPORTING": "1",
            }
        )
        if config.max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(config.max_thinking_tokens)

        if config.native_auth:
            return env

        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{config.api_key_env} is not set. Export it, add it to .env, or "
                "pass --native-auth to use Claude Code's own authentication."
            )
        if not config.model:
            raise RuntimeError("--model is required unless --native-auth is used.")

        env.update(
            {
                "ANTHROPIC_BASE_URL": config.effective_base_url
                or config.base_url
                or "https://openrouter.ai/api",
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": config.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": config.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": config.model,
                "CLAUDE_CODE_SUBAGENT_MODEL": config.model,
            }
        )
        return env

    # -----------------------------------------------------------------
    # parse_stream
    # -----------------------------------------------------------------

    def _tokens_from_model_usage(self, model_usage: dict[str, Any]) -> dict[str, int]:
        """Aggregate token counts in the shape _token_totals() produces.

        input/output are populated rather than prompt/completion: Claude
        Code speaks the Anthropic dialect. token_all_tokens is the sum
        either way, so it stays the column that compares directly against
        multi-agent OpenRouter runs (which populate prompt/completion
        instead).
        """
        totals = _empty_tokens()
        for usage in (model_usage or {}).values():
            totals["input_tokens"] += _safe_int(usage.get("inputTokens"))
            totals["output_tokens"] += _safe_int(usage.get("outputTokens"))
        totals["all_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        return totals

    def parse_stream(
        self, stdout: str, *, returncode: int, timed_out: bool, duration: float
    ) -> HarnessTelemetry:
        """Extract per-call token usage and the final result event.

        Two shapes of the stream have to be handled:

          * Assistant events are emitted once per content block, so a
            single LLM response arrives as several events sharing one
            message.id (a thinking block and a tool_use block, say).
            Deduplicating by id is what makes llm_calls_total mean "LLM
            round trips" — directly comparable to the multi-agent runs'
            llm_call_log length — rather than double-counting.

          * Per-message usage comes back all zeros through the OpenRouter
            Anthropic-compat endpoint; only the result event's modelUsage
            carries real counts. Per-call usage is still preferred when it
            is populated (the native Anthropic path does fill it in), with
            modelUsage as the fallback. See token_usage_source in the
            emitted payload.

          * A session can end with Claude Code reporting is_error=false and
            subtype="success" while the model never did anything. Observed
            with deepseek-v4-flash: the model returns a turn with no text
            and no tool_use (a genuinely empty completion), or emits its
            native tool-call syntax as literal text inside a "thinking"
            block instead of a structured tool_use block, which the
            OpenRouter shim does not parse — Claude Code sees a
            content-free turn either way. Claude Code has exactly one
            built-in recovery: a synthetic user turn ("[Your previous
            response had no visible output...]", isSynthetic: true).
            `stalled` is true when that nudge fired and no tool_use
            appeared in any assistant turn afterward — i.e. the harness's
            own recovery attempt also failed and it gave up.
            is_error/returncode cannot detect this: Claude Code still
            reports success.

          * thinking_tokens_total is pulled out of modelUsage explicitly
            (NOT buried in the raw stream) because it is the single most
            diagnostic number for the failure above: the DSML leak has
            only ever been observed inside a thinking block, and
            --max-thinking-tokens 0 does NOT reliably prevent it.
        """
        calls_by_id: dict[str, dict[str, Any]] = {}
        result_event: dict[str, Any] = {}
        last_synthetic_nudge_index: int | None = None
        tool_use_after_last_nudge = False

        for index, line in enumerate(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if etype == "assistant":
                message = event.get("message") or {}
                usage = message.get("usage") or {}
                message_id = message.get("id") or f"anon-{len(calls_by_id)}"
                if last_synthetic_nudge_index is not None and any(
                    c.get("type") == "tool_use" for c in (message.get("content") or [])
                ):
                    tool_use_after_last_nudge = True
                if message_id in calls_by_id:
                    continue
                calls_by_id[message_id] = {
                    "agent": "harness",
                    "iteration": None,
                    "model": message.get("model"),
                    "message_id": message_id,
                    "prompt": "",  # full transcript lives in harness_stream.jsonl
                    "response": "",
                    "timestamp": datetime.now().isoformat(),
                    "token_usage": {
                        "input_tokens": _safe_int(usage.get("input_tokens")),
                        "output_tokens": _safe_int(usage.get("output_tokens")),
                        "cache_read_input_tokens": _safe_int(
                            usage.get("cache_read_input_tokens")
                        ),
                        "cache_creation_input_tokens": _safe_int(
                            usage.get("cache_creation_input_tokens")
                        ),
                    },
                }
            elif etype == "user" and event.get("isSynthetic"):
                last_synthetic_nudge_index = index
                tool_use_after_last_nudge = False
            elif etype == "result":
                result_event = event

        llm_calls = list(calls_by_id.values())
        model_usage = result_event.get("modelUsage") or {}

        per_call_tokens = _token_totals(llm_calls)
        if per_call_tokens["all_tokens"] > 0:
            tokens, token_source = per_call_tokens, "per_call"
        else:
            tokens, token_source = self._tokens_from_model_usage(model_usage), "model_usage"

        stalled = (
            last_synthetic_nudge_index is not None and not tool_use_after_last_nudge
        )

        thinking_tokens_total = sum(
            _safe_int(usage.get("thinkingTokens")) for usage in model_usage.values()
        )

        return {
            "llm_call_log": llm_calls,
            "token_usage": tokens,
            "token_usage_source": token_source,
            "num_turns": _safe_int(result_event.get("num_turns")),
            "session_id": result_event.get("session_id"),
            "model_usage": model_usage,
            "thinking_tokens_total": thinking_tokens_total,
            "is_error": bool(result_event.get("is_error")) or returncode != 0,
            "stalled": stalled,
            "stop_reason": result_event.get("stop_reason"),
            "subtype": result_event.get("subtype"),
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "harness_reported_cost_usd": result_event.get("total_cost_usd"),
        }

    # -----------------------------------------------------------------
    # write_debug_transcripts
    # -----------------------------------------------------------------

    def _format_tool_input(self, args: dict[str, Any], indent: str = "    ") -> list[str]:
        lines: list[str] = []
        for key, value in args.items():
            if isinstance(value, str) and "\n" in value:
                lines.append(f"{indent}{key}:")
                lines.append(f"{indent}{'-' * 60}")
                for content_line in value.splitlines():
                    lines.append(f"{indent}| {content_line}")
                lines.append(f"{indent}{'-' * 60}")
            else:
                lines.append(f"{indent}{key}: {value}")
        return lines

    def _format_tool_result_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block))
            return "\n".join(parts)
        return json.dumps(content)

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
        """Render the raw stream-json into two human-readable .txt files.

        state_dir/debug_log_path are unused here: Claude Code already wrote
        its own wire-level debug log directly to debug_log_path via
        --debug-file (see build_command) — nothing to copy.

        harness_transcript.txt is the full conversation in order: system
        prompt, initial user prompt, then every thinking/text/tool_use block
        and every tool result exactly as they occurred, plus the final
        result summary. harness_tool_calls.txt is the same session reduced
        to just the MCP/file tool calls and their results — for scanning a
        run without reading the full transcript. Both are written even for
        a stalled or failed session: a transcript showing exactly nothing
        happened, or where it broke off, is itself the debugging signal.

        Kept deliberately dumb (string formatting, no dependency on
        parse_stream's dedup/aggregation logic) so a change to token
        accounting can never silently change what gets recorded for
        debugging.
        """
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

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")

            if etype == "assistant":
                for block in (event.get("message") or {}).get("content") or []:
                    btype = block.get("type")
                    if btype == "thinking":
                        text = (block.get("thinking") or "").strip()
                        if text:
                            transcript_lines.append(f"[ASSISTANT — thinking]\n{text}\n\n")
                    elif btype == "text":
                        text = (block.get("text") or "").strip()
                        if text:
                            transcript_lines.append(f"[ASSISTANT — text]\n{text}\n\n")
                    elif btype == "tool_use":
                        tool_call_index += 1
                        name = block.get("name", "?")
                        args = block.get("input") or {}
                        block_lines = self._format_tool_input(args)
                        entry = (
                            f"[ASSISTANT — tool_use #{tool_call_index}: {name}]\n"
                            + ("\n".join(block_lines) + "\n" if block_lines else "  (no arguments)\n")
                            + "\n"
                        )
                        transcript_lines.append(entry)
                        tool_call_lines.append(entry)
            elif etype == "user":
                content = (event.get("message") or {}).get("content")
                if event.get("isSynthetic"):
                    text = ""
                    if isinstance(content, list):
                        text = "\n".join(
                            b.get("text", "") for b in content if b.get("type") == "text"
                        )
                    transcript_lines.append(
                        f"[HARNESS NUDGE — empty-turn recovery]\n{text.strip()}\n\n"
                    )
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            text = self._format_tool_result_text(block.get("content")).strip()
                            entry = f"[TOOL RESULT]\n{text}\n\n"
                            transcript_lines.append(entry)
                            tool_call_lines.append(entry)
            elif etype == "result":
                summary = (
                    f"{'=' * 78}\n"
                    f"Session Result\n"
                    f"  is_error    : {event.get('is_error')}\n"
                    f"  subtype     : {event.get('subtype')}\n"
                    f"  stop_reason : {event.get('stop_reason')}\n"
                    f"  num_turns   : {event.get('num_turns')}\n"
                    f"  result text : {event.get('result', '')!r}\n"
                    f"{'=' * 78}\n"
                )
                transcript_lines.append(summary)
                tool_call_lines.append(f"\n{summary}")

        transcript_path.write_text("".join(transcript_lines), encoding="utf-8")
        if tool_call_index == 0:
            tool_call_lines.insert(1, "(no tool calls were made in this session)\n\n")
        tool_calls_path.write_text("".join(tool_call_lines), encoding="utf-8")
