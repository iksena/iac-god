"""The HarnessDriver interface and the common telemetry shape drivers emit.

Kept as a plain dict shape (via TypedDict, for documentation and type
checking only — nothing at runtime enforces it) rather than a dataclass,
because every consumer in harness_baseline.py already reads this as
dict[str, Any] and changing that would mean touching run_scenario,
finalize_scenario, the CSV writer and both existing regression tests for no
behavioural benefit.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from baselines.evaluation import ScenarioConfig
    from baselines.harness_baseline import BaselineConfig


class HarnessTelemetry(TypedDict):
    """What driver.parse_stream() must return. See parse_harness_stream in
    baselines/drivers/claude_code.py for the reference implementation and the
    fullest documentation of each field's meaning and edge cases."""

    llm_call_log: list[dict[str, Any]]
    token_usage: dict[str, int]
    token_usage_source: str
    num_turns: int
    session_id: str | None
    model_usage: dict[str, Any]
    thinking_tokens_total: int
    is_error: bool
    stalled: bool
    stop_reason: str | None
    subtype: str | None
    returncode: int
    timed_out: bool
    duration_seconds: float
    harness_reported_cost_usd: float | None


class HarnessDriver(Protocol):
    """One implementation per coding harness (Claude Code, OpenCode, ...).

    Instantiated once per scenario attempt by run_harness() in
    harness_baseline.py, which owns everything generic: spawning the
    subprocess, streaming stdout to disk, the timeout/process-group-kill
    logic, and calling these four methods in order. A driver never touches
    the subprocess directly — it only describes what to run and how to make
    sense of what came back.
    """

    #: Short, filesystem-safe identifier. Used for the harness's isolated
    #: state directory name (workdir/.<name>_state) and appears in
    #: BaselineConfig.harness / the results CSV.
    name: str

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
        """Full argv for the harness subprocess.

        Responsible for writing any config file the harness itself reads
        (MCP servers, provider/model routing, tool restrictions) into
        workdir — run_harness() does not know that shape and never inspects
        it. state_dir is this attempt's isolated state/config directory
        (already created); debug_log_path is where a wire-level debug log
        should be written if the harness supports one (optional — a driver
        that has no equivalent may ignore it).
        """
        ...

    def build_env(self, config: "BaselineConfig", state_dir: Path) -> dict[str, str]:
        """Environment for the harness subprocess.

        Responsible for both routing (auth token / base URL / model
        aliases) and isolation (redirecting the harness's own state/config
        directories into state_dir so nothing leaks to or from the
        researcher's real installation, and nothing persists between
        scenarios).
        """
        ...

    def parse_stream(
        self, stdout: str, *, returncode: int, timed_out: bool, duration: float
    ) -> HarnessTelemetry:
        """Parse raw stdout into the common telemetry shape.

        stdout is whatever the harness wrote in its streaming/JSON-lines
        output mode over the full process lifetime (already captured to
        disk by run_harness() before this is called).
        """
        ...

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
        """Render stdout into the two human-readable debug .txt files.

        Deliberately independent of parse_stream's dedup/aggregation logic
        (own pass over the same stdout) so a change to token accounting can
        never silently change what gets recorded for debugging. Must be
        written even for a stalled/failed session — an empty or truncated
        transcript is itself the debugging signal.

        state_dir and debug_log_path are passed again (build_command already
        saw them) so a driver whose harness cannot be told to write its
        wire-level debug log to an arbitrary path directly (unlike Claude
        Code's --debug-file) can locate that log inside its own state_dir
        after the process exits and copy/rename it to debug_log_path here,
        keeping the artifact filename convention (harness_debug.log)
        consistent across drivers. A driver with no such log may ignore
        both.
        """
        ...
