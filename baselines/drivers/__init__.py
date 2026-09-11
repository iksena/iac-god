"""Pluggable harness drivers.

A driver owns exactly the parts of running a coding harness that differ
between harnesses: the CLI invocation, the subprocess environment, the
harness-specific config file (MCP servers, provider/model routing, tool
restrictions), and parsing that harness's own event-stream format into the
common telemetry shape the rest of baselines/ already consumes.

Everything else — the scenario loop, the stall-then-passed retry logic,
finalize_scenario's re-validation, the CSV/summary writers, the retry proxy,
the MCP server and validators themselves — is harness-agnostic and lives in
baselines/harness_baseline.py, baselines/mcp_server.py and
baselines/evaluation.py unchanged. Adding a third harness means adding a
third driver here, not touching any of that.
"""
from baselines.drivers.base import HarnessDriver
from baselines.drivers.claude_code import ClaudeCodeDriver
from baselines.drivers.opencode import OpenCodeDriver

DRIVERS: dict[str, type[HarnessDriver]] = {
    "claude_code": ClaudeCodeDriver,
    "opencode": OpenCodeDriver,
}


def get_driver(name: str) -> HarnessDriver:
    try:
        return DRIVERS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown harness {name!r}. Available: {', '.join(sorted(DRIVERS))}"
        ) from None
