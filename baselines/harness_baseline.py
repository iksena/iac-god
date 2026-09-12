"""Single-agent harness baseline runner.

Runs the same benchmark CSVs as benchmark.py, but replaces the LangGraph
multi-agent pipeline with an off-the-shelf coding harness (Claude Code,
OpenCode, ... — see baselines/drivers/) driving IaCGOD's own validators
through an MCP server.

    python -m baselines.harness_baseline \
        --harness opencode \
        --iac-type terraform \
        --dataset data/tf_eval_benchmark_real_aws.csv \
        --model deepseek/deepseek-v4-flash \
        --deploy-target aws --max-iterations 30

Control is inverted relative to benchmark.py: the harness owns the loop and
calls validate_iac / deploy_iac / submit_template itself. This process only
sets up each scenario, launches the harness, and scores what comes back.
Everything in this module is harness-agnostic; what differs between harnesses
(the CLI invocation, its config file, parsing its output) lives entirely in
baselines/drivers/ — see baselines/drivers/base.py for the interface.

Scoring is deliberately independent of the harness's own claims. submit_template
records an assertion; the verdict here is recomputed from the recorded
validation results, re-validating (and deploying, when configured) whenever the
submitted artifact was never actually validated in that exact form. Observed in
smoke testing: a harness will rationalise an error it cannot fix and submit
anyway.

Emits results.csv / results.jsonl / summary.json in the same shape benchmark.py
does, plus runs/<run_id>/ artifacts written by the same ResearchRecorder, so the
existing aggregation scripts read baseline and multi-agent runs identically.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_common import (
    CSV_RESULT_FIELDS,
    _append_jsonl,
    _build_summary,
    _extract_iteration_records,
    _extract_policy_metrics,
    _filter_rows_in_failed_csv,
    _filter_rows_not_in_completed_csv,
    _load_iteration_snapshots,
    _row_number_from_csv,
    _row_slice,
    _row_to_csv,
    _safe_int,
)
from config import DeployConfig, DeployTarget
from tools.deploy_validator import validate_deployment
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder

from baselines.drivers import DRIVERS, get_driver
from baselines.drivers.base import HarnessDriver
from baselines.evaluation import (
    STATE_FILENAME,
    ScenarioConfig,
    _sha,
    template_filename,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Extra columns beyond CSV_RESULT_FIELDS. The first 18 columns stay identical
# and in order, so --exclude-completed-csv and the aggregation scripts keep
# working across both runners.
BASELINE_CSV_EXTRA = [
    "harness",
    "harness_model",
    "harness_turns",
    "cost_usd_openrouter",
    "iteration_cap_reached",
    "submitted_by_harness",
    "submitted_matched_validated",
    "final_verdict_source",
    "harness_stalled",
    "harness_stall_retries_used",
    "harness_stalled_attempt_run_ids",
    "thinking_tokens_total",
]
BASELINE_CSV_FIELDS = CSV_RESULT_FIELDS + BASELINE_CSV_EXTRA

# Consecutive scenarios that never reached the model before the run aborts.
DEAD_ROW_ABORT_THRESHOLD = 3

# Console warning threshold for thinking_tokens_total. Not a hard cutoff —
# there is no evidence a "normal" amount exists, only that the one confirmed
# DSML-leak-despite---max-thinking-tokens-0 case burned 10,935. Picked low
# enough to flag that case and similar ones without being noisy on ordinary
# reasoning.
THINKING_TOKENS_WARN_THRESHOLD = 3000


@dataclass
class BaselineConfig:
    """Structurally satisfies benchmark_common.SummaryConfig."""

    dataset_path: Path
    output_dir: Path
    start_row: int
    max_rows: int | None
    rows: list[int] | None
    exclude_completed_csv: Path | None
    retry_errors: bool
    max_iterations: int
    iac_type: str
    deploy_target: str
    model: str | None
    harness: str
    harness_model: str
    base_url: str | None
    api_key_env: str
    native_auth: bool
    scenario_timeout: int
    max_stall_retries: int
    harness_effort: str | None
    max_thinking_tokens: int | None
    use_retry_proxy: bool
    retry_proxy_max_retries: int
    sleep_between_rows: float
    runs_dir: Path
    keep_workspace: bool

    # Present only to satisfy SummaryConfig; the harness owns provider routing.
    provider: str = "harness"
    openrouter_provider_only: str | None = None
    openrouter_min_quantization: str | None = None
    openrouter_reasoning_effort: str | None = None
    openrouter_reasoning_max_tokens: int | None = None
    skip_security: bool = False

    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    # Set by main() once the retry proxy is up: the URL the harness
    # subprocess actually connects to (http://127.0.0.1:<port>). base_url
    # itself is left untouched so summary.json keeps reporting the real
    # upstream (https://openrouter.ai/api) rather than an ephemeral local
    # port that means nothing once the run ends.
    effective_base_url: str | None = None


# ---------------------------------------------------------------------------
# OpenRouter pricing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Retry proxy — see baselines/retry_proxy.py for the full rationale.
#
# Root cause of the stalled-session failure mode (confirmed by a controlled
# 15-call burst against the raw OpenRouter endpoint): deepseek-v4-flash is
# load-balanced across many upstream providers essentially at random per
# request, and a small fraction of draws (any provider, not one specific
# backend — confirmed across multiple runs) burn their entire output budget
# on a thinking block and return zero visible content. One shared local
# proxy is started for the whole benchmark run (not per scenario — it is
# stateless and cheap), and ANTHROPIC_BASE_URL is pointed at it instead of
# directly at OpenRouter. Every scenario's requests flow through it
# transparently; Claude Code and the rest of the pipeline are unaware it
# exists.
# ---------------------------------------------------------------------------


def start_retry_proxy(max_retries: int, log_path: Path) -> tuple[subprocess.Popen, int]:
    """Launch baselines/retry_proxy.py and return (process, bound_port).

    Reads the port off the proxy's stdout (it prints exactly one line: the
    port number, once bound) rather than guessing a free port ourselves,
    which would race with the proxy's own bind.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "baselines.retry_proxy",
         "--port", "0", "--max-retries", str(max_retries), "--log-file", str(log_path)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    port_line = proc.stdout.readline().strip()
    if not port_line.isdigit():
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.terminate()
        raise RuntimeError(f"retry_proxy failed to start: {stderr or port_line}")
    return proc, int(port_line)


def stop_retry_proxy(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def fetch_openrouter_pricing() -> dict[str, dict[str, float]]:
    """Map model id -> per-token prompt/completion price.

    Claude Code prices every run with Anthropic rates, so its total_cost_usd is
    wrong by a large factor for third-party models (observed ~60x for
    deepseek-v4-flash). Cost is therefore recomputed here from token counts and
    the provider's published rates; the harness-reported figure is discarded.
    """
    try:
        import requests

        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        resp.raise_for_status()
        pricing: dict[str, dict[str, float]] = {}
        for entry in resp.json().get("data", []):
            price = entry.get("pricing") or {}
            try:
                pricing[entry["id"]] = {
                    "prompt": float(price.get("prompt", 0) or 0),
                    "completion": float(price.get("completion", 0) or 0),
                }
            except (TypeError, ValueError):
                continue
        return pricing
    except Exception as exc:
        print(f"[Baseline] Could not fetch OpenRouter pricing ({exc}); cost will be null.")
        return {}


def compute_cost(model_usage: dict[str, Any], pricing: dict[str, dict[str, float]]) -> float | None:
    if not pricing or not model_usage:
        return None
    total = 0.0
    seen = False
    for model, usage in model_usage.items():
        rates = pricing.get(model) or pricing.get(model.lstrip("~"))
        if not rates:
            continue
        seen = True
        total += _safe_int(usage.get("inputTokens")) * rates["prompt"]
        total += _safe_int(usage.get("outputTokens")) * rates["completion"]
    return round(total, 6) if seen else None


# ---------------------------------------------------------------------------
# Harness invocation
#
# Generic across harnesses: everything harness-specific (the CLI
# invocation, its subprocess environment, its own config file, how to parse
# its output into the common telemetry shape) is delegated to a
# baselines.drivers.HarnessDriver — see baselines/drivers/ for
# ClaudeCodeDriver and OpenCodeDriver. This section owns only what stays the
# same regardless of which harness is running: subprocess spawning,
# streaming stdout to disk, and the timeout/process-group-kill logic.
# ---------------------------------------------------------------------------


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the harness's process group, then SIGKILL anything that survives.

    The group, not the process: every driver's harness spawns the MCP
    server as a child, and killing only the harness would leave a validator
    process holding the scenario's run directory open.
    """
    import signal

    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_harness(
    config: BaselineConfig,
    scenario: ScenarioConfig,
    prompt: str,
    workdir: Path,
    stream_path: Path,
    driver: HarnessDriver,
) -> dict[str, Any]:
    """Launch the harness (via driver) and return its parsed telemetry.

    Also writes two human-readable debug files alongside stream_path (see
    driver.write_debug_transcripts): harness_transcript.txt (the full
    conversation) and harness_tool_calls.txt (just the MCP/file tool calls
    and their results), plus harness_debug.log where the driver has a
    wire-level log to offer.
    """
    system_prompt = (REPO_ROOT / "baselines" / "prompts" / f"{config.iac_type}.md").read_text()

    # Isolated per-attempt directory for the harness's OWN installation
    # state (session files, auth cache, etc. — never the MCP server's
    # scenario state, which lives under ScenarioConfig.workdir separately).
    # What specifically gets redirected here, and why, is documented in
    # each driver's build_env.
    state_dir = workdir / f".{driver.name}_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = stream_path.with_name("harness_debug.log")

    cmd = driver.build_command(
        config=config,
        scenario=scenario,
        prompt=prompt,
        workdir=workdir,
        system_prompt=system_prompt,
        state_dir=state_dir,
        debug_log_path=debug_log_path,
    )

    stderr_path = stream_path.with_suffix(".stderr.txt")
    started = time.time()
    timed_out = False

    # Streamed straight to disk rather than buffered: a 30-iteration scenario
    # on a large template produces a stream far too big to hold in memory, and
    # a partial transcript is exactly what is wanted when a run is killed.
    #
    # start_new_session puts the harness in its own process group so that a
    # timeout can take down the MCP server it spawned as well. subprocess's own
    # timeout kills only the direct child, which over a 250-row sweep would
    # leave an orphaned validator process behind for every scenario that hung.
    resolved_workdir = str(workdir.resolve())
    # Popen's cwd= sets the process's real working directory, but leaves the
    # inherited PWD env var pointing at whatever directory this Python
    # process happened to be started from. That mismatch is invisible to
    # most tools (which call getcwd()) but broke OpenCode: it reads
    # process.env.PWD during startup and bootstrapped a second, conflicting
    # instance against the researcher's actual repo checkout instead of the
    # isolated workdir, which surfaced as an opaque "UnknownError" — one
    # that never reproduced when testing manually from an interactive shell,
    # because `cd`ing there keeps PWD in sync automatically.
    env = driver.build_env(config, state_dir)
    env["PWD"] = resolved_workdir

    with stream_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.Popen(
            cmd,
            cwd=resolved_workdir,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=config.scenario_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            _terminate_process_group(proc)
            err.write(f"\n[Baseline] Timed out after {config.scenario_timeout}s\n")

    stdout = stream_path.read_text(encoding="utf-8", errors="replace")
    if not stderr_path.read_text(encoding="utf-8", errors="replace").strip():
        stderr_path.unlink(missing_ok=True)

    driver.write_debug_transcripts(
        stdout,
        run_id=scenario.run_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        harness_model_alias=config.harness_model,
        routed_model=config.model,
        state_dir=state_dir,
        debug_log_path=debug_log_path,
        transcript_path=stream_path.with_name("harness_transcript.txt"),
        tool_calls_path=stream_path.with_name("harness_tool_calls.txt"),
    )

    # state_dir is the harness's own installation bookkeeping (session DB,
    # auth cache, and — for OpenCode specifically — a full npm install of
    # its plugin dependencies under .opencode_state/config, tens of MB per
    # scenario). Nothing in it is scenario evidence; anything worth keeping
    # has already been copied out by write_debug_transcripts above (e.g.
    # OpenCode's internal log -> harness_debug.log). Discarded unconditionally,
    # independent of --keep-workspace: that flag means "keep this scenario's
    # own artifacts (template.yaml, etc.) for inspection," not "keep the
    # harness's internal plumbing too" — leaving it in made --keep-workspace
    # runs balloon to ~60-70MB/row and broke a plain file copy of the run
    # directory (large binary/db files silently dropped mid-copy).
    shutil.rmtree(state_dir, ignore_errors=True)

    return driver.parse_stream(
        stdout,
        returncode=returncode,
        timed_out=timed_out,
        duration=round(time.time() - started, 3),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def finalize_scenario(
    config: BaselineConfig,
    scenario: ScenarioConfig,
    workdir: Path,
    recorder: ResearchRecorder,
) -> dict[str, Any]:
    """Recompute the verdict from the artifact the harness left behind.

    Four cases, in order:
      1. Nothing was ever written        -> fail, no template.
      2. Artifact matches its last       -> reuse the recorded results.
         validation, and deploy either
         ran or is not required
      3. Artifact matches, static passed,
         but deploy never ran            -> deploy it now (IaCGOD always
                                            deploys when static passes, so
                                            skipping it would flatter the
                                            baseline).
      4. Artifact never validated in     -> validate it now, and deploy if
         this exact form                    static passes.

    Re-validation here never increments iterations_used: it measures the
    artifact, it is not a generate->validate cycle the harness performed.
    """
    state_path = recorder.output_dir / STATE_FILENAME
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}

    submitted = state.get("submitted") or {}
    candidate = Path(submitted["file_path"]) if submitted.get("file_path") else (
        workdir / template_filename(config.iac_type)
    )
    if not candidate.is_absolute():
        candidate = workdir / candidate

    if not candidate.exists() or not candidate.read_text(encoding="utf-8").strip():
        return {
            "template": "",
            "validation_results": [],
            "deploy_result": None,
            "passed": False,
            "verdict_source": "no_template_produced",
            "state": state,
        }

    template = candidate.read_text(encoding="utf-8")
    sha = _sha(template)
    deploy_required = config.deploy_target != "none"

    matches = sha == state.get("last_validated_sha")
    static_passed = bool(state.get("last_static_passed")) and matches
    results = state.get("last_validation_results", []) if matches else []
    deploy_result = state.get("last_deploy_result") if matches else None

    verdict_source = "recorded"

    if not matches:
        verdict_source = "revalidated"
        results, _all, _dep = run_all_validators(
            template,
            iac_type=config.iac_type,
            deploy_config=DeployConfig(target=DeployTarget.NONE),
        )
        static_passed = all(r["passed"] for r in results)
        deploy_result = None

    if deploy_required and static_passed and deploy_result is None:
        if verdict_source == "recorded":
            verdict_source = "deployed_at_finalize"
        deploy_result = dict(
            validate_deployment(
                template,
                deploy_config=DeployConfig(target=DeployTarget(config.deploy_target)),
                iac_type=config.iac_type,
            )
        )
        recorder.record_deployment_log(
            # One past the harness's last iteration: deployment logs are keyed
            # by iteration number, and a finalization deploy must not clobber
            # the log of a deploy the harness itself performed at that index.
            iteration=_safe_int(state.get("iteration_count")) + 1,
            iac_type=config.iac_type,
            target=deploy_result.get("target", config.deploy_target),
            deployment_logs=deploy_result.get("deployment_logs", []),
            passed=bool(deploy_result.get("passed")),
            duration_seconds=float(deploy_result.get("duration_seconds", 0.0) or 0.0),
            failed_resources=deploy_result.get("failed_resources", []),
        )

    deploy_ok = True
    if deploy_required:
        deploy_ok = bool(deploy_result and deploy_result.get("passed"))

    return {
        "template": template,
        "validation_results": results,
        "deploy_result": deploy_result,
        "passed": bool(static_passed and deploy_ok),
        "verdict_source": verdict_source,
        "state": state,
    }


def _empty_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }


def _run_one_attempt(
    config: BaselineConfig, prompt: str, driver: HarnessDriver
) -> tuple[str, ResearchRecorder, Path, ScenarioConfig, dict[str, Any], dict[str, Any]]:
    """Launch one fresh harness process for one scenario attempt.

    Fully self-contained: its own run_id, ResearchRecorder, and workspace.
    Called more than once by run_scenario when an attempt stalls (see
    driver.parse_stream's `stalled` detection) — each retry is a clean
    restart, not a resumption, since a stalled session's own context is what
    produced the stall.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    recorder = ResearchRecorder(run_id=run_id, output_dir=str(config.runs_dir))

    # Absolute: the harness runs with cwd set to this directory, so any
    # relative path handed to it (--mcp-config, the template path in tool
    # args) would be re-resolved against the workspace and doubled.
    workdir = (recorder.output_dir / "workspace").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    scenario = ScenarioConfig(
        run_id=run_id,
        iac_type=config.iac_type,
        workdir=workdir,
        runs_dir=config.runs_dir,
        max_iterations=config.max_iterations,
        deploy_target=config.deploy_target,
        user_request=prompt,
    )

    telemetry = run_harness(
        config,
        scenario,
        prompt,
        workdir,
        recorder.output_dir / "harness_stream.jsonl",
        driver,
    )
    final = finalize_scenario(config, scenario, workdir, recorder)
    return run_id, recorder, workdir, scenario, telemetry, final


def run_scenario(
    config: BaselineConfig,
    row: dict[str, str],
    row_number: int,
    prompt: str,
    driver: HarnessDriver,
) -> dict[str, Any]:
    started = time.time()

    # Retry loop for stalled sessions (see parse_harness_stream): Claude Code
    # can report is_error=false / subtype="success" after its own built-in
    # empty-turn recovery also comes back empty — observed with
    # deepseek-v4-flash returning genuinely empty completions, or emitting its
    # native tool-call syntax as literal text inside a "thinking" block that
    # the OpenRouter shim never parses into a tool_use. Either way the harness
    # never got a real turn, so this mirrors the empty-completion retry
    # agents/llm_client.py already gives the multi-agent path — a stalled
    # attempt is not treated as an evaluated result unless every retry also
    # stalls. Each attempt is a fully independent process; earlier stalled
    # attempts' artifacts stay on disk under their own run_id for inspection
    # but are not what gets scored.
    #
    # `telemetry["stalled"]` (did Claude Code's own session end in an empty
    # turn) and `final["passed"]` (does the artifact on disk actually
    # validate and deploy, from finalize_scenario re-checking it directly)
    # are independent — a session can stall on a LATER turn (e.g. the model
    # failing to call submit_template) after already producing a genuinely
    # passing, deployed template on an EARLIER turn. Observed directly: a
    # session that deployed a real 12-resource stack successfully, then
    # stalled trying to acknowledge it. Retrying on `stalled` alone would
    # discard that completed, verified work and restart the whole scenario
    # from a blank template — each attempt is a fresh process with no memory
    # of the last, so nothing carries over. `final["passed"]` is checked
    # first: once the artifact is confirmed working, stop, regardless of
    # what the session itself did afterward.
    stalled_run_ids: list[str] = []
    attempt = 1
    while True:
        run_id, recorder, workdir, scenario, telemetry, final = _run_one_attempt(
            config, prompt, driver
        )
        if final["passed"] or not telemetry["stalled"] or attempt > config.max_stall_retries:
            if telemetry["stalled"] and final["passed"]:
                print(
                    f"[Baseline] row {row_number}: attempt {attempt} stalled after "
                    f"already passing (deployed/validated) — keeping this result, "
                    f"not retrying"
                )
            break
        stalled_run_ids.append(run_id)
        print(
            f"[Baseline] row {row_number}: attempt {attempt} stalled "
            f"(Claude Code's empty-turn recovery also came back empty) — retrying"
        )
        attempt += 1

    state = final["state"]

    iterations_used = _safe_int(state.get("iteration_count"))
    tokens = telemetry["token_usage"]
    policy_metrics = _extract_policy_metrics(final["validation_results"])
    iteration_records = _extract_iteration_records(
        _load_iteration_snapshots(recorder.output_dir)
    )

    recorder.save_final_report(
        {
            "run_id": run_id,
            "user_request": prompt,
            "current_iteration": iterations_used,
            "validation_passed": final["passed"],
            "objectives": [],
            "final_template": final["template"],
            "remediation_history": [],
            "llm_call_log": telemetry["llm_call_log"],
            "validation_results": final["validation_results"],
            "deploy_validation_result": final["deploy_result"],
        }
    )

    if not config.keep_workspace:
        shutil.rmtree(workdir, ignore_errors=True)

    status = "ok"
    error_message = None
    if telemetry["timed_out"]:
        status, error_message = "harness_timeout", "Harness exceeded --scenario-timeout"
    elif telemetry["stalled"] and not final["passed"]:
        # A stall on an attempt whose artifact already passed is NOT this
        # status — see the retry loop above, which accepts that attempt
        # instead of discarding it. Only a stall with no passing artifact
        # behind it is a genuine "the harness never completed a real
        # attempt" outcome.
        status = "harness_stalled"
        error_message = (
            f"Harness never produced a usable turn after "
            f"{len(stalled_run_ids) + 1} attempt(s) — its own empty-turn "
            f"recovery also came back empty each time. Not a validation "
            f"failure on the merits; the model/shim pairing did not "
            f"complete a genuine attempt."
        )
    elif telemetry["is_error"]:
        status = "harness_error"
        error_message = f"Harness exited rc={telemetry['returncode']} subtype={telemetry['subtype']}"

    submitted = state.get("submitted") or {}
    return {
        "row_number": row_number,
        "ground_truth_path": row.get("ground_truth_path"),
        "prompt": prompt,
        "run_id": run_id,
        "status": status,
        "error_message": error_message,
        "error_traceback": None,
        "final_validation_passed": final["passed"],
        "iterations_used": iterations_used,
        "llm_calls_total": len(telemetry["llm_call_log"]),
        "token_usage": tokens,
        "iteration_records": iteration_records,
        "validation_results_final": final["validation_results"],
        "remediation_history": [],
        "policy_metrics": policy_metrics,
        "duration_seconds": round(time.time() - started, 3),
        # baseline-specific
        "harness": config.harness,
        "harness_model": config.model or config.harness_model,
        "harness_turns": telemetry["num_turns"],
        "token_usage_source": telemetry["token_usage_source"],
        "harness_session_id": telemetry["session_id"],
        "model_usage": telemetry["model_usage"],
        "cost_usd_openrouter": compute_cost(telemetry["model_usage"], config.pricing),
        "harness_reported_cost_usd": telemetry["harness_reported_cost_usd"],
        "iteration_cap_reached": bool(state.get("cap_reached")),
        "submitted_by_harness": bool(submitted),
        "submitted_matched_validated": bool(submitted.get("matches_last_validated")),
        "final_verdict_source": final["verdict_source"],
        "harness_stalled": bool(telemetry["stalled"]),
        "harness_stall_retries_used": len(stalled_run_ids),
        "harness_stalled_attempt_run_ids": ",".join(stalled_run_ids) or None,
        "thinking_tokens_total": telemetry["thinking_tokens_total"],
    }


def _baseline_csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = _row_to_csv(payload)
    for key in BASELINE_CSV_EXTRA:
        row[key] = payload.get(key)
    return row


def _append_baseline_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=BASELINE_CSV_FIELDS).writerow(
            _baseline_csv_row(payload)
        )


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    # One driver instance for the whole run, not per scenario: drivers are
    # stateless (every method takes whatever scenario-specific context it
    # needs as arguments — see baselines/drivers/base.py), so there is
    # nothing to gain from re-instantiating per row, and reusing one avoids
    # ever accidentally relying on instance state that would leak between
    # scenarios if the driver were ever made stateful later.
    driver = get_driver(config.harness)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.json"
    jsonl_path = config.output_dir / "results.jsonl"
    csv_path = config.output_dir / "results.csv"

    jsonl_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=BASELINE_CSV_FIELDS).writeheader()

    with config.dataset_path.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    if config.rows:
        requested = set(config.rows)
        selected = [
            r
            for r in all_rows
            if (n := _row_number_from_csv(r)) is not None and n in requested
        ]
    else:
        selected = _row_slice(all_rows, config.start_row, config.max_rows)

    selected_row_count = len(selected)
    filtered_row_count = 0
    if config.exclude_completed_csv is not None:
        if config.retry_errors:
            selected, filtered_row_count = _filter_rows_in_failed_csv(
                selected, config.exclude_completed_csv,
                match_ground_truth_path=not bool(config.rows),
            )
        else:
            selected, filtered_row_count = _filter_rows_not_in_completed_csv(
                selected, config.exclude_completed_csv,
                match_ground_truth_path=not bool(config.rows),
            )

    started_at = datetime.now().isoformat()
    started_ts = time.time()

    rows_out: list[dict[str, Any]] = []
    consecutive_dead_rows = 0
    pass_count = pass_at_1_count = 0
    total_iterations = total_llm_calls = 0
    total_policy = total_passed_policy = total_failed_policy = total_filtered_failed = 0
    scenario_ppr_sum = scenario_unfiltered_sum = 0.0
    scenario_ppr_count = runtime_error_runs = 0
    stalled_count = 0
    aggregate_tokens = _empty_tokens()
    total_cost = 0.0

    print(f"\n{'=' * 80}")
    print(
        f"Harness baseline | harness={config.harness} | model={config.model or 'native'} "
        f"| iac_type={config.iac_type} | rows={len(selected)} | deploy={config.deploy_target}"
    )
    print(f"{'=' * 80}")

    def write_summary(attempted: int) -> dict[str, Any]:
        summary = _build_summary(
            config=config,
            selected_row_count=selected_row_count,
            filtered_row_count=filtered_row_count,
            attempted=attempted,
            pass_count=pass_count,
            pass_at_1_count=pass_at_1_count,
            total_iterations=total_iterations,
            total_llm_calls=total_llm_calls,
            aggregate_tokens=aggregate_tokens,
            total_policy_count=total_policy,
            total_passed_policy_count=total_passed_policy,
            total_failed_policy_count=total_failed_policy,
            total_filtered_failed_policy_count=total_filtered_failed,
            scenario_ppr_sum=scenario_ppr_sum,
            scenario_unfiltered_compliance_sum=scenario_unfiltered_sum,
            scenario_ppr_count=scenario_ppr_count,
            runtime_error_runs=runtime_error_runs,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            elapsed=round(time.time() - started_ts, 3),
        )
        # rows_stalled: sessions where Claude Code's own empty-turn recovery
        # also came back empty on every attempt (see README §6). These are
        # not validation failures on the merits — the harness never completed
        # a genuine attempt — so pass_rate (over ALL evaluated rows, matching
        # benchmark.py's semantics for direct comparability) is reported
        # alongside pass_rate_excl_stalled, computed over the rows that
        # actually got a fair attempt. A large gap between the two is itself
        # a finding: it says how much of the observed failure rate is
        # model/shim instability rather than the model's IaC generation
        # ability.
        evaluated = summary.get("rows_evaluated", 0)
        non_stalled = max(evaluated - stalled_count, 0)
        summary.update(
            {
                "harness": config.harness,
                "harness_model": config.model or config.harness_model,
                "harness_alias": config.harness_model,
                "base_url": None if config.native_auth else config.base_url,
                "retry_proxy_used": config.effective_base_url is not None,
                "retry_proxy_max_retries": config.retry_proxy_max_retries
                if config.effective_base_url is not None
                else None,
                "total_cost_usd_openrouter": round(total_cost, 6),
                "runs_dir": str(config.runs_dir),
                "rows_stalled": stalled_count,
                "pass_rate_excl_stalled": (pass_count / non_stalled) if non_stalled else None,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    write_summary(0)

    for i, row in enumerate(selected, start=1):
        row_number = _safe_int(row.get("row_number"), config.start_row + i - 1)
        prompt = (row.get("prompt") or "").strip()
        print(f"\n[Baseline] ({i}/{len(selected)}) row {row_number} [{config.iac_type}]")

        if not prompt:
            payload: dict[str, Any] = {
                "row_number": row_number,
                "ground_truth_path": row.get("ground_truth_path"),
                "prompt": prompt,
                "run_id": None,
                "status": "skipped_empty_prompt",
                "error_message": "Prompt is empty",
                "final_validation_passed": False,
                "iterations_used": 0,
                "llm_calls_total": 0,
                "token_usage": _empty_tokens(),
                "iteration_records": [],
                "validation_results_final": [],
                "remediation_history": [],
                "policy_metrics": _extract_policy_metrics([]),
                "duration_seconds": 0.0,
            }
        else:
            row_started = time.time()
            try:
                payload = run_scenario(config, row, row_number, prompt, driver)
            except Exception as exc:
                runtime_error_runs += 1
                payload = {
                    "row_number": row_number,
                    "ground_truth_path": row.get("ground_truth_path"),
                    "prompt": prompt,
                    "status": "runtime_error",
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(),
                    # run_scenario's own internal timer never returns when it
                    # raises before completing — this outer timer is the only
                    # way to still capture how long the scenario ran before
                    # crashing (e.g. a 20-minute stall vs. an instant
                    # failure), matching benchmark.py's equivalent path.
                    "duration_seconds": round(time.time() - row_started, 3),
                }
                print(f"[Baseline] Row {row_number} failed: {exc}")
                print(traceback.format_exc())

        # Aggregate only rows that actually produced a scored run. Skipped
        # (empty prompt) and runtime-error rows are recorded but excluded, so
        # they cannot contribute a spurious 1.0 to avg_ppr — matching how
        # benchmark.py leaves its counters untouched on those paths. A harness
        # timeout or non-zero exit still yields a real artifact and verdict, so
        # it counts as an evaluated failure.
        if payload.get("run_id") and payload.get("status") != "runtime_error":
            if payload.get("status") == "harness_stalled":
                stalled_count += 1
            if payload.get("final_validation_passed"):
                pass_count += 1
                if _safe_int(payload.get("iterations_used")) == 1:
                    pass_at_1_count += 1
            total_iterations += _safe_int(payload.get("iterations_used"))
            total_llm_calls += _safe_int(payload.get("llm_calls_total"))
            for key in aggregate_tokens:
                aggregate_tokens[key] += _safe_int(
                    (payload.get("token_usage") or {}).get(key)
                )
            pm = payload.get("policy_metrics") or {}
            total_policy += _safe_int(pm.get("total_policies"))
            total_passed_policy += _safe_int(pm.get("passed_policies"))
            total_failed_policy += _safe_int(pm.get("failed_policies_all_severity"))
            total_filtered_failed += _safe_int(pm.get("filtered_failed_policies"))
            scenario_ppr_sum += float(pm.get("scenario_policy_pass_rate", 0.0) or 0.0)
            scenario_unfiltered_sum += float(
                pm.get("unfiltered_compliance_rate", 0.0) or 0.0
            )
            scenario_ppr_count += 1
            total_cost += float(payload.get("cost_usd_openrouter") or 0.0)

            if payload.get("status") == "harness_stalled":
                outcome = "STALLED (not a validation failure — see README §6)"
            elif payload.get("final_validation_passed"):
                outcome = "PASS"
            else:
                outcome = "FAIL"
            thinking_total = _safe_int(payload.get("thinking_tokens_total"))
            # Surfaced inline, not just in the CSV: this is the number that
            # would have caught the --max-thinking-tokens gap immediately
            # instead of requiring a manual grep through thousands of raw
            # thinking_tokens ping lines in harness_stream.jsonl. See
            # parse_harness_stream's docstring — --max-thinking-tokens 0 does
            # NOT reliably prevent this on long, cache-heavy turns.
            thinking_flag = (
                f" | ⚠ thinking={thinking_total}" if thinking_total > THINKING_TOKENS_WARN_THRESHOLD else ""
            )
            print(
                f"[Baseline] row {row_number}: {outcome} "
                f"| iters={payload.get('iterations_used')} "
                f"| calls={payload.get('llm_calls_total')} "
                f"| verdict={payload.get('final_verdict_source')} "
                f"| ${payload.get('cost_usd_openrouter')}"
                f"{thinking_flag}"
            )

        # Fail fast on a misconfiguration. A scenario that makes zero LLM
        # calls and produces no template did not fail on its merits — the
        # harness never really ran (bad model routing, an unreadable
        # --mcp-config, a missing CLI). Left unchecked this churns through the
        # whole dataset in seconds and yields a CSV of uniform false negatives
        # that looks like a result.
        harness_never_ran = (
            _safe_int(payload.get("llm_calls_total")) == 0
            and payload.get("final_verdict_source") == "no_template_produced"
            and payload.get("status") != "skipped_empty_prompt"
        )
        consecutive_dead_rows = consecutive_dead_rows + 1 if harness_never_ran else 0

        rows_out.append(payload)
        _append_jsonl(jsonl_path, payload)
        _append_baseline_csv(csv_path, payload)
        write_summary(len(rows_out))

        if consecutive_dead_rows >= DEAD_ROW_ABORT_THRESHOLD:
            stderr_hint = ""
            if payload.get("run_id"):
                stderr_hint = (
                    f"\nInspect: {config.runs_dir / payload['run_id'] / 'harness_stream.stderr.txt'}"
                )
            raise SystemExit(
                f"\n[Baseline] Aborting: {consecutive_dead_rows} consecutive scenarios "
                f"made zero LLM calls and produced no template. The harness is not "
                f"running — this is a setup problem, not a benchmark result."
                f"{stderr_hint}\n"
                f"Last error: {payload.get('error_message')}\n"
                f"Partial results kept in {config.output_dir}"
            )

        if config.sleep_between_rows and i < len(selected):
            time.sleep(config.sleep_between_rows)

    summary = write_summary(len(rows_out))
    (config.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows_out}, indent=2), encoding="utf-8"
    )

    print(f"\n[Baseline] Finished in {summary['duration_seconds']}s -> {config.output_dir}")
    excl = summary.get("pass_rate_excl_stalled")
    excl_str = f"{excl:.3f}" if excl is not None else "n/a"
    print(
        f"[Baseline] pass_rate={summary['pass_rate']:.3f} "
        f"pass@1={summary['pass_at_1']:.3f} cost=${total_cost:.4f}"
    )
    if stalled_count:
        print(
            f"[Baseline] {stalled_count} row(s) stalled (harness never completed a "
            f"real attempt, not a validation failure — see README §6). "
            f"pass_rate_excl_stalled={excl_str}"
        )
    return {"summary": summary, "rows": rows_out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the benchmark through a single-agent coding harness."
    )
    p.add_argument("--iac-type", choices=["cloudformation", "terraform"], default="cloudformation")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--rows", type=str, default=None,
                   help="Comma-separated row_number values to run (e.g. 1,6,7)")
    p.add_argument("--exclude-completed-csv", type=Path, default=None)
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Requires --exclude-completed-csv. Instead of excluding rows found in "
            "that CSV, run ONLY the rows that did not get a clean pass there: "
            "final_validation_passed was False, OR status was anything other "
            "than 'ok' (harness_stalled, harness_timeout, harness_error, "
            "runtime_error, skipped_empty_prompt, ...) (row_number match, or "
            "ground_truth_path match when --rows is not used)."
        ),
    )
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--deploy-target", choices=["none", "localstack", "aws"], default="aws")

    p.add_argument("--harness", choices=sorted(DRIVERS), default="claude_code",
                   help="Which coding harness drives the loop. claude_code: Anthropic-"
                        "compat routing via ANTHROPIC_BASE_URL (see baselines/README.md "
                        "for the OpenRouter provider-roulette issue this hits with "
                        "third-party models). opencode: native multi-provider routing, "
                        "no Anthropic-shape translation involved for OpenRouter models — "
                        "see baselines/drivers/opencode.py for what has and has not been "
                        "verified.")
    p.add_argument("--model", type=str, default=None,
                   help="Model id the harness routes to, e.g. deepseek/deepseek-v4-flash. "
                        "Mapped onto every Claude Code model alias.")
    p.add_argument("--harness-model", type=str, default="sonnet",
                   help="Alias passed to `claude --model` (default: sonnet)")
    p.add_argument("--base-url", type=str, default="https://openrouter.ai/api")
    p.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    p.add_argument("--native-auth", action="store_true",
                   help="Use Claude Code's own auth and models; ignore --model/--base-url.")

    p.add_argument("--scenario-timeout", type=int, default=1800,
                   help="Seconds before a scenario's harness process is killed (default 1800)")
    p.add_argument("--max-stall-retries", type=int, default=1,
                   help="Extra fresh attempts when a session stalls: Claude Code's own "
                        "built-in empty-turn recovery (a synthetic nudge) also came back "
                        "empty, so the harness never got a real turn. Observed with "
                        "deepseek-v4-flash returning genuinely empty completions or leaking "
                        "its tool-call syntax into a thinking block instead of a structured "
                        "tool_use, which the OpenRouter shim does not parse. Each retry is a "
                        "full fresh process (default 1 retry, i.e. 2 attempts total).")
    p.add_argument("--harness-effort", type=str, default=None,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Passed through as `claude --effort <level>`. EXPERIMENTAL and "
                        "unverified for third-party models routed through the OpenRouter "
                        "shim: offered as a lever to try against the stalled-session "
                        "failure mode (README §6), whose stalls correlate with unusually "
                        "long reasoning chains in the observed logs, on the hypothesis "
                        "that a lower effort level may shorten them. Not confirmed to help.")
    p.add_argument("--max-thinking-tokens", type=int, default=None,
                   help="Sets MAX_THINKING_TOKENS for the harness subprocess. Pass 0 to ask "
                        "the harness to disable extended thinking. PARTIAL MITIGATION ONLY, "
                        "NOT a fix: reliably honoured on short, fresh sessions (0 cache "
                        "reads; 14/14 clean in one batch) but overridden by the model on "
                        "long, cache-heavy mid-repair-loop turns — one confirmed case burned "
                        "10,935 thinking tokens with this set to 0 and cache_read_input_tokens "
                        "in the tens of thousands, with the DSML leak recurring inside that "
                        "thinking block (README §6: deepseek-v4-flash's native tool-call "
                        "syntax leaking as literal text inside thinking instead of a "
                        "structured tool_use, which the OpenRouter shim does not parse). "
                        "Likely deepseek-v4-flash's own always-reasoning behaviour taking "
                        "precedence over the request once it judges a turn hard enough — not "
                        "something this flag can force. Watch the thinking_tokens_total CSV "
                        "column (and the inline ⚠ warning above "
                        f"{THINKING_TOKENS_WARN_THRESHOLD} tokens) to see when it did not "
                        "hold, rather than assuming 0 means disabled.")
    p.add_argument("--no-retry-proxy", action="store_true",
                   help=(
                       "Disable the local retry proxy (see baselines/retry_proxy.py). "
                       "On by default: it sits between the harness and OpenRouter and "
                       "transparently re-issues a request when the response comes back "
                       "with no real content (no tool_use, no non-empty text) — the "
                       "confirmed stalled-session signature, root-caused to OpenRouter's "
                       "per-request provider load-balancing (any of several upstream "
                       "providers can return this on a given draw; ~93%% of raw calls "
                       "succeed in one measured burst, so a fresh retry resolves most "
                       "cases before the harness ever sees a failure). Ignored under "
                       "--native-auth (routes to Claude's own API, not OpenRouter)."
                   ))
    p.add_argument("--retry-proxy-max-retries", type=int, default=3,
                   help="Retries per API call inside the proxy, before it gives up and "
                        "returns the last (empty) response as-is. Distinct from "
                        "--max-stall-retries, which restarts the whole harness process; "
                        "this retries a single request and is much cheaper.")
    p.add_argument("--sleep-between-rows", type=float, default=0.0)
    p.add_argument("--keep-workspace", action="store_true",
                   help="Keep runs/<run_id>/workspace instead of deleting it after scoring")
    p.add_argument("--no-pricing", action="store_true",
                   help="Skip the OpenRouter pricing fetch; cost columns will be null.")

    args = p.parse_args()
    if args.retry_errors and args.exclude_completed_csv is None:
        p.error("--retry-errors requires --exclude-completed-csv")
    if not args.native_auth and not args.model:
        p.error("--model is required unless --native-auth is given")
    return args


def main() -> None:
    args = parse_args()

    if args.dataset is not None:
        dataset_path = args.dataset
    elif args.iac_type == "terraform":
        dataset_path = Path("data") / "tf_eval_benchmark_real_aws.csv"
    else:
        dataset_path = Path("data") / "cfn_eval_benchmark_real_aws.csv"

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_runs") / f"baseline_{args.harness}_{args.iac_type}_{ts}"

    rows = None
    if args.rows:
        rows = [int(r.strip()) for r in args.rows.split(",") if r.strip().isdigit()]

    harness_binary = {"claude_code": "claude", "opencode": "opencode"}.get(args.harness)
    if harness_binary and shutil.which(harness_binary) is None:
        raise SystemExit(
            f"The `{harness_binary}` CLI was not found on PATH. Install it before "
            f"running --harness {args.harness}."
        )

    config = BaselineConfig(
        dataset_path=dataset_path,
        output_dir=output_dir.resolve(),
        start_row=args.start_row,
        max_rows=args.max_rows,
        rows=rows,
        exclude_completed_csv=args.exclude_completed_csv,
        retry_errors=args.retry_errors,
        max_iterations=args.max_iterations,
        iac_type=args.iac_type,
        deploy_target=args.deploy_target,
        model=args.model,
        harness=args.harness,
        harness_model=args.harness_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        native_auth=args.native_auth,
        scenario_timeout=args.scenario_timeout,
        max_stall_retries=args.max_stall_retries,
        harness_effort=args.harness_effort,
        max_thinking_tokens=args.max_thinking_tokens,
        use_retry_proxy=not args.no_retry_proxy,
        retry_proxy_max_retries=args.retry_proxy_max_retries,
        sleep_between_rows=args.sleep_between_rows,
        runs_dir=args.runs_dir.resolve(),
        keep_workspace=args.keep_workspace,
        pricing={} if args.no_pricing else fetch_openrouter_pricing(),
    )

    proxy_proc = None
    if config.use_retry_proxy and not config.native_auth:
        config.runs_dir.mkdir(parents=True, exist_ok=True)
        proxy_log = config.runs_dir / f"retry_proxy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        proxy_proc, proxy_port = start_retry_proxy(config.retry_proxy_max_retries, proxy_log)
        print(f"[Baseline] retry proxy listening on 127.0.0.1:{proxy_port} (log: {proxy_log})")
        # The real OpenRouter base_url this run would otherwise have used is
        # what the proxy forwards to (baselines/retry_proxy.py:UPSTREAM_BASE)
        # — overriding it here, rather than changing UPSTREAM_BASE, keeps the
        # proxy a fixed, reusable piece rather than one wired to this run's
        # particular --base-url.
        config.effective_base_url = f"http://127.0.0.1:{proxy_port}"

    try:
        run_baseline(config)
    finally:
        if proxy_proc is not None:
            stop_retry_proxy(proxy_proc)


if __name__ == "__main__":
    import logging

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    main()
