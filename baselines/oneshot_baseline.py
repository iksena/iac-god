"""One-shot baseline runner.

The weakest control condition: a single LLM call per scenario, using the
original IaCGen first-generation prompt (baselines/oneshot_prompts.py), with
no repair loop at all. The generated template is then scored by exactly the
validators IaCGOD itself uses — run_all_validators() (static stages, then the
live deploy when static passes), the same call agents/validator.py makes.

    python -m baselines.oneshot_baseline \
        --iac-type cloudformation \
        --dataset data/cfn_eval_benchmark_real_aws.csv \
        --provider deepseek --model deepseek-v4-flash \
        --deploy-target aws

LLM configuration goes through config.configure_llm and agents/llm_client.py,
the same path the multi-agent pipeline uses, so --provider/--model and the
retry-on-empty-completion behaviour match benchmark.py.

One generate->validate cycle is exactly one iteration, so iterations_used is
1 for every scored row and pass@1 equals pass_rate. Output shape (results.csv
/ results.jsonl / summary.json and runs/<run_id>/ written by ResearchRecorder)
matches benchmark.py and harness_baseline.py so existing aggregation scripts
read all three identically.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import _build_client, _call_llm_with_history
from baselines.oneshot_prompts import build_messages, extract_template
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
    _token_totals,
)
from config import DEFAULT_CONFIG, DeployConfig, DeployTarget, configure_llm
from tools.deploy_cleanup import cleanup_scenario_resources
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder

# Extra columns beyond CSV_RESULT_FIELDS (the first 18 stay identical and in
# order, so --exclude-completed-csv and the aggregation scripts keep working).
ONESHOT_CSV_EXTRA = ["extraction_method"]
ONESHOT_CSV_FIELDS = CSV_RESULT_FIELDS + ONESHOT_CSV_EXTRA


@dataclass
class OneShotConfig:
    """Structurally satisfies benchmark_common.SummaryConfig."""

    dataset_path: Path
    output_dir: Path
    runs_dir: Path
    start_row: int
    max_rows: int | None
    rows: list[int] | None
    exclude_completed_csv: Path | None
    retry_errors: bool
    iac_type: str
    deploy_target: str
    provider: str
    model: str | None
    openrouter_provider_only: str | None
    openrouter_min_quantization: str | None
    openrouter_reasoning_effort: str | None
    openrouter_reasoning_max_tokens: int | None
    disable_reasoning: bool
    max_tokens: int | None
    no_max_tokens: bool
    skip_security: bool
    sleep_between_rows: float
    # Always one generate->validate cycle; present for SummaryConfig.
    max_iterations: int = 1


def _empty_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _oneshot_csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = _row_to_csv(payload)
    for key in ONESHOT_CSV_EXTRA:
        row[key] = payload.get(key)
    return row


def _append_oneshot_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=ONESHOT_CSV_FIELDS).writerow(_oneshot_csv_row(payload))


# ---------------------------------------------------------------------------
# One scenario
# ---------------------------------------------------------------------------


def run_scenario(
    config: OneShotConfig, row: dict[str, str], row_number: int, prompt: str
) -> dict[str, Any]:
    started = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    recorder = ResearchRecorder(run_id=run_id, output_dir=str(config.runs_dir))

    system, user_content = build_messages(config.iac_type, prompt)

    # llm_client retries transient failures / empty completions itself; an
    # exception here is a real failure and propagates to the row loop, which
    # records it as runtime_error (retryable via --retry-errors).
    client, model = _build_client()
    content, usage = _call_llm_with_history(
        client, model, system, [{"role": "user", "content": user_content}]
    )
    # An LLM backend (e.g. a CLI-wrapping proxy) can hand back an error message in the
    # completion slot. That is an infrastructure/safeguard event, not the model's answer:
    # scoring it would record a bogus "raw" template and a model FAIL. Raise instead so the
    # row is recorded as runtime_error (retryable via --retry-errors).
    if content.lstrip().startswith("API Error:"):
        raise RuntimeError(
            "LLM backend returned an error message instead of a completion "
            f"(not scored as a model answer): {content.strip()[:300]}"
        )
    template, extraction_method = extract_template(content, config.iac_type)

    llm_record = recorder.record_llm_call(
        state={"current_iteration": 1},
        agent="oneshot",
        model=usage.get("reported_model") or model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content,
        token_usage=usage,
    )
    llm_call_log = [llm_record]

    if template.strip():
        # The same call agents/validator.py makes: static stages, then the
        # live deploy only if static passed (and the target is not "none").
        results, passed, deploy_result = run_all_validators(
            template,
            iac_type=config.iac_type,
            deploy_config=DeployConfig(target=DeployTarget(config.deploy_target)),
            skip_security=config.skip_security,
        )
        if deploy_result["target"] != "skipped":
            recorder.record_deployment_log(
                iteration=1,
                iac_type=config.iac_type,
                target=deploy_result["target"],
                deployment_logs=deploy_result.get("deployment_logs", []),
                passed=bool(deploy_result["passed"]),
                duration_seconds=float(deploy_result.get("duration_seconds", 0.0) or 0.0),
                failed_resources=deploy_result.get("failed_resources", []),
            )
        iterations_used = 1
    else:
        # Nothing extractable: not a validation verdict on the merits.
        results, passed, deploy_result = [], False, None
        extraction_method = "no_template_produced"
        iterations_used = 0

    history = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": content},
    ]
    state = {
        "current_iteration": iterations_used or 1,
        "objectives": [],
        "iac_template": template,
        "validation_results": results,
        "validation_passed": bool(passed),
        "deploy_validation_result": deploy_result,
        "remediation_history": [],
        "engineer_history": history,
    }
    recorder.save_iteration_snapshot(state)
    recorder.save_final_report(
        {
            **state,
            "user_request": prompt,
            "final_template": template,
            "llm_call_log": llm_call_log,
        }
    )

    iteration_records = _extract_iteration_records(_load_iteration_snapshots(recorder.output_dir))
    return {
        "row_number": row_number,
        "ground_truth_path": row.get("ground_truth_path"),
        "prompt": prompt,
        "run_id": run_id,
        "status": "ok",
        "error_message": None,
        "error_traceback": None,
        "final_validation_passed": bool(passed),
        "iterations_used": iterations_used,
        "llm_calls_total": 1,
        "token_usage": _token_totals(llm_call_log),
        "iteration_records": iteration_records,
        "validation_results_final": results,
        "remediation_history": [],
        "policy_metrics": _extract_policy_metrics(results),
        "duration_seconds": round(time.time() - started, 3),
        "extraction_method": extraction_method,
    }


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_oneshot(config: OneShotConfig) -> dict[str, Any]:
    configure_llm(
        config.provider,
        config.model,
        openrouter_provider_only=config.openrouter_provider_only,
        openrouter_min_quantization=config.openrouter_min_quantization,
        openrouter_reasoning_effort=config.openrouter_reasoning_effort,
        openrouter_reasoning_max_tokens=config.openrouter_reasoning_max_tokens,
        disable_reasoning=config.disable_reasoning,
        max_tokens=config.max_tokens,
        no_max_tokens=config.no_max_tokens,
    )
    # Reflect the resolved model (provider defaults apply when --model is
    # unset) in summary.json rather than reporting null.
    config.model = DEFAULT_CONFIG.model

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.json"
    jsonl_path = config.output_dir / "results.jsonl"
    csv_path = config.output_dir / "results.csv"

    # Truncated at the start of every invocation: point --output-dir somewhere
    # new when resuming, and --exclude-completed-csv at the previous file.
    jsonl_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=ONESHOT_CSV_FIELDS).writeheader()

    with config.dataset_path.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    if config.rows:
        requested = set(config.rows)
        selected = [
            r for r in all_rows
            if (n := _row_number_from_csv(r)) is not None and n in requested
        ]
    else:
        selected = _row_slice(all_rows, config.start_row, config.max_rows)

    selected_row_count = len(selected)
    filtered_row_count = 0
    if config.exclude_completed_csv is not None:
        if config.retry_errors:
            selected, filtered_row_count = _filter_rows_in_failed_csv(
                selected, config.exclude_completed_csv, match_ground_truth_path=True
            )
        else:
            selected, filtered_row_count = _filter_rows_not_in_completed_csv(
                selected, config.exclude_completed_csv, match_ground_truth_path=True
            )

    started_at = datetime.now().isoformat()
    started_ts = time.time()

    rows_out: list[dict[str, Any]] = []
    pass_count = total_llm_calls = total_iterations = 0
    total_policy = total_passed_policy = total_failed_policy = total_filtered_failed = 0
    scenario_ppr_sum = scenario_unfiltered_sum = 0.0
    scenario_ppr_count = runtime_error_runs = 0
    aggregate_tokens = _empty_tokens()

    print(f"\n{'=' * 80}")
    print(
        f"One-shot baseline | provider={config.provider} | model={config.model} "
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
            # One cycle per scenario: every pass is a first-attempt pass.
            pass_at_1_count=pass_count,
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
        summary["baseline"] = "oneshot"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    write_summary(0)

    for i, row in enumerate(selected, start=1):
        row_number = _safe_int(row.get("row_number"), config.start_row + i - 1)
        prompt = (row.get("prompt") or "").strip()
        print(f"\n[OneShot] ({i}/{len(selected)}) row {row_number} [{config.iac_type}]")

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
                payload = run_scenario(config, row, row_number, prompt)
            except Exception as exc:
                runtime_error_runs += 1
                payload = {
                    "row_number": row_number,
                    "ground_truth_path": row.get("ground_truth_path"),
                    "prompt": prompt,
                    "status": "runtime_error",
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(),
                    "duration_seconds": round(time.time() - row_started, 3),
                }
                print(f"[OneShot] Row {row_number} failed: {exc}")
                print(traceback.format_exc())
            finally:
                # Same per-scenario AWS sweep benchmark.py and the harness
                # baseline run (a no-op for localstack/none).
                try:
                    cleanup_scenario_resources(
                        DeployConfig(target=DeployTarget(config.deploy_target))
                    )
                except Exception as cleanup_exc:
                    print(
                        f"[OneShot] Warning: scenario-finished cleanup for row "
                        f"{row_number} failed unexpectedly: {cleanup_exc}"
                    )

        # Aggregate only rows that produced a scored run (matching benchmark.py:
        # skipped and runtime-error rows are recorded but excluded).
        if payload.get("run_id") and payload.get("status") != "runtime_error":
            if payload.get("final_validation_passed"):
                pass_count += 1
            total_iterations += _safe_int(payload.get("iterations_used"))
            total_llm_calls += _safe_int(payload.get("llm_calls_total"))
            for key in aggregate_tokens:
                aggregate_tokens[key] += _safe_int((payload.get("token_usage") or {}).get(key))
            pm = payload.get("policy_metrics") or {}
            total_policy += _safe_int(pm.get("total_policies"))
            total_passed_policy += _safe_int(pm.get("passed_policies"))
            total_failed_policy += _safe_int(pm.get("failed_policies_all_severity"))
            total_filtered_failed += _safe_int(pm.get("filtered_failed_policies"))
            scenario_ppr_sum += float(pm.get("scenario_policy_pass_rate", 0.0) or 0.0)
            scenario_unfiltered_sum += float(pm.get("unfiltered_compliance_rate", 0.0) or 0.0)
            scenario_ppr_count += 1

            print(
                f"[OneShot] row {row_number}: "
                f"{'PASS' if payload.get('final_validation_passed') else 'FAIL'} "
                f"| extraction={payload.get('extraction_method')} "
                f"| tokens={(payload.get('token_usage') or {}).get('all_tokens')}"
            )

        rows_out.append(payload)
        _append_jsonl(jsonl_path, payload)
        _append_oneshot_csv(csv_path, payload)
        write_summary(len(rows_out))

        if config.sleep_between_rows and i < len(selected):
            time.sleep(config.sleep_between_rows)

    summary = write_summary(len(rows_out))
    (config.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows_out}, indent=2), encoding="utf-8"
    )
    print(f"\n[OneShot] Finished in {summary['duration_seconds']}s -> {config.output_dir}")
    print(f"[OneShot] pass_rate={summary['pass_rate']:.3f}")
    return {"summary": summary, "rows": rows_out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot baseline: IaCGen's first-generation prompt, no repair loop, "
        "scored by IaCGOD's own validators."
    )
    p.add_argument("--iac-type", choices=["cloudformation", "terraform"], default="cloudformation")
    p.add_argument("--dataset", type=Path, default=Path("data/cfn_eval_benchmark_real_aws.csv"))
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
        help="Requires --exclude-completed-csv. Run ONLY the rows that did not get a clean "
        "pass there (final_validation_passed False, or status other than 'ok').",
    )
    p.add_argument("--deploy-target", choices=["none", "localstack", "aws"], default="aws")
    p.add_argument("--provider", choices=["openrouter", "claude", "openai", "deepseek"],
                   default="openrouter")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--openrouter-provider-only", type=str, default=None)
    p.add_argument("--openrouter-min-quantization", type=str, default=None,
                   choices=["int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"])
    p.add_argument("--openrouter-reasoning-effort", type=str, default=None)
    p.add_argument("--openrouter-reasoning-max-tokens", type=int, default=None)
    p.add_argument("--disable-reasoning", action="store_true",
                   help="OpenRouter: omit the reasoning field. DeepSeek direct: send "
                        "thinking={'type':'disabled'}. Avoids reasoning eating the whole "
                        "token budget, but costs accuracy (tested: a row that passes with "
                        "thinking failed cfn-lint without it) — prefer a larger --max-tokens.")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-max-tokens", action="store_true")
    p.add_argument("--skip-security", action="store_true")
    p.add_argument("--sleep-between-rows", type=float, default=0.0)

    args = p.parse_args()
    if args.retry_errors and args.exclude_completed_csv is None:
        p.error("--retry-errors requires --exclude-completed-csv")
    return args


def main() -> None:
    args = parse_args()
    rows = None
    if args.rows:
        rows = [int(r.strip()) for r in args.rows.split(",") if r.strip().isdigit()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("benchmark_runs") / f"oneshot_{args.iac_type}_{ts}"

    config = OneShotConfig(
        dataset_path=args.dataset,
        output_dir=output_dir.resolve(),
        runs_dir=args.runs_dir.resolve(),
        start_row=args.start_row,
        max_rows=args.max_rows,
        rows=rows,
        exclude_completed_csv=args.exclude_completed_csv,
        retry_errors=args.retry_errors,
        iac_type=args.iac_type,
        deploy_target=args.deploy_target,
        provider=args.provider,
        model=args.model,
        openrouter_provider_only=args.openrouter_provider_only,
        openrouter_min_quantization=args.openrouter_min_quantization,
        openrouter_reasoning_effort=args.openrouter_reasoning_effort,
        openrouter_reasoning_max_tokens=args.openrouter_reasoning_max_tokens,
        disable_reasoning=args.disable_reasoning,
        max_tokens=args.max_tokens,
        no_max_tokens=args.no_max_tokens,
        skip_security=args.skip_security,
        sleep_between_rows=args.sleep_between_rows,
    )
    run_oneshot(config)


if __name__ == "__main__":
    main()
