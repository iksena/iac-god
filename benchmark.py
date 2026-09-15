# benchmark.py
import argparse
import csv
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from main import run_pipeline


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    output_dir: Path
    start_row: int
    max_rows: int | None
    rows: list[int] | None
    exclude_completed_csv: Path | None
    retry_errors: bool
    max_iterations: int
    provider: str           # "openrouter" | "claude" | "openai"
    model: str | None
    deploy_target: str
    openrouter_provider_only: str | None
    openrouter_min_quantization: str | None
    openrouter_reasoning_effort: str | None
    openrouter_reasoning_max_tokens: int | None
    skip_security: bool
    iac_type: str           # "cloudformation" | "terraform"

# ---------------------------------------------------------------------------
# Shared scoring / aggregation / row-selection helpers.
# Defined in benchmark_common.py so that alternative harness runners
# (baselines/harness_baseline.py) can reuse them without importing the
# LangGraph stack that `from main import run_pipeline` pulls in above.
# Re-exported here under their original names: benchmark.py's public surface
# is unchanged.
# ---------------------------------------------------------------------------
from benchmark_common import (  # noqa: F401
    CSV_RESULT_FIELDS,
    _append_csv,
    _append_jsonl,
    _build_summary,
    _extract_iteration_records,
    _extract_policy_metrics,
    _filter_rows_in_failed_csv,
    _filter_rows_not_in_completed_csv,
    _load_completed_scenario_keys,
    _load_failed_scenario_keys,
    _load_iteration_snapshots,
    _row_number_from_csv,
    _row_slice,
    _row_to_csv,
    _safe_int,
    _token_totals,
)

def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.json"
    jsonl_path = config.output_dir / "results.jsonl"
    csv_path = config.output_dir / "results.csv"

    jsonl_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_RESULT_FIELDS)
        writer.writeheader()

    with config.dataset_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    if config.rows:
        requested_rows = set(config.rows)
        selected_rows = [
            row
            for row in all_rows
            if (row_number := _row_number_from_csv(row)) is not None and row_number in requested_rows
        ]
    else:
        selected_rows = _row_slice(all_rows, config.start_row, config.max_rows)

    selected_row_count = len(selected_rows)

    filtered_row_count = 0
    if config.exclude_completed_csv is not None:
        if config.retry_errors:
            selected_rows, filtered_row_count = _filter_rows_in_failed_csv(
                selected_rows,
                config.exclude_completed_csv,
                match_ground_truth_path=not bool(config.rows),
            )
        else:
            selected_rows, filtered_row_count = _filter_rows_not_in_completed_csv(
                selected_rows,
                config.exclude_completed_csv,
                match_ground_truth_path=not bool(config.rows),
            )
    rows_after_filter = len(selected_rows)

    started_at = datetime.now().isoformat()
    started_ts = time.time()

    rows_out: list[dict[str, Any]] = []
    pass_count = 0
    pass_at_1_count = 0
    total_iterations = 0
    total_llm_calls = 0
    total_policy_count = 0
    total_passed_policy_count = 0
    total_failed_policy_count = 0
    total_filtered_failed_policy_count = 0
    scenario_ppr_sum = 0.0
    scenario_unfiltered_compliance_sum = 0.0
    scenario_ppr_count = 0
    runtime_error_runs = 0
    aggregate_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }

    print(f"\n{'=' * 80}")
    print(
        f"Benchmark start | iac_type={config.iac_type} | provider={config.provider} "
        f"| model={config.model or 'env default'} | rows={rows_after_filter} "
        f"| dataset={config.dataset_path}"
    )
    if config.exclude_completed_csv is not None:
        if config.retry_errors:
            print(
                f"[Benchmark] Retrying {rows_after_filter} previously-failed row(s) "
                f"from prior results CSV: {config.exclude_completed_csv}"
            )
        else:
            print(
                f"[Benchmark] Excluding {filtered_row_count} row(s) using prior results CSV: "
                f"{config.exclude_completed_csv}"
            )
    print(f"{'=' * 80}")

    initial_summary = _build_summary(
        config=config,
        selected_row_count=selected_row_count,
        filtered_row_count=filtered_row_count,
        attempted=0,
        pass_count=0,
        pass_at_1_count=0,
        total_iterations=0,
        total_llm_calls=0,
        aggregate_tokens=aggregate_tokens,
        total_policy_count=0,
        total_passed_policy_count=0,
        total_failed_policy_count=0,
        total_filtered_failed_policy_count=0,
        scenario_ppr_sum=0.0,
        scenario_unfiltered_compliance_sum=0.0,
        scenario_ppr_count=0,
        runtime_error_runs=0,
        started_at=started_at,
        completed_at=started_at,
        elapsed=0.0,
    )
    summary_path.write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")

    for i, row in enumerate(selected_rows, start=1):
        row_number = _safe_int(row.get("row_number"), config.start_row + i - 1)
        prompt = (row.get("prompt") or "").strip()

        if not prompt:
            print(f"\n[Benchmark] Skipping row {row_number}: empty prompt")
            result_payload = {
                "row_number": row_number,
                "ground_truth_path": row.get("ground_truth_path"),
                "prompt": prompt,
                "run_id": None,
                "status": "skipped_empty_prompt",
                "error_message": "Prompt is empty",
                "final_validation_passed": False,
                "iterations_used": 0,
                "llm_calls_total": 0,
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "all_tokens": 0,
                },
                "iteration_records": [],
                "validation_results_final": [],
                "remediation_history": [],
                "policy_metrics": {
                    "total_policies": 0,
                    "passed_policies": 0,
                    "failed_policies_all_severity": 0,
                    "filtered_failed_policies": 0,
                    "scenario_policy_pass_rate": 0.0,
                    "filtered_compliance_rate": 0.0,
                    "unfiltered_compliance_rate": 0.0,
                },
                "duration_seconds": 0.0,
                "error_traceback": None,
            }
            rows_out.append(result_payload)
            _append_jsonl(jsonl_path, result_payload)
            _append_csv(csv_path, result_payload)

            interim_summary = _build_summary(
                config=config,
                selected_row_count=selected_row_count,
                filtered_row_count=filtered_row_count,
                attempted=len(rows_out),
                pass_count=pass_count,
                pass_at_1_count=pass_at_1_count,
                total_iterations=total_iterations,
                total_llm_calls=total_llm_calls,
                aggregate_tokens=aggregate_tokens,
                total_policy_count=total_policy_count,
                total_passed_policy_count=total_passed_policy_count,
                total_failed_policy_count=total_failed_policy_count,
                total_filtered_failed_policy_count=total_filtered_failed_policy_count,
                scenario_ppr_sum=scenario_ppr_sum,
                scenario_unfiltered_compliance_sum=scenario_unfiltered_compliance_sum,
                scenario_ppr_count=scenario_ppr_count,
                runtime_error_runs=runtime_error_runs,
                started_at=started_at,
                completed_at=datetime.now().isoformat(),
                elapsed=round(time.time() - started_ts, 3),
            )
            summary_path.write_text(json.dumps(interim_summary, indent=2), encoding="utf-8")
            continue

        print(f"\n[Benchmark] ({i}/{len(selected_rows)}) Running row {row_number} [{config.iac_type}]")

        row_started = time.time()
        status = "ok"
        error_message = None
        result_payload: dict[str, Any]

        try:
            final_state = run_pipeline(
                user_request=prompt,
                max_iterations=config.max_iterations,
                provider=config.provider,
                model=config.model,
                deploy_target=config.deploy_target,
                openrouter_provider_only=config.openrouter_provider_only,
                openrouter_min_quantization=config.openrouter_min_quantization,
                openrouter_reasoning_effort=config.openrouter_reasoning_effort,
                openrouter_reasoning_max_tokens=config.openrouter_reasoning_max_tokens,
                skip_security=config.skip_security,
                iac_type=config.iac_type,
            )

            run_id = final_state["run_id"]
            run_dir = Path("runs") / run_id
            snapshots = _load_iteration_snapshots(run_dir)
            iteration_records = _extract_iteration_records(snapshots)

            llm_calls = final_state.get("llm_call_log", [])
            tokens = _token_totals(llm_calls)
            total_runs_iterations = _safe_int(final_state.get("current_iteration"), 0)
            final_passed = bool(final_state.get("validation_passed"))
            policy_metrics = _extract_policy_metrics(final_state.get("validation_results", []))

            if final_passed:
                pass_count += 1
            if final_passed and total_runs_iterations == 1:
                pass_at_1_count += 1

            total_iterations += total_runs_iterations
            total_llm_calls += len(llm_calls)
            for key in aggregate_tokens:
                aggregate_tokens[key] += tokens[key]

            total_policy_count += _safe_int(policy_metrics.get("total_policies"), 0)
            total_passed_policy_count += _safe_int(policy_metrics.get("passed_policies"), 0)
            total_failed_policy_count += _safe_int(
                policy_metrics.get("failed_policies_all_severity"), 0,
            )
            total_filtered_failed_policy_count += _safe_int(
                policy_metrics.get("filtered_failed_policies"), 0,
            )

            scenario_ppr_sum += float(policy_metrics.get("scenario_policy_pass_rate", 0.0) or 0.0)
            scenario_unfiltered_compliance_sum += float(
                policy_metrics.get("unfiltered_compliance_rate", 0.0) or 0.0
            )
            scenario_ppr_count += 1

            result_payload = {
                "row_number": row_number,
                "ground_truth_path": row.get("ground_truth_path"),
                "prompt": prompt,
                "run_id": run_id,
                "status": status,
                "error_message": error_message,
                "final_validation_passed": final_passed,
                "iterations_used": total_runs_iterations,
                "llm_calls_total": len(llm_calls),
                "token_usage": tokens,
                "iteration_records": iteration_records,
                "validation_results_final": final_state.get("validation_results", []),
                "remediation_history": final_state.get("remediation_history", []),
                "policy_metrics": policy_metrics,
                "duration_seconds": round(time.time() - row_started, 3),
            }
        except Exception as exc:
            status = "runtime_error"
            error_message = str(exc)
            error_traceback = traceback.format_exc()
            runtime_error_runs += 1
            result_payload = {
                "row_number": row_number,
                "ground_truth_path": row.get("ground_truth_path"),
                "prompt": prompt,
                "status": status,
                "error_message": error_message,
                "error_traceback": error_traceback,
                "duration_seconds": round(time.time() - row_started, 3),
            }
            print(f"[Benchmark] Row {row_number} failed: {error_message}")
            print("[Benchmark] Traceback:")
            print(error_traceback)

        rows_out.append(result_payload)

        _append_jsonl(jsonl_path, result_payload)
        _append_csv(csv_path, result_payload)
        interim_summary = _build_summary(
            config=config,
            selected_row_count=selected_row_count,
            filtered_row_count=filtered_row_count,
            attempted=len(rows_out),
            pass_count=pass_count,
            pass_at_1_count=pass_at_1_count,
            total_iterations=total_iterations,
            total_llm_calls=total_llm_calls,
            aggregate_tokens=aggregate_tokens,
            total_policy_count=total_policy_count,
            total_passed_policy_count=total_passed_policy_count,
            total_failed_policy_count=total_failed_policy_count,
            total_filtered_failed_policy_count=total_filtered_failed_policy_count,
            scenario_ppr_sum=scenario_ppr_sum,
            scenario_unfiltered_compliance_sum=scenario_unfiltered_compliance_sum,
            scenario_ppr_count=scenario_ppr_count,
            runtime_error_runs=runtime_error_runs,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            elapsed=round(time.time() - started_ts, 3),
        )
        summary_path.write_text(json.dumps(interim_summary, indent=2), encoding="utf-8")

    completed_at = datetime.now().isoformat()
    elapsed = round(time.time() - started_ts, 3)
    attempted = len(rows_out)

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
        total_policy_count=total_policy_count,
        total_passed_policy_count=total_passed_policy_count,
        total_failed_policy_count=total_failed_policy_count,
        total_filtered_failed_policy_count=total_filtered_failed_policy_count,
        scenario_ppr_sum=scenario_ppr_sum,
        scenario_unfiltered_compliance_sum=scenario_unfiltered_compliance_sum,
        scenario_ppr_count=scenario_ppr_count,
        runtime_error_runs=runtime_error_runs,
        started_at=started_at,
        completed_at=completed_at,
        elapsed=elapsed,
    )

    (config.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows_out}, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[Benchmark] Finished in {elapsed}s")
    print(f"[Benchmark] Output dir: {config.output_dir}")
    print(f"[Benchmark] pass_rate={summary['pass_rate']:.3f} pass@1={summary['pass_at_1']:.3f}")

    return {"summary": summary, "rows": rows_out}


def _default_output_dir(iac_type: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("benchmark_runs") / f"{iac_type}_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark IaCGOD pipeline over a CSV dataset of prompts."
    )
    parser.add_argument(
        "--iac-type",
        choices=["cloudformation", "terraform"],
        default="cloudformation",
        help="IaC language to generate. Determines default dataset path when --dataset is omitted.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to CSV file with columns: row_number, prompt[, ground_truth_path]. "
            "Defaults to data/iac_basic.csv for cloudformation or "
            "data/tf_basic.csv for terraform when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for benchmark artifacts (default: benchmark_runs/<iac_type>_<timestamp>)",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--rows", 
        type=str, 
        default=None, 
        help="Comma-separated list of CSV row_number values to execute (e.g., 1,6,7,142)"
    )
    parser.add_argument(
        "--exclude-completed-csv",
        type=Path,
        default=None,
        help=(
            "CSV file of prior benchmark results to exclude from this run. "
            "Rows are skipped when row_number matches; when --rows is not used, "
            "ground_truth_path matches are also excluded. "
            "Combine with --retry-errors to instead select only the failed rows."
        ),
    )
    parser.add_argument(
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
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--provider",
        choices=["openrouter", "claude", "openai"],
        default="openrouter",
        help=(
            "LLM provider. "
            "'openai' reads OPENAI_API_KEY + OPENAI_MODEL from .env; "
            "use --model to override the model for this run."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the model for this benchmark run. "
            "openai examples: o3-mini, o3, o4-mini, gpt-4o, codex-mini-latest"
        ),
    )
    parser.add_argument(
        "--openrouter-provider-only",
        type=str,
        default=None,
        help="Comma-separated OpenRouter provider slugs to allow",
    )
    parser.add_argument(
        "--openrouter-min-quantization",
        type=str,
        choices=["int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"],
        default=None,
    )
    parser.add_argument(
        "--openrouter-reasoning-effort",
        type=str,
        default=None,
        help=(
            "Reasoning effort hint sent as reasoning.effort (maps to OpenRouter's "
            "provider.reasoning.effort). Typical values: low, medium, high "
            "(some models also accept max/none). Lower effort leaves more of the "
            "shared max_tokens budget for actual content instead of reasoning — "
            "useful for always-reasoning models that can return empty content "
            "when reasoning consumes the whole token budget."
        ),
    )
    parser.add_argument(
        "--openrouter-reasoning-max-tokens",
        type=int,
        default=None,
        help=(
            "Explicit token budget for reasoning, sent as reasoning.max_tokens "
            "(support varies by model/provider — unsupported models are expected "
            "to just ignore it). Reasoning tokens share the same completion "
            "budget as content on OpenRouter, so this is ADDED on top of the "
            "configured max_tokens rather than carved out of it, keeping "
            "max_tokens a guaranteed content budget even when it's a fixed, "
            "research-controlled parameter that can't itself be changed."
        ),
    )
    parser.add_argument(
        "--deploy-target",
        choices=["none", "localstack", "aws"],
        default="localstack",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip the deploy stage entirely (equivalent to --deploy-target none, and overrides it).",
    )
    parser.add_argument(
        "--skip-security",
        action="store_true",
        help=(
            "Skip the security misconfiguration scan (trivy) stage. Structural "
            "validation (yaml/cfn-lint or tflint/terraform-validate) still runs; "
            "deploy still gates on structural validation passing unless "
            "--skip-deploy / --deploy-target none is also used."
        ),
    )
    args = parser.parse_args()
    if args.retry_errors and args.exclude_completed_csv is None:
        parser.error("--retry-errors requires --exclude-completed-csv")
    if args.skip_deploy:
        args.deploy_target = "none"
    return args


if __name__ == "__main__":
    import logging
    logging.getLogger("neo4j").setLevel(logging.ERROR)

    args = parse_args()

    # Resolve dataset: explicit --dataset wins; otherwise pick by iac_type.
    if args.dataset is not None:
        dataset_path = args.dataset
    elif args.iac_type == "terraform":
        dataset_path = Path("data") / "tf_basic.csv"
    else:
        dataset_path = Path("data") / "iac_basic.csv"

    output_dir = args.output_dir or _default_output_dir(args.iac_type)

    specific_rows = None
    if args.rows:
        specific_rows = [int(r.strip()) for r in args.rows.split(",") if r.strip().isdigit()]

    cfg = BenchmarkConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        start_row=args.start_row,
        max_rows=args.max_rows,
        rows=specific_rows,
        exclude_completed_csv=args.exclude_completed_csv,
        retry_errors=args.retry_errors,
        max_iterations=args.max_iterations,
        provider=args.provider,
        model=args.model,
        deploy_target=args.deploy_target,
        openrouter_provider_only=args.openrouter_provider_only,
        openrouter_min_quantization=args.openrouter_min_quantization,
        openrouter_reasoning_effort=args.openrouter_reasoning_effort,
        openrouter_reasoning_max_tokens=args.openrouter_reasoning_max_tokens,
        skip_security=args.skip_security,
        iac_type=args.iac_type,
    )
    run_benchmark(cfg)
