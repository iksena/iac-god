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
import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    output_dir: Path
    start_row: int
    max_rows: int | None
    max_iterations: int
    provider: str           # "openrouter" | "claude" | "openai"
    model: str | None
    deploy_target: str
    openrouter_provider_only: str | None
    openrouter_min_quantization: str | None


CSV_RESULT_FIELDS = [
    "row_number",
    "ground_truth_path",
    "run_id",
    "status",
    "final_validation_passed",
    "iterations_used",
    "llm_calls_total",
    "token_all_tokens",
    "token_input_tokens",
    "token_output_tokens",
    "token_prompt_tokens",
    "token_completion_tokens",
    "scenario_policy_pass_rate",
    "filtered_compliance_rate",
    "unfiltered_compliance_rate",
    "duration_seconds",
    "error_message",
    "error_traceback",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _token_totals(llm_call_log: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }

    for call in llm_call_log:
        usage = call.get("token_usage") or {}
        for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
            totals[key] += _safe_int(usage.get(key), 0)

    totals["all_tokens"] = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["prompt_tokens"]
        + totals["completion_tokens"]
    )
    return totals


def _load_iteration_snapshots(run_dir: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("iteration_*.json")):
        try:
            snapshots.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return snapshots


def _extract_iteration_records(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for snap in snapshots:
        validation_results = snap.get("validation_results", [])
        stage_summary = []
        total_errors = 0

        for result in validation_results:
            errors = result.get("errors", [])
            stage_summary.append(
                {
                    "stage": result.get("stage"),
                    "passed": bool(result.get("passed")),
                    "error_count": len(errors),
                    "errors": errors,
                    "policy_stats": result.get("policy_stats"),
                    "scenario_policy_pass_rate": result.get("scenario_policy_pass_rate"),
                    "filtered_compliance_rate": result.get("filtered_compliance_rate"),
                }
            )
            total_errors += len(errors)

        policy_metrics = _extract_policy_metrics(validation_results)

        records.append(
            {
                "iteration": snap.get("iteration"),
                "validation_passed": bool(snap.get("validation_passed")),
                "total_errors": total_errors,
                "stages": stage_summary,
                "policy_metrics": policy_metrics,
                "timestamp": snap.get("timestamp"),
            }
        )

    return records


def _row_slice(rows: list[dict[str, str]], start_row: int, max_rows: int | None) -> list[dict[str, str]]:
    if start_row < 0:
        start_row = 0
    sliced = rows[start_row:]
    if max_rows is not None and max_rows >= 0:
        return sliced[:max_rows]
    return sliced


def _extract_policy_metrics(validation_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_policies = 0
    passed_policies = 0
    failed_policies_all_severity = 0
    filtered_failed_policies = 0

    for result in validation_results:
        if result.get("stage") not in {"checkov", "trivy"}:
            continue
        stats = result.get("policy_stats") or {}
        total_policies += _safe_int(stats.get("total_policies"), 0)
        passed_policies += _safe_int(stats.get("passed_policies"), 0)
        failed_policies_all_severity += _safe_int(
            stats.get("failed_policies"),
            0,
        )
        filtered_failed_policies += _safe_int(stats.get("filtered_failed_policies"), 0)

    if total_policies > 0 and failed_policies_all_severity == 0 and passed_policies > 0:
        failed_policies_all_severity = max(total_policies - passed_policies, 0)

    if total_policies > 0:
        scenario_ppr = passed_policies / total_policies
        scenario_fcr = (total_policies - filtered_failed_policies) / total_policies
        scenario_unfiltered_compliance = (
            total_policies - failed_policies_all_severity
        ) / total_policies
    else:
        scenario_ppr = 1.0
        scenario_fcr = 1.0
        scenario_unfiltered_compliance = 1.0

    return {
        "total_policies": total_policies,
        "passed_policies": passed_policies,
        "failed_policies_all_severity": failed_policies_all_severity,
        "filtered_failed_policies": filtered_failed_policies,
        "scenario_policy_pass_rate": scenario_ppr,
        "filtered_compliance_rate": scenario_fcr,
        "unfiltered_compliance_rate": scenario_unfiltered_compliance,
    }


def _build_summary(
    config: BenchmarkConfig,
    selected_row_count: int,
    attempted: int,
    pass_count: int,
    pass_at_1_count: int,
    total_iterations: int,
    total_llm_calls: int,
    aggregate_tokens: dict[str, int],
    total_policy_count: int,
    total_passed_policy_count: int,
    total_failed_policy_count: int,
    total_filtered_failed_policy_count: int,
    scenario_ppr_sum: float,
    scenario_unfiltered_compliance_sum: float,
    scenario_ppr_count: int,
    runtime_error_runs: int,
    started_at: str,
    completed_at: str,
    elapsed: float,
) -> dict[str, Any]:
    evaluated_runs = max(attempted - runtime_error_runs, 0)
    avg_ppr = (scenario_ppr_sum / scenario_ppr_count) if scenario_ppr_count else 0.0
    total_fcr = (
        (total_policy_count - total_filtered_failed_policy_count) / total_policy_count
        if total_policy_count
        else 0.0
    )
    total_unfiltered_compliance_rate = (
        (total_policy_count - total_failed_policy_count) / total_policy_count
        if total_policy_count
        else 0.0
    )
    avg_unfiltered_compliance_rate = (
        scenario_unfiltered_compliance_sum / scenario_ppr_count
        if scenario_ppr_count
        else 0.0
    )

    return {
        "dataset": str(config.dataset_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": elapsed,
        "provider": config.provider,
        "model": config.model,
        "deploy_target": config.deploy_target,
        "openrouter_provider_only": config.openrouter_provider_only,
        "openrouter_min_quantization": config.openrouter_min_quantization,
        "max_iterations": config.max_iterations,
        "rows_requested": selected_row_count,
        "rows_attempted": attempted,
        "rows_evaluated": evaluated_runs,
        "runtime_error_runs": runtime_error_runs,
        "rows_passed": pass_count,
        "pass_rate": (pass_count / evaluated_runs) if evaluated_runs else 0.0,
        "pass_at_1": (pass_at_1_count / evaluated_runs) if evaluated_runs else 0.0,
        "total_iterations": total_iterations,
        "avg_iterations": (total_iterations / attempted) if attempted else 0.0,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls": (total_llm_calls / attempted) if attempted else 0.0,
        "avg_ppr": avg_ppr,
        "total_fcr": total_fcr,
        "avg_unfiltered_compliance_rate": avg_unfiltered_compliance_rate,
        "total_unfiltered_compliance_rate": total_unfiltered_compliance_rate,
        "policy_totals": {
            "total_policies": total_policy_count,
            "passed_policies": total_passed_policy_count,
            "failed_policies_all_severity": total_failed_policy_count,
            "filtered_failed_policies": total_filtered_failed_policy_count,
            "scenarios_with_policy_metrics": scenario_ppr_count,
        },
        "token_usage_total": aggregate_tokens,
        "token_usage_avg": {
            key: (value / attempted if attempted else 0.0)
            for key, value in aggregate_tokens.items()
        },
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _row_to_csv(payload: dict[str, Any]) -> dict[str, Any]:
    token_usage = payload.get("token_usage", {})
    policy_metrics = payload.get("policy_metrics", {})
    return {
        "row_number": payload.get("row_number"),
        "ground_truth_path": payload.get("ground_truth_path"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "final_validation_passed": payload.get("final_validation_passed"),
        "iterations_used": payload.get("iterations_used"),
        "llm_calls_total": payload.get("llm_calls_total"),
        "token_all_tokens": token_usage.get("all_tokens"),
        "token_input_tokens": token_usage.get("input_tokens"),
        "token_output_tokens": token_usage.get("output_tokens"),
        "token_prompt_tokens": token_usage.get("prompt_tokens"),
        "token_completion_tokens": token_usage.get("completion_tokens"),
        "scenario_policy_pass_rate": policy_metrics.get("scenario_policy_pass_rate"),
        "filtered_compliance_rate": policy_metrics.get("filtered_compliance_rate"),
        "unfiltered_compliance_rate": policy_metrics.get("unfiltered_compliance_rate"),
        "duration_seconds": payload.get("duration_seconds"),
        "error_message": payload.get("error_message"),
        "error_traceback": payload.get("error_traceback"),
    }


def _append_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_RESULT_FIELDS)
        writer.writerow(_row_to_csv(payload))


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

    selected_rows = _row_slice(all_rows, config.start_row, config.max_rows)
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
    print(f"Benchmark start | provider={config.provider} | model={config.model or 'env default'} | rows={len(selected_rows)} | dataset={config.dataset_path}")
    print(f"{'=' * 80}")

    initial_summary = _build_summary(
        config=config,
        selected_row_count=len(selected_rows),
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
                selected_row_count=len(selected_rows),
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

        print(f"\n[Benchmark] ({i}/{len(selected_rows)}) Running row {row_number}")

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
            selected_row_count=len(selected_rows),
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
        selected_row_count=len(selected_rows),
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


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("benchmark_runs") / ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark IaCGOD pipeline over a CSV dataset of prompts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "iac_basic.csv",
        help="Path to CSV file with columns: row_number, prompt, ground_truth_path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for benchmark artifacts (default: benchmark_runs/<timestamp>)",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
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
        "--deploy-target",
        choices=["none", "localstack", "aws"],
        default="localstack",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()
    cfg = BenchmarkConfig(
        dataset_path=args.dataset,
        output_dir=output_dir,
        start_row=args.start_row,
        max_rows=args.max_rows,
        max_iterations=args.max_iterations,
        provider=args.provider,
        model=args.model,
        deploy_target=args.deploy_target,
        openrouter_provider_only=args.openrouter_provider_only,
        openrouter_min_quantization=args.openrouter_min_quantization,
    )
    run_benchmark(cfg)
