import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from main import run_pipeline


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    output_dir: Path
    start_row: int
    max_rows: int | None
    max_iterations: int
    provider: str
    model: str | None


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
                }
            )
            total_errors += len(errors)

        records.append(
            {
                "iteration": snap.get("iteration"),
                "validation_passed": bool(snap.get("validation_passed")),
                "total_errors": total_errors,
                "stages": stage_summary,
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


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with config.dataset_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    selected_rows = _row_slice(all_rows, config.start_row, config.max_rows)
    started_at = datetime.now(timezone.utc).isoformat()
    started_ts = time.time()

    rows_out: list[dict[str, Any]] = []
    pass_count = 0
    pass_at_1_count = 0
    total_iterations = 0
    total_llm_calls = 0
    aggregate_tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "all_tokens": 0,
    }

    print(f"\n{'=' * 80}")
    print(f"Benchmark start | rows={len(selected_rows)} | dataset={config.dataset_path}")
    print(f"{'=' * 80}")

    for i, row in enumerate(selected_rows, start=1):
        row_number = _safe_int(row.get("row_number"), config.start_row + i - 1)
        prompt = (row.get("prompt") or "").strip()

        if not prompt:
            print(f"\n[Benchmark] Skipping row {row_number}: empty prompt")
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
            )

            run_id = final_state["run_id"]
            run_dir = Path("runs") / run_id
            snapshots = _load_iteration_snapshots(run_dir)
            iteration_records = _extract_iteration_records(snapshots)

            llm_calls = final_state.get("llm_call_log", [])
            tokens = _token_totals(llm_calls)
            total_runs_iterations = _safe_int(final_state.get("current_iteration"), 0)
            final_passed = bool(final_state.get("validation_passed"))

            if final_passed:
                pass_count += 1
            if final_passed and total_runs_iterations == 1:
                pass_at_1_count += 1

            total_iterations += total_runs_iterations
            total_llm_calls += len(llm_calls)
            for key in aggregate_tokens:
                aggregate_tokens[key] += tokens[key]

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
                "duration_seconds": round(time.time() - row_started, 3),
            }
        except Exception as exc:  # Keep benchmark running even if one row fails.
            status = "runtime_error"
            error_message = str(exc)
            result_payload = {
                "row_number": row_number,
                "ground_truth_path": row.get("ground_truth_path"),
                "prompt": prompt,
                "status": status,
                "error_message": error_message,
                "duration_seconds": round(time.time() - row_started, 3),
            }
            print(f"[Benchmark] Row {row_number} failed: {error_message}")

        rows_out.append(result_payload)

    completed_at = datetime.now(timezone.utc).isoformat()
    elapsed = round(time.time() - started_ts, 3)
    attempted = len(rows_out)

    summary = {
        "dataset": str(config.dataset_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": elapsed,
        "provider": config.provider,
        "model": config.model,
        "max_iterations": config.max_iterations,
        "rows_requested": len(selected_rows),
        "rows_attempted": attempted,
        "rows_passed": pass_count,
        "pass_rate": (pass_count / attempted) if attempted else 0.0,
        "pass_at_1": (pass_at_1_count / attempted) if attempted else 0.0,
        "total_iterations": total_iterations,
        "avg_iterations": (total_iterations / attempted) if attempted else 0.0,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls": (total_llm_calls / attempted) if attempted else 0.0,
        "token_usage_total": aggregate_tokens,
        "token_usage_avg": {
            key: (value / attempted if attempted else 0.0)
            for key, value in aggregate_tokens.items()
        },
    }

    # Persist benchmark outputs.
    (config.output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows_out}, indent=2),
        encoding="utf-8",
    )

    with (config.output_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row) + "\n")

    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n[Benchmark] Finished in {elapsed}s")
    print(f"[Benchmark] Output dir: {config.output_dir}")
    print(f"[Benchmark] pass_rate={summary['pass_rate']:.3f} pass@1={summary['pass_at_1']:.3f}")

    return {"summary": summary, "rows": rows_out}


def _default_output_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("benchmark_runs") / ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark IaCGOD pipeline over a CSV dataset of prompts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "iac_basic.csv",
        help="Path to CSV file containing columns: row_number, prompt, ground_truth_path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write benchmark artifacts (default: benchmark_runs/<timestamp>)",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--provider", choices=["openrouter", "claude"], default="openrouter")
    parser.add_argument("--model", type=str, default=None)
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
    )
    run_benchmark(cfg)
