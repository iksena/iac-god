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
    "duration_seconds",
    "error_message",
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


def _build_summary(
    config: BenchmarkConfig,
    selected_row_count: int,
    attempted: int,
    pass_count: int,
    pass_at_1_count: int,
    total_iterations: int,
    total_llm_calls: int,
    aggregate_tokens: dict[str, int],
    started_at: str,
    completed_at: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "dataset": str(config.dataset_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": elapsed,
        "provider": config.provider,
        "model": config.model,
        "max_iterations": config.max_iterations,
        "rows_requested": selected_row_count,
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _row_to_csv(payload: dict[str, Any]) -> dict[str, Any]:
    token_usage = payload.get("token_usage", {})
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
        "duration_seconds": payload.get("duration_seconds"),
        "error_message": payload.get("error_message"),
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

    # Initialize incremental output files.
    jsonl_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_RESULT_FIELDS)
        writer.writeheader()

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

    # Write an initial empty summary so dashboards can start polling immediately.
    initial_summary = _build_summary(
        config=config,
        selected_row_count=len(selected_rows),
        attempted=0,
        pass_count=0,
        pass_at_1_count=0,
        total_iterations=0,
        total_llm_calls=0,
        aggregate_tokens=aggregate_tokens,
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
                "duration_seconds": 0.0,
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
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
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

        # Incremental persistence after every finished row.
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
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            elapsed=round(time.time() - started_ts, 3),
        )
        summary_path.write_text(json.dumps(interim_summary, indent=2), encoding="utf-8")

    completed_at = datetime.now(timezone.utc).isoformat()
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
        started_at=started_at,
        completed_at=completed_at,
        elapsed=elapsed,
    )

    # Persist benchmark outputs.
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
