# benchmark_common.py
"""Shared benchmark scoring, aggregation and row-selection helpers.

Extracted verbatim from benchmark.py so that alternative harness runners
(baselines/harness_baseline.py) can produce byte-compatible results.csv /
results.jsonl / summary.json artifacts without importing the LangGraph
multi-agent stack via `main.run_pipeline`.

benchmark.py re-imports every name defined here, so its public surface is
unchanged.

The `config` argument accepted by _build_summary is duck-typed: any object
exposing dataset_path, iac_type, provider, model, deploy_target,
openrouter_provider_only, openrouter_min_quantization,
openrouter_reasoning_effort, openrouter_reasoning_max_tokens,
skip_security and max_iterations will work.
"""
import csv
import json
from pathlib import Path
from typing import Any, Protocol


class SummaryConfig(Protocol):
    """Fields _build_summary() reads off the run configuration.

    Structurally satisfied by the BenchmarkConfig dataclass (multi-agent
    runs) and by BaselineConfig (single-agent harness runs), so both emit
    an identically-shaped summary.json."""

    dataset_path: Path
    iac_type: str
    provider: str
    model: str | None
    deploy_target: str
    openrouter_provider_only: str | None
    openrouter_min_quantization: str | None
    openrouter_reasoning_effort: str | None
    openrouter_reasoning_max_tokens: int | None
    skip_security: bool
    max_iterations: int


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
        pass
    try:
        return int(float(value))
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


def _row_number_from_csv(row: dict[str, str]) -> int | None:
    raw_row_number = (row.get("row_number") or "").strip()
    if not raw_row_number:
        return None
    try:
        return int(raw_row_number)
    except ValueError:
        pass
    try:
        return int(float(raw_row_number))
    except ValueError:
        return None


def _load_completed_scenario_keys(csv_path: Path | None) -> set[str]:
    if csv_path is None:
        return set()
    if not csv_path.exists():
        raise FileNotFoundError(f"Completed-runs CSV not found: {csv_path}")

    completed_keys: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ground_truth_path = (row.get("ground_truth_path") or "").strip()
            if ground_truth_path:
                completed_keys.add(f"ground_truth_path:{ground_truth_path}")

            row_number = _row_number_from_csv(row)
            if row_number is not None:
                completed_keys.add(f"row_number:{row_number}")

    return completed_keys


def _filter_rows_not_in_completed_csv(
    rows: list[dict[str, str]],
    completed_csv: Path | None,
    *,
    match_ground_truth_path: bool = True,
) -> tuple[list[dict[str, str]], int]:
    completed_keys = _load_completed_scenario_keys(completed_csv)
    if not completed_keys:
        return rows, 0

    filtered_rows: list[dict[str, str]] = []
    skipped_count = 0
    for row in rows:
        ground_truth_path = (row.get("ground_truth_path") or "").strip()
        row_number = _row_number_from_csv(row)

        # ground_truth_path is the stable scenario identity; row_number is
        # NOT stable across dataset revisions (a completed_csv built from a
        # differently-numbered dataset can reuse the same row_number for an
        # unrelated scenario). When a ground_truth_path is available, trust
        # it exclusively — only fall back to row_number when it's missing.
        if match_ground_truth_path and ground_truth_path:
            is_match = f"ground_truth_path:{ground_truth_path}" in completed_keys
        else:
            is_match = row_number is not None and f"row_number:{row_number}" in completed_keys

        if is_match:
            skipped_count += 1
            continue

        filtered_rows.append(row)

    return filtered_rows, skipped_count


def _load_failed_scenario_keys(csv_path: Path | None) -> set[str]:
    if csv_path is None:
        return set()
    if not csv_path.exists():
        raise FileNotFoundError(f"Prior results CSV not found: {csv_path}")

    failed_keys: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            final_validation_passed = (row.get("final_validation_passed") or "").strip().lower()
            status = (row.get("status") or "").strip()
            # A row counts as "failed" (worth retrying) either because the
            # artifact was scored and didn't pass, or because the run itself
            # never reached a genuine verdict on the merits — harness_stalled,
            # harness_timeout, harness_error, runtime_error,
            # skipped_empty_prompt, etc. Those statuses can carry
            # final_validation_passed=False already (e.g. harness_stalled), but
            # not always (a runtime_error row has no final_validation_passed at
            # all; a timed-out or errored harness could coincidentally leave a
            # last artifact that happens to validate). status != "ok" is
            # checked independently so none of those cases slip through
            # needing final_validation_passed to also be exactly "false". An
            # empty/missing status (older result CSVs predating this column)
            # is not itself treated as a failure signal, so old CSVs keep
            # their prior final_validation_passed-only behaviour.
            if final_validation_passed != "false" and not (status and status != "ok"):
                continue

            ground_truth_path = (row.get("ground_truth_path") or "").strip()
            if ground_truth_path:
                failed_keys.add(f"ground_truth_path:{ground_truth_path}")

            row_number = _row_number_from_csv(row)
            if row_number is not None:
                failed_keys.add(f"row_number:{row_number}")

    return failed_keys


def _filter_rows_in_failed_csv(
    rows: list[dict[str, str]],
    completed_csv: Path | None,
    *,
    match_ground_truth_path: bool = True,
) -> tuple[list[dict[str, str]], int]:
    """Keep only rows whose row_number/ground_truth_path had
    final_validation_passed=False in completed_csv. Inverse selection of
    _filter_rows_not_in_completed_csv."""
    failed_keys = _load_failed_scenario_keys(completed_csv)
    if not failed_keys:
        return [], 0

    selected_rows: list[dict[str, str]] = []
    for row in rows:
        ground_truth_path = (row.get("ground_truth_path") or "").strip()
        row_number = _row_number_from_csv(row)

        # ground_truth_path is the stable scenario identity; row_number is
        # NOT stable across dataset revisions (completed_csv can be built
        # from a differently-numbered dataset than --dataset, so the same
        # row_number can coincidentally mean an unrelated, already-passing
        # scenario). When a ground_truth_path is available, trust it
        # exclusively — only fall back to row_number when it's missing.
        if match_ground_truth_path and ground_truth_path:
            is_match = f"ground_truth_path:{ground_truth_path}" in failed_keys
        else:
            is_match = row_number is not None and f"row_number:{row_number}" in failed_keys

        if is_match:
            selected_rows.append(row)

    skipped_count = len(rows) - len(selected_rows)
    return selected_rows, skipped_count


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
    config: SummaryConfig,
    selected_row_count: int,
    filtered_row_count: int,
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
        "iac_type": config.iac_type,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": elapsed,
        "provider": config.provider,
        "model": config.model,
        "deploy_target": config.deploy_target,
        "openrouter_provider_only": config.openrouter_provider_only,
        "openrouter_min_quantization": config.openrouter_min_quantization,
        "openrouter_reasoning_effort": config.openrouter_reasoning_effort,
        "openrouter_reasoning_max_tokens": config.openrouter_reasoning_max_tokens,
        "skip_security": config.skip_security,
        "max_iterations": config.max_iterations,
        "rows_requested": selected_row_count,
        "rows_filtered_out": filtered_row_count,
        "rows_selected": selected_row_count - filtered_row_count,
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


