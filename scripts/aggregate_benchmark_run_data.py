import os
import json
import shutil
import time
import pandas as pd


def _count_stage_errors(stage_record):
    """Return a robust count of errors for a stage record."""
    errors = stage_record.get("errors", [])
    if isinstance(errors, list):
        return len([e for e in errors if e])
    if isinstance(errors, str):
        return 1 if errors.strip() else 0

    # Fallback: treat an explicit failed stage with no structured list as one error.
    if not stage_record.get("passed", True):
        return 1
    return 0


def _classify_stage(stage_name):
    stage = (stage_name or "").strip().lower().replace("_", "-")

    # if stage in {"yaml", "cfn-lint", "cfnlint"}:
    #     return "yaml_cfn_lint"
    
    if stage in {"yaml"}:
        return "yaml"
    
    if stage in {"cfn-lint"}:
        return "cfn-lint"

    if stage in {
        "trivy",
        "checkov",
        "security",
        "security-scan",
        "iac-security",
        "kics",
        "tfsec",
    }:
        return "security"

    if stage in {"deployment", "deploy", "stack-deploy", "cloudformation-deploy"}:
        return "deployment"

    return "other"


def _extract_stage_errors(stage_record):
    """Return stage errors as a normalized list of non-empty strings."""
    errors = stage_record.get("errors", [])
    if isinstance(errors, list):
        return [str(e).strip() for e in errors if str(e).strip()]
    if isinstance(errors, str):
        return [errors.strip()] if errors.strip() else []
    return []


def _normalize_path_inputs(paths):
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def _collect_files_recursive(paths, extension, exclude_filenames=None):
    """Collect files from paths or folders recursively that end with ``extension``."""
    exclude_filenames = set(exclude_filenames or [])
    collected = []

    for path in _normalize_path_inputs(paths):
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file_name in files:
                    if not file_name.lower().endswith(extension):
                        continue
                    if file_name in exclude_filenames:
                        continue
                    collected.append(os.path.join(root, file_name))
        elif os.path.isfile(path) and path.lower().endswith(extension):
            if os.path.basename(path) not in exclude_filenames:
                collected.append(path)

    return sorted(dict.fromkeys(collected))


def merge_results_from_paths(
    csv_paths,
    merged_csv_path="results_merged.csv",
    jsonl_paths=None,
    merged_jsonl_path="results_merged.jsonl",
):
    """Recursively expand CSV and JSONL inputs, then merge them."""
    csv_files = _collect_files_recursive(
        csv_paths,
        ".csv",
        exclude_filenames={os.path.basename(merged_csv_path)},
    )
    jsonl_files = _collect_files_recursive(
        jsonl_paths,
        ".jsonl",
        exclude_filenames={os.path.basename(merged_jsonl_path)},
    )

    if not csv_files and not jsonl_files:
        print("No CSV or JSONL files found to merge.")
        return

    merge_results(
        csv_paths=csv_files,
        merged_csv_path=merged_csv_path,
        jsonl_paths=jsonl_files,
        merged_jsonl_path=merged_jsonl_path,
    )


def merge_results_from_directory(
    base_dir,
    merged_csv_path="results_merged.csv",
    merged_jsonl_path="results_merged.jsonl",
):
    """Find CSV and JSONL files recursively under ``base_dir`` and merge them."""
    resolved_csv_path = (
        merged_csv_path
        if os.path.isabs(merged_csv_path)
        else os.path.join(base_dir, merged_csv_path)
    )
    resolved_jsonl_path = (
        merged_jsonl_path
        if os.path.isabs(merged_jsonl_path)
        else os.path.join(base_dir, merged_jsonl_path)
    )

    merge_results_from_paths(
        csv_paths=[base_dir],
        merged_csv_path=resolved_csv_path,
        jsonl_paths=[base_dir],
        merged_jsonl_path=resolved_jsonl_path,
    )


def filter_runtime_error_rows(
    input_csv,
    output_csv="results_without_runtime_error.csv",
    status_col="status",
):
    """Write a copy of ``input_csv`` with ``runtime_error`` rows removed."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if status_col not in df.columns:
        raise ValueError(f"Column '{status_col}' not found in {input_csv}")

    status_values = df[status_col].fillna("").astype(str).str.strip().str.lower()
    filtered_df = df[status_values != "runtime_error"].copy()

    filtered_df.to_csv(output_csv, index=False)
    removed_rows = len(df) - len(filtered_df)
    print(
        f"Filtered {removed_rows} runtime_error row(s) from '{input_csv}' into '{output_csv}'"
    )
    return filtered_df

def merge_results_with_reports(input_csv="results.csv", base_dir="runs", output_csv="results_merged.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: The input CSV '{input_csv}' does not exist.")
        return

    # 1. Load the existing results
    df_results = pd.read_csv(input_csv)
    
    # 2. Extract additional data from the runs directory
    aggregated_extra_data = []
    
    if os.path.exists(base_dir):
        for run_id in os.listdir(base_dir):
            run_dir = os.path.join(base_dir, run_id)
            if not os.path.isdir(run_dir):
                continue

            report_path = os.path.join(run_dir, "final_report.json")
            if not os.path.exists(report_path):
                continue

            with open(report_path, "r", encoding="utf-8") as f:
                try:
                    report = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading JSON in {report_path}")
                    continue

            # Build the base extra columns dictionary
            row_extra = {
                "run_id": report.get("run_id", run_id),
                "policy_total_policies": report.get("policy_metrics", {}).get("total_policies", 0),
                "policy_passed_policies": report.get("policy_metrics", {}).get("passed_policies", 0),
                "final_template": report.get("final_template", "None"),
            }

            # Extract full deploy_validation_result details
            deploy_res = report.get("deploy_validation_result", {})
            row_extra["deploy_target"] = deploy_res.get("target", "none")
            row_extra["deploy_passed"] = deploy_res.get("passed", False)
            row_extra["deploy_stack_id"] = deploy_res.get("stack_id", "None")
            row_extra["deploy_error_message"] = deploy_res.get("error_message") or "None"
            row_extra["deploy_logs"] = deploy_res.get("deployment_logs") or "None"
            row_extra["deploy_duration_seconds"] = deploy_res.get("duration_seconds", 0.0)
            
            failed_res = deploy_res.get("failed_resources", [])
            row_extra["deploy_failed_resources"] = json.dumps(failed_res) if failed_res else "None"
            
            completed_res = deploy_res.get("completed_resources", [])
            row_extra["deploy_completed_resources"] = " | ".join(completed_res) if completed_res else "None"

            # Extract Latest Iteration Errors separated by stage
            rem_history = report.get("remediation_history", [])
            if rem_history:
                # Total stage error counts with sequential gating:
                # YAML/CFN_LINT -> SECURITY -> DEPLOYMENT.
                total_yaml_errors = 0
                total_cfn_lint_errors = 0
                total_security_errors = 0
                total_deployment_errors = 0

                all_yaml_errors = []
                all_cfn_lint_errors = []
                all_security_errors = []
                all_deployment_errors = []

                for iteration in rem_history:
                    stage_records = iteration.get("errors", [])

                    yaml_count = 0
                    cfn_lint_count = 0
                    security_count = 0
                    deployment_count = 0

                    for err_stage in stage_records:
                        stage_group = _classify_stage(err_stage.get("stage", "unknown"))
                        stage_error_count = _count_stage_errors(err_stage)
                        stage_errors_list = _extract_stage_errors(err_stage)

                        if stage_group == "yaml":
                            yaml_count += stage_error_count
                            all_yaml_errors.extend(stage_errors_list)
                        elif stage_group == "cfn-lint":
                            cfn_lint_count += stage_error_count
                            all_cfn_lint_errors.extend(stage_errors_list)
                        elif stage_group == "security":
                            security_count += stage_error_count
                            all_security_errors.extend(stage_errors_list)
                        elif stage_group == "deployment":
                            deployment_count += stage_error_count
                            all_deployment_errors.extend(stage_errors_list)

                    # Gate later stages if earlier stages have errors.
                    if yaml_count > 0:
                        security_count = 0
                        deployment_count = 0
                    elif cfn_lint_count > 0:
                        security_count = 0
                        deployment_count = 0
                    elif security_count > 0:
                        deployment_count = 0

                    total_yaml_errors += yaml_count
                    total_cfn_lint_errors += cfn_lint_count
                    total_security_errors += security_count
                    total_deployment_errors += deployment_count

                row_extra["total_yaml_errors"] = total_yaml_errors
                row_extra["total_cfn_lint_errors"] = total_cfn_lint_errors
                row_extra["total_security_errors"] = total_security_errors
                # row_extra["total_deployment_errors"] = total_deployment_errors

                row_extra["all_yaml_errors"] = " | ".join(all_yaml_errors) if all_yaml_errors else "None"
                row_extra["all_cfn_lint_errors"] = " | ".join(all_cfn_lint_errors) if all_cfn_lint_errors else "None"
                row_extra["all_security_errors"] = " | ".join(all_security_errors) if all_security_errors else "None"
                # row_extra["all_deployment_errors"] = " | ".join(all_deployment_errors) if all_deployment_errors else "None"

                last_iteration = rem_history[-1]
                for err_stage in last_iteration.get("errors", []):
                    stage_name = err_stage.get("stage", "unknown")
                    stage_errors = err_stage.get("errors", [])
                    
                    # Create a specific column for this stage's errors
                    if not err_stage.get("passed", True) and stage_errors:
                        row_extra[f"latest_error_{stage_name}"] = " | ".join(stage_errors)
                    else:
                        row_extra[f"latest_error_{stage_name}"] = "None"

            # Extract specific validation stage passes (yaml, cfn-lint, etc.)
            val_results = report.get("validation_results", [])
            for stage_res in val_results:
                stage_name = stage_res.get("stage", "unknown")
                row_extra[f"val_stage_{stage_name}_passed"] = stage_res.get("passed", False)

            aggregated_extra_data.append(row_extra)
    else:
        print(f"Warning: The directory '{base_dir}' does not exist. No extra data will be merged.")

    # 3. Merge the dataframes
    if aggregated_extra_data:
        df_extra = pd.DataFrame(aggregated_extra_data)
        
        # Merge on 'run_id' using a left join so we keep all rows from results.csv
        df_merged = pd.merge(df_results, df_extra, on="run_id", how="left")
    else:
        print("No extra report data found to merge. Saving a copy of the input CSV.")
        df_merged = df_results.copy()

    # 4. Save the merged dataframe
    df_merged.to_csv(output_csv, index=False)
    print(f"Successfully created merged results at '{output_csv}'")

def merge_results(csv_paths, merged_csv_path, jsonl_paths=None, merged_jsonl_path=None):
    """
    Merges multiple CSV and JSONL files from the same model into single files.
    """
    # Merge CSVs while preserving original row order across files.
    if csv_paths:
        dfs = []
        for file_idx, f in enumerate(csv_paths):
            if os.path.exists(f):
                df = pd.read_csv(f)
                df = df.reset_index(drop=True)
                df["_file_order"] = file_idx
                df["_row_order"] = df.index
                dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)

            # If a `row_number` column exists, prefer sorting by it across all files.
            if "row_number" in merged_df.columns:
                merged_df["row_number"] = pd.to_numeric(merged_df["row_number"], errors="coerce")
                merged_df = merged_df.sort_values(
                    by=["row_number", "_file_order", "_row_order"],
                    na_position="last",
                ).reset_index(drop=True)
            else:
                # Fall back to file-order then original within-file order
                merged_df = merged_df.sort_values(by=["_file_order", "_row_order"]).reset_index(drop=True)

            # Remove helper ordering columns before saving
            drop_cols = [c for c in ("_file_order", "_row_order") if c in merged_df.columns]
            if drop_cols:
                merged_df = merged_df.drop(columns=drop_cols)

            merged_df.to_csv(merged_csv_path, index=False)
            print(f"Merged {len(dfs)} CSVs into {merged_csv_path}")
        else:
            print("No valid CSV files found to merge.")
    
    # Merge JSONLs
    if jsonl_paths and merged_jsonl_path:
        valid_jsonls = [f for f in jsonl_paths if os.path.exists(f)]
        if valid_jsonls:
            with open(merged_jsonl_path, 'w', encoding='utf-8') as outfile:
                for fname in valid_jsonls:
                    with open(fname, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
            print(f"Merged {len(valid_jsonls)} JSONLs into {merged_jsonl_path}")


def move_run_folders_from_csv(
    input_csv="results_merged.csv",
    runs_dir="runs",
    target_subfolder_name="moved_runs",
    run_id_col="run_id",
    dry_run=True,
):
    """Move folders under ``runs_dir`` whose folder name exactly matches any
    value in the ``run_id_col`` column of ``input_csv`` into a single
    subfolder named ``target_subfolder_name``.

    Parameters:
    - input_csv: path to CSV that contains a column with run IDs.
    - runs_dir: base directory to search for run folders (searched recursively).
    - target_subfolder_name: name of the subfolder inside runs_dir to move matches into.
    - run_id_col: column name in the CSV containing run IDs.
    - dry_run: if True, only print planned moves and do not perform them.

    Returns a list of (src_path, dest_path) for folders that would be / were moved.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str)
    if run_id_col not in df.columns:
        raise ValueError(f"Column '{run_id_col}' not found in {input_csv}")

    run_ids = set(df[run_id_col].dropna().astype(str).str.strip())
    if not run_ids:
        return []

    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    target_dir = os.path.join(runs_dir, target_subfolder_name)
    os.makedirs(target_dir, exist_ok=True)

    moved = []

    # Walk the runs directory tree looking for directories whose basename matches a run_id
    for root, dirs, files in os.walk(runs_dir):
        # Skip descending into the target folder itself
        if os.path.abspath(root).startswith(os.path.abspath(target_dir)):
            continue

        # iterate over a copy because we may modify dirs in-place
        for d in list(dirs):
            if d in run_ids:
                src = os.path.join(root, d)
                dest = os.path.join(target_dir, d)

                # If already in target directory, skip
                if os.path.abspath(src) == os.path.abspath(dest):
                    dirs.remove(d)
                    continue

                # Avoid overwriting an existing folder in the target
                if os.path.exists(dest):
                    dest = dest + f".dup_{int(time.time())}"

                if dry_run:
                    print(f"[DRY RUN] Would move: '{src}' -> '{dest}'")
                else:
                    shutil.move(src, dest)
                    print(f"Moved: '{src}' -> '{dest}'")

                moved.append((src, dest))

                # Prevent os.walk from descending into the moved directory
                try:
                    dirs.remove(d)
                except ValueError:
                    pass

    return moved

if __name__ == "__main__":
    # filter_runtime_error_rows(
    #     input_csv='benchmark_runs/terraform_20260813_214252/results.csv',
    #     output_csv='benchmark_runs/terraform_20260813_214252/results_without_runtime_error.csv',
    #     status_col='status',
    # )

    merge_results_from_directory(
        base_dir="./benchmark_runs/terraform_20260813_001412 TFEval",
    )

    # move_run_folders_from_csv(
    #     input_csv='benchmark_runs/terraform_20260801_141119 NeoDPIaCEval DeepseekV4Flash/results_merged.csv',
    #     runs_dir='runs',
    #     target_subfolder_name='NeoDPIaCEval_TF_DeepseekV4Flash_security_runs',
    #     run_id_col='run_id',
    #     dry_run=False,
    # )

    # merge_results_with_reports(
    #     input_csv="benchmark_runs/terraform_20260801_141119 NeoDPIaCEval DeepseekV4Flash/results_merged.csv", 
    #     base_dir="runs/NeoDPIaCEval_TF_DeepseekV4Flash_security_runs", 
    #     output_csv="benchmark_runs/terraform_20260801_141119 NeoDPIaCEval DeepseekV4Flash/NeoDPIaCEval_TF_DeepseekV4Flash_security_runs.csv"
    # )

    # merge_results(
    #     csv_paths=[
    #         './benchmark_runs/20260508_013226 Retriever + Security/results.csv', 
    #         './benchmark_runs/20260509_012338/results.csv',
    #         './benchmark_runs/20260509_141904/results.csv',
    #         './benchmark_runs/20260509_235120/results.csv',
    #         './benchmark_runs/20260510_095430/results.csv',
    #         './benchmark_runs/20260510_181824/results.csv',
    #         './benchmark_runs/20260511_025801/results.csv',
    #         './benchmark_runs/20260511_153550/results.csv',
    #         './benchmark_runs/20260511_194305/results.csv',
    #     ], 
    #     merged_csv_path='./benchmark_runs/20260508_013226 Retriever + Security/results_merged.csv',
    #     jsonl_paths=[
    #         './benchmark_runs/20260508_013226 Retriever + Security/results.jsonl', 
    #         './benchmark_runs/20260509_012338/results.jsonl',
    #         './benchmark_runs/20260509_141904/results.jsonl',
    #         './benchmark_runs/20260509_235120/results.jsonl',
    #         './benchmark_runs/20260510_095430/results.jsonl',
    #         './benchmark_runs/20260510_181824/results.jsonl',
    #         './benchmark_runs/20260511_025801/results.jsonl',
    #         './benchmark_runs/20260511_153550/results.jsonl',
    #         './benchmark_runs/20260511_194305/results.jsonl',
    #     ],
    #     merged_jsonl_path='./benchmark_runs/20260508_013226 Retriever + Security/results_merged.jsonl'
    # )

    # merge_results(
    #     csv_paths=[
    #         'benchmark_runs/20260515_010830 o3mini deploy/results.csv', 
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_115058 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_170603 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_212014 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_004211 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_130833 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_230459 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260517_140710 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260517_200045 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260519_234526 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_003217 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_183435 o3mini deploy/results.csv',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_225840 o3mini deploy/results.csv',
    #     ], 
    #     merged_csv_path='./benchmark_runs/20260515_010830 o3mini deploy/results_merged.csv',
    #     jsonl_paths=[
    #         'benchmark_runs/20260515_010830 o3mini deploy/results.jsonl', 
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_115058 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_170603 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260515_212014 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_004211 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_130833 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260516_230459 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260517_140710 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260517_200045 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260519_234526 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_003217 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_183435 o3mini deploy/results.jsonl',
    #         'benchmark_runs/20260515_010830 o3mini deploy/20260520_225840 o3mini deploy/results.jsonl',
    #     ],
    #     merged_jsonl_path='./benchmark_runs/20260515_010830 o3mini deploy/results_merged.jsonl'
    # )

    # merge_results(
    #     csv_paths=[
    #         './benchmark_runs/20260430_025759 RAG Tool/results.csv', 
    #         './benchmark_runs/20260505_014850 RAG Tool/results.csv',
    #         './benchmark_runs/20260505_202135 RAG Tool/results.csv',
    #         './benchmark_runs/20260506_104349 RAG Tool/results.csv',
    #         './benchmark_runs/20260507_002939 RAG Tool/results.csv',
    #     ], 
    #     merged_csv_path='./benchmark_runs/20260430_025759 RAG Tool/results_merged.csv',
    #     jsonl_paths=[
    #         './benchmark_runs/20260430_025759 RAG Tool/results.jsonl', 
    #         './benchmark_runs/20260505_014850 RAG Tool/results.jsonl',
    #         './benchmark_runs/20260505_202135 RAG Tool/results.jsonl',
    #         './benchmark_runs/20260506_104349 RAG Tool/results.jsonl',
    #         './benchmark_runs/20260507_002939 RAG Tool/results.jsonl',
    #     ],
    #     merged_jsonl_path='./benchmark_runs/20260430_025759 RAG Tool/results_merged.jsonl'
    # )
    # merge_results(
    #     csv_paths=[
    #         './benchmark_runs/20260427_163601 Grok Hybrid 15-Itr 0-29/results.csv', 
    #         './benchmark_runs/20260428_120119 Grok Hybrid 15-Itr 30-47/results.csv',
    #         './benchmark_runs/20260428_140414 Grok Hybrid 15-Itr 48-60/results.csv',
    #         './benchmark_runs/20260428_170008 Grok Hybrid 15-Itr 61-91/results.csv',
    #         './benchmark_runs/20260428_230436 Grok Hybrid 15-Itr 92-112/results.csv',
    #         './benchmark_runs/20260429_143042 Grok Hybrid 15-Itr 112-121/results.csv',
    #         './benchmark_runs/20260429_163803 Grok Hybrid 15-Itr 122-137/results.csv',
    #         './benchmark_runs/20260503_163359 Grok Hybrid 15-Itr 138-152/results.csv',
    #         './benchmark_runs/20260504_120843 Grok Hybrid 15-Itr 10/results.csv',
    #         './benchmark_runs/20260504_141843 Grok Hybrid 15-Itr 27/results.csv',
    #         './benchmark_runs/20260504_162919 Grok Hybrid 15-Itr 84/results.csv',
    #         './benchmark_runs/20260504_172854 Grok Hybrid 15-Itr 96/results.csv',
    #         './benchmark_runs/20260505_012857 Grok Hybrid 15-Itr 118/results.csv',
    #     ], 
    #     merged_csv_path='./benchmark_runs/20260427_163601 Grok Hybrid 15-Itr 0-29/results_merged.csv',
    #     jsonl_paths=[
    #         './benchmark_runs/20260427_163601 Grok Hybrid 15-Itr 0-29/results.jsonl', 
    #         './benchmark_runs/20260428_120119 Grok Hybrid 15-Itr 30-47/results.jsonl',
    #         './benchmark_runs/20260428_140414 Grok Hybrid 15-Itr 48-60/results.jsonl',
    #         './benchmark_runs/20260428_170008 Grok Hybrid 15-Itr 61-91/results.jsonl',
    #         './benchmark_runs/20260428_230436 Grok Hybrid 15-Itr 92-112/results.jsonl',
    #         './benchmark_runs/20260429_143042 Grok Hybrid 15-Itr 112-121/results.jsonl',
    #         './benchmark_runs/20260429_163803 Grok Hybrid 15-Itr 122-137/results.jsonl',
    #         './benchmark_runs/20260503_163359 Grok Hybrid 15-Itr 138-152/results.jsonl',
    #         './benchmark_runs/20260504_120843 Grok Hybrid 15-Itr 10/results.jsonl',
    #         './benchmark_runs/20260504_141843 Grok Hybrid 15-Itr 27/results.jsonl',
    #         './benchmark_runs/20260504_162919 Grok Hybrid 15-Itr 84/results.jsonl',
    #         './benchmark_runs/20260504_172854 Grok Hybrid 15-Itr 96/results.jsonl',
    #         './benchmark_runs/20260505_012857 Grok Hybrid 15-Itr 118/results.jsonl',
    #     ],
    #     merged_jsonl_path='./benchmark_runs/20260427_163601 Grok Hybrid 15-Itr 0-29/results_merged.jsonl'
    # )