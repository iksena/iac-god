import os
import json
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
    # Merge CSVs
    if csv_paths:
        dfs = [pd.read_csv(f) for f in csv_paths if os.path.exists(f)]
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
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

if __name__ == "__main__":
    # You can change input_csv, base_dir, and output_csv as needed
    merge_results_with_reports(
        input_csv="./benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/results_merged.csv", 
        base_dir="runs/", 
        output_csv="./benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/grok_hybrid_rag_results_agg.csv"
    )
    merge_results(
        csv_paths=[
            './benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/results.csv', 
            './benchmark_runs/20260422_110630/results.csv',
            './benchmark_runs/20260422_135849/results.csv',
            './benchmark_runs/20260422_164425/results.csv',
            './benchmark_runs/20260422_185827/results.csv',
            './benchmark_runs/20260422_220303/results.csv',
            './benchmark_runs/20260423_012254/results.csv',
            './benchmark_runs/20260423_101506/results.csv',
            './benchmark_runs/20260423_103455/results.csv',
            './benchmark_runs/20260423_103742/results.csv',
        ], 
        merged_csv_path='./benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/results_merged.csv',
        jsonl_paths=[
            './benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/results.jsonl', 
            './benchmark_runs/20260422_110630/results.jsonl',
            './benchmark_runs/20260422_135849/results.jsonl',
            './benchmark_runs/20260422_164425/results.jsonl',
            './benchmark_runs/20260422_185827/results.jsonl',
            './benchmark_runs/20260422_220303/results.jsonl',
            './benchmark_runs/20260423_012254/results.jsonl',
            './benchmark_runs/20260423_103455/results.jsonl',
            './benchmark_runs/20260423_103742/results.jsonl',
        ],
        merged_jsonl_path='./benchmark_runs/20260422_012409 Hybrid RAG Grok4.1 Fast 0-27/results_merged.jsonl'
    )