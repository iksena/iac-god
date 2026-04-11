import os
import json
import pandas as pd

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
            }

            # Extract full deploy_validation_result details
            deploy_res = report.get("deploy_validation_result", {})
            row_extra["deploy_target"] = deploy_res.get("target", "none")
            row_extra["deploy_passed"] = deploy_res.get("passed", False)
            row_extra["deploy_stack_id"] = deploy_res.get("stack_id", "None")
            row_extra["deploy_error_message"] = deploy_res.get("error_message") or "None"
            row_extra["deploy_duration_seconds"] = deploy_res.get("duration_seconds", 0.0)
            
            failed_res = deploy_res.get("failed_resources", [])
            row_extra["deploy_failed_resources"] = json.dumps(failed_res) if failed_res else "None"
            
            completed_res = deploy_res.get("completed_resources", [])
            row_extra["deploy_completed_resources"] = " | ".join(completed_res) if completed_res else "None"

            # Extract Latest Iteration Errors separated by stage
            rem_history = report.get("remediation_history", [])
            if rem_history:
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

if __name__ == "__main__":
    # You can change input_csv, base_dir, and output_csv as needed
    merge_results_with_reports(
        input_csv="benchmark_runs/CFN Schema Context Deterministic 2/results.csv", 
        base_dir="runs/archive/", 
        output_csv="benchmark_runs/CFN Schema Context Deterministic 2/results_merged.csv"
    )