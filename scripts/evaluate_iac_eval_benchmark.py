import argparse
import csv
import json
import time
import tempfile
import subprocess
import shutil
import os
import sys
from pathlib import Path
from datasets import load_dataset

# Import your repository's existing deployment configuration and reset tools
from config import DeployConfig, DeployTarget
from tools.deploy_validator import _reset_target_state, _delete_all_non_default_vpcs

def parse_args():
    parser = argparse.ArgumentParser(description="Test Terraform deployability from autoiac-project/iac-eval with resume capabilities.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--deploy-target", choices=["localstack", "aws", "none"], default="localstack")
    parser.add_argument("--localstack-endpoint", type=str, default="http://localhost:4566")
    parser.add_argument("--aws-region", type=str, default="us-east-1")
    parser.add_argument("--aws-profile", type=str, default="default")
    parser.add_argument("--output-csv", type=Path, default=Path("tf_evaluation_results.csv"))
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for terraform apply")
    return parser.parse_args()

def validate_terraform_deployment(tf_code: str, target: str, localstack_endpoint: str, region: str, timeout: int) -> dict:
    """Executes the Terraform lifecycle in an ephemeral directory."""
    if target == "none":
        return {"passed": True, "duration_seconds": 0.0, "error_message": None, "logs": "Skipped deployment."}

    start_time = time.time()
    temp_dir = tempfile.mkdtemp(prefix="iac_eval_tf_")
    
    main_tf_path = Path(temp_dir) / "main.tf"
    main_tf_path.write_text(tf_code, encoding="utf-8")

    env = None
    if target == "localstack":
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = "test"
        env["AWS_SECRET_ACCESS_KEY"] = "test"
        env["AWS_REGION"] = region
        env["AWS_ENDPOINT_URL"] = localstack_endpoint
        env["TF_VAR_skip_credentials_validation"] = "true"
        env["TF_VAR_skip_metadata_api_check"] = "true"
        env["TF_VAR_skip_requesting_account_id"] = "true"

    logs = []
    passed = False
    error_message = None

    try:
        # 1. Terraform Init
        init_res = subprocess.run(
            ["terraform", "init", "-no-color"], 
            cwd=temp_dir, capture_output=True, text=True, env=env
        )
        logs.append(f"--- INIT STDOUT ---\n{init_res.stdout}\n--- INIT STDERR ---\n{init_res.stderr}")
        
        if init_res.returncode != 0:
            error_message = "Terraform Init Failed"
            return {"passed": False, "duration_seconds": round(time.time() - start_time, 2), "error_message": error_message, "logs": "\n".join(logs)}

        # 2. Terraform Apply
        apply_res = subprocess.run(
            ["terraform", "apply", "-auto-approve", "-no-color"], 
            cwd=temp_dir, capture_output=True, text=True, env=env, timeout=timeout
        )
        logs.append(f"--- APPLY STDOUT ---\n{apply_res.stdout}\n--- APPLY STDERR ---\n{apply_res.stderr}")

        if apply_res.returncode == 0:
            passed = True
        else:
            passed = False
            error_message = apply_res.stderr.strip().split("\n")[-1] if apply_res.stderr else "Apply failed with no stderr."

    except subprocess.TimeoutExpired:
        passed = False
        error_message = f"Terraform apply timed out after {timeout} seconds."
        logs.append(error_message)
    except Exception as e:
        passed = False
        error_message = f"Execution error: {str(e)}"
        logs.append(error_message)
    finally:
        # 3. Terraform Destroy (Cleanup)
        if passed or (error_message and "Terraform Init Failed" not in error_message):
            destroy_res = subprocess.run(
                ["terraform", "destroy", "-auto-approve", "-no-color"], 
                cwd=temp_dir, capture_output=True, text=True, env=env
            )
            logs.append(f"--- DESTROY STDOUT ---\n{destroy_res.stdout}\n--- DESTROY STDERR ---\n{destroy_res.stderr}")
        
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "passed": passed,
        "duration_seconds": round(time.time() - start_time, 2),
        "error_message": error_message,
        "logs": "\n".join(logs)
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Filter input CSV for rows where deployment passed and export prompts.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input results CSV file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("filtered_prompts.csv"), help="Path to save the output filtered CSV.")
    return parser.parse_args()

def filter_iac_eval():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    filtered_records = []

    print(f"Reading and processing '{args.input}'...")
    with args.input.open(mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        
        # Guard against column presence and subtle casing variations
        fieldnames = reader.fieldnames or []
        prompt_key = "Prompt" if "Prompt" in fieldnames else ("prompt" if "prompt" in fieldnames else None)
        deploy_key = "deploy_passed"
        
        if not prompt_key:
            print(f"Error: Missing 'Prompt' or 'prompt' column. Found fields: {fieldnames}", file=sys.stderr)
            sys.exit(1)
            
        if deploy_key not in fieldnames:
            print(f"Error: Missing '{deploy_key}' validation column. Found fields: {fieldnames}", file=sys.stderr)
            sys.exit(1)

        # Assign row_number first sequentially (1-indexed based on original dataset position)
        for original_row_idx, row in enumerate(reader, start=1):
            deploy_value = str(row[deploy_key]).strip().lower()
            
            # Filter rows where deploy_passed is True
            if deploy_value == "true":
                filtered_records.append({
                    "row_number": original_row_idx,
                    "prompt": row[prompt_key]
                })

    print(f"Filtering complete. Found {len(filtered_records)} matching records.")
    
    # Write out the target schema containing only row_number and prompt
    print(f"Writing filtered metrics to '{args.output}'...")
    with args.output.open(mode="w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["row_number", "prompt"])
        writer.writeheader()
        writer.writerows(filtered_records)

    print("Done execution successfully.")

def main():
    args = parse_args()
    
    deploy_config = DeployConfig(
        target=DeployTarget(args.deploy_target),
        localstack_endpoint=args.localstack_endpoint,
        aws_profile=args.aws_profile,
        aws_region=args.aws_region,
    )
    
    print(f"Loading dataset 'autoiac-project/iac-eval' (split: {args.split})...")
    dataset = load_dataset("autoiac-project/iac-eval", split=args.split)
    
    original_columns = dataset.column_names
    eval_columns = ["deploy_target", "deploy_passed", "deploy_duration_seconds", "error_message", "deployment_logs"]
    all_columns = original_columns + eval_columns

    # ---------------------------------------------------------
    # RESUME LOGIC: Check how many rows have already been evaluated
    # ---------------------------------------------------------
    evaluated_count = 0
    file_mode = "w"
    
    if args.output_csv.exists():
        with args.output_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1: # We have a header and at least one data row
                evaluated_count = len(rows) - 1
                file_mode = "a"
                
    if evaluated_count > 0:
        print(f"Found existing results file. Resuming from row {evaluated_count + 1}...")

    with args.output_csv.open(file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        if file_mode == "w":
            writer.writeheader()
        
        for i, row in enumerate(dataset):
            # Skip already evaluated rows
            if i < evaluated_count:
                continue
                
            print(f"\n[{i+1}/{len(dataset)}] Evaluating scenario...")
            
            # ---------------------------------------------------------
            # ENVIRONMENT RESET: Guarantee clean slate before apply
            # ---------------------------------------------------------
            if args.deploy_target != "none":
                print(f"  -> Resetting {args.deploy_target} state...")
                _reset_target_state(deploy_config)
                if args.deploy_target == "aws":
                    _delete_all_non_default_vpcs(deploy_config) # Free up VPC quotas in AWS

            tf_code = row.get("Reference output", "")
            
            if not tf_code.strip():
                print("  -> Skipping: No Terraform code found in 'Reference output'.")
                out_row = {**row, "deploy_passed": False, "error_message": "Empty template"}
                writer.writerow(out_row)
                csvfile.flush() # Ensure it writes immediately
                continue
                
            print(f"  -> Deploying via Terraform to {args.deploy_target}...")
            
            result = validate_terraform_deployment(
                tf_code=tf_code,
                target=args.deploy_target,
                localstack_endpoint=args.localstack_endpoint,
                region=args.aws_region,
                timeout=args.timeout
            )
            
            print(f"  -> Passed: {result['passed']} | Duration: {result['duration_seconds']}s")
            if not result['passed']:
                print(f"  -> Error: {result['error_message']}")

            out_row = {**row}
            out_row["deploy_target"] = args.deploy_target
            out_row["deploy_passed"] = result["passed"]
            out_row["deploy_duration_seconds"] = result["duration_seconds"]
            out_row["error_message"] = result["error_message"]
            out_row["deployment_logs"] = result["logs"]
            
            # ---------------------------------------------------------
            # INCREMENTAL SAVE
            # ---------------------------------------------------------
            writer.writerow(out_row)
            csvfile.flush() # Force OS to write buffer to disk
            
    print(f"\nEvaluation complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    # main()
    filter_iac_eval()