#!/usr/bin/env python3
"""Merge benchmark attempts into final per-experiment results, check whether each
benchmark is fulfilled, and group run folders - in a way that a later rsync cannot disturb.

Layout (see also scripts/sync_from_server.sh):
  benchmark_runs/   server mirror. Read-only input; never edited here, may be overwritten by a sync.
  runs/             server mirror + organiser folders. Complete flat run folders are moved into
                    runs/<organizer>/<run_id>/ ; the sync script excludes every complete local run
                    (at any depth), so moved runs are not downloaded again.
  final_results/    OUTPUT of this script (local only, never synced), one folder per experiment:
                      results_final.csv   one row per benchmark row (best valid attempt), with final_template
                      attempts_all.csv    every attempt with its validity verdict
                      status.json         fulfilment: expected / valid / invalid / missing rows
                      rerun_rows.txt      rows that need a rerun (invalid + missing)
                      rerun.sh            command to rerun them
                    plus final_results/SUMMARY.csv|md across experiments.
Everything is recomputed from the raw inputs, so after each sync just run:
    scripts/finalize_results.py               # report + write final_results/, plan folder moves (dry run)
    scripts/finalize_results.py --apply       # also move complete flat run folders into organiser folders
    scripts/finalize_results.py --only CFN_RealAWS_OpenCode_DSV4F
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import tempfile

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "benchmark_runs")
RUNS = os.path.join(ROOT, "runs")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "final_results")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
csv.field_size_limit(sys.maxsize)

# ----------------------------------------------------------------------------------------------
# Manifest. dirs = top-level folders in benchmark_runs/ (their nested batch folders are included).
# dataset = benchmark CSV the experiment is supposed to cover; min_difficulty restricts to L3-5.
# organizer = folder under runs/ that collects the run folders of this experiment.
# ----------------------------------------------------------------------------------------------
CFN_REAL = "cfn_eval_benchmark_real_aws.csv"
CFN_REAL_345 = "cfn_eval_benchmark_real_aws_diff345.csv"
TF_REAL = "tf_eval_benchmark_real_aws.csv"
CFN_ABL = "cfn_eval_benchmark_ablation.csv"
TF_ABL = "tf_eval_benchmark_ablation.csv"
OC_CFN_CMD = """export OLLAMA_DUMMY_KEY=ollama
python -m baselines.harness_baseline --harness opencode --iac-type cloudformation \\
  --dataset data/{dataset} --rows "{rows}" --deploy-target aws --max-iterations 15 \\
  --model deepseek-v4-flash:cloud --base-url http://localhost:11434 --api-key-env OLLAMA_DUMMY_KEY \\
  --provider-name ollama --no-retry-proxy --scenario-timeout 14400 \\
  --runs-dir runs/OpencodeDSV4F_CFNEvalRealAWS_Full_runs \\
  --output-dir benchmark_runs/{out_dir} --keep-workspace"""

EXPERIMENTS = [
    # --- CloudFormation, live AWS, full real-AWS benchmark -----------------------------------
    dict(name="CFN_RealAWS_DSV4F", dirs=["cloudformation_20260917_170146_DSV4F_Full"], dataset=CFN_REAL,
         organizer="CFNEvalRealAWS_DeepseekV4Flash_sec_runs"),
    dict(name="CFN_RealAWS_GLM53F", dirs=["cloudformation_20260922_235243_GLM53Flash"], dataset=CFN_REAL,
         organizer="CFNEvalRealAWS_GLM53Flash_sec_runs"),
    dict(name="CFN_RealAWS_Gemini38F", dirs=["cloudformation_20260929_193317_Gemini38F"], dataset=CFN_REAL,
         organizer="CFNEvalRealAWS_Gemini38Flash_runs"),
    dict(name="CFN_RealAWS_OpenCode_DSV4F",
         dirs=["baseline_opencode_cloudformation_20260917_232736_DSV4F_Full",
               "baseline_opencode_cloudformation_20261002_212629", "baseline_opencode_cloudformation_20261002_213054"],
         dataset=CFN_REAL, organizer="OpencodeDSV4F_CFNEvalRealAWS_Full_runs",
         rerun_cmd=OC_CFN_CMD, rerun_out="baseline_opencode_cloudformation_20260917_232736_DSV4F_Full/batch4"),
    dict(name="CFN_RealAWS_Opus55", dirs=["cloudformation_20260925_001704_Opus55", "cloudformation_20260925_202548"],
         dataset=CFN_REAL, organizer="CFNEvalRealAWS_Opus55_runs"),
    # --- CloudFormation, older level 3-5 subsets / other benchmarks ---------------------------
    dict(name="CFN_RealAWS_Gemini36F_L345", dirs=["cloudformation_20260915_154934_Gemini36Flash"], dataset=CFN_REAL_345,
         organizer="Gemini36Flash_CFNEvalRealAWS_runs"),
    dict(name="CFN_RealAWS_DSV4F_DeployOnly_L345", dirs=["cloudformation_20260908_115729_DSV4F_Dep"], dataset=CFN_REAL_345,
         organizer="CFNEvalRealAWS_DeepseekV4Flash_deployability_runs"),
    dict(name="CFN_DPIaCEval_o3mini_DeployOnly", dirs=["cloudformation_20260922_232123_DPIaCEval_O3Mini"],
         dataset="iac_with_difficulty_levels.csv", organizer="DPIaCEval_O3Mini_deployability_runs"),
    # --- Terraform, live AWS ------------------------------------------------------------------
    dict(name="TF_RealAWS_DSV4F", dirs=["terraform_20260920_135237_DSV4_Full"], dataset=TF_REAL,
         organizer="TFEvalRealAWS_DeepseekV4Flash_sec_runs"),
    dict(name="TF_RealAWS_GLM53F", dirs=["terraform_20260925_102452_GLM53F", "terraform_20260927_143840", "terraform_20260928_120051"],
         dataset=TF_REAL, organizer="TFEvalRealAWS_GLM53Flash_sec_runs"),
    dict(name="TF_RealAWS_Gemini38F", dirs=["terraform_20260929_151853_Gemini38F"], dataset=TF_REAL,
         organizer="TFEvalRealAWS_Gemini38Flash_runs"),
    dict(name="TF_RealAWS_OpenCode_DSV4F", dirs=sorted(os.path.basename(p) for p in glob.glob(os.path.join(BENCH, "baseline_opencode_terraform_202609*"))),
         dataset=TF_REAL, organizer="TFEvalRealAWS_OpencodeDSV4F_runs"),
    dict(name="TF_RealAWS_Opus55", dirs=["terraform_20261002_115801"], dataset=TF_REAL, organizer="TFEvalRealAWS_Opus55_runs"),
    dict(name="TF_IaCEval_DSV4F_LintOnly", dirs=["terraform_20260908_171434_IaCEval"], dataset="iac_eval_benchmark.csv",
         organizer="IaCEval_DeepseekV4Flash_lint_runs", expected_n=372),
    # --- Ablations (LocalStack, 50 scenarios per language) ------------------------------------
    dict(name="CFN_Ablation_IaCGOD", dirs=["cloudformation_20260921_132308_Ablation_DSV4F", "cloudformation_20260928_223453_Ablation_Normal"],
         dataset=CFN_ABL, organizer="Ablation_IaCGOD_CFN_runs"),
    dict(name="CFN_Ablation_NoDenseRAG", dirs=["cloudformation_20260926_150304_Ablation_NoDenseRAG_DSV4F", "cloudformation_20260928_213100_Ablation_NoDenseRAG"],
         dataset=CFN_ABL, organizer="Ablation_NoDenseRAG_CFN_runs"),
    dict(name="CFN_Ablation_OpenCodeRetriever", dirs=["baseline_opencode_retriever_cloudformation_20260919_173443_Ablation_HarnessAndRetriever",
                                                      "baseline_opencode_retriever_cloudformation_20260929_011151", "baseline_opencode_retriever_cloudformation_20260929_011116"],
         dataset=CFN_ABL, organizer="Ablation_OpenCodeRetriever_CFN_runs"),
    dict(name="CFN_Ablation_NoGraphRAG", dirs=["cloudformation_20260929_235232_Ablation_NoGraphRAG"], dataset=CFN_ABL,
         organizer="Ablation_NoGraphRAG_CFN_runs"),
    dict(name="CFN_Ablation_NoPlanner", dirs=["cloudformation_20260930_130454_Ablation_NoPlanner"], dataset=CFN_ABL,
         organizer="Ablation_NoPlanner_CFN_runs"),
    dict(name="CFN_Ablation_TrivyAllSeverities", dirs=["cloudformation_20261001_225307_Ablation_TrivyAllSeverities"], dataset=CFN_ABL,
         organizer="Ablation_TrivyAllSeverities_CFN_runs"),
    dict(name="TF_Ablation_IaCGOD", dirs=["terraform_20260920_044840_Ablation_DSV4F", "terraform_20260929_002056_Ablation_Normal"],
         dataset=TF_ABL, organizer="Ablation_IaCGOD_TF_runs"),
    dict(name="TF_Ablation_NoDenseRAG", dirs=["terraform_20260926_174102_Ablation_NoDenseRAG_DSV4F", "terraform_20260927_162437",
                                              "terraform_20260927_162437_Ablation_NoDenseRAG", "terraform_20260928_001624"],
         dataset=TF_ABL, organizer="Ablation_NoDenseRAG_TF_runs"),
    dict(name="TF_Ablation_OpenCodeRetriever", dirs=["baseline_opencode_retriever_terraform_20260919_211653_Ablation_HarnessAndRetriever",
                                                     "baseline_opencode_retriever_terraform_20260929_053600"],
         dataset=TF_ABL, organizer="Ablation_OpenCodeRetriever_TF_runs"),
    dict(name="TF_Ablation_NoGraphRAG", dirs=["terraform_20260929_142359_Ablation_NoGraphRAG"], dataset=TF_ABL,
         organizer="Ablation_NoGraphRAG_TF_runs"),
    dict(name="TF_Ablation_NoPlanner", dirs=["terraform_20260930_152800_Ablation_NoPlanner"], dataset=TF_ABL,
         organizer="Ablation_NoPlanner_TF_runs"),
    dict(name="TF_Ablation_TrivyAllSeverities", dirs=["terraform_20261002_103230_Ablation_TrivyAllSeverities"], dataset=TF_ABL,
         organizer="Ablation_TrivyAllSeverities_TF_runs"),
    # Deliberately NOT listed: terraform_20260929_220511_Invalid_Ablation_NoGraphRAG (marked invalid), runs before 2026-09.
]

# ----------------------------------------------------------------------------------------------
# Validity: a result is INVALID when it says nothing about the model - the harness/pipeline died,
# or the failure is the environment's (provider install, disk, stalled/internal AWS errors,
# leftover resources). A passed run is always valid (env noise only inflates its iteration count).
# ----------------------------------------------------------------------------------------------
BAD_STATUS = {"harness_timeout": "harness timeout", "harness_error": "harness error", "harness_stalled": "harness stalled",
              "runtime_error": "pipeline runtime error"}
ENV_RE = re.compile("|".join([
    r"Failed to install provider", r"plugin-cache", r"text file busy", r"no space left on device",
    r"Could not connect to the endpoint", r"Connection was closed", r"already exists", r"AlreadyExists",
    r"Stalled in CREATE_IN_PROGRESS", r"HandlerErrorCode: InternalFailure", r"InternalError", r"Resource handler returned message: \"null\"",
    r"tflocal apply timed out", r"timed out after", r"Throttling", r"Rate exceeded", r"ServiceUnavailable",
    r"RequestLimitExceeded", r"LimitExceeded", r"No valid credential sources",
]), re.I)
ERR_COLS = ["error_message", "deploy_error_message", "latest_error_terraform-validate", "latest_error_tflint",
            "latest_error_cfn-lint", "latest_error_yaml", "latest_error_trivy"]


def _txt(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def classify(row) -> tuple[str, str]:
    status = _txt(row.get("status"))
    passed = str(row.get("final_validation_passed")).strip().lower() == "true"
    if passed:
        return "valid_pass", ""
    if status in BAD_STATUS:
        return "invalid_harness" if status.startswith("harness") else "invalid_runtime", BAD_STATUS[status]
    for col in ERR_COLS:
        m = ENV_RE.search(_txt(row.get(col)))
        if m:
            return "invalid_environment", f"{col}: {m.group(0)}"
    return "valid_fail", ""


RANK = {"valid_pass": 0, "valid_fail": 1, "invalid_environment": 2, "invalid_runtime": 3, "invalid_harness": 3}

# ----------------------------------------------------------------------------------------------


def read_csv_safe(path, **kw):
    return pd.read_csv(path, engine="python", **kw)


def load_attempts(exp) -> pd.DataFrame:
    frames, seen_files = [], set()
    for d in exp["dirs"]:
        base = os.path.join(BENCH, d)
        # sources: every batch results.csv, plus the merged/exported CSVs of the experiment folder
        # (older experiments only kept their complete table there). Derived subsets are skipped.
        cands = glob.glob(os.path.join(base, "**", "results.csv"), recursive=True)
        cands += glob.glob(os.path.join(base, "results_merged.csv"))
        cands += [q for q in glob.glob(os.path.join(base, "*.csv"))
                  if os.path.basename(q) not in ("results.csv", "results_merged.csv")
                  and not re.search(r"diff345|without_runtime|retry_error|\.bak", os.path.basename(q))]
        for p in cands:
            if p in seen_files:
                continue
            seen_files.add(p)
            try:
                x = read_csv_safe(p)
            except Exception as e:  # empty / unreadable file
                continue
            if x.empty or "row_number" not in x:
                continue
            x["_batch"] = os.path.relpath(p, BENCH)
            frames.append(x)
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames, ignore_index=True)
    a["_exp"] = exp["name"]
    # the same attempt can appear in a parent folder and in its nested/sibling copy
    has_id = a["run_id"].notna()
    a = pd.concat([a[has_id].drop_duplicates("run_id", keep="last"), a[~has_id]], ignore_index=True)
    return a


def expected_rows(exp) -> list[int]:
    ds = read_csv_safe(os.path.join(DATA, exp["dataset"]))
    return sorted(int(r) for r in ds["row_number"].dropna().unique())


def build_enrichment(thin: pd.DataFrame) -> pd.DataFrame:
    """final_report.json data for every run_id, in ONE pass over runs/ (organiser subfolders included)."""
    import aggregate_benchmark_run_data as agg
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "thin.csv"), os.path.join(tmp, "enriched.csv")
        thin.drop_duplicates("run_id")[["run_id"]].dropna().to_csv(src, index=False)
        agg.merge_results_with_reports(input_csv=src, base_dir=RUNS, output_csv=dst)
        return read_csv_safe(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="really move complete flat run folders into organiser folders")
    ap.add_argument("--only", nargs="*", help="experiment names to process")
    args = ap.parse_args()

    exps = [e for e in EXPERIMENTS if not args.only or e["name"] in args.only]
    attempts = {e["name"]: load_attempts(e) for e in exps}
    all_thin = pd.concat([a for a in attempts.values() if not a.empty], ignore_index=True)
    enrich = build_enrichment(all_thin)
    enrich_cols = [c for c in enrich.columns if c != "run_id"]
    os.makedirs(OUT, exist_ok=True)

    summary, move_plan = [], []
    for e in exps:
        a = attempts[e["name"]]
        out = os.path.join(OUT, e["name"])
        os.makedirs(out, exist_ok=True)
        exp_rows = expected_rows(e)
        if a.empty:
            a = pd.DataFrame(columns=["row_number", "run_id", "status", "final_validation_passed"])
        a = a.copy()
        # enrich: report data wins; keep results.csv columns that the report does not have
        report = enrich.set_index("run_id")
        extra = [c for c in report.columns if c not in a.columns]
        a = a.merge(report[extra], left_on="run_id", right_index=True, how="left") if extra else a
        # columns that exist in both: the CSV export may hold blanks (e.g. final_template lost on export) -> fill from final_report.json
        for c in [c for c in report.columns if c in a.columns and c != "run_id"]:
            blank = a[c].isna() | (a[c].astype(str).str.strip() == "")
            if blank.any():
                a[c] = a[c].astype(object).where(~blank, a["run_id"].map(report[c]).astype(object))
        verdict = a.apply(classify, axis=1, result_type="expand") if len(a) else pd.DataFrame(columns=[0, 1])
        a["validity"], a["invalid_reason"] = (verdict[0], verdict[1]) if len(a) else ([], [])
        a["_rank"] = a["validity"].map(RANK)
        a["_ts"] = a["run_id"].fillna("").astype(str)
        best = (a.sort_values(["row_number", "_rank", "_ts"], ascending=[True, True, False])
                  .drop_duplicates("row_number", keep="first"))
        in_scope = best[best.row_number.isin(exp_rows)]
        valid = in_scope[in_scope.validity.isin(["valid_pass", "valid_fail"])]
        invalid = in_scope[~in_scope.validity.isin(["valid_pass", "valid_fail"])]
        missing = sorted(set(exp_rows) - set(in_scope.row_number.astype(int)))
        expected_n = e.get("expected_n", len(exp_rows))
        rerun = sorted(set(invalid.row_number.astype(int)) | set(missing))
        if "expected_n" in e:           # intentionally subsetted benchmark: do not report excluded rows as missing
            missing_report = []
            rerun = sorted(set(invalid.row_number.astype(int)))
        else:
            missing_report = missing
        fulfilled = (len(valid) >= expected_n) if "expected_n" in e else (not invalid.shape[0] and not missing)
        st = dict(experiment=e["name"], dataset=e["dataset"], expected=expected_n, rows_with_attempt=int(len(in_scope)),
                  valid=int(len(valid)), valid_pass=int((valid.validity == "valid_pass").sum()),
                  valid_fail=int((valid.validity == "valid_fail").sum()), invalid=int(len(invalid)),
                  missing=len(missing_report), fulfilled=bool(fulfilled),
                  pass_rate_over_valid=round(float((valid.validity == "valid_pass").mean()), 4) if len(valid) else None,
                  pass_rate_over_expected=round(float((valid.validity == "valid_pass").sum() / expected_n), 4) if expected_n else None,
                  invalid_rows={int(r.row_number): r.invalid_reason for r in invalid.itertuples()},
                  missing_rows=missing_report, rerun_rows=rerun, attempts=int(len(a)),
                  extra_rows_outside_dataset=int((~best.row_number.isin(exp_rows)).sum()) if len(best) else 0,
                  organizer=e["organizer"])
        keep = [c for c in best.columns if not c.startswith("_")]
        in_scope[keep].sort_values("row_number").to_csv(os.path.join(out, "results_final.csv"), index=False)
        a[[c for c in ["row_number", "run_id", "status", "final_validation_passed", "iterations_used", "validity",
                       "invalid_reason", "_batch"] if c in a.columns]].sort_values(["row_number", "run_id"]).to_csv(
            os.path.join(out, "attempts_all.csv"), index=False)
        json.dump(st, open(os.path.join(out, "status.json"), "w"), indent=1)
        open(os.path.join(out, "rerun_rows.txt"), "w").write(",".join(map(str, rerun)) + "\n")
        write_rerun(e, out, rerun)
        summary.append({k: v for k, v in st.items() if k not in ("invalid_rows", "missing_rows", "rerun_rows")})
        # folder moves: only COMPLETE run folders that still sit flat under runs/
        for rid in a["run_id"].dropna().unique():
            src = os.path.join(RUNS, str(rid))
            if os.path.isdir(src) and os.path.exists(os.path.join(src, "final_report.json")):
                move_plan.append((src, os.path.join(RUNS, e["organizer"], str(rid)), e["name"]))

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "SUMMARY.csv"), index=False)
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write(s[["experiment", "expected", "valid", "valid_pass", "valid_fail", "invalid", "missing", "fulfilled",
                   "pass_rate_over_expected"]].to_markdown(index=False) + "\n")
    print(open(os.path.join(OUT, "SUMMARY.md")).read())

    # ---- group run folders ---------------------------------------------------------------
    todo = [(s_, d_, n) for s_, d_, n in move_plan if not os.path.exists(d_)]
    print(f"[group] {len(todo)} complete flat run folders to move into organiser folders "
          f"({'APPLYING' if args.apply else 'dry run, add --apply'})")
    by = pd.Series([n for *_, n in todo]).value_counts() if todo else pd.Series(dtype=int)
    print(by.to_string())
    if args.apply:
        for src, dst, _ in todo:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        print("[group] moved", len(todo))


def write_rerun(e, out, rerun):
    path = os.path.join(out, "rerun.sh")
    rows = ",".join(map(str, rerun))
    if not rerun:
        open(path, "w").write("# nothing to rerun: benchmark fulfilled\n")
        return
    try:
        sj = json.load(open(glob.glob(os.path.join(BENCH, e["dirs"][0], "summary.json"))[0]))
    except Exception:
        sj = {}
    ds = f"data/{e['dataset']}"
    if e.get("rerun_cmd"):
        cmd = e["rerun_cmd"].format(dataset=e["dataset"], rows=rows, out_dir=e["rerun_out"])
    elif sj.get("provider") == "harness":
        cmd = ("# harness run: fill in the provider flags you used originally (see the experiment's Notion page)\n"
               f"python -m baselines.harness_baseline --harness {sj.get('harness')} --iac-type {sj.get('iac_type')} --dataset {ds} "
               f"--rows \"{rows}\" --deploy-target {sj.get('deploy_target')} --max-iterations {sj.get('max_iterations')} "
               f"--model {sj.get('model')} --no-retry-proxy --scenario-timeout 14400 --keep-workspace \\\n"
               f"  --output-dir benchmark_runs/{e['dirs'][0]}/rerun_$(date +%Y%m%d_%H%M%S)")
    else:
        flags = [f"--iac-type {sj.get('iac_type')}", f"--dataset {ds}", f"--rows \"{rows}\"", f"--provider {sj.get('provider')}",
                 f"--model {sj.get('model')}", f"--deploy-target {sj.get('deploy_target')}",
                 f"--max-iterations {sj.get('max_iterations')}"]
        for k, f in (("openrouter_provider_only", "--openrouter-provider-only"),
                     ("openrouter_reasoning_max_tokens", "--openrouter-reasoning-max-tokens"),
                     ("openrouter_reasoning_effort", "--openrouter-reasoning-effort"),
                     ("openrouter_min_quantization", "--openrouter-min-quantization")):
            if sj.get(k):
                flags.append(f"{f} \"{sj[k]}\"")
        if sj.get("no_max_tokens"):
            flags.append("--no-max-tokens")
        if sj.get("skip_security"):
            flags.append("--skip-security")
        cmd = ("# reconstructed from summary.json of the first run folder; check against how you launched it\n"
               "python benchmark.py " + " ".join(flags) +
               f" \\\n  --output-dir benchmark_runs/{e['dirs'][0]}/rerun_$(date +%Y%m%d_%H%M%S)")
    open(path, "w").write(cmd + "\n")


if __name__ == "__main__":
    main()
