"""Convert the IaC-Eval dataset (autoiac-project/iac-eval, Terraform) into a
benchmark.py-compatible CSV: row_number, ground_truth_path, prompt, difficulty
— the same shape as data/tf_benchmark_diff_345.csv and the other existing
*_benchmark_*.csv files, so it can be run directly with:

    python benchmark.py --iac-type terraform --dataset data/iac_eval_benchmark.csv

Source columns (IaC-Eval's data.csv): Resource, Prompt, Rego intent,
Difficulty, Reference output, Intent.

Unlike the scraped-GitHub CFN/Terraform datasets, whose ground_truth_path
values point into an external corpus that isn't part of this repo,
IaC-Eval embeds its reference Terraform inline (the "Reference output"
column). Since we have that content, it's materialized to a local .tf file
per row under --ground-truth-dir and ground_truth_path points at it — so the
reference is actually inspectable, not just a label. Nothing in benchmark.py
reads this file back in; it's metadata only, exactly like the existing
datasets' ground_truth_path values.

By default this fetches the dataset fresh from HuggingFace
(https://huggingface.co/datasets/autoiac-project/iac-eval, split "test", 458
rows). Pass --input to convert a local copy instead (.csv or .parquet).
"""
import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

IAC_EVAL_CSV_URL = "https://huggingface.co/datasets/autoiac-project/iac-eval/resolve/main/data.csv"

REQUIRED_COLUMNS = ["Prompt", "Reference output", "Difficulty"]


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len] or "scenario"


def _load_source(input_path: Path | None) -> pd.DataFrame:
    if input_path is None:
        print(f"Fetching IaC-Eval dataset from {IAC_EVAL_CSV_URL} ...")
        resp = requests.get(IAC_EVAL_CSV_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    elif input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Source is missing required column(s) {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )
    return df


def convert(
    input_path: Path | None,
    output_csv: Path,
    ground_truth_dir: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    df = _load_source(input_path)
    if limit is not None:
        df = df.head(limit)

    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_number in range(len(df)):
        rec = df.iloc[row_number]
        prompt = str(rec["Prompt"]).strip()
        reference = str(rec["Reference output"])
        difficulty = rec["Difficulty"]
        resource = str(rec.get("Resource", "")) if "Resource" in df.columns else ""

        slug = _slugify(resource.split(",")[0]) if resource else "scenario"
        gt_filename = f"row_{row_number:04d}_{slug}.tf"
        gt_path = ground_truth_dir / gt_filename
        gt_path.write_text(reference, encoding="utf-8")

        rows.append({
            "row_number": row_number,
            "ground_truth_path": f"{ground_truth_dir.name}/{gt_filename}",
            "prompt": prompt,
            "difficulty": int(difficulty),
        })

    out_df = pd.DataFrame(rows, columns=["row_number", "ground_truth_path", "prompt", "difficulty"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the IaC-Eval (Terraform) dataset into a benchmark.py-compatible CSV."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Local .csv or .parquet copy of IaC-Eval. Omit to fetch fresh from HuggingFace.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/iac_eval_benchmark.csv"),
        help="Where to write the converted CSV (default: data/iac_eval_benchmark.csv).",
    )
    parser.add_argument(
        "--ground-truth-dir", type=Path, default=Path("data/iac_eval_ground_truth"),
        help="Directory to materialize per-row reference .tf files into.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only convert the first N rows (useful for a quick smoke test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_df = convert(args.input, args.output_csv, args.ground_truth_dir, args.limit)
    print(f"\nWrote {len(out_df)} rows to {args.output_csv}")
    print(f"Ground-truth Terraform files written to {args.ground_truth_dir}/")
    print("\nDifficulty distribution:")
    print(out_df["difficulty"].value_counts().sort_index())
