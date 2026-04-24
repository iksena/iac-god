# 07_test_g_retrieval_by_stage.py
"""
Tests G-Retrieval (GraphRAG pipeline) against representative error messages
sampled from each validation stage: yaml, cfn-lint, trivy (security), deploy.

Data sources:
  - grok_secure_neo4j_results_agg.csv
  - grok_deployable_results_agg.csv
"""

import pandas as pd
import json
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from execute_g_retrieval import execute_g_retrieval

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATHS = [
    "benchmark_runs/Deployable Grok4.1Fast 0-143/grok_deployable_results_agg.csv",
    "benchmark_runs/20260419_002337 Secure Neo4J Grok4.1 Fast/grok_secure_neo4j_results_agg.csv",
]

STAGE_COLUMNS = {
    "yaml":     "latest_error_yaml",
    "cfn-lint": "latest_error_cfn-lint",
    "trivy":    "latest_error_trivy",
    "deploy":   "deploy_error_message",
}

# How many samples to draw per stage (stratified across both CSVs)
SAMPLES_PER_STAGE = 5

# Exclude boilerplate non-errors
SKIP_PREFIXES = ["Skipped:", "Unexpected error: 'reason'"]

OUTPUT_DIR = Path("./g_retrieval_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class StageQuery:
    stage: str
    source_csv: str
    raw_error: str
    natural_query: str  # transformed for G-Retrieval

@dataclass
class RetrievalResult:
    stage: str
    source_csv: str
    raw_error: str
    natural_query: str
    retrieved_resources: list[str]
    num_chunks_retrieved: int
    prompt_length: int
    prompt_snippet: str
    success: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────
# STAGE 1: LOAD AND SAMPLE ERRORS
# ─────────────────────────────────────────────
def load_and_sample(csv_paths: list[str], stage_cols: dict, n: int) -> list[StageQuery]:
    """
    Load CSVs, filter real errors per stage, sample n per stage,
    and return StageQuery objects with natural language queries.
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["_source"] = Path(path).stem
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    queries = []
    for stage, col in stage_cols.items():
        if col not in combined.columns:
            print(f"[WARN] Column '{col}' not found — skipping stage '{stage}'")
            continue

        errors = combined[col].dropna()
        errors = errors[~errors.apply(
            lambda e: any(e.startswith(prefix) for prefix in SKIP_PREFIXES)
        )]
        errors = errors.drop_duplicates()

        sampled = errors.sample(min(n, len(errors)), random_state=42)

        for raw_error in sampled:
            source_row = combined[combined[col] == raw_error].iloc[0]
            queries.append(StageQuery(
                stage=stage,
                source_csv=source_row["_source"],
                raw_error=raw_error,
                natural_query=error_to_natural_query(stage, raw_error),
            ))

    return queries


# ─────────────────────────────────────────────
# STAGE 2: ERROR → NATURAL LANGUAGE QUERY
# ─────────────────────────────────────────────
def error_to_natural_query(stage: str, error: str) -> str:
    """
    Converts a raw validator error string into a natural language
    query suitable for semantic search in G-Retrieval.

    Strategy per stage:
    - yaml:     Extract the rule tag + description → ask how to fix YAML syntax
    - cfn-lint: Extract the E/W code + property name → ask for correct CFN schema
    - trivy:    Extract the AWS rule ID + resource type → ask for secure config
    - deploy:   Use the AWS error message directly → ask how to fix the stack op
    """
    # Take only the first error if pipe-delimited
    first_error = error.split("|")[0].strip()

    if stage == "yaml":
        # e.g. "[indentation] line 71: wrong indentation: expected 12 but found 14"
        tag_match = re.search(r'\[(\w+)\]', first_error)
        desc_match = re.search(r':\s(.+?)(?:\s\(|$)', first_error)
        tag = tag_match.group(1) if tag_match else "syntax"
        desc = desc_match.group(1) if desc_match else first_error[:100]
        return (
            f"How do I fix a YAML {tag} error in a CloudFormation template? "
            f"The error is: {desc}"
        )

    elif stage == "cfn-lint":
        # e.g. "[E3002] ... Additional properties are not allowed ('KmsMasterKeyID' was unexpected)"
        code_match = re.search(r'\[(E|W\d+)\]', first_error)
        prop_match = re.search(r"'([\w:]+)' was unexpected", first_error)
        resource_match = re.search(r"Resource type '([\w:]+)'", first_error)
        code = code_match.group(0) if code_match else ""
        if prop_match:
            prop = prop_match.group(1)
            return (
                f"What is the correct CloudFormation property name? "
                f"cfn-lint {code} reports '{prop}' is not allowed or unexpected."
            )
        elif resource_match:
            res = resource_match.group(1)
            return (
                f"Does the CloudFormation resource type '{res}' exist? "
                f"cfn-lint {code} says it does not exist in the region."
            )
        else:
            return (
                f"How do I fix this CloudFormation cfn-lint {code} error: {first_error[:200]}"
            )

    elif stage == "trivy":
        # e.g. "[AWS-0028] HIGH: aws_instance should activate session tokens..."
        rule_match = re.search(r'\[(AWS-\d+)\]', first_error)
        severity_match = re.search(r'(CRITICAL|HIGH|MEDIUM|LOW):', first_error)
        desc_match = re.search(r'(?:CRITICAL|HIGH|MEDIUM|LOW):\s(.+?)\s—', first_error)
        rule = rule_match.group(1) if rule_match else "security rule"
        sev = severity_match.group(1) if severity_match else ""
        desc = desc_match.group(1) if desc_match else first_error[:150]
        return (
            f"How do I configure a CloudFormation resource to fix a {sev} security issue? "
            f"Trivy rule {rule}: {desc}"
        )

    elif stage == "deploy":
        # e.g. "An error occurred (ValidationError) when calling CreateStack..."
        return (
            f"How do I fix this AWS CloudFormation deployment error: {first_error[:300]}"
        )

    return first_error[:300]


# ─────────────────────────────────────────────
# STAGE 3: RUN G-RETRIEVAL + CAPTURE METRICS
# ─────────────────────────────────────────────
def run_retrieval_tests(queries: list[StageQuery]) -> list[RetrievalResult]:
    results = []
    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Stage={q.stage} | {q.natural_query[:80]}...")
        try:
            prompt = execute_g_retrieval(q.natural_query)

            # Parse retrieved resources from prompt (between CONTEXT: and USER QUERY:)
            resources = extract_resources_from_prompt(prompt)

            result = RetrievalResult(
                stage=q.stage,
                source_csv=q.source_csv,
                raw_error=q.raw_error[:300],
                natural_query=q.natural_query,
                retrieved_resources=resources,
                num_chunks_retrieved=len(resources),
                prompt_length=len(prompt) if prompt else 0,
                prompt_snippet=prompt[:500] if prompt else "",
                success=prompt is not None and len(prompt) > 100,
            )
        except Exception as e:
            result = RetrievalResult(
                stage=q.stage,
                source_csv=q.source_csv,
                raw_error=q.raw_error[:300],
                natural_query=q.natural_query,
                retrieved_resources=[],
                num_chunks_retrieved=0,
                prompt_length=0,
                prompt_snippet="",
                success=False,
                error=str(e),
            )
        results.append(result)
    return results


def extract_resources_from_prompt(prompt: str) -> list[str]:
    """Extract 'Extracting minimal subgraph for X' lines from G-Retrieval stdout."""
    if not prompt:
        return []
    # Resources appear in CONTEXT section headers, heuristic parse
    matches = re.findall(r'Resource:\s*([\w:]+)', prompt)
    return list(set(matches))


# ─────────────────────────────────────────────
# STAGE 4: REPORT
# ─────────────────────────────────────────────
def save_results(results: list[RetrievalResult]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSONL log
    jsonl_path = OUTPUT_DIR / f"g_retrieval_test_{ts}.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # Summary CSV
    df = pd.DataFrame([asdict(r) for r in results])
    df["retrieved_resources"] = df["retrieved_resources"].apply(lambda x: "; ".join(x))
    csv_path = OUTPUT_DIR / f"g_retrieval_summary_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # Print per-stage stats
    print("\n\n===== G-RETRIEVAL TEST SUMMARY =====")
    print(f"Total queries: {len(results)}")
    print(f"Success rate: {sum(r.success for r in results)}/{len(results)}")
    print()
    for stage in STAGE_COLUMNS.keys():
        stage_r = [r for r in results if r.stage == stage]
        if not stage_r:
            continue
        success = sum(r.success for r in stage_r)
        avg_len = sum(r.prompt_length for r in stage_r) / len(stage_r)
        avg_chunks = sum(r.num_chunks_retrieved for r in stage_r) / len(stage_r)
        print(f"[{stage:10s}] success={success}/{len(stage_r)} | "
              f"avg_prompt_len={avg_len:.0f} | avg_resources={avg_chunks:.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  → {jsonl_path.name}")
    print(f"  → {csv_path.name}")


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== G-Retrieval Stage-Based Error Query Tester ===\n")

    print("Step 1: Loading and sampling errors from CSVs...")
    queries = load_and_sample(CSV_PATHS, STAGE_COLUMNS, n=SAMPLES_PER_STAGE)
    print(f"Total queries to test: {len(queries)}")

    print("\nStep 2: Running G-Retrieval for each query...")
    results = run_retrieval_tests(queries)

    print("\nStep 3: Saving results...")
    save_results(results)