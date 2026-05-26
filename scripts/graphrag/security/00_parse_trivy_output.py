#!/usr/bin/env python3
"""
00_parse_trivy_output.py

Parses Trivy's ``--format json`` output into typed TrivyFinding objects
that the retrieval pipeline can use as structured lookup keys.

Trivy JSON structure (relevant fields only)
-------------------------------------------
{
  "Results": [
    {
      "Target": "path/to/template.yaml",
      "Type": "cloudformation",
      "Misconfigs": [
        {
          "ID":       "AVD-AWS-0086",
          "Title":    "S3 Bucket does not have encryption enabled",
          "Severity": "HIGH",
          "CauseMetadata": {
            "Resource":  "AWS::S3::Bucket",
            "StartLine": 12
          }
        }
      ]
    }
  ]
}

Design
------
The ``check_id`` field (e.g. "AVD-AWS-0086") is the primary lookup key into
both the Neo4j SecurityCheck graph and the ChromaDB metadata index.

Path A in the retrieval pipeline is deterministic: given a known check_id,
no vector search is needed — a single Cypher traversal returns the full
context (description, impact, CFN remediation, good example, Rego policy,
CFN schema for the violating resource).

Path B (probabilistic ChromaDB fallback) is used only when check_id is not
found in the graph, which can happen for:
  - Custom Trivy checks not yet scraped into trivy_enriched.csv
  - Trivy findings with a non-standard ID format (e.g., user-defined)

Usage
-----
    # Programmatic
    from scripts.graphrag.security.parse_trivy_output import (
        parse_trivy_json, TrivyFinding
    )
    findings = parse_trivy_json(Path("trivy_output.json"))

    # CLI smoke-test
    python scripts/graphrag/security/00_parse_trivy_output.py \\
        --input path/to/trivy_output.json

Environment variables
---------------------
    None — this module is pure parsing with no network or DB calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrivyFinding:
    """A single misconfiguration finding produced by ``trivy config``.

    Attributes
    ----------
    check_id:
        AVD check identifier, e.g. "AVD-AWS-0086".  This is the primary
        lookup key into Neo4j (SecurityCheck.check_id) and ChromaDB metadata.
    cfn_resource_type:
        CloudFormation resource type reported by Trivy, e.g.
        "AWS::S3::Bucket".  Used by Path A to scope the CFN schema join.
        May be empty string if Trivy did not report a resource type.
    severity:
        One of CRITICAL / HIGH / MEDIUM / LOW / UNKNOWN (upper-case).
    title:
        Short human-readable title from Trivy, e.g.
        "S3 Bucket does not have encryption enabled".
    file_path:
        Path to the CFN template file that triggered the finding.
    start_line:
        Line number in the template file where the violation begins.
        0 if not reported by Trivy.
    """
    check_id:          str
    cfn_resource_type: str
    severity:          str
    title:             str
    file_path:         str = ""
    start_line:        int = 0

    def __post_init__(self) -> None:
        self.severity = (self.severity or "UNKNOWN").upper()

    @property
    def is_known_avd_id(self) -> bool:
        """True when check_id follows the canonical AVD pattern (AVD-XXX-NNNN)."""
        import re
        return bool(re.match(r'^AVD-[A-Z]+-\d+$', self.check_id))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_trivy_json(
    trivy_output_path: Path,
    *,
    target_type: str = "cloudformation",
) -> list[TrivyFinding]:
    """Parse a Trivy ``--format json`` output file into TrivyFinding objects.

    Parameters
    ----------
    trivy_output_path:
        Path to the JSON file produced by ``trivy config --format json``.
    target_type:
        Only process results whose ``Type`` field matches this value.
        Defaults to ``"cloudformation"`` — pass ``None`` to process all types.

    Returns
    -------
    list[TrivyFinding]
        Deduplicated list of findings ordered as they appear in the Trivy
        output (file order, then finding order within each file).  Findings
        with an empty check_id are silently skipped.
    """
    if not trivy_output_path.exists():
        raise FileNotFoundError(f"Trivy output not found: {trivy_output_path}")

    with trivy_output_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    findings: list[TrivyFinding] = []
    seen_keys: set[tuple[str, str]] = set()  # (check_id, file_path) dedup

    results = data.get("Results", [])
    if not results:
        # Trivy wraps the results array differently for some versions.
        # Try the top-level list fallback.
        results = data if isinstance(data, list) else []

    for result in results:
        result_type = (result.get("Type") or "").lower()
        if target_type is not None and result_type != target_type.lower():
            continue

        file_path = result.get("Target", "")
        misconfigs = result.get("Misconfigs") or result.get("MisconfSummary", [])

        # Trivy sometimes uses "Misconfigurations" as the key
        if not misconfigs:
            misconfigs = result.get("Misconfigurations", [])

        for mc in (misconfigs or []):
            check_id = (mc.get("ID") or mc.get("AVD-ID") or "").strip()
            if not check_id:
                continue

            cause = mc.get("CauseMetadata") or {}
            cfn_resource_type = (cause.get("Resource") or "").strip()
            start_line        = int(cause.get("StartLine") or cause.get("startLine") or 0)

            # Deduplicate on (check_id, file_path) — same check can appear
            # multiple times if the same resource is declared more than once.
            dedup_key = (check_id, file_path)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            findings.append(TrivyFinding(
                check_id          = check_id,
                cfn_resource_type = cfn_resource_type,
                severity          = (mc.get("Severity") or "UNKNOWN").upper(),
                title             = (mc.get("Title") or mc.get("Message") or "").strip(),
                file_path         = file_path,
                start_line        = start_line,
            ))

    return findings


def parse_trivy_findings_for_resources(
    trivy_output_path: Path,
    resource_types: set[str],
    *,
    target_type: str = "cloudformation",
) -> list[TrivyFinding]:
    """Parse and filter findings to a specific set of CFN resource types.

    Useful when the retrieval pipeline is called after selecting which
    resources in the template to focus on.

    Parameters
    ----------
    trivy_output_path:
        Path to the JSON file produced by ``trivy config --format json``.
    resource_types:
        Set of CFN resource type strings to keep, e.g.
        ``{"AWS::S3::Bucket", "AWS::IAM::Role"}``.
        Pass an empty set to return all findings (no filtering).
    target_type:
        Only process results whose ``Type`` field matches this value.

    Returns
    -------
    list[TrivyFinding]
        Filtered and deduplicated findings.
    """
    all_findings = parse_trivy_json(trivy_output_path, target_type=target_type)
    if not resource_types:
        return all_findings
    return [
        f for f in all_findings
        if f.cfn_resource_type in resource_types
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a Trivy --format json output file and print the structured "
            "TrivyFinding objects. Use this to verify that the parser correctly "
            "extracts check_id, cfn_resource_type, severity, and title from "
            "your Trivy scan output before running the retrieval pipeline."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the Trivy JSON output file.",
    )
    parser.add_argument(
        "--type", "-t",
        default="cloudformation",
        help="Filter by Trivy result type (default: cloudformation).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    try:
        findings = parse_trivy_json(input_path, target_type=args.type)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not findings:
        print("No findings found (check --type filter and Trivy output structure).")
        sys.exit(0)

    print(f"Parsed {len(findings)} finding(s) from: {input_path}\n")
    avd_count    = sum(1 for f in findings if f.is_known_avd_id)
    custom_count = len(findings) - avd_count
    print(f"  Known AVD IDs  : {avd_count}  (Path A — deterministic Neo4j lookup)")
    print(f"  Custom/unknown : {custom_count}  (Path B — ChromaDB semantic fallback)")
    print()

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    for f in sorted(findings, key=lambda x: sev_order.get(x.severity, 4)):
        known = "[AVD]" if f.is_known_avd_id else "[custom]"
        print(
            f"  {known:<10} {f.severity:<8} {f.check_id:<20} "
            f"{f.cfn_resource_type:<30} {f.title[:60]}"
        )
        if f.file_path:
            print(f"             file: {f.file_path}  line: {f.start_line}")


if __name__ == "__main__":
    main()
