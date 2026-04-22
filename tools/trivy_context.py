# infraskill/trivy_context.py
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "trivy_enriched.csv"
_REMEDIATION_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "avd_remediation_map.csv"
)


def _extract_check_id(finding: Any) -> str:
    if isinstance(finding, dict):
        return str(finding.get("check_id") or finding.get("rule_id") or "").strip()
    return str(
        getattr(finding, "check_id", None) or getattr(finding, "rule_id", None) or ""
    ).strip()


def _id_candidates(check_id: str) -> list[str]:
    raw = (check_id or "").strip()
    if not raw:
        return []

    base = raw.upper()
    # Trivy IDs are supported only in AWS-#### or AVD-AWS-#### forms.
    if re.fullmatch(r"AWS-\d{4}", base):
        return [base, f"AVD-{base}"]
    if re.fullmatch(r"AVD-AWS-\d{4}", base):
        return [base, base.removeprefix("AVD-")]

    return []


@lru_cache(maxsize=1)
def _load_policy_map() -> dict[str, dict]:
    """Load CSV once and return {check_id: policy_row}."""
    if not _CSV_PATH.exists():
        return {}
    with _CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {
            str(row["check_id"]).strip().upper(): row
            for row in reader
            if row.get("check_id")
        }


@lru_cache(maxsize=1)
def _load_remediation_map() -> dict[str, dict]:
    """Load remediation CSV once and return {check_id: remediation_row}."""
    if not _REMEDIATION_CSV_PATH.exists():
        return {}
    with _REMEDIATION_CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {
            str(row["check_id"]).strip().upper(): row
            for row in reader
            if row.get("check_id")
        }


def get_trivy_policy_context(findings: list[Any]) -> str:
    """
    Given failed Trivy findings, return formatted source-code blocks for matched
    check IDs suitable for remediation prompts.
    """
    policy_map = _load_policy_map()
    blocks: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        check_id = _extract_check_id(finding)
        if not check_id or check_id in seen:
            continue
        seen.add(check_id)

        row = None
        for candidate in _id_candidates(check_id):
            row = policy_map.get(candidate)
            if row:
                break
        if row:
            description = str(row.get("description") or "").strip()
            remediation_cfn = str(row.get("remediation_cfn") or "").strip()
            cfn_good_example = str(row.get("cfn_good_example") or "").strip()
            description_block = f"\nDescription: {description}" if description else ""
            remediation_block = (
                f"\nRemediation (CloudFormation): {remediation_cfn}"
                if remediation_cfn
                else ""
            )
            cfn_example_block = (
                f"\nCloudFormation Good Example:\n```yaml\n{cfn_good_example}\n```"
                if cfn_good_example
                else ""
            )
            blocks.append(
                f"### [{row.get('check_id', check_id)}] {row.get('check_name', '')}"
                f"{description_block}{remediation_block}{cfn_example_block}\n"
                f"```rego\n{row.get('source_code', '')}\n```"
            )

    return "\n\n".join(blocks)


def get_trivy_remediation_context(findings: list[Any]) -> str:
    """
    Given failed Trivy findings, return remediation context from
    avd_remediation_map.csv including only check_id, title, description,
    remediation_cfn, and cfn_good_example.
    """
    remediation_map = _load_remediation_map()
    blocks: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        check_id = _extract_check_id(finding)
        if not check_id or check_id in seen:
            continue
        seen.add(check_id)

        row = None
        for candidate in _id_candidates(check_id):
            row = remediation_map.get(candidate)
            if row:
                break

        if row:
            mapped_check_id = str(row.get("check_id") or check_id).strip()
            title = str(row.get("title") or "").strip()
            description = str(row.get("description") or "").strip()
            remediation_cfn = str(row.get("remediation_cfn") or "").strip()
            cfn_good_example = str(row.get("cfn_good_example") or "").strip()

            header = f"### [{mapped_check_id}]"
            if title:
                header = f"{header} {title}"

            section_lines = [header]
            if description:
                section_lines.append(f"Description: {description}")
            if remediation_cfn:
                section_lines.append(f"Remediation (CloudFormation): {remediation_cfn}")
            if cfn_good_example:
                section_lines.append(
                    "CloudFormation Good Example:\n"
                    f"```yaml\n{cfn_good_example}\n```"
                )

            blocks.append("\n".join(section_lines))

    return "\n\n".join(blocks)
