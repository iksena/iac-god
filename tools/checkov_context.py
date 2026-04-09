# infraskill/checkov_context.py
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "checkov_cfn_policy_map.csv"


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
    candidates: list[str] = [base]

    # CKV/CKV2 may appear with either hyphens or underscores.
    if base.startswith("CKV"):
        candidates.append(base.replace("-", "_"))
        candidates.append(base.replace("_", "-"))

    # Trivy findings may emit AWS-#### while policy map uses AVD-AWS-####.
    if re.fullmatch(r"AWS-\d{4}", base):
        candidates.append(f"AVD-{base}")
    if re.fullmatch(r"AVD-AWS-\d{4}", base):
        candidates.append(base.removeprefix("AVD-"))

    # De-duplicate while preserving priority order.
    return list(dict.fromkeys(candidates))

@lru_cache(maxsize=1)
def _load_policy_map() -> dict[str, dict]:
    """Load CSV once and return {check_id: {check_name, source_code}}."""
    if not _CSV_PATH.exists():
        return {}
    with _CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {
            str(row["check_id"]).strip().upper(): row
            for row in reader
            if row.get("check_id")
        }

def get_checkov_policy_context(findings: list[Any]) -> str:
    """
    Given a list of failed Checkov ValidationFindings, return a
    formatted string containing the policy source code for each,
    suitable for embedding directly into a remediation prompt.
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
            description_block = f"\nDescription: {description}" if description else ""
            blocks.append(
                f"### [{row['check_id']}] {row['check_name']}{description_block}\n"
                f"```python\n{row['source_code']}\n```"
            )

    return "\n\n".join(blocks)