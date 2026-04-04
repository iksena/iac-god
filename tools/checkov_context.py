# infraskill/checkov_context.py
from __future__ import annotations

import csv
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

@lru_cache(maxsize=1)
def _load_policy_map() -> dict[str, dict]:
    """Load CSV once and return {check_id: {check_name, source_code}}."""
    if not _CSV_PATH.exists():
        return {}
    with _CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["check_id"]: row for row in reader}

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
        row = policy_map.get(check_id)
        if row:
            blocks.append(
                f"### [{check_id}] {row['check_name']}\n"
                f"```python\n{row['source_code']}\n```"
            )

    return "\n\n".join(blocks)