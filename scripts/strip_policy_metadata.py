#!/usr/bin/env python3
"""Remove leading metadata comment blocks from policy source_code in a CSV.

Usage:
  python scripts/strip_policy_metadata.py data/trivy_cfn_policy_map.csv --in-place
  python scripts/strip_policy_metadata.py data/trivy_cfn_policy_map.csv -o data/trivy_clean.csv
  python scripts/strip_policy_metadata.py data/trivy_cfn_policy_map.csv --merge-remediation data/avd_remediation_map.csv -o data/trivy_enriched.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def strip_leading_comment_block(source: str) -> str:
    """Drop only the initial run of comment lines and nearby blank lines.

    This preserves inline comments and any comments that appear after real code starts.
    """
    if not source:
        return source

    lines = source.splitlines()
    i = 0

    # Skip an initial comment header like:
    #   # METADATA
    #   # title: ...
    #   # ...
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("#"):
            i += 1
            continue

        if stripped == "":
            # Allow blanks while still in the leading metadata section.
            i += 1
            continue

        # First real code/content line.
        break

    # No leading comments found.
    if i == 0:
        return source

    # Trim any extra blank lines before the first real content.
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    return "\n".join(lines[i:])


def _normalize_check_id(check_id: str) -> str:
    raw = str(check_id or "").strip().upper()
    if raw.startswith("AVD-"):
        return raw.removeprefix("AVD-")
    return raw


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV appears to have no header: {path}")
        rows = list(reader)
    return fieldnames, rows


def process_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    fieldnames, rows = _read_csv_rows(input_path)
    if "source_code" not in fieldnames:
        raise ValueError("CSV must include a 'source_code' column")

    changed_rows = 0
    for row in rows:
        original = row.get("source_code", "")
        cleaned = strip_leading_comment_block(original)
        if cleaned != original:
            row["source_code"] = cleaned
            changed_rows += 1

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), changed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip leading metadata comments from source_code in a policy CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (required unless --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV with cleaned content",
    )
    parser.add_argument(
        "--merge-remediation",
        type=Path,
        default=None,
        help="Merge remediation columns from this CSV into the input policy CSV",
    )
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("Use either --in-place or --output, not both")
    if not args.in_place and args.output is None:
        parser.error("Provide --output or use --in-place")
    if args.merge_remediation is not None and not args.merge_remediation.exists():
        parser.error(f"Remediation CSV not found: {args.merge_remediation}")

    return args


def merge_policy_map_with_remediation(
    policy_path: Path,
    remediation_path: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    policy_fieldnames, policy_rows = _read_csv_rows(policy_path)
    remediation_fieldnames, remediation_rows = _read_csv_rows(remediation_path)

    if "check_id" not in policy_fieldnames:
        raise ValueError("Policy CSV must include a 'check_id' column")
    if "check_id" not in remediation_fieldnames:
        raise ValueError("Remediation CSV must include a 'check_id' column")

    remediation_lookup: dict[str, dict[str, str]] = {}
    for row in remediation_rows:
        normalized = _normalize_check_id(row.get("check_id", ""))
        if normalized:
            remediation_lookup[normalized] = row

    merged_fieldnames = list(policy_fieldnames)
    for fieldname in remediation_fieldnames:
        if fieldname not in merged_fieldnames:
            merged_fieldnames.append(fieldname)

    matched_rows = 0
    populated_cells = 0
    merged_rows: list[dict[str, str]] = []

    for row in policy_rows:
        merged_row = dict(row)
        normalized = _normalize_check_id(merged_row.get("check_id", ""))
        remediation_row = remediation_lookup.get(normalized)

        if remediation_row:
            matched_rows += 1
            for fieldname in remediation_fieldnames:
                if fieldname == "check_id":
                    continue

                remediation_value = str(remediation_row.get(fieldname) or "").strip()
                if not remediation_value:
                    continue

                if not str(merged_row.get(fieldname) or "").strip():
                    merged_row[fieldname] = remediation_value
                    populated_cells += 1

        for fieldname in merged_fieldnames:
            merged_row.setdefault(fieldname, "")

        merged_rows.append(merged_row)

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    return len(policy_rows), matched_rows, populated_cells


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_csv
    output_path: Path = input_path if args.in_place else args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.merge_remediation is not None:
        total_rows, matched_rows, populated_cells = merge_policy_map_with_remediation(
            input_path,
            args.merge_remediation,
            output_path,
        )
        print(
            f"Merged {total_rows} policy rows with {matched_rows} remediation matches; "
            f"populated {populated_cells} cells"
        )
        print(f"Wrote: {output_path}")
        return

    total_rows, changed_rows = process_csv(input_path, output_path)
    print(f"Processed {total_rows} rows; cleaned {changed_rows} source_code entries")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
