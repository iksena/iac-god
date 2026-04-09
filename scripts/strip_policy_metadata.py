#!/usr/bin/env python3
"""Remove leading metadata comment blocks from policy source_code in a CSV.

Usage:
  python scripts/strip_policy_metadata.py data/trivy_cfn_policy_map.csv --in-place
  python scripts/strip_policy_metadata.py data/trivy_cfn_policy_map.csv -o data/trivy_clean.csv
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


def process_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV appears to have no header")
        if "source_code" not in fieldnames:
            raise ValueError("CSV must include a 'source_code' column")

        rows = list(reader)

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
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("Use either --in-place or --output, not both")
    if not args.in_place and args.output is None:
        parser.error("Provide --output or use --in-place")

    return args


def main() -> None:
    args = parse_args()
    input_path: Path = args.input_csv
    output_path: Path = input_path if args.in_place else args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    total_rows, changed_rows = process_csv(input_path, output_path)
    print(f"Processed {total_rows} rows; cleaned {changed_rows} source_code entries")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
