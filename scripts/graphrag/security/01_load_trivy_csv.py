#!/usr/bin/env python3
"""
01_load_trivy_csv.py

Stage 1a – Parse trivy_enriched.csv into a clean security_checks.json.

Service → CFN prefix mapping strategy
---------------------------------------
Instead of a hardcoded dict (which would go stale), the mapping is derived
at runtime from two sources already in the repo:

  1. PRIMARY: data/cfn_spec.json
     ResourceTypes keys have the form 'AWS::<Namespace>::<ResourceType>'.
     We extract the <Namespace> segment (e.g. 'ApiGateway', 'S3') and
     case-insensitively match it against the Trivy service name.

  2. FALLBACK: data/trivy_cfn_policy_map.csv
     Some rows contain CFN YAML examples with 'Type: AWS::...' lines.
     We parse those to extract additional AWS:: prefixes.

Data quality
------------
impact fields are cleaned of HTML comment scaffold placeholders at load time
(AVD uses <!-- Add Impact here --> for empty fields).
Additional enrichment fields such as remediation_console, raw_text, and the
CloudFormation / Terraform examples are preserved in the JSON output so later
stages can use them directly.

No rows are filtered out. All 723 checks are valid public AVD data.
The prior "Get Demo" values in description were caused by an unscoped scraper
that picked up the nav CTA button; 00_scrape_avd_docs.py now scopes to
div.content.vulnerability_content and overwrites those stale values.

Output
------
  data/security_checks.json  – keyed by check_id
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "trivy_enriched.csv"
CFN_SPEC_PATH = REPO_ROOT / "data" / "cfn_spec.json"
TRIVY_CFN_MAP_PATH = REPO_ROOT / "data" / "trivy_cfn_policy_map.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "security_checks.json"

# HTML comment scaffold AVD uses as placeholder for empty impact fields.
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)


def clean_impact(raw: str) -> str:
    """Strip HTML comment placeholders from the impact field."""
    if not raw:
        return ""
    return _HTML_COMMENT_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Build service → CFN prefix map
# ---------------------------------------------------------------------------

def build_service_to_cfn_prefix(cfn_spec_path: Path, trivy_map_path: Path) -> dict[str, str]:
    namespace_to_prefix: dict[str, str] = {}

    if cfn_spec_path.exists():
        print(f"  Loading CFN spec from {cfn_spec_path} ...")
        with cfn_spec_path.open(encoding="utf-8") as fh:
            try:
                spec = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Could not parse cfn_spec.json: {e}")
                spec = {}

        resource_types = (
            spec.get("ResourceTypes", {})
            or spec.get("resource_types", {})
            or spec
        )
        for key in resource_types:
            m = re.match(r'^AWS::([^:]+)::', str(key))
            if m:
                namespace = m.group(1)
                namespace_to_prefix[namespace.lower()] = f"AWS::{namespace}::"

        print(f"  Found {len(namespace_to_prefix)} unique CFN namespaces in spec.")
    else:
        print(f"  WARNING: cfn_spec.json not found at {cfn_spec_path}.")

    service_from_yaml: dict[str, set[str]] = defaultdict(set)

    if trivy_map_path.exists():
        with trivy_map_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                service = row.get("service", "").strip().lower()
                if not service:
                    continue
                for value in row.values():
                    for match in re.finditer(r'Type:\s*(AWS::[A-Za-z0-9]+::)', str(value)):
                        service_from_yaml[service].add(match.group(1))
        print(f"  Found YAML-derived CFN prefixes for {len(service_from_yaml)} services.")
    else:
        print(f"  WARNING: trivy_cfn_policy_map.csv not found.")

    mapping: dict[str, str] = dict(namespace_to_prefix)
    for service, prefixes in service_from_yaml.items():
        if service not in mapping:
            best = min(prefixes, key=lambda p: _levenshtein(service, p.split("::")[1].lower()))
            mapping[service] = best

    ALIASES = {
        "vpc": "AWS::EC2::",
        "elb": "AWS::ElasticLoadBalancing::",
        "elbv2": "AWS::ElasticLoadBalancingV2::",
        "msk": "AWS::MSK::",
        "mq": "AWS::AmazonMQ::",
        "neptune": "AWS::Neptune::",
        "glacier": "AWS::Glacier::",
    }
    for alias, prefix in ALIASES.items():
        if alias not in mapping:
            mapping[alias] = prefix

    return mapping


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def clean_list_field(raw: str) -> list[str]:
    if not raw or raw.strip() == "":
        return []
    inner = raw.strip().lstrip("[").rstrip("]")
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    return [a or b for a, b in items if a or b]


def clean_links_field(raw: str) -> list[str]:
    if not raw or raw.strip() == "":
        return []
    if raw.startswith("["):
        return clean_list_field(raw)
    return [u.strip() for u in raw.split("|") if u.strip()]


def load_csv(csv_path: Path, service_map: dict) -> dict:
    checks: dict = {}

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            check_id = row.get("check_id", "").strip()
            if not check_id:
                continue

            service = row.get("service", "").strip().lower()
            cfn_prefix = service_map.get(service, "")

            checks[check_id] = {
                "check_id": check_id,
                "check_name": row.get("check_name", "").strip(),
                "severity": row.get("severity", "").strip().upper(),
                "short_code": row.get("short_code", "").strip(),
                "description": row.get("description", "").strip(),
                "service": service,
                "cfn_resource_prefix": cfn_prefix,
                "framework": row.get("framework", "").strip(),
                "source_file_url": row.get("source_file_url", "").strip(),
                "source_code": row.get("source_code", "").strip(),
                "avd_url": row.get("avd_url", "").strip(),
                "title": row.get("title", "").strip(),
                # Clean HTML comment placeholders at load time
                "impact": clean_impact(row.get("impact", "")),
                "remediation_console": row.get("remediation_console", "").strip(),
                "remediation_cfn": clean_list_field(row.get("remediation_cfn", "")),
                "remediation_tf": clean_list_field(row.get("remediation_tf", "")),
                "cfn_good_example": row.get("cfn_good_example", "").strip(),
                "tf_good_example": row.get("tf_good_example", "").strip(),
                "raw_text": row.get("raw_text", "").strip(),
                "links": clean_links_field(row.get("links", "")),
            }

    return checks


def main():
    print("Building service → CFN prefix mapping from repository data...")
    service_map = build_service_to_cfn_prefix(CFN_SPEC_PATH, TRIVY_CFN_MAP_PATH)
    print(f"  Mapping covers {len(service_map)} service tokens.\n")

    print("  Derived mapping (sorted):")
    for svc, prefix in sorted(service_map.items()):
        print(f"    {svc:<30} →  {prefix}")

    print(f"\nReading CSV from: {CSV_PATH}")
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    checks = load_csv(CSV_PATH, service_map)
    print(f"\nLoaded {len(checks)} unique security checks.")

    severity_counts = Counter(c["severity"] for c in checks.values())
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev or '(empty)'}: {count}")

    unmapped = sorted({c["service"] for c in checks.values() if not c["cfn_resource_prefix"]})
    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} services have no CFN prefix resolved: {unmapped}")
        print("  Add them to the ALIASES dict in build_service_to_cfn_prefix() if needed.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(checks)} checks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
