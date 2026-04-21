"""
scrape_avd_remediation.py

Crawls the avd_docs/aws/ directory tree from the aquasecurity/trivy-checks
GitHub repository and builds a CSV mapping each check ID to its full
remediation description, links, and compliant code examples.

No web scraping needed — the AVD website is generated from these markdown
files, so reading them directly from GitHub is faster, reliable, and
version-pinned to the exact content on avd.aquasec.com.

URL structure discovered:
  avd_docs/aws/<service>/<AWS-XXXX>/docs.md          → description + links
  avd_docs/aws/<service>/<AWS-XXXX>/CloudFormation.md → CFn remediation + example
  avd_docs/aws/<service>/<AWS-XXXX>/Terraform.md      → TF remediation + example

Output CSV columns:
  check_id, avd_url, service, title, description, impact,
  remediation_cfn, remediation_tf, cfn_good_example,
  tf_good_example, links

Usage:
    python Code/tools/scrape_avd_remediation.py
    python Code/tools/scrape_avd_remediation.py --output Code/data/avd_remediation_map.csv
    python Code/tools/scrape_avd_remediation.py --service ec2 s3 iam
"""

import csv
import os
import re
import sys
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ───────────────────────────────────────────────────────────────────

GITHUB_API  = "https://api.github.com"
REPO_OWNER  = "aquasecurity"
REPO_NAME   = "trivy-checks"
BRANCH      = "main"
RAW_BASE    = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"
AVD_WEB_BASE = "https://avd.aquasec.com/misconfig/aws"

AVD_DOCS_ROOT = "avd_docs/aws"

OUTPUT_DIR  = Path("Code/data")
OUTPUT_FILE = OUTPUT_DIR / "avd_remediation_map.csv"

CSV_COLUMNS = [
    "check_id",          # e.g. AVD-AWS-0099
    "avd_url",           # canonical AVD website URL for reference
    "service",           # e.g. ec2, s3, iam
    "title",             # check title extracted from docs.md first line
    "description",       # why this is a problem (from docs.md)
    "impact",            # impact section (from docs.md)
    "remediation_cfn",   # one-line remediation action (from CloudFormation.md)
    "remediation_tf",    # one-line remediation action (from Terraform.md)
    "cfn_good_example",  # compliant CloudFormation YAML/JSON snippet
    "tf_good_example",   # compliant Terraform HCL snippet
    "links",             # pipe-separated reference URLs
]

MAX_WORKERS = 20

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
_token = os.environ.get("GITHUB_TOKEN")
if _token:
    HEADERS["Authorization"] = f"Bearer {_token}"


# ─── GitHub API helpers ────────────────────────────────────────────────────────

def get_full_tree() -> list[dict]:
    """
    Fetch the full recursive git tree and return all blobs under avd_docs/aws/.
    One API call covers everything — no pagination needed.
    """
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/{BRANCH}?recursive=1"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("truncated"):
        print("[WARN] GitHub tree was truncated — some docs may be missing.", file=sys.stderr)

    return [
        item for item in data.get("tree", [])
        if item["type"] == "blob"
        and item["path"].startswith(AVD_DOCS_ROOT + "/")
        and item["path"].endswith(".md")
    ]


def fetch_raw(path: str) -> str:
    """Fetch a raw file from GitHub with retry."""
    url = f"{RAW_BASE}/{path}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 2:
                print(f"[ERROR] Failed to fetch {path}: {e}", file=sys.stderr)
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


# ─── Markdown parsing ──────────────────────────────────────────────────────────

def parse_docs_md(content: str) -> dict:
    """
    Parse docs.md.

    Typical structure:
      <title line (sometimes)>

      <description paragraph(s)>

      ### Impact
      <impact text>

      {{ remediationActions }}   ← placeholder, replaced by framework .md files

      ### Links
      - https://...
    """
    result = {
        "title":       "",
        "description": "",
        "impact":      "",
        "links":       [],
    }
    if not content.strip():
        return result

    lines = content.splitlines()

    sections = {"description": [], "impact": [], "links": []}
    current = "description"

    for line in lines:
        stripped = line.strip()

        # Section headers
        if re.match(r"^#{1,3}\s+Impact", stripped, re.IGNORECASE):
            current = "impact"
            continue
        if re.match(r"^#{1,3}\s+Links?", stripped, re.IGNORECASE):
            current = "links"
            continue
        # Skip the remediation placeholder
        if "remediationActions" in stripped or "remediationAction" in stripped:
            continue
        # Skip horizontal rules / empty section separators
        if stripped in ("---", "***", "___"):
            continue

        if current == "links":
            # Collect bullet URLs
            url_match = re.search(r"https?://\S+", stripped)
            if url_match:
                sections["links"].append(url_match.group(0).rstrip(")>.,"))
        else:
            sections[current].append(line)

    # Extract title: first non-empty line of description that reads like a heading
    desc_lines = [l for l in sections["description"] if l.strip()]
    if desc_lines:
        first = desc_lines[0].strip().lstrip("#").strip()
        # Heuristic: if it's short (< 120 chars) and doesn't end with a period,
        # treat it as a title rather than description prose
        if len(first) < 120 and not first.endswith("."):
            result["title"] = first
            desc_lines = desc_lines[1:]

    result["description"] = "\n".join(desc_lines).strip()
    result["impact"]      = "\n".join(sections["impact"]).strip()
    result["links"]       = sections["links"]

    return result


def parse_framework_md(content: str) -> dict:
    """
    Parse a CloudFormation.md or Terraform.md file.

    Typical structure:
      <one-line remediation action>

      ```yaml / hcl
      <compliant code example>
      ```
    """
    result = {"remediation": "", "good_example": ""}
    if not content.strip():
        return result

    lines = content.splitlines()

    # Collect lines before the first code fence → remediation text
    # Collect content inside the first code fence → good_example
    pre_fence = []
    in_fence  = False
    fence_lines = []
    fence_found = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") and not fence_found:
            if in_fence:
                # closing fence
                fence_found = True
                in_fence = False
            else:
                in_fence = True
            continue
        if in_fence:
            fence_lines.append(line)
        elif not fence_found:
            pre_fence.append(line)

    # Remediation = first meaningful non-empty line before the code block
    remediation_lines = [l.strip() for l in pre_fence if l.strip()]
    result["remediation"] = remediation_lines if remediation_lines else ""

    result["good_example"] = "\n".join(fence_lines).strip()
    return result


# ─── Per-check processing ──────────────────────────────────────────────────────

def build_avd_url(service: str, check_id: str) -> str:
    """
    Construct the canonical AVD website URL.
    e.g. service=ec2, check_id=AVD-AWS-0099
    → https://avd.aquasec.com/misconfig/aws/ec2/aws-0099/
    """
    slug = check_id.lower().replace("avd-", "")  # avd-aws-0099 → aws-0099
    return f"{AVD_WEB_BASE}/{service}/{slug}/"


def process_check(check_id: str, service: str, file_map: dict[str, str]) -> dict:
    """
    Fetch and parse all three .md files for one check and return a CSV row.

    file_map: { "docs": raw_path, "cfn": raw_path, "tf": raw_path }
    """
    row = {col: "" for col in CSV_COLUMNS}
    row["check_id"] = check_id
    row["service"]  = service
    row["avd_url"]  = build_avd_url(service, check_id)

    # docs.md
    if file_map.get("docs"):
        docs = parse_docs_md(fetch_raw(file_map["docs"]))
        row["title"]       = docs["title"]
        row["description"] = docs["description"]
        row["impact"]      = docs["impact"]
        row["links"]       = " | ".join(docs["links"])

    # CloudFormation.md
    if file_map.get("cfn"):
        cfn = parse_framework_md(fetch_raw(file_map["cfn"]))
        row["remediation_cfn"]  = cfn["remediation"]
        row["cfn_good_example"] = cfn["good_example"]

    # Terraform.md
    if file_map.get("tf"):
        tf = parse_framework_md(fetch_raw(file_map["tf"]))
        row["remediation_tf"]  = tf["remediation"]
        row["tf_good_example"] = tf["good_example"]

    return row


# ─── Discovery: group tree blobs by check ─────────────────────────────────────

def discover_checks(
    tree_items: list[dict],
    service_filter: list[str] | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """
    Walk the git tree and group files by (service, check_id).

    Returns:
        { (service, "AVD-AWS-XXXX"): {"docs": path, "cfn": path, "tf": path} }

    Path structure: avd_docs/aws/<service>/<AWS-XXXX>/<filename>.md
    """
    checks: dict[tuple[str, str], dict[str, str]] = {}

    for item in tree_items:
        parts = item["path"].split("/")
        # Expected: ["avd_docs", "aws", <service>, <AWS-XXXX>, <file>.md]
        if len(parts) != 5:
            continue

        _, _, service, raw_check_id, filename = parts

        if service_filter and service not in service_filter:
            continue

        # Normalise to AVD-AWS-XXXX
        check_id = f"AVD-{raw_check_id.upper()}" if not raw_check_id.upper().startswith("AVD-") else raw_check_id.upper()

        key = (service, check_id)
        if key not in checks:
            checks[key] = {}

        fname_lower = filename.lower()
        if fname_lower == "docs.md":
            checks[key]["docs"] = item["path"]
        elif "cloudformation" in fname_lower:
            checks[key]["cfn"] = item["path"]
        elif "terraform" in fname_lower:
            checks[key]["tf"] = item["path"]
        # Other framework files (Kubernetes.md, Dockerfile.md) are ignored
        # but can be added here if needed

    return checks


# ─── Main ──────────────────────────────────────────────────────────────────────

def generate_avd_csv(
    output_path: Path = OUTPUT_FILE,
    service_filter: list[str] | None = None,
):
    print(f"[INFO] Fetching full git tree from {REPO_OWNER}/{REPO_NAME} ...")
    tree_items = get_full_tree()
    print(f"[INFO] Found {len(tree_items)} .md blobs under avd_docs/aws/")

    checks = discover_checks(tree_items, service_filter)
    print(f"[INFO] Discovered {len(checks)} distinct checks"
          + (f" (filtered to: {service_filter})" if service_filter else ""))

    if not checks:
        print("[ERROR] No checks discovered. Verify the repo structure or service filter.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []

    print(f"[INFO] Fetching markdown content with {MAX_WORKERS} parallel workers ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_check, check_id, service, file_map): (service, check_id)
            for (service, check_id), file_map in checks.items()
        }
        for i, future in enumerate(as_completed(future_map), 1):
            result = future.result()
            if result:
                rows.append(result)
            if i % 50 == 0 or i == len(checks):
                print(f"[INFO]   {i}/{len(checks)} checks processed ...")

    # Sort by check_id for deterministic output
    rows.sort(key=lambda r: r["check_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Wrote {len(rows)} rows to {output_path}")
    print(f"[INFO] Sample:")
    for r in rows[:3]:
        print(f"  {r['check_id']} ({r['service']}): {r['remediation_cfn'] or r['remediation_tf'] or '(no remediation text)'}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a CSV mapping Trivy/AVD check IDs to remediation descriptions "
                    "by reading avd_docs/ from aquasecurity/trivy-checks on GitHub."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--service", "-s",
        nargs="+",
        metavar="SERVICE",
        default=None,
        help="Filter to specific AWS services, e.g. --service ec2 s3 iam rds"
    )
    args = parser.parse_args()

    generate_avd_csv(
        output_path=args.output,
        service_filter=args.service,
    )