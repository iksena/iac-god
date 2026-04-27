import re
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RULES_MD_URL = "https://raw.githubusercontent.com/aws-cloudformation/cfn-lint/main/docs/rules.md"
RAW_BASE = "https://raw.githubusercontent.com/aws-cloudformation/cfn-lint/main"
OUTPUT_PATH = "data/cfn_lint_rules.json"

def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode()
    except Exception:
        return None

def resolve_source_url(path: str) -> str:
    """Convert relative path like ../src/cfnlint/rules/... to raw GitHub URL."""
    # path starts with '../' relative to docs/, so strip the leading ../
    clean = re.sub(r'^\.\./', '', path)
    return f"{RAW_BASE}/{clean}"

def parse_rules(content: str) -> list[dict]:
    rules = []
    for line in content.splitlines():
        m = re.match(
            r'\|\s*\[([EWI]\d+)[^\]]*\]\(([^)]+)\)\s*\|'   # rule id + source path
            r'\s*([^|]+)\|'                                   # title
            r'\s*([^|]+)\|'                                   # description
            r'\s*([^|]*)\|'                                   # config
            r'\s*\[Source\]\(([^)]*)\)\s*\|'                 # docs source URL
            r'\s*([^|]*)\|',                                  # tags
            line
        )
        if not m:
            continue
        rule_id = m.group(1)
        source_path = m.group(2).strip()
        title = m.group(3).strip()
        description = m.group(4).strip()
        config = m.group(5).strip()
        docs_url = m.group(6).strip()
        tags_raw = m.group(7).strip()
        tags = re.findall(r'`([^`]+)`', tags_raw)

        rules.append({
            "id": rule_id,
            "title": title,
            "description": description,
            "config": config,
            "docs_url": docs_url,
            "tags": tags,
            "source_path": source_path,
            "source_raw_url": resolve_source_url(source_path),
            "source_code": None,  # filled in next step
        })
    return rules

def fetch_source_codes(rules: list[dict], max_workers: int = 10) -> list[dict]:
    def _fetch_one(rule):
        url = rule["source_raw_url"]
        code = fetch(url)
        return rule["id"], code

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, r): r["id"] for r in rules}
        for i, future in enumerate(as_completed(futures), 1):
            rule_id, code = future.result()
            for r in rules:
                if r["id"] == rule_id:
                    r["source_code"] = code
                    break
            if i % 50 == 0:
                print(f"  Fetched source for {i}/{len(rules)} rules...")

    return rules

# --- Main ---
print("Fetching cfn-lint rules.md ...")
md_content = fetch(RULES_MD_URL)
if not md_content:
    raise RuntimeError("Failed to fetch rules.md")

rules = parse_rules(md_content)
print(f"Parsed {len(rules)} rules.")

print("Fetching source code for each rule (parallel)...")
rules = fetch_source_codes(rules)

fetched = sum(1 for r in rules if r["source_code"])
print(f"Source code fetched: {fetched}/{len(rules)}")

Path("data").mkdir(exist_ok=True)
output = {r["id"]: r for r in rules}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"[JSON] Written {len(rules)} rules → {OUTPUT_PATH}")