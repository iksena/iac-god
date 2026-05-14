#!/usr/bin/env python3
"""
02_scrape_avd_docs.py

Stage 1b – Scrape Aqua Security AVD pages for each check in security_checks.json.

For each check with an avd_url:
  - Fetch the AVD HTML page
  - Extract: page title, description paragraphs, impact block, remediation block,
    example code snippets (CloudFormation + Terraform)
  - Persist enriched HTML to data/scraped_avd/<check_id>.html
  - Merge enriched fields back into security_checks.json

This mirrors the pattern of 02_scrape_cfn_docs.py + 03_parse_and_merge.py in the CFN pipeline.

Usage:
    python scripts/graphrag/security/02_scrape_avd_docs.py

Dependencies: requests, beautifulsoup4
"""

import json
import sys
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: Install dependencies with: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKS_JSON = REPO_ROOT / "data" / "security_checks.json"
SCRAPED_DIR = REPO_ROOT / "data" / "scraped_avd"

REQUEST_DELAY = 0.5  # seconds between requests – be polite
SESSION_HEADERS = {
    "User-Agent": "iac-god-graphrag-builder/1.0 (research)",
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_page(url: str, session: requests.Session) -> str | None:
    """Fetch a URL with retry on transient errors. Returns raw HTML or None."""
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15, headers=SESSION_HEADERS)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"  404 – skipping {url}")
                return None
            print(f"  HTTP error {e} (attempt {attempt + 1}/3)")
        except requests.RequestException as e:
            print(f"  Request error: {e} (attempt {attempt + 1}/3)")
        time.sleep(1.0)
    return None


def extract_avd_content(html: str, check_id: str) -> dict:
    """
    Parse an AVD page and extract semantic fields.

    AVD page structure (as of 2025):
      - <h1> or <title> → page title
      - <div class="content"> or <article> → main body
      - Sections headed by <h2>/<h3>: Description, Impact, Resolution/Remediation
      - <code> or <pre> blocks for examples
    """
    soup = BeautifulSoup(html, "html.parser")
    extracted: dict = {"check_id": check_id}

    # --- Title
    h1 = soup.find("h1")
    extracted["page_title"] = h1.get_text(strip=True) if h1 else ""

    # --- Description: first 3 non-empty paragraphs from main content
    main = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.body
    paragraphs = []
    if main:
        for p in main.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)
                if len(paragraphs) >= 3:
                    break
    extracted["page_description"] = " ".join(paragraphs)

    # --- Impact / Remediation blocks via section headings
    impact_texts = []
    remediation_texts = []

    if main:
        headings = main.find_all(["h2", "h3"])
        for heading in headings:
            heading_text = heading.get_text(strip=True).lower()
            # Collect sibling text until next heading
            sibling_texts = []
            for sib in heading.find_next_siblings():
                if sib.name in ("h2", "h3"):
                    break
                text = sib.get_text(separator=" ", strip=True)
                if text:
                    sibling_texts.append(text)

            if "impact" in heading_text:
                impact_texts.extend(sibling_texts)
            elif any(k in heading_text for k in ("remediat", "resolut", "fix", "how to")):
                remediation_texts.extend(sibling_texts)

    extracted["page_impact"] = " ".join(impact_texts)
    extracted["page_remediation"] = " ".join(remediation_texts)

    # --- Code examples: first CloudFormation YAML block and first Terraform block
    cfn_example = ""
    tf_example = ""
    code_blocks = main.find_all(["code", "pre"]) if main else []
    for block in code_blocks:
        text = block.get_text()
        if not cfn_example and ("Type: AWS::" in text or "AWSTemplateFormatVersion" in text):
            cfn_example = text.strip()
        elif not tf_example and ("resource \"aws_" in text or "provider \"aws\"" in text):
            tf_example = text.strip()
        if cfn_example and tf_example:
            break

    extracted["page_cfn_example"] = cfn_example
    extracted["page_tf_example"] = tf_example

    return extracted


def main():
    if not CHECKS_JSON.exists():
        print(f"ERROR: Run 01_load_trivy_csv.py first. Missing: {CHECKS_JSON}", file=sys.stderr)
        sys.exit(1)

    with CHECKS_JSON.open(encoding="utf-8") as fh:
        checks: dict = json.load(fh)

    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    total = len(checks)
    enriched = 0
    skipped = 0

    for i, (check_id, check) in enumerate(checks.items(), 1):
        avd_url = check.get("avd_url", "").strip()
        if not avd_url:
            skipped += 1
            continue

        out_file = SCRAPED_DIR / f"{check_id}.html"

        # Skip re-download if already cached
        if out_file.exists():
            print(f"[{i}/{total}] {check_id} – cached")
            html = out_file.read_text(encoding="utf-8")
        else:
            print(f"[{i}/{total}] {check_id} – fetching {avd_url}")
            html = fetch_page(avd_url, session)
            if html is None:
                skipped += 1
                continue
            out_file.write_text(html, encoding="utf-8")
            time.sleep(REQUEST_DELAY)

        # Extract and merge enriched fields
        extracted = extract_avd_content(html, check_id)
        check["page_title"] = extracted["page_title"]
        check["page_description"] = extracted["page_description"]
        check["page_impact"] = extracted["page_impact"]
        check["page_remediation"] = extracted["page_remediation"]
        # Only overwrite example if CSV was empty
        if not check.get("cfn_good_example"):
            check["cfn_good_example"] = extracted["page_cfn_example"]
        if not check.get("tf_good_example"):
            check["tf_good_example"] = extracted["page_tf_example"]

        enriched += 1

    # Write enriched JSON back
    with CHECKS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, ensure_ascii=False)

    print(f"\nDone. Enriched: {enriched}, Skipped/no-url: {skipped}, Total: {total}")
    print(f"Updated: {CHECKS_JSON}")
    print(f"HTML cache: {SCRAPED_DIR}")


if __name__ == "__main__":
    main()
