#!/usr/bin/env python3
"""
00_scrape_avd_docs.py

Stage 0 – Scrape all AWS misconfiguration checks from avd.aquasec.com.

Why this exists
---------------
trivy_enriched.csv only contains checks that happened to be mapped during
a previous enrichment pass. The AVD website is the authoritative source and
contains additional checks (especially console-only / cloud-provider checks)
that Trivy's Rego policies do not cover.

What it does
------------
1. Fetch the AWS index page  https://avd.aquasec.com/misconfig/aws/
   → extract all service slugs (e.g. 'ec2', 'rds', 's3', …)

2. For each service, paginate through  /misconfig/aws/<service>/?page=N
   → collect every (check_name, check_slug) pair

3. For each check, fetch  /misconfig/aws/<service>/<check_slug>/
   → extract title, description paragraphs, remediation steps
   → detect gated/enterprise checks ("Get Demo" / paywall pages) and
     mark them so downstream stages can skip them

4. Merge results into trivy_enriched.csv
   - Rows already in the CSV (matched on avd_url suffix or check_id) → update
     avd_url, title, description fields if they were blank
   - Brand-new checks → append as new rows with service + avd_url populated;
     check_id derived from the URL slug (e.g. 'AVD-AWS-EC2-default-security-group')
   - Gated / enterprise rows are SKIPPED entirely (no value without the data)

5. Write data/avd_scraped.json as a raw cache (re-running skips already-fetched
   URLs unless --force is passed on the command line)

Usage
-----
    python scripts/graphrag/security/00_scrape_avd_docs.py
    python scripts/graphrag/security/00_scrape_avd_docs.py --force   # ignore cache

Dependencies: requests, beautifulsoup4  (both already in requirements if you
ran the CFN scraper; add with: pip install requests beautifulsoup4)
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://avd.aquasec.com/misconfig/aws/"
CRAWL_DELAY = 0.5          # seconds between requests (polite crawl)
MAX_RETRIES = 3
RETRY_DELAY = 5            # seconds to wait on 429 / 5xx

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "trivy_enriched.csv"
CACHE_PATH = REPO_ROOT / "data" / "avd_scraped.json"

# Phrases that indicate a gated / enterprise-only check page.
# When the description or page body contains any of these, the check has no
# actionable public data and should be excluded.
GATED_PHRASES = (
    "get demo",
    "request a demo",
    "contact us for",
    "sign up to view",
    "available in aqua",
)

# HTML comment scaffold that AVD uses as placeholder for empty impact fields.
# These must be stripped at ingest time so they don't pollute downstream data.
IMPACT_PLACEHOLDER_RE = re.compile(
    r'<!--.*?-->',
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; iac-god-scraper/1.0; "
        "+https://github.com/iksena/iac-god)"
    )
})


def fetch(url: str, retries: int = MAX_RETRIES) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            if resp.status_code == 200:
                time.sleep(CRAWL_DELAY)
                return BeautifulSoup(resp.text, "html.parser")
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    [{resp.status_code}] {url} – retrying in {wait}s …")
                time.sleep(wait)
                continue
            print(f"    [SKIP {resp.status_code}] {url}")
            return None
        except requests.RequestException as exc:
            print(f"    [ERROR] {url}: {exc}")
            time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# Helper: detect gated pages
# ---------------------------------------------------------------------------

def is_gated(text: str) -> bool:
    """Return True if the page body indicates a paywall / enterprise gate."""
    normalised = text.lower()
    return any(phrase in normalised for phrase in GATED_PHRASES)


# ---------------------------------------------------------------------------
# Helper: clean impact text
# ---------------------------------------------------------------------------

def clean_impact(raw: str) -> str:
    """
    Strip HTML comment placeholders that AVD uses for empty impact fields.

    AVD source uses:
        <!-- Add Impact here -->
        <!-- DO NOT CHANGE -->
    as scaffold that should never appear in user-facing output.
    """
    cleaned = IMPACT_PLACEHOLDER_RE.sub("", raw)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Step 1 – discover service slugs from the index page
# ---------------------------------------------------------------------------

def scrape_service_slugs() -> list[str]:
    """Return list of service slugs, e.g. ['ec2', 'rds', 's3', …]"""
    print(f"Fetching index: {BASE_URL}")
    soup = fetch(BASE_URL)
    if not soup:
        raise RuntimeError("Failed to fetch AVD AWS index page.")

    slugs: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match links like /misconfig/aws/ec2/  (exactly two path components after /misconfig/aws/)
        m = re.match(r'^/misconfig/aws/([a-z0-9\-]+)/?$', href)
        if m:
            slug = m.group(1)
            if slug not in slugs:
                slugs.append(slug)

    print(f"  Found {len(slugs)} service slugs: {slugs}")
    return slugs


# ---------------------------------------------------------------------------
# Step 2 – collect all check slugs for a service (paginated)
# ---------------------------------------------------------------------------

def scrape_check_slugs(service_slug: str) -> list[dict]:
    """
    Return list of {name, slug, url} for every check under a service.
    Handles the 'Next >>' pagination link.
    """
    checks: list[dict] = []
    seen_slugs: set[str] = set()
    page_url = f"{BASE_URL}{service_slug}/"

    while page_url:
        soup = fetch(page_url)
        if not soup:
            break

        # Check links look like /misconfig/aws/ec2/default-security-group/
        pattern = re.compile(
            rf'^/misconfig/aws/{re.escape(service_slug)}/([a-z0-9\-]+)/?$'
        )
        for a in soup.find_all("a", href=True):
            m = pattern.match(a["href"])
            if m:
                check_slug = m.group(1)
                if check_slug not in seen_slugs:
                    seen_slugs.add(check_slug)
                    checks.append({
                        "name": a.get_text(strip=True),
                        "slug": check_slug,
                        "url": urljoin("https://avd.aquasec.com", a["href"]),
                    })

        # Follow "Next >>" pagination
        next_link = soup.find("a", string=re.compile(r'Next', re.I))
        if next_link and next_link.get("href"):
            next_href = next_link["href"]
            page_url = urljoin("https://avd.aquasec.com", next_href)
        else:
            page_url = None

    return checks


# ---------------------------------------------------------------------------
# Step 3 – scrape a single check detail page
# ---------------------------------------------------------------------------

def scrape_check_detail(url: str) -> dict:
    """
    Extract structured data from a check detail page.

    Returns a dict with keys:
      title, description, remediation_steps (list[str]), raw_text, gated (bool)

    gated=True means the page is behind a paywall/enterprise gate and has no
    actionable data.  Callers should skip gated entries.
    """
    soup = fetch(url)
    if not soup:
        return {}

    raw_text = soup.get_text(" ", strip=True)

    # Detect gated pages early – no point parsing further
    if is_gated(raw_text):
        return {
            "title": "",
            "description": "",
            "remediation_steps": [],
            "raw_text": "",
            "gated": True,
        }

    # Title is in the <title> tag or first <h1>/<h2>
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True).replace(" - Aqua Vulnerability Database", "").strip()
    if not title:
        h = soup.find(["h1", "h2"])
        title = h.get_text(strip=True) if h else ""

    # All paragraph text – first non-empty para is usually the description
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    description = paragraphs[0] if paragraphs else ""

    # Remediation steps: list items or paragraphs after the first
    remediation_steps: list[str] = []
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if text:
            remediation_steps.append(text)
    if not remediation_steps and len(paragraphs) > 1:
        remediation_steps = paragraphs[1:]

    return {
        "title": title,
        "description": description,
        "remediation_steps": remediation_steps,
        "raw_text": raw_text[:4000],
        "gated": False,
    }


# ---------------------------------------------------------------------------
# Step 4 – load cache / build full dataset
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(cache_path: Path, data: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def build_avd_dataset(force: bool = False) -> dict[str, dict]:
    """
    Crawl the full AVD AWS section, using the on-disk cache to skip
    already-fetched URLs unless force=True.

    Returns a dict keyed by check URL.  Gated entries are retained in the
    cache (so we don't re-fetch them) but are marked gated=True so
    merge_into_csv can skip them.
    """
    cache = {} if force else load_cache(CACHE_PATH)
    changed = False

    service_slugs = scrape_service_slugs()

    for service_slug in service_slugs:
        print(f"\nService: {service_slug}")
        check_list = scrape_check_slugs(service_slug)
        print(f"  {len(check_list)} checks found.")

        for check in check_list:
            url = check["url"]
            if url in cache and not force:
                continue

            print(f"  Scraping: {check['slug']} …", end=" ", flush=True)
            detail = scrape_check_detail(url)
            cache[url] = {
                "service_slug": service_slug,
                "check_slug": check["slug"],
                "check_name": check["name"],
                "url": url,
                **detail,
            }
            changed = True
            if detail.get("gated"):
                print("GATED (skipped)")
            else:
                print("done" if detail else "EMPTY")

    if changed:
        save_cache(CACHE_PATH, cache)
        print(f"\nCache updated → {CACHE_PATH}")

    return cache


# ---------------------------------------------------------------------------
# Step 5 – merge into trivy_enriched.csv
# ---------------------------------------------------------------------------

# Derive a pseudo check_id from the URL slug so new rows have a stable key.
# Format: AVD-AWS-<SERVICE>-<slug>  (uppercased, hyphens preserved)
def slug_to_check_id(service_slug: str, check_slug: str) -> str:
    return f"AVD-AWS-{service_slug}-{check_slug}".upper()


def load_existing_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Returns (rows, fieldnames)."""
    if not csv_path.exists():
        return [], []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, list(fieldnames)


def merge_into_csv(avd_data: dict, csv_path: Path) -> None:
    rows, fieldnames = load_existing_csv(csv_path)

    # Build lookup: avd_url (normalised) → row index
    url_to_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        avd_url = row.get("avd_url", "").strip().rstrip("/")
        if avd_url:
            url_to_idx[avd_url] = i

    # Ensure the CSV has all required columns
    extra_cols = ["avd_url", "title", "description", "remediation_console", "raw_text"]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row.setdefault(col, "")

    added = 0
    updated = 0
    skipped_gated = 0

    for url, item in avd_data.items():
        # Skip enterprise/gated checks – they have no actionable data
        if item.get("gated"):
            skipped_gated += 1
            continue

        normalised_url = url.rstrip("/")
        remediation_text = "\n".join(item.get("remediation_steps", []))

        if normalised_url in url_to_idx:
            idx = url_to_idx[normalised_url]
            row = rows[idx]
            if not row.get("title"):
                row["title"] = item.get("title", "")
                updated += 1
            if not row.get("description"):
                row["description"] = item.get("description", "")
            if not row.get("remediation_console"):
                row["remediation_console"] = remediation_text
            if not row.get("raw_text"):
                row["raw_text"] = item.get("raw_text", "")
            # Always clean impact placeholders on existing rows
            if row.get("impact"):
                row["impact"] = clean_impact(row["impact"])
        else:
            new_row = {col: "" for col in fieldnames}
            new_row.update({
                "check_id": slug_to_check_id(item["service_slug"], item["check_slug"]),
                "check_name": item.get("check_name", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "service": item["service_slug"],
                "framework": "cloudformation",
                "avd_url": url,
                "remediation_console": remediation_text,
                "raw_text": item.get("raw_text", ""),
                "severity": "",
                "source_file_url": "",
                "source_code": "",
                "impact": "",
                "remediation_cfn": "",
                "remediation_tf": "",
                "cfn_good_example": "",
                "tf_good_example": "",
                "links": "",
            })
            rows.append(new_row)
            url_to_idx[normalised_url] = len(rows) - 1
            added += 1

    # Write back
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV merge complete:")
    print(f"  Updated existing rows : {updated}")
    print(f"  Appended new rows     : {added}")
    print(f"  Skipped gated rows    : {skipped_gated}")
    print(f"  Total rows in CSV     : {len(rows)}")
    print(f"  Output                : {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape AVD AWS misconfig pages.")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore the on-disk cache and re-fetch all pages."
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip merging into trivy_enriched.csv (just update the JSON cache)."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AVD AWS Misconfig Scraper")
    print("=" * 60)

    avd_data = build_avd_dataset(force=args.force)

    total = len(avd_data)
    gated = sum(1 for v in avd_data.values() if v.get("gated"))
    print(f"\nTotal checks scraped/cached : {total}")
    print(f"  Gated (enterprise-only)   : {gated}")
    print(f"  Actionable (public)        : {total - gated}")

    if not args.no_merge:
        print(f"\nMerging into {CSV_PATH} …")
        merge_into_csv(avd_data, CSV_PATH)
    else:
        print("\n--no-merge set. Skipping CSV merge.")

    print("\nDone.")


if __name__ == "__main__":
    main()
