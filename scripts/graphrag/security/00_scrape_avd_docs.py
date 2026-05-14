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

4. Merge results into trivy_enriched.csv
   - Rows already in the CSV (matched on avd_url) → update title/description
     if they were blank OR if the existing value is a known bad/stale value
     (e.g. "Get Demo" written by a prior headless-browser enrichment pass).
   - Brand-new checks → append as new rows with service + avd_url populated;
     check_id derived from the URL slug.

5. Write data/avd_scraped.json as a raw cache (re-running skips already-fetched
   URLs unless --force is passed on the command line)

Background on "Get Demo" stale values
--------------------------------------
AVD serves all check pages as SSR HTML — requests.get() receives real content.
However, a prior enrichment pass used a headless browser that hit a JS-rendered
paywall variant for some enterprise-adjacent checks, storing "Get Demo" as the
description in trivy_enriched.csv.  The old merge logic had an
`if not row.get("description")` guard that prevented these stale values from
being overwritten.  This version replaces that guard with is_bad_description()
so stale paywall text is always replaced by live content.

Usage
-----
    python scripts/graphrag/security/00_scrape_avd_docs.py
    python scripts/graphrag/security/00_scrape_avd_docs.py --force   # ignore cache

Dependencies: requests, beautifulsoup4
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
CRAWL_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 5

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "trivy_enriched.csv"
CACHE_PATH = REPO_ROOT / "data" / "avd_scraped.json"

# Descriptions that indicate stale/bad data from a prior enrichment pass.
# These should always be overwritten by the freshly scraped live content.
BAD_DESCRIPTIONS = frozenset({
    "get demo",
    "request a demo",
    "get a demo",
    "contact us",
    "available in aqua",
})

# HTML comment scaffold that AVD uses as placeholder for empty impact fields.
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL | re.IGNORECASE)


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
# Helpers
# ---------------------------------------------------------------------------

def is_bad_description(text: str) -> bool:
    """
    Return True if the description is a known stale/bad value that should be
    replaced by freshly scraped content.

    This catches values written by prior enrichment passes that hit a JS-rendered
    paywall variant of AVD (e.g. "Get Demo", "Request a demo").
    """
    return text.strip().lower() in BAD_DESCRIPTIONS


def clean_impact(raw: str) -> str:
    """Strip HTML comment placeholders from the impact field."""
    if not raw:
        return ""
    return _HTML_COMMENT_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# Step 1 – discover service slugs
# ---------------------------------------------------------------------------

def scrape_service_slugs() -> list[str]:
    print(f"Fetching index: {BASE_URL}")
    soup = fetch(BASE_URL)
    if not soup:
        raise RuntimeError("Failed to fetch AVD AWS index page.")

    slugs: list[str] = []
    for a in soup.find_all("a", href=True):
        m = re.match(r'^/misconfig/aws/([a-z0-9\-]+)/?$', a["href"])
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
    checks: list[dict] = []
    seen_slugs: set[str] = set()
    page_url = f"{BASE_URL}{service_slug}/"

    while page_url:
        soup = fetch(page_url)
        if not soup:
            break

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

        next_link = soup.find("a", string=re.compile(r'Next', re.I))
        if next_link and next_link.get("href"):
            page_url = urljoin("https://avd.aquasec.com", next_link["href"])
        else:
            page_url = None

    return checks


# ---------------------------------------------------------------------------
# Step 3 – scrape a single check detail page
# ---------------------------------------------------------------------------

def scrape_check_detail(url: str) -> dict:
    soup = fetch(url)
    if not soup:
        return {}

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True) \
            .replace(" - Aqua Vulnerability Database", "").strip()
    if not title:
        h = soup.find(["h1", "h2"])
        title = h.get_text(strip=True) if h else ""

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]
    description = paragraphs[0] if paragraphs else ""

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
        "raw_text": soup.get_text(" ", strip=True)[:4000],
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
            print("done" if detail else "EMPTY")

    if changed:
        save_cache(CACHE_PATH, cache)
        print(f"\nCache updated → {CACHE_PATH}")

    return cache


# ---------------------------------------------------------------------------
# Step 5 – merge into trivy_enriched.csv
# ---------------------------------------------------------------------------

def slug_to_check_id(service_slug: str, check_slug: str) -> str:
    return f"AVD-AWS-{service_slug}-{check_slug}".upper()


def load_existing_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    if not csv_path.exists():
        return [], []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, list(fieldnames)


def merge_into_csv(avd_data: dict, csv_path: Path) -> None:
    rows, fieldnames = load_existing_csv(csv_path)

    url_to_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        avd_url = row.get("avd_url", "").strip().rstrip("/")
        if avd_url:
            url_to_idx[avd_url] = i

    extra_cols = ["avd_url", "title", "description", "remediation_console", "raw_text"]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row.setdefault(col, "")

    added = 0
    updated = 0
    overwritten_stale = 0

    for url, item in avd_data.items():
        normalised_url = url.rstrip("/")
        scraped_desc = item.get("description", "")
        remediation_text = "\n".join(item.get("remediation_steps", []))

        if normalised_url in url_to_idx:
            idx = url_to_idx[normalised_url]
            row = rows[idx]

            existing_desc = row.get("description", "")
            # Overwrite if blank OR if it's a known stale/bad value from a
            # prior headless-browser enrichment pass (e.g. "Get Demo").
            if not existing_desc or is_bad_description(existing_desc):
                if is_bad_description(existing_desc) and scraped_desc:
                    overwritten_stale += 1
                row["description"] = scraped_desc
                updated += 1

            if not row.get("title"):
                row["title"] = item.get("title", "")
            if not row.get("remediation_console"):
                row["remediation_console"] = remediation_text
            if not row.get("raw_text"):
                row["raw_text"] = item.get("raw_text", "")

            # Always clean impact HTML comment placeholders
            if row.get("impact"):
                row["impact"] = clean_impact(row["impact"])
        else:
            new_row = {col: "" for col in fieldnames}
            new_row.update({
                "check_id": slug_to_check_id(item["service_slug"], item["check_slug"]),
                "check_name": item.get("check_name", ""),
                "title": item.get("title", ""),
                "description": scraped_desc,
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

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV merge complete:")
    print(f"  Updated existing rows         : {updated}")
    print(f"    of which overwritten stale  : {overwritten_stale}")
    print(f"  Appended new rows             : {added}")
    print(f"  Total rows in CSV             : {len(rows)}")
    print(f"  Output                        : {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape AVD AWS misconfig pages.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore the on-disk cache and re-fetch all pages.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging into trivy_enriched.csv.")
    args = parser.parse_args()

    print("=" * 60)
    print("AVD AWS Misconfig Scraper")
    print("=" * 60)

    avd_data = build_avd_dataset(force=args.force)
    print(f"\nTotal checks scraped/cached: {len(avd_data)}")

    if not args.no_merge:
        print(f"\nMerging into {CSV_PATH} …")
        merge_into_csv(avd_data, CSV_PATH)
    else:
        print("\n--no-merge set. Skipping CSV merge.")

    print("\nDone.")


if __name__ == "__main__":
    main()
