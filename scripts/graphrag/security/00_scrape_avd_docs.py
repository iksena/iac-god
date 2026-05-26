#!/usr/bin/env python3
"""
00_scrape_avd_docs.py

Stage 0 – Scrape all AWS misconfiguration checks from avd.aquasec.com.

What it does
------------
1. Fetch the AWS index page  https://avd.aquasec.com/misconfig/aws/
   → extract all service slugs (e.g. 'ec2', 'rds', 's3', …)

2. For each service, paginate through  /misconfig/aws/<service>/?page=N
   → collect every (check_name, check_slug) pair

3. For each check, fetch  /misconfig/aws/<service>/<check_slug>/
   → extract title, description, remediation steps
   → IMPORTANT: content is scoped to div.content.vulnerability_content
     to avoid picking up nav/CTA elements like the "Get Demo" button
     that appear earlier in DOM order than the main description.

4. Merge results into trivy_enriched.csv
   - Rows already in the CSV (matched on avd_url) → update title/description
     if they were blank OR if the existing value is a known stale value
     written by a prior enrichment pass (e.g. "Get Demo" captured from a
     nav button by an unscoped scraper).
   - Brand-new checks → append as new rows.

5. Write data/avd_scraped.json as a raw cache.

Usage
-----
    python scripts/graphrag/security/00_scrape_avd_docs.py
    python scripts/graphrag/security/00_scrape_avd_docs.py --force

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
from bs4 import BeautifulSoup, Tag

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

# Descriptions known to be stale nav/CTA text from unscoped prior scrapes.
# These are overwritten (not filtered) by the corrected scraper.
STALE_DESCRIPTIONS = frozenset({
    "get demo",
    "request a demo",
    "get a demo",
})

# HTML comment scaffold AVD uses as placeholder for empty impact fields.
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

def is_stale_description(text: str) -> bool:
    """True if the description is nav/CTA text captured by an unscoped prior scrape."""
    return text.strip().lower() in STALE_DESCRIPTIONS


def clean_impact(raw: str) -> str:
    """Strip HTML comment placeholders (AVD scaffold for empty impact fields)."""
    if not raw:
        return ""
    return _HTML_COMMENT_RE.sub("", raw).strip()


def get_content_scope(soup: BeautifulSoup) -> Tag:
    """
    Return the main content container for a check detail page.

    AVD check pages structure their content inside:
        <div class="content vulnerability_content">...</div>

    This container holds the description paragraphs, remediation steps, and
    code examples. Scoping to it avoids nav elements ("Get Demo" CTA button
    lives in div.field.is-grouped earlier in DOM order).

    Falls back to the full soup if the container is not found.
    """
    container = soup.find("div", class_="vulnerability_content")
    return container if container else soup


def extract_avd_content(soup: BeautifulSoup) -> dict:
    """Extract structured AVD fields from a check detail page."""
    # Title from <title> tag
    title = ""
    if soup.title:
        title = (
            soup.title.get_text(strip=True)
            .replace(" - Aqua Vulnerability Database", "")
            .replace(" | Vulnerability Database | Aqua Security", "")
            .strip()
        )
    if not title:
        h = soup.find(["h1", "h2"])
        title = h.get_text(strip=True) if h else ""

    content = get_content_scope(soup)

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in content.find_all("p")
        if p.get_text(strip=True)
    ]
    description = paragraphs[0] if paragraphs else ""

    impact_texts: list[str] = []
    remediation_steps: list[str] = []
    for heading in content.find_all(["h2", "h3"]):
        heading_text = heading.get_text(strip=True).lower()
        sibling_texts: list[str] = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ("h2", "h3"):
                break
            text = sibling.get_text(" ", strip=True)
            if text:
                sibling_texts.append(text)

        if "impact" in heading_text:
            impact_texts.extend(sibling_texts)
        elif any(k in heading_text for k in ("remediat", "resolut", "fix", "how to")):
            remediation_steps.extend(sibling_texts)

    if not remediation_steps and len(paragraphs) > 1:
        remediation_steps = paragraphs[1:]

    cfn_example = ""
    tf_example = ""
    for block in content.find_all(["code", "pre"]):
        text = block.get_text().strip()
        if not cfn_example and ("Type: AWS::" in text or "AWSTemplateFormatVersion" in text):
            cfn_example = text
        elif not tf_example and ("resource \"aws_" in text or "provider \"aws\"" in text):
            tf_example = text
        if cfn_example and tf_example:
            break

    impact = clean_impact(" ".join(impact_texts))
    remediation_console = "\n".join(remediation_steps)

    return {
        "title": title,
        "description": description,
        "impact": impact,
        "remediation_steps": remediation_steps,
        "remediation_console": remediation_console,
        "cfn_good_example": cfn_example,
        "tf_good_example": tf_example,
        "raw_text": content.get_text(" ", strip=True)[:4000],
    }


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
    """
    Extract structured data from a check detail page.

    Scopes all content extraction to div.content.vulnerability_content to
    avoid nav/CTA elements that appear earlier in the DOM.
    """
    soup = fetch(url)
    if not soup:
        return {}
    return extract_avd_content(soup)


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

    extra_cols = [
        "avd_url",
        "title",
        "description",
        "impact",
        "remediation_console",
        "cfn_good_example",
        "tf_good_example",
        "raw_text",
    ]
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
        scraped_impact = item.get("impact", "")
        remediation_text = "\n".join(item.get("remediation_steps", []))

        if normalised_url in url_to_idx:
            idx = url_to_idx[normalised_url]
            row = rows[idx]

            existing_desc = row.get("description", "")
            # Overwrite if blank OR if it's stale nav text from a prior scrape
            if not existing_desc or is_stale_description(existing_desc):
                if is_stale_description(existing_desc) and scraped_desc:
                    overwritten_stale += 1
                row["description"] = scraped_desc
                updated += 1

            if not row.get("title"):
                row["title"] = item.get("title", "")
            if not row.get("impact"):
                row["impact"] = scraped_impact
            if not row.get("remediation_console"):
                row["remediation_console"] = remediation_text
            if not row.get("cfn_good_example"):
                row["cfn_good_example"] = item.get("cfn_good_example", "")
            if not row.get("tf_good_example"):
                row["tf_good_example"] = item.get("tf_good_example", "")
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
                "impact": scraped_impact,
                "service": item["service_slug"],
                "framework": "cloudformation",
                "avd_url": url,
                "remediation_console": remediation_text,
                "cfn_good_example": item.get("cfn_good_example", ""),
                "tf_good_example": item.get("tf_good_example", ""),
                "raw_text": item.get("raw_text", ""),
                "severity": "",
                "source_file_url": "",
                "source_code": "",
                "remediation_cfn": "",
                "remediation_tf": "",
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
