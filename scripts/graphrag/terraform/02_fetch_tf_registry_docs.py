"""02_fetch_tf_registry_docs.py

Fetch resource documentation from the Terraform Registry REST API (v2).
This replaces HTML scraping with a structured JSON API that returns
descriptions and full markdown content (including HCL examples) for every
resource in the hashicorp/aws provider.

API reference: https://registry.terraform.io/v2/provider-docs

Output
------
tf_registry_docs.json   — dict keyed by registry slug (e.g. "r/s3_bucket")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://registry.terraform.io/v2/provider-docs"
OUTPUT_FILE = Path("tf_registry_docs.json")

# Courtesy rate-limit: wait this many seconds between paginated requests.
_SLEEP_BETWEEN_PAGES = 0.4
# Maximum items per page (Registry API cap is 100).
_PAGE_SIZE = 100


def _build_initial_params() -> dict:
    return {
        "filter[provider-slug]": "hashicorp/aws",
        "filter[category]": "resources",
        "page[size]": _PAGE_SIZE,
    }


def fetch_all_resource_docs() -> dict[str, dict]:
    """Paginate the Registry API and collect all resource docs.

    Returns a dict keyed by the registry slug (e.g. "r/s3_bucket") with
    the following shape per entry::

        {
            "title":       str,
            "description": str,
            "content":     str,   # full markdown including HCL examples
            "subcategory": str,
        }
    """
    docs: dict[str, dict] = {}
    params: dict | None = _build_initial_params()
    next_url: str | None = BASE_URL
    page_num = 0

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        while next_url:
            page_num += 1
            print(f"[02] Fetching page {page_num} ... ", end="", flush=True)

            response = client.get(next_url, params=params)
            response.raise_for_status()
            payload = response.json()

            items = payload.get("data", [])
            for item in items:
                attrs = item.get("attributes", {})
                slug = attrs.get("slug", "")
                if not slug:
                    continue
                docs[slug] = {
                    "title":       attrs.get("title", ""),
                    "description": attrs.get("description", ""),
                    "content":     attrs.get("content", ""),
                    "subcategory": attrs.get("subcategory", ""),
                }

            print(f"{len(items)} resources fetched (running total: {len(docs)})")

            # Pagination: follow `links.next` if present; clear params after
            # first request because the next URL already embeds them.
            links = payload.get("links", {})
            next_url = links.get("next")  # None when on the last page
            params = None  # params are baked into next_url

            if next_url:
                time.sleep(_SLEEP_BETWEEN_PAGES)

    return docs


def extract_hcl_examples(markdown: str) -> list[str]:
    """Extract HCL code blocks from a markdown string.

    Equivalent to the YAML-block extraction in cfn_parse_and_merge.py but
    targeting ```hcl``` fences used in Terraform registry docs.
    """
    examples: list[str] = []
    in_block = False
    current: list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if not in_block and stripped.startswith("```hcl"):
            in_block = True
            current = []
        elif in_block and stripped == "```":
            in_block = False
            block = "\n".join(current).strip()
            if block:
                examples.append(block)
        elif in_block:
            current.append(line)

    return examples


if __name__ == "__main__":
    docs = fetch_all_resource_docs()

    # Attach pre-parsed HCL examples so downstream merge script doesn't have
    # to re-parse the markdown.
    for slug, entry in docs.items():
        entry["hcl_examples"] = extract_hcl_examples(entry["content"])

    OUTPUT_FILE.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"\n[02] Saved {len(docs)} resource docs to: {OUTPUT_FILE}")
