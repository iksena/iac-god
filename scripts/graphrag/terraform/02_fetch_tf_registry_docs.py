"""02_fetch_tf_registry_docs.py

Fetch resource documentation from the Terraform Registry REST API (v2).

How the Registry v2 API actually works
---------------------------------------
The list endpoint   GET /v2/provider-docs?filter[provider-version]=<id>&...
returns lightweight stubs (slug, title, subcategory) but NO content field.

To get the full markdown you must fetch each doc individually:
    GET /v2/provider-docs/<doc-id>

Step 0: resolve the numeric provider-version ID for hashicorp/aws @ 6.47.0
        GET /v2/providers/hashicorp/aws?include=provider-versions
        → scan `included` array for version == "6.47.0", grab its `id`

Step 1: paginate the list endpoint (filter[provider-version]=<id>, category=resources)
        to collect every (id, slug, title, subcategory) tuple.

Step 2: fetch each doc by ID to get its full markdown content.
        We parallelise this with a small thread pool to stay within ~60 s.

Output
------
tf_registry_docs.json   — dict keyed by terraform resource name (e.g. "aws_s3_bucket")
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROVIDER_NAMESPACE = "hashicorp"
PROVIDER_NAME      = "aws"
PROVIDER_VERSION   = "6.47.0"
OUTPUT_FILE        = Path("tf_registry_docs.json")

_BASE      = "https://registry.terraform.io"
_PAGE_SIZE = 100          # Registry API max
_SLEEP     = 0.25         # seconds between list pages
_WORKERS   = 8            # parallel doc-content fetches
_TIMEOUT   = 30           # per-request timeout (s)


# ---------------------------------------------------------------------------
# Step 0 — resolve numeric provider-version ID
# ---------------------------------------------------------------------------

def _resolve_version_id(client: httpx.Client) -> str:
    """Return the numeric Registry ID for PROVIDER_VERSION (e.g. '97945')."""
    url = f"{_BASE}/v2/providers/{PROVIDER_NAMESPACE}/{PROVIDER_NAME}?include=provider-versions"
    r = client.get(url)
    r.raise_for_status()
    payload = r.json()

    for item in payload.get("included", []):
        if item.get("attributes", {}).get("version") == PROVIDER_VERSION:
            ver_id = item["id"]
            print(f"[02] Resolved {PROVIDER_NAMESPACE}/{PROVIDER_NAME}@{PROVIDER_VERSION} → version ID {ver_id}")
            return str(ver_id)

    raise RuntimeError(
        f"Could not find version {PROVIDER_VERSION} for {PROVIDER_NAMESPACE}/{PROVIDER_NAME} "
        f"in Registry response. Check PROVIDER_VERSION constant."
    )


# ---------------------------------------------------------------------------
# Step 1 — paginate list endpoint to collect all doc stubs
# ---------------------------------------------------------------------------

def _fetch_doc_stubs(client: httpx.Client, version_id: str) -> list[dict]:
    """Return list of {id, slug, title, subcategory} for every resource doc."""
    stubs: list[dict] = []
    next_url: str | None = f"{_BASE}/v2/provider-docs"
    params: dict | None = {
        "filter[provider-version]": version_id,
        "filter[category]":        "resources",
        "filter[language]":        "hcl",
        "page[size]":              _PAGE_SIZE,
    }
    page = 0

    while next_url:
        page += 1
        print(f"[02] List page {page} … ", end="", flush=True)

        r = client.get(next_url, params=params)
        r.raise_for_status()
        payload = r.json()

        items = payload.get("data", [])
        for item in items:
            attrs = item.get("attributes", {})
            stubs.append({
                "id":          item["id"],
                "slug":        attrs.get("slug", ""),
                "title":       attrs.get("title", ""),
                "subcategory": attrs.get("subcategory", ""),
            })

        print(f"{len(items)} stubs (total: {len(stubs)})")

        links    = payload.get("links", {})
        next_url = links.get("next")
        params   = None          # next_url already embeds params

        if next_url:
            time.sleep(_SLEEP)

    return stubs


# ---------------------------------------------------------------------------
# Step 2 — fetch full markdown content for each doc
# ---------------------------------------------------------------------------

def _fetch_one_doc(doc_id: str, slug: str) -> tuple[str, str]:
    """Fetch the markdown content for a single doc. Returns (slug, content)."""
    url = f"{_BASE}/v2/provider-docs/{doc_id}"
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as c:
        r = c.get(url, headers={"Accept": "application/json"})
        r.raise_for_status()
        content = r.json()["data"]["attributes"].get("content", "")
    return slug, content


def _fetch_all_contents(stubs: list[dict]) -> dict[str, str]:
    """Parallel-fetch markdown content for every stub. Returns {slug: content}."""
    results: dict[str, str] = {}
    total   = len(stubs)
    done    = 0

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one_doc, s["id"], s["slug"]): s
            for s in stubs
        }
        for future in as_completed(futures):
            stub = futures[future]
            try:
                slug, content = future.result()
                results[slug] = content
            except Exception as exc:
                print(f"\n[02] WARNING: failed to fetch doc {stub['id']} ({stub['slug']}): {exc}")
                results[stub["slug"]] = ""
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[02] Content fetched: {done}/{total}", flush=True)

    return results


# ---------------------------------------------------------------------------
# HCL example extractor (used by downstream scripts)
# ---------------------------------------------------------------------------

def extract_hcl_examples(markdown: str) -> list[str]:
    """Extract ```hcl … ``` code blocks from a markdown string."""
    examples: list[str] = []
    in_block = False
    current:  list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if not in_block and stripped.startswith("```hcl"):
            in_block = True
            current  = []
        elif in_block and stripped == "```":
            in_block = False
            block    = "\n".join(current).strip()
            if block:
                examples.append(block)
        elif in_block:
            current.append(line)

    return examples


# ---------------------------------------------------------------------------
# slug → terraform resource name
# ---------------------------------------------------------------------------

def _slug_to_tf_name(slug: str) -> str:
    """Convert a registry slug to the Terraform resource name.

    Registry slugs omit the leading 'aws_' and use the path 'r/<rest>'.
    Examples:
        's3_bucket'               → 'aws_s3_bucket'
        'instance'                → 'aws_instance'
        'vpc'                     → 'aws_vpc'
    """
    # slug may already include 'aws_' if the registry changed format
    if slug.startswith("aws_"):
        return slug
    return f"aws_{slug}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        version_id = _resolve_version_id(client)
        stubs      = _fetch_doc_stubs(client, version_id)

    print(f"\n[02] Fetched {len(stubs)} resource stubs. Now downloading content …\n")
    contents = _fetch_all_contents(stubs)

    # Build final output keyed by terraform resource name
    docs: dict[str, dict] = {}
    for stub in stubs:
        slug    = stub["slug"]
        content = contents.get(slug, "")
        tf_name = _slug_to_tf_name(slug)
        docs[tf_name] = {
            "title":        stub["title"],
            "subcategory":  stub["subcategory"],
            "content":      content,
            "hcl_examples": extract_hcl_examples(content),
        }

    OUTPUT_FILE.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"\n[02] Saved {len(docs)} resource docs → {OUTPUT_FILE}")

    # Sanity check
    s3 = docs.get("aws_s3_bucket", {})
    n_examples = len(s3.get("hcl_examples", []))
    print(f"[02] Sanity: aws_s3_bucket — {n_examples} HCL examples in docs.")
