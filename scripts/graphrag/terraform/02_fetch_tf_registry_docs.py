"""02_fetch_tf_registry_docs.py

Fetch resource documentation from the Terraform Registry REST API (v2).

How the Registry v2 API actually works
---------------------------------------
The list endpoint does NOT return a `links.next` cursor — it uses
offset-based pagination via `page[number]`.

    GET /v2/provider-docs
        ?filter[provider-version]=<numeric-id>
        &filter[category]=resources
        &filter[language]=hcl
        &page[size]=100
        &page[number]=<N>

Slug format: slugs in this API have NO leading 'aws_'.
    e.g.  's3_bucket' → terraform resource 'aws_s3_bucket'

The list response only has stub fields (slug, title, subcategory, path).
Full markdown content requires a separate fetch per doc:
    GET /v2/provider-docs/<doc-id>

Code fence format: the Registry uses ```terraform (NOT ```hcl).
    Both are accepted by extract_hcl_examples for safety.

Total docs for hashicorp/aws@6.47.0: 1,657 resources across 17 pages.

Output
------
tf_registry_docs.json — dict keyed by terraform resource name ('aws_s3_bucket')
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
_PAGE_SIZE = 100
_SLEEP     = 0.2    # seconds between list-page requests
_WORKERS   = 10     # parallel content fetches
_TIMEOUT   = 30     # per-request timeout (s)

# The AWS provider Registry docs use ```terraform fences, not ```hcl.
# Accept both to be safe.
_HCL_FENCES = {"terraform", "hcl"}


# ---------------------------------------------------------------------------
# Step 0 — resolve numeric provider-version ID
# ---------------------------------------------------------------------------

def _resolve_version_id(client: httpx.Client) -> str:
    url = f"{_BASE}/v2/providers/{PROVIDER_NAMESPACE}/{PROVIDER_NAME}?include=provider-versions"
    r = client.get(url)
    r.raise_for_status()
    for item in r.json().get("included", []):
        if item.get("attributes", {}).get("version") == PROVIDER_VERSION:
            ver_id = str(item["id"])
            print(f"[02] Resolved {PROVIDER_NAMESPACE}/{PROVIDER_NAME}@{PROVIDER_VERSION} → version ID {ver_id}")
            return ver_id
    raise RuntimeError(f"Version {PROVIDER_VERSION} not found in Registry response.")


# ---------------------------------------------------------------------------
# Step 1 — collect all doc stubs via page[number] pagination
# ---------------------------------------------------------------------------

def _fetch_doc_stubs(client: httpx.Client, version_id: str) -> list[dict]:
    """Page through all resource docs and return stub list.

    The v2 API uses page[number]=N (1-indexed), NOT cursor/links.next.
    Stop when a page returns fewer than page[size] items.
    """
    stubs: list[dict] = []
    page = 0

    while True:
        page += 1
        print(f"[02] List page {page} … ", end="", flush=True)

        r = client.get(
            f"{_BASE}/v2/provider-docs",
            params={
                "filter[provider-version]": version_id,
                "filter[category]":        "resources",
                "filter[language]":        "hcl",
                "page[size]":              _PAGE_SIZE,
                "page[number]":            page,
            },
        )
        r.raise_for_status()
        items = r.json().get("data", [])

        for item in items:
            attrs = item["attributes"]
            stubs.append({
                "id":          item["id"],
                "slug":        attrs.get("slug", ""),
                "title":       attrs.get("title", ""),
                "subcategory": attrs.get("subcategory", ""),
                "path":        attrs.get("path", ""),
            })

        print(f"{len(items)} stubs (total: {len(stubs)})")

        if len(items) < _PAGE_SIZE:
            break  # last page

        time.sleep(_SLEEP)

    return stubs


# ---------------------------------------------------------------------------
# Step 2 — fetch full markdown content per doc (parallelised)
# ---------------------------------------------------------------------------

def _fetch_one(doc_id: str) -> str:
    url = f"{_BASE}/v2/provider-docs/{doc_id}"
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as c:
        r = c.get(url, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()["data"]["attributes"].get("content", "")


def _fetch_all_contents(stubs: list[dict]) -> dict[str, str]:
    """Returns {doc_id: content}."""
    results: dict[str, str] = {}
    total = len(stubs)
    done  = 0

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, s["id"]): s for s in stubs}
        for future in as_completed(futures):
            stub = futures[future]
            try:
                results[stub["id"]] = future.result()
            except Exception as exc:
                print(f"\n[02] WARNING: failed doc {stub['id']} ({stub['slug']}): {exc}")
                results[stub["id"]] = ""
            done += 1
            if done % 100 == 0 or done == total:
                print(f"[02] Content fetched: {done}/{total}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug_to_tf_name(slug: str) -> str:
    """Registry slugs omit the 'aws_' prefix; add it back.

    's3_bucket'  → 'aws_s3_bucket'
    'instance'   → 'aws_instance'
    """
    return slug if slug.startswith("aws_") else f"aws_{slug}"


def extract_hcl_examples(markdown: str) -> list[str]:
    """Extract fenced Terraform/HCL code blocks from Registry markdown.

    The Terraform Registry AWS provider uses ```terraform fences (not ```hcl).
    Both are accepted. Closing fence must be a bare ``` on its own line.
    """
    examples: list[str] = []
    in_block = False
    current:  list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("```"):
                lang = stripped[3:].strip().lower()
                if lang in _HCL_FENCES:
                    in_block = True
                    current  = []
        else:
            if stripped == "```":
                in_block = False
                block    = "\n".join(current).strip()
                if block:
                    examples.append(block)
            else:
                current.append(line)

    return examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        version_id = _resolve_version_id(client)
        stubs      = _fetch_doc_stubs(client, version_id)

    print(f"\n[02] {len(stubs)} stubs collected. Fetching full content …\n")
    contents = _fetch_all_contents(stubs)  # {doc_id: markdown}

    docs: dict[str, dict] = {}
    for stub in stubs:
        tf_name = _slug_to_tf_name(stub["slug"])
        content = contents.get(stub["id"], "")
        docs[tf_name] = {
            "title":        stub["title"],
            "subcategory":  stub["subcategory"],
            "content":      content,
            "hcl_examples": extract_hcl_examples(content),
        }

    OUTPUT_FILE.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"\n[02] Saved {len(docs)} resource docs → {OUTPUT_FILE}")

    # Sanity checks
    s3 = docs.get("aws_s3_bucket", {})
    print(f"[02] Sanity aws_s3_bucket: {len(s3.get('hcl_examples', []))} HCL examples, "
          f"{len(s3.get('content', ''))} content chars")
    print(f"[02] Sample resource names: {list(docs.keys())[:5]}")
