"""02_fetch_tf_registry_docs.py

Fetch resource AND data-source documentation from the Terraform Registry
REST API (v2).

How the Registry v2 API actually works
---------------------------------------
The list endpoint does NOT return a `links.next` cursor — it uses
offset-based pagination via `page[number]`.

    GET /v2/provider-docs
        ?filter[provider-version]=<numeric-id>
        &filter[category]=resources          # or data-sources
        &filter[language]=hcl
        &page[size]=100
        &page[number]=<N>

Slug format: slugs in this API have NO leading 'aws_'.
    e.g.  's3_bucket' → terraform resource 'aws_s3_bucket'
          'ami'       → terraform data source 'aws_ami'

The list response only has stub fields (slug, title, subcategory, path).
Full markdown content requires a separate fetch per doc:
    GET /v2/provider-docs/<doc-id>

Code fence format: the Registry uses ```terraform (NOT ```hcl).
    Both are accepted by extract_hcl_examples for safety.

Categories fetched
------------------
  resources    — managed resources  (resource "aws_s3_bucket" ...)
  data-sources — data lookup blocks (data "aws_ami" ...)

Output
------
tf_registry_docs.json — dict keyed by terraform name ('aws_s3_bucket',
                         'aws_ami', 'docker_container', etc.)  Each entry
                         carries an is_data_source boolean for downstream
                         tracing.

Multi-provider coverage
------------------------
PROVIDERS lists every provider to pull docs for. Slugs from different
providers never collide in tf_registry_docs.json because _slug_to_tf_name
prefixes each slug with its own provider's resource prefix (e.g. 'aws_' or
'docker_') rather than a single hardcoded 'aws_' — see _slug_to_tf_name.
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

# Must match scripts/graphrag/terraform/01_download_tf_schema.py's PROVIDERS
# (namespace/name/version) so registry_docs keys line up with schema keys in
# 03_parse_and_merge_tf.py.
PROVIDERS = [
    {"namespace": "hashicorp",  "name": "aws",    "version": "6.47.0", "resource_prefix": "aws_"},
    {"namespace": "kreuzwerker", "name": "docker", "version": "4.5.0",  "resource_prefix": "docker_"},
]
OUTPUT_FILE        = Path("tf_registry_docs.json")

# Both categories must be ingested so the Remediator has schema docs for
# every block type the engineer_prompt.py permits.
CATEGORIES = ["resources", "data-sources"]

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

def _resolve_version_id(client: httpx.Client, provider: dict) -> str:
    namespace, name, version = provider["namespace"], provider["name"], provider["version"]
    url = f"{_BASE}/v2/providers/{namespace}/{name}?include=provider-versions"
    r = client.get(url)
    r.raise_for_status()
    for item in r.json().get("included", []):
        if item.get("attributes", {}).get("version") == version:
            ver_id = str(item["id"])
            print(f"[02] Resolved {namespace}/{name}@{version} → version ID {ver_id}")
            return ver_id
    raise RuntimeError(f"Version {version} not found in Registry response for {namespace}/{name}.")


# ---------------------------------------------------------------------------
# Step 1 — collect all doc stubs via page[number] pagination
# ---------------------------------------------------------------------------

def _fetch_doc_stubs(
    client: httpx.Client,
    version_id: str,
    category: str,
) -> list[dict]:
    """Page through all docs for *category* and return stub list.

    The v2 API uses page[number]=N (1-indexed), NOT cursor/links.next.
    Stop when a page returns fewer than page[size] items.

    Args:
        client:     shared httpx.Client
        version_id: numeric provider-version ID from _resolve_version_id()
        category:   'resources' or 'data-sources'
    """
    stubs: list[dict] = []
    page = 0

    while True:
        page += 1
        print(f"[02] [{category}] List page {page} … ", end="", flush=True)

        r = client.get(
            f"{_BASE}/v2/provider-docs",
            params={
                "filter[provider-version]": version_id,
                "filter[category]":        category,
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
                "id":             item["id"],
                "slug":           attrs.get("slug", ""),
                "title":          attrs.get("title", ""),
                # Some providers' Registry API responses (e.g. kreuzwerker/
                # docker) return an explicit JSON null for "subcategory"
                # rather than omitting the key. dict.get(key, default) only
                # substitutes default when the KEY is absent, so a present-
                # but-null value passes straight through as None and later
                # crashes any `.strip()` call downstream. `or ""` normalises
                # both "missing" and "present but null" to "".
                "subcategory":    attrs.get("subcategory") or "",
                "path":           attrs.get("path", ""),
                "is_data_source": category == "data-sources",
            })

        print(f"{len(items)} stubs (total this category: {len(stubs)})")

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

def _slug_to_tf_name(slug: str, resource_prefix: str) -> str:
    """Registry slugs omit the resource-type prefix; add it back.

    's3_bucket'  + 'aws_'    → 'aws_s3_bucket'
    'container'  + 'docker_' → 'docker_container'
    """
    return slug if slug.startswith(resource_prefix) else f"{resource_prefix}{slug}"


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
    all_stubs: list[dict] = []
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        for provider in PROVIDERS:
            print(f"\n[02] ##### Provider: {provider['namespace']}/{provider['name']}@{provider['version']} #####")
            version_id = _resolve_version_id(client, provider)

            for category in CATEGORIES:
                print(f"\n[02] === Fetching category: {category} ===")
                category_stubs = _fetch_doc_stubs(client, version_id, category)
                # Tag each stub with the resource_prefix of the provider it
                # came from, so _slug_to_tf_name below prefixes it correctly
                # (e.g. 'container' → 'docker_container', not 'aws_container').
                for stub in category_stubs:
                    stub["resource_prefix"] = provider["resource_prefix"]
                all_stubs.extend(category_stubs)
                print(f"[02] {len(category_stubs)} stubs collected for '{category}'")

    print(f"\n[02] {len(all_stubs)} total stubs collected across {len(PROVIDERS)} provider(s). "
          f"Fetching full content …\n")
    contents = _fetch_all_contents(all_stubs)  # {doc_id: markdown}

    docs: dict[str, dict] = {}
    for stub in all_stubs:
        tf_name = _slug_to_tf_name(stub["slug"], stub["resource_prefix"])
        content = contents.get(stub["id"], "")
        docs[tf_name] = {
            "title":          stub["title"],
            "subcategory":    stub["subcategory"],
            "is_data_source": stub["is_data_source"],
            "content":        content,
            "hcl_examples":   extract_hcl_examples(content),
        }

    OUTPUT_FILE.write_text(json.dumps(docs, indent=2), encoding="utf-8")

    resource_count     = sum(1 for d in docs.values() if not d["is_data_source"])
    data_source_count  = sum(1 for d in docs.values() if d["is_data_source"])
    print(f"\n[02] Saved {len(docs)} total docs → {OUTPUT_FILE}")
    print(f"     Managed resources : {resource_count}")
    print(f"     Data sources      : {data_source_count}")

    # Sanity checks
    s3 = docs.get("aws_s3_bucket", {})
    print(f"[02] Sanity aws_s3_bucket   : {len(s3.get('hcl_examples', []))} HCL examples, "
          f"{len(s3.get('content', ''))} content chars")
    ebs = docs.get("aws_elastic_beanstalk_solution_stack", {})
    print(f"[02] Sanity aws_elastic_beanstalk_solution_stack: "
          f"is_data_source={ebs.get('is_data_source')}, "
          f"{len(ebs.get('hcl_examples', []))} HCL examples")

    docker_container = docs.get("docker_container", {})
    print(f"[02] Sanity docker_container: present={'docker_container' in docs}, "
          f"{len(docker_container.get('hcl_examples', []))} HCL examples")
    docker_image = docs.get("docker_image", {})
    print(f"[02] Sanity docker_image    : present={'docker_image' in docs}, "
          f"{len(docker_image.get('hcl_examples', []))} HCL examples")
