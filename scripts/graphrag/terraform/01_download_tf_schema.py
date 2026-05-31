"""01_download_tf_schema.py

Download the pre-built AWS provider schema JSON from the official HashiCorp
GitHub release artifacts.  This is the canonical machine-readable schema,
identical to the output of `terraform providers schema -json`, published on
every provider release since v4.x — no Terraform binary or `terraform init`
required.

Output
------
tf_schema_raw.json   — raw schema as returned by the release artifact
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pin to a specific provider version for reproducibility.
# Update this constant when you want to refresh the knowledge graph.
PROVIDER_VERSION = "5.97.0"

_RELEASE_URL = (
    "https://github.com/hashicorp/terraform-provider-aws/releases/download"
    f"/v{PROVIDER_VERSION}"
    f"/terraform-provider-aws_{PROVIDER_VERSION}_SCHEMA.json"
)

OUTPUT_FILE = Path("tf_schema_raw.json")


def download_schema(url: str, output: Path) -> dict:
    """Stream the schema JSON from the GitHub release artifact."""
    print(f"[01] Downloading AWS provider schema v{PROVIDER_VERSION} ...")
    print(f"     URL: {url}")

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()

    schema = response.json()

    # Validate the top-level shape expected by downstream scripts.
    provider_key = "registry.terraform.io/hashicorp/aws"
    provider_schemas = schema.get("provider_schemas", {})
    if provider_key not in provider_schemas:
        # Some release artifacts nest under a slightly different key; try to
        # detect it and warn instead of hard-failing.
        found_keys = list(provider_schemas.keys())
        print(
            f"[01] WARNING: expected key '{provider_key}' not found. "
            f"Found: {found_keys}. Downstream scripts may need adjustment.",
            file=sys.stderr,
        )
    else:
        resource_schemas = (
            provider_schemas[provider_key].get("resource_schemas", {})
        )
        print(f"[01] Schema contains {len(resource_schemas):,} resource types.")

    output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"[01] Saved to: {output}")
    return schema


if __name__ == "__main__":
    download_schema(_RELEASE_URL, OUTPUT_FILE)
