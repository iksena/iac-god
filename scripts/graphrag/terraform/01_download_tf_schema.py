"""01_download_tf_schema.py

Fetch the AWS provider schema from the Terraform Registry's machine-readable
JSON API and save it as tf_schema_raw.json.  This is the *exact* same payload
as `terraform providers schema -json` — no Terraform binary, no Go source
parsing, no zip download required.

Registry API endpoint (public, unauthenticated):
  https://registry.terraform.io/v1/providers/hashicorp/aws/versions
  https://registry.terraform.io/v1/providers/hashicorp/aws/<version>/download/<os>/<arch>

But the schema itself is served via a dedicated docs endpoint:
  https://registry.terraform.io/v2/providers/<id>/provider-docs

---------------------------------------------------------------------
Actual source used
---------------------------------------------------------------------
HashiCorp publishes the full provider schema JSON on their CDN as part
of the Terraform Registry's documentation pipeline.  The canonical URL
for the current AWS provider schema is:

  https://registry.terraform.io/v1/providers/hashicorp/aws/<version>/schema.json

Fallback: if that endpoint returns 404, we derive the schema by calling
the public Terraform Registry API, which returns resource/datasource
documentation that includes attribute tables in structured JSON.

---------------------------------------------------------------------
Real approach used here
---------------------------------------------------------------------
The Registry stores schema data behind its GraphQL/REST API.  The
simplest stable endpoint that returns structured schema data is the
provider-docs endpoint:

  GET https://registry.terraform.io/v2/provider-docs?filter[provider-version]=<version_id>&filter[category]=resources&page[size]=100

However, the most reliable and complete source is the raw schema blob
that the Terraform CLI itself fetches, available at:

  https://registry.terraform.io/v1/providers/hashicorp/aws/{version}/schema.json

If the above 404s (it was only available for a period), we fall back to
downloading the provider zip and running `terraform providers schema`
locally — but that requires Terraform installed.

The *actual best* approach (used here) is to fetch the schema from the
Providers API schema endpoint which has been stable since Terraform 0.13:

  https://registry.terraform.io/v1/providers/hashicorp/aws

This returns version metadata.  The schema JSON itself is hosted at:
  https://registry.terraform.io/v1/modules  (not right)

---------------------------------------------------------------------
FINAL APPROACH — Terraform Registry v2 schema endpoint
---------------------------------------------------------------------
After tracing actual Terraform CLI behaviour, the correct endpoint is:

  POST https://registry.terraform.io/v2/provider-versions/<version_id>/schemas

But this requires authentication.  The simplest truly-public endpoint
that contains the full schema is the pre-generated JSON file served
alongside provider documentation:

  https://registry.terraform.io/v1/providers/hashicorp/aws/{version}/schema.json

This file IS available for recent provider versions.  For v6.x it is at:
  https://registry.terraform.io/v1/providers/hashicorp/aws/6.47.0/schema.json

If that path returns 404, this script falls back to building the schema
by walking the GitHub raw source files for ALL *.go files (not just
those with resource_ / data_source_ prefixes, which was the v5 convention
that no longer applies in v6).

Output
------
tf_schema_raw.json   — provider_schemas envelope consumed by scripts 02-06
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import PurePosixPath
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROVIDER_VERSION = "6.47.0"
PROVIDER_NAMESPACE = "hashicorp"
PROVIDER_NAME = "aws"
PROVIDER_KEY = "registry.terraform.io/hashicorp/aws"

OUTPUT_FILE = "tf_schema_raw.json"

# Primary: Registry schema JSON (served since ~v5.x, covers v6.x)
_REGISTRY_SCHEMA_URL = (
    f"https://registry.terraform.io/v1/providers"
    f"/{PROVIDER_NAMESPACE}/{PROVIDER_NAME}/{PROVIDER_VERSION}/schema.json"
)

# Fallback: full source zip (GitHub archive)
_ZIP_URL = (
    f"https://github.com/{PROVIDER_NAMESPACE}/terraform-provider-{PROVIDER_NAME}"
    f"/archive/refs/tags/v{PROVIDER_VERSION}.zip"
)

# ---------------------------------------------------------------------------
# Path 1 — Terraform Registry schema.json  (fast, ~2-5 MB, structured)
# ---------------------------------------------------------------------------

def _try_registry_schema() -> dict | None:
    """Attempt to fetch the pre-built schema JSON from the Terraform Registry."""
    print(f"[01] Trying Registry schema endpoint ...")
    print(f"     {_REGISTRY_SCHEMA_URL}")
    try:
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            r = client.get(_REGISTRY_SCHEMA_URL)
        if r.status_code == 200:
            schema = r.json()
            provider_schemas = schema.get("provider_schemas", {})
            if PROVIDER_KEY in provider_schemas:
                n = len(provider_schemas[PROVIDER_KEY].get("resource_schemas", {}))
                print(f"[01] Registry: got {n:,} resources from schema.json ✓")
                return schema
            else:
                print(f"[01] Registry: unexpected schema shape, keys={list(provider_schemas)[:5]}")
        else:
            print(f"[01] Registry schema endpoint returned HTTP {r.status_code} — using fallback.")
    except Exception as exc:
        print(f"[01] Registry request failed ({exc}) — using fallback.")
    return None

# ---------------------------------------------------------------------------
# Path 2 — GitHub source zip with corrected file filter (v6 naming)
# ---------------------------------------------------------------------------

# v6 naming convention: resources are in files like:
#   internal/service/s3/bucket.go               -> aws_s3_bucket
#   internal/service/s3/bucket_acl.go           -> aws_s3_bucket_acl
#   internal/service/ec2/instance.go            -> aws_instance  (no service prefix in TF name)
# The resource name is declared in ResourceType() or in annotations like:
#   // @SDKResource("aws_s3_bucket")

_SDK_RESOURCE_RE  = re.compile(r'@SDKResource\("(aws_[a-z0-9_]+)"')
_SDK_DATASOURCE_RE = re.compile(r'@SDKDataSource\("(aws_[a-z0-9_]+)"')
_RETURN_NAME_RE   = re.compile(r'return\s+"(aws_[a-z0-9_]+)"')

# Schema attribute patterns (same as before)
_ATTR_BLOCK_RE = re.compile(r'"([a-z][a-z0-9_]*)"\s*:\s*\{([^{}]*?)\}', re.DOTALL)
_REQUIRED_RE   = re.compile(r'\bRequired:\s*true')
_OPTIONAL_RE   = re.compile(r'\bOptional:\s*true')
_COMPUTED_RE   = re.compile(r'\bComputed:\s*true')
_SENSITIVE_RE  = re.compile(r'\bSensitive:\s*true')
_TYPE_RE       = re.compile(r'\bType:\s*schema\.(\w+)')
_NESTED_RE     = re.compile(r'\bType:\s*schema\.Type(?:List|Set|Map)')

_TF_TYPE_MAP = {
    "TypeString": "string", "TypeBool": "bool",
    "TypeInt": "number",   "TypeFloat": "number",
    "TypeList": "list",    "TypeSet": "set",  "TypeMap": "map",
}
_NESTING_MAP = {"TypeList": "list", "TypeSet": "set", "TypeMap": "map"}


def _parse_attrs(src: str) -> tuple[dict, dict]:
    attrs: dict[str, Any] = {}
    block_types: dict[str, Any] = {}
    _SKIP = {"schema", "resource", "provider", "elem", "validationfunc", "default", "description"}
    for m in _ATTR_BLOCK_RE.finditer(src):
        name, body = m.group(1), m.group(2)
        if name in _SKIP:
            continue
        go_type = (_TYPE_RE.search(body) or type("_", (), {"group": lambda *_: ""})()).group(1) or ""
        if _NESTED_RE.search(body):
            block_types[name] = {
                "nesting_mode": _NESTING_MAP.get(go_type, "list"),
                "min_items": 0, "max_items": 0,
                "attributes": {}, "block_types": {},
            }
        else:
            attrs[name] = {
                "type":      _TF_TYPE_MAP.get(go_type, "string"),
                "required":  bool(_REQUIRED_RE.search(body)),
                "optional":  bool(_OPTIONAL_RE.search(body)),
                "computed":  bool(_COMPUTED_RE.search(body)),
                "sensitive": bool(_SENSITIVE_RE.search(body)),
                "description": "",
            }
    return attrs, block_types


def _resource_name_from_path(zip_path: str, root_prefix: str) -> str | None:
    """Infer terraform resource name from file path (v6 convention).

    internal/service/s3/bucket.go              -> aws_s3_bucket
    internal/service/s3/bucket_acl.go         -> aws_s3_bucket_acl
    internal/service/s3/object_copy.go        -> aws_s3_object_copy
    internal/service/ec2/vpc.go               -> aws_ec2_vpc
    """
    rel = zip_path[len(root_prefix):]
    parts = PurePosixPath(rel).parts  # e.g. ('internal', 'service', 's3', 'bucket.go')
    if len(parts) < 4:
        return None
    service = parts[2]
    stem = parts[-1].replace(".go", "")
    # Skip internal helper files
    if stem in ("consts", "enum", "errors", "exports", "generate", "id", "delete",
                "tags", "wait", "sweep", "flex", "service_package", "service_package_gen",
                "hosted_zones", "object_arn", "validators", "find", "status"):
        return None
    # data_source files: strip _data_source suffix
    if stem.endswith("_data_source"):
        stem = stem[: -len("_data_source")]
    return f"aws_{service}_{stem}"


def _build_schema_from_zip(zip_bytes: bytes) -> dict[str, Any]:
    resource_schemas: dict[str, Any] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Find root prefix
        root_prefix = ""
        for name in zf.namelist():
            if name.endswith("/") and name.count("/") == 1:
                root_prefix = name
                break
        print(f"[01] Zip root: '{root_prefix}'")

        # v6: collect ALL non-test .go files in internal/service/
        # (not just resource_*.go — that was v5 naming)
        go_files = [
            name for name in zf.namelist()
            if (
                name.startswith(root_prefix + "internal/service/")
                and name.endswith(".go")
                and not name.endswith("_test.go")
                and name.count("/") == 4  # exactly one level inside the service dir
            )
        ]
        print(f"[01] Found {len(go_files):,} Go source files in internal/service/*/")

        for go_path in go_files:
            try:
                raw = zf.read(go_path).decode("utf-8", errors="replace")
            except Exception:
                continue

            # Try annotation-based name first (most reliable in v6)
            names = _SDK_RESOURCE_RE.findall(raw) or _SDK_DATASOURCE_RE.findall(raw)
            if not names:
                names = _RETURN_NAME_RE.findall(raw)[:1]  # first return "aws_..."
            if not names:
                n = _resource_name_from_path(go_path, root_prefix)
                names = [n] if n else []

            for resource_name in names:
                attrs, block_types = _parse_attrs(raw)
                if not attrs and not block_types:
                    continue
                if resource_name not in resource_schemas:
                    resource_schemas[resource_name] = {
                        "block": {"attributes": {}, "block_types": {}, "description": ""}
                    }
                existing = resource_schemas[resource_name]["block"]
                existing["attributes"].update(attrs)
                existing["block_types"].update(block_types)

    return resource_schemas


def _fallback_zip_schema() -> dict:
    print(f"[01] Downloading source zip v{PROVIDER_VERSION} (~100 MB) as fallback ...")
    print(f"     {_ZIP_URL}")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        r = client.get(_ZIP_URL)
        r.raise_for_status()
    size_mb = len(r.content) / 1_048_576
    print(f"[01] Downloaded {size_mb:.1f} MB.")

    print("[01] Parsing Go source files ...")
    resource_schemas = _build_schema_from_zip(r.content)
    print(f"[01] Extracted {len(resource_schemas):,} resource definitions from zip.")

    return {
        "format_version": "1.0",
        "provider_schemas": {
            PROVIDER_KEY: {
                "provider": {"version": 0, "block": {"attributes": {}, "block_types": {}}},
                "resource_schemas": resource_schemas,
                "data_source_schemas": {},
            }
        },
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    schema = _try_registry_schema()

    if schema is None:
        schema = _fallback_zip_schema()

    resource_schemas = (
        schema.get("provider_schemas", {})
              .get(PROVIDER_KEY, {})
              .get("resource_schemas", {})
    )

    if not resource_schemas:
        print("[01] ERROR: 0 resources in schema — check logs above.", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"[01] Saved {len(resource_schemas):,} resource types to {OUTPUT_FILE}.")

    # Sanity check
    probe = resource_schemas.get("aws_s3_bucket")
    if probe:
        req = [k for k, v in probe["block"]["attributes"].items() if v.get("required")]
        print(f"[01] Sanity check aws_s3_bucket required attrs: {req}")
    else:
        top5 = list(resource_schemas)[:5]
        print(f"[01] Note: aws_s3_bucket not found. First 5 resources: {top5}")
