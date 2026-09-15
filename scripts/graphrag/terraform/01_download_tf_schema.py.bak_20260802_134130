"""01_download_tf_schema.py

Strategy (in order of preference):
  1. Use terraform CLI if already installed → `terraform providers schema -json`
  2. Auto-install terraform CLI (single binary, ~60 MB) → same command
  3. Fall back to Go source zip parsing (current behaviour, known attribute-nesting bug)
"""
from __future__ import annotations

import io, json, os, platform, shutil, stat, subprocess, sys, tempfile, zipfile
from pathlib import Path

import httpx

PROVIDER_VERSION   = "6.47.0"
PROVIDER_KEY       = "registry.terraform.io/hashicorp/aws"
TERRAFORM_VERSION  = "1.15.5"
OUTPUT_FILE        = "tf_schema_raw.json"

# --------------------------------------------------------------------------
# Path 1 — use terraform CLI if already on PATH
# --------------------------------------------------------------------------

def _run_terraform_schema(tf_bin: str) -> dict | None:
    """Create a scratch workspace, run terraform init + providers schema -json."""
    main_tf = """
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "6.47.0" }
  }
}
provider "aws" {
  region                      = "us-east-1"
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
  access_key                  = "mock"
  secret_key                  = "mock"
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        tf_file = Path(tmp) / "main.tf"
        tf_file.write_text(main_tf)

        print(f"[01] Running terraform init (provider download ~30 MB) ...")
        init = subprocess.run(
            [tf_bin, "init", "-no-color"],
            cwd=tmp, capture_output=True, text=True, timeout=300
        )
        if init.returncode != 0:
            print(f"[01] terraform init failed:\n{init.stderr[-1000:]}")
            return None

        print("[01] Running terraform providers schema -json ...")
        schema_proc = subprocess.run(
            [tf_bin, "providers", "schema", "-json", "-no-color"],
            cwd=tmp, capture_output=True, text=True, timeout=120
        )
        if schema_proc.returncode != 0:
            print(f"[01] terraform providers schema failed:\n{schema_proc.stderr[-500:]}")
            return None

        schema = json.loads(schema_proc.stdout)
        n = len(
            schema.get("provider_schemas", {})
                  .get(PROVIDER_KEY, {})
                  .get("resource_schemas", {})
        )
        print(f"[01] terraform CLI: {n:,} resources ✓")
        return schema


# --------------------------------------------------------------------------
# Path 2 — auto-install terraform binary
# --------------------------------------------------------------------------

def _install_terraform() -> str | None:
    """Download the terraform binary for the current platform into a temp dir."""
    os_name = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(
        platform.system(), "linux"
    )
    machine = platform.machine().lower()
    arch = "amd64" if machine in ("x86_64", "amd64") else \
           "arm64"  if machine in ("aarch64", "arm64") else "amd64"

    url = (
        f"https://releases.hashicorp.com/terraform/{TERRAFORM_VERSION}/"
        f"terraform_{TERRAFORM_VERSION}_{os_name}_{arch}.zip"
    )
    print(f"[01] Downloading terraform {TERRAFORM_VERSION} binary ...")
    print(f"     {url}")

    try:
        with httpx.Client(follow_redirects=True, timeout=300) as client:
            r = client.get(url)
            r.raise_for_status()
    except Exception as exc:
        print(f"[01] Failed to download terraform binary: {exc}")
        return None

    tf_dir = Path(tempfile.mkdtemp(prefix="tf_bin_"))
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for member in zf.namelist():
            if "terraform" in member.lower() and not member.endswith("/"):
                zf.extract(member, tf_dir)
                tf_path = tf_dir / member
                tf_path.chmod(tf_path.stat().st_mode | stat.S_IEXEC)
                print(f"[01] terraform binary installed at {tf_path}")
                return str(tf_path)
    return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    schema = None

    # Try existing terraform on PATH first
    tf_on_path = shutil.which("terraform") or shutil.which("tofu")
    if tf_on_path:
        print(f"[01] Found terraform at {tf_on_path}")
        schema = _run_terraform_schema(tf_on_path)

    # Auto-install if not found or failed
    if schema is None:
        tf_bin = _install_terraform()
        if tf_bin:
            schema = _run_terraform_schema(tf_bin)

    # Final fallback: existing Go source zip parse (with accuracy caveat)
    if schema is None:
        print("[01] WARNING: falling back to Go source parsing (attribute nesting may be imprecise)")
        schema = _fallback_zip_schema()   # keep existing function

    resource_schemas = (
        schema.get("provider_schemas", {})
              .get(PROVIDER_KEY, {})
              .get("resource_schemas", {})
    )

    if not resource_schemas:
        print("[01] ERROR: 0 resources extracted.", file=sys.stderr)
        sys.exit(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"[01] Saved {len(resource_schemas):,} resource types to {OUTPUT_FILE}.")

    probe = resource_schemas.get("aws_s3_bucket", {})
    req_attrs = [
        k for k, v in probe.get("block", {}).get("attributes", {}).items()
        if v.get("required")
    ]
    print(f"[01] Sanity check aws_s3_bucket required attrs: {req_attrs}")
    # Correct answer: [] — aws_s3_bucket has NO required attrs (bucket name is optional/computed)