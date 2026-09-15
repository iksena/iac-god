#!/usr/bin/env python3
"""
prepare_aws_account.py
=======================
Prepares (or tears down) the AWS-account-level prerequisites some ground-truth
benchmark scenarios assume already exist, before those scenarios can deploy --
scanned from the aws_services column already parsed for every row of:

    data_analysis/cfn_benchmark/dataset/final_benchmark_real_aws_with_prompts.csv   (CFN)
    data_analysis/iac_benchmark/dataset/final_benchmark_real_aws_with_prompts.csv   (Terraform)

Both data_analysis/CFN_Benchmark_Analytics.ipynb and .../TF_Benchmark_Analytics.ipynb
already have their own "pre-flight"/"post-flight" notebook cells doing exactly
this (same idempotent boto3 logic reused verbatim below). This script exists
to run that same setup/teardown as a standalone CLI, in one pass across BOTH
benchmarks' scenario sets, without opening either notebook.

What actually needs preparing, and why
----------------------------------------
- AWS Config (AWS::Config::ConfigRule / ConformancePack, aws_config_config_rule /
  aws_config_conformance_pack, ...): needs an active Configuration Recorder +
  Delivery Channel to already exist before a rule/pack resource can attach to
  it. A scenario that creates its OWN recorder is excluded from this check --
  AWS allows only ONE recorder per account/region, so pre-creating one would
  make that scenario fail with a conflict instead of succeed (confirmed
  against the current Terraform benchmark: its only 2 Config-rule-using
  scenarios both already create their own recorder, so this script correctly
  finds nothing to prepare for Config on the TF side today).
- Security Hub (AWS::SecurityHub::AutomationRule / aws_securityhub_automation_rule,
  ...): needs Security Hub itself already enabled. A scenario that enables its
  own hub (AWS::SecurityHub::Hub / aws_securityhub_account) is excluded, same
  one-per-account reasoning.
- ImageBuilder (AWS::ImageBuilder::* / aws_imagebuilder_*, or a direct Inspector
  reference): AMI distribution needs Amazon Inspector already enabled for EC2,
  and the account's "Block Public Access for AMIs" setting disabled.

GuardDuty and Organizations-scoped resources are DELIBERATELY never touched:
every GuardDuty-referencing scenario in both benchmarks already creates its
own detector (one per account/region -- pre-creating one would break them,
not help them), and Organizations-scoped resources need a real AWS
Organizations management account with member accounts, which no single-account
setting can satisfy. Both are still reported for visibility.

CFN vs. Terraform: --type switches matching STRATEGY, not just which file to read
------------------------------------------------------------------------------------
CFN's aws_services column already holds clean, human-readable AWS service
names (e.g. "Config", "SecurityHub", "GuardDuty") -- matched here by EXACT
token, never substring (a substring match on "Config" would false-positive on
the unrelated "AppConfig" service, which is genuinely present in this
benchmark's own data). Terraform's aws_services column instead holds raw
Terraform resource TYPE identifiers (e.g. "aws_config_config_rule",
"aws_securityhub_account") -- matched here by resource-type-family prefix.
They are genuinely different data shapes, not just different filenames, which
is why --type switches the whole detection strategy rather than just the CSV
path.

Cost / safety
--------------
Nothing here is free once enabled: Config bills per configuration-item-
recorded + per-rule-evaluation, Security Hub has a 30-day trial then bills per
check/finding, Inspector for EC2 bills per instance-hour scanned (free first
15 days). Expect well under $1 for a normal benchmark pass. Every action is
idempotent (checks current state before acting) and every setup action has a
matching teardown action via --teardown. Nothing calls AWS without an explicit
confirmation unless --yes is passed, and --dry-run never calls AWS at all.

Usage
------
    python3 prepare_aws_account.py --profile default --type both
    python3 prepare_aws_account.py --profile senatwo --type tf --dry-run
    python3 prepare_aws_account.py --profile default --type cfn --teardown --yes
"""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../research/
CFN_CSV_DEFAULT = REPO_ROOT / "data_analysis" / "cfn_benchmark" / "dataset" / "final_benchmark_real_aws_with_prompts.csv"
TF_CSV_DEFAULT = REPO_ROOT / "data_analysis" / "iac_benchmark" / "dataset" / "final_benchmark_real_aws_with_prompts.csv"

# A single canonical stack name regardless of --type: AWS Config allows only
# ONE recorder per account/region no matter which benchmark asked for it, so
# using per-type stack names (as the two notebooks do, since each is run
# independently) would just mean the second one silently never gets created
# -- and its own teardown would then have nothing to find. One name here
# keeps setup/teardown bookkeeping correct when --type both combines them.
STACK_NAME = "benchmark-eval-gt-config-prereq"

_CONFIG_PREREQ_TEMPLATE = """
AWSTemplateFormatVersion: '2010-09-09'
Description: Minimal AWS Config recorder + delivery channel prerequisite, so
  Config-rule-based ground-truth scenarios have something to attach to.
Resources:
  ConfigBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'benchmark-eval-gt-config-${AWS::AccountId}-${AWS::Region}'
  ConfigRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: config.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWS_ConfigRole
      Policies:
        - PolicyName: ConfigBucketDelivery
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:PutObject
                  - s3:GetBucketAcl
                Resource:
                  - !GetAtt ConfigBucket.Arn
                  - !Sub '${ConfigBucket.Arn}/*'
  ConfigRecorder:
    Type: AWS::Config::ConfigurationRecorder
    Properties:
      Name: default
      RoleARN: !GetAtt ConfigRole.Arn
      RecordingGroup:
        AllSupported: true
        IncludeGlobalResourceTypes: true
  DeliveryChannel:
    Type: AWS::Config::DeliveryChannel
    Properties:
      S3BucketName: !Ref ConfigBucket
"""


# ── CSV scanning: figure out which prerequisites are actually needed ─────────

def _split_tokens(value, sep=","):
    if not value:
        return []
    return [t.strip() for t in str(value).split(sep) if t.strip()]


def _read_rows(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scan_cfn(csv_path):
    """CFN's aws_services column: clean, human-readable service names, comma-separated.
    Matched by exact token -- substring matching would false-positive on
    unrelated services that merely contain the same word (e.g. "AppConfig").
    """
    rows = _read_rows(csv_path)
    result = {
        "type": "CFN",
        "csv_path": str(csv_path),
        "total": len(rows),
        "needs_config": [],
        "needs_security_hub": [],
        "needs_imagebuilder": [],
        "guardduty_scenarios": [],
        "guardduty_self_contained": 0,
        "organizations_scenarios": [],
    }
    for r in rows:
        services = _split_tokens(r.get("aws_services"), sep=",")
        resource_types = _split_tokens(r.get("resource_types"), sep="|")
        ident = r.get("dest_file") or r.get("content_hash") or "?"

        if "Config" in services:
            self_contained = any("ConfigurationRecorder" in rt for rt in resource_types)
            if not self_contained:
                result["needs_config"].append(ident)

        if "SecurityHub" in services:
            self_contained = any(rt == "AWS::SecurityHub::Hub" for rt in resource_types)
            if not self_contained:
                result["needs_security_hub"].append(ident)

        if "ImageBuilder" in services or "Inspector" in services:
            result["needs_imagebuilder"].append(ident)

        if "GuardDuty" in services:
            result["guardduty_scenarios"].append(ident)
            if any(rt == "AWS::GuardDuty::Detector" for rt in resource_types):
                result["guardduty_self_contained"] += 1

        if any(s == "Organizations" or s.startswith("Organizations") for s in services):
            result["organizations_scenarios"].append(ident)

    return result


def scan_tf(csv_path):
    """Terraform's aws_services column: raw Terraform resource TYPE identifiers
    (aws_*), comma-separated -- matched by resource-type-family prefix, since
    there's no separate human-readable service-name column to match exactly.
    """
    rows = _read_rows(csv_path)
    result = {
        "type": "Terraform",
        "csv_path": str(csv_path),
        "total": len(rows),
        "needs_config": [],
        "needs_security_hub": [],
        "needs_imagebuilder": [],
        "guardduty_scenarios": [],
        "guardduty_self_contained": 0,
        "organizations_scenarios": [],
    }
    config_rule_prefixes = (
        "aws_config_config_rule",
        "aws_config_conformance_pack",
        "aws_config_organization_managed_rule",
        "aws_config_organization_custom_rule",
        "aws_config_organization_conformance_pack",
    )
    for r in rows:
        tokens = _split_tokens(r.get("aws_services"), sep=",")
        ident = r.get("folder_path") or r.get("scenario_id") or "?"

        has_config_rule = any(t.startswith(config_rule_prefixes) for t in tokens)
        has_own_recorder = "aws_config_configuration_recorder" in tokens
        if has_config_rule and not has_own_recorder:
            result["needs_config"].append(ident)

        has_securityhub_resource = any(
            t.startswith("aws_securityhub_") and t != "aws_securityhub_account" for t in tokens
        )
        has_own_hub = "aws_securityhub_account" in tokens
        if has_securityhub_resource and not has_own_hub:
            result["needs_security_hub"].append(ident)

        if any(t.startswith("aws_imagebuilder_") or t.startswith("aws_inspector") for t in tokens):
            result["needs_imagebuilder"].append(ident)

        if any(t.startswith("aws_guardduty_") for t in tokens):
            result["guardduty_scenarios"].append(ident)
            if "aws_guardduty_detector" in tokens:
                result["guardduty_self_contained"] += 1

        if any(t.startswith("aws_organizations_") for t in tokens):
            result["organizations_scenarios"].append(ident)

    return result


def print_scan_report(results):
    print("=" * 78)
    print("Account-prerequisite scan (from each benchmark's own aws_services column)")
    print("=" * 78)
    for r in results:
        print(f"\n[{r['type']}] {r['csv_path']}  ({r['total']} scenarios)")
        print(f"  Needs pre-existing AWS Config recorder : {len(r['needs_config'])}")
        for ident in r["needs_config"][:10]:
            print(f"      - {ident}")
        print(f"  Needs pre-enabled Security Hub         : {len(r['needs_security_hub'])}")
        for ident in r["needs_security_hub"][:10]:
            print(f"      - {ident}")
        print(f"  Needs Inspector-for-EC2 + AMI-public-access-off (ImageBuilder): "
              f"{len(r['needs_imagebuilder'])}")
        for ident in r["needs_imagebuilder"][:10]:
            print(f"      - {ident}")
        n_gd = len(r["guardduty_scenarios"])
        print(f"  GuardDuty-referencing scenarios (NOT touched, self-contained "
              f"{r['guardduty_self_contained']}/{n_gd}): {n_gd}")
        n_org = len(r["organizations_scenarios"])
        print(f"  Organizations-scoped scenarios (out of scope, NOT touched): {n_org}")
    print()


def merge_needs(results):
    return {
        "config": any(r["needs_config"] for r in results),
        "security_hub": any(r["needs_security_hub"] for r in results),
        "imagebuilder": any(r["needs_imagebuilder"] for r in results),
    }


# ── boto3 actions (idempotent, reused from the two notebooks' own pre-flight /
# post-flight cells -- see CFN_Benchmark_Analytics.ipynb section 3/6 and
# TF_Benchmark_Analytics.ipynb section 3/6 for the original, independently
# proven versions this is kept in sync with) ──────────────────────────────────

def preflight_config(session):
    import botocore

    cfn = session.client("cloudformation")
    configservice = session.client("config")

    existing = configservice.describe_configuration_recorders().get("ConfigurationRecorders", [])
    if existing:
        print(f"[ok] AWS Config already has a recorder ({existing[0]['name']}) -- nothing to do.")
        return

    try:
        desc = cfn.describe_stacks(StackName=STACK_NAME)
        status = desc["Stacks"][0]["StackStatus"]
        print(f"[ok] Prereq stack {STACK_NAME!r} already exists (status={status}).")
        return
    except botocore.exceptions.ClientError:
        pass  # doesn't exist yet, fall through to create it

    print(f"[..] Creating AWS Config prerequisite stack {STACK_NAME!r}...")
    cfn.create_stack(
        StackName=STACK_NAME,
        TemplateBody=_CONFIG_PREREQ_TEMPLATE,
        Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
    )
    waiter = cfn.get_waiter("stack_create_complete")
    waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    print("[ok] Config recorder + delivery channel created.")

    configservice.start_configuration_recorder(ConfigurationRecorderName="default")
    print("[ok] Configuration recorder started.")


def preflight_security_hub(session):
    import botocore

    securityhub = session.client("securityhub")
    try:
        securityhub.describe_hub()
        print("[ok] Security Hub already enabled -- nothing to do.")
        return
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("InvalidAccessException", "ResourceNotFoundException"):
            raise

    print("[..] Enabling Security Hub...")
    securityhub.enable_security_hub(EnableDefaultStandards=False)
    print("[ok] Security Hub enabled (default standards left off to avoid extra check volume/cost).")


def preflight_inspector_ec2(session):
    inspector2 = session.client("inspector2")
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    status = inspector2.batch_get_account_status(accountIds=[account_id])
    resource_state = status["accounts"][0]["resourceState"] if status["accounts"] else {}
    ec2_state = resource_state.get("ec2", {}).get("status")
    if ec2_state == "ENABLED":
        print("[ok] Amazon Inspector already enabled for EC2 -- nothing to do.")
        return

    print("[..] Enabling Amazon Inspector for EC2...")
    inspector2.enable(accountIds=[account_id], resourceTypes=["EC2"])
    print("[ok] Amazon Inspector enabled for EC2 (needed for ImageBuilder AMI distribution).")


def preflight_ami_block_public_access(session):
    ec2 = session.client("ec2")
    current = ec2.get_image_block_public_access_state()
    if current.get("ImageBlockPublicAccessState") == "unblocked":
        print('[ok] EC2 AMI block-public-access already disabled -- nothing to do.')
        return

    print('[..] Disabling EC2 "Block Public Access for AMIs"...')
    ec2.disable_image_block_public_access()
    print("[ok] AMI block-public-access disabled (needed for ImageBuilder distribution to succeed).")


def postflight_config(session):
    import botocore

    cfn = session.client("cloudformation")
    configservice = session.client("config")
    s3 = session.client("s3")

    try:
        desc = cfn.describe_stacks(StackName=STACK_NAME)
        status = desc["Stacks"][0]["StackStatus"]
    except botocore.exceptions.ClientError:
        print(f"[ok] Prereq stack {STACK_NAME!r} doesn't exist -- nothing to tear down.")
        return

    print(f"[..] Tearing down AWS Config prerequisite stack {STACK_NAME!r} (status={status})...")
    try:
        configservice.stop_configuration_recorder(ConfigurationRecorderName="default")
        print("    Stopped configuration recorder.")
    except botocore.exceptions.ClientError as e:
        print(f"    (recorder already stopped or missing: {e.response.get('Error', {}).get('Code', '')})")

    bucket_name = None
    for res in cfn.list_stack_resources(StackName=STACK_NAME)["StackResourceSummaries"]:
        if res["LogicalResourceId"] == "ConfigBucket":
            bucket_name = res["PhysicalResourceId"]
            break
    if bucket_name:
        paginator = s3.get_paginator("list_object_versions")
        to_delete = []
        for page in paginator.paginate(Bucket=bucket_name):
            for v in page.get("Versions", []) + page.get("DeleteMarkers", []):
                to_delete.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        for i in range(0, len(to_delete), 1000):
            batch = to_delete[i:i + 1000]
            if batch:
                s3.delete_objects(Bucket=bucket_name, Delete={"Objects": batch})
        if to_delete:
            print(f"    Emptied {len(to_delete)} object(s) from {bucket_name} before stack deletion.")

    cfn.delete_stack(StackName=STACK_NAME)
    waiter = cfn.get_waiter("stack_delete_complete")
    waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    print("[ok] Config recorder + delivery channel + bucket + role deleted.")


def postflight_security_hub(session):
    import botocore

    securityhub = session.client("securityhub")
    try:
        securityhub.describe_hub()
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("InvalidAccessException", "ResourceNotFoundException"):
            print("[ok] Security Hub already disabled -- nothing to do.")
            return
        raise

    print("[..] Disabling Security Hub...")
    securityhub.disable_security_hub()
    print("[ok] Security Hub disabled.")


def postflight_inspector_ec2(session):
    inspector2 = session.client("inspector2")
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    status = inspector2.batch_get_account_status(accountIds=[account_id])
    resource_state = status["accounts"][0]["resourceState"] if status["accounts"] else {}
    ec2_state = resource_state.get("ec2", {}).get("status")
    if ec2_state in (None, "DISABLED", "DISABLING"):
        print("[ok] Amazon Inspector for EC2 already disabled -- nothing to do.")
        return

    print("[..] Disabling Amazon Inspector for EC2...")
    inspector2.disable(accountIds=[account_id], resourceTypes=["EC2"])
    print("[ok] Amazon Inspector for EC2 disabled.")


def postflight_ami_block_public_access(session):
    ec2 = session.client("ec2")
    current = ec2.get_image_block_public_access_state()
    if current.get("ImageBlockPublicAccessState") == "block-new-sharing":
        print("[ok] EC2 AMI block-public-access already restored -- nothing to do.")
        return

    print('[..] Restoring EC2 "Block Public Access for AMIs" to its default...')
    ec2.enable_image_block_public_access(ImageBlockPublicAccessState="block-new-sharing")
    print("[ok] AMI block-public-access restored (this setting has no cost either way -- restored for symmetry).")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare or tear down AWS-account-level prerequisites for the CFN/Terraform "
                    "real-AWS benchmark scenarios, based on each benchmark's own parsed aws_services.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type", choices=["cfn", "tf", "both"], default="both",
        help="Which benchmark's scenarios to scan/prepare for -- CFN's aws_services is a clean "
             "service-name list, Terraform's is raw resource-type identifiers, so this switches "
             "the matching strategy, not just the file read (default: both).",
    )
    parser.add_argument("--profile", required=True, help="AWS CLI profile to use (e.g. default, senatwo).")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"),
                        help="AWS region (default: $AWS_REGION or us-east-1).")
    parser.add_argument(
        "--teardown", action="store_true",
        help="Tear down/disable everything this script (or the notebooks' own pre-flight cells) "
             "may have turned on, instead of setting it up. Always attempted unconditionally for "
             "all 4 settings (idempotent no-op if something was never enabled), regardless of "
             "--type, so a stale setting left over from a differently-scoped run doesn't linger.",
    )
    parser.add_argument("--cfn-csv", type=Path, default=CFN_CSV_DEFAULT,
                        help=f"Path to the CFN benchmark CSV (default: {CFN_CSV_DEFAULT}).")
    parser.add_argument("--tf-csv", type=Path, default=TF_CSV_DEFAULT,
                        help=f"Path to the Terraform benchmark CSV (default: {TF_CSV_DEFAULT}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only scan and report what would happen -- never calls AWS.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the interactive confirmation prompt before calling AWS.")
    args = parser.parse_args()

    results = []
    if args.type in ("cfn", "both"):
        results.append(scan_cfn(args.cfn_csv))
    if args.type in ("tf", "both"):
        results.append(scan_tf(args.tf_csv))

    print_scan_report(results)

    if args.dry_run:
        print("Dry run -- no AWS calls made. Rerun without --dry-run to actually "
              f"{'tear down' if args.teardown else 'set up'} the prerequisites above.")
        return

    needs = merge_needs(results)
    action_word = "TEAR DOWN" if args.teardown else "SET UP"
    if not args.teardown:
        planned = [name for name, on in
                   [("AWS Config recorder", needs["config"]),
                    ("Security Hub", needs["security_hub"]),
                    ("Inspector-for-EC2 + AMI-public-access-off", needs["imagebuilder"])]
                   if on]
        if not planned:
            print("Nothing needs preparing for the selected --type -- exiting without calling AWS.")
            return
        print(f"About to {action_word}: {', '.join(planned)} "
              f"(profile={args.profile!r}, region={args.region!r}).")
    else:
        print(f"About to {action_word} everything this script can turn on/off "
              f"(AWS Config recorder, Security Hub, Inspector-for-EC2, AMI-public-access) "
              f"(profile={args.profile!r}, region={args.region!r}).")

    if not args.yes:
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted -- no AWS calls made.")
            return

    import boto3

    session = boto3.Session(profile_name=args.profile, region_name=args.region)

    if args.teardown:
        postflight_config(session)
        postflight_security_hub(session)
        postflight_inspector_ec2(session)
        postflight_ami_block_public_access(session)
        print("\n[ok] Teardown complete.")
    else:
        if needs["config"]:
            preflight_config(session)
        else:
            print("[--] Skipping AWS Config -- no selected scenario needs a pre-existing recorder.")
        if needs["security_hub"]:
            preflight_security_hub(session)
        else:
            print("[--] Skipping Security Hub -- no selected scenario needs a pre-enabled hub.")
        if needs["imagebuilder"]:
            preflight_inspector_ec2(session)
            preflight_ami_block_public_access(session)
        else:
            print("[--] Skipping Inspector-for-EC2/AMI-public-access -- no selected scenario needs it.")
        print("\n[ok] Setup complete.")
        if any(r["guardduty_scenarios"] for r in results):
            print("[ok] GuardDuty was intentionally NOT pre-enabled -- referencing scenarios create "
                  "their own detector, and an account can only have one per region.")
        if any(r["organizations_scenarios"] for r in results):
            print("[ok] Organizations-scoped scenarios were intentionally NOT touched -- they need a "
                  "real AWS Organizations management account, which no single-account setting can satisfy.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
