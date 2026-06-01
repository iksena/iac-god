import re
import time
import uuid
import boto3
import datetime
import requests
import subprocess
import tempfile
from pathlib import Path
from botocore.exceptions import ClientError
from config import DeployConfig, DeployTarget
from state import DeployValidationResult


# ---------------------------------------------------------------------------
# CloudFormation client factory
# ---------------------------------------------------------------------------

def _build_cfn_client(deploy_config: DeployConfig):
    """
    Build a boto3 CloudFormation client pointed at either LocalStack or real AWS.
    For LocalStack, endpoint_url redirects all API calls to localhost.
    """
    if deploy_config.target == DeployTarget.LOCALSTACK:
        return boto3.client(
            "cloudformation",
            endpoint_url=deploy_config.localstack_endpoint,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
    else:
        session = boto3.Session(profile_name=deploy_config.aws_profile)
        return session.client("cloudformation", region_name=deploy_config.aws_region)


# ---------------------------------------------------------------------------
# Error message formatting
# ---------------------------------------------------------------------------

def _format_failed_resources(failed_resources: list[dict]) -> str:
    """
    Build a human-readable error message that names each responsible resource.

    Format per resource:
        <LogicalResourceId|resource_address>: <status_reason>

    Multiple failures are joined with " | " so the message stays on one line
    while still being parseable by the remediator prompt.
    """
    if not failed_resources:
        return "Deployment failed (no resource-level detail available)"
    parts = [
        f"{r['logical_name']}: {r['status_reason']}"
        for r in failed_resources
        if r.get("logical_name") and r.get("status_reason")
    ]
    return " | ".join(parts) if parts else "Deployment failed (unknown reason)"


# ---------------------------------------------------------------------------
# LocalStack reset helpers
# ---------------------------------------------------------------------------

def _reset_localstack_state(deploy_config: DeployConfig):
    """
    Two-phase reset for LocalStack:

    Phase 1 — HTTP state reset
        POST /_localstack/state/reset clears all service state (S3, IAM, …).
        This is the broad greenfield reset.

    Phase 2 — Explicit stack deletion
        The HTTP reset may return 200 while CloudFormation stacks are still
        present in LocalStack's internal database.  We therefore list every
        iac-god-eval-* stack and explicitly delete each one through the
        CloudFormation API before proceeding.
    """
    # Phase 1: broad service reset
    try:
        resp = requests.post(
            f"{deploy_config.localstack_endpoint}/_localstack/state/reset",
            timeout=10,
        )
        if resp.status_code == 200:
            print("[Deploy] LocalStack state reset OK")
        else:
            print(f"[Deploy] LocalStack reset returned HTTP {resp.status_code} — proceeding")
    except requests.exceptions.ConnectionError:
        print("[Deploy] ⚠️  Could not connect to LocalStack for reset. Is it running?")
    except Exception as e:
        print(f"[Deploy] Reset error: {e}")

    time.sleep(deploy_config.localstack_reset_wait)

    # Phase 2: verify + explicitly delete any surviving evaluation stacks
    _delete_surviving_eval_stacks(deploy_config)


def _delete_surviving_eval_stacks(deploy_config: DeployConfig):
    """
    List all CloudFormation stacks visible to the target and delete any that
    carry the iac-god-eval- prefix.
    """
    cfn_client = _build_cfn_client(deploy_config)
    stack_prefix = "iac-god-eval-"

    active_statuses = [
        "CREATE_IN_PROGRESS", "CREATE_FAILED", "CREATE_COMPLETE",
        "ROLLBACK_IN_PROGRESS", "ROLLBACK_FAILED", "ROLLBACK_COMPLETE",
        "DELETE_IN_PROGRESS", "DELETE_FAILED",
        "UPDATE_IN_PROGRESS", "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
        "UPDATE_COMPLETE", "UPDATE_ROLLBACK_IN_PROGRESS",
        "UPDATE_ROLLBACK_FAILED", "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS",
        "UPDATE_ROLLBACK_COMPLETE", "REVIEW_IN_PROGRESS",
        "IMPORT_IN_PROGRESS", "IMPORT_COMPLETE",
        "IMPORT_ROLLBACK_IN_PROGRESS", "IMPORT_ROLLBACK_FAILED",
        "IMPORT_ROLLBACK_COMPLETE",
    ]

    try:
        paginator = cfn_client.get_paginator("list_stacks")
        targets: list[tuple[str, str]] = []

        for page in paginator.paginate(StackStatusFilter=active_statuses):
            for summary in page.get("StackSummaries", []):
                name = summary.get("StackName", "")
                sid = summary.get("StackId", "")
                if name.startswith(stack_prefix) and sid:
                    targets.append((sid, name))

        if not targets:
            print("[Deploy] No surviving evaluation stacks found — clean slate confirmed")
            return

        print(f"[Deploy] Deleting {len(targets)} surviving evaluation stack(s)...")
        for stack_id, stack_name in targets:
            print(f"  [Deploy] Deleting '{stack_name}'...")
            try:
                cfn_client.delete_stack(StackName=stack_id)
                _wait_for_stack_deletion(
                    cfn_client, stack_id, stack_name,
                    deploy_config.stack_deletion_timeout,
                )
                print(f"  [Deploy] '{stack_name}' deleted ✓")
            except Exception as e:
                print(f"  [Deploy] Warning: could not delete '{stack_name}': {e}")

    except Exception as e:
        print(f"[Deploy] Stack sweep error: {e}")


# ---------------------------------------------------------------------------
# AWS reset helper
# ---------------------------------------------------------------------------

def _reset_aws_state(deploy_config: DeployConfig):
    print("[Deploy] AWS state reset: scanning for prior evaluation stacks...")
    _delete_surviving_eval_stacks(deploy_config)


# ---------------------------------------------------------------------------
# Unified reset dispatcher
# ---------------------------------------------------------------------------

def _reset_target_state(deploy_config: DeployConfig):
    """Run pre-deployment state reset for the configured target."""
    if deploy_config.target == DeployTarget.LOCALSTACK:
        _reset_localstack_state(deploy_config)
    elif deploy_config.target == DeployTarget.AWS:
        _reset_aws_state(deploy_config)


# ---------------------------------------------------------------------------
# Stack deletion waiter
# ---------------------------------------------------------------------------

def _wait_for_stack_deletion(cfn_client, stack_id: str, stack_name: str, timeout: int):
    start = time.time()
    while time.time() - start < timeout:
        try:
            stack = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]
            status = stack["StackStatus"]
            if status == "DELETE_COMPLETE":
                return
            if status in ("ROLLBACK_COMPLETE", "CREATE_FAILED", "DELETE_FAILED"):
                try:
                    cfn_client.delete_stack(StackName=stack_name)
                except Exception:
                    pass
        except ClientError as e:
            if "does not exist" in str(e):
                return
            raise
        time.sleep(3)

    print(f"[Deploy] ⚠️  Stack deletion timed out after {timeout}s")
    try:
        current_status = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]["StackStatus"]
        print(f"[Deploy] Stack '{stack_name}' left in status: {current_status}")
        if current_status == "DELETE_FAILED":
            cfn_client.delete_stack(StackName=stack_id, RetainResources=[])
            print(f"[Deploy] Issued force-delete for '{stack_name}'")
    except ClientError as e:
        if "does not exist" in str(e):
            return
        print(f"[Deploy] Could not force-delete '{stack_name}': {e}")


# ---------------------------------------------------------------------------
# VPC quota pre-flight
# ---------------------------------------------------------------------------

def _delete_all_non_default_vpcs(deploy_config: DeployConfig) -> None:
    if deploy_config.target != DeployTarget.AWS:
        return

    session = boto3.Session(profile_name=deploy_config.aws_profile)
    ec2 = session.client("ec2", region_name=deploy_config.aws_region)

    try:
        vpcs = ec2.describe_vpcs()["Vpcs"]
    except Exception as e:
        print(f"[Deploy] VPC pre-flight: could not list VPCs: {e}")
        return

    non_default = [v for v in vpcs if not v.get("IsDefault", False)]
    if not non_default:
        return

    print(f"[Deploy] VPC pre-flight: deleting {len(non_default)} non-default VPC(s) to free quota...")

    for vpc in non_default:
        vpc_id = vpc["VpcId"]
        try:
            igws = ec2.describe_internet_gateways(
                Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}]
            )["InternetGateways"]
            for igw in igws:
                igw_id = igw["InternetGatewayId"]
                ec2.detach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
                ec2.delete_internet_gateway(InternetGatewayId=igw_id)

            subnets = ec2.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )["Subnets"]
            for subnet in subnets:
                ec2.delete_subnet(SubnetId=subnet["SubnetId"])

            rts = ec2.describe_route_tables(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )["RouteTables"]
            for rt in rts:
                is_main = any(
                    assoc.get("Main") for assoc in rt.get("Associations", [])
                )
                if not is_main:
                    ec2.delete_route_table(RouteTableId=rt["RouteTableId"])

            sgs = ec2.describe_security_groups(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )["SecurityGroups"]
            for sg in sgs:
                if sg["GroupName"] != "default":
                    ec2.delete_security_group(GroupId=sg["GroupId"])

            ec2.delete_vpc(VpcId=vpc_id)
            print(f"[Deploy] VPC pre-flight: deleted {vpc_id} ✓")

        except Exception as e:
            print(f"[Deploy] VPC pre-flight: could not fully delete {vpc_id}: {e}")


def _check_vpc_quota(deploy_config: DeployConfig) -> str | None:
    if deploy_config.target != DeployTarget.AWS:
        return None

    session = boto3.Session(profile_name=deploy_config.aws_profile)
    ec2 = session.client("ec2", region_name=deploy_config.aws_region)

    try:
        vpcs = ec2.describe_vpcs()["Vpcs"]
        quota = 5
        try:
            sq = session.client("service-quotas", region_name=deploy_config.aws_region)
            quota = int(
                sq.get_service_quota(
                    ServiceCode="vpc", QuotaCode="L-F678F1CE"
                )["Quota"]["Value"]
            )
        except Exception:
            pass

        if len(vpcs) >= quota:
            return (
                f"VPC_QUOTA_EXHAUSTED: {len(vpcs)}/{quota} VPCs in use in "
                f"{deploy_config.aws_region} — free VPC quota before deploying"
            )
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Parameter validation helper (CloudFormation)
# ---------------------------------------------------------------------------

def _required_parameter_keys(cfn_client, template: str) -> tuple[list[str], str | None]:
    try:
        response = cfn_client.validate_template(TemplateBody=template)
    except ClientError as e:
        return [], f"validate_template failed: {e}"

    required: list[str] = []
    for param in response.get("Parameters", []):
        key = param.get("ParameterKey")
        has_default = "DefaultValue" in param
        if key and not has_default:
            required.append(key)
    return required, None


# ---------------------------------------------------------------------------
# Event poller helper (CloudFormation)
# ---------------------------------------------------------------------------

def _drain_stack_events(
    cfn_client,
    stack_id: str,
    seen_events: set[str],
    last_timestamp: datetime.datetime,
    failed_resources: list[dict],
    completed_resources: list[str],
    deploy_logs: list[str],
) -> None:
    try:
        events = cfn_client.describe_stack_events(StackName=stack_id)["StackEvents"]
    except Exception:
        return

    for event in sorted(events, key=lambda x: x["Timestamp"]):
        if event["EventId"] in seen_events or event["Timestamp"] <= last_timestamp:
            continue
        seen_events.add(event["EventId"])
        rid = event["LogicalResourceId"]
        status = event["ResourceStatus"]
        reason = event.get("ResourceStatusReason", "N/A")
        print(f"  [Deploy] {rid}: {status} — {reason}")
        deploy_logs.append(f"{rid}: {status} - {reason}")
        if status == "CREATE_FAILED":
            failed_resources.append({"logical_name": rid, "status_reason": reason})
        elif status == "CREATE_COMPLETE":
            completed_resources.append(rid)


# ---------------------------------------------------------------------------
# Custom resource type helper (CloudFormation)
# ---------------------------------------------------------------------------

def _get_resource_type(cfn_client, stack_id: str, logical_id: str) -> str:
    try:
        resp = cfn_client.describe_stack_resource(
            StackName=stack_id, LogicalResourceId=logical_id
        )
        return resp["StackResourceDetail"].get("ResourceType", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HCL provider block stripper (Terraform)
# ---------------------------------------------------------------------------

def _strip_provider_blocks(hcl: str) -> str:
    """
    Remove all top-level provider "aws" { ... } blocks from an HCL string.

    Uses a brace-depth counter to handle multi-line blocks correctly.
    This ensures the LLM-generated provider block never collides with the
    harness-injected LocalStack provider override.
    """
    lines = hcl.splitlines(keepends=True)
    result = []
    depth = 0
    skipping = False

    for line in lines:
        stripped = line.strip()

        if not skipping and re.match(r'^provider\s+"aws"\s*\{', stripped):
            skipping = True
            depth = stripped.count("{") - stripped.count("}")
            continue  # drop opening line

        if skipping:
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0:
                skipping = False  # block closed — resume collecting
            continue  # drop all lines inside the block

        result.append(line)

    return "".join(result)


# ---------------------------------------------------------------------------
# Terraform deploy via `terraform apply` against LocalStack
# ---------------------------------------------------------------------------

def _validate_terraform_deployment(
    template: str,
    deploy_config: DeployConfig,
    start_time: float,
) -> DeployValidationResult:
    """
    Deploy a Terraform HCL template against the configured target.

    Strategy:
      - Strip any provider "aws" block from the LLM-generated template to
        prevent duplicate-provider errors when the harness injects its own
        provider override block (which carries LocalStack endpoint overrides).
      - Write the sanitised template to a temp dir as main.tf.
      - Inject a provider block that routes to LocalStack (or real AWS) via
        environment variables consumed by the AWS provider.
      - Run: terraform init -backend=false && terraform apply -auto-approve
      - Parse stdout/stderr for resource-level error lines.
      - Always run `terraform destroy -auto-approve` to clean up, mirroring
        the CloudFormation path's stack deletion.

    The stdout format for terraform apply errors is:
        Error: <summary>\n\n  on main.tf line N, in resource "type" "name":\n  <detail>
    """
    target_name = deploy_config.target.value
    deploy_logs: list[str] = []

    # Build provider env: LocalStack injects endpoint overrides;
    # real AWS relies on profile/region from deploy_config.
    env_overrides: dict[str, str] = {}
    if deploy_config.target == DeployTarget.LOCALSTACK:
        ls_ep = deploy_config.localstack_endpoint  # e.g. http://localhost:4566
        env_overrides = {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        # Build a provider override block that uses localstack-specific endpoints
        provider_override = f"""
provider "aws" {{
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  endpoints {{
    apigateway     = "{ls_ep}"
    cloudformation = "{ls_ep}"
    cloudwatch     = "{ls_ep}"
    dynamodb       = "{ls_ep}"
    ec2            = "{ls_ep}"
    es             = "{ls_ep}"
    firehose       = "{ls_ep}"
    iam            = "{ls_ep}"
    kinesis        = "{ls_ep}"
    lambda         = "{ls_ep}"
    route53        = "{ls_ep}"
    redshift       = "{ls_ep}"
    s3             = "{ls_ep}"
    secretsmanager = "{ls_ep}"
    ses            = "{ls_ep}"
    sns            = "{ls_ep}"
    sqs            = "{ls_ep}"
    ssm            = "{ls_ep}"
    stepfunctions  = "{ls_ep}"
    sts            = "{ls_ep}"
  }}
}}
"""
    else:
        # Real AWS: rely on environment / instance profile
        provider_override = ""

    import os
    import copy
    run_env = copy.copy(os.environ)
    run_env.update(env_overrides)

    with tempfile.TemporaryDirectory() as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"

        # Strip any provider "aws" block from the LLM output before combining
        # with the harness-injected override to prevent duplicate-provider errors.
        if provider_override:
            sanitised_template = _strip_provider_blocks(template)
            combined = provider_override + "\n" + sanitised_template
        else:
            combined = template

        tf_path.write_text(combined, encoding="utf-8")

        # ----------------------------------------------------------------
        # terraform init
        # ----------------------------------------------------------------
        print(f"[Deploy] Running terraform init in {tmpdir}...")
        init_result = subprocess.run(
            ["terraform", "init", "-backend=false", "-input=false", "-no-color"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120, env=run_env,
        )
        if init_result.returncode != 0:
            err = (init_result.stderr or init_result.stdout).strip()
            deploy_logs.append(f"terraform init failed: {err}")
            return DeployValidationResult(
                target=target_name, passed=False, stack_id=None,
                completed_resources=[], failed_resources=[{"logical_name": "init", "status_reason": err}],
                error_message=f"init: {err}",
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        # ----------------------------------------------------------------
        # terraform apply
        # ----------------------------------------------------------------
        timeout = deploy_config.stack_creation_timeout
        print(f"[Deploy] Running terraform apply (timeout={timeout}s)...")
        try:
            apply_result = subprocess.run(
                ["terraform", "apply", "-auto-approve", "-input=false", "-no-color"],
                cwd=tmpdir, capture_output=True, text=True, timeout=timeout, env=run_env,
            )
        except subprocess.TimeoutExpired:
            timeout_msg = f"terraform apply timed out after {timeout}s"
            deploy_logs.append(timeout_msg)
            return DeployValidationResult(
                target=target_name, passed=False, stack_id=None,
                completed_resources=[], failed_resources=[{"logical_name": "apply", "status_reason": timeout_msg}],
                error_message=timeout_msg,
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        apply_output = (apply_result.stdout or "") + (apply_result.stderr or "")
        deploy_logs.extend(apply_output.splitlines())

        if apply_result.returncode == 0:
            # Parse completed resources from apply output lines like:
            #   aws_vpc.main: Creation complete after 1s [id=vpc-xxx]
            completed: list[str] = re.findall(
                r'^([\w.]+):\s+Creation complete', apply_output, re.MULTILINE
            )
            print(f"[Deploy] ✅ terraform apply succeeded ({len(completed)} resources created)")

            # ----------------------------------------------------------------
            # terraform destroy (cleanup)
            # ----------------------------------------------------------------
            print("[Deploy] Cleaning up with terraform destroy...")
            subprocess.run(
                ["terraform", "destroy", "-auto-approve", "-input=false", "-no-color"],
                cwd=tmpdir, capture_output=True, text=True, timeout=timeout, env=run_env,
            )

            return DeployValidationResult(
                target=target_name, passed=True, stack_id=None,
                completed_resources=completed, failed_resources=[],
                error_message=None,
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        # ----------------------------------------------------------------
        # Parse failures from apply output
        # Error block pattern:
        #   Error: <summary>
        #     on main.tf line N, in resource "type" "name":
        #       N: <hcl line>
        #   <detail>
        # ----------------------------------------------------------------
        failed_resources: list[dict] = []
        error_blocks = re.split(r'(?m)^\u2502?\s*Error:', apply_output)
        resource_re = re.compile(
            r'on \S+\.tf line (\d+), in resource "([^"]+)"\s+"([^"]+)"'
        )
        for block in error_blocks[1:]:  # skip text before first Error:
            lines = block.strip().splitlines()
            summary = lines[0].strip() if lines else "unknown error"
            resource_addr = "apply"
            for line in lines[1:6]:  # look for resource ref near the top
                m = resource_re.search(line)
                if m:
                    resource_addr = f"{m.group(2)}.{m.group(3)}"
                    break
            # Find detail lines (non-empty, non-header lines after resource ref)
            detail_lines = [
                l.strip() for l in lines
                if l.strip()
                and not l.strip().startswith("on ")
                and l.strip() != summary
                and not re.match(r'^\d+:', l.strip())
            ]
            detail = " ".join(detail_lines[:3])  # keep first 3 detail lines
            status_reason = f"{summary}" + (f" — {detail}" if detail else "")
            failed_resources.append({"logical_name": resource_addr, "status_reason": status_reason})

        if not failed_resources:
            # Fallback: surface raw stderr
            raw_err = (apply_result.stderr or apply_result.stdout or "unknown apply error").strip()
            failed_resources = [{"logical_name": "apply", "status_reason": raw_err[:500]}]

        error_msg = _format_failed_resources(failed_resources)
        print(f"[Deploy] ❌ terraform apply failed: {error_msg}")

        # Always destroy on failure to clean up partial state
        subprocess.run(
            ["terraform", "destroy", "-auto-approve", "-input=false", "-no-color"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120, env=run_env,
        )

        return DeployValidationResult(
            target=target_name, passed=False, stack_id=None,
            completed_resources=[], failed_resources=failed_resources,
            error_message=error_msg,
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )


# ---------------------------------------------------------------------------
# Main deploy validator (public API)
# ---------------------------------------------------------------------------

def validate_deployment(
    template: str,
    deploy_config: DeployConfig,
    iac_type: str = "cloudformation",
) -> DeployValidationResult:
    """
    Stage: Attempt to deploy the IaC template to the configured target.

    Branches on iac_type:
      - "terraform":      runs terraform apply via _validate_terraform_deployment()
      - "cloudformation": original boto3/CloudFormation path (unchanged)

    The Terraform path:
      - Strips any LLM-generated provider "aws" block before writing main.tf
        to prevent duplicate-provider errors when the harness injects its own
        LocalStack provider override block.
      - Injects a LocalStack provider override block when target==localstack
      - Runs terraform init + terraform apply + terraform destroy (cleanup)
      - Parses Error blocks from apply output into FailedResource entries
        using Terraform resource addresses (e.g. aws_vpc.main) as logical_name

    The CloudFormation path:
      - Greenfield reset (LocalStack HTTP reset + CFN stack sweep; AWS: stack sweep)
      - VPC quota pre-flight for real AWS
      - Creates CFN stack with CREATE_FAILED event polling
      - Deletes stack on success/failure
    """
    if deploy_config.target == DeployTarget.NONE:
        return DeployValidationResult(
            target="skipped",
            passed=True,
            stack_id=None,
            completed_resources=[],
            failed_resources=[],
            error_message=None,
            duration_seconds=0.0,
            deployment_logs=[],
        )

    start_time = time.time()

    # ------------------------------------------------------------------
    # Terraform deploy path
    # ------------------------------------------------------------------
    if iac_type == "terraform":
        # Reset LocalStack state before Terraform apply too
        _reset_target_state(deploy_config)
        return _validate_terraform_deployment(template, deploy_config, start_time)

    # ------------------------------------------------------------------
    # CloudFormation deploy path (original logic preserved exactly)
    # ------------------------------------------------------------------
    target_name = deploy_config.target.value
    deploy_logs: list[str] = []

    _reset_target_state(deploy_config)

    _delete_all_non_default_vpcs(deploy_config)
    vpc_error = _check_vpc_quota(deploy_config)
    if vpc_error:
        deploy_logs.append(vpc_error)
        failed = [{"logical_name": "VPC", "status_reason": vpc_error}]
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=[],
            failed_resources=failed,
            error_message=_format_failed_resources(failed),
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )

    cfn_client = _build_cfn_client(deploy_config)
    stack_name = f"iac-god-eval-{uuid.uuid4().hex[:8]}"

    try:
        required_params, param_error = _required_parameter_keys(cfn_client, template)
        if param_error:
            deploy_logs.append(param_error)
            return DeployValidationResult(
                target=target_name, passed=False, stack_id=None,
                completed_resources=[],
                failed_resources=[{"logical_name": "template", "status_reason": param_error}],
                error_message=f"template: {param_error}",
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        if required_params:
            failed = [
                {"logical_name": name, "status_reason": "Required parameter has no Default value"}
                for name in required_params
            ]
            error_msg = _format_failed_resources(failed)
            deploy_logs.append(error_msg)
            return DeployValidationResult(
                target=target_name, passed=False, stack_id=None,
                completed_resources=[], failed_resources=failed,
                error_message=error_msg,
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        print(f"[Deploy] Creating stack '{stack_name}' on {target_name}...")
        deploy_logs.append(f"Creating stack '{stack_name}' on {target_name}")
        create_response = cfn_client.create_stack(
            StackName=stack_name,
            TemplateBody=template,
            OnFailure="DELETE",
            Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
        )
        stack_id = create_response["StackId"]

        seen_events: set[str] = set()
        last_timestamp = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        )
        failed_resources: list[dict] = []
        completed_resources: list[str] = []
        deadline = time.time() + deploy_config.stack_creation_timeout

        STALL_TIMEOUT = 15 * 60
        last_event_time = time.time()
        last_active_resource: str | None = None
        prev_seen_count = 0

        while time.time() < deadline:
            _drain_stack_events(
                cfn_client, stack_id, seen_events, last_timestamp,
                failed_resources, completed_resources, deploy_logs,
            )

            if len(seen_events) > prev_seen_count:
                last_event_time = time.time()
                prev_seen_count = len(seen_events)
                if deploy_logs:
                    last_log = deploy_logs[-1]
                    colon_idx = last_log.find(":")
                    if colon_idx > 0:
                        last_active_resource = last_log[:colon_idx].strip()

            stack = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]
            stack_status = stack["StackStatus"]

            if stack_status == "CREATE_COMPLETE":
                print(
                    f"[Deploy] ✅ Stack deployed successfully "
                    f"({len(completed_resources)} resources)"
                )
                cfn_client.delete_stack(StackName=stack_id)
                if deploy_config.target == DeployTarget.AWS:
                    _wait_for_stack_deletion(
                        cfn_client, stack_id, stack_name,
                        deploy_config.stack_deletion_timeout,
                    )
                return DeployValidationResult(
                    target=target_name, passed=True, stack_id=stack_id,
                    completed_resources=completed_resources, failed_resources=[],
                    error_message=None,
                    duration_seconds=round(time.time() - start_time, 2),
                    deployment_logs=deploy_logs,
                )

            if stack_status in (
                "CREATE_FAILED", "ROLLBACK_COMPLETE",
                "ROLLBACK_FAILED", "DELETE_COMPLETE",
            ):
                time.sleep(1)
                _drain_stack_events(
                    cfn_client, stack_id, seen_events, last_timestamp,
                    failed_resources, completed_resources, deploy_logs,
                )

                error_msg = _format_failed_resources(failed_resources)
                if not failed_resources:
                    error_msg = f"Stack entered terminal status: {stack_status}"

                print(f"[Deploy] ❌ Deployment failed: {error_msg}")
                _wait_for_stack_deletion(
                    cfn_client, stack_id, stack_name,
                    deploy_config.stack_deletion_timeout,
                )
                return DeployValidationResult(
                    target=target_name, passed=False, stack_id=stack_id,
                    completed_resources=completed_resources, failed_resources=failed_resources,
                    error_message=error_msg,
                    duration_seconds=round(time.time() - start_time, 2),
                    deployment_logs=deploy_logs,
                )

            stall_elapsed = time.time() - last_event_time
            if (
                last_active_resource
                and stall_elapsed > STALL_TIMEOUT
                and stack_status == "CREATE_IN_PROGRESS"
            ):
                resource_type = _get_resource_type(cfn_client, stack_id, last_active_resource)
                is_custom = bool(
                    re.match(r"^Custom::|^AWS::CloudFormation::CustomResource$", resource_type)
                )
                stall_reason = (
                    f"Stalled in CREATE_IN_PROGRESS for >{int(stall_elapsed)}s"
                    + (
                        " — Lambda-backed custom resource likely failed to send cfn-response "
                        "(add ServiceTimeout, ensure Lambda can reach the CloudFormation "
                        "response URL, or replace with a native CloudFormation resource)"
                        if is_custom
                        else " — resource has not progressed; possible dependency or quota issue"
                    )
                )
                stall_failed = [{"logical_name": last_active_resource, "status_reason": stall_reason}]
                stall_msg = _format_failed_resources(stall_failed)
                deploy_logs.append(f"STALL_DETECTED: {stall_msg}")
                print(f"[Deploy] ⚠️  Stall detected: {stall_msg}")

                try:
                    cfn_client.delete_stack(StackName=stack_id)
                    print(f"[Deploy] Cancellation requested for stalled stack '{stack_name}'")
                    if deploy_config.target == DeployTarget.AWS:
                        _wait_for_stack_deletion(
                            cfn_client, stack_id, stack_name,
                            deploy_config.stack_deletion_timeout,
                        )
                except Exception as e:
                    print(f"[Deploy] Could not cancel stalled stack: {e}")

                return DeployValidationResult(
                    target=target_name, passed=False, stack_id=stack_id,
                    completed_resources=completed_resources, failed_resources=stall_failed,
                    error_message=stall_msg,
                    duration_seconds=round(time.time() - start_time, 2),
                    deployment_logs=deploy_logs,
                )

            time.sleep(2)

        stall_elapsed = round(time.time() - last_event_time)
        stall_context = (
            f" | Last active resource: {last_active_resource} "
            f"(no new events for {stall_elapsed}s)"
            if last_active_resource else ""
        )
        timeout_msg = (
            f"Stack creation timed out after {deploy_config.stack_creation_timeout}s "
            f"({round(time.time() - start_time, 2)}s elapsed)" + stall_context
        )
        deploy_logs.append(timeout_msg)

        timeout_failed: list[dict] = list(failed_resources)
        if last_active_resource and not any(
            r["logical_name"] == last_active_resource for r in timeout_failed
        ):
            resource_type = _get_resource_type(cfn_client, stack_id, last_active_resource)
            is_custom = bool(
                re.match(r"^Custom::|^AWS::CloudFormation::CustomResource$", resource_type)
            )
            timeout_reason = (
                f"Stack creation timed out — resource stalled in CREATE_IN_PROGRESS "
                f"for {stall_elapsed}s"
                + (" (custom resource — add ServiceTimeout or replace with native resource)" if is_custom else "")
            )
            timeout_failed.append({"logical_name": last_active_resource, "status_reason": timeout_reason})

        print(f"[Deploy] ❌ {timeout_msg}")

        try:
            cfn_client.delete_stack(StackName=stack_id)
            print(f"[Deploy] Cancellation requested for timed-out stack '{stack_name}'")
            if deploy_config.target == DeployTarget.AWS:
                _wait_for_stack_deletion(
                    cfn_client, stack_id, stack_name,
                    deploy_config.stack_deletion_timeout,
                )
        except Exception as e:
            print(f"[Deploy] Could not cancel timed-out stack: {e}")

        return DeployValidationResult(
            target=target_name, passed=False, stack_id=stack_id,
            completed_resources=completed_resources, failed_resources=timeout_failed,
            error_message=_format_failed_resources(timeout_failed) if timeout_failed else timeout_msg,
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )

    except ClientError as e:
        msg = str(e)
        deploy_logs.append(msg)
        return DeployValidationResult(
            target=target_name, passed=False, stack_id=None,
            completed_resources=[],
            failed_resources=[{"logical_name": "stack", "status_reason": msg}],
            error_message=f"stack: {msg}",
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )
    except Exception as e:
        msg = f"Unexpected error: {e}"
        deploy_logs.append(msg)
        return DeployValidationResult(
            target=target_name, passed=False, stack_id=None,
            completed_resources=[],
            failed_resources=[{"logical_name": "stack", "status_reason": msg}],
            error_message=f"stack: {msg}",
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )
