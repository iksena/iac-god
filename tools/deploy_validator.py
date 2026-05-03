# tools/deploy_validator.py
import time
import uuid
import boto3
import datetime
import requests
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
        <LogicalResourceId>: <status_reason>

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
        CloudFormation API before proceeding.  This mirrors the guidance in
        the LocalStack CloudFormation docs (delete-stack is fully supported).
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
    carry the iac-god-eval- prefix.  Safe to call against both LocalStack and
    AWS; for AWS this is also used by _reset_aws_state().

    We query all non-deleted statuses so we catch stacks stuck in
    ROLLBACK_COMPLETE or CREATE_FAILED that would otherwise block a new
    create_stack call with the same name.
    """
    cfn_client = _build_cfn_client(deploy_config)
    stack_prefix = "iac-god-eval-"

    # Statuses that represent a stack that still physically exists
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
        targets: list[tuple[str, str]] = []  # (stack_id, stack_name)

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
                    cfn_client,
                    stack_id,
                    stack_name,
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
    """
    Best-effort cleanup of prior IaCGOD evaluation stacks in real AWS.
    Reuses _delete_surviving_eval_stacks so the deletion logic is not
    duplicated between LocalStack and AWS paths.
    """
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
    """
    Block until the stack reaches DELETE_COMPLETE or the timeout elapses.
    Critical for LocalStack where resource cleanup is asynchronous.
    """
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


# ---------------------------------------------------------------------------
# Parameter validation helper
# ---------------------------------------------------------------------------

def _required_parameter_keys(cfn_client, template: str) -> tuple[list[str], str | None]:
    """
    Return parameter keys that require explicit values (no Default).

    We intentionally do NOT auto-fill values because the benchmark goal is
    to test whether the LLM produced a self-deployable template.
    """
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
# Event poller helper
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
    """
    Fetch and process all unseen CloudFormation stack events, updating
    failed_resources, completed_resources, and deploy_logs in-place.
    """
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
# Main deploy validator
# ---------------------------------------------------------------------------

def validate_deployment(
    template: str,
    deploy_config: DeployConfig,
) -> DeployValidationResult:
    """
    Stage 5: Attempt to deploy the CloudFormation template to the target.

    Greenfield guarantee:
      - LocalStack: HTTP state reset + explicit deletion of any surviving
        iac-god-eval-* stacks via the CloudFormation API.
      - AWS: deletion of any surviving iac-god-eval-* stacks.

    Error messages always identify the responsible resource(s) by logical ID
    so the remediator prompt contains actionable context.
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
    target_name = deploy_config.target.value
    deploy_logs: list[str] = []

    # Greenfield reset (LocalStack: HTTP reset + stack sweep; AWS: stack sweep)
    _reset_target_state(deploy_config)

    cfn_client = _build_cfn_client(deploy_config)
    stack_name = f"iac-god-eval-{uuid.uuid4().hex[:8]}"

    try:
        # ------------------------------------------------------------------
        # Parameter pre-check
        # ------------------------------------------------------------------
        required_params, param_error = _required_parameter_keys(cfn_client, template)
        if param_error:
            deploy_logs.append(param_error)
            return DeployValidationResult(
                target=target_name,
                passed=False,
                stack_id=None,
                completed_resources=[],
                failed_resources=[{"logical_name": "template", "status_reason": param_error}],
                error_message=f"template: {param_error}",
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        if required_params:
            failed = [
                {
                    "logical_name": name,
                    "status_reason": "Required parameter has no Default value",
                }
                for name in required_params
            ]
            error_msg = _format_failed_resources(failed)
            deploy_logs.append(error_msg)
            return DeployValidationResult(
                target=target_name,
                passed=False,
                stack_id=None,
                completed_resources=[],
                failed_resources=failed,
                error_message=error_msg,
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        # ------------------------------------------------------------------
        # Create stack
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Poll loop
        # ------------------------------------------------------------------
        while time.time() < deadline:
            _drain_stack_events(
                cfn_client, stack_id, seen_events, last_timestamp,
                failed_resources, completed_resources, deploy_logs,
            )

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
                    target=target_name,
                    passed=True,
                    stack_id=stack_id,
                    completed_resources=completed_resources,
                    failed_resources=[],
                    error_message=None,
                    duration_seconds=round(time.time() - start_time, 2),
                    deployment_logs=deploy_logs,
                )

            if stack_status in (
                "CREATE_FAILED", "ROLLBACK_COMPLETE",
                "ROLLBACK_FAILED", "DELETE_COMPLETE",
            ):
                # One final drain to capture any events that arrived between
                # the last poll and the terminal status check.
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
                    target=target_name,
                    passed=False,
                    stack_id=stack_id,
                    completed_resources=completed_resources,
                    failed_resources=failed_resources,
                    error_message=error_msg,
                    duration_seconds=round(time.time() - start_time, 2),
                    deployment_logs=deploy_logs,
                )

            time.sleep(2)

        # ------------------------------------------------------------------
        # Timeout
        # ------------------------------------------------------------------
        timeout_msg = (
            f"Stack creation timed out after {deploy_config.stack_creation_timeout}s"
        )
        deploy_logs.append(timeout_msg)
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=completed_resources,
            failed_resources=failed_resources,
            error_message=timeout_msg,
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )

    except ClientError as e:
        msg = str(e)
        deploy_logs.append(msg)
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
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
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=[],
            failed_resources=[{"logical_name": "stack", "status_reason": msg}],
            error_message=f"stack: {msg}",
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )
