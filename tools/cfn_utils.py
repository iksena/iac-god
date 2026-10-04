"""Low-level CloudFormation helpers shared by tools/deploy_validator.py (the
actual deploy attempt) and tools/deploy_cleanup.py (pre/post-deployment
cleanup). Split out on its own so neither of those two modules has to import
from the other -- both import from here instead, avoiding a circular import
between "run a deployment" and "clean up after one."
"""

import time
import boto3
from botocore.exceptions import ClientError
from config import DeployConfig, DeployTarget


# ---------------------------------------------------------------------------
# CloudFormation client factory
# ---------------------------------------------------------------------------

def build_cfn_client(deploy_config: DeployConfig):
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

def format_failed_resources(failed_resources: list[dict]) -> str:
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
# Stack deletion waiter
# ---------------------------------------------------------------------------

def _failed_resource_ids(cfn_client, stack_id: str) -> list[str]:
    ids: list[str] = []
    try:
        for page in cfn_client.get_paginator("list_stack_resources").paginate(StackName=stack_id):
            ids += [r["LogicalResourceId"] for r in page["StackResourceSummaries"]
                    if r["ResourceStatus"] == "DELETE_FAILED"]
    except ClientError:
        pass
    return ids


def wait_for_stack_deletion(cfn_client, stack_id: str, stack_name: str, timeout: int):
    """Block until the stack is gone or *timeout* elapses.

    A stack that lands in DELETE_FAILED (typically a custom resource whose
    delete handler itself fails, or a non-empty bucket) will fail again
    identically if simply re-deleted, so each such failure is retried with the
    failed resources *retained* (RetainResources) -- the stack then actually
    disappears instead of looping DELETE_IN_PROGRESS -> DELETE_FAILED until the
    timeout. Anything retained that was real is tagged and picked up by the
    tag-scoped orphan sweep in deploy_cleanup.py. delete_stack is issued at
    most once per failure, not on every poll.
    """
    start = time.time()
    retain_attempts = 0
    reissued_plain_delete = False
    while time.time() - start < timeout:
        try:
            stack = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]
            status = stack["StackStatus"]
            if status == "DELETE_COMPLETE":
                return
            if status == "DELETE_FAILED" and retain_attempts < 3:
                retain_attempts += 1
                failed = _failed_resource_ids(cfn_client, stack_id)
                print(f"[Deploy] Stack '{stack_name}' DELETE_FAILED on {failed or 'unknown resource(s)'} "
                      f"-- retrying with those retained")
                try:
                    cfn_client.delete_stack(StackName=stack_id, RetainResources=failed)
                except ClientError as e:
                    print(f"[Deploy] Retain-delete of '{stack_name}' failed: {e}")
            elif status in ("ROLLBACK_COMPLETE", "CREATE_FAILED") and not reissued_plain_delete:
                reissued_plain_delete = True
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
            cfn_client.delete_stack(StackName=stack_id, RetainResources=_failed_resource_ids(cfn_client, stack_id))
            print(f"[Deploy] Issued force-delete for '{stack_name}'")
    except ClientError as e:
        if "does not exist" in str(e):
            return
        print(f"[Deploy] Could not force-delete '{stack_name}': {e}")
