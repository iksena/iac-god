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

def wait_for_stack_deletion(cfn_client, stack_id: str, stack_name: str, timeout: int):
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
