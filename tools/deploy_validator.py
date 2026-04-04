# tools/deploy_validator.py
import time
import uuid
import boto3
import datetime
import requests
from botocore.exceptions import ClientError
from config import DeployConfig, DeployTarget
from state import DeployValidationResult


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
            # LocalStack does not require real credentials
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
    else:
        # Real AWS — uses standard credential chain (profile, env vars, instance role)
        session = boto3.Session(profile_name=deploy_config.aws_profile)
        return session.client("cloudformation", region_name=deploy_config.aws_region)


def _reset_localstack_state(deploy_config: DeployConfig):
    """
    Hard-reset all LocalStack services before each deployment attempt.
    This is the greenfield reset — ensures each iteration starts from a clean slate.
    Adapted from IaCGen cloud_evaluation.py.
    """
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


def _wait_for_stack_deletion(cfn_client, stack_id: str, stack_name: str, timeout: int):
    """
    Block until the stack reaches DELETE_COMPLETE or timeout.
    Critical for LocalStack where resource cleanup is asynchronous.
    Adapted from IaCGen cloud_evaluation.py.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            stack = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]
            status = stack["StackStatus"]
            if status == "DELETE_COMPLETE":
                return
            if status in ["ROLLBACK_COMPLETE", "CREATE_FAILED"]:
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


def _required_parameter_keys(cfn_client, template: str) -> tuple[list[str], str | None]:
    """
    Return parameter keys that require explicit values (no Default).

    We intentionally do NOT auto-fill values because benchmark goal is to test
    whether the LLM produced a self-deployable template.
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


def validate_deployment(
    template: str,
    deploy_config: DeployConfig,
) -> DeployValidationResult:
    """
    Stage 5: Attempt to deploy the CloudFormation template to the target environment.

    Greenfield guarantee: LocalStack state is reset before every attempt so each
    iteration starts from a clean environment with no leftover resources.

    For AWS target, a unique stack name is used and the stack is always cleaned up
    after the test (success or failure) to avoid cost accumulation.
    """
    if deploy_config.target == DeployTarget.NONE:
        return DeployValidationResult(
            target="skipped",
            passed=True,     # Treated as vacuously passing — not a blocker
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

    # --- Greenfield reset for LocalStack ---
    if deploy_config.target == DeployTarget.LOCALSTACK:
        _reset_localstack_state(deploy_config)

    cfn_client = _build_cfn_client(deploy_config)
    stack_name = f"iac-god-eval-{uuid.uuid4().hex[:8]}"

    try:
        required_params, param_error = _required_parameter_keys(cfn_client, template)
        if param_error:
            deploy_logs.append(param_error)
            return DeployValidationResult(
                target=target_name,
                passed=False,
                stack_id=None,
                completed_resources=[],
                failed_resources=[{"resource": "template", "reason": param_error}],
                error_message=param_error,
                duration_seconds=round(time.time() - start_time, 2),
                deployment_logs=deploy_logs,
            )

        if required_params:
            message = (
                "Template is not self-deployable: missing required CloudFormation "
                f"parameters with no defaults: {', '.join(required_params)}"
            )
            deploy_logs.append(message)
            return DeployValidationResult(
                target=target_name,
                passed=False,
                stack_id=None,
                completed_resources=[],
                failed_resources=[
                    {
                        "resource": "parameters",
                        "reason": f"{name} must have a value",
                    }
                    for name in required_params
                ],
                error_message=message,
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
        last_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
        failed_resources: list[dict] = []
        completed_resources: list[str] = []
        deadline = time.time() + deploy_config.stack_creation_timeout

        while time.time() < deadline:
            # Poll stack events
            events = cfn_client.describe_stack_events(StackName=stack_id)["StackEvents"]
            new_events = [
                e for e in events
                if e["EventId"] not in seen_events and e["Timestamp"] > last_timestamp
            ]
            for event in sorted(new_events, key=lambda x: x["Timestamp"]):
                seen_events.add(event["EventId"])
                rid = event["LogicalResourceId"]
                status = event["ResourceStatus"]
                reason = event.get("ResourceStatusReason", "N/A")
                print(f"  [Deploy] {rid}: {status} — {reason}")
                deploy_logs.append(f"{rid}: {status} - {reason}")
                if status == "CREATE_FAILED":
                    failed_resources.append({"resource": rid, "reason": reason})
                elif status == "CREATE_COMPLETE":
                    completed_resources.append(rid)

            # Check terminal stack status
            stack = cfn_client.describe_stacks(StackName=stack_id)["Stacks"][0]
            stack_status = stack["StackStatus"]

            if stack_status == "CREATE_COMPLETE":
                print(f"[Deploy] ✅ Stack deployed successfully ({len(completed_resources)} resources)")
                cfn_client.delete_stack(StackName=stack_id)
                # For AWS, wait for deletion to avoid cost; for LocalStack, next iteration resets anyway
                if deploy_config.target == DeployTarget.AWS:
                    _wait_for_stack_deletion(cfn_client, stack_id, stack_name,
                                             deploy_config.stack_deletion_timeout)
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

            elif stack_status in ("CREATE_FAILED", "ROLLBACK_COMPLETE", "ROLLBACK_FAILED", "DELETE_COMPLETE"):
                error_msg = (
                    failed_resources[0]["reason"] if failed_resources
                    else f"Stack entered terminal status: {stack_status}"
                )
                print(f"[Deploy] ❌ Deployment failed: {error_msg}")
                _wait_for_stack_deletion(cfn_client, stack_id, stack_name,
                                         deploy_config.stack_deletion_timeout)
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

        # Timeout hit
        deploy_logs.append(
            f"Stack creation timed out after {deploy_config.stack_creation_timeout}s"
        )
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=completed_resources,
            failed_resources=failed_resources,
            error_message=f"Stack creation timed out after {deploy_config.stack_creation_timeout}s",
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )

    except ClientError as e:
        deploy_logs.append(str(e))
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=[],
            failed_resources=[{"resource": "stack", "reason": str(e)}],
            error_message=str(e),
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )
    except Exception as e:
        deploy_logs.append(f"Unexpected error: {str(e)}")
        return DeployValidationResult(
            target=target_name,
            passed=False,
            stack_id=None,
            completed_resources=[],
            failed_resources=[{"resource": "stack", "reason": str(e)}],
            error_message=f"Unexpected error: {str(e)}",
            duration_seconds=round(time.time() - start_time, 2),
            deployment_logs=deploy_logs,
        )