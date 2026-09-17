"""Pre- and post-deployment cleanup for the benchmark harness's LocalStack/AWS
targets. Split out of tools/deploy_validator.py (which now holds only the
actual deploy-attempt logic) so "how do we tear things down" and "how do we
run a deployment" can be read, tested, and changed independently.

Two call sites, both from deploy_validator.validate_deployment():
  - _reset_target_state(deploy_config): per-ITERATION pre-flight reset, run
    before every single deployment attempt.
  - cleanup_scenario_resources(deploy_config): per-SCENARIO final sweep, run
    once a scenario has exhausted all its iterations (see benchmark.py).

Everything here for the AWS target is deliberately scoped one of two ways,
matching the two different risks it addresses:
  - Resources tagged ManagedBy=iac-god-eval (_delete_orphaned_eval_resources):
    strictly scoped to what THIS harness itself created and tagged, safe to
    run unattended in an account with anything else in it.
  - Every non-default VPC and its dependency chain (_delete_all_non_default_vpcs):
    deliberately NOT tag-scoped, because VPC quota is consumed by any
    non-default VPC regardless of tags -- this assumes a dedicated research
    AWS account with nothing else expected to live in it (see
    ../nuke-config.yml), same assumption the rest of this project makes.
"""

import boto3
from botocore.exceptions import ClientError
from config import DeployConfig, DeployTarget
import requests
import time

from tools.cfn_utils import build_cfn_client, wait_for_stack_deletion
from scripts.nuke_vpc_dependencies import (
    Actions as _NukeActions,
    list_target_vpcs,
    nuke_vpc_network,
    nuke_elb,
    wait_for_elbv2_deleted,
    nuke_nat_gateways,
    wait_for_nat_gateways_deleted,
    nuke_rds,
    nuke_rds_subnet_groups,
    wait_for_rds_gone,
    nuke_elasticache,
    nuke_elasticache_subnet_groups,
    nuke_efs_mount_targets,
    nuke_efs_file_systems,
    nuke_elastic_ips,
    nuke_vpc_endpoints,
    nuke_peering_connections,
    nuke_transit_gateway_attachments,
    nuke_vpn,
    nuke_egress_only_igw,
    nuke_one_kms_key,
)

# Applied to every CloudFormation stack this harness creates (and, by CFN's
# stack-level tag propagation, to every resource within it that supports
# tagging). Stack-name-prefix matching (see _delete_surviving_eval_stacks)
# only finds *stacks*; an LLM-chosen resource name (e.g. an S3 bucket) can be
# anything and won't carry that prefix. A resource that outlives its stack's
# deletion (DeletionPolicy: Retain, or a resource CloudFormation can't
# auto-delete) still carries this tag, which is how
# _delete_orphaned_eval_resources finds and removes it regardless of name --
# critical for S3 specifically, since bucket names are globally unique and a
# single orphan permanently blocks every future run of that same scenario.
EVAL_TAG_KEY = "ManagedBy"
EVAL_TAG_VALUE = "iac-god-eval"


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
    cfn_client = build_cfn_client(deploy_config)
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
                wait_for_stack_deletion(
                    cfn_client, stack_id, stack_name,
                    deploy_config.stack_deletion_timeout,
                )
                print(f"  [Deploy] '{stack_name}' deleted ✓")
            except Exception as e:
                print(f"  [Deploy] Warning: could not delete '{stack_name}': {e}")

    except Exception as e:
        print(f"[Deploy] Stack sweep error: {e}")


def _empty_and_delete_bucket(s3_client, bucket_name: str) -> None:
    """Empty every object version + delete marker, then delete the bucket
    itself.

    delete_bucket refuses a non-empty bucket, and a versioning-enabled
    bucket's real contents include every historical version and delete
    marker, not just the current keys — list_object_versions (not
    list_objects_v2) is required to actually find and remove all of them.
    """
    paginator = s3_client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket_name):
        to_delete = [
            {"Key": v["Key"], "VersionId": v["VersionId"]}
            for v in page.get("Versions", []) + page.get("DeleteMarkers", [])
        ]
        for i in range(0, len(to_delete), 1000):  # delete_objects caps at 1000/call
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={"Objects": to_delete[i:i + 1000]},
            )

    s3_client.delete_bucket(Bucket=bucket_name)


def _delete_orphaned_ecs_cluster(ecs_client, cluster_arn: str) -> None:
    """Drain and delete an ECS cluster.

    delete_cluster refuses a cluster with active services or tasks, so those
    have to be torn down first — mirrors what an LLM-generated ECS stack's
    own rollback would normally do, since this cluster outlived its stack.
    """
    services = ecs_client.list_services(cluster=cluster_arn).get("serviceArns", [])
    for service_arn in services:
        ecs_client.update_service(cluster=cluster_arn, service=service_arn, desiredCount=0)
        ecs_client.delete_service(cluster=cluster_arn, service=service_arn, force=True)

    tasks = ecs_client.list_tasks(cluster=cluster_arn).get("taskArns", [])
    for task_arn in tasks:
        ecs_client.stop_task(cluster=cluster_arn, task=task_arn)

    ecs_client.delete_cluster(cluster=cluster_arn)


def _delete_orphaned_eval_resources(deploy_config: DeployConfig):
    """Find and delete resources tagged EVAL_TAG_KEY=EVAL_TAG_VALUE (see
    deploy_validator.validate_deployment's create_stack call) that survived
    their owning stack's deletion.

    Stack-name-prefix matching (_delete_surviving_eval_stacks, which always
    runs first) only finds *stacks*. A resource an LLM's template caused to
    outlive its stack — DeletionPolicy: Retain, or a resource CloudFormation
    couldn't auto-delete (a non-empty S3 bucket, an ECS cluster with an
    orphaned service) — can have any name at all, so only the tag reliably
    identifies it as ours. Tags live on the resource itself, so they persist
    even after the owning stack is gone.

    Every resource type below is scoped strictly to this tag -- safe to run
    unattended regardless of what else lives in the account/region.
    """
    session = boto3.Session(profile_name=deploy_config.aws_profile)
    tagging_client = session.client(
        "resourcegroupstaggingapi", region_name=deploy_config.aws_region
    )

    try:
        paginator = tagging_client.get_paginator("get_resources")
        mappings = []
        for page in paginator.paginate(
            TagFilters=[{"Key": EVAL_TAG_KEY, "Values": [EVAL_TAG_VALUE]}],
        ):
            mappings.extend(page.get("ResourceTagMappingList", []))
    except Exception as e:
        print(f"[Deploy] Orphan resource sweep error (non-fatal): {e}")
        return

    if not mappings:
        return

    s3_buckets: list[str] = []
    cognito_pool_ids: list[str] = []
    ecs_cluster_arns: list[str] = []
    log_group_names: list[str] = []
    kms_key_ids: list[str] = []
    elasticache_cluster_ids: list[str] = []
    elasticache_replgroup_ids: list[str] = []
    elbv2_arns: list[str] = []
    secret_arns: list[str] = []
    webacl_arns: list[str] = []
    codebuild_project_names: list[str] = []
    other: list[str] = []
    for mapping in mappings:
        arn = mapping.get("ResourceARN", "")
        # S3 bucket ARNs: arn:aws:s3:::bucket-name — no account/region
        # segment, and no further "/" (an object-level ARN would have one).
        s3_prefix = "arn:aws:s3:::"
        if arn.startswith(s3_prefix) and "/" not in arn[len(s3_prefix):]:
            s3_buckets.append(arn[len(s3_prefix):])
        elif ":cognito-idp:" in arn and "userpool/" in arn:
            cognito_pool_ids.append(arn.rsplit("/", 1)[1])
        elif ":ecs:" in arn and ":cluster/" in arn:
            ecs_cluster_arns.append(arn)
        elif ":logs:" in arn and ":log-group:" in arn:
            log_group_names.append(arn.split(":log-group:", 1)[1])
        elif ":kms:" in arn and ":key/" in arn:
            kms_key_ids.append(arn.rsplit("/", 1)[1])
        elif ":elasticache:" in arn and ":cluster:" in arn:
            elasticache_cluster_ids.append(arn.rsplit(":", 1)[1])
        elif ":elasticache:" in arn and ":replicationgroup:" in arn:
            elasticache_replgroup_ids.append(arn.rsplit(":", 1)[1])
        elif ":elasticloadbalancing:" in arn and ":loadbalancer/" in arn:
            elbv2_arns.append(arn)
        elif ":secretsmanager:" in arn and ":secret:" in arn:
            secret_arns.append(arn)
        elif ":wafv2:" in arn and "/webacl/" in arn:
            webacl_arns.append(arn)
        elif ":codebuild:" in arn and ":project/" in arn:
            codebuild_project_names.append(arn.rsplit("/", 1)[1])
        else:
            other.append(arn)

    if other:
        preview = ", ".join(other[:10]) + (" ..." if len(other) > 10 else "")
        print(
            f"[Deploy] ⚠️  {len(other)} other tagged resource(s) found with no "
            f"owning stack (not auto-cleaned, needs a type-specific delete): {preview}"
        )

    if s3_buckets:
        print(f"[Deploy] {len(s3_buckets)} orphaned eval S3 bucket(s) found — emptying and deleting...")
        s3_client = session.client("s3", region_name=deploy_config.aws_region)
        for bucket_name in s3_buckets:
            try:
                _empty_and_delete_bucket(s3_client, bucket_name)
                print(f"  [Deploy] Deleted orphaned bucket '{bucket_name}' ✓")
            except Exception as e:
                print(f"  [Deploy] Warning: could not delete orphaned bucket '{bucket_name}': {e}")

    if cognito_pool_ids:
        print(f"[Deploy] {len(cognito_pool_ids)} orphaned eval Cognito user pool(s) found — deleting...")
        cognito_client = session.client("cognito-idp", region_name=deploy_config.aws_region)
        for pool_id in cognito_pool_ids:
            try:
                cognito_client.delete_user_pool(UserPoolId=pool_id)
                print(f"  [Deploy] Deleted orphaned user pool '{pool_id}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned user pool '{pool_id}': {e}")

    if ecs_cluster_arns:
        print(f"[Deploy] {len(ecs_cluster_arns)} orphaned eval ECS cluster(s) found — draining and deleting...")
        ecs_client = session.client("ecs", region_name=deploy_config.aws_region)
        for cluster_arn in ecs_cluster_arns:
            try:
                _delete_orphaned_ecs_cluster(ecs_client, cluster_arn)
                print(f"  [Deploy] Deleted orphaned ECS cluster '{cluster_arn}' ✓")
            except ClientError as e:
                print(f"  [Deploy] Warning: could not delete orphaned ECS cluster '{cluster_arn}': {e}")

    if log_group_names:
        print(f"[Deploy] {len(log_group_names)} orphaned eval CloudWatch log group(s) found — deleting...")
        logs_client = session.client("logs", region_name=deploy_config.aws_region)
        for log_group_name in log_group_names:
            try:
                logs_client.delete_log_group(logGroupName=log_group_name)
                print(f"  [Deploy] Deleted orphaned log group '{log_group_name}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned log group '{log_group_name}': {e}")

    if elbv2_arns:
        print(f"[Deploy] {len(elbv2_arns)} orphaned eval load balancer(s) found — clearing deletion "
              f"protection and deleting...")
        elbv2_client = session.client("elbv2", region_name=deploy_config.aws_region)
        for lb_arn in elbv2_arns:
            try:
                attrs = elbv2_client.describe_load_balancer_attributes(LoadBalancerArn=lb_arn).get("Attributes", [])
                if any(a.get("Key") == "deletion_protection.enabled" and a.get("Value") == "true" for a in attrs):
                    elbv2_client.modify_load_balancer_attributes(
                        LoadBalancerArn=lb_arn,
                        Attributes=[{"Key": "deletion_protection.enabled", "Value": "false"}],
                    )
                elbv2_client.delete_load_balancer(LoadBalancerArn=lb_arn)
                print(f"  [Deploy] Deleted orphaned load balancer '{lb_arn}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "LoadBalancerNotFoundException":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned load balancer '{lb_arn}': {e}")

    if elasticache_cluster_ids or elasticache_replgroup_ids:
        n = len(elasticache_cluster_ids) + len(elasticache_replgroup_ids)
        print(f"[Deploy] {n} orphaned eval ElastiCache resource(s) found — deleting...")
        ec_client = session.client("elasticache", region_name=deploy_config.aws_region)
        for repl_id in elasticache_replgroup_ids:
            try:
                ec_client.delete_replication_group(ReplicationGroupId=repl_id)
                print(f"  [Deploy] Deleted orphaned ElastiCache replication group '{repl_id}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ReplicationGroupNotFoundFault":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned replication group '{repl_id}': {e}")
        for cluster_id in elasticache_cluster_ids:
            try:
                ec_client.delete_cache_cluster(CacheClusterId=cluster_id)
                print(f"  [Deploy] Deleted orphaned ElastiCache cluster '{cluster_id}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "CacheClusterNotFoundFault":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned cache cluster '{cluster_id}': {e}")

    if secret_arns:
        print(f"[Deploy] {len(secret_arns)} orphaned eval Secrets Manager secret(s) found — deleting...")
        sm_client = session.client("secretsmanager", region_name=deploy_config.aws_region)
        for secret_arn in secret_arns:
            try:
                sm_client.delete_secret(SecretId=secret_arn, ForceDeleteWithoutRecovery=True)
                print(f"  [Deploy] Deleted orphaned secret '{secret_arn}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned secret '{secret_arn}': {e}")

    if codebuild_project_names:
        print(f"[Deploy] {len(codebuild_project_names)} orphaned eval CodeBuild project(s) found — deleting...")
        cb_client = session.client("codebuild", region_name=deploy_config.aws_region)
        for name in codebuild_project_names:
            try:
                cb_client.delete_project(name=name)
                print(f"  [Deploy] Deleted orphaned CodeBuild project '{name}' ✓")
            except ClientError as e:
                print(f"  [Deploy] Warning: could not delete orphaned CodeBuild project '{name}': {e}")

    if webacl_arns:
        print(f"[Deploy] {len(webacl_arns)} orphaned eval WAF web ACL(s) found — disassociating and deleting...")
        for webacl_arn in webacl_arns:
            try:
                # arn:aws:wafv2:<region>:<account>:<scope>/webacl/<name>/<id>
                parts = webacl_arn.split(":")[-1].split("/")  # ['<scope>', 'webacl', '<name>', '<id>']
                scope = "CLOUDFRONT" if parts[0] == "global" else "REGIONAL"
                name, webacl_id = parts[2], parts[3]
                waf_region = "us-east-1" if scope == "CLOUDFRONT" else deploy_config.aws_region
                waf_client = session.client("wafv2", region_name=waf_region)
                for resource_arn in waf_client.list_resources_for_web_acl(
                    WebACLArn=webacl_arn, ResourceType="APPLICATION_LOAD_BALANCER"
                ).get("ResourceArns", []):
                    waf_client.disassociate_web_acl(ResourceArn=resource_arn)
                lock_token = waf_client.get_web_acl(Name=name, Scope=scope, Id=webacl_id)["LockToken"]
                waf_client.delete_web_acl(Name=name, Scope=scope, Id=webacl_id, LockToken=lock_token)
                print(f"  [Deploy] Deleted orphaned WAF web ACL '{name}' ✓")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "WAFNonexistentItemException":
                    continue
                print(f"  [Deploy] Warning: could not delete orphaned WAF web ACL '{webacl_arn}': {e}")

    if kms_key_ids:
        print(f"[Deploy] {len(kms_key_ids)} orphaned eval KMS key(s) found — scheduling deletion...")
        kms_client = session.client("kms", region_name=deploy_config.aws_region)
        try:
            caller_arn = session.client("sts", region_name=deploy_config.aws_region).get_caller_identity()["Arn"]
        except Exception:
            caller_arn = None
        nuke_actions = _NukeActions(dry_run=False)
        for key_id in kms_key_ids:
            nuke_one_kms_key(kms_client, key_id, nuke_actions, caller_arn)


# ---------------------------------------------------------------------------
# AWS reset helper
# ---------------------------------------------------------------------------

def _reset_aws_state(deploy_config: DeployConfig):
    print("[Deploy] AWS state reset: scanning for prior evaluation stacks...")
    _delete_surviving_eval_stacks(deploy_config)
    _delete_orphaned_eval_resources(deploy_config)
    # VPC/ALB/RDS/ElastiCache/EFS/NAT dependency-chain cleanup -- shared by
    # both CFN and Terraform (see _reset_target_state), since VPC quota
    # exhaustion isn't specific to either IaC language.
    _delete_all_non_default_vpcs(deploy_config)


# ---------------------------------------------------------------------------
# Unified reset dispatcher (per-ITERATION pre-flight — public API)
# ---------------------------------------------------------------------------

def reset_target_state(deploy_config: DeployConfig):
    """Run pre-deployment state reset for the configured target. Called by
    deploy_validator.validate_deployment() before every single deployment
    attempt (i.e. once per iteration)."""
    if deploy_config.target == DeployTarget.LOCALSTACK:
        _reset_localstack_state(deploy_config)
    elif deploy_config.target == DeployTarget.AWS:
        _reset_aws_state(deploy_config)


# ---------------------------------------------------------------------------
# Scenario-finished cleanup (per-SCENARIO final sweep — public API)
# ---------------------------------------------------------------------------

def cleanup_scenario_resources(deploy_config: DeployConfig) -> None:
    """Run once a scenario has exhausted all its iterations (passed or hit
    max_iterations) -- the benchmark harness's per-scenario boundary, as
    opposed to reset_target_state's per-iteration pre-flight reset.

    A no-op for LOCALSTACK/NONE targets (nothing to reset that a container
    restart doesn't already handle, and no real-money cost at stake).

    For AWS, this is deliberately *not* a second, different implementation:
    it just re-runs the same pre-flight reset (reset_target_state) that
    every iteration already goes through. That's the right thing to do here
    for two reasons a per-iteration-only reset can't cover on its own:
      1. Resources whose deletion the *last* iteration's cleanup kicked off
         asynchronously (an ALB, a NAT gateway, an ElastiCache cluster) may
         not have actually finished disappearing by the time that iteration
         ended -- running the reset again now, with nothing else queued
         behind it, gives those a second, unhurried pass instead of leaving
         them for whatever the *next scenario's* first iteration happens to
         trigger.
      2. If this is the last scenario in the whole benchmark run, there is
         no "next iteration" to ever trigger reset_target_state again --
         without this call, anything still standing here would simply be
         left running (and billing) until someone notices and runs
         scripts/nuke_vpc_dependencies.py or aws-nuke by hand.
    """
    if deploy_config.target != DeployTarget.AWS:
        return
    print("[Deploy] Scenario finished — running final AWS resource cleanup...")
    reset_target_state(deploy_config)


# ---------------------------------------------------------------------------
# VPC quota pre-flight
# ---------------------------------------------------------------------------

def _delete_all_non_default_vpcs(deploy_config: DeployConfig) -> None:
    """VPC quota pre-flight: tear down every non-default VPC and everything
    that can block its deletion, so a stray VPC an earlier iteration's
    template left behind never exhausts the account's (default: 5) VPC quota
    or blocks the next deployment attempt.

    Deliberately account/region-wide rather than tagged-resource-scoped like
    _delete_orphaned_eval_resources -- VPC quota is consumed by ANY
    non-default VPC regardless of tags, and this project's AWS eval targets
    are dedicated research accounts with nothing else expected to live here
    (see ../nuke-config.yml). Reuses scripts/nuke_vpc_dependencies.py's
    proven per-VPC teardown (ALB deletion-protection clear + wait, NAT
    gateway clear + wait, force-detached ENIs, security-group rule
    stripping, RDS/ElastiCache/EFS in the subnet, ...) instead of
    reimplementing it -- the old inline version here only handled
    IGW/subnet/route-table/security-group and left every one of those
    dependency-violation failure modes unaddressed.
    """
    if deploy_config.target != DeployTarget.AWS:
        return

    session = boto3.Session(profile_name=deploy_config.aws_profile)
    region = deploy_config.aws_region
    ec2 = session.client("ec2", region_name=region)
    actions = _NukeActions(dry_run=False)

    vpcs = list_target_vpcs(ec2, include_default=False, only_ids=None)
    if not vpcs:
        return

    print(f"[Deploy] VPC pre-flight: {len(vpcs)} non-default VPC(s) found — clearing dependencies...")

    def _safe(fn, description: str) -> None:
        try:
            fn()
        except Exception as e:
            print(f"[Deploy] VPC pre-flight: {description} failed unexpectedly: {e} (continuing)")

    elbv2_arns: list = []
    nat_ids: list = []
    _safe(lambda: elbv2_arns.extend(nuke_elb(session, region, actions)), "load balancer cleanup")
    _safe(lambda: nuke_rds(session, region, actions), "RDS cleanup")
    _safe(lambda: nuke_elasticache(session, region, actions), "ElastiCache cleanup")
    _safe(lambda: nuke_efs_mount_targets(session, region, actions), "EFS mount target cleanup")
    _safe(lambda: nat_ids.extend(nuke_nat_gateways(session, region, actions)), "NAT gateway cleanup")

    _safe(lambda: wait_for_elbv2_deleted(session.client("elbv2", region_name=region), elbv2_arns),
          "load balancer deletion wait")
    _safe(lambda: wait_for_nat_gateways_deleted(ec2, nat_ids), "NAT gateway deletion wait")
    _safe(lambda: wait_for_rds_gone(session.client("rds", region_name=region)), "RDS deletion wait")

    _safe(lambda: nuke_rds_subnet_groups(session, region, actions), "RDS subnet group cleanup")
    _safe(lambda: nuke_elasticache_subnet_groups(session, region, actions), "ElastiCache subnet group cleanup")
    _safe(lambda: nuke_efs_file_systems(session, region, actions), "EFS file system cleanup")
    _safe(lambda: nuke_elastic_ips(session, region, actions), "Elastic IP cleanup")
    _safe(lambda: nuke_vpc_endpoints(session, region, actions), "VPC endpoint cleanup")
    _safe(lambda: nuke_peering_connections(session, region, actions), "VPC peering cleanup")
    _safe(lambda: nuke_transit_gateway_attachments(session, region, actions), "Transit Gateway attachment cleanup")
    _safe(lambda: nuke_vpn(session, region, actions), "VPN gateway/connection cleanup")
    _safe(lambda: nuke_egress_only_igw(session, region, actions), "egress-only IGW cleanup")

    for vpc in vpcs:
        _safe(lambda v=vpc: nuke_vpc_network(ec2, v, actions, dry_run=False),
              f"VPC {vpc['VpcId']} network teardown")


def check_vpc_quota(deploy_config: DeployConfig) -> str | None:
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
