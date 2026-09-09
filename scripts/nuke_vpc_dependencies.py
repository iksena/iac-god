#!/usr/bin/env python3
"""
nuke_vpc_dependencies.py
=========================
Complement to `aws-nuke` (driven by ../nuke-config.yml) for the one failure
mode it routinely can't recover from on its own: a VPC (or its subnets /
security groups / internet gateway) left behind with `DependencyViolation`
because something is still attached to it -- a NAT gateway, an ENI owned by
a Lambda/RDS/ELB/EKS control plane, a VPC endpoint, a peering connection, an
EFS mount target, and so on. aws-nuke deletes each resource type in its own
pass without re-deriving the attachment graph between them, so anything with
a multi-step teardown (detach -> wait -> delete) tends to get stuck.

This script does the attachment graph explicitly, in two phases, per region:

  Phase 1 (account/region-wide): tear down every compute/managed-service
  resource that can be holding a VPC hostage -- ECS services/tasks/clusters,
  EKS nodegroups/fargate profiles/clusters, Lambda functions, Auto Scaling
  Groups, EC2 instances, ELB/ALB/NLB + target groups, RDS instances/clusters
  + subnet groups, ElastiCache clusters/replication groups + subnet groups,
  EFS mount targets + file systems, NAT Gateways, Elastic IPs, VPC Endpoints,
  VPC Peering Connections, Transit Gateway VPC Attachments, VPN
  Gateways/Connections/Customer Gateways, Egress-Only Internet Gateways.
  Async deletions (EC2 termination, NAT gateway teardown, RDS deletion) are
  polled to completion before moving on, since their dependent ENIs are
  exactly what blocks the VPC network deletion in phase 2.

  Phase 2 (per VPC): detach + delete the Internet Gateway, force-detach +
  delete any remaining ENIs the account itself owns, strip every security
  group's rules before deleting the groups (removes SG-to-SG cross
  references that block deletion order), delete non-default NACLs, subnets,
  non-main route tables, reset + delete custom DHCP option sets, and finally
  delete_vpc.

Some ENIs (Lambda hyperplane ENIs in particular) are owned and released by
AWS itself, asynchronously, and cannot be force-deleted by the account --
the script reports these by ID/description instead of retrying forever;
re-run the script a few minutes later and they'll be gone.

Beyond the pure attachment graph, phase 1 also unlocks/stops a few specific
things that are otherwise impossible to delete at all, based on real
aws-nuke failure logs against these accounts:

  - Load balancers with `deletion_protection.enabled=true` -- disabled before
    delete_load_balancer (the ALB/NLB equivalent of RDS DeletionProtection).
  - Self-owned AMIs still referencing an EBS snapshot -- deregistered before
    the snapshot delete is attempted (InvalidSnapshot.InUse otherwise).
  - Leftover EBS volumes (non-DeleteOnTermination) -- force-detached, then
    deleted.
  - S3 buckets with Object Lock enabled -- legal holds removed and
    GOVERNANCE-mode retention bypassed per object version before aws-nuke's
    own bucket-emptying pass runs. COMPLIANCE-mode retention has no bypass by
    design (not even for the account root) -- those objects are reported and
    skipped until their retention window actually expires.
  - KMS keys denying `kms:ScheduleKeyDeletion` to the caller -- the script
    tries to self-grant via PutKeyPolicy and retry; if the key's policy also
    denies PutKeyPolicy, only the AWS account's true root user (not just an
    IAM admin) can recover it, and the script says so instead of looping.
  - Glue database drops blocked by Lake Formation ("Insufficient Lake
    Formation permission(s)") -- the caller is added as a Lake Formation
    Data Lake Administrator so aws-nuke's own GlueDatabase pass can succeed.

Not handled: aws-nuke's own `LakeFormationPermission` revoke calls that fail
with `InvalidInputException: Table name and table wildcard cannot both be
present` are a bug in aws-nuke's own request construction, not a
permissions or locking problem this script (or any account-side fix) can
work around.

Safety
-------
- Every mutating call goes through Actions.do(), which no-ops and just
  prints under --dry-run.
- Refuses to run against any account ID listed under `blocklist:` in
  nuke-config.yml.
- Without --yes, requires typing the account ID back to confirm before any
  AWS call is made (--dry-run never needs this).
- Every teardown step is independently try/except'd and idempotent (missing
  resources / already-deleted / AccessDenied are logged and skipped, not
  fatal) -- safe to re-run repeatedly until a region reports clean.

This is meant for the disposable, single-purpose research AWS accounts
listed in ../nuke-config.yml -- it deletes real resources with no undo.

Usage
------
    python3 scripts/nuke_vpc_dependencies.py --profile default --dry-run
    python3 scripts/nuke_vpc_dependencies.py --profile senatwo --regions us-east-1,ap-southeast-2
    python3 scripts/nuke_vpc_dependencies.py --profile default --vpc-id vpc-0123456789abcdef0 --yes

Typical order of operations with aws-nuke:
    1. Run this script first to clear the dependency graph and delete VPCs.
    2. Run aws-nuke -c nuke-config.yml to sweep up everything else
       (IAM, S3, CloudWatch, KMS, etc.) that never had a detach problem.
    3. If aws-nuke still reports a handful of stuck VPC resources, re-run
       this script -- it's idempotent.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import botocore
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUKE_CONFIG = REPO_ROOT / "nuke-config.yml"

# AWS-managed ENI descriptions/owners we must never try to force-detach --
# they belong to a service control plane and are released asynchronously.
_AWS_MANAGED_ENI_HINT = "AWS-managed"


def log(msg: str) -> None:
    print(f"[Nuke] {msg}")


# ---------------------------------------------------------------------------
# Small helpers shared by every resource-type function
# ---------------------------------------------------------------------------

def paginate(client, method: str, key: str, **kwargs) -> list:
    """List every item of *key* from *method*, paginating when possible and
    swallowing (with a log line) any error so one missing permission or an
    unavailable service in a region never aborts the whole run."""
    try:
        paginator = client.get_paginator(method)
        items = []
        for page in paginator.paginate(**kwargs):
            items.extend(page.get(key, []))
        return items
    except botocore.exceptions.OperationNotPageableError:
        pass
    except Exception as e:
        log(f"  (list {method} failed: {e})")
        return []
    try:
        response = getattr(client, method)(**kwargs)
        return response.get(key, [])
    except Exception as e:
        log(f"  (list {method} failed: {e})")
        return []


def wait_until(check, description: str, timeout: int = 300, interval: int = 10) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if check():
                log(f"{description}: done")
                return True
        except Exception:
            pass
        time.sleep(interval)
    log(f"{description}: timed out after {timeout}s (continuing anyway)")
    return False


def _safe(fn, description: str) -> None:
    try:
        fn()
    except Exception as e:
        log(f"{description} step failed unexpectedly: {e} (continuing)")


class Actions:
    """Wraps every mutating AWS call so --dry-run is a single code path."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def do(self, description: str, fn):
        if self.dry_run:
            log(f"[dry-run] would {description}")
            return None
        try:
            result = fn()
            log(f"{description} ✓")
            return result
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            log(f"{description} ✗ ({code})")
            return None
        except Exception as e:
            log(f"{description} ✗ ({e})")
            return None


# ---------------------------------------------------------------------------
# Phase 1 -- account/region-wide compute & managed-service teardown
# ---------------------------------------------------------------------------

def nuke_ecs(session, region, actions):
    ecs = session.client("ecs", region_name=region)
    for cluster_arn in paginate(ecs, "list_clusters", "clusterArns"):
        services = paginate(ecs, "list_services", "serviceArns", cluster=cluster_arn)
        for svc in services:
            actions.do(f"scale ECS service {svc} to 0",
                       lambda c=cluster_arn, s=svc: ecs.update_service(cluster=c, service=s, desiredCount=0))
        for svc in services:
            actions.do(f"delete ECS service {svc}",
                       lambda c=cluster_arn, s=svc: ecs.delete_service(cluster=c, service=s, force=True))
        for task in paginate(ecs, "list_tasks", "taskArns", cluster=cluster_arn):
            actions.do(f"stop ECS task {task}",
                       lambda c=cluster_arn, t=task: ecs.stop_task(cluster=c, task=t, reason="nuke_vpc_dependencies"))
        for ci in paginate(ecs, "list_container_instances", "containerInstanceArns", cluster=cluster_arn):
            actions.do(f"deregister ECS container instance {ci}",
                       lambda c=cluster_arn, x=ci: ecs.deregister_container_instance(cluster=c, containerInstance=x, force=True))
        actions.do(f"delete ECS cluster {cluster_arn}", lambda c=cluster_arn: ecs.delete_cluster(cluster=c))


def nuke_eks(session, region, actions):
    eks = session.client("eks", region_name=region)
    for cluster_name in paginate(eks, "list_clusters", "clusters"):
        for ng in paginate(eks, "list_nodegroups", "nodegroups", clusterName=cluster_name):
            actions.do(f"delete EKS nodegroup {cluster_name}/{ng}",
                       lambda c=cluster_name, n=ng: eks.delete_nodegroup(clusterName=c, nodegroupName=n))
        for fp in paginate(eks, "list_fargate_profiles", "fargateProfileNames", clusterName=cluster_name):
            actions.do(f"delete EKS fargate profile {cluster_name}/{fp}",
                       lambda c=cluster_name, f=fp: eks.delete_fargate_profile(clusterName=c, fargateProfileName=f))
        # Nodegroups/Fargate profiles take minutes to actually disappear, so this
        # cluster delete will often fail on the first pass -- that's expected;
        # re-running the script later picks it back up.
        actions.do(f"delete EKS cluster {cluster_name}", lambda c=cluster_name: eks.delete_cluster(name=c))


def nuke_lambda(session, region, actions):
    lam = session.client("lambda", region_name=region)
    for fn in paginate(lam, "list_functions", "Functions"):
        name = fn["FunctionName"]
        actions.do(f"delete Lambda function {name}", lambda n=name: lam.delete_function(FunctionName=n))


def nuke_asg(session, region, actions):
    asg = session.client("autoscaling", region_name=region)
    for group in paginate(asg, "describe_auto_scaling_groups", "AutoScalingGroups"):
        name = group["AutoScalingGroupName"]
        actions.do(f"force-delete Auto Scaling Group {name}",
                   lambda n=name: asg.delete_auto_scaling_group(AutoScalingGroupName=n, ForceDelete=True))
    for lc in paginate(asg, "describe_launch_configurations", "LaunchConfigurations"):
        name = lc["LaunchConfigurationName"]
        actions.do(f"delete launch configuration {name}",
                   lambda n=name: asg.delete_launch_configuration(LaunchConfigurationName=n))


def nuke_ec2_instances(session, region, actions) -> list:
    ec2 = session.client("ec2", region_name=region)
    reservations = paginate(
        ec2, "describe_instances", "Reservations",
        Filters=[{"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}],
    )
    ids = [i["InstanceId"] for r in reservations for i in r["Instances"]]
    if ids:
        actions.do(f"terminate {len(ids)} EC2 instance(s)", lambda: ec2.terminate_instances(InstanceIds=ids))
    return ids


def nuke_amis_and_snapshots(session, region, actions):
    """A self-owned EBS snapshot still backing an AMI fails to delete with
    `InvalidSnapshot.InUse` -- deregister the AMI first (which is what
    actually releases the snapshot), then delete the snapshot."""
    ec2 = session.client("ec2", region_name=region)
    snapshot_ids = set()
    for image in paginate(ec2, "describe_images", "Images", Owners=["self"]):
        image_id = image["ImageId"]
        for mapping in image.get("BlockDeviceMappings", []):
            ebs = mapping.get("Ebs")
            if ebs and ebs.get("SnapshotId"):
                snapshot_ids.add(ebs["SnapshotId"])
        actions.do(f"deregister AMI {image_id}", lambda i=image_id: ec2.deregister_image(ImageId=i))

    if snapshot_ids and not actions.dry_run:
        time.sleep(5)  # the InUse association clears almost immediately, but not synchronously
    for snapshot_id in snapshot_ids:
        actions.do(f"delete EBS snapshot {snapshot_id} (freed by AMI deregistration)",
                   lambda s=snapshot_id: ec2.delete_snapshot(SnapshotId=s))


def nuke_ebs_volumes(session, region, actions):
    """Volumes created with DeleteOnTermination=false survive their instance's
    termination and linger as 'in-use' (attached to a now-terminated
    instance) until force-detached."""
    ec2 = session.client("ec2", region_name=region)
    for volume in paginate(ec2, "describe_volumes", "Volumes"):
        vol_id = volume["VolumeId"]
        for attachment in volume.get("Attachments", []):
            if attachment.get("State") == "attached":
                actions.do(f"force-detach EBS volume {vol_id}",
                           lambda v=vol_id, i=attachment["InstanceId"]: ec2.detach_volume(VolumeId=v, InstanceId=i, Force=True))
        actions.do(f"delete EBS volume {vol_id}", lambda v=vol_id: ec2.delete_volume(VolumeId=v))


def nuke_elb(session, region, actions) -> list:
    """Deletes classic + v2 load balancers and their target groups. Returns
    the v2 ARNs so the caller can wait for their (asynchronous) deletion --
    their ENIs, and anything they hold onto (Elastic IPs, security groups),
    aren't released until the load balancer is actually gone, not just when
    delete_load_balancer returns."""
    elb = session.client("elb", region_name=region)
    for lb in paginate(elb, "describe_load_balancers", "LoadBalancerDescriptions"):
        name = lb["LoadBalancerName"]
        actions.do(f"delete classic ELB {name}", lambda n=name: elb.delete_load_balancer(LoadBalancerName=n))

    elbv2 = session.client("elbv2", region_name=region)
    v2_arns = []
    for lb in paginate(elbv2, "describe_load_balancers", "LoadBalancers"):
        arn, name = lb["LoadBalancerArn"], lb["LoadBalancerName"]
        v2_arns.append(arn)

        # ALB/NLB deletion protection blocks delete_load_balancer outright
        # (OperationNotPermitted) -- unlock it first, same idea as the RDS
        # DeletionProtection handling in nuke_rds() below.
        try:
            attrs = elbv2.describe_load_balancer_attributes(LoadBalancerArn=arn).get("Attributes", [])
        except Exception:
            attrs = []
        if any(a.get("Key") == "deletion_protection.enabled" and a.get("Value") == "true" for a in attrs):
            actions.do(f"disable deletion protection on load balancer {name}",
                       lambda a=arn: elbv2.modify_load_balancer_attributes(
                           LoadBalancerArn=a, Attributes=[{"Key": "deletion_protection.enabled", "Value": "false"}]))

        actions.do(f"delete load balancer {name}", lambda a=arn: elbv2.delete_load_balancer(LoadBalancerArn=a))

    for tg in paginate(elbv2, "describe_target_groups", "TargetGroups"):
        arn, name = tg["TargetGroupArn"], tg["TargetGroupName"]
        actions.do(f"delete target group {name}", lambda a=arn: elbv2.delete_target_group(TargetGroupArn=a))

    return v2_arns


def wait_for_elbv2_deleted(elbv2, arns: list):
    if not arns:
        return

    def gone(arn):
        try:
            elbv2.describe_load_balancers(LoadBalancerArns=[arn])
            return False
        except botocore.exceptions.ClientError as e:
            return e.response.get("Error", {}).get("Code") == "LoadBalancerNotFound"
        except Exception:
            return False

    wait_until(lambda: all(gone(a) for a in arns),
               "waiting for load balancer(s) to finish deleting (releases their ENIs/EIPs)",
               timeout=300, interval=10)


def nuke_rds(session, region, actions):
    rds = session.client("rds", region_name=region)
    for cluster in paginate(rds, "describe_db_clusters", "DBClusters"):
        cid = cluster["DBClusterIdentifier"]
        if cluster.get("DeletionProtection"):
            actions.do(f"disable deletion protection on RDS cluster {cid}",
                       lambda c=cid: rds.modify_db_cluster(DBClusterIdentifier=c, DeletionProtection=False, ApplyImmediately=True))
    for inst in paginate(rds, "describe_db_instances", "DBInstances"):
        iid = inst["DBInstanceIdentifier"]
        if inst.get("DeletionProtection"):
            actions.do(f"disable deletion protection on RDS instance {iid}",
                       lambda i=iid: rds.modify_db_instance(DBInstanceIdentifier=i, DeletionProtection=False, ApplyImmediately=True))
    for inst in paginate(rds, "describe_db_instances", "DBInstances"):
        iid = inst["DBInstanceIdentifier"]
        actions.do(f"delete RDS instance {iid}",
                   lambda i=iid: rds.delete_db_instance(DBInstanceIdentifier=i, SkipFinalSnapshot=True, DeleteAutomatedBackups=True))
    for cluster in paginate(rds, "describe_db_clusters", "DBClusters"):
        cid = cluster["DBClusterIdentifier"]
        actions.do(f"delete RDS cluster {cid}", lambda c=cid: rds.delete_db_cluster(DBClusterIdentifier=c, SkipFinalSnapshot=True))


def nuke_rds_subnet_groups(session, region, actions):
    rds = session.client("rds", region_name=region)
    for sg in paginate(rds, "describe_db_subnet_groups", "DBSubnetGroups"):
        name = sg["DBSubnetGroupName"]
        actions.do(f"delete RDS subnet group {name}", lambda n=name: rds.delete_db_subnet_group(DBSubnetGroupName=n))


def nuke_elasticache(session, region, actions):
    ec = session.client("elasticache", region_name=region)
    for rg in paginate(ec, "describe_replication_groups", "ReplicationGroups"):
        rgid = rg["ReplicationGroupId"]
        actions.do(f"delete ElastiCache replication group {rgid}",
                   lambda r=rgid: ec.delete_replication_group(ReplicationGroupId=r))
    for cluster in paginate(ec, "describe_cache_clusters", "CacheClusters"):
        if cluster.get("ReplicationGroupId"):
            continue  # deleted via its replication group above
        cid = cluster["CacheClusterId"]
        actions.do(f"delete ElastiCache cluster {cid}", lambda c=cid: ec.delete_cache_cluster(CacheClusterId=c))


def nuke_elasticache_subnet_groups(session, region, actions):
    ec = session.client("elasticache", region_name=region)
    for sg in paginate(ec, "describe_cache_subnet_groups", "CacheSubnetGroups"):
        name = sg["CacheSubnetGroupName"]
        if name == "default":
            continue
        actions.do(f"delete ElastiCache subnet group {name}",
                   lambda n=name: ec.delete_cache_subnet_group(CacheSubnetGroupName=n))


def nuke_efs_mount_targets(session, region, actions):
    efs = session.client("efs", region_name=region)
    for fs in paginate(efs, "describe_file_systems", "FileSystems"):
        fsid = fs["FileSystemId"]
        for mt in paginate(efs, "describe_mount_targets", "MountTargets", FileSystemId=fsid):
            mtid = mt["MountTargetId"]
            actions.do(f"delete EFS mount target {mtid}", lambda m=mtid: efs.delete_mount_target(MountTargetId=m))


def nuke_efs_file_systems(session, region, actions):
    efs = session.client("efs", region_name=region)
    for fs in paginate(efs, "describe_file_systems", "FileSystems"):
        fsid = fs["FileSystemId"]
        actions.do(f"delete EFS file system {fsid}", lambda f=fsid: efs.delete_file_system(FileSystemId=f))


def nuke_nat_gateways(session, region, actions) -> list:
    ec2 = session.client("ec2", region_name=region)
    nats = paginate(ec2, "describe_nat_gateways", "NatGateways",
                    Filter=[{"Name": "state", "Values": ["available", "pending"]}])
    ids = []
    for nat in nats:
        nid = nat["NatGatewayId"]
        ids.append(nid)
        actions.do(f"delete NAT gateway {nid}", lambda n=nid: ec2.delete_nat_gateway(NatGatewayId=n))
    return ids


def nuke_elastic_ips(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    for addr in paginate(ec2, "describe_addresses", "Addresses"):
        public_ip = addr.get("PublicIp", "?")
        assoc_id = addr.get("AssociationId")
        alloc_id = addr.get("AllocationId")
        if assoc_id:
            actions.do(f"disassociate Elastic IP {public_ip}", lambda a=assoc_id: ec2.disassociate_address(AssociationId=a))
        if alloc_id:
            actions.do(f"release Elastic IP {public_ip}", lambda a=alloc_id: ec2.release_address(AllocationId=a))


def nuke_vpc_endpoints(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    ids = [e["VpcEndpointId"] for e in paginate(ec2, "describe_vpc_endpoints", "VpcEndpoints")
           if e.get("State") not in ("deleted", "deleting")]
    if ids:
        actions.do(f"delete {len(ids)} VPC endpoint(s)", lambda: ec2.delete_vpc_endpoints(VpcEndpointIds=ids))


def nuke_peering_connections(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    for pcx in paginate(ec2, "describe_vpc_peering_connections", "VpcPeeringConnections"):
        if pcx.get("Status", {}).get("Code") in ("deleted", "deleting", "rejected", "expired", "failed"):
            continue
        pid = pcx["VpcPeeringConnectionId"]
        actions.do(f"delete VPC peering connection {pid}",
                   lambda p=pid: ec2.delete_vpc_peering_connection(VpcPeeringConnectionId=p))


def nuke_transit_gateway_attachments(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    for att in paginate(ec2, "describe_transit_gateway_vpc_attachments", "TransitGatewayVpcAttachments"):
        if att.get("State") in ("deleted", "deleting"):
            continue
        aid = att["TransitGatewayAttachmentId"]
        actions.do(f"delete transit gateway VPC attachment {aid}",
                   lambda a=aid: ec2.delete_transit_gateway_vpc_attachment(TransitGatewayAttachmentId=a))


def nuke_vpn(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    for conn in paginate(ec2, "describe_vpn_connections", "VpnConnections"):
        if conn.get("State") in ("deleted", "deleting"):
            continue
        cid = conn["VpnConnectionId"]
        actions.do(f"delete VPN connection {cid}", lambda c=cid: ec2.delete_vpn_connection(VpnConnectionId=c))

    for vgw in paginate(ec2, "describe_vpn_gateways", "VpnGateways"):
        if vgw.get("State") in ("deleted", "deleting"):
            continue
        vid = vgw["VpnGatewayId"]
        for att in vgw.get("VpcAttachments", []):
            if att.get("State") == "attached":
                actions.do(f"detach VPN gateway {vid} from {att['VpcId']}",
                           lambda v=vid, vpc=att["VpcId"]: ec2.detach_vpn_gateway(VpnGatewayId=v, VpcId=vpc))
        actions.do(f"delete VPN gateway {vid}", lambda v=vid: ec2.delete_vpn_gateway(VpnGatewayId=v))

    for cgw in paginate(ec2, "describe_customer_gateways", "CustomerGateways"):
        if cgw.get("State") in ("deleted", "deleting"):
            continue
        cid = cgw["CustomerGatewayId"]
        actions.do(f"delete customer gateway {cid}", lambda c=cid: ec2.delete_customer_gateway(CustomerGatewayId=c))


def nuke_egress_only_igw(session, region, actions):
    ec2 = session.client("ec2", region_name=region)
    for eigw in paginate(ec2, "describe_egress_only_internet_gateways", "EgressOnlyInternetGateways"):
        eid = eigw["EgressOnlyInternetGatewayId"]
        actions.do(f"delete egress-only internet gateway {eid}",
                   lambda e=eid: ec2.delete_egress_only_internet_gateway(EgressOnlyInternetGatewayId=e))


# ---------------------------------------------------------------------------
# "Unlock" helpers -- resources that are technically deletable but denied by
# a lock/policy of their own (KMS key policy, S3 Object Lock, Lake Formation)
# rather than by an attachment. aws-nuke has no notion of fixing these itself.
# ---------------------------------------------------------------------------

def nuke_kms_keys(session, region, actions):
    kms = session.client("kms", region_name=region)
    try:
        caller_arn = session.client("sts", region_name=region).get_caller_identity()["Arn"]
    except Exception as e:
        log(f"  (could not resolve caller identity for KMS self-grant: {e})")
        caller_arn = None

    for key in paginate(kms, "list_keys", "Keys"):
        key_id = key["KeyId"]
        try:
            meta = kms.describe_key(KeyId=key_id)["KeyMetadata"]
        except Exception:
            continue
        if meta.get("KeyManager") != "CUSTOMER" or meta.get("KeyState") in ("PendingDeletion", "PendingReplicaDeletion"):
            continue

        if actions.dry_run:
            # Whether AWS would deny kms:ScheduleKeyDeletion (requiring the
            # self-grant retry below) can only be known by actually calling
            # it -- which a dry run must never do. Report the plan generically
            # instead of guessing which branch a live run would take.
            log(f"[dry-run] would schedule deletion of KMS key {key_id} (7-day window; if its key "
                f"policy denies that, would self-grant via PutKeyPolicy and retry once)")
            continue

        try:
            kms.schedule_key_deletion(KeyId=key_id, PendingWindowInDays=7)
            log(f"schedule deletion of KMS key {key_id} (7-day window) ✓")
            continue
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") != "AccessDeniedException" or not caller_arn:
                log(f"schedule deletion of KMS key {key_id} ✗ ({e.response.get('Error', {}).get('Code', e)})")
                continue
        except Exception as e:
            log(f"schedule deletion of KMS key {key_id} ✗ ({e})")
            continue

        # Denied by the key's own resource policy, not by IAM -- try
        # self-granting via PutKeyPolicy (only works if PutKeyPolicy itself
        # is allowed by the current policy) and retry once. Both calls are
        # real AWS mutations, only reachable once the dry-run guard above has
        # already returned -- i.e. never during --dry-run.
        log(f"  KMS key {key_id} denies kms:ScheduleKeyDeletion to {caller_arn} -- "
            f"attempting to self-grant via its key policy")
        try:
            policy_doc = json.loads(kms.get_key_policy(KeyId=key_id, PolicyName="default")["Policy"])
            policy_doc.setdefault("Statement", []).append({
                "Sid": "NukeVpcDependenciesSelfGrant",
                "Effect": "Allow",
                "Principal": {"AWS": caller_arn},
                "Action": "kms:*",
                "Resource": "*",
            })
            kms.put_key_policy(KeyId=key_id, PolicyName="default", Policy=json.dumps(policy_doc))
            log(f"grant {caller_arn} kms:* on key {key_id} via key policy ✓")
            kms.schedule_key_deletion(KeyId=key_id, PendingWindowInDays=7)
            log(f"schedule deletion of KMS key {key_id} (7-day window) ✓")
        except Exception as e:
            log(f"  could not self-grant on KMS key {key_id}: {e} -- this key needs the AWS account's "
                f"actual root user (not just an IAM admin) to fix its key policy before it can be deleted.")


def unlock_locked_s3_buckets(session, region, actions):
    """Removes legal holds and bypasses GOVERNANCE-mode retention on every
    object version in every Object-Lock-enabled bucket in *region*, so
    aws-nuke's own (lock-unaware) bucket-emptying pass can then succeed.
    COMPLIANCE-mode retention has no bypass, by AWS design -- those versions
    are reported and left alone until their retention window expires.

    Also clears each bucket's resource-based policy first: confirmed against
    a real account, a Terraform-state bucket's policy carried an explicit
    Deny that blocked even the account's own admin user from
    GetBucketObjectLockConfiguration -- an explicit Deny in a resource
    policy always overrides an IAM Allow, so no identity-side permission fix
    can work around it, only removing/editing the policy can. Deleting it
    outright is safe here since the whole account is being torn down; if the
    Deny also covers DeleteBucketPolicy itself, that call fails too and the
    bucket needs the account's true root user (or whichever principal the
    policy does allow) to clear it."""
    s3 = session.client("s3", region_name=region)
    try:
        buckets = s3.list_buckets().get("Buckets", [])
    except Exception as e:
        log(f"  (list S3 buckets failed: {e})")
        return

    for bucket in buckets:
        name = bucket["Name"]
        try:
            loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
            bucket_region = loc or "us-east-1"
            if bucket_region != region:
                continue
        except botocore.exceptions.ClientError:
            # GetBucketLocation itself can be blocked by the same explicit-Deny
            # bucket policy this function exists to clear (confirmed against a
            # real account) -- skipping the bucket here would mean the unlock
            # logic below never even runs. Fall through and try it against the
            # current region's client instead of assuming it belongs elsewhere;
            # a genuinely wrong-region attempt just fails harmlessly below.
            pass

        existing_policy = None
        needs_policy_clear = False
        try:
            existing_policy = s3.get_bucket_policy(Bucket=name).get("Policy")
            needs_policy_clear = True
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") != "NoSuchBucketPolicy":
                needs_policy_clear = True  # can't even read it -- still worth a blind delete attempt
        if needs_policy_clear:
            cleared = actions.do(
                f"delete bucket policy on s3://{name} (an explicit Deny there can block Object Lock "
                f"checks, object deletes, or the bucket delete itself, even for the account's own admin)",
                lambda n=name: s3.delete_bucket_policy(Bucket=n),
            )
            if cleared is None and not actions.dry_run:
                log(f"  could not remove the policy on s3://{name} either (or even read it, if "
                    f"GetBucketPolicy was also denied) -- its explicit Deny covers DeleteBucketPolicy "
                    f"too, which no IAM identity in this account can override. This needs either the "
                    f"AWS account's true root-user login (not an IAM user, even one named 'root') or "
                    f"an AWS Support case for an S3 bucket-policy self-lockout -- aws-nuke will keep "
                    f"failing to empty/delete s3://{name} until then.")

        try:
            lock_cfg = s3.get_object_lock_configuration(Bucket=name)
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code != "ObjectLockConfigurationNotFoundError":
                log(f"  could not check Object Lock status on s3://{name} ({code}) -- it may still "
                    f"have locked objects blocking its deletion; if its bucket policy denied this "
                    f"check, that's likely the same policy the block above just tried to clear")
            continue
        if lock_cfg.get("ObjectLockConfiguration", {}).get("ObjectLockEnabled") != "Enabled":
            continue

        log(f"  bucket {name} has Object Lock enabled -- clearing legal holds / GOVERNANCE retention")
        for page in s3.get_paginator("list_object_versions").paginate(Bucket=name):
            for version in page.get("Versions", []) + page.get("DeleteMarkers", []):
                key, version_id = version["Key"], version["VersionId"]

                try:
                    hold = s3.get_object_legal_hold(Bucket=name, Key=key, VersionId=version_id)
                    if hold.get("LegalHold", {}).get("Status") == "ON":
                        actions.do(f"remove legal hold on s3://{name}/{key}",
                                   lambda n=name, k=key, v=version_id: s3.put_object_legal_hold(
                                       Bucket=n, Key=k, VersionId=v, LegalHold={"Status": "OFF"}))
                except botocore.exceptions.ClientError:
                    pass  # no legal hold on this version

                mode, retain_until = None, None
                try:
                    retention = s3.get_object_retention(Bucket=name, Key=key, VersionId=version_id).get("Retention", {})
                    mode, retain_until = retention.get("Mode"), retention.get("RetainUntilDate")
                except botocore.exceptions.ClientError:
                    pass  # no retention set on this version

                if mode == "COMPLIANCE":
                    log(f"    s3://{name}/{key} (v{version_id}) is under COMPLIANCE retention until "
                        f"{retain_until} -- cannot be deleted by anyone (including root) until then. Skipping.")
                    continue

                extra = {"BypassGovernanceRetention": True} if mode == "GOVERNANCE" else {}
                actions.do(f"delete s3://{name}/{key}" + (" (bypassing GOVERNANCE retention)" if mode else ""),
                           lambda n=name, k=key, v=version_id, ex=extra: s3.delete_object(Bucket=n, Key=k, VersionId=v, **ex))


def unlock_lake_formation(session, region, actions):
    """Glue database drops fail with 'Insufficient Lake Formation
    permission(s): Required Drop on <db>' regardless of IAM policy, and --
    confirmed against a real account -- regardless of Data Lake
    Administrator status too: DataLakeAdmins controls who can manage LF
    settings/registrations, but does NOT retroactively grant Drop on
    already-registered databases. The actual fix needed is an explicit
    grant_permissions(ALL) per database to the caller, in addition to (not
    instead of) admin status."""
    try:
        lf = session.client("lakeformation", region_name=region)
        glue = session.client("glue", region_name=region)
        caller_arn = session.client("sts", region_name=region).get_caller_identity()["Arn"]
        settings = lf.get_data_lake_settings().get("DataLakeSettings", {})
    except Exception as e:
        log(f"  (Lake Formation not reachable/available in this region: {e})")
        return

    admins = settings.get("DataLakeAdmins", [])
    if not any(a.get("DataLakePrincipalIdentifier") == caller_arn for a in admins):
        settings["DataLakeAdmins"] = admins + [{"DataLakePrincipalIdentifier": caller_arn}]
        actions.do(f"grant {caller_arn} Lake Formation Data Lake Administrator",
                   lambda s=settings: lf.put_data_lake_settings(DataLakeSettings=s))

    for db in paginate(glue, "get_databases", "DatabaseList"):
        name = db["Name"]
        actions.do(f"grant {caller_arn} ALL Lake Formation permission on Glue database {name} (unblocks its drop)",
                   lambda n=name: lf.grant_permissions(
                       Principal={"DataLakePrincipalIdentifier": caller_arn},
                       Resource={"Database": {"Name": n}},
                       Permissions=["ALL"],
                   ))


# ---------------------------------------------------------------------------
# Async waits -- give phase 1's slow deletions a chance to actually finish
# before phase 2 tries to delete the VPC that depends on them.
# ---------------------------------------------------------------------------

def wait_for_instances_terminated(ec2, instance_ids):
    if not instance_ids:
        return

    def check():
        reservations = ec2.describe_instances(InstanceIds=instance_ids)["Reservations"]
        states = {i["State"]["Name"] for r in reservations for i in r["Instances"]}
        return states <= {"terminated"}

    wait_until(check, "waiting for EC2 instance termination", timeout=300)


def wait_for_nat_gateways_deleted(ec2, nat_ids):
    if not nat_ids:
        return

    def check():
        nats = ec2.describe_nat_gateways(NatGatewayIds=nat_ids)["NatGateways"]
        return all(n["State"] == "deleted" for n in nats)

    wait_until(check, "waiting for NAT gateway deletion", timeout=420, interval=15)


def wait_for_rds_gone(rds):
    def check():
        return not paginate(rds, "describe_db_instances", "DBInstances") and \
            not paginate(rds, "describe_db_clusters", "DBClusters")

    wait_until(check, "waiting for RDS instances/clusters to finish deleting", timeout=600, interval=15)


# ---------------------------------------------------------------------------
# Phase 2 -- per-VPC network teardown
# ---------------------------------------------------------------------------

def list_target_vpcs(ec2, include_default: bool, only_ids: list) -> list:
    vpcs = paginate(ec2, "describe_vpcs", "Vpcs")
    if only_ids:
        vpcs = [v for v in vpcs if v["VpcId"] in only_ids]
    if not include_default:
        vpcs = [v for v in vpcs if not v.get("IsDefault")]
    return vpcs


def detach_delete_igws(ec2, vpc_id, actions):
    igws = paginate(ec2, "describe_internet_gateways", "InternetGateways",
                    Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])
    for igw in igws:
        gid = igw["InternetGatewayId"]
        actions.do(f"detach internet gateway {gid} from {vpc_id}",
                   lambda g=gid: ec2.detach_internet_gateway(InternetGatewayId=g, VpcId=vpc_id))
        actions.do(f"delete internet gateway {gid}", lambda g=gid: ec2.delete_internet_gateway(InternetGatewayId=g))


def delete_enis(ec2, vpc_id, actions, dry_run: bool) -> list:
    """Force-detach and delete account-owned ENIs. AWS-service-managed ENIs
    (RequesterManaged=True -- Lambda hyperplane, RDS, ELB, EKS control plane)
    can't be force-deleted by the account and are reported instead, since
    they release themselves asynchronously."""
    remaining = []
    for eni in paginate(ec2, "describe_network_interfaces", "NetworkInterfaces",
                        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]):
        eni_id = eni["NetworkInterfaceId"]
        if eni.get("RequesterManaged"):
            remaining.append((eni_id, eni.get("Description", "")))
            continue

        attachment = eni.get("Attachment") or {}
        if attachment.get("AttachmentId") and attachment.get("Status") == "attached":
            attachment_id = attachment["AttachmentId"]
            actions.do(f"force-detach ENI {eni_id}",
                       lambda a=attachment_id: ec2.detach_network_interface(AttachmentId=a, Force=True))
            if not dry_run:
                wait_until(
                    lambda e=eni_id: ec2.describe_network_interfaces(NetworkInterfaceIds=[e])
                    ["NetworkInterfaces"][0]["Status"] == "available",
                    f"waiting for ENI {eni_id} to detach", timeout=60, interval=5,
                )
        actions.do(f"delete ENI {eni_id}", lambda e=eni_id: ec2.delete_network_interface(NetworkInterfaceId=e))
    return remaining


def strip_and_delete_security_groups(ec2, vpc_id, actions):
    sgs = paginate(ec2, "describe_security_groups", "SecurityGroups",
                   Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    # Strip every group's rules FIRST -- security groups routinely reference
    # each other (SG-to-SG ingress rules), which blocks deletion order unless
    # the cross references are removed before any group is deleted.
    for sg in sgs:
        gid = sg["GroupId"]
        if sg.get("IpPermissions"):
            actions.do(f"revoke ingress rules on security group {gid}",
                       lambda g=gid, p=sg["IpPermissions"]: ec2.revoke_security_group_ingress(GroupId=g, IpPermissions=p))
        if sg.get("IpPermissionsEgress"):
            actions.do(f"revoke egress rules on security group {gid}",
                       lambda g=gid, p=sg["IpPermissionsEgress"]: ec2.revoke_security_group_egress(GroupId=g, IpPermissions=p))
    for sg in sgs:
        if sg["GroupName"] == "default":
            continue
        gid = sg["GroupId"]
        actions.do(f"delete security group {gid} ({sg['GroupName']})", lambda g=gid: ec2.delete_security_group(GroupId=g))


def delete_nacls(ec2, vpc_id, actions):
    for nacl in paginate(ec2, "describe_network_acls", "NetworkAcls",
                         Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]):
        if nacl.get("IsDefault"):
            continue
        nid = nacl["NetworkAclId"]
        actions.do(f"delete network ACL {nid}", lambda n=nid: ec2.delete_network_acl(NetworkAclId=n))


def delete_subnets(ec2, vpc_id, actions):
    for subnet in paginate(ec2, "describe_subnets", "Subnets", Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]):
        sid = subnet["SubnetId"]
        actions.do(f"delete subnet {sid}", lambda s=sid: ec2.delete_subnet(SubnetId=s))


def delete_route_tables(ec2, vpc_id, actions):
    for rt in paginate(ec2, "describe_route_tables", "RouteTables", Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]):
        if any(a.get("Main") for a in rt.get("Associations", [])):
            continue  # the main route table is deleted implicitly with the VPC
        rid = rt["RouteTableId"]
        for assoc in rt.get("Associations", []):
            assoc_id = assoc.get("RouteTableAssociationId")
            if assoc_id and not assoc.get("Main"):
                actions.do(f"disassociate route table {rid}",
                           lambda a=assoc_id: ec2.disassociate_route_table(AssociationId=a))
        actions.do(f"delete route table {rid}", lambda r=rid: ec2.delete_route_table(RouteTableId=r))


def reset_dhcp_options(ec2, vpc, actions):
    dopt_id = vpc.get("DhcpOptionsId")
    if not dopt_id or dopt_id == "default":
        return
    vpc_id = vpc["VpcId"]
    actions.do(f"reset {vpc_id} to the default DHCP options set",
               lambda v=vpc_id: ec2.associate_dhcp_options(DhcpOptionsId="default", VpcId=v))
    other_users = paginate(ec2, "describe_vpcs", "Vpcs", Filters=[{"Name": "dhcp-options-id", "Values": [dopt_id]}])
    if not any(v["VpcId"] != vpc_id for v in other_users):
        actions.do(f"delete DHCP options set {dopt_id}", lambda d=dopt_id: ec2.delete_dhcp_options(DhcpOptionsId=d))


def nuke_vpc_network(ec2, vpc, actions, dry_run: bool):
    vpc_id = vpc["VpcId"]
    kind = "default" if vpc.get("IsDefault") else "non-default"
    log(f"--- VPC {vpc_id} ({kind}) ---")

    detach_delete_igws(ec2, vpc_id, actions)
    remaining_enis = delete_enis(ec2, vpc_id, actions, dry_run)
    strip_and_delete_security_groups(ec2, vpc_id, actions)
    delete_nacls(ec2, vpc_id, actions)
    delete_subnets(ec2, vpc_id, actions)
    delete_route_tables(ec2, vpc_id, actions)
    reset_dhcp_options(ec2, vpc, actions)

    if remaining_enis and not dry_run:
        log(f"  {len(remaining_enis)} AWS-managed ENI(s) still attached -- these release on their "
            f"own within a few minutes (Lambda hyperplane / RDS / ELB / EKS control plane); "
            f"re-run this script afterward to finish deleting {vpc_id}:")
        for eni_id, desc in remaining_enis:
            log(f"    - {eni_id}: {desc or '(no description)'}")

    actions.do(f"delete VPC {vpc_id}", lambda v=vpc_id: ec2.delete_vpc(VpcId=v))


# ---------------------------------------------------------------------------
# Per-region orchestration
# ---------------------------------------------------------------------------

def nuke_region(session, region: str, actions: Actions, args) -> None:
    log(f"=== Region {region} ===")
    ec2 = session.client("ec2", region_name=region)
    elbv2 = session.client("elbv2", region_name=region)
    rds = session.client("rds", region_name=region)

    _safe(lambda: nuke_ecs(session, region, actions), "ECS")
    _safe(lambda: nuke_eks(session, region, actions), "EKS")
    _safe(lambda: nuke_lambda(session, region, actions), "Lambda")
    _safe(lambda: nuke_asg(session, region, actions), "Auto Scaling")
    instance_ids = []
    _safe(lambda: instance_ids.extend(nuke_ec2_instances(session, region, actions)), "EC2 instances")
    _safe(lambda: nuke_amis_and_snapshots(session, region, actions), "AMIs and EBS snapshots")
    elbv2_arns = []
    _safe(lambda: elbv2_arns.extend(nuke_elb(session, region, actions)), "ELB/ALB/NLB")
    _safe(lambda: nuke_rds(session, region, actions), "RDS")
    _safe(lambda: nuke_elasticache(session, region, actions), "ElastiCache")
    _safe(lambda: nuke_efs_mount_targets(session, region, actions), "EFS mount targets")
    nat_ids = []
    _safe(lambda: nat_ids.extend(nuke_nat_gateways(session, region, actions)), "NAT gateways")

    if not args.dry_run:
        _safe(lambda: wait_for_instances_terminated(ec2, instance_ids), "EC2 termination wait")
        _safe(lambda: wait_for_elbv2_deleted(elbv2, elbv2_arns), "Load balancer deletion wait")
        _safe(lambda: wait_for_nat_gateways_deleted(ec2, nat_ids), "NAT gateway deletion wait")
        _safe(lambda: wait_for_rds_gone(rds), "RDS deletion wait")

    _safe(lambda: nuke_ebs_volumes(session, region, actions), "EBS volumes")
    _safe(lambda: nuke_rds_subnet_groups(session, region, actions), "RDS subnet groups")
    _safe(lambda: nuke_elasticache_subnet_groups(session, region, actions), "ElastiCache subnet groups")
    _safe(lambda: nuke_efs_file_systems(session, region, actions), "EFS file systems")
    _safe(lambda: nuke_elastic_ips(session, region, actions), "Elastic IPs")
    _safe(lambda: nuke_vpc_endpoints(session, region, actions), "VPC endpoints")
    _safe(lambda: nuke_peering_connections(session, region, actions), "VPC peering connections")
    _safe(lambda: nuke_transit_gateway_attachments(session, region, actions), "Transit Gateway attachments")
    _safe(lambda: nuke_vpn(session, region, actions), "VPN gateways/connections")
    _safe(lambda: nuke_egress_only_igw(session, region, actions), "Egress-only internet gateways")
    _safe(lambda: nuke_kms_keys(session, region, actions), "KMS keys")
    _safe(lambda: unlock_locked_s3_buckets(session, region, actions), "S3 Object Lock buckets")
    _safe(lambda: unlock_lake_formation(session, region, actions), "Lake Formation admin grant")

    vpcs = list_target_vpcs(ec2, args.include_default_vpc, args.vpc_id)
    if not vpcs:
        log("No target VPCs found in this region.")
        return
    for vpc in vpcs:
        _safe(lambda v=vpc: nuke_vpc_network(ec2, v, actions, args.dry_run), f"VPC {vpc['VpcId']} network teardown")


# ---------------------------------------------------------------------------
# Config loading, safety gate, CLI
# ---------------------------------------------------------------------------

def load_nuke_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_regions(args, cfg: dict) -> list:
    # "global" is a pseudo-region aws-nuke uses for IAM/S3/etc, not a real AWS
    # region -- passing it to a regional EC2/ECS/... endpoint just hangs on
    # retries against a nonexistent host, so it's filtered out unconditionally.
    if args.regions:
        candidates = [r.strip() for r in args.regions.split(",") if r.strip()]
    else:
        candidates = list(cfg.get("regions", []))
    regions = [r for r in candidates if r != "global"]
    return regions or [args.region]


def confirm_account(session, cfg: dict, args, regions: list) -> str:
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]

    blocklist = {str(b) for b in cfg.get("blocklist", [])}
    if account_id in blocklist:
        log(f"Account {account_id} is in nuke-config.yml's blocklist -- refusing to run.")
        sys.exit(1)

    alias = "?"
    try:
        aliases = session.client("iam").list_account_aliases().get("AccountAliases", [])
        if aliases:
            alias = aliases[0]
    except Exception:
        pass

    print("=" * 72)
    print(f"Account : {account_id} ({alias})")
    print(f"Profile : {args.profile}")
    print(f"Regions : {', '.join(regions)}")
    print(f"VPCs    : {'all VPCs, including default' if args.include_default_vpc else 'non-default VPCs only'}"
          + (f" (restricted to: {', '.join(args.vpc_id)})" if args.vpc_id else ""))
    print(f"Mode    : {'DRY RUN -- no AWS calls will mutate anything' if args.dry_run else 'LIVE -- resources WILL be deleted, no undo'}")
    print("=" * 72)

    if args.dry_run or args.yes:
        return account_id

    typed = input(f"Type the account ID ({account_id}) to confirm: ").strip()
    if typed != account_id:
        log("Confirmation did not match -- aborting.")
        sys.exit(1)
    return account_id


def main():
    parser = argparse.ArgumentParser(
        description="Detach and delete everything blocking VPC deletion (NAT gateways, ENIs, endpoints, "
                    "peering, load balancers, RDS/ElastiCache/EFS, ECS/EKS, VPN gateways, ...) then delete "
                    "the VPCs themselves. Complements aws-nuke, which routinely fails on this exact "
                    "dependency graph. See the module docstring for full details.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", required=True, help="AWS CLI profile to use (e.g. default, senatwo).")
    parser.add_argument("--region", default="us-east-1",
                        help="Fallback single region if --regions and nuke-config.yml regions are both unset.")
    parser.add_argument("--regions", default=None,
                        help="Comma-separated regions to nuke. Defaults to the `regions:` list in "
                             "--nuke-config (minus 'global').")
    parser.add_argument("--nuke-config", type=Path, default=DEFAULT_NUKE_CONFIG,
                        help=f"Path to nuke-config.yml, read for regions + account blocklist "
                             f"(default: {DEFAULT_NUKE_CONFIG}).")
    parser.add_argument("--vpc-id", action="append", default=None,
                        help="Restrict phase 2 (VPC network teardown) to this VPC ID. Repeatable. "
                             "Phase 1's account-wide compute/service teardown always runs regardless.")
    parser.add_argument("--include-default-vpc", action="store_true",
                        help="Also tear down each region's default VPC (skipped by default).")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done -- makes no AWS calls that mutate anything.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the interactive account-ID confirmation prompt.")
    args = parser.parse_args()

    cfg = load_nuke_config(args.nuke_config)
    regions = resolve_regions(args, cfg)

    session = boto3.Session(profile_name=args.profile)
    confirm_account(session, cfg, args, regions)

    actions = Actions(dry_run=args.dry_run)
    for region in regions:
        nuke_region(session, region, actions, args)

    log("Done. Re-run this script (and/or aws-nuke) if any region still reports leftover "
        "AWS-managed ENIs or a VPC that failed to delete -- most of those clear within a few minutes.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
