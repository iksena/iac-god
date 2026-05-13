#!/usr/bin/env python3
"""
01_load_trivy_csv.py

Stage 1a – Parse trivy_enriched.csv into a clean security_checks.json.

The CSV has multiple rows per logical check:
  - Primary rows: check_id is set, contain check metadata + Rego source
  - Supplement rows: check_id is empty, contain extra YAML fixture data

This script:
  1. Reads all rows, groups by check_id (skipping blank-id supplement rows)
  2. Resolves `service` → AWS CloudFormation resource type prefix via SERVICE_TO_CFN_PREFIX
  3. Emits data/security_checks.json keyed by check_id

Output path: data/security_checks.json
"""

import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "trivy_enriched.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "security_checks.json"

# ---------------------------------------------------------------------------
# Service → CloudFormation resource prefix mapping
# Maps Trivy service names to the AWS:: prefix used in the CFN graph.
# ---------------------------------------------------------------------------
SERVICE_TO_CFN_PREFIX = {
    "accessanalyzer": "AWS::AccessAnalyzer::",
    "apigateway": "AWS::ApiGateway::",
    "athena": "AWS::Athena::",
    "cloudfront": "AWS::CloudFront::",
    "cloudtrail": "AWS::CloudTrail::",
    "cloudwatch": "AWS::CloudWatch::",
    "codebuild": "AWS::CodeBuild::",
    "config": "AWS::Config::",
    "documentdb": "AWS::DocDB::",
    "dynamodb": "AWS::DynamoDB::",
    "ec2": "AWS::EC2::",
    "ecr": "AWS::ECR::",
    "ecs": "AWS::ECS::",
    "efs": "AWS::EFS::",
    "eks": "AWS::EKS::",
    "elasticache": "AWS::ElastiCache::",
    "elasticsearch": "AWS::Elasticsearch::",
    "elb": "AWS::ElasticLoadBalancing::",
    "elbv2": "AWS::ElasticLoadBalancingV2::",
    "emr": "AWS::EMR::",
    "glacier": "AWS::Glacier::",
    "glue": "AWS::Glue::",
    "iam": "AWS::IAM::",
    "kinesis": "AWS::Kinesis::",
    "kms": "AWS::KMS::",
    "lambda": "AWS::Lambda::",
    "mq": "AWS::AmazonMQ::",
    "msk": "AWS::MSK::",
    "neptune": "AWS::Neptune::",
    "rds": "AWS::RDS::",
    "redshift": "AWS::Redshift::",
    "s3": "AWS::S3::",
    "sagemaker": "AWS::SageMaker::",
    "secretsmanager": "AWS::SecretsManager::",
    "sns": "AWS::SNS::",
    "sqs": "AWS::SQS::",
    "ssm": "AWS::SSM::",
    "vpc": "AWS::EC2::",  # VPC resources live under EC2 in CFN
    "waf": "AWS::WAF::",
    "workspaces": "AWS::WorkSpaces::",
}


def clean_list_field(raw: str) -> list[str]:
    """Parse fields that look like Python list literals: "['a', 'b']" → ['a', 'b']."""
    if not raw or raw.strip() == "":
        return []
    # Strip outer brackets
    inner = raw.strip().lstrip("[").rstrip("]")
    # Split on ', ' boundaries between quoted items
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    return [a or b for a, b in items if a or b]


def clean_links_field(raw: str) -> list[str]:
    """Parse link fields that may be pipe-separated or list literals."""
    if not raw or raw.strip() == "":
        return []
    if raw.startswith("["):
        return clean_list_field(raw)
    return [u.strip() for u in raw.split("|") if u.strip()]


def load_csv(csv_path: Path) -> dict:
    checks: dict = {}

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            check_id = row.get("check_id", "").strip()
            if not check_id:
                # Supplement row – skip for now; enrichment handled in stage 2
                continue

            service = row.get("service", "").strip().lower()
            cfn_prefix = SERVICE_TO_CFN_PREFIX.get(service, "")

            checks[check_id] = {
                "check_id": check_id,
                "check_name": row.get("check_name", "").strip(),
                "severity": row.get("severity", "").strip().upper(),
                "short_code": row.get("short_code", "").strip(),
                "description": row.get("description", "").strip(),
                "service": service,
                "cfn_resource_prefix": cfn_prefix,
                "framework": row.get("framework", "").strip(),
                "source_file_url": row.get("source_file_url", "").strip(),
                "source_code": row.get("source_code", "").strip(),
                "avd_url": row.get("avd_url", "").strip(),
                "title": row.get("title", "").strip(),
                "impact": row.get("impact", "").strip(),
                "remediation_cfn": clean_list_field(row.get("remediation_cfn", "")),
                "remediation_tf": clean_list_field(row.get("remediation_tf", "")),
                "cfn_good_example": row.get("cfn_good_example", "").strip(),
                "tf_good_example": row.get("tf_good_example", "").strip(),
                "links": clean_links_field(row.get("links", "")),
            }

    return checks


def main():
    print(f"Reading CSV from: {CSV_PATH}")
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    checks = load_csv(CSV_PATH)
    print(f"Loaded {len(checks)} unique security checks.")

    # Show severity breakdown
    from collections import Counter
    severity_counts = Counter(c["severity"] for c in checks.values())
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev}: {count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(checks)} checks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
