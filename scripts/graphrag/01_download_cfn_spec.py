import urllib.request
import json
import os
import gzip

# AWS publishes the official schema here
SPEC_URL = "https://d1uauaxba7bl26.cloudfront.net/latest/gzip/CloudFormationResourceSpecification.json"
OUTPUT_FILE = "cfn_resource_spec.json"

def download_cfn_spec():
    print(f"Downloading CloudFormation Resource Specification...")
    response = urllib.request.urlopen(SPEC_URL)
    raw = response.read()

    # AWS endpoint returns gzip-compressed content at this URL.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    data = json.loads(raw.decode('utf-8'))
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Total Resources Found: {len(data.get('ResourceTypes', {}))}")
    print(f"Total Property Types (Nested Blocks) Found: {len(data.get('PropertyTypes', {}))}")

if __name__ == "__main__":
    download_cfn_spec()