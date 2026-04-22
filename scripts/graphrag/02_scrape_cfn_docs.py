import json
import os
import requests
import time

SPEC_FILE = "cfn_resource_spec.json"
OUTPUT_DIR = "scraped_html"

def generate_doc_url(resource_name):
    """
    Converts 'AWS::EC2::Instance' to 'https://docs.aws.amazon.com/.../aws-resource-ec2-instance.html'
    """
    formatted_name = resource_name.lower().replace("::", "-")
    return f"https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-{formatted_name.split('aws-')[-1]}.html"

def scrape_docs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(SPEC_FILE, 'r') as f:
        spec = json.load(f)
    
    resources = list(spec.get("ResourceTypes", {}).keys())
    
    # For testing, let's limit to 5 resources. Remove this slice to run on all ~1000+ resources.
    target_resources = resources 

    for res in target_resources:
        url = generate_doc_url(res)
        file_name = res.replace("::", "_") + ".html"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        if os.path.exists(file_path):
            print(f"Skipping {res}, already downloaded.")
            continue
            
        print(f"Fetching {res} -> {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                print(f"Failed to fetch {res}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            
        time.sleep(0.5) # Polite scraping delay

if __name__ == "__main__":
    scrape_docs()