import os
import json
from bs4 import BeautifulSoup

SPEC_FILE = "cfn_resource_spec.json"
HTML_DIR = "scraped_html"
OUTPUT_KG_FILE = "cfn_knowledge_graph.json"

def parse_html_for_semantics(html_path):
    """Extracts description and YAML examples from the AWS HTML doc."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    semantics = {
        "description": "",
        "examples": []
    }

    # 1. Extract main description (usually the first few paragraphs in the main body)
    main_body = soup.find('div', id='main-col-body')
    if main_body:
        paragraphs = main_body.find_all('p', recursive=False)
        semantics["description"] = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])

    # 2. Extract YAML Examples
    # AWS docs typically use <code class="yaml"> or similar for YAML blocks
    code_blocks = soup.find_all('code')
    for block in code_blocks:
        # Simple heuristic: if it contains 'Type: AWS::', it's likely a CFN YAML block
        text = block.get_text()
        if "Type: AWS::" in text and "{" not in text: # Avoid JSON
            semantics["examples"].append(text.strip())

    return semantics

def build_knowledge_graph_data():
    print("Merging structural JSON with semantic HTML data...")
    with open(SPEC_FILE, 'r') as f:
        spec = json.load(f)

    kg_data = {}
    
    for filename in os.listdir(HTML_DIR):
        if not filename.endswith(".html"):
            continue
            
        resource_name = filename.replace(".html", "").replace("_", "::")
        html_path = os.path.join(HTML_DIR, filename)
        
        # Get structure from spec
        structure = spec.get("ResourceTypes", {}).get(resource_name, {})
        
        # Get semantics from HTML
        semantics = parse_html_for_semantics(html_path)
        
        # Merge
        kg_data[resource_name] = {
            "name": resource_name,
            "description": semantics["description"],
            "properties": structure.get("Properties", {}),
            "examples": semantics["examples"]
        }
        print(f"Processed {resource_name}: Found {len(semantics['examples'])} examples.")

    with open(OUTPUT_KG_FILE, 'w') as f:
        json.dump(kg_data, f, indent=4)
    print(f"Knowledge Graph data saved to {OUTPUT_KG_FILE}")

if __name__ == "__main__":
    build_knowledge_graph_data()