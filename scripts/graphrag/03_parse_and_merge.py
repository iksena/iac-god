import os
import json
from bs4 import BeautifulSoup

SPEC_FILE      = "cfn_resource_spec.json"
HTML_DIR       = "scraped_html"
OUTPUT_KG_FILE = "cfn_knowledge_graph.json"


def parse_html_for_resource(html_path: str, resource_name: str) -> dict:
    """Extract resource description, per-property descriptions, and YAML examples.

    Per-property descriptions are keyed by lowercased property name so they
    can be matched case-insensitively against the spec property names in
    build_knowledge_graph_data().

    AWS docs structure used:
      - Resource description: first 3 <p> tags inside <div id="main-col-body">
      - Property descriptions: <dt id="cfn-<ns>-<res>-<prop>"> → next <dd> sibling
      - YAML examples: <code> blocks containing "Type: AWS::" without JSON braces
    """
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    result = {"description": "", "property_descriptions": {}, "examples": []}

    # ── Resource-level description ─────────────────────────────────────────
    main_body = soup.find("div", id="main-col-body")
    if main_body:
        paras = main_body.find_all("p", recursive=False)
        result["description"] = " ".join(p.get_text(strip=True) for p in paras[:3])

    # ── Per-property descriptions ──────────────────────────────────────────
    # AWS docs use <dt id="cfn-<namespace>-<resource>-<property>"> anchors.
    # e.g. for AWS::Timestream::Table / RetentionProperties:
    #   <dt id="cfn-timestream-table-retentionproperties">
    # We strip the resource prefix to get the lowercased property name.
    prefix = "cfn-" + resource_name.lower().replace("::", "-") + "-"
    for dt in soup.find_all("dt"):
        anchor = dt.get("id", "")
        if anchor.startswith(prefix):
            raw_prop = anchor[len(prefix):]          # e.g. "retentionproperties"
            dd = dt.find_next_sibling("dd")
            if dd:
                desc = " ".join(p.get_text(strip=True) for p in dd.find_all("p")[:2])
                if desc:
                    result["property_descriptions"][raw_prop] = desc

    # ── YAML examples ──────────────────────────────────────────────────────
    for block in soup.find_all("code"):
        text = block.get_text()
        if "Type: AWS::" in text and "{" not in text:
            result["examples"].append(text.strip())

    return result


def build_knowledge_graph_data():
    """Build cfn_knowledge_graph.json from the spec as the authoritative source.

    Strategy:
      1. Iterate every resource in cfn_resource_spec.json (source of truth).
      2. For each resource, enrich with scraped HTML semantics when available.
         Resources without HTML are still included with empty description/examples.
      3. Merge per-property HTML descriptions into spec property entries using
         case-insensitive property name matching.

    This guarantees cfn_knowledge_graph.json always contains all resources
    from the spec, regardless of scrape completeness.
    """
    print("Loading spec...")
    with open(SPEC_FILE) as f:
        spec = json.load(f)

    # Pre-index scraped HTML files: lowercased resource name → filepath.
    # e.g. "aws::timestream::table" → "scraped_html/AWS_Timestream_Table.html"
    html_index: dict[str, str] = {}
    if os.path.isdir(HTML_DIR):
        for fname in os.listdir(HTML_DIR):
            if fname.endswith(".html"):
                rname = fname.replace(".html", "").replace("_", "::").lower()
                html_index[rname] = os.path.join(HTML_DIR, fname)

    kg_data: dict = {}
    resource_types = spec.get("ResourceTypes", {})
    scraped_count = 0
    prop_desc_count = 0

    for resource_name, structure in resource_types.items():
        html_path = html_index.get(resource_name.lower())

        if html_path:
            semantics = parse_html_for_resource(html_path, resource_name)
            scraped_count += 1
        else:
            semantics = {"description": "", "property_descriptions": {}, "examples": []}

        # Merge per-property HTML descriptions back into spec property entries.
        # Match by lowercasing both sides so casing differences don't break lookups.
        # e.g. spec key "RetentionProperties" matches HTML anchor "retentionproperties".
        prop_desc_index = semantics["property_descriptions"]  # already lowercased
        merged_properties: dict = {}

        for prop_name, prop_data in structure.get("Properties", {}).items():
            enriched = dict(prop_data)
            html_desc = prop_desc_index.get(prop_name.lower(), "")
            if html_desc:
                enriched["Description"] = html_desc
                prop_desc_count += 1
            merged_properties[prop_name] = enriched

        kg_data[resource_name] = {
            "name":        resource_name,
            "description": semantics["description"],
            "properties":  merged_properties,
            "examples":    semantics["examples"],
        }

    with open(OUTPUT_KG_FILE, "w") as f:
        json.dump(kg_data, f, indent=4)

    spec_only = len(kg_data) - scraped_count
    print(f"Knowledge Graph built:")
    print(f"  Total resources    : {len(kg_data)}")
    print(f"  With scraped HTML  : {scraped_count}")
    print(f"  Spec-only (no HTML): {spec_only}")
    print(f"  Property descriptions merged from HTML: {prop_desc_count}")
    print(f"  Saved to           : {OUTPUT_KG_FILE}")


if __name__ == "__main__":
    build_knowledge_graph_data()
