# scripts/build_cfn_graph.py
import json
import pathlib
import pickle
import urllib.request
import networkx as nx

# Official AWS CloudFormation Resource Specification — stable public URL
# ~15MB JSON, all 1,100+ resource types + property types for us-east-1
CFN_SPEC_URL = (
    "https://d1uauaxba7bl26.cloudfront.net/latest/gzip/CloudFormationResourceSpecification.json"
)
SPEC_CACHE   = pathlib.Path("data/cfn_spec.json")
OUTPUT_PATH  = pathlib.Path("data/cfn_graph.pkl")


def load_cfn_schemas() -> dict:
    """Download (and cache) the AWS CloudFormation Resource Specification."""
    if SPEC_CACHE.exists():
        print(f"Loading cached spec from {SPEC_CACHE}")
        with open(SPEC_CACHE) as f:
            return json.load(f)

    print("Downloading CloudFormation Resource Specification from AWS...")
    SPEC_CACHE.parent.mkdir(exist_ok=True)

    # The URL serves gzip — urllib handles Content-Encoding automatically
    req = urllib.request.Request(CFN_SPEC_URL, headers={"Accept-Encoding": "gzip"})
    import gzip, io
    with urllib.request.urlopen(CFN_SPEC_URL) as resp:
        raw = resp.read()

    # Try decompressing — AWS serves it gzip-encoded
    try:
        spec = json.loads(gzip.decompress(raw))
    except Exception:
        spec = json.loads(raw)  # already plain JSON

    with open(SPEC_CACHE, "w") as f:
        json.dump(spec, f)
    print(f"Spec cached to {SPEC_CACHE}")
    return spec

def build_graph_from_cloudspec(spec: dict):
    G = nx.DiGraph()
    resource_types = spec.get("ResourceTypes", {})
    property_types = spec.get("PropertyTypes", {})

    for rtype, rdata in resource_types.items():
        G.add_node(rtype, ntype="ResourceType", name=rtype,
                   docs=rdata.get("Documentation", ""))
        for prop_name, prop_data in rdata.get("Properties", {}).items():
            prop_id = f"{rtype}/{prop_name}"
            G.add_node(prop_id, ntype="Property", name=prop_name,
                       required=prop_data.get("Required", False),
                       type=prop_data.get("Type", ""),
                       update_type=prop_data.get("UpdateType", ""),
                       primitive_type=prop_data.get("PrimitiveType", ""))
            G.add_edge(rtype, prop_id, etype="HAS_PROPERTY")

    for ptype, pdata in property_types.items():
        pt_node = f"PropertyType/{ptype}"
        G.add_node(pt_node, ntype="PropertyType", name=ptype)
        for prop_name, prop_data in pdata.get("Properties", {}).items():
            prop_id = f"{pt_node}/{prop_name}"
            G.add_node(prop_id, ntype="Property", name=prop_name,
                       required=prop_data.get("Required", False),
                       type=prop_data.get("Type", ""),
                       primitive_type=prop_data.get("PrimitiveType", ""))
            G.add_edge(pt_node, prop_id, etype="HAS_PROPERTY")
        resource_prefix = ptype.rsplit(".", 1)[0]
        if resource_prefix in resource_types:
            G.add_edge(resource_prefix, pt_node, etype="HAS_PROPERTY_TYPE")

    return G


def main():
    spec = load_cfn_schemas()
    print(f"Spec version: {spec.get('ResourceSpecificationVersion', 'unknown')}")

    G = build_graph_from_cloudspec(spec)

    n_resources = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "ResourceType")
    n_props     = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "Property")
    n_ptypes    = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "PropertyType")

    print(f"Graph built successfully:")
    print(f"  Resources    : {n_resources}")
    print(f"  PropertyTypes: {n_ptypes}")
    print(f"  Properties   : {n_props}")
    print(f"  Nodes        : {G.number_of_nodes()}")
    print(f"  Edges        : {G.number_of_edges()}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to     : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()