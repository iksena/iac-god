import requests
import json
from neo4j import GraphDatabase

# Download latest CFN Specification
CFN_URL = "https://d1uauaxba7bl26.cloudfront.net/latest/gzip/CloudFormationResourceSpecification.json"
cfn_spec = requests.get(CFN_URL).json()

class CFNGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def build_graph(self, spec):
        with self.driver.session() as session:
            # 1. Ingest Resources and their primitive properties
            for res_name, res_data in spec.get("ResourceTypes", {}).items():
                session.run("""
                    MERGE (r:Resource {name: $res_name})
                    SET r.docs = $docs
                """, res_name=res_name, docs=res_data.get("Documentation", ""))
                
                self._ingest_properties(session, res_name, "Resource", res_data.get("Properties", {}))

            # 2. Ingest Nested PropertyTypes (e.g., AWS::S3::Bucket.CorsRule)
            for prop_type_name, prop_data in spec.get("PropertyTypes", {}).items():
                session.run("""
                    MERGE (pt:PropertyType {name: $pt_name})
                """, pt_name=prop_type_name)
                
                self._ingest_properties(session, prop_type_name, "PropertyType", prop_data.get("Properties", {}))

    def _ingest_properties(self, session, parent_name, parent_label, properties):
        for prop_name, prop_details in properties.items():
            required = prop_details.get("Required", False)
            primitive_type = prop_details.get("PrimitiveType", "Complex")
            item_type = prop_details.get("ItemType", prop_details.get("Type", ""))

            # Create Property Node and link it to Parent
            session.run(f"""
                MATCH (parent:{parent_label} {{name: $parent_name}})
                MERGE (p:Property {{id: $prop_id}})
                SET p.name = $prop_name, p.type = $type, p.required = $required
                MERGE (parent)-[:HAS_PROPERTY]->(p)
            """, parent_name=parent_name, prop_id=f"{parent_name}/{prop_name}", 
                 prop_name=prop_name, type=primitive_type, required=required)

            # If it references a nested PropertyType, link them
            if item_type and item_type != "List" and item_type != "Map":
                # ItemTypes in CFN can be just "CorsRule" which implicitly means "AWS::S3::Bucket.CorsRule"
                full_type_name = item_type if "::" in item_type else f"{parent_name.split('.')[0]}.{item_type}"
                session.run("""
                    MATCH (p:Property {id: $prop_id})
                    MERGE (pt:PropertyType {name: $type_name})
                    MERGE (p)-[:USES_TYPE]->(pt)
                """, prop_id=f"{parent_name}/{prop_name}", type_name=full_type_name)

# Execute Build
builder = CFNGraphBuilder("bolt://localhost:7687", "neo4j", "password")
builder.build_graph(cfn_spec)
builder.close()
print("CFN Knowledge Graph Built Successfully!")