# 04_import_cfn_to_neo4j.py
import json
from neo4j import GraphDatabase
import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def clear_database(session):
    print("Clearing existing database...")
    session.run("MATCH (n) DETACH DELETE n")

def import_cfn_graph(session, data):
    print("Importing CloudFormation Knowledge Graph...")
    for res_name, res_data in data.items():
        # Create Resource Node
        session.run("""
            MERGE (r:Resource {name: $name})
            SET r.description = $desc
        """, name=res_name, desc=res_data.get("description", ""))

        # Create Properties and Relationships
        for prop_name, prop_details in res_data.get("properties", {}).items():
            required = prop_details.get("Required", False)
            prop_type = prop_details.get("Type", prop_details.get("PrimitiveType", "Unknown"))
            
            # Determine if it's a primitive property or a complex nested type
            if "ItemType" in prop_details or prop_type not in ["String", "Integer", "Boolean", "Timestamp", "Double", "Long", "Json"]:
                node_label = "NestedType"
                rel_type = "HAS_NESTED_TYPE"
            else:
                node_label = "Property"
                rel_type = "HAS_PROPERTY"

            session.run(f"""
                MATCH (r:Resource {{name: $res_name}})
                MERGE (p:{node_label} {{id: $prop_id}})
                SET p.name        = $prop_name,
                    p.type        = $prop_type,
                    p.required    = $req,
                    p.update_type = $update_type,
                    p.doc_url     = $doc_url,
                    p.description = $description,
                    p.primitive_item_type = $primitive_item_type,
                    p.duplicates_allowed  = $duplicates_allowed
                MERGE (r)-[:{rel_type}]->(p)
            """,
                res_name=res_name,
                prop_id=f"{res_name}.{prop_name}",
                prop_name=prop_name,
                prop_type=prop_type,
                req=required,
                update_type=prop_details.get("UpdateType", "Unknown"),
                doc_url=prop_details.get("Documentation", ""),
                description=prop_details.get("Description", ""),
                primitive_item_type=prop_details.get("PrimitiveItemType", ""),
                duplicates_allowed=prop_details.get("DuplicatesAllowed", False),
            )

            # After creating the NestedType node, add this block:
            referenced_type = prop_details.get("ItemType") or prop_details.get("Type")
            if referenced_type and referenced_type not in ["String", "Integer", "Boolean", "Timestamp", "Double", "Long", "Json", "List", "Map"]:
                ref_id = f"{res_name}.{referenced_type}"
                session.run("""
                    MERGE (t:NestedType {id: $ref_id})
                    SET t.name = $ref_name
                    WITH t
                    MATCH (p:NestedType {id: $prop_id})
                    MERGE (p)-[:REFERENCES_TYPE]->(t)
                """, ref_id=ref_id, ref_name=referenced_type, prop_id=f"{res_name}.{prop_name}")

        # Create Example Nodes
        for i, example_code in enumerate(res_data.get("examples", [])):
            session.run("""
                MATCH (r:Resource {name: $res_name})
                MERGE (e:Example {id: $ex_id})
                SET e.code = $code, e.index = $index
                MERGE (r)-[:HAS_EXAMPLE]->(e)
            """, res_name=res_name, ex_id=f"{res_name}_ex_{i}", code=example_code, index=i)

def create_indexes(session):
    print("Creating indexes...")
    session.run("CREATE INDEX resource_name IF NOT EXISTS FOR (r:Resource) ON (r.name)")
    session.run("CREATE INDEX property_id IF NOT EXISTS FOR (p:Property) ON (p.id)")
    session.run("CREATE INDEX nested_type_id IF NOT EXISTS FOR (t:NestedType) ON (t.id)")

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with open("cfn_knowledge_graph.json", "r") as f:
        kg_data = json.load(f)
    
    with driver.session() as session:
        create_indexes(session)
        clear_database(session)
        import_cfn_graph(session, kg_data)
    driver.close()
    print("Import complete!")