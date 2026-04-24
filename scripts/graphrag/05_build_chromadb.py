# 05_build_chromadb.py
import json
import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "../cfn-chroma-docker"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def build_vector_db():
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    with open("cfn_knowledge_graph.json", "r") as f:
        kg_data = json.load(f)
        
    documents = []
    print("Chunking CloudFormation data...")
    
    for res_name, res_data in kg_data.items():
        res_description = res_data.get("description", "")

        for prop_name, prop_details in res_data.get("properties", {}).items():
            prop_type = prop_details.get("Type", prop_details.get("PrimitiveType", "Unknown"))
            required = prop_details.get("Required", False)
            update_type = prop_details.get("UpdateType", "Unknown")
            doc_url = prop_details.get("Documentation", "")
            description = prop_details.get("Description", "")

            text_content = (
                f"Resource: {res_name}\n"
                f"Resource Description: {res_description}\n"
                f"Property: {prop_name}\n"
                f"Property ID: {res_name}.{prop_name}\n"
                f"Type: {prop_type}\n"
                f"Required: {required}\n"
                f"Update Type: {update_type}\n"
                f"Description: {description}\n"
                f"Documentation: {doc_url}\n"
            )

            doc = Document(
                page_content=text_content,
                metadata={
                    "resource_name": res_name,
                    "property_name": prop_name,
                    "property_id": f"{res_name}.{prop_name}",  # aligns with Neo4j node ID
                    "required": required,
                    "update_type": update_type,
                    "type": prop_type,
                }
            )
            documents.append(doc)
    
    for i, example_code in enumerate(res_data.get("examples", [])):
        doc = Document(
            page_content=f"CloudFormation example for {res_name}:\n{example_code}",
            metadata={
                "resource_name": res_name,
                "chunk_type": "example",
                "example_index": i,
                "property_id": None,   # can be enriched later if example is tagged
            }
        )
        documents.append(doc)

    print(f"Created {len(documents)} document chunks. Ingesting into ChromaDB Docker container...")
    
    # Connect to Docker ChromaDB
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    
    # Store in Docker ChromaDB under a specific collection name
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        client=chroma_client,
        collection_name="cfn_schema_properties"
    )
    print("Vector database successfully built inside Docker!")

if __name__ == "__main__":
    build_vector_db()