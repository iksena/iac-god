# 05_build_chromadb.py
import json
import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DB_DIR = "../cfn_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def build_vector_db():
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    with open("cfn_knowledge_graph.json", "r") as f:
        kg_data = json.load(f)
        
    documents = []
    print("Chunking CloudFormation data...")
    
    for res_name, res_data in kg_data.items():
        # Create a semantic text chunk for the resource
        text_content = f"Resource: {res_name}\n"
        text_content += f"Description: {res_data.get('description', '')}\n"
        text_content += "Properties:\n"
        
        # Add basic property descriptions to the text chunk
        for prop_name, prop_details in res_data.get("properties", {}).items():
            text_content += f"- {prop_name}: {prop_details.get('Documentation', 'No description available')}\n"

        # Create a LangChain Document with critical metadata
        doc = Document(
            page_content=text_content,
            metadata={"resource_name": res_name} # This metadata is critical for Stage 2
        )
        documents.append(doc)

    print(f"Created {len(documents)} document chunks. Ingesting into ChromaDB Docker container...")
    
    # Connect to Docker ChromaDB
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    
    # Store in Docker ChromaDB under a specific collection name
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        client=chroma_client,
        collection_name="cloudformation_docs"
    )
    print("Vector database successfully built inside Docker!")

if __name__ == "__main__":
    build_vector_db()