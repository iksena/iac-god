# 06_execute_g_retrieval.py
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from neo4j_client import query_knowledge_graph, format_prompt_from_neo4j_result 

CHROMA_DB_DIR = "../cfn_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def execute_g_retrieval(user_query: str):
    print(f"--- Starting G-Retrieval for Query: '{user_query}' ---\n")
    
    # ==========================================
    # STAGE 1: Semantic Search (Vector Database)
    # ==========================================
    print("Stage 1: Performing Semantic Search in ChromaDB Docker...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Strictly use the HTTP Client (do NOT use persist_directory here)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="cloudformation_docs", # Must match ingestion exactly
        embedding_function=embeddings
    )
    
    # Retrieve top 5 most similar chunks
    top_chunks = vectorstore.similarity_search(user_query, k=5)
    
    if not top_chunks:
        print("WARNING: Vector search returned no results. Check database connection or collection name.")
        return
        
    # Extract unique resource names from the metadata
    identified_resources = set()
    for chunk in top_chunks:
        if "resource_name" in chunk.metadata:
            identified_resources.add(chunk.metadata["resource_name"])
        
    print(f"Identified entry points: {list(identified_resources)}\n")
    
    # ==========================================
    # STAGE 2: Graph Traversal (Neo4j)
    # ==========================================
    print("Stage 2: Performing Graph Traversal in Neo4j...")
    final_llm_context = ""
    
    for resource in identified_resources:
        print(f"Extracting minimal subgraph for {resource}...")
        
        # Query Neo4j for the exact structural schema of this resource
        resource_data = query_knowledge_graph(resource_name=resource)
        
        # Format it into a clean string for the LLM
        formatted_context = format_prompt_from_neo4j_result(resource_data)
        
        final_llm_context += formatted_context + "\n\n"
        
    # ==========================================
    # FINAL: Combine with User Prompt
    # ==========================================
    final_prompt = f"""You are an expert AWS CloudFormation architect.
Use the following official documentation context to answer the user query.

CONTEXT:
{final_llm_context}

USER QUERY: {user_query}

Generate the appropriate CloudFormation YAML code.
"""
    return final_prompt

if __name__ == "__main__":
    test_query = "Create an Amazon S3 bucket that has versioning enabled."
    llm_prompt = execute_g_retrieval(test_query)
    
    print("\n--- Final Output to send to LLM ---")
    print(llm_prompt)