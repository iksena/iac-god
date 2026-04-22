# debug_chroma.py
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

def raw_chroma_search():
    print("1. Connecting to Docker ChromaDB...")
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_collection("cloudformation_docs")
    
    print("2. Embedding the test query...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_text = "Create an Amazon S3 bucket that has versioning enabled."
    query_vector = embeddings.embed_query(query_text)
    
    print("3. Executing Vector Search...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )
    
    print("\n--- Search Results ---")
    for i in range(len(results['documents'][0])):
        print(f"Match {i+1}:")
        print(f"Resource: {results['metadatas'][0][i]['resource_name']}")
        print(f"Distance Score: {results['distances'][0][i]}")
        print("-" * 20)

def check_chroma():
    print("Connecting to ChromaDB on localhost:8000...")
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # List all collections
        collections = client.list_collections()
        print(f"\nFound {len(collections)} collections:")
        
        for col in collections:
            collection = client.get_collection(col.name)
            count = collection.count()
            print(f"- Collection '{col.name}' has {count} document chunks.")
            
            if count > 0:
                # Print a sample to ensure metadata exists
                sample = collection.peek(1)
                print(f"  Sample Metadata: {sample['metadatas'][0]}")
                
    except Exception as e:
        print(f"Failed to connect or query ChromaDB: {e}")

if __name__ == "__main__":
    # check_chroma()
    raw_chroma_search()