# scripts/build_cfn_rag_index.py
"""
Build BM25 + FAISS indexes from the cfn_graph.pkl for RAG.
Run: python scripts/build_cfn_rag_index.py
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).resolve().parents[1] / "data"
GRAPH_PATH   = DATA / "cfn_graph.pkl"
CORPUS_PATH  = DATA / "cfn_rag_corpus.jsonl"
FAISS_PATH   = DATA / "cfn_rag_faiss.index"
BM25_PATH    = DATA / "cfn_rag_bm25.pkl"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

def _load_graph():
    with GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj

def build_corpus(G) -> list[dict]:
    """
    One doc per resource node + one doc per property node.
    The text field is what gets embedded/indexed.
    """
    docs = []
    for node_id, nd in G.nodes(data=True):
        ntype = nd.get("ntype", "")

        if ntype in ("ResourceType", "Resource"):
            text = (
                f"{node_id} CloudFormation resource schema. "
                f"Type: {node_id}. "
                f"{nd.get('docs', '')}" # Include official AWS docs if available
            )
            docs.append({
                "doc_id": node_id,
                "resource_type": node_id,
                "property_name": "",
                "text": text,
            })

        elif ntype == "Property":
            parts = str(node_id).rsplit("/", 1)
            rtype = parts[0] if len(parts) == 2 else node_id
            prop  = parts[1] if len(parts) == 2 else nd.get("name", "")
            
            prim  = nd.get("primitive_type") or nd.get("type") or "Any"
            req   = "required" if nd.get("required") else "optional"
            upd   = nd.get("update_type", "")
            
            text  = (
                f"{rtype} property {prop}. "
                f"Type: {prim}. {req.capitalize()} property. "
                + (f"UpdateType: {upd}. " if upd else "")
                + f"Part of {rtype} CloudFormation resource."
            )
            
            docs.append({
                "doc_id": node_id,
                "resource_type": rtype,
                "property_name": prop,
                "text": text,
            })

    return docs

def main():
    print("Loading graph...")
    G = _load_graph()

    print("Building corpus...")
    docs = build_corpus(G)
    print(f"  {len(docs)} documents generated.")

    with CORPUS_PATH.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    print(f"  Saved {CORPUS_PATH}")

    texts   = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    print("Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with BM25_PATH.open("wb") as fh:
        pickle.dump((bm25, doc_ids), fh)
    print(f"  Saved {BM25_PATH}")

    print("Building FAISS index...")
    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode(texts, batch_size=256,
                           normalize_embeddings=True,
                           show_progress_bar=True).astype("float32")
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  Saved {FAISS_PATH}  ({index.ntotal} vectors, dim={dim})")

if __name__ == "__main__":
    main()