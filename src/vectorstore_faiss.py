# src/vectorstore_faiss.py
import faiss
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from embeddings import get_embedding_gemini, get_embedding_local
from sentence_transformers import SentenceTransformer

def build_faiss(csv_in="data/disease_kb.csv", index_out="data/faiss.index", meta_out="data/meta.pkl", use_local=True):
    df = pd.read_csv(csv_in)
    texts = df['combined'].fillna("").tolist()

    if use_local:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.array([get_embedding_local(model, t) for t in tqdm(texts)])
    else:
        embeddings = np.array([get_embedding_gemini(t) for t in tqdm(texts)])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # gunakan inner product (simpan normalized vectors)
    # normalisasi agar IP ~ cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_out)

    meta = df.to_dict(orient='records')
    with open(meta_out, "wb") as f:
        pickle.dump(meta, f)

    print("FAISS index and meta saved.")

if __name__ == "__main__":
    build_faiss()
