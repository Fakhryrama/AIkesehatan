# src/embeddings.py
import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ============================
#   Load dataset disease_kb
# ============================

DATA_PATH = "../data/disease_kb.csv"
OUTPUT_INDEX = "../data/faiss.index"
OUTPUT_META = "../data/meta.pkl"

def load_kb():
    df = pd.read_csv(DATA_PATH)
    # Pastikan ada kolom-kolom ini:
    # Disease, Symptoms, Description, Precautions
    df = df.fillna("")
    docs = []
    for _, row in df.iterrows():
        text = f"{row['Disease']} | {row['Symptoms']} | {row['Description']}"
        docs.append(text)
    return df, docs

# ============================
#   Embedding model (multilingual)
# ============================

def load_model():
    print("Loading multilingual embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model

def build_embeddings(model, docs):
    print("Encoding documents...")
    vectors = model.encode(docs, show_progress_bar=True)
    vectors = np.array(vectors).astype("float32")

    # Normalize
    faiss.normalize_L2(vectors)
    return vectors

# ============================
#   Save FAISS index + metadata
# ============================

def save_faiss(vectors, df):
    dim = vectors.shape[1]

    print(f"Building FAISS index with dim={dim} ...")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    print("Saving index...")
    faiss.write_index(index, OUTPUT_INDEX)

    print("Saving metadata...")
    meta = df.to_dict("records")
    with open(OUTPUT_META, "wb") as f:
        pickle.dump(meta, f)

    print("DONE: FAISS index + metadata saved.")


# ============================
#   MAIN ENTRY
# ============================

if __name__ == "__main__":
    df, docs = load_kb()
    model = load_model()
    vectors = build_embeddings(model, docs)
    save_faiss(vectors, df)
