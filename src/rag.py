# src/rag.py
import faiss, pickle, numpy as np
from embeddings import get_embedding_gemini, get_embedding_local
from sentence_transformers import SentenceTransformer
from typing import List

class RAGEngine:
    def __init__(self, index_path="data/faiss.index", meta_path="data/meta.pkl", use_local=True):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        self.use_local = use_local
        if use_local:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, text: str):
        if self.use_local:
            vec = get_embedding_local(self.model, text)
        else:
            vec = get_embedding_gemini(text)
        vec = np.array(vec).astype('float32')
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    def retrieve(self, query: str, top_k=3):
        qvec = self.embed(query).reshape(1, -1)
        D, I = self.index.search(qvec, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            item = self.meta[idx].copy()
            item['score'] = float(score)
            results.append(item)
        return results

# Usage example
if __name__ == "__main__":
    engine = RAGEngine(use_local=True)
    print(engine.retrieve("sakit kepala, mual, berputar"))
