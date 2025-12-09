# src/rag.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class RAGEngine:
    """
    Tool 1: Retrieval / RAG engine berbasis FAISS.

    - Memuat index FAISS + metadata (meta.pkl)
    - Menggunakan SentenceTransformer multilingual untuk embedding query
    - Mengembalikan list dict: {Disease, Symptoms, Description, Precautions, score}
    """

    def __init__(
        self,
        index_path: str = "../data/faiss.index",
        meta_path: str = "../data/meta.pkl",
        use_local: bool = True,
        min_score: Optional[float] = 0.40,
    ):
        """
        :param index_path: path ke file FAISS index
        :param meta_path: path ke metadata pickle
        :param use_local: saat ini hanya mendukung embedding lokal (SentenceTransformer)
        :param min_score: threshold skor kemiripan (inner product, 0–1).
                          Jika None, tidak difilter.
        """
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        self.use_local = use_local
        self.min_score = min_score

        if use_local:
            # multilingual: cocok untuk input Bahasa Indonesia + data Inggris
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        else:
            # Kalau suatu saat mau pakai embedding dari Gemini, bisa diisi di sini.
            self.model = None

    def embed(self, text: str) -> np.ndarray:
        """
        Mengubah teks menjadi vektor embedding (float32, L2-normalized).
        Saat ini hanya mendukung embedding lokal (SentenceTransformer).
        """
        if not self.use_local:
            raise NotImplementedError(
                "Embedding non-lokal (mis. Gemini) belum dikonfigurasi di RAGEngine."
            )

        # langsung encode pakai SentenceTransformer
        vec = self.model.encode(text, show_progress_bar=False)
        vec = np.array(vec).astype("float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Mengambil top_k dokumen paling mirip.
        Hanya mengembalikan dokumen dengan score >= min_score (jika diset).
        """
        qvec = self.embed(query).reshape(1, -1)
        D, I = self.index.search(qvec, top_k)

        results: List[dict] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue

            item = self.meta[idx].copy()
            item["score"] = float(score)

            if (self.min_score is not None) and (score < self.min_score):
                # Terlalu tidak mirip, skip
                continue

            results.append(item)

        return results


# Tes cepat (opsional)
if __name__ == "__main__":
    engine = RAGEngine(use_local=True, min_score=0.40)
    print(engine.retrieve("kepala belakang terasa pening sejak dua hari, kadang mual"))
