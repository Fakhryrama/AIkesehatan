# src/embeddings.py
import os
import numpy as np

# --- placeholder: ganti dengan panggilan Gemini embeddings sesuai SDK kamu ---
def get_embedding_gemini(text: str) -> list:
    """
    Contoh pseudocode: implementasikan panggilan ke Gemini embeddings.
    Return: list/ndarray vector
    """
    # TODO: ganti dengan code resmi provider
    raise NotImplementedError("Replace with your Gemini embedding call")

# --- fallback with sentence-transformers for dev/testing ---
def get_embedding_local(model, text):
    vec = model.encode(text, show_progress_bar=False)
    return vec / (np.linalg.norm(vec) + 1e-9)
