# src/preprocess.py
import pandas as pd
import re

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'\s+,', ',', s)
    s = re.sub(r'[^a-z0-9, ()./-]', ' ', s)  # sesuaikan bahasa
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def preprocess(in_path="data/Disease_Dataset_akhir.csv", out_path="data/disease_kb.csv"):
    df = pd.read_csv(in_path)
    df['Symptoms'] = df['Symptoms'].fillna("")
    df['Disease'] = df['Disease'].fillna("")
    df['Description'] = df['Description'].fillna("")
    df['Precautions'] = df['Precautions'].fillna("")

    df['symptom_list'] = df['Symptoms'].apply(lambda x: [s.strip() for s in clean_text(x).split(',') if s.strip()])
    df['combined'] = df.apply(lambda r: " | ".join([r['Disease'], r['Symptoms'], r['Description'], r['Precautions']]), axis=1)
    df['combined'] = df['combined'].apply(clean_text)

    df.to_csv(out_path, index=False)
    print(f"Saved cleaned KB to {out_path}")

if __name__ == "__main__":
    preprocess()
