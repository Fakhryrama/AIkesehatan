# src/fix_kb_json.py

import pandas as pd
import json
from pathlib import Path

# 1. Lokasi file
BASE_DIR = Path(__file__).resolve().parent.parent  # folder AIkesehatan
csv_path = BASE_DIR / "data" / "disease_kb.csv"
json_path = BASE_DIR / "data" / "disease_kb_clean.json"

print(f"Membaca CSV dari: {csv_path}")

# 2. Baca CSV dan bersihkan NaN
df = pd.read_csv(csv_path)

# Pastikan kolom yang kamu sebutkan ada semua
expected_cols = ["Disease", "Symptoms", "Description", "Precautions", "symptom_list", "combined"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("WARNING: Kolom berikut tidak ditemukan di CSV:", missing)

# Ganti NaN dengan string kosong supaya JSON valid
df = df.fillna("")

# 3. Konversi ke list of dict
records = df.to_dict(orient="records")
print(f"Total record: {len(records)}")

# 4. Simpan sebagai JSON rapi
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Selesai! JSON bersih disimpan ke: {json_path}")
print("Embedding TIDAK diubah, faiss.index & meta.pkl tetap dipakai seperti sebelumnya.")
