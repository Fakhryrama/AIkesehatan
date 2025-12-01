import pandas as pd
import json

def build_kb(csv_in="data/disease_kb.csv", json_out="data/disease_kb.json"):
    df = pd.read_csv(csv_in)
    records = df.to_dict(orient="records")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("KB JSON saved:", json_out)

if __name__ == "__main__":
    build_kb()
