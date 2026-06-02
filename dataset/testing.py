from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


TARGET = "demand"
EXCLUDED_CSV_NAMES = {"train.csv", "test.csv", "real_test.csv"}
EXCLUDED_CSV_TAGS = ("ext", "leak")


def find_dataset_root() -> Path:
    return Path("C:/Users/bagri/Downloads/e88186124ec611f1/dataset/")


def main():
    root = find_dataset_root()
    real_test_path = root / "real_test.csv"

    if not real_test_path.exists():
        print(f"Error: Could not find real_test.csv under {root}")
        return

    print(f"Loading real_test.csv from: {real_test_path}")
    df_real = pd.read_csv(real_test_path)

    if TARGET not in df_real.columns:
        print("Error: real_test.csv does not contain 'demand' column")
        return

    real_demand = df_real[TARGET].to_numpy(dtype=np.float64)
    real_len = len(df_real)
    print(f"Target size: {real_len} rows")

    results = []
    print(f"Scanning submission-shaped CSV files under: {root} ...")
    for path in sorted(root.rglob("*.csv"), key=lambda p: str(p.relative_to(root)).lower()):
        name = path.name.lower()
        if name in EXCLUDED_CSV_NAMES or any(tag in name for tag in EXCLUDED_CSV_TAGS):
            continue
        try:
            df_temp = pd.read_csv(path)
        except Exception:
            continue
        if len(df_temp) != real_len or TARGET not in df_temp.columns:
            continue

        pred_demand = pd.to_numeric(df_temp[TARGET], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        score = r2_score(real_demand, pred_demand) * 100
        results.append((str(path.relative_to(root)), score))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=========================================")
    print("BY R2 SCORE (0-100 SCALE)")
    print("=========================================\n")
    for i, (name, score) in enumerate(results):
        print(f"{i:2d}. {name:60s} | R2 Score: {score:.6f}")


if __name__ == "__main__":
    main()
