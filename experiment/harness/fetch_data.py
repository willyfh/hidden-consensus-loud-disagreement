"""Fetch Adult/Census Income dataset once and save a static local CSV.

Static local copy (rather than per-replicate live fetch) so all N replicate
agents see byte-identical data, and so the experiment doesn't depend on
network reliability across dozens of runs.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

print("Fetching adult (census income) dataset from OpenML...")
data = fetch_openml("adult", version=2, as_frame=True, parser="auto")

df = data.frame
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target distribution:\n{df[data.target_names[0] if hasattr(data, 'target_names') else df.columns[-1]].value_counts()}")

out_path = EXPERIMENT_DIR / "data" / "adult_income.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
