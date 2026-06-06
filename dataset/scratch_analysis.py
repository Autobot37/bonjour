import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train.csv")
df["tmin"] = df["timestamp"].map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
df["hour"] = df["tmin"] // 60

test = pd.read_csv("dataset/test.csv")
test["tmin"] = test["timestamp"].map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
test["hour"] = test["tmin"] // 60

print("=== Test set structure ===")
print("Day values:", test["day"].unique())
print("Hour range:", test["hour"].min(), "to", test["hour"].max())
print("Unique geohashes in test:", test["geohash"].nunique())
print("Total rows:", len(test))

train_geos = set(df["geohash"].unique())
test_geos = set(test["geohash"].unique())
print("Test geos in train:", len(test_geos & train_geos), "/", len(test_geos))
print("Test geos NOT in train:", len(test_geos - train_geos))

d49 = df[df["day"] == 49]
d49_geos = set(d49["geohash"].unique())
print("Test geos with Day49 morning data:", len(test_geos & d49_geos), "/", len(test_geos))

print("\nRoadType distribution in test:")
print(test["RoadType"].value_counts())

# Key question: how well does the simple "scale Day48 profile" approach work?
d48 = df[df["day"] == 48]

# Build Day 48 hourly profile per geohash
d48_profile = d48.groupby(["geohash", "tmin"])["demand"].mean().reset_index()
d48_morning_avg = d48[d48["hour"] <= 2].groupby("geohash")["demand"].mean()
d49_morning_avg = d49.groupby("geohash")["demand"].mean()

# For geohashes in both mornings, compute the shift
common = d48_morning_avg.index.intersection(d49_morning_avg.index)
shift_ratio = d49_morning_avg[common] / d48_morning_avg[common]
shift_ratio = shift_ratio.replace([np.inf, -np.inf], np.nan).dropna()

print("\n=== Shift Ratio (D49 morning / D48 morning) ===")
print("Count:", len(shift_ratio))
print("Mean:", shift_ratio.mean())
print("Median:", shift_ratio.median())

# Check: for the overlapping morning window (h0-2), how well does 
# "D48_morning * shift_ratio" predict D49_morning per timestamp?
d48_morning_ts = d48[d48["hour"] <= 2].groupby(["geohash", "tmin"])["demand"].mean().reset_index()
d49_morning_ts = d49.groupby(["geohash", "tmin"])["demand"].mean().reset_index()

merged = d48_morning_ts.merge(d49_morning_ts, on=["geohash", "tmin"], suffixes=("_48", "_49"))
merged["shift"] = merged["geohash"].map(shift_ratio)
merged["pred_simple"] = merged["demand_48"] * merged["shift"]
merged = merged.dropna()

from sklearn.metrics import r2_score
r2_simple = r2_score(merged["demand_49"], merged["pred_simple"])
print(f"\nR2 of simple scaling on overlapping morning: {r2_simple:.4f}")

# What about just using D48 demand directly (no shift)?
r2_raw = r2_score(merged["demand_49"], merged["demand_48"])
print(f"R2 of raw D48 values (no shift): {r2_raw:.4f}")

# What about global shift (single multiplier)?
global_shift = d49_morning_avg.mean() / d48_morning_avg.mean()
merged["pred_global"] = merged["demand_48"] * global_shift
r2_global = r2_score(merged["demand_49"], merged["pred_global"])
print(f"R2 of global shift ({global_shift:.4f}x): {r2_global:.4f}")
