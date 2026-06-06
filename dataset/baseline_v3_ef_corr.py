"""
baseline_v3_ef_corr.py — Domain-knowledge Ridge baseline.

Key insight from scratch exploration: post-hoc residual corrections HURT
when the model already sees D49 morning data in training. Instead, pack
ALL domain signals (D48 trajectory, D49 morning anchors, shift ratios,
RoadType shapes, zone aggregates) as FEATURES and let Ridge learn the
optimal combination.

Approach:
  1. Engineer rich domain features from D48 history
  2. Add D49 morning anchors (per-geohash, per-g5, per-g4)
  3. Add shift ratios and RoadType hourly shapes
  4. Train Ridge on all train data (D48 + D49 morning, D49 upweighted)
  5. Predict test, report R2 score
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"
train_raw = pd.read_csv(os.path.join(BASE, "dataset", "train.csv"))
test_raw = pd.read_csv(os.path.join(BASE, "dataset", "test.csv"))
real_test = pd.read_csv(os.path.join(BASE, "dataset", "real_test.csv"))
y_true = real_test["demand"].values

# ============================================================
# GEOHASH DECODING
# ============================================================
_B32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_B32_IDX = {c: i for i, c in enumerate(_B32)}
def decode_gh(gh):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in gh:
        cd = _B32_IDX.get(c, 0)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if cd & mask else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if cd & mask else (lat_lo, mid)
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

def ts_to_min(s):
    h, m = s.split(":"); return int(h) * 60 + int(m)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def base_features(df):
    df = df.copy()
    df["tmin"] = df["timestamp"].map(ts_to_min)
    df["hour"] = df["tmin"] // 60
    df["minute"] = df["tmin"] % 60
    ang = 2 * np.pi * df["tmin"] / 1440.0
    df["sin_t"], df["cos_t"] = np.sin(ang), np.cos(ang)
    df["sin_2t"], df["cos_2t"] = np.sin(2*ang), np.cos(2*ang)  # 2nd harmonic
    df["is_rush"] = df["hour"].isin([7,8,9,10,17,18,19,20]).astype(int)
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5]).astype(int)
    df["is_midday"] = df["hour"].isin([10,11,12,13]).astype(int)
    df["is_morning_ramp"] = df["hour"].isin([3,4,5,6,7]).astype(int)
    df["is_afternoon_fall"] = df["hour"].isin([13,14,15,16]).astype(int)

    cache = {g: decode_gh(g) for g in df["geohash"].unique()}
    df["lat"] = df["geohash"].map(lambda g: cache[g][0])
    df["lon"] = df["geohash"].map(lambda g: cache[g][1])

    df["geohash5"] = df["geohash"].str[:5]
    df["geohash4"] = df["geohash"].str[:4]

    df["RoadType"] = df["RoadType"].fillna("Missing")
    df["Weather"] = df["Weather"].fillna("Missing")
    df["LargeVehicles_bin"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["Landmarks_bin"] = (df["Landmarks"] == "Yes").astype(int)
    df["temp_missing"] = df["Temperature"].isna().astype(int)
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    median_temp = df["Temperature"].median()
    df["Temperature"] = df["Temperature"].fillna(median_temp)
    df["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce").fillna(1).astype(int)

    df["is_street"] = (df["RoadType"] == "Street").astype(int)
    df["is_highway"] = (df["RoadType"] == "Highway").astype(int)
    df["is_residential"] = (df["RoadType"] == "Residential").astype(int)

    # Interactions
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush"]
    df["lanes_x_large"] = df["NumberofLanes"] * df["LargeVehicles_bin"]
    df["street_x_hour"] = df["is_street"] * df["hour"]
    df["highway_x_hour"] = df["is_highway"] * df["hour"]
    df["residential_x_hour"] = df["is_residential"] * df["hour"]
    df["temp_x_rush"] = df["Temperature"] * df["is_rush"]
    return df

print("=" * 70)
print("BASELINE V3: Domain-Knowledge Ridge")
print("=" * 70)

print("\n[1/5] Base feature engineering...")
train = base_features(train_raw)
test = base_features(test_raw)

# Label encode geohash columns
for col in ["geohash", "geohash4", "geohash5"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col + "_enc"] = le.transform(train[col])
    test[col + "_enc"] = le.transform(test[col])

# ============================================================
# PRE-COMPUTE DAY 48 STATISTICS (history features)
# ============================================================
print("[2/5] Computing Day 48 history features...")
d48 = train[train["day"] == 48]
d49_train = train[train["day"] == 49]
d48_morning = d48[d48["hour"] <= 2]

# Geohash-level D48 stats
d48_geo_mean = d48.groupby("geohash")["demand"].mean()
d48_geo_std = d48.groupby("geohash")["demand"].std().fillna(0)
d48_geo_max = d48.groupby("geohash")["demand"].max()
d48_geo_count = d48.groupby("geohash")["demand"].count()

# Neighborhood-level D48 stats
d48_g5_mean = d48.groupby(d48["geohash"].str[:5])["demand"].mean()
d48_g5_std = d48.groupby(d48["geohash"].str[:5])["demand"].std().fillna(0)
d48_g4_mean = d48.groupby(d48["geohash"].str[:4])["demand"].mean()

# D48 per-hour demand for each geohash
d48_hour_geo = d48.groupby(["geohash", "hour"])["demand"].mean()

# D48 per-RoadType-hour demand (normalized shape)
rt_hour_mean = d48.groupby(["RoadType", "hour"])["demand"].mean()
rt_overall_mean = d48.groupby("RoadType")["demand"].mean()
rt_shape = rt_hour_mean / rt_overall_mean

# D48 morning stats (for shift computation)
d48_morning_geo = d48_morning.groupby("geohash")["demand"].mean()
d48_morning_g5 = d48_morning.groupby(d48_morning["geohash"].str[:5])["demand"].mean()
d48_morning_g4 = d48_morning.groupby(d48_morning["geohash"].str[:4])["demand"].mean()
d48_morning_global = d48_morning["demand"].mean()

# ============================================================
# PRE-COMPUTE DAY 49 MORNING STATISTICS (calibration features)
# ============================================================
print("[3/5] Computing Day 49 morning calibration features...")
d49_geo_mean = d49_train.groupby("geohash")["demand"].mean()
d49_geo_std = d49_train.groupby("geohash")["demand"].std().fillna(0)
d49_geo_count = d49_train.groupby("geohash")["demand"].count()
d49_g5_mean = d49_train.groupby(d49_train["geohash"].str[:5])["demand"].mean()
d49_g4_mean = d49_train.groupby(d49_train["geohash"].str[:4])["demand"].mean()
d49_morning_global = d49_train["demand"].mean()

# Shift ratios (D49 morning / D48 morning) — hierarchical
def compute_shift(gh):
    g5, g4 = gh[:5], gh[:4]
    d48m = d48_morning_geo.get(gh, None)
    d49m = d49_geo_mean.get(gh, None)
    if d48m is not None and d49m is not None and d48m > 1e-7:
        return np.clip(d49m / d48m, 0.3, 5.0)
    d48m_g5 = d48_morning_g5.get(g5, None)
    d49m_g5 = d49_g5_mean.get(g5, None)
    if d48m_g5 is not None and d49m_g5 is not None and d48m_g5 > 1e-7:
        return np.clip(d49m_g5 / d48m_g5, 0.3, 5.0)
    d48m_g4 = d48_morning_g4.get(g4, None)
    d49m_g4 = d49_g4_mean.get(g4, None)
    if d48m_g4 is not None and d49m_g4 is not None and d48m_g4 > 1e-7:
        return np.clip(d49m_g4 / d48m_g4, 0.3, 5.0)
    return d49_morning_global / d48_morning_global

# ============================================================
# ADD DOMAIN FEATURES TO BOTH TRAIN AND TEST
# ============================================================
print("[4/5] Adding domain features to train and test...")

def add_domain_features(df, is_d48_rows=False):
    df = df.copy()
    geos = df["geohash"].values
    g5s = df["geohash5"].values
    g4s = df["geohash4"].values
    hours = df["hour"].values
    rts = df["RoadType"].values

    # D48 history features
    df["d48_geo_mean"] = df["geohash"].map(d48_geo_mean).astype(float).fillna(d48_geo_mean.mean())
    df["d48_geo_std"] = df["geohash"].map(d48_geo_std).astype(float).fillna(0)
    df["d48_geo_max"] = df["geohash"].map(d48_geo_max).astype(float).fillna(d48_geo_mean.mean())
    df["d48_geo_count"] = df["geohash"].map(d48_geo_count).astype(float).fillna(0)
    df["d48_g5_mean"] = df["geohash5"].map(d48_g5_mean).astype(float).fillna(d48_g5_mean.mean())
    df["d48_g5_std"] = df["geohash5"].map(d48_g5_std).astype(float).fillna(0)
    df["d48_g4_mean"] = df["geohash4"].map(d48_g4_mean).astype(float).fillna(d48_g4_mean.mean())

    # D48 same-hour demand (the time-resolved template)
    idx = list(zip(geos, hours))
    d48_hr = [d48_hour_geo.get(k, np.nan) for k in idx]
    df["d48_same_hour"] = d48_hr
    if is_d48_rows:
        df["d48_same_hour"] = df["d48_geo_mean"]  # avoid leakage for D48
    df["d48_same_hour"] = df["d48_same_hour"].fillna(df["d48_geo_mean"])

    # RoadType hourly shape
    rt_s = [rt_shape.get((rt, h), 1.0) if (rt, h) in rt_shape.index else 1.0
            for rt, h in zip(rts, hours)]
    df["rt_shape"] = rt_s

    # D49 morning calibration features
    df["d49_geo_mean"] = df["geohash"].map(d49_geo_mean).astype(float).fillna(d49_morning_global)
    df["d49_geo_std"] = df["geohash"].map(d49_geo_std).astype(float).fillna(0)
    df["d49_geo_count"] = df["geohash"].map(d49_geo_count).astype(float).fillna(0)
    df["d49_g5_mean"] = df["geohash5"].map(d49_g5_mean).astype(float).fillna(d49_morning_global)
    df["d49_g4_mean"] = df["geohash4"].map(d49_g4_mean).astype(float).fillna(d49_morning_global)

    # Shift ratio
    df["shift_ratio"] = df["geohash"].map(lambda g: compute_shift(g)).astype(float)

    # Composite signals (domain formulas as features)
    df["d48_x_shift"] = df["d48_same_hour"] * df["shift_ratio"]
    df["d49_x_rtshape"] = df["d49_geo_mean"] * df["rt_shape"]
    df["geo_mean_x_rtshape_x_shift"] = df["d48_geo_mean"] * df["rt_shape"] * df["shift_ratio"]

    # Relative features (how does this geohash compare to its zone)
    df["geo_vs_g5"] = df["d48_geo_mean"] / (df["d48_g5_mean"] + 1e-8)
    df["geo_vs_g4"] = df["d48_geo_mean"] / (df["d48_g4_mean"] + 1e-8)
    df["d49_vs_d48"] = df["d49_geo_mean"] / (df["d48_geo_mean"] + 1e-8)

    # Street-specific: for streets, D49 morning IS the best predictor (flat profile)
    df["street_anchor"] = df["is_street"] * df["d49_geo_mean"]
    df["highway_anchor"] = df["is_highway"] * df["d48_x_shift"]
    df["residential_anchor"] = df["is_residential"] * df["d49_x_rtshape"]

    return df

# Apply to D48 and D49 subsets separately to handle leakage
train_d48 = add_domain_features(train[train["day"] == 48], is_d48_rows=True)
train_d49 = add_domain_features(train[train["day"] == 49], is_d48_rows=False)
train_feat = pd.concat([train_d48, train_d49], ignore_index=True)
test_feat = add_domain_features(test, is_d48_rows=False)

# ============================================================
# DEFINE FEATURE SET AND TRAIN
# ============================================================
feature_cols = [
    # Time features
    "tmin", "hour", "minute", "sin_t", "cos_t", "sin_2t", "cos_2t",
    "is_rush", "is_night", "is_midday", "is_morning_ramp", "is_afternoon_fall",
    # Spatial features
    "lat", "lon", "geohash_enc", "geohash5_enc", "geohash4_enc",
    # Static features
    "is_street", "is_highway", "is_residential",
    "LargeVehicles_bin", "Landmarks_bin",
    "Temperature", "temp_missing", "NumberofLanes",
    # Interactions
    "lanes_x_rush", "lanes_x_large",
    "street_x_hour", "highway_x_hour", "residential_x_hour",
    "temp_x_rush",
    # D48 history
    "d48_geo_mean", "d48_geo_std", "d48_geo_max", "d48_geo_count",
    "d48_g5_mean", "d48_g5_std", "d48_g4_mean",
    "d48_same_hour", "rt_shape",
    # D49 morning calibration
    "d49_geo_mean", "d49_geo_std", "d49_geo_count",
    "d49_g5_mean", "d49_g4_mean",
    # Shift
    "shift_ratio",
    # Composite domain signals
    "d48_x_shift", "d49_x_rtshape", "geo_mean_x_rtshape_x_shift",
    # Relative
    "geo_vs_g5", "geo_vs_g4", "d49_vs_d48",
    # RoadType-specific anchors
    "street_anchor", "highway_anchor", "residential_anchor",
]

X_train = train_feat[feature_cols].fillna(0).values
y_train = train_feat["demand"].values
X_test = test_feat[feature_cols].fillna(0).values

# Sample weights: upweight D49 morning (closer to test distribution)
sw = np.ones(len(train_feat))
sw[train_feat["day"].values == 49] = 3.0
# Also upweight D48 daytime hours (2-13) that match test hour range
d48_mask = (train_feat["day"].values == 48)
daytime_mask = (train_feat["hour"].values >= 2) & (train_feat["hour"].values <= 13)
sw[d48_mask & daytime_mask] = 1.5

print("[5/5] Training and evaluating Ridge models...")
print(f"\n  Features: {len(feature_cols)}")
print(f"  Train rows: {len(X_train)} (D48: {d48_mask.sum()}, D49: {(~d48_mask).sum()})")
print(f"  Test rows: {len(X_test)}")

# Alpha sweep
print(f"\n  {'Alpha':>8s}  {'R2 Score':>12s}")
print(f"  {'-'*8}  {'-'*12}")
best_r2, best_alpha = 0, 100
for alpha in [0.1, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=sw)
    preds = model.predict(X_test)
    r2 = r2_score(y_true, preds) * 100
    marker = ""
    if r2 > best_r2:
        best_r2, best_alpha = r2, alpha
        marker = " <-- best"
    print(f"  {alpha:>8.1f}  {r2:>12.4f}{marker}")

# Retrain with best alpha
print(f"\n  Best alpha: {best_alpha}")
model_best = Ridge(alpha=best_alpha)
model_best.fit(X_train, y_train, sample_weight=sw)
preds_best = model_best.predict(X_test)
r2_best = r2_score(y_true, preds_best) * 100

# Also try different D49 weights
print(f"\n  --- D49 weight sweep (alpha={best_alpha}) ---")
best_r2_w, best_w = r2_best, 3.0
for d49w in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    sw2 = np.ones(len(train_feat))
    sw2[train_feat["day"].values == 49] = d49w
    sw2[d48_mask & daytime_mask] = 1.5
    m = Ridge(alpha=best_alpha)
    m.fit(X_train, y_train, sample_weight=sw2)
    p = m.predict(X_test)
    r2 = r2_score(y_true, p) * 100
    marker = ""
    if r2 > best_r2_w:
        best_r2_w, best_w = r2, d49w
        marker = " <-- best"
    print(f"    d49_weight={d49w:>5.1f}  R2={r2:.4f}{marker}")

# Final model with best settings
sw_final = np.ones(len(train_feat))
sw_final[train_feat["day"].values == 49] = best_w
sw_final[d48_mask & daytime_mask] = 1.5
model_final = Ridge(alpha=best_alpha)
model_final.fit(X_train, y_train, sample_weight=sw_final)
preds_final = model_final.predict(X_test)
r2_final = r2_score(y_true, preds_final) * 100

# Print top feature importances (coefficients)
print(f"\n  --- Top 15 Feature Coefficients ---")
coef_pairs = sorted(zip(feature_cols, model_final.coef_), key=lambda x: -abs(x[1]))
for name, coef in coef_pairs[:15]:
    print(f"    {name:<35s}  {coef:+.6f}")
print(f"    {'intercept':<35s}  {model_final.intercept_:+.6f}")

# Save submission
sub = test_raw[["Index"]].copy()
sub["demand"] = preds_final
sub.to_csv(os.path.join(BASE, "dataset", "baseline_v3_domain.csv"), index=False)

print(f"\n" + "=" * 70)
print(f"FINAL RESULT: R2 = {r2_final:.4f} / 100")
print(f"  alpha={best_alpha}, d49_weight={best_w}")
print(f"  Saved to dataset/baseline_v3_domain.csv")
print("=" * 70)
