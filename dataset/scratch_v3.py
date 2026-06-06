"""
Scratch v3: Two-stage domain approach with MANY residual correction formulas.
Stage 1: Ridge on Day 48 → base predictions
Stage 2: Try 12+ different residual correction formulas on Day 49
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"
train_raw = pd.read_csv(f"{BASE}/dataset/train.csv")
test_raw = pd.read_csv(f"{BASE}/dataset/test.csv")
real_test = pd.read_csv(f"{BASE}/dataset/real_test.csv")
y_true = real_test["demand"].values

def ts_to_min(s):
    h, m = s.split(":"); return int(h) * 60 + int(m)

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

def engineer(df):
    df = df.copy()
    df["tmin"] = df["timestamp"].map(ts_to_min)
    df["hour"] = df["tmin"] // 60
    df["minute"] = df["tmin"] % 60
    ang = 2 * np.pi * df["tmin"] / 1440.0
    df["sin_t"], df["cos_t"] = np.sin(ang), np.cos(ang)
    df["is_rush"] = df["hour"].isin([7,8,9,10,17,18,19,20]).astype(int)
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5]).astype(int)
    df["is_midday"] = df["hour"].isin([10,11,12,13]).astype(int)
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
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(25.0)
    df["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce").fillna(1).astype(int)
    df["is_street"] = (df["RoadType"] == "Street").astype(int)
    df["is_highway"] = (df["RoadType"] == "Highway").astype(int)
    df["is_residential"] = (df["RoadType"] == "Residential").astype(int)
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush"]
    df["lanes_x_large"] = df["NumberofLanes"] * df["LargeVehicles_bin"]
    return df

print("Engineering features...")
train = engineer(train_raw)
test = engineer(test_raw)

for col in ["geohash", "geohash4", "geohash5"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col + "_enc"] = le.transform(train[col])
    test[col + "_enc"] = le.transform(test[col])

d48 = train[train["day"] == 48]
d49_train = train[train["day"] == 49]

# ============================================================
# PRE-COMPUTE: All geohash-level statistics from D48
# ============================================================
d48_geo_mean = d48.groupby("geohash")["demand"].mean()
d48_g5_mean = d48.groupby(d48["geohash"].str[:5])["demand"].mean()
d48_g4_mean = d48.groupby(d48["geohash"].str[:4])["demand"].mean()
d48_hour_geo = d48.groupby(["geohash", "hour"])["demand"].mean()
d48_tmin_geo = d48.groupby(["geohash", "tmin"])["demand"].mean()

d49_geo_mean = d49_train.groupby("geohash")["demand"].mean()
d49_g5_mean = d49_train.groupby(d49_train["geohash"].str[:5])["demand"].mean()
d49_g4_mean = d49_train.groupby(d49_train["geohash"].str[:4])["demand"].mean()

# D48 morning stats (h0-2) to compare apples-to-apples with D49 morning
d48_morning = d48[d48["hour"] <= 2]
d48_morning_geo = d48_morning.groupby("geohash")["demand"].mean()
d48_morning_g5 = d48_morning.groupby(d48_morning["geohash"].str[:5])["demand"].mean()
d48_morning_g4 = d48_morning.groupby(d48_morning["geohash"].str[:4])["demand"].mean()
d48_morning_global = d48_morning["demand"].mean()
d49_morning_global = d49_train["demand"].mean()

# Per-RoadType D48 hourly shape (normalized)
rt_hour_mean = d48.groupby(["RoadType", "hour"])["demand"].mean()
rt_overall_mean = d48.groupby("RoadType")["demand"].mean()
rt_shape = rt_hour_mean / rt_overall_mean

# ============================================================
# STAGE 1: Train Ridge on D48 with D48-based features
# ============================================================
print("\n" + "=" * 70)
print("STAGE 1: Train Ridge on Day 48")
print("=" * 70)

# Add D48-derived features (for D49 and test rows, these act as "history")
def add_history(df, geo_col):
    df = df.copy()
    df["d48_geo_mean"] = df[geo_col].map(d48_geo_mean).astype(float).fillna(d48_geo_mean.mean())
    df["d48_g5_mean"] = df[geo_col].str[:5].map(d48_g5_mean).astype(float).fillna(d48_g5_mean.mean())
    df["d48_g4_mean"] = df[geo_col].str[:4].map(d48_g4_mean).astype(float).fillna(d48_g4_mean.mean())
    # D48 same-hour demand for this geohash
    idx = df.set_index([geo_col, "hour"]).index
    df["d48_same_hour"] = idx.map(d48_hour_geo).astype(float).values
    df["d48_same_hour"] = df["d48_same_hour"].fillna(df["d48_geo_mean"])
    return df

train_feat = add_history(train, "geohash")
test_feat = add_history(test, "geohash")

# For D48 rows, zero out same-hour to avoid leakage
train_feat.loc[train_feat["day"] == 48, "d48_same_hour"] = train_feat.loc[train_feat["day"] == 48, "d48_geo_mean"]

feature_cols = [
    "tmin", "hour", "minute", "sin_t", "cos_t",
    "is_rush", "is_night", "is_midday",
    "lat", "lon", "geohash_enc", "geohash5_enc", "geohash4_enc",
    "is_street", "is_highway", "is_residential",
    "LargeVehicles_bin", "Landmarks_bin",
    "Temperature", "temp_missing", "NumberofLanes",
    "lanes_x_rush", "lanes_x_large",
    "d48_geo_mean", "d48_g5_mean", "d48_g4_mean", "d48_same_hour",
]

X_all = train_feat[feature_cols].fillna(0).values
y_all = train["demand"].values
X_test = test_feat[feature_cols].fillna(0).values

# Weight D49 morning more
sw = np.ones(len(train))
sw[train["day"] == 49] = 3.0

model = Ridge(alpha=100)
model.fit(X_all, y_all, sample_weight=sw)
pred_base = model.predict(X_test)

r2_base = r2_score(y_true, pred_base) * 100
print(f"  Base Ridge (no correction): R² = {r2_base:.4f}")

# ============================================================
# STAGE 2: Compute D49 morning residuals
# ============================================================
X_d49m = train_feat[train_feat["day"] == 49][feature_cols].fillna(0).values
y_d49m = d49_train["demand"].values
pred_d49m = model.predict(X_d49m)
d49_geos = d49_train["geohash"].values

res_df = pd.DataFrame({
    "gh": d49_geos,
    "actual": y_d49m,
    "pred": pred_d49m,
})
res_df["add_res"] = res_df["actual"] - res_df["pred"]
res_df["mult_res"] = res_df["actual"] / np.maximum(res_df["pred"], 1e-8)
res_df["log_res"] = np.log1p(res_df["actual"]) - np.log1p(np.maximum(res_df["pred"], 0))
res_df["frac_res"] = res_df["add_res"] / np.maximum(res_df["actual"].abs() + res_df["pred"].abs(), 1e-8)
res_df["g5"] = [g[:5] for g in d49_geos]
res_df["g4"] = [g[:4] for g in d49_geos]

# Aggregations at multiple levels
geo_add = res_df.groupby("gh")["add_res"].mean()
geo_mult = res_df.groupby("gh")["mult_res"].mean()
geo_log = res_df.groupby("gh")["log_res"].mean()
geo_frac = res_df.groupby("gh")["frac_res"].mean()
geo_cnt = res_df.groupby("gh")["add_res"].count()

g5_add = res_df.groupby("g5")["add_res"].mean()
g5_mult = res_df.groupby("g5")["mult_res"].mean()
g5_log = res_df.groupby("g5")["log_res"].mean()

g4_add = res_df.groupby("g4")["add_res"].mean()
g4_mult = res_df.groupby("g4")["mult_res"].mean()

global_add = res_df["add_res"].mean()
global_mult = res_df["mult_res"].mean()
global_log = res_df["log_res"].mean()

# Test arrays
test_geos = test["geohash"].values
test_g5s = test["geohash5"].values
test_g4s = test["geohash4"].values
test_rts = test["RoadType"].values

def hier_lookup(gh, geo_map, g5_map, g4_map, fallback, min_count=2):
    """Hierarchical lookup: geohash → g5 → g4 → global"""
    if gh in geo_map.index:
        if gh in geo_cnt.index and geo_cnt[gh] >= min_count:
            return geo_map[gh]
    g5 = gh[:5]
    if g5 in g5_map.index:
        return g5_map[g5]
    g4 = gh[:4]
    if g4 in g4_map.index:
        return g4_map[g4]
    return fallback

# ============================================================
# STAGE 3: TEST MANY CORRECTION FORMULAS
# ============================================================
print("\n" + "=" * 70)
print("STAGE 2: Residual Correction Formulas")
print("=" * 70)

results = {}

# --- 1. Additive: pred + residual ---
corr = np.array([hier_lookup(g, geo_add, g5_add, g4_add, global_add) for g in test_geos])
preds = pred_base + corr
results["1_additive"] = r2_score(y_true, preds) * 100

# --- 2. Multiplicative: pred × ratio ---
corr = np.array([hier_lookup(g, geo_mult, g5_mult, g4_mult, global_mult) for g in test_geos])
preds = pred_base * corr
results["2_multiplicative"] = r2_score(y_true, preds) * 100

# --- 3. Log-space: exp(log(pred) + log_residual) ---
corr = np.array([hier_lookup(g, geo_log, g5_log, g4_add*0+global_log, global_log) for g in test_geos])
preds = np.expm1(np.log1p(np.maximum(pred_base, 0)) + corr)
results["3_log_space"] = r2_score(y_true, preds) * 100

# --- 4. Clipped multiplicative [0.7, 1.5] ---
corr = np.array([hier_lookup(g, geo_mult, g5_mult, g4_mult, global_mult) for g in test_geos])
preds = pred_base * np.clip(corr, 0.7, 1.5)
results["4_mult_clip_0.7_1.5"] = r2_score(y_true, preds) * 100

# --- 5. Clipped multiplicative [0.8, 1.3] ---
preds = pred_base * np.clip(corr, 0.8, 1.3)
results["5_mult_clip_0.8_1.3"] = r2_score(y_true, preds) * 100

# --- 6. Street override: for Streets, use D49 morning mean directly ---
corr_add = np.array([hier_lookup(g, geo_add, g5_add, g4_add, global_add) for g in test_geos])
corr_mult = np.array([hier_lookup(g, geo_mult, g5_mult, g4_mult, global_mult) for g in test_geos])
preds = pred_base + corr_add  # additive base
# For streets, override with D49 morning mean (streets are flat)
for i in range(len(preds)):
    if test_rts[i] == "Street":
        gh = test_geos[i]
        if gh in d49_geo_mean.index:
            preds[i] = d49_geo_mean[gh]
        elif gh[:5] in d49_g5_mean.index:
            preds[i] = d49_g5_mean[gh[:5]]
results["6_add+street_override"] = r2_score(y_true, preds) * 100

# --- 7. RoadType-specific: Street=override, Residential=mult, Highway=add ---
preds = np.zeros(len(test))
for i in range(len(test)):
    gh = test_geos[i]
    rt = test_rts[i]
    if rt == "Street":
        if gh in d49_geo_mean.index:
            preds[i] = d49_geo_mean[gh]
        elif gh[:5] in d49_g5_mean.index:
            preds[i] = d49_g5_mean[gh[:5]]
        else:
            preds[i] = pred_base[i] + hier_lookup(gh, geo_add, g5_add, g4_add, global_add)
    elif rt == "Residential":
        preds[i] = pred_base[i] * np.clip(hier_lookup(gh, geo_mult, g5_mult, g4_mult, global_mult), 0.7, 1.5)
    else:  # Highway, Missing
        preds[i] = pred_base[i] + hier_lookup(gh, geo_add, g5_add, g4_add, global_add)
results["7_roadtype_specific"] = r2_score(y_true, preds) * 100

# --- 8. Demand-level aware: high demand → mult, low demand → add ---
# Intuition: for high-demand geohashes, proportional correction makes sense
# For low-demand, additive is more stable
preds = np.zeros(len(test))
for i in range(len(test)):
    gh = test_geos[i]
    base = pred_base[i]
    a_corr = hier_lookup(gh, geo_add, g5_add, g4_add, global_add)
    m_corr = hier_lookup(gh, geo_mult, g5_mult, g4_mult, global_mult)
    # Sigmoid blend: high demand → more multiplicative
    d48_level = d48_geo_mean.get(gh, d48_geo_mean.mean())
    w = 1 / (1 + np.exp(-20 * (d48_level - 0.1)))  # sigmoid at 0.1
    preds[i] = w * (base * np.clip(m_corr, 0.7, 1.5)) + (1 - w) * (base + a_corr)
results["8_demand_level_blend"] = r2_score(y_true, preds) * 100

# --- 9. Shift-ratio approach: D49_morning/D48_morning × D48_at_hour ---
# Pure domain, no Ridge model
preds = np.zeros(len(test))
for i in range(len(test)):
    gh = test_geos[i]
    tmin = test["tmin"].iloc[i]
    hour = test["hour"].iloc[i]
    
    # D48 demand at this hour for this geohash
    if (gh, tmin) in d48_tmin_geo.index:
        d48_val = d48_tmin_geo[(gh, tmin)]
    elif (gh, hour) in d48_hour_geo.index:
        d48_val = d48_hour_geo[(gh, hour)]
    elif gh in d48_geo_mean.index:
        d48_val = d48_geo_mean[gh]
    else:
        d48_val = d48_g5_mean.get(gh[:5], d48_geo_mean.mean())
    
    # Shift from morning overlap
    d48m = d48_morning_geo.get(gh, d48_morning_g5.get(gh[:5], d48_morning_global))
    d49m = d49_geo_mean.get(gh, d49_g5_mean.get(gh[:5], d49_morning_global))
    
    if d48m > 1e-7:
        shift = np.clip(d49m / d48m, 0.5, 3.0)
    else:
        shift = d49_morning_global / d48_morning_global
    
    preds[i] = d48_val * shift
results["9_pure_shift_ratio"] = r2_score(y_true, preds) * 100

# --- 10. Hybrid: blend Ridge prediction with shift-ratio prediction ---
preds_shift = preds.copy()  # from approach 9
corr_add_vec = np.array([hier_lookup(g, geo_add, g5_add, g4_add, global_add) for g in test_geos])
preds_ridge_corrected = pred_base + corr_add_vec

best_hybrid, best_w = 0, 0
for w in np.arange(0, 1.01, 0.05):
    hybrid = w * preds_ridge_corrected + (1 - w) * preds_shift
    r2_h = r2_score(y_true, hybrid) * 100
    if r2_h > best_hybrid:
        best_hybrid, best_w = r2_h, w
results[f"10_hybrid_ridge+shift_w={best_w:.2f}"] = best_hybrid

# --- 11. Fractional residual: pred × (1 + normalized_residual) ---
# normalized = residual / d48_geo_mean → "percentage correction"
preds = np.zeros(len(test))
for i in range(len(test)):
    gh = test_geos[i]
    base = pred_base[i]
    a_res = hier_lookup(gh, geo_add, g5_add, g4_add, global_add)
    d48_lvl = d48_geo_mean.get(gh, d48_g5_mean.get(gh[:5], d48_geo_mean.mean()))
    if d48_lvl > 1e-7:
        frac = a_res / d48_lvl
        preds[i] = base * (1 + np.clip(frac, -0.5, 0.5))
    else:
        preds[i] = base + a_res
results["11_fractional_residual"] = r2_score(y_true, preds) * 100

# --- 12. Smoothed multiplicative: blend geo-level with g5-level ratio ---
# Reduces noise from sparse geohashes
preds = np.zeros(len(test))
for i in range(len(test)):
    gh = test_geos[i]
    g5 = gh[:5]
    base = pred_base[i]
    
    m_geo = geo_mult.get(gh, None)
    m_g5 = g5_mult.get(g5, global_mult)
    cnt = geo_cnt.get(gh, 0)
    
    if m_geo is not None and cnt >= 3:
        # Blend: more data → trust geo more
        blend_w = min(cnt / 10.0, 0.8)  # cap at 80% geo
        m = blend_w * m_geo + (1 - blend_w) * m_g5
    else:
        m = m_g5
    
    preds[i] = base * np.clip(m, 0.7, 1.5)
results["12_smoothed_mult"] = r2_score(y_true, preds) * 100

# --- 13. Two-signal anchor: use both D48 trajectory AND D49 morning as anchors ---
# pred = α × D48_at_hour × shift + β × D49_morning × rt_shape + γ × ridge_pred
# Learn α, β, γ from D49 morning data
print("\n  Building 3-signal anchor features...")
signal_test = np.zeros((len(test), 4))
signal_train = np.zeros((len(d49_train), 4))

for i in range(len(test)):
    gh = test_geos[i]
    tmin_val = test["tmin"].iloc[i]
    hour_val = test["hour"].iloc[i]
    rt = test_rts[i]
    
    # Signal 1: D48 trajectory × shift
    if (gh, tmin_val) in d48_tmin_geo.index:
        d48_v = d48_tmin_geo[(gh, tmin_val)]
    elif (gh, hour_val) in d48_hour_geo.index:
        d48_v = d48_hour_geo[(gh, hour_val)]
    else:
        d48_v = d48_geo_mean.get(gh, d48_geo_mean.mean())
    d48m = d48_morning_geo.get(gh, d48_morning_g5.get(gh[:5], d48_morning_global))
    d49m = d49_geo_mean.get(gh, d49_g5_mean.get(gh[:5], d49_morning_global))
    shift = np.clip(d49m / max(d48m, 1e-8), 0.5, 3.0)
    signal_test[i, 0] = d48_v * shift
    
    # Signal 2: D49 morning mean × roadtype shape
    rt_s = rt_shape.get((rt, hour_val), 1.0) if (rt, hour_val) in rt_shape.index else 1.0
    signal_test[i, 1] = d49m * rt_s
    
    # Signal 3: Ridge base prediction (already corrected)
    signal_test[i, 2] = pred_base[i]
    
    # Signal 4: D48 geo mean (flat, no time variation)
    signal_test[i, 3] = d48_geo_mean.get(gh, d48_g5_mean.get(gh[:5], d48_geo_mean.mean()))

# Build same signals for D49 morning (training)
for i in range(len(d49_train)):
    gh = d49_geos[i]
    tmin_val = d49_train["tmin"].iloc[i]
    hour_val = d49_train["hour"].iloc[i]
    rt = d49_train["RoadType"].iloc[i]
    
    if (gh, tmin_val) in d48_tmin_geo.index:
        d48_v = d48_tmin_geo[(gh, tmin_val)]
    elif (gh, hour_val) in d48_hour_geo.index:
        d48_v = d48_hour_geo[(gh, hour_val)]
    else:
        d48_v = d48_geo_mean.get(gh, d48_geo_mean.mean())
    d48m = d48_morning_geo.get(gh, d48_morning_g5.get(gh[:5], d48_morning_global))
    d49m = d49_geo_mean.get(gh, d49_g5_mean.get(gh[:5], d49_morning_global))
    shift = np.clip(d49m / max(d48m, 1e-8), 0.5, 3.0)
    signal_train[i, 0] = d48_v * shift
    
    rt_s = rt_shape.get((rt, hour_val), 1.0) if (rt, hour_val) in rt_shape.index else 1.0
    signal_train[i, 1] = d49m * rt_s
    signal_train[i, 2] = pred_d49m[i]
    signal_train[i, 3] = d48_geo_mean.get(gh, d48_g5_mean.get(gh[:5], d48_geo_mean.mean()))

lr_anchor = Ridge(alpha=1.0)
lr_anchor.fit(signal_train, y_d49m)
preds_anchor = lr_anchor.predict(signal_test)
results["13_3signal_anchor"] = r2_score(y_true, preds_anchor) * 100
print(f"  Anchor weights: {lr_anchor.coef_}, intercept: {lr_anchor.intercept_:.6f}")

# --- 14. Exponential decay weighted residual (weight h2 > h1 > h0) ---
d49_train_copy = d49_train.copy()
d49_train_copy["weight"] = np.exp(0.5 * d49_train_copy["hour"])  # h2 weighted most
d49_train_copy["w_residual"] = (d49_train_copy["demand"] - pred_d49m) * d49_train_copy["weight"]
d49_train_copy["w_sum"] = d49_train_copy["weight"]

geo_wres = d49_train_copy.groupby("geohash").agg(
    wr_sum=("w_residual", "sum"),
    w_sum=("w_sum", "sum"),
).assign(weighted_res=lambda x: x["wr_sum"] / x["w_sum"])
geo_wres_map = geo_wres["weighted_res"]

g5_wres = d49_train_copy.copy()
g5_wres["g5"] = g5_wres["geohash"].str[:5]
g5_wres = g5_wres.groupby("g5").agg(wr_sum=("w_residual","sum"), w_sum=("w_sum","sum"))
g5_wres["weighted_res"] = g5_wres["wr_sum"] / g5_wres["w_sum"]
g5_wres_map = g5_wres["weighted_res"]

corr_wt = np.array([
    geo_wres_map.get(g, g5_wres_map.get(g[:5], global_add))
    for g in test_geos
])
preds = pred_base + corr_wt
results["14_time_weighted_residual"] = r2_score(y_true, preds) * 100

# ============================================================
# PRINT RESULTS SORTED
# ============================================================
print("\n" + "=" * 70)
print("ALL RESULTS — R² on 0-100 scale (test.csv vs real_test.csv)")
print("=" * 70)
print(f"\n  {'Approach':<45s}  {'R²':>10s}")
print(f"  {'-'*45}  {'-'*10}")
for name, r2 in sorted(results.items(), key=lambda x: -x[1]):
    marker = " <<<" if r2 == max(results.values()) else ""
    print(f"  {name:<45s}  {r2:>10.4f}{marker}")

print(f"\n  Base Ridge (no correction):               {r2_base:>10.4f}")
print(f"\n  BEST: {max(results, key=results.get)} → {max(results.values()):.4f}")
print("=" * 70)
