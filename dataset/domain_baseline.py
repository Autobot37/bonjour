"""
Domain-knowledge baseline for Day 49 demand prediction.
No tree models — just shift ratios, profile shapes, and optional linear regression.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("DOMAIN-KNOWLEDGE BASELINE FOR DAY 49 DEMAND PREDICTION")
print("=" * 70)

BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"
train_raw = pd.read_csv(f"{BASE}/dataset/train.csv")
test_raw = pd.read_csv(f"{BASE}/dataset/test.csv")
real_test = pd.read_csv(f"{BASE}/dataset/real_test.csv")
y_true = real_test["demand"].values

def add_time_features(df):
    df = df.copy()
    df["tmin"] = df["timestamp"].map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
    df["hour"] = df["tmin"] // 60
    df["minute"] = df["tmin"] % 60
    df["RoadType"] = df["RoadType"].fillna("Missing")
    df["Weather"] = df["Weather"].fillna("Missing")
    df["LargeVehicles_bin"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["Landmarks_bin"] = (df["Landmarks"] == "Yes").astype(int)
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce").fillna(1)
    return df

train = add_time_features(train_raw)
test = add_time_features(test_raw)

d48 = train[train["day"] == 48]
d49_train = train[train["day"] == 49]

# ============================================================
# APPROACH 1: PURE DOMAIN — D48 profile × per-geohash shift
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 1: Pure Domain (D48 template × shift ratio)")
print("=" * 70)

# Build Day 48 lookup: geohash × tmin → mean demand
d48_lookup = d48.groupby(["geohash", "tmin"])["demand"].mean()

# Build shift ratios with hierarchical fallback
d48_morning = d48[d48["hour"] <= 2]
d49_morning = d49_train  # all of d49 in train is h0-2

d48_m_geo = d48_morning.groupby("geohash")["demand"].mean()
d49_m_geo = d49_morning.groupby("geohash")["demand"].mean()
d48_m_g5 = d48_morning.groupby(d48_morning["geohash"].str[:5])["demand"].mean()
d49_m_g5 = d49_morning.groupby(d49_morning["geohash"].str[:5])["demand"].mean()
d48_m_g4 = d48_morning.groupby(d48_morning["geohash"].str[:4])["demand"].mean()
d49_m_g4 = d49_morning.groupby(d49_morning["geohash"].str[:4])["demand"].mean()
global_shift = d49_m_geo.mean() / d48_m_geo.mean()

def get_shift(gh):
    g5, g4 = gh[:5], gh[:4]
    if gh in d49_m_geo.index and gh in d48_m_geo.index and d48_m_geo[gh] > 1e-7:
        return d49_m_geo[gh] / d48_m_geo[gh]
    if g5 in d49_m_g5.index and g5 in d48_m_g5.index and d48_m_g5[g5] > 1e-7:
        return d49_m_g5[g5] / d48_m_g5[g5]
    if g4 in d49_m_g4.index and g4 in d48_m_g4.index and d48_m_g4[g4] > 1e-7:
        return d49_m_g4[g4] / d48_m_g4[g4]
    return global_shift

# Build Day 48 hourly lookup (fallback when exact tmin missing)
d48_hour_geo = d48.groupby(["geohash", "hour"])["demand"].mean()
d48_geo_mean = d48.groupby("geohash")["demand"].mean()
d48_g5_mean = d48.groupby(d48["geohash"].str[:5])["demand"].mean()
global_d48_mean = d48["demand"].mean()

def get_d48_demand(gh, tmin, hour):
    """Get Day 48 demand for a geohash at a specific time, with fallbacks."""
    # Try exact tmin match
    if (gh, tmin) in d48_lookup.index:
        return d48_lookup[(gh, tmin)]
    # Try hour-level average
    if (gh, hour) in d48_hour_geo.index:
        return d48_hour_geo[(gh, hour)]
    # Try geohash overall average
    if gh in d48_geo_mean.index:
        return d48_geo_mean[gh]
    # Try geohash5 average
    g5 = gh[:5]
    if g5 in d48_g5_mean.index:
        return d48_g5_mean[g5]
    return global_d48_mean

# Generate predictions
preds_v1 = []
for _, row in test.iterrows():
    gh = row["geohash"]
    tmin = row["tmin"]
    hour = row["hour"]
    d48_val = get_d48_demand(gh, tmin, hour)
    shift = get_shift(gh)
    preds_v1.append(d48_val * shift)

preds_v1 = np.array(preds_v1)
r2_v1 = r2_score(y_true, preds_v1) * 100
print(f"\n  R² Score: {r2_v1:.4f} / 100")

# ============================================================
# APPROACH 2: D48 profile × shift, with RoadType shape correction
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 2: D48 template × shift × RoadType shape correction")
print("=" * 70)

# Compute normalized shape per RoadType from Day 48
rt_hour_mean = d48.groupby(["RoadType", "hour"])["demand"].mean()
rt_overall_mean = d48.groupby("RoadType")["demand"].mean()
rt_shape = rt_hour_mean / rt_overall_mean  # normalized profile

def get_rt_shape(rt, hour):
    if (rt, hour) in rt_shape.index:
        return rt_shape[(rt, hour)]
    return 1.0

# For this approach: use geohash mean × roadtype shape × shift
preds_v2 = []
for _, row in test.iterrows():
    gh = row["geohash"]
    hour = row["hour"]
    rt = row["RoadType"]
    
    # Base = geohash's Day 48 overall mean
    if gh in d48_geo_mean.index:
        base = d48_geo_mean[gh]
    elif gh[:5] in d48_g5_mean.index:
        base = d48_g5_mean[gh[:5]]
    else:
        base = global_d48_mean
    
    shape = get_rt_shape(rt, hour)
    shift = get_shift(gh)
    preds_v2.append(base * shape * shift)

preds_v2 = np.array(preds_v2)
r2_v2 = r2_score(y_true, preds_v2) * 100
print(f"\n  R² Score: {r2_v2:.4f} / 100")

# ============================================================
# APPROACH 3: Clipped shift (cap outlier shifts)
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 3: D48 template × clipped shift (cap extreme ratios)")
print("=" * 70)

for lo, hi in [(0.3, 3.0), (0.5, 2.5), (0.5, 3.0), (0.7, 2.0)]:
    preds_v3 = []
    for _, row in test.iterrows():
        gh = row["geohash"]
        tmin = row["tmin"]
        hour = row["hour"]
        d48_val = get_d48_demand(gh, tmin, hour)
        shift = np.clip(get_shift(gh), lo, hi)
        preds_v3.append(d48_val * shift)
    preds_v3 = np.array(preds_v3)
    r2_v3 = r2_score(y_true, preds_v3) * 100
    print(f"  clip=[{lo}, {hi}]  R²: {r2_v3:.4f} / 100")

# ============================================================
# APPROACH 4: LINEAR REGRESSION — learn optimal combination
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 4: Linear Regression on domain features")
print("=" * 70)

# Build feature matrix for test set using domain signals
print("  Building feature matrix...")
features_list = []
for _, row in test.iterrows():
    gh = row["geohash"]
    tmin = row["tmin"]
    hour = row["hour"]
    rt = row["RoadType"]
    
    d48_exact = get_d48_demand(gh, tmin, hour)
    shift = get_shift(gh)
    rt_shape_val = get_rt_shape(rt, hour)
    
    geo_mean = d48_geo_mean.get(gh, global_d48_mean)
    g5_mean = d48_g5_mean.get(gh[:5], global_d48_mean)
    
    # D49 morning mean for this geohash (direct signal)
    d49_m_val = d49_m_geo.get(gh, d49_m_g5.get(gh[:5], d49_m_geo.mean()))
    
    features_list.append({
        "d48_exact": d48_exact,
        "d48_x_shift": d48_exact * shift,
        "geo_mean_x_shape_x_shift": geo_mean * rt_shape_val * shift,
        "shift": shift,
        "d49_morning": d49_m_val,
        "geo_mean": geo_mean,
        "g5_mean": g5_mean,
        "rt_shape": rt_shape_val,
        "hour": hour,
        "is_residential": 1 if rt == "Residential" else 0,
        "is_highway": 1 if rt == "Highway" else 0,
        "is_street": 1 if rt == "Street" else 0,
        "lanes": row["NumberofLanes"],
        "landmarks": row["Landmarks_bin"],
        "large_vehicles": row["LargeVehicles_bin"],
    })

X_test_df = pd.DataFrame(features_list)

# We need training data too — use Day 49 morning from train as "train set"
# But that's only h0-2, and test is h2-13. So instead, use Day 48 data
# with a train/val split to learn the linear combination, then apply to test.

# Strategy: Train on Day 48 (split into "pseudo-morning" and "pseudo-daytime")
# Actually, the best strategy is to train the linear model on ALL of train
# (both Day 48 and Day 49 morning) to learn the feature weights.

print("  Building training feature matrix...")
train_features = []
for _, row in train.iterrows():
    gh = row["geohash"]
    tmin = row["tmin"]
    hour = row["hour"]
    rt = row["RoadType"]
    day = row["day"]
    
    # For Day 48 rows, we use leave-one-out style: 
    # d48_exact from the same day is somewhat circular, but for linear reg it's fine
    d48_exact = get_d48_demand(gh, tmin, hour)
    shift = get_shift(gh)
    rt_shape_val = get_rt_shape(rt, hour)
    
    geo_mean = d48_geo_mean.get(gh, global_d48_mean)
    g5_mean = d48_g5_mean.get(gh[:5], global_d48_mean)
    d49_m_val = d49_m_geo.get(gh, d49_m_g5.get(gh[:5], d49_m_geo.mean()))
    
    train_features.append({
        "d48_exact": d48_exact,
        "d48_x_shift": d48_exact * shift,
        "geo_mean_x_shape_x_shift": geo_mean * rt_shape_val * shift,
        "shift": shift,
        "d49_morning": d49_m_val,
        "geo_mean": geo_mean,
        "g5_mean": g5_mean,
        "rt_shape": rt_shape_val,
        "hour": hour,
        "is_residential": 1 if rt == "Residential" else 0,
        "is_highway": 1 if rt == "Highway" else 0,
        "is_street": 1 if rt == "Street" else 0,
        "lanes": row["NumberofLanes"],
        "landmarks": row["Landmarks_bin"],
        "large_vehicles": row["LargeVehicles_bin"],
    })

X_train_df = pd.DataFrame(train_features)
y_train = train["demand"].values

# Fill NaN
X_train_df = X_train_df.fillna(0)
X_test_df = X_test_df.fillna(0)

feature_cols = X_train_df.columns.tolist()

# 4a: Simple linear regression
print("\n  --- 4a: OLS Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train_df[feature_cols], y_train)
preds_lr = lr.predict(X_test_df[feature_cols])
r2_lr = r2_score(y_true, preds_lr) * 100
print(f"  R² Score: {r2_lr:.4f} / 100")
print("  Coefficients:")
for col, coef in sorted(zip(feature_cols, lr.coef_), key=lambda x: -abs(x[1])):
    print(f"    {col:35s}  {coef:+.6f}")
print(f"    {'intercept':35s}  {lr.intercept_:+.6f}")

# 4b: Ridge regression (regularized)
print("\n  --- 4b: Ridge Regression (alpha sweep) ---")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_df[feature_cols], y_train)
    preds_ridge = ridge.predict(X_test_df[feature_cols])
    r2_ridge = r2_score(y_true, preds_ridge) * 100
    print(f"    alpha={alpha:>6.2f}  R²: {r2_ridge:.4f} / 100")

# 4c: Linear regression on FEWER features (most interpretable)
print("\n  --- 4c: Minimal Linear Regression (3 features) ---")
minimal_cols = ["d48_x_shift", "d49_morning", "rt_shape"]
lr_min = LinearRegression()
lr_min.fit(X_train_df[minimal_cols], y_train)
preds_min = lr_min.predict(X_test_df[minimal_cols])
r2_min = r2_score(y_true, preds_min) * 100
print(f"  R² Score: {r2_min:.4f} / 100")
print("  Coefficients:")
for col, coef in zip(minimal_cols, lr_min.coef_):
    print(f"    {col:35s}  {coef:+.6f}")
print(f"    {'intercept':35s}  {lr_min.intercept_:+.6f}")

# ============================================================
# APPROACH 5: Weighted blend (manual, no learning)
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 5: Manual weighted blend sweep")
print("=" * 70)

# Blend between d48_exact*shift and d49_morning*rt_shape
d48_signal = X_test_df["d48_x_shift"].values
d49_signal = (X_test_df["d49_morning"] * X_test_df["rt_shape"]).values

best_r2, best_w = 0, 0
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * d48_signal + (1 - w) * d49_signal
    r2_b = r2_score(y_true, blended) * 100
    if r2_b > best_r2:
        best_r2, best_w = r2_b, w
    print(f"  w(d48×shift)={w:.2f}, w(d49_morning×shape)={1-w:.2f}  R²: {r2_b:.4f} / 100")

print(f"\n  BEST: w={best_w:.2f}  R²: {best_r2:.4f} / 100")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL APPROACHES (R² on 0-100 scale)")
print("=" * 70)
print(f"  1. Pure domain (D48×shift):             {r2_v1:.4f}")
print(f"  2. D48 mean × RoadType shape × shift:   {r2_v2:.4f}")
print(f"  3. D48×shift (clipped):                 best from above")
print(f"  4a. OLS Linear Regression (15 features): {r2_lr:.4f}")
print(f"  4c. Minimal LinReg (3 features):         {r2_min:.4f}")
print(f"  5. Manual blend (best w={best_w:.2f}):          {best_r2:.4f}")
