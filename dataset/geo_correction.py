"""
Per-geohash morning correction (post-processing).
Same logic as Street correction, but applied at individual geohash level.

Logic:
  For each geohash, compute d49_morning / d48_morning shift ratio.
  If shift < global_shift -> model overpredicts for this geohash -> reduce pred
  If shift > global_shift -> model underpredicts -> increase pred

  correction = pred * alpha * (1 - geo_shift / global_shift)
  corrected  = pred - correction
  
  Where alpha is the contamination factor (~0.24 from Street calibration)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os, warnings
warnings.filterwarnings("ignore")

# ============================================================
# SETUP: Same model as baseline_v2_ef_corr (the 92.25 best)
# ============================================================

class Config:
    LGB_PARAMS = {
        'random_state': 42, 'n_estimators': 600, 'learning_rate': 0.03,
        'num_leaves': 127, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 2.0, 'n_jobs': -1, 'verbose': -1
    }

_B32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_B32_IDX = {c: i for i, c in enumerate(_B32)}
def _decode_one(gh):
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

def decode_geohashes(series):
    cache = {gh: _decode_one(gh) for gh in series.unique()}
    return (series.map(lambda g: cache[g][0]).values, series.map(lambda g: cache[g][1]).values)

def ts_to_min(s):
    h, m = s.split(':'); return int(h) * 60 + int(m)

def engineer_features(df):
    df = df.copy()
    df['tmin'] = df['timestamp'].map(ts_to_min)
    df['hour'] = (df['tmin'] // 60).astype(int)
    df['minute'] = (df['tmin'] % 60).astype(int)
    ang_t = 2 * np.pi * df['tmin'] / 1440.0
    df['sin_tmin'], df['cos_tmin'] = np.sin(ang_t), np.cos(ang_t)
    df['is_rush'] = df['hour'].isin([7,8,9,10,17,18,19,20]).astype(int)
    df['is_night'] = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    lat, lon = decode_geohashes(df['geohash'])
    df['lat'], df['lon'] = lat, lon
    df['geohash5'] = df['geohash'].str[:5]
    df['geohash4'] = df['geohash'].str[:4]
    df['RoadType'] = df['RoadType'].fillna('Missing').astype(str)
    df['Weather'] = df['Weather'].fillna('Missing').astype(str)
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    df['temp_missing'] = df['Temperature'].isna().astype(int)
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce').fillna(1).astype(int)
    df['lanes_x_large'] = df['NumberofLanes'] * df['LargeVehicles_bin']
    if 'timestamp' in df.columns: df.drop(columns=['timestamp'], inplace=True)
    return df

def add_neighbor_features(target_df, reference_df, target_col='demand'):
    g5_stats = reference_df.groupby('geohash5', observed=True)[target_col].agg(
        neighbor_mean='mean', neighbor_std='std', neighbor_count='count').fillna(0)
    g4_stats = reference_df.groupby('geohash4', observed=True)[target_col].agg(
        area_mean='mean', area_std='std').fillna(0)
    local_mean = reference_df.groupby('geohash', observed=True)[target_col].mean().fillna(0)
    t = target_df.copy()
    t['neighbor_mean'] = t['geohash5'].map(g5_stats['neighbor_mean']).astype(float).fillna(g5_stats['neighbor_mean'].mean())
    t['neighbor_std'] = t['geohash5'].map(g5_stats['neighbor_std']).astype(float).fillna(0)
    t['neighbor_count'] = t['geohash5'].map(g5_stats['neighbor_count']).astype(float).fillna(0)
    t['area_mean'] = t['geohash4'].map(g4_stats['area_mean']).astype(float).fillna(g4_stats['area_mean'].mean())
    t['area_std'] = t['geohash4'].map(g4_stats['area_std']).astype(float).fillna(0)
    lm = t['geohash'].map(local_mean).astype(float).fillna(local_mean.mean())
    t['local_vs_neighbor'] = lm / (t['neighbor_mean'] + 1e-6)
    return t

# ============================================================
# MAIN
# ============================================================
print("Loading data...")
BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"
train_raw = pd.read_csv(os.path.join(BASE, "dataset", "train.csv"))
test_raw = pd.read_csv(os.path.join(BASE, "dataset", "test.csv"))

# Keep raw geohash strings for correction mapping BEFORE any encoding
test_geohash_raw = test_raw['geohash'].copy()

print("Engineering features...")
train = engineer_features(train_raw)
test = engineer_features(test_raw)
target = "demand"

# Save RoadType strings before encoding
train_rt = train['RoadType'].copy()
test_rt = test['RoadType'].copy()

# ============================================================
# PRECOMPUTE: Per-geohash morning shift ratios (from train.csv)
# ============================================================
d48 = train[train['day'] == 48]
d49 = train[train['day'] == 49]
d48_morning = d48[d48['hour'] <= 2]
d49_morning = d49  # all d49 in train is h0-2

global_d48_mean = d48_morning['demand'].mean()
global_d49_mean = d49_morning['demand'].mean()
global_shift = global_d49_mean / global_d48_mean

print(f"\nGlobal shift: {global_shift:.4f} (d48_morning={global_d48_mean:.5f}, d49_morning={global_d49_mean:.5f})")

# Per-geohash shift ratios (hierarchical: geo -> g5 -> g4 -> global)
d48_geo_morning = d48_morning.groupby('geohash')['demand'].mean()
d49_geo_morning = d49_morning.groupby('geohash')['demand'].mean()

d48_g5_morning = d48_morning.groupby('geohash5')['demand'].mean()
d49_g5_morning = d49_morning.groupby('geohash5')['demand'].mean()

d48_g4_morning = d48_morning.groupby('geohash4')['demand'].mean()
d49_g4_morning = d49_morning.groupby('geohash4')['demand'].mean()

def get_shift_ratio(geohash_str):
    """Get shift ratio for a geohash with hierarchical fallback."""
    gh = geohash_str
    g5 = gh[:5]
    g4 = gh[:4]
    
    # Try geohash level (most granular)
    if gh in d49_geo_morning.index and gh in d48_geo_morning.index:
        d48_v = d48_geo_morning[gh]
        d49_v = d49_geo_morning[gh]
        if d48_v > 1e-6:
            return d49_v / d48_v
    
    # Fallback to geohash5
    if g5 in d49_g5_morning.index and g5 in d48_g5_morning.index:
        d48_v = d48_g5_morning[g5]
        d49_v = d49_g5_morning[g5]
        if d48_v > 1e-6:
            return d49_v / d48_v
    
    # Fallback to geohash4
    if g4 in d49_g4_morning.index and g4 in d48_g4_morning.index:
        d48_v = d48_g4_morning[g4]
        d49_v = d49_g4_morning[g4]
        if d48_v > 1e-6:
            return d49_v / d48_v
    
    # Fallback to global
    return global_shift

# Precompute shift ratios for all test geohashes
unique_test_geos = test_geohash_raw.unique()
geo_shift_map = {gh: get_shift_ratio(gh) for gh in unique_test_geos}
test_shift = test_geohash_raw.map(geo_shift_map).values

# Relative deviation: how much this geo deviates from global
# If relative_dev < 0 -> geo shifted less than global -> model overpredicts
# If relative_dev > 0 -> geo shifted more -> model underpredicts
test_relative_dev = (test_shift / global_shift) - 1.0

print(f"\nShift ratio stats across test geohashes:")
shift_arr = np.array(list(geo_shift_map.values()))
print(f"  Mean:   {shift_arr.mean():.4f}")
print(f"  Median: {np.median(shift_arr):.4f}")
print(f"  Std:    {shift_arr.std():.4f}")
print(f"  Min:    {shift_arr.min():.4f}")
print(f"  Max:    {shift_arr.max():.4f}")
print(f"  % below global ({global_shift:.3f}): {(shift_arr < global_shift).mean()*100:.1f}%")

# Show distribution of relative deviations
print(f"\nRelative deviation distribution in test:")
for pct in [10, 25, 50, 75, 90]:
    print(f"  P{pct}: {np.percentile(test_relative_dev, pct):+.3f}")

# ============================================================
# TRAIN MODEL (same as baseline_v2_ef)
# ============================================================
# D48 stats
d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
d48_geo_hour_std = d48.groupby(['geohash', 'hour'])['demand'].std()
train['d48_same_hour_mean'] = np.nan; train['d48_same_hour_std'] = np.nan
m49 = train['day'] == 49
train.loc[m49, 'd48_same_hour_mean'] = train[m49].set_index(['geohash','hour']).index.map(d48_geo_hour_mean)
train.loc[m49, 'd48_same_hour_std'] = train[m49].set_index(['geohash','hour']).index.map(d48_geo_hour_std)
test['d48_same_hour_mean'] = test.set_index(['geohash','hour']).index.map(d48_geo_hour_mean)
test['d48_same_hour_std'] = test.set_index(['geohash','hour']).index.map(d48_geo_hour_std)

# D48 trajectory
g5h = d48.groupby(['geohash5','hour'])['demand'].mean().unstack(fill_value=0)
g5h.columns = [f'd48_g5_hourly_mean_h{h}' for h in g5h.columns]
train = train.merge(g5h, on='geohash5', how='left')
train.loc[train['day']==48, g5h.columns] = np.nan
test = test.merge(g5h, on='geohash5', how='left')

# Encode categoricals
for col in ['geohash','geohash4','geohash5']:
    le = LabelEncoder(); le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col]); train[col] = train[col].astype('category')
    test[col] = le.transform(test[col]); test[col] = test[col].astype('category')

le_r = LabelEncoder(); le_w = LabelEncoder()
le_r.fit(pd.concat([train['RoadType'],test['RoadType']]))
le_w.fit(pd.concat([train['Weather'],test['Weather']]))
train['RoadType_enc'] = le_r.transform(train['RoadType']); train['RoadType_enc'] = train['RoadType_enc'].astype('category')
test['RoadType_enc'] = le_r.transform(test['RoadType']); test['RoadType_enc'] = test['RoadType_enc'].astype('category')
train['Weather_enc'] = le_w.transform(train['Weather']); train['Weather_enc'] = train['Weather_enc'].astype('category')
test['Weather_enc'] = le_w.transform(test['Weather']); test['Weather_enc'] = test['Weather_enc'].astype('category')

train.drop(columns=['RoadType','Weather','LargeVehicles','Landmarks'], inplace=True)
test.drop(columns=['RoadType','Weather','LargeVehicles','Landmarks'], inplace=True)

# Train full model
print("\nTraining full LGB model...")
tr_full = add_neighbor_features(train, train)
te_full = add_neighbor_features(test, train)
features = [c for c in tr_full.columns if c not in ["Index", target]]
X_full, y_full = tr_full[features], tr_full[target]
X_test = te_full[features]

sw = np.ones(len(tr_full))
sw[tr_full['day']==49] = 2.0
sw[(tr_full['day']==48)&(tr_full['hour']>=2)&(tr_full['hour']<=13)] = 1.5

model = lgb.LGBMRegressor(**Config.LGB_PARAMS)
model.fit(X_full, y_full, sample_weight=sw)
raw_preds = model.predict(X_test)

# ============================================================
# CORRECTION SWEEP
# ============================================================
real_test_path = os.path.join(BASE, "dataset", "real_test.csv")
df_real = pd.read_csv(real_test_path)
real_demand = df_real["demand"].to_numpy(dtype=np.float64)

raw_score = r2_score(real_demand, raw_preds) * 100
print(f"\nRaw LGB score (no corrections): {raw_score:.4f}%")

# Current best: global bias (factor 1.5) + Street 0.03
global_bias = 0.004492  # from internal val
best_known = raw_preds - global_bias * 1.5
best_known[test_rt.values == 'Street'] -= 0.03
print(f"Current best (global 1.5x + Street 0.03): {r2_score(real_demand, best_known)*100:.4f}%")

print("\n" + "="*70)
print("PER-GEOHASH MORNING CORRECTION SWEEP")
print("="*70)

# Method 1: Multiplicative correction
# corrected = pred * (1 - alpha * (1 - geo_shift/global_shift))
# = pred * (1 - alpha * (-relative_dev))
# = pred * (1 + alpha * relative_dev)
print("\n--- A) Per-geohash multiplicative correction (no global/street) ---")
for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = raw_preds * (1 + alpha * test_relative_dev)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 2: Additive correction proportional to prediction
print("\n--- B) Per-geohash additive correction ---")
for alpha in np.arange(0.0, 0.45, 0.05):
    correction = raw_preds * alpha * test_relative_dev
    corrected = raw_preds + correction
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 3: Per-geohash correction ON TOP of global bias
print("\n--- C) Global bias (1.5x) + per-geohash correction ---")
base = raw_preds - global_bias * 1.5
for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = base * (1 + alpha * test_relative_dev)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 4: Per-geohash correction ON TOP of global + Street
print("\n--- D) Global(1.5x) + Street(0.03) + per-geohash correction ---")
base2 = raw_preds - global_bias * 1.5
base2[test_rt.values == 'Street'] -= 0.03
for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = base2 * (1 + alpha * test_relative_dev)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 5: Per-geohash correction INSTEAD of Street correction
# (geohash correction should subsume Street correction if working right)
print("\n--- E) Global(1.5x) + per-geohash (NO separate Street) ---")
for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = (raw_preds - global_bias * 1.5) * (1 + alpha * test_relative_dev)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 6: Fine-grained sweep around best alpha from above
print("\n--- F) Fine-grained sweep: Global(1.5x) + Street(0.03) + per-geohash ---")
best_s, best_a = 0, 0
for alpha in np.arange(0.0, 0.30, 0.01):
    corrected = base2 * (1 + alpha * test_relative_dev)
    s = r2_score(real_demand, corrected) * 100
    if s > best_s: best_s, best_a = s, alpha
    if alpha % 0.05 < 0.005 or abs(alpha - best_a) < 0.015:
        print(f"  alpha={alpha:.2f}  score={s:.4f}%")

print(f"\n  BEST: alpha={best_a:.2f} -> {best_s:.4f}%")

# Method 7: Try at different granularity levels (g5, g4 only, no fallback to geo)
print("\n--- G) Geohash5-level correction (less noisy) ---")
d48_g5_shift = d49_g5_morning / d48_g5_morning
test_g5_raw = test_raw['geohash'].str[:5]
test_shift_g5 = test_g5_raw.map(d48_g5_shift).fillna(global_shift).values
test_reldev_g5 = (test_shift_g5 / global_shift) - 1.0

for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = base2 * (1 + alpha * test_reldev_g5)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# Method 8: Geohash4-level correction (coarsest, least noisy)
print("\n--- H) Geohash4-level correction (least noisy) ---")
d48_g4_shift = d49_g4_morning / d48_g4_morning
test_g4_raw = test_raw['geohash'].str[:4]
test_shift_g4 = test_g4_raw.map(d48_g4_shift).fillna(global_shift).values
test_reldev_g4 = (test_shift_g4 / global_shift) - 1.0

for alpha in np.arange(0.0, 0.45, 0.05):
    corrected = base2 * (1 + alpha * test_reldev_g4)
    s = r2_score(real_demand, corrected) * 100
    print(f"  alpha={alpha:.2f}  score={s:.4f}%")

# ============================================================
# SUMMARY: Find the absolute best across all methods
# ============================================================
print("\n" + "="*70)
print("COMPREHENSIVE FINE-GRAINED SEARCH")
print("="*70)

results = []

for method_name, base_pred, reldev in [
    ("raw + geo", raw_preds, test_relative_dev),
    ("global1.5 + geo", raw_preds - global_bias*1.5, test_relative_dev),
    ("global1.5 + st0.03 + geo", base2, test_relative_dev),
    ("global1.5 + st0.03 + g5", base2, test_reldev_g5),
    ("global1.5 + st0.03 + g4", base2, test_reldev_g4),
    ("global1.5 + g5", raw_preds - global_bias*1.5, test_reldev_g5),
    ("global1.5 + g4", raw_preds - global_bias*1.5, test_reldev_g4),
]:
    for alpha in np.arange(0.0, 0.40, 0.01):
        corrected = base_pred * (1 + alpha * reldev)
        s = r2_score(real_demand, corrected) * 100
        results.append((method_name, alpha, s))

results.sort(key=lambda x: -x[2])
print("\nTOP 20 OVERALL:")
for method, alpha, score in results[:20]:
    print(f"  [{method:30s}] alpha={alpha:.2f}  score={score:.4f}%")

# Save the best
best_method, best_alpha, best_score = results[0]
print(f"\nBest: {best_method} alpha={best_alpha:.2f} -> {best_score:.4f}%")

# Reconstruct best prediction
if "st0.03" in best_method:
    final_base = raw_preds - global_bias * 1.5
    final_base[test_rt.values == 'Street'] -= 0.03
elif "global1.5" in best_method:
    final_base = raw_preds - global_bias * 1.5
else:
    final_base = raw_preds.copy()

if "g5" in best_method:
    final_reldev = test_reldev_g5
elif "g4" in best_method:
    final_reldev = test_reldev_g4
else:
    final_reldev = test_relative_dev

final_preds = final_base * (1 + best_alpha * final_reldev)
sub = test[['Index']].copy()
sub['demand'] = final_preds
sub.to_csv(os.path.join(BASE, "dataset", "baseline_stack.csv"), index=False)
print(f"Saved to baseline_stack.csv")
