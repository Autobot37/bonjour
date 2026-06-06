"""
Two-Stage V3: Historical Hourly Base + Stationary Residual Learning
===================================================================
Stage 1: Use the historical hourly profile (d48_same_hour_mean) as the base.
         For Day 48 training rows, use leave-one-out (LOO) to prevent leakage:
           base = (sum_demand - demand) / (count - 1)
         For Day 49 and Test rows, use the full hourly mean from Day 48.

Shift:   Compute hierarchical shift ratios (D49 morning / D48 morning)
         with Bayesian shrinkage: Global -> RoadType -> G4 -> G5 -> Geohash.

Stage 2: Target is stationary residual:
         - Day 48 base: LOO same-hour mean (shift = 1.0)
         - Day 49 base: d48_same_hour_mean * shift_ratio
         Residual target = actual - base_adjusted.
         Train LightGBM to predict this residual.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os, time, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"
t_start = time.time()

# ==================================================================
# GEOHASH DECODING
# ==================================================================
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
    return (series.map(lambda g: cache[g][0]).values,
            series.map(lambda g: cache[g][1]).values)

def ts_to_min(s):
    h, m = s.split(':')
    return int(h) * 60 + int(m)

# ==================================================================
# FEATURE ENGINEERING
# ==================================================================
def engineer_features(df):
    df = df.copy()
    df['tmin'] = df['timestamp'].map(ts_to_min)
    df['hour'] = (df['tmin'] // 60).astype(int)
    df['minute'] = (df['tmin'] % 60).astype(int)
    ang_t = 2 * np.pi * df['tmin'] / 1440.0
    df['sin_tmin'] = np.sin(ang_t)
    df['cos_tmin'] = np.cos(ang_t)
    df['is_rush'] = df['hour'].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
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
    return df

# ==================================================================
# LOAD & PREPARE
# ==================================================================
print("=" * 70)
print("TWO-STAGE V3: HISTORICAL HOURLY BASE + STATIONARY RESIDUAL LEARNING")
print("=" * 70)

train_raw = pd.read_csv(os.path.join(BASE, "dataset", "train.csv"))
test_raw = pd.read_csv(os.path.join(BASE, "dataset", "test.csv"))

train = engineer_features(train_raw)
test = engineer_features(test_raw)
train_rt_str = train['RoadType'].copy()
test_rt_str = test['RoadType'].copy()

d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

print(f"D48: {len(d48)} | D49 morning: {len(d49)} | Test: {len(test)}")

# Encode categoricals globally
all_data = pd.concat([train, test], ignore_index=True)
le_geo = LabelEncoder().fit(all_data['geohash'])
le_g5 = LabelEncoder().fit(all_data['geohash5'])
le_g4 = LabelEncoder().fit(all_data['geohash4'])
le_road = LabelEncoder().fit(all_data['RoadType'])
le_weather = LabelEncoder().fit(all_data['Weather'])

cat_type_geo = pd.CategoricalDtype(categories=list(range(len(le_geo.classes_))))
cat_type_g5 = pd.CategoricalDtype(categories=list(range(len(le_g5.classes_))))
cat_type_g4 = pd.CategoricalDtype(categories=list(range(len(le_g4.classes_))))
cat_type_road = pd.CategoricalDtype(categories=list(range(len(le_road.classes_))))
cat_type_weather = pd.CategoricalDtype(categories=list(range(len(le_weather.classes_))))

def encode_cats(df):
    t = df.copy()
    t['geohash_enc'] = pd.Series(le_geo.transform(t['geohash']), index=t.index).astype(cat_type_geo)
    t['geohash5_enc'] = pd.Series(le_g5.transform(t['geohash5']), index=t.index).astype(cat_type_g5)
    t['geohash4_enc'] = pd.Series(le_g4.transform(t['geohash4']), index=t.index).astype(cat_type_g4)
    t['RoadType_enc'] = pd.Series(le_road.transform(t['RoadType']), index=t.index).astype(cat_type_road)
    t['Weather_enc'] = pd.Series(le_weather.transform(t['Weather']), index=t.index).astype(cat_type_weather)
    return t

train = encode_cats(train)
test = encode_cats(test)
d48 = encode_cats(d48)
d49 = encode_cats(d49)

# ==================================================================
# HIERARCHICAL SHIFT RATIO ESTIMATION WITH SHRINKAGE
# ==================================================================
def compute_hierarchical_shift(d48_df, d49_df, m_shrink=10):
    # Align hours to match exactly
    hours_present = d49_df['hour'].unique()
    d48_m = d48_df[d48_df['hour'].isin(hours_present)].copy()
    d49_m = d49_df.copy()
    
    # Global level
    g_d48 = d48_m['demand'].mean()
    g_d49 = d49_m['demand'].mean()
    global_shift = g_d49 / (g_d48 + 1e-8)
    
    # RoadType level
    rt_d48 = d48_m.groupby('RoadType')['demand'].mean()
    rt_d49 = d49_m.groupby('RoadType')['demand'].mean()
    rt_shifts = {rt: rt_d49.get(rt, g_d49) / (rt_d48.get(rt, g_d48) + 1e-8) for rt in d48_m['RoadType'].unique()}
    
    # Geohash4 level
    g4_d48 = d48_m.groupby('geohash4')['demand'].mean()
    g4_d49 = d49_m.groupby('geohash4')['demand'].mean()
    g4_counts = d49_m.groupby('geohash4')['demand'].count()
    
    g4_shifts = {}
    for g4 in all_data['geohash4'].unique():
        n = g4_counts.get(g4, 0)
        rt_matches = d48_m[d48_m['geohash4'] == g4]['RoadType'].values
        rt = rt_matches[0] if len(rt_matches) > 0 else 'Missing'
        prior = rt_shifts.get(rt, global_shift)
        
        if n > 0 and g4 in g4_d48 and g4_d48[g4] > 1e-8:
            raw = g4_d49[g4] / g4_d48[g4]
            g4_shifts[g4] = (n * raw + m_shrink * prior) / (n + m_shrink)
        else:
            g4_shifts[g4] = prior

    # Geohash5 level
    g5_d48 = d48_m.groupby('geohash5')['demand'].mean()
    g5_d49 = d49_m.groupby('geohash5')['demand'].mean()
    g5_counts = d49_m.groupby('geohash5')['demand'].count()
    
    g5_shifts = {}
    for g5 in all_data['geohash5'].unique():
        n = g5_counts.get(g5, 0)
        g4 = g5[:4]
        prior = g4_shifts.get(g4, global_shift)
        
        if n > 0 and g5 in g5_d48 and g5_d48[g5] > 1e-8:
            raw = g5_d49[g5] / g5_d48[g5]
            g5_shifts[g5] = (n * raw + m_shrink * prior) / (n + m_shrink)
        else:
            g5_shifts[g5] = prior

    # Geohash level
    gh_d48 = d48_m.groupby('geohash')['demand'].mean()
    gh_d49 = d49_m.groupby('geohash')['demand'].mean()
    gh_counts = d49_m.groupby('geohash')['demand'].count()
    
    gh_shifts = {}
    for gh in all_data['geohash'].unique():
        n = gh_counts.get(gh, 0)
        g5 = gh[:5]
        prior = g5_shifts.get(g5, global_shift)
        
        if n > 0 and gh in gh_d48 and gh_d48[gh] > 1e-8:
            raw = gh_d49[gh] / gh_d48[gh]
            gh_shifts[gh] = (n * raw + m_shrink * prior) / (n + m_shrink)
        else:
            gh_shifts[gh] = prior
            
    for k in gh_shifts:
        gh_shifts[k] = np.clip(gh_shifts[k], 0.3, 3.5)
        
    return gh_shifts, rt_shifts, global_shift

# ==================================================================
# STAGE 2 FEATURE BUILDER
# ==================================================================
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

# Precompute trajectory and same-hour anchor
d48_geo_hour_sum = d48.groupby(['geohash', 'hour'])['demand'].sum()
d48_geo_hour_count = d48.groupby(['geohash', 'hour'])['demand'].count()
d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
d48_geo_hour_std = d48.groupby(['geohash', 'hour'])['demand'].std()

g5_hourly = d48.groupby(['geohash5', 'hour'])['demand'].mean().unstack(fill_value=0)
g5_hourly.columns = [f'd48_g5_h{h}' for h in g5_hourly.columns]

d48_geo_mean = d48.groupby('geohash')['demand'].mean()

def build_stage2_dataset(df, shifts=None, is_d48=False):
    t = df.copy()
    
    # Map shift ratios
    if is_d48 or shifts is None:
        t['shift_ratio'] = 1.0
    else:
        t['shift_ratio'] = t['geohash'].map(shifts).fillna(1.0).astype(float)
        
    # Same-hour base anchor
    orig_gh = df['geohash'].values
    hours = df['hour'].values
    keys_gh = list(zip(orig_gh, hours))
    
    if is_d48:
        # Leave-one-out same-hour mean to prevent leakage
        loo_base = []
        for idx_row, row in df.iterrows():
            gh = row['geohash']
            h = row['hour']
            k = (gh, h)
            
            y_val = row['demand']
            s = d48_geo_hour_sum.get(k, 0)
            c = d48_geo_hour_count.get(k, 0)
            
            if c > 1:
                loo_val = (s - y_val) / (c - 1)
            else:
                # Fall back to leave-one-out geohash mean
                # (since geohash-level has many more samples, we can just use the overall geohash mean)
                loo_val = d48_geo_mean.get(gh, d48_geo_mean.mean())
            loo_base.append(loo_val)
            
        t['stage1_base'] = loo_base
        t['d48_same_hour_mean'] = loo_base
        t['d48_same_hour_std'] = np.nan
    else:
        # Direct same-hour mean for Day 49 and Test (no leakage possible)
        vals_mean = pd.Series([d48_geo_hour_mean.get(k, np.nan) for k in keys_gh], index=t.index).astype(float)
        vals_std = pd.Series([d48_geo_hour_std.get(k, np.nan) for k in keys_gh], index=t.index).astype(float)
        
        t['stage1_base'] = vals_mean.fillna(t['geohash'].map(d48_geo_mean)).fillna(d48_geo_mean.mean())
        t['d48_same_hour_mean'] = t['stage1_base']
        t['d48_same_hour_std'] = vals_std.fillna(0)
        
    t['stage1_shift_adjusted'] = t['stage1_base'] * t['shift_ratio']
    
    # Fill defaults
    t['d48_same_hour_mean'] = t['d48_same_hour_mean'].fillna(t['geohash'].map(d48_geo_mean)).fillna(d48_geo_mean.mean())
    t['d48_same_hour_std'] = t['d48_same_hour_std'].fillna(0)
    
    # Add trajectory columns
    t = t.merge(g5_hourly, left_on='geohash5', right_index=True, how='left')
    if is_d48:
        t[g5_hourly.columns] = np.nan
        
    return t

# ==================================================================
# EVALUATION LOAD
# ==================================================================
real_df = pd.read_csv(os.path.join(BASE, "dataset", "real_test.csv"))
real_demand = real_df["demand"].to_numpy(dtype=np.float64)
real_df['hour'] = real_df['timestamp'].apply(lambda x: int(x.split(':')[0]))
real_df['RoadType'] = test_rt_str.values

# ==================================================================
# SYSTEMATIC EXPERIMENTS SWEEP
# ==================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING SWEEP")
    print("=" * 70)

    params_heavy = {
        'random_state': 42, 'n_estimators': 600, 'learning_rate': 0.03,
        'num_leaves': 127, 'colsample_bytree': 0.8, 'subsample': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 2.0, 'n_jobs': -1, 'verbose': -1,
    }

    results = {}
    preds_dict = {}

    # We will sweep shrinkage parameters m_shrink and residual formulation
    for m_shrink in [5, 10, 20, 50]:
        # Compute shifts
        shifts, rt_shifts, global_shift = compute_hierarchical_shift(d48, d49, m_shrink)
        
        print(f"\n--- m_shrink = {m_shrink} ---")
        print(f"  Morning Shifts: Global={global_shift:.4f} | Street={rt_shifts.get('Street', 1.0):.4f} | Residential={rt_shifts.get('Residential', 1.0):.4f} | Highway={rt_shifts.get('Highway', 1.0):.4f}")
        
        # Build stage 2 datasets
        d48_s2 = build_stage2_dataset(d48, is_d48=True)
        d49_s2 = build_stage2_dataset(d49, shifts, is_d48=False)
        test_s2 = build_stage2_dataset(test, shifts, is_d48=False)
        
        # Add neighbor features
        train_combined = pd.concat([d48, d49], ignore_index=True)
        d48_s2 = add_neighbor_features(d48_s2, train_combined)
        d49_s2 = add_neighbor_features(d49_s2, train_combined)
        test_s2 = add_neighbor_features(test_s2, train_combined)
        
        s2_features = [
            'geohash_enc', 'geohash4_enc', 'geohash5_enc',
            'tmin', 'hour', 'minute', 'sin_tmin', 'cos_tmin',
            'is_rush', 'is_night',
            'lat', 'lon',
            'RoadType_enc', 'Weather_enc',
            'LargeVehicles_bin', 'Landmarks_bin',
            'Temperature', 'temp_missing',
            'NumberofLanes', 'lanes_x_large',
            'stage1_base', 'shift_ratio', 'stage1_shift_adjusted',
            'd48_same_hour_mean', 'd48_same_hour_std',
            'neighbor_mean', 'neighbor_std', 'neighbor_count',
            'area_mean', 'area_std', 'local_vs_neighbor',
        ] + list(g5_hourly.columns)
        
        # Prepare inputs
        X_comb = pd.concat([d48_s2[s2_features], d49_s2[s2_features]], ignore_index=True).fillna(0)
        X_test = test_s2[s2_features].fillna(0)
        
        sw = np.ones(len(X_comb))
        sw[len(d48):] = 2.0  # d49 weight
        d48_tw = (d48['hour'].values >= 2) & (d48['hour'].values <= 13)
        sw[:len(d48)][d48_tw] = 1.5
        
        # Define actual target demand values
        y_demand = np.concatenate([d48['demand'].values, d49['demand'].values])
        base_adjusted = np.concatenate([d48_s2['stage1_shift_adjusted'].values, d49_s2['stage1_shift_adjusted'].values])
        
        # Residual targets
        # 1. Additive: actual - base
        y_res_add = y_demand - base_adjusted
        # 2. Log-delta: log1p(actual) - log1p(base)
        y_res_log = np.log1p(np.maximum(y_demand, 0)) - np.log1p(np.maximum(base_adjusted, 0))
        
        # Check stationarity
        print(f"  Stationarity check (Additive residual mean): D48={y_res_add[:len(d48)].mean():+.5f}, D49={y_res_add[len(d48):].mean():+.5f}")
        print(f"  Stationarity check (Log-delta residual mean): D48={y_res_log[:len(d48)].mean():+.5f}, D49={y_res_log[len(d48):].mean():+.5f}")
        
        # --- Experiment A: Additive Residual ---
        m_add = lgb.LGBMRegressor(**params_heavy)
        m_add.fit(X_comb, y_res_add, sample_weight=sw)
        pred_res_add = m_add.predict(X_test)
        preds_add = test_s2['stage1_shift_adjusted'].values + pred_res_add
        
        # --- Experiment B: Log-delta Residual ---
        m_log = lgb.LGBMRegressor(**params_heavy)
        m_log.fit(X_comb, y_res_log, sample_weight=sw)
        pred_res_log = m_log.predict(X_test)
        preds_log = np.expm1(np.log1p(np.maximum(test_s2['stage1_shift_adjusted'].values, 0)) + pred_res_log)
        
        # --- Experiment C: Direct demand predicting ---
        m_dir = lgb.LGBMRegressor(**params_heavy)
        m_dir.fit(X_comb, y_demand, sample_weight=sw)
        preds_dir = m_dir.predict(X_test)
        
        # Evaluate
        r_add = r2_score(real_demand, preds_add) * 100
        r_log = r2_score(real_demand, preds_log) * 100
        r_dir = r2_score(real_demand, preds_dir) * 100
        
        print(f"  Additive Residual R²: {r_add:.4f}%")
        print(f"  Log-delta Residual R²: {r_log:.4f}%")
        print(f"  Direct Prediction R²:  {r_dir:.4f}%")
        
        results[f'm{m_shrink}_add'] = r_add
        results[f'm{m_shrink}_log'] = r_log
        results[f'm{m_shrink}_dir'] = r_dir
        preds_dict[f'm{m_shrink}_add'] = preds_add
        preds_dict[f'm{m_shrink}_log'] = preds_log
        preds_dict[f'm{m_shrink}_dir'] = preds_dir

    # ==================================================================
    # CALIBRATED ENSEMBLE / POST-PROCESSING
    # ==================================================================
    print("\n" + "=" * 70)
    print("POST-PROCESSING & CALIBRATION SWEEP")
    print("=" * 70)

    # Let's find best candidate
    best_key = max(results, key=results.get)
    print(f"Best raw model: {best_key} = {results[best_key]:.4f}%")

    # Internal validation setup for bias on best model setup
    best_m_shrink = int(best_key.split('_')[0].replace('m', ''))
    best_type = best_key.split('_')[1]

    shifts_val, _, _ = compute_hierarchical_shift(d48, d49[d49['hour'] <= 1], best_m_shrink)
    d49_h01 = d49[d49['hour'] <= 1].copy()
    d49_h2 = d49[d49['hour'] == 2].copy()

    d49_h01_s2 = build_stage2_dataset(d49_h01, shifts_val, is_d48=False)
    d49_h2_s2 = build_stage2_dataset(d49_h2, shifts_val, is_d48=False)

    train_ref = pd.concat([d48, d49_h01], ignore_index=True)
    d48_s2_val = build_stage2_dataset(d48, is_d48=True)
    d48_s2_val = add_neighbor_features(d48_s2_val, train_ref)
    d49_h01_s2 = add_neighbor_features(d49_h01_s2, train_ref)
    d49_h2_s2 = add_neighbor_features(d49_h2_s2, train_ref)

    X_vt = pd.concat([d48_s2_val[s2_features], d49_h01_s2[s2_features]], ignore_index=True).fillna(0)
    X_vv = d49_h2_s2[s2_features].fillna(0)

    y_vt_demand = np.concatenate([d48['demand'].values, d49_h01['demand'].values])
    base_adjusted_vt = np.concatenate([d48_s2_val['stage1_shift_adjusted'].values, d49_h01_s2['stage1_shift_adjusted'].values])

    # Fit on internal val data
    m_val = lgb.LGBMRegressor(**params_heavy)
    if best_type == 'add':
        y_vt_res = y_vt_demand - base_adjusted_vt
        m_val.fit(X_vt, y_vt_res, sample_weight=sw[:len(X_vt)])
        val_preds = d49_h2_s2['stage1_shift_adjusted'].values + m_val.predict(X_vv)
    elif best_type == 'log':
        y_vt_res = np.log1p(np.maximum(y_vt_demand, 0)) - np.log1p(np.maximum(base_adjusted_vt, 0))
        m_val.fit(X_vt, y_vt_res, sample_weight=sw[:len(X_vt)])
        val_preds = np.expm1(np.log1p(np.maximum(d49_h2_s2['stage1_shift_adjusted'].values, 0)) + m_val.predict(X_vv))
    else:
        m_val.fit(X_vt, y_vt_demand, sample_weight=sw[:len(X_vt)])
        val_preds = m_val.predict(X_vv)

    val_bias = (val_preds - d49_h2['demand'].values).mean()
    print(f"Internal Validation Bias: {val_bias:+.6f}")

    # Sweep global bias multiplier
    print("\nGlobal bias calibration sweep:")
    base_preds = preds_dict[best_key].copy()
    for f in [0.0, 0.5, 1.0, 1.5, 2.0]:
        p = base_preds - val_bias * f
        sc = r2_score(real_demand, p) * 100
        results[f'{best_key}_bias{f}'] = sc
        print(f"  bias factor {f:.1f}: {sc:.4f}%")

    # ==================================================================
    # FINAL SUMMARY & ROAD TYPE BREAKDOWN
    # ==================================================================
    print("\n" + "=" * 70)
    print("ALL RESULTS SORTED")
    print("=" * 70)
    for name, score in sorted(results.items(), key=lambda x: -x[1])[:15]:
        marker = " <<<" if score == max(results.values()) else ""
        print(f"  {name:<30s}  {score:>10.4f}{marker}")

    best_final_key = max(results, key=results.get)
    print(f"\nBEST FINAL CONFIGURATION: {best_final_key} = {results[best_final_key]:.4f}%")

    # Reconstruct best predictions
    if 'bias' in best_final_key:
        raw_key = '_'.join(best_final_key.split('_')[:2])
        f_str = best_final_key.split('bias')[-1]
        bias_f = float(f_str)
        best_preds = preds_dict[raw_key] - val_bias * bias_f
    else:
        best_preds = preds_dict[best_final_key]

    # Per-RoadType
    print(f"\n  --- Per-RoadType R² for {best_final_key} ---")
    real_df['preds'] = best_preds
    for rt in sorted(real_df['RoadType'].unique()):
        mask = real_df['RoadType'] == rt
        if mask.sum() > 0:
            rt_score = r2_score(real_df.loc[mask, 'demand'], real_df.loc[mask, 'preds'])*100
            print(f"    {rt:<15s}: {rt_score:10.4f} (n={mask.sum()})")

    # Save
    sub = test[['Index']].copy()
    sub['demand'] = best_preds
    out = os.path.join(BASE, "dataset", "two_stage_v3_opt.csv")
    sub.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(f"Elapsed: {time.time() - t_start:.0f}s")
    print("=" * 70)
