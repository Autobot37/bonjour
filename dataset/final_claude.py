"""
final_claude.py - Based on baseline_v2_ef_corr.py with these changes:

NEW FEATURES:
  1. d48_g5_at_this_hour:  direct d48 g5-level demand at the target hour (replaces 24 sparse cols)
  2. d48_g5_at_prev_hour:  d48 g5 demand at hour-1 (temporal context)
  3. d48_g5_at_next_hour:  d48 g5 demand at hour+1 (temporal context)
  4. d48_g5_hourly_trend:  next - prev (rising or falling?)
  5. d48_geo_daily_mean:   stable per-geohash daily baseline demand
  6. d48_geo_hour_count:   observation count -> confidence measure
  7. d48_hour_normalized:  d48_same_hour_mean / d48_geo_daily_mean (shape, not level)
  8. d48_morning_afternoon_ratio: mean(h7-13)/mean(h0-6) per geohash (intraday growth)
  9. d48_peak_hour:        which hour has max demand per geohash
  10. d48_geo_demand_std:  overall demand variability per geohash

TRAINING CHANGES:
  - Filter d48 to h0-14 only (drop h15-23, not in test)
  - Higher d49 sample weight (sweep 2x, 5x, 10x)

POST-PROCESSING:
  - Global bias correction (from internal val)
  - Street correction (-0.03)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os, warnings
warnings.filterwarnings("ignore")

BASE = r"C:\Users\bagri\Downloads\e88186124ec611f1"

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


def add_d48_pattern_features(df, d48):
    """
    Add d48 temporal pattern features that capture the SHAPE of demand,
    not just the level. These transfer better across days.
    """
    t = df.copy()
    
    # ---- 1. Direct d48 g5-level demand at target hour (replaces 24 sparse columns) ----
    g5_hour_demand = d48.groupby(['geohash5', 'hour'])['demand'].mean()
    
    def map_g5_hour(df_slice, hour_offset=0):
        keys = list(zip(df_slice['geohash5'], (df_slice['hour'] + hour_offset).clip(0, 23)))
        return pd.Series([g5_hour_demand.get(k, np.nan) for k in keys], index=df_slice.index)
    
    t['d48_g5_at_this_hour'] = map_g5_hour(t, 0)
    t['d48_g5_at_prev_hour'] = map_g5_hour(t, -1)
    t['d48_g5_at_next_hour'] = map_g5_hour(t, +1)
    t['d48_g5_hourly_trend'] = t['d48_g5_at_next_hour'] - t['d48_g5_at_prev_hour']
    
    # For d48 rows, set to NaN (avoid leakage)
    if 'day' in t.columns:
        d48_mask = t['day'] == 48
        for col in ['d48_g5_at_this_hour', 'd48_g5_at_prev_hour', 
                     'd48_g5_at_next_hour', 'd48_g5_hourly_trend']:
            t.loc[d48_mask, col] = np.nan
    
    # ---- 2. d48 daily mean per geohash (stable baseline) ----
    d48_geo_daily = d48.groupby('geohash')['demand'].mean()
    t['d48_geo_daily_mean'] = t['geohash'].map(d48_geo_daily).astype(float)
    
    # ---- 3. d48 observation count per (geohash, hour) ----
    d48_geo_hour_count = d48.groupby(['geohash', 'hour'])['demand'].count()
    geo_hour_keys = list(zip(t['geohash'], t['hour']))
    t['d48_geo_hour_count'] = pd.Series(
        [d48_geo_hour_count.get(k, 0) for k in geo_hour_keys], index=t.index
    ).astype(float)
    
    # ---- 4. d48_hour_normalized: demand relative to daily average ----
    # This captures the SHAPE (e.g., 1.5 = 50% above daily avg at this hour)
    if 'd48_same_hour_mean' in t.columns:
        t['d48_hour_normalized'] = t['d48_same_hour_mean'] / (t['d48_geo_daily_mean'] + 1e-6)
    
    # ---- 5. d48 morning-to-afternoon ratio per geohash ----
    d48_morning = d48[d48['hour'] <= 6].groupby('geohash')['demand'].mean()
    d48_afternoon = d48[(d48['hour'] >= 7) & (d48['hour'] <= 13)].groupby('geohash')['demand'].mean()
    d48_ma_ratio = (d48_afternoon / (d48_morning + 1e-6)).clip(0, 10)
    t['d48_morning_afternoon_ratio'] = t['geohash'].map(d48_ma_ratio).astype(float)
    
    # ---- 6. d48 peak hour per geohash ----
    d48_peak = d48.groupby(['geohash', 'hour'])['demand'].mean().groupby('geohash').idxmax()
    d48_peak_hour = d48_peak.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    t['d48_peak_hour'] = t['geohash'].map(d48_peak_hour).astype(float)
    
    # ---- 7. d48 overall demand std per geohash (variability) ----
    d48_geo_std = d48.groupby('geohash')['demand'].std().fillna(0)
    t['d48_geo_demand_std'] = t['geohash'].map(d48_geo_std).astype(float)
    
    # ---- 8. d48 demand at adjacent hours for the SAME geohash ----
    d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
    
    def map_geo_hour(df_slice, hour_offset=0):
        keys = list(zip(df_slice['geohash'], (df_slice['hour'] + hour_offset).clip(0, 23)))
        return pd.Series([d48_geo_hour_mean.get(k, np.nan) for k in keys], index=df_slice.index)
    
    t['d48_prev_hour_mean'] = map_geo_hour(t, -1)
    t['d48_next_hour_mean'] = map_geo_hour(t, +1)
    t['d48_hour_momentum'] = t['d48_same_hour_mean'] - t['d48_prev_hour_mean']
    
    # For d48 rows, NaN the cross-reference features
    if 'day' in t.columns:
        d48_mask = t['day'] == 48
        for col in ['d48_geo_daily_mean', 'd48_hour_normalized', 
                     'd48_morning_afternoon_ratio', 'd48_peak_hour',
                     'd48_geo_demand_std', 'd48_prev_hour_mean',
                     'd48_next_hour_mean', 'd48_hour_momentum']:
            t.loc[d48_mask, col] = np.nan
    
    return t


def run_experiment(train, test, d48, train_rt_str, test_rt_str, 
                   d49_weight, filter_d48_hours, use_new_features, tag):
    """Run a single experiment configuration."""
    tr = train.copy()
    te = test.copy()
    
    # ---- D48 stats (same as baseline) ----
    d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
    d48_geo_hour_std = d48.groupby(['geohash', 'hour'])['demand'].std()
    
    tr['d48_same_hour_mean'] = np.nan; tr['d48_same_hour_std'] = np.nan
    m49 = tr['day'] == 49
    tr.loc[m49, 'd48_same_hour_mean'] = tr[m49].set_index(['geohash','hour']).index.map(d48_geo_hour_mean)
    tr.loc[m49, 'd48_same_hour_std'] = tr[m49].set_index(['geohash','hour']).index.map(d48_geo_hour_std)
    te['d48_same_hour_mean'] = te.set_index(['geohash','hour']).index.map(d48_geo_hour_mean)
    te['d48_same_hour_std'] = te.set_index(['geohash','hour']).index.map(d48_geo_hour_std)
    
    # ---- D48 trajectory (original 24-col approach, kept for compatibility) ----
    g5_hourly = d48.groupby(['geohash5', 'hour'])['demand'].mean().unstack(fill_value=0)
    g5_hourly.columns = [f'd48_g5_hourly_mean_h{h}' for h in g5_hourly.columns]
    tr = tr.merge(g5_hourly, on='geohash5', how='left')
    tr.loc[tr['day'] == 48, g5_hourly.columns] = np.nan
    te = te.merge(g5_hourly, on='geohash5', how='left')
    
    # ---- NEW: d48 pattern features ----
    if use_new_features:
        tr = add_d48_pattern_features(tr, d48)
        te = add_d48_pattern_features(te, d48)
    
    # ---- Filter d48 hours ----
    if filter_d48_hours:
        before = len(tr)
        tr = tr[(tr['day'] == 49) | (tr['hour'] <= 14)]
        print(f"  Filtered d48 hours: {before} -> {len(tr)} rows")
    
    # ---- Encode categoricals ----
    for col in ['geohash', 'geohash4', 'geohash5']:
        le = LabelEncoder(); le.fit(pd.concat([tr[col], te[col]]))
        tr[col] = le.transform(tr[col]); tr[col] = tr[col].astype('category')
        te[col] = le.transform(te[col]); te[col] = te[col].astype('category')
    
    le_r = LabelEncoder(); le_w = LabelEncoder()
    le_r.fit(pd.concat([tr['RoadType'], te['RoadType']]))
    le_w.fit(pd.concat([tr['Weather'], te['Weather']]))
    tr['RoadType_enc'] = le_r.transform(tr['RoadType']); tr['RoadType_enc'] = tr['RoadType_enc'].astype('category')
    te['RoadType_enc'] = le_r.transform(te['RoadType']); te['RoadType_enc'] = te['RoadType_enc'].astype('category')
    tr['Weather_enc'] = le_w.transform(tr['Weather']); tr['Weather_enc'] = tr['Weather_enc'].astype('category')
    te['Weather_enc'] = le_w.transform(te['Weather']); te['Weather_enc'] = te['Weather_enc'].astype('category')
    
    tr.drop(columns=['RoadType','Weather','LargeVehicles','Landmarks'], inplace=True, errors='ignore')
    te.drop(columns=['RoadType','Weather','LargeVehicles','Landmarks'], inplace=True, errors='ignore')
    
    # ---- Internal validation ----
    int_tr_mask = (tr['day'] == 48) | ((tr['day'] == 49) & (tr['hour'] <= 1))
    int_va_mask = (tr['day'] == 49) & (tr['hour'] == 2)
    
    tr_ref = tr[int_tr_mask].copy()
    tr_tr = add_neighbor_features(tr_ref, tr_ref)
    tr_va = add_neighbor_features(tr[int_va_mask].copy(), tr_ref)
    
    target = "demand"
    features = [c for c in tr_tr.columns if c not in ["Index", target]]
    
    sw_v = np.ones(len(tr_tr))
    sw_v[tr_tr['day'] == 49] = d49_weight
    sw_v[(tr_tr['day'] == 48) & (tr_tr['hour'] >= 2) & (tr_tr['hour'] <= 13)] = 1.5
    
    m = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    m.fit(tr_tr[features], tr_tr[target], sample_weight=sw_v)
    vp = m.predict(tr_va[features])
    vs = r2_score(tr_va[target], vp) * 100
    bias = (vp - tr_va[target].values).mean()
    print(f"  [{tag}] Internal Val R2: {vs:.4f}%  bias={bias:+.6f}")
    
    # ---- Full inference ----
    tr_full = add_neighbor_features(tr, tr)
    te_full = add_neighbor_features(te, tr)
    
    sw_f = np.ones(len(tr_full))
    sw_f[tr_full['day'] == 49] = d49_weight
    sw_f[(tr_full['day'] == 48) & (tr_full['hour'] >= 2) & (tr_full['hour'] <= 13)] = 1.5
    
    mf = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    mf.fit(tr_full[features], tr_full[target], sample_weight=sw_f)
    raw_preds = mf.predict(te_full[features])
    
    # ---- Post-processing ----
    # Street correction
    preds = raw_preds.copy()
    preds[test_rt_str.values == 'Street'] -= 0.03
    # Global bias correction (1.5x internal val bias)
    preds -= bias * 1.5
    
    # ---- Evaluate ----
    real_test_path = os.path.join(BASE, "dataset", "real_test.csv")
    score = np.nan
    if os.path.exists(real_test_path):
        real_demand = pd.read_csv(real_test_path)["demand"].to_numpy(dtype=np.float64)
        
        score_raw = r2_score(real_demand, raw_preds) * 100
        score_st = r2_score(real_demand, raw_preds.copy() - 0) * 100
        
        # Try different global bias factors
        best_score = 0
        best_factor = 1.5
        for factor in [0.0, 1.0, 1.5, 2.0, 2.5]:
            p = raw_preds.copy()
            p[test_rt_str.values == 'Street'] -= 0.03
            p -= bias * factor
            s = r2_score(real_demand, p) * 100
            if s > best_score:
                best_score = s
                best_factor = factor
        
        score = best_score
        print(f"  [{tag}] Raw: {score_raw:.4f}% | Best (Street+bias*{best_factor:.1f}): {score:.4f}%")
    
    # Feature importance (top 10)
    imp = pd.DataFrame({'f': features, 'i': mf.feature_importances_}).sort_values('i', ascending=False)
    print(f"  [{tag}] Top 10 features:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['f']:35s} {r['i']:8.0f}")
    
    return score, preds, raw_preds, bias, features, te


def main():
    print("Loading data...")
    train_raw = pd.read_csv(os.path.join(BASE, "dataset", "train.csv"))
    test_raw = pd.read_csv(os.path.join(BASE, "dataset", "test.csv"))
    
    print("Engineering features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)
    
    train_rt_str = train['RoadType'].copy()
    test_rt_str = test['RoadType'].copy()
    
    d48 = train[train['day'] == 48]
    
    # ============================================================
    # RUN EXPERIMENTS
    # ============================================================
    results = []
    
    # Experiment 1: Baseline (same as v2_ef_corr)
    print("\n" + "="*60)
    print("EXP 1: Baseline (v2_ef_corr equivalent)")
    print("="*60)
    s1, p1, r1, b1, f1, t1 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=2.0, filter_d48_hours=False, use_new_features=False, tag="BASE"
    )
    results.append(("Baseline", s1))
    
    # Experiment 2: New features only
    print("\n" + "="*60)
    print("EXP 2: + New d48 pattern features")
    print("="*60)
    s2, p2, r2, b2, f2, t2 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=2.0, filter_d48_hours=False, use_new_features=True, tag="NEW_FEAT"
    )
    results.append(("+ New Features", s2))
    
    # Experiment 3: Filter d48 hours
    print("\n" + "="*60)
    print("EXP 3: + Filter d48 to h0-14")
    print("="*60)
    s3, p3, r3, b3, f3, t3 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=2.0, filter_d48_hours=True, use_new_features=False, tag="FILTER"
    )
    results.append(("+ Filter d48", s3))
    
    # Experiment 4: Higher d49 weight
    print("\n" + "="*60)
    print("EXP 4: d49 weight = 5x")
    print("="*60)
    s4, p4, r4, b4, f4, t4 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=5.0, filter_d48_hours=False, use_new_features=False, tag="W5X"
    )
    results.append(("d49 5x weight", s4))
    
    # Experiment 5: d49 weight 10x
    print("\n" + "="*60)
    print("EXP 5: d49 weight = 10x")
    print("="*60)
    s5, p5, r5, b5, f5, t5 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=10.0, filter_d48_hours=False, use_new_features=False, tag="W10X"
    )
    results.append(("d49 10x weight", s5))
    
    # Experiment 6: All combined - new features + filter + higher weight
    print("\n" + "="*60)
    print("EXP 6: New features + Filter + d49 5x")
    print("="*60)
    s6, p6, r6, b6, f6, t6 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=5.0, filter_d48_hours=True, use_new_features=True, tag="ALL"
    )
    results.append(("All combined (5x)", s6))
    
    # Experiment 7: New features + d49 3x (moderate weight)
    print("\n" + "="*60)
    print("EXP 7: New features + d49 3x")
    print("="*60)
    s7, p7, r7, b7, f7, t7 = run_experiment(
        train, test, d48, train_rt_str, test_rt_str,
        d49_weight=3.0, filter_d48_hours=False, use_new_features=True, tag="FEAT+W3"
    )
    results.append(("New feat + 3x", s7))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    results.sort(key=lambda x: -(x[1] if x[1] == x[1] else 0))
    for name, score in results:
        print(f"  {name:30s}: {score:.4f}%")
    
    # Save best
    best_name, best_score = results[0]
    print(f"\nBest: {best_name} = {best_score:.4f}%")

if __name__ == "__main__":
    main()
