"""
===========================================================================
Gridlock 2.0 -- curr_best_fix.py
===========================================================================
Predicting ONLY by fixed equation (LB004) and not ML models.
Prints internal and test scores comparing with real_test.csv.
===========================================================================
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# Configuration
SEED       = 42
TARGET     = 'demand'
ID         = 'Index'
ANCHOR_END = 120
TGT_LO, TGT_HI = 135, 825
SMOOTH_M   = 20.0
ROLL_WIN   = 5
TAU        = 240.0
W_ROLL     = 0.25
N_FOLDS    = 5

t_start = time.time()

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(SCRIPT_DIR, 'train.csv')
test_path = os.path.join(SCRIPT_DIR, 'test.csv')
real_test_path = os.path.join(SCRIPT_DIR, 'real_test.csv')

# ==========================================================================
# SECTION 1: DATA LOADING + BASE FEATURES
# ==========================================================================
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

def add_base_features(df):
    df = df.copy()
    df['tmin'] = df['timestamp'].map(ts_to_min)
    df['hour'] = (df['tmin'] // 60).astype(int)
    df['minute'] = (df['tmin'] % 60).astype(int)
    ang_t = 2 * np.pi * df['tmin'] / 1440.0
    df['sin_tmin'] = np.sin(ang_t)
    df['cos_tmin'] = np.cos(ang_t)
    df['is_rush']  = df['hour'].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    lat, lon = decode_geohashes(df['geohash'])
    df['lat'], df['lon'] = lat, lon
    df['geohash5'] = df['geohash'].str[:5]
    df['geohash4'] = df['geohash'].str[:4]
    df['RoadType'] = df['RoadType'].fillna('Missing').astype(str)
    df['Weather']  = df['Weather'].fillna('Missing').astype(str)
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin']     = (df['Landmarks'] == 'Yes').astype(int)
    df['temp_missing'] = df['Temperature'].isna().astype(int)
    df['Temperature']  = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce').fillna(1).astype(int)
    df['lanes_x_large'] = df['NumberofLanes'] * df['LargeVehicles_bin']
    return df

print("Loading and preprocessing datasets...")
train = add_base_features(pd.read_csv(train_path))
test  = add_base_features(pd.read_csv(test_path))

le_road = LabelEncoder()
le_weather = LabelEncoder()
le_road.fit(pd.concat([train['RoadType'], test['RoadType']]))
le_weather.fit(pd.concat([train['Weather'], test['Weather']]))
train['RoadType_enc'] = le_road.transform(train['RoadType'])
test['RoadType_enc']  = le_road.transform(test['RoadType'])
train['Weather_enc']  = le_weather.transform(train['Weather'])
test['Weather_enc']   = le_weather.transform(test['Weather'])

d48 = train[train.day == 48].copy()
d49 = train[train.day == 49].copy()

# ==========================================================================
# SECTION 2: SMOOTHED MEAN ENCODING
# ==========================================================================
class SmoothedMeanEncoder:
    def __init__(self, keys, m):
        self.keys, self.m = list(keys), float(m)
    def fit(self, df, target):
        g = df.groupby(self.keys, observed=True)[target]
        self.sum_, self.cnt_ = g.sum(), g.count()
        return self
    def transform(self, df, prior):
        idx = (pd.MultiIndex.from_frame(df[self.keys])
               if len(self.keys) > 1
               else pd.Index(df[self.keys[0]].values))
        s = np.nan_to_num(self.sum_.reindex(idx).to_numpy())
        c = np.nan_to_num(self.cnt_.reindex(idx).to_numpy())
        return (s + self.m * np.asarray(prior, dtype=float)) / (c + self.m)

def compute_encoders(history_df, target=TARGET, m=SMOOTH_M):
    enc = {'global_mean': float(history_df[target].mean())}
    for name, keys in [
        ('hour', ['hour']), ('roadtype', ['RoadType']),
        ('rt_hour', ['RoadType', 'hour']),
        ('g4', ['geohash4']), ('g5', ['geohash5']),
        ('geo', ['geohash']), ('geo_hour', ['geohash', 'hour']),
    ]:
        enc[name] = SmoothedMeanEncoder(keys, m).fit(history_df, target)
    return enc

def geohash_hour_encoded(enc, df):
    gm = np.full(len(df), enc['global_mean'])
    hour_mean = enc['hour'].transform(df, gm)
    rt_hour   = enc['rt_hour'].transform(df, hour_mean)
    return enc['geo_hour'].transform(df, rt_hour)

# ==========================================================================
# SECTION 3: DENOISED ANALOG
# ==========================================================================
def build_denoised_analog(analog_day_df):
    d = analog_day_df.sort_values(['geohash', 'tmin'])
    roll = d.groupby('geohash')[TARGET].transform(
        lambda s: s.rolling(ROLL_WIN, center=True, min_periods=1).mean()
    )
    roll_series = pd.Series(
        roll.values,
        index=pd.MultiIndex.from_arrays([d.geohash, d.tmin])
    ).sort_index()
    enc = compute_encoders(analog_day_df)
    def apply_fn(df):
        ar = roll_series.reindex(
            pd.MultiIndex.from_arrays([df.geohash, df.tmin])
        ).to_numpy()
        gh = geohash_hour_encoded(enc, df)
        return np.where(np.isnan(ar), gh, W_ROLL * ar + (1 - W_ROLL) * gh)
    return apply_fn

# ==========================================================================
# SECTION 4: MORNING FEATURES
# ==========================================================================
def morning_features(morning_df):
    m = morning_df[morning_df.tmin <= ANCHOR_END].sort_values(['geohash', 'tmin'])
    g = m.groupby('geohash')[TARGET]
    last = m.loc[m.groupby('geohash')['tmin'].idxmax()].set_index('geohash')[TARGET]
    return pd.DataFrame({'persistence': last, 'morning_std': g.std().fillna(0.0)})

# ==========================================================================
# SECTION 5: NEIGHBOR DEMAND FEATURES (Grab SEA AI)
# ==========================================================================
d48_g5_stats = d48.groupby('geohash5')[TARGET].agg(
    neighbor_mean='mean',
    neighbor_std='std',
    neighbor_count='count',
).fillna(0)

d48_g4_stats = d48.groupby('geohash4')[TARGET].agg(
    area_mean='mean',
    area_std='std',
).fillna(0)

# ==========================================================================
# SECTION 6: FEATURE ASSEMBLY
# ==========================================================================
def assemble_features(target_df, analog_fn, morn_feats, g5_stats, g4_stats):
    t = target_df.copy()
    t['analog'] = analog_fn(t)

    # Morning features
    t = t.merge(morn_feats, left_on='geohash', right_index=True, how='left')
    t['persistence'] = t['persistence'].fillna(t['analog'])
    t['morning_std'] = t['morning_std'].fillna(0.0)

    # Horizon + LB004
    t['horizon'] = t['tmin'] - ANCHOR_END
    t['w_exp'] = np.exp(-t['horizon'] / TAU)
    t['lb004_pred'] = np.clip(
        t['w_exp'] * t['persistence'] + (1 - t['w_exp']) * t['analog'], 0, 1)
    t['analog_minus_persist'] = t['analog'] - t['persistence']

    # Neighbor features
    t = t.merge(g5_stats, left_on='geohash5', right_index=True, how='left')
    t['neighbor_mean']  = t['neighbor_mean'].fillna(g5_stats['neighbor_mean'].mean())
    t['neighbor_std']   = t['neighbor_std'].fillna(0)
    t['neighbor_count'] = t['neighbor_count'].fillna(0)

    t = t.merge(g4_stats, left_on='geohash4', right_index=True, how='left')
    t['area_mean'] = t['area_mean'].fillna(g4_stats['area_mean'].mean())
    t['area_std']  = t['area_std'].fillna(0)

    # Ratio of this location vs its neighborhood
    t['local_vs_neighbor'] = t['analog'] / (t['neighbor_mean'] + 1e-6)

    return t

# ==========================================================================
# SECTION 8: BUILD TRAINING FRAME (Internal Validation Score)
# ==========================================================================
print("Evaluating internal validation score...")
tr_tgt = d48[(d48.tmin >= TGT_LO) & (d48.tmin <= TGT_HI)].copy().reset_index(drop=True)
morn_rows = d48[d48.tmin <= ANCHOR_END]
morn48 = morning_features(d48)

# OOF analog (leak-free)
oof_analog = np.full(len(tr_tgt), np.nan)
kf_analog = KFold(N_FOLDS, shuffle=True, random_state=SEED)
for fold_idx, (tr_idx, va_idx) in enumerate(kf_analog.split(tr_tgt)):
    src = pd.concat([morn_rows, tr_tgt.iloc[tr_idx]], ignore_index=True)
    fn_fold = build_denoised_analog(src)
    oof_analog[va_idx] = fn_fold(tr_tgt.iloc[va_idx])

_oof_s = pd.Series(oof_analog, index=tr_tgt.index)
oof_fn = lambda df: _oof_s.reindex(df.index).values

tr_frame = assemble_features(tr_tgt, oof_fn, morn48, d48_g5_stats, d48_g4_stats)
ytr = tr_frame[TARGET].values
lb004_oof = tr_frame['lb004_pred'].values
internal_r2 = r2_score(ytr, lb004_oof)

# ==========================================================================
# SECTION 9: BUILD TEST FRAME (Test Score)
# ==========================================================================
print("Evaluating test score...")
fn48_full = build_denoised_analog(d48)
morn49 = morning_features(d49)
test_frame = assemble_features(test, fn48_full, morn49, d48_g5_stats, d48_g4_stats)
lb004_test = test_frame['lb004_pred'].values

# Save predictions as submission CSV
output_csv_path = os.path.join(SCRIPT_DIR, 'curr_best_fix.csv')
sub = pd.DataFrame({ID: test_frame[ID].values, TARGET: lb004_test})
sub.to_csv(output_csv_path, index=False)
print(f"Saved test predictions to {output_csv_path}")

test_r2 = None
if os.path.exists(real_test_path):
    df_real = pd.read_csv(real_test_path)
    real_demand = df_real[TARGET].to_numpy(dtype=np.float64)
    pred_demand = pd.to_numeric(pd.Series(lb004_test), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    test_r2 = r2_score(real_demand, pred_demand)

# ==========================================================================
# SECTION 10: DISPLAY RESULTS
# ==========================================================================
print("\n=========================================")
print("FIXED EQUATION (LB004) EVALUATION")
print(f"INTERNAL VALIDATION R2 SCORE: {internal_r2 * 100:.6f}")
if test_r2 is not None:
    print(f"REAL LEADERBOARD R2 SCORE (0-100): {test_r2 * 100:.6f}")
else:
    print("REAL LEADERBOARD R2 SCORE: real_test.csv not found")
print("=========================================")

elapsed = time.time() - t_start
print(f"Done in {elapsed:.1f}s")
