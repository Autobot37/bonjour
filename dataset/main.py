"""
===========================================================================
Gridlock 2.0 -- curr_best91.py
===========================================================================
All proven techniques that got LB=91.08 (v4), PLUS:
  - MLPRegressor from sklearn (neural net) added to ensemble
  - Neighbor geohash features (from Grab SEA AI challenge research)
  - Multi-seed averaging for variance reduction

Architecture:
  FINAL = alpha * LB004 + (1-alpha) * RidgeStack(LGB_res, XGB_res, CB, HGB_res, MLP_res)
  Proven best: alpha = 0.45

Key proven settings (DO NOT CHANGE):
  TAU = 240, W_ROLL = 0.25, residual learning, 22 base features
===========================================================================
"""

import os, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

# ==========================================================================
# PROVEN CONFIG (from v4, LB=91.08)
# ==========================================================================
SEED       = 42
TARGET     = 'demand'
ID         = 'Index'
ANCHOR_END = 120
TGT_LO, TGT_HI = 135, 1440
SMOOTH_M   = 20.0
ROLL_WIN   = 5
TAU        = 240.0
W_ROLL     = 0.25
N_FOLDS    = 5

t_start = time.time()
print(f'Libraries: LGB={HAS_LGB} | XGB={HAS_XGB} | CB={HAS_CB}')

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

def geohash_neighbors(gh):
    """Get 8 neighboring geohashes by incrementing/decrementing last char."""
    # Simplified: use geohash5 prefix grouping for spatial neighbors
    return gh[:5]  # group by 5-char prefix

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

train = add_base_features(pd.read_csv('train.csv'))
test  = add_base_features(pd.read_csv('test.csv'))
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
print(f'd48: {d48.shape} | d49: {d49.shape} | test: {test.shape}')

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
# SECTION 5: NEIGHBOR DEMAND FEATURES (from Grab SEA AI challenge)
# ==========================================================================
# Key insight from Grab challenge: demand in one area correlates with
# neighboring areas. We approximate this by computing the average demand
# of all geohashes sharing the same 5-character prefix (same local region).
# This is computed from d48 FULL data for both train and test -> no shift.

print('\n--- Computing neighbor features (Grab SEA AI) ---')
# Neighbor = same geohash5 prefix (nearby locations)
d48_g5_stats = d48.groupby('geohash5')[TARGET].agg(
    neighbor_mean='mean',
    neighbor_std='std',
    neighbor_count='count',
).fillna(0)

# Per geohash4 (broader area) stats
d48_g4_stats = d48.groupby('geohash4')[TARGET].agg(
    area_mean='mean',
    area_std='std',
).fillna(0)

print(f'  Geohash5 groups: {len(d48_g5_stats)}')
print(f'  Geohash4 groups: {len(d48_g4_stats)}')

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

    # Neighbor features (from Grab SEA AI challenge research)
    # Same d48 data for both train and test -> consistent distributions
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
# SECTION 7: FEATURE LIST
# ==========================================================================
# Original 22 proven features + 6 new neighbor/area features
FEATURES = [
    # === PROVEN 22 features from v4 (LB=91.08) ===
    'hour', 'minute', 'sin_tmin', 'cos_tmin', 'is_rush', 'is_night',
    'lat', 'lon',
    'NumberofLanes', 'LargeVehicles_bin', 'Landmarks_bin',
    'RoadType_enc', 'Weather_enc',
    'temp_missing', 'Temperature', 'lanes_x_large',
    'analog', 'persistence', 'morning_std',
    'horizon', 'w_exp',
    'analog_minus_persist',
    # === NEW: Neighbor/area features (Grab SEA AI inspired) ===
    'neighbor_mean',        # avg demand in same geohash5 region
    'neighbor_std',         # demand variability in region
    'neighbor_count',       # data density in region
    'area_mean',            # broader area (geohash4) demand
    'area_std',             # broader area variability
    'local_vs_neighbor',    # how this location compares to neighbors
]

print(f'\nFeatures: {len(FEATURES)} (22 proven + {len(FEATURES)-22} new)')

# ==========================================================================
# SECTION 8: BUILD TRAINING FRAME
# ==========================================================================
print('\n' + '='*60)
print('BUILDING TRAINING FRAME')
print('='*60)

tr_tgt = d48[(d48.tmin >= TGT_LO) & (d48.tmin <= TGT_HI)].copy().reset_index(drop=True)
morn_rows = d48[d48.tmin <= ANCHOR_END]
morn48 = morning_features(d48)

# OOF analog (leak-free)
print('Computing OOF analog...')
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
residual_oof = ytr - lb004_oof

Xtr = tr_frame[FEATURES].copy().fillna(0.0)
print(f'Training: X={Xtr.shape}')
print(f'LB004 OOF R^2 = {r2_score(ytr, lb004_oof):.6f}')
print(f'Residual: mean={residual_oof.mean():.6f}  std={residual_oof.std():.4f}')

# ==========================================================================
# SECTION 9: BUILD TEST FRAME
# ==========================================================================
print('\n' + '='*60)
print('BUILDING TEST FRAME')
print('='*60)

fn48_full = build_denoised_analog(d48)
morn49 = morning_features(d49)
test_frame = assemble_features(test, fn48_full, morn49, d48_g5_stats, d48_g4_stats)
lb004_test = test_frame['lb004_pred'].values
Xte = test_frame[FEATURES].copy().fillna(0.0)
print(f'Test: X={Xte.shape}')

# Distribution check for new features
print('\n--- Distribution Check (new features) ---')
for f in FEATURES[22:]:
    tr_m = Xtr[f].mean()
    te_m = Xte[f].mean()
    ratio = te_m / tr_m if abs(tr_m) > 1e-8 else float('nan')
    print(f'  {f:25s}  train={tr_m:8.4f}  test={te_m:8.4f}  ratio={ratio:.2f}')

# ==========================================================================
# SECTION 10: MODEL CONFIGS
# ==========================================================================
# Tree models: proven v4 configs
lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 50,
    'reg_alpha': 1.0, 'reg_lambda': 10.0,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'n_estimators': 800, 'verbose': -1, 'random_state': SEED,
}
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 50,
    'reg_alpha': 1.0, 'reg_lambda': 10.0,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'n_estimators': 800, 'verbosity': 0, 'random_state': SEED,
}
cb_params = {
    'iterations': 800, 'learning_rate': 0.03, 'depth': 5,
    'l2_leaf_reg': 10.0, 'random_seed': SEED, 'verbose': 0,
}
hgb_params = {
    'max_iter': 800, 'learning_rate': 0.03, 'max_leaf_nodes': 31,
    'min_samples_leaf': 50, 'l2_regularization': 10.0,
    'early_stopping': False, 'random_state': SEED,
}

# NEW: MLP (neural net) config
# - 3 hidden layers with decreasing width
# - Early stopping to prevent overfitting
# - Standardized inputs (critical for neural nets)
mlp_params = {
    'hidden_layer_sizes': (128, 64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,           # L2 regularization
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 15,
    'random_state': SEED,
    'verbose': False,
}

# Model registry: trees + MLP
MODEL_NAMES = []
if HAS_LGB: MODEL_NAMES.append('LGB')
if HAS_XGB: MODEL_NAMES.append('XGB')
if HAS_CB:  MODEL_NAMES.append('CB')
MODEL_NAMES.append('HGB')
MODEL_NAMES.append('MLP')  # NEW

print(f'\nModels: {MODEL_NAMES}')

# ==========================================================================
# SECTION 11: 5-FOLD CV (DIRECT + RESIDUAL)
# ==========================================================================
print('\n' + '='*60)
print(f'{N_FOLDS}-FOLD CV: DIRECT + RESIDUAL')
print('='*60 + '\n')

kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
Xtr_np = Xtr.values.astype(np.float32)

oof_direct = {n: np.zeros(len(ytr)) for n in MODEL_NAMES}
oof_resid  = {n: np.zeros(len(ytr)) for n in MODEL_NAMES}
fmodels_dir = {n: [] for n in MODEL_NAMES}
fmodels_res = {n: [] for n in MODEL_NAMES}
# MLP needs scalers stored per fold
mlp_scalers = {'dir': [], 'res': []}

for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr_np)):
    X_f_tr, X_f_va = Xtr_np[tr_idx], Xtr_np[va_idx]
    y_f_tr, y_f_va = ytr[tr_idx], ytr[va_idx]
    r_f_tr = residual_oof[tr_idx]
    lb004_f_va = lb004_oof[va_idx]

    dir_r2, res_r2 = {}, {}

    # ---------- Tree models ----------
    for name, cls, params in [
        ('LGB', lgb.LGBMRegressor if HAS_LGB else None, lgb_params),
        ('XGB', xgb.XGBRegressor if HAS_XGB else None, xgb_params),
        ('CB', CatBoostRegressor if HAS_CB else None, cb_params),
        ('HGB', HistGradientBoostingRegressor, hgb_params),
    ]:
        if cls is None:
            continue
        # Direct
        m_d = cls(**params); m_d.fit(X_f_tr, y_f_tr)
        p_d = np.clip(m_d.predict(X_f_va), 0, 1)
        oof_direct[name][va_idx] = p_d
        dir_r2[name] = r2_score(y_f_va, p_d)
        fmodels_dir[name].append(m_d)
        # Residual
        m_r = cls(**params); m_r.fit(X_f_tr, r_f_tr)
        p_r = np.clip(lb004_f_va + m_r.predict(X_f_va), 0, 1)
        oof_resid[name][va_idx] = p_r
        res_r2[name] = r2_score(y_f_va, p_r)
        fmodels_res[name].append(m_r)

    # ---------- MLP (needs feature scaling) ----------
    scaler_d = StandardScaler().fit(X_f_tr)
    X_f_tr_sc = scaler_d.transform(X_f_tr)
    X_f_va_sc = scaler_d.transform(X_f_va)

    # MLP Direct
    mlp_d = MLPRegressor(**mlp_params)
    mlp_d.fit(X_f_tr_sc, y_f_tr)
    p_mlp_d = np.clip(mlp_d.predict(X_f_va_sc), 0, 1)
    oof_direct['MLP'][va_idx] = p_mlp_d
    dir_r2['MLP'] = r2_score(y_f_va, p_mlp_d)
    fmodels_dir['MLP'].append(mlp_d)
    mlp_scalers['dir'].append(scaler_d)

    # MLP Residual
    scaler_r = StandardScaler().fit(X_f_tr)
    X_f_tr_sc_r = scaler_r.transform(X_f_tr)
    X_f_va_sc_r = scaler_r.transform(X_f_va)
    mlp_r = MLPRegressor(**mlp_params)
    mlp_r.fit(X_f_tr_sc_r, r_f_tr)
    p_mlp_r = np.clip(lb004_f_va + mlp_r.predict(X_f_va_sc_r), 0, 1)
    oof_resid['MLP'][va_idx] = p_mlp_r
    res_r2['MLP'] = r2_score(y_f_va, p_mlp_r)
    fmodels_res['MLP'].append(mlp_r)
    mlp_scalers['res'].append(scaler_r)

    print(f'Fold {fold_idx+1}:  ' + '  '.join(
        f'{n}:d={dir_r2[n]:.4f}/r={res_r2[n]:.4f}' for n in MODEL_NAMES))

# ==========================================================================
# SECTION 12: SELECT BEST APPROACH PER MODEL
# ==========================================================================
print(f'\n{"="*60}')
print(f'{"Model":6s}  {"Direct":>10s}  {"Residual":>10s}  {"Winner":>8s}')
oof_best, fmodels_best, approach_best = {}, {}, {}
for name in MODEL_NAMES:
    r2_d = r2_score(ytr, oof_direct[name])
    r2_r = r2_score(ytr, oof_resid[name])
    winner = 'resid' if r2_r > r2_d else 'direct'
    if r2_r > r2_d:
        oof_best[name] = oof_resid[name]
        fmodels_best[name] = ('resid', fmodels_res[name])
    else:
        oof_best[name] = oof_direct[name]
        fmodels_best[name] = ('direct', fmodels_dir[name])
    approach_best[name] = winner
    print(f'{name:6s}  {r2_d:10.6f}  {r2_r:10.6f}  {winner:>8s}')
print(f'{"="*60}')

# ==========================================================================
# SECTION 13: STACKING ENSEMBLE
# ==========================================================================
print('\n' + '='*60)
print('STACKING')
print('='*60)

stack_train = np.column_stack([oof_best[n] for n in MODEL_NAMES])

# Ridge stacking
ridge = Ridge(alpha=1.0)
ridge.fit(stack_train, ytr)
stacked_oof = np.clip(ridge.predict(stack_train), 0, 1)
stacked_r2  = r2_score(ytr, stacked_oof)

# Weighted average search (exhaustive for 5 models)
best_w_r2, best_weights = -1e9, None
n_m = len(MODEL_NAMES)
if n_m == 5:
    # Grid search for 5 models (coarser step to keep it tractable)
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            for w2 in np.arange(0, 1.05 - w0 - w1, 0.1):
                for w3 in np.arange(0, 1.05 - w0 - w1 - w2, 0.1):
                    w4 = max(1.0 - w0 - w1 - w2 - w3, 0.0)
                    wt = np.array([w0, w1, w2, w3, w4])
                    r2w = r2_score(ytr, np.clip(stack_train @ wt, 0, 1))
                    if r2w > best_w_r2:
                        best_w_r2, best_weights = r2w, wt.copy()

print(f'Ridge Stack  R^2: {stacked_r2:.6f}  Score: {max(0,100*stacked_r2):.2f}')
print(f'Best Wt Avg  R^2: {best_w_r2:.6f}  Score: {max(0,100*best_w_r2):.2f}')
print(f'Ridge coefs: {dict(zip(MODEL_NAMES, ridge.coef_))}')
print(f'Best weights: {dict(zip(MODEL_NAMES, best_weights))}')

# Also try ensemble WITHOUT MLP (to see if MLP helps or hurts)
tree_names = [n for n in MODEL_NAMES if n != 'MLP']
stack_trees = np.column_stack([oof_best[n] for n in tree_names])
ridge_trees = Ridge(alpha=1.0)
ridge_trees.fit(stack_trees, ytr)
trees_oof = np.clip(ridge_trees.predict(stack_trees), 0, 1)
trees_r2 = r2_score(ytr, trees_oof)
print(f'\nTrees-only Stack R^2: {trees_r2:.6f}  Score: {max(0,100*trees_r2):.2f}')
print(f'MLP added:       R^2: {stacked_r2:.6f}  Score: {max(0,100*stacked_r2):.2f}')
print(f'MLP delta: {(stacked_r2 - trees_r2)*100:+.4f} points')

# Use best method
ensemble_methods = {'Ridge_Stack': stacked_r2, 'Best_Wt': best_w_r2, 'Trees_Only': trees_r2}
best_method = max(ensemble_methods, key=ensemble_methods.get)
print(f'\nBest method: {best_method}')

if best_method == 'Trees_Only':
    model_oof = trees_oof
    print('  -> Using trees-only ensemble (MLP did not help)')
elif best_method == 'Ridge_Stack':
    model_oof = stacked_oof
else:
    model_oof = np.clip(stack_train @ best_weights, 0, 1)

# ==========================================================================
# SECTION 14: TEST PREDICTIONS (FOLD-MODEL AVERAGING)
# ==========================================================================
print('\n' + '='*60)
print('TEST PREDICTIONS')
print('='*60 + '\n')

Xte_np = Xte.values.astype(np.float32)
test_preds = {}

for name in MODEL_NAMES:
    approach, models = fmodels_best[name]
    preds = []
    for i, fm in enumerate(models):
        if name == 'MLP':
            # MLP needs scaled input
            scaler_key = 'res' if approach == 'resid' else 'dir'
            X_sc = mlp_scalers[scaler_key][i].transform(Xte_np)
            if approach == 'direct':
                preds.append(np.clip(fm.predict(X_sc), 0, 1))
            else:
                preds.append(np.clip(lb004_test + fm.predict(X_sc), 0, 1))
        else:
            if approach == 'direct':
                preds.append(np.clip(fm.predict(Xte_np), 0, 1))
            else:
                preds.append(np.clip(lb004_test + fm.predict(Xte_np), 0, 1))
    test_preds[name] = np.mean(preds, axis=0)
    m = test_preds[name]
    print(f'{name:6s} ({approach}): mean={m.mean():.4f}  min={m.min():.4f}  max={m.max():.4f}')

# Correlation matrix between models
print('\n--- Model Correlation Matrix ---')
names = MODEL_NAMES
corr_matrix = np.corrcoef([test_preds[n] for n in names])
print(f'{"":8s}  ' + '  '.join(f'{n:6s}' for n in names))
for i, n in enumerate(names):
    print(f'{n:8s}  ' + '  '.join(f'{corr_matrix[i,j]:.4f}' for j in range(len(names))))

# Apply stacking
stack_test = np.column_stack([test_preds[n] for n in MODEL_NAMES])
stack_test_trees = np.column_stack([test_preds[n] for n in tree_names])

if best_method == 'Trees_Only':
    model_test = np.clip(ridge_trees.predict(stack_test_trees), 0, 1)
elif best_method == 'Ridge_Stack':
    model_test = np.clip(ridge.predict(stack_test), 0, 1)
else:
    model_test = np.clip(stack_test @ best_weights, 0, 1)

print(f'\nEnsemble test ({best_method}): mean={model_test.mean():.4f}')

# ==========================================================================
# SECTION 15: MULTI-SEED AVERAGING (variance reduction)
# ==========================================================================
# Train 2 additional seeds and average for reduced variance
print('\n' + '='*60)
print('MULTI-SEED AVERAGING')
print('='*60)

seed_preds = [model_test.copy()]  # seed=42 already done

for extra_seed in [123, 777]:
    print(f'\n  --- Seed {extra_seed} ---')
    kf_s = KFold(N_FOLDS, shuffle=True, random_state=extra_seed)
    oof_s = {n: np.zeros(len(ytr)) for n in tree_names}
    fmod_s = {n: [] for n in tree_names}

    for fold_idx, (tr_idx, va_idx) in enumerate(kf_s.split(Xtr_np)):
        X_f_tr, X_f_va = Xtr_np[tr_idx], Xtr_np[va_idx]
        y_f_tr = ytr[tr_idx]
        r_f_tr = residual_oof[tr_idx]
        lb004_f_va = lb004_oof[va_idx]

        for name, cls, params in [
            ('LGB', lgb.LGBMRegressor if HAS_LGB else None, lgb_params),
            ('XGB', xgb.XGBRegressor if HAS_XGB else None, xgb_params),
            ('CB', CatBoostRegressor if HAS_CB else None, cb_params),
            ('HGB', HistGradientBoostingRegressor, hgb_params),
        ]:
            if cls is None or name not in tree_names:
                continue
            p_copy = dict(params)
            # Update seed params
            if 'random_state' in p_copy: p_copy['random_state'] = extra_seed
            if 'random_seed' in p_copy: p_copy['random_seed'] = extra_seed

            approach = approach_best.get(name, 'direct')
            m = cls(**p_copy)
            if approach == 'resid':
                m.fit(X_f_tr, r_f_tr)
                oof_s[name][va_idx] = np.clip(lb004_f_va + m.predict(X_f_va), 0, 1)
            else:
                m.fit(X_f_tr, y_f_tr)
                oof_s[name][va_idx] = np.clip(m.predict(X_f_va), 0, 1)
            fmod_s[name].append(m)

    # Test predictions for this seed
    test_s = {}
    for name in tree_names:
        approach = approach_best.get(name, 'direct')
        preds = []
        for fm in fmod_s[name]:
            if approach == 'resid':
                preds.append(np.clip(lb004_test + fm.predict(Xte_np), 0, 1))
            else:
                preds.append(np.clip(fm.predict(Xte_np), 0, 1))
        test_s[name] = np.mean(preds, axis=0)

    stack_s = np.column_stack([test_s[n] for n in tree_names])
    pred_s = np.clip(ridge_trees.predict(stack_s), 0, 1)
    seed_preds.append(pred_s)
    print(f'  Seed {extra_seed}: mean={pred_s.mean():.4f}')

# Average across seeds
model_test_multiseed = np.mean(seed_preds, axis=0)
print(f'\nMulti-seed average ({len(seed_preds)} seeds): mean={model_test_multiseed.mean():.4f}')

# ==========================================================================
# SECTION 16: GENERATE ALL SUBMISSIONS
# ==========================================================================
print('\n' + '='*60)
print('GENERATING SUBMISSIONS')
print('='*60)

submissions = {}

# v4 reproductions (single seed)
for a_pct in [40, 43, 44, 45, 46, 47, 50]:
    a = a_pct / 100.0
    submissions[f'best_a{a_pct:02d}'] = np.clip(a*lb004_test + (1-a)*model_test, 0, 1)

# Multi-seed versions
for a_pct in [40, 43, 44, 45, 46, 47, 50]:
    a = a_pct / 100.0
    submissions[f'multi_a{a_pct:02d}'] = np.clip(a*lb004_test + (1-a)*model_test_multiseed, 0, 1)

# Pure model
submissions['best_pure'] = model_test
submissions['multi_pure'] = model_test_multiseed

for name, pred in submissions.items():
    sub = pd.DataFrame({ID: test_frame[ID].values, TARGET: pred})
    sub.to_csv(f'x_{name}.csv', index=False)
    print(f'  {name:20s}  mean={pred.mean():.4f}  min={pred.min():.4f}  max={pred.max():.4f}')

# Default
pd.DataFrame({ID: test_frame[ID].values,
              TARGET: submissions['best_a45']}).to_csv('x_submission.csv', index=False)
pd.DataFrame({ID: test_frame[ID].values,
              TARGET: submissions['best_a45']}).to_csv('x_heavy_submission.csv', index=False)

# ==========================================================================
# SECTION 17: RESULTS SUMMARY
# ==========================================================================
elapsed = time.time() - t_start
print(f'\n{"="*60}')
print(f'FINAL RESULTS (elapsed: {elapsed:.0f}s)')
print(f'{"="*60}\n')

print(f'--- Per-Model OOF ---')
for name in MODEL_NAMES:
    r2 = r2_score(ytr, oof_best[name])
    print(f'{name:6s}  R^2={r2:.6f}  Score={max(0,100*r2):.2f}  ({approach_best[name]})')

print(f'\n--- Ensemble OOF ---')
print(f'Ridge Stack (all): {stacked_r2:.6f}  Score: {max(0,100*stacked_r2):.2f}')
print(f'Trees only:        {trees_r2:.6f}  Score: {max(0,100*trees_r2):.2f}')
print(f'Best method:       {best_method}')
print(f'MLP delta:         {(stacked_r2 - trees_r2)*100:+.4f} points')

print(f'\n--- Blend Grid ---')
for a_pct in [40, 43, 44, 45, 46, 47, 50]:
    a = a_pct / 100.0
    r2 = r2_score(ytr, np.clip(a*lb004_oof + (1-a)*model_oof, 0, 1))
    print(f'  a={a:.2f}  OOF={max(0,100*r2):.2f}  -> best_a{a_pct:02d}.csv / multi_a{a_pct:02d}.csv')

# Feature importance
if HAS_LGB and fmodels_best.get('LGB'):
    print(f'\n--- Top Features (LGB) ---')
    imp = fmodels_best['LGB'][1][0].feature_importances_
    for fname, importance in sorted(zip(FEATURES, imp), key=lambda x: x[1], reverse=True)[:15]:
        print(f'  {fname:25s}  {importance:8.0f}')

# Save results
results_file = 'results_utk.csv'
new_rows = []
for name in MODEL_NAMES:
    new_rows.append({
        'model': f'best91_{name}',
        'final_r2': r2_score(ytr, oof_best[name]),
        'mean_r2': r2_score(ytr, oof_best[name]),
        'fold_1': 0, 'fold_2': 0, 'fold_3': 0, 'fold_4': 0, 'fold_5': 0,
    })
new_rows.append({'model': f'best91_stack', 'final_r2': stacked_r2, 'mean_r2': stacked_r2,
                 'fold_1': 0, 'fold_2': 0, 'fold_3': 0, 'fold_4': 0, 'fold_5': 0})
new_df = pd.DataFrame(new_rows)
if os.path.exists(results_file):
    try:
        existing = pd.read_csv(results_file)
        updated = pd.concat([existing, new_df], ignore_index=True)
    except Exception:
        updated = new_df
else:
    updated = new_df
updated.to_csv(results_file, index=False)

print(f'\n{"="*60}')
print('PRIORITY SUBMISSIONS:')
print('  best_a45.csv   = proven blend (single seed)')
print('  multi_a45.csv  = multi-seed averaged (should be more stable)')
print('  multi_a43.csv  = more model weight, multi-seed')
print('  multi_a47.csv  = more LB004 weight, multi-seed')
print(f'{"="*60}')
print('DONE!')
print("HELLO")