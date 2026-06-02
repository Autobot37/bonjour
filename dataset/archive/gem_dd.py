"""
===========================================================================
Gridlock 3.0 -- curr_best91_ddm.py (Drift-Diffusion Enhanced)
===========================================================================
All proven techniques that got LB=91.08 (v4), PLUS:
  - MLPRegressor from sklearn (neural net) added to ensemble
  - Neighbor geohash features (from Grab SEA AI challenge research)
  - Multi-seed averaging for variance reduction
  - NEW: Discrete Drift-Diffusion Model (DDM) Prior
         Simulates spatial demand spread using graph Laplacian and 
         directional drift before passing residuals to ML models.

Architecture:
  FINAL = RidgeStack(LGB_res, XGB_res, CB_res, HGB_res, MLP_res)
  Residual Target = target - HybridPrior
  HybridPrior = 0.5 * LB004 + 0.5 * DDM_Prior
===========================================================================
"""

import os, warnings, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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
# PROVEN CONFIG + NEW DDM CONFIG
# ==========================================================================
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

# DDM (Drift-Diffusion) Parameters
DDM_DIFFUSION_COEFF = 0.05  # Rate at which demand spills to neighbors
DDM_DRIFT_COEFF     = 0.01  # Rate at which demand moves toward hotspots
DDM_NEIGHBOR_RADIUS = 0.02  # Spatial radius for adjacency graph (approx degrees)

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
# SECTION 2 & 3: SMOOTHED MEAN ENCODING & DENOISED ANALOG
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
               if len(self.keys) > 1 else pd.Index(df[self.keys[0]].values))
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

def build_denoised_analog(analog_day_df):
    d = analog_day_df.sort_values(['geohash', 'tmin'])
    roll = d.groupby('geohash')[TARGET].transform(
        lambda s: s.rolling(ROLL_WIN, center=True, min_periods=1).mean()
    )
    roll_series = pd.Series(roll.values, index=pd.MultiIndex.from_arrays([d.geohash, d.tmin])).sort_index()
    enc = compute_encoders(analog_day_df)
    def apply_fn(df):
        ar = roll_series.reindex(pd.MultiIndex.from_arrays([df.geohash, df.tmin])).to_numpy()
        gh = geohash_hour_encoded(enc, df)
        return np.where(np.isnan(ar), gh, W_ROLL * ar + (1 - W_ROLL) * gh)
    return apply_fn

# ==========================================================================
# SECTION 4 & 5: MORNING FEATURES & NEIGHBOR STATS
# ==========================================================================
def morning_features(morning_df):
    m = morning_df[morning_df.tmin <= ANCHOR_END].sort_values(['geohash', 'tmin'])
    g = m.groupby('geohash')[TARGET]
    last = m.loc[m.groupby('geohash')['tmin'].idxmax()].set_index('geohash')[TARGET]
    return pd.DataFrame({'persistence': last, 'morning_std': g.std().fillna(0.0)})

print('\n--- Computing neighbor features ---')
d48_g5_stats = d48.groupby('geohash5')[TARGET].agg(
    neighbor_mean='mean', neighbor_std='std', neighbor_count='count'
).fillna(0)
d48_g4_stats = d48.groupby('geohash4')[TARGET].agg(
    area_mean='mean', area_std='std'
).fillna(0)

# ==========================================================================
# SECTION 5.5: DRIFT-DIFFUSION PDE MODULE (NEW)
# ==========================================================================
print('\n--- Building Spatial Graph for Drift-Diffusion PDE ---')

def build_spatial_laplacian(df):
    unique_gh = df['geohash'].unique()
    gh_to_idx = {gh: i for i, gh in enumerate(unique_gh)}
    coords = np.array([_decode_one(gh) for gh in unique_gh])
    
    # Distance matrix (Euclidean approximation for small lat/lon diffs)
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    # Adjacency Matrix: 1 if neighbors, 0 otherwise
    A = (dist_matrix < DDM_NEIGHBOR_RADIUS).astype(float)
    np.fill_diagonal(A, 0)
    
    # Degree Matrix
    D = np.diag(A.sum(axis=1))
    
    # Graph Laplacian
    L = D - A
    
    # Centroid vector for drift (flowing toward spatial center of mass)
    lat_center, lon_center = coords[:, 0].mean(), coords[:, 1].mean()
    drift_vector = np.zeros_like(coords)
    drift_vector[:, 0] = np.sign(lat_center - coords[:, 0]) # Drift toward center lat
    drift_vector[:, 1] = np.sign(lon_center - coords[:, 1]) # Drift toward center lon
    
    return unique_gh, gh_to_idx, L, drift_vector

global_unique_gh, gh_to_idx, laplacian_matrix, global_drift = build_spatial_laplacian(train)

def apply_ddm_prior(df, morn_feats):
    """
    Solves Advection-Diffusion: dU/dt = -D(Laplacian*U) + v(Drift)
    """
    # 1. Initialize U(t=0) with morning persistence
    U0 = np.zeros(len(global_unique_gh))
    for gh, val in morn_feats['persistence'].dropna().items():
        if gh in gh_to_idx:
            U0[gh_to_idx[gh]] = val
            
    # Calculate global target mean as background fallback
    bg_mean = U0[U0 > 0].mean() if U0.sum() > 0 else 0.5
    U0[U0 == 0] = bg_mean
    
    # Create mapping array for fast lookup
    results = np.zeros(len(df))
    horizons = (df['tmin'] - ANCHOR_END).values
    df_gh_indices = df['geohash'].map(gh_to_idx).fillna(-1).values.astype(int)
    
    # Run simulation forward (max horizon)
    max_h = horizons.max()
    if max_h <= 0: return U0[df_gh_indices]
    
    U_t = U0.copy()
    history = [U_t.copy()]
    
    # Forward Euler time integration (1 step = 1 minute)
    for t in range(1, int(max_h) + 1):
        diffusion = -DDM_DIFFUSION_COEFF * laplacian_matrix.dot(U_t)
        # Simplified linear drift toward center
        drift = DDM_DRIFT_COEFF * (bg_mean - U_t)
        
        dU = diffusion + drift
        U_t = np.clip(U_t + dU, 0, 1)
        history.append(U_t.copy())
        
    history = np.array(history)
    
    # Map predictions back to the dataframe based on horizon
    for i in range(len(df)):
        h = max(0, int(horizons[i]))
        gh_idx = df_gh_indices[i]
        if gh_idx != -1:
            results[i] = history[h, gh_idx]
        else:
            results[i] = bg_mean
            
    return results

# ==========================================================================
# SECTION 6: FEATURE ASSEMBLY (HYBRID PRIOR)
# ==========================================================================
def assemble_features(target_df, analog_fn, morn_feats, g5_stats, g4_stats):
    t = target_df.copy()
    t['analog'] = analog_fn(t)

    # Morning features
    t = t.merge(morn_feats, left_on='geohash', right_index=True, how='left')
    t['persistence'] = t['persistence'].fillna(t['analog'])
    t['morning_std'] = t['morning_std'].fillna(0.0)

    # Calculate Horizons
    t['horizon'] = t['tmin'] - ANCHOR_END
    t['w_exp'] = np.exp(-t['horizon'] / TAU)
    
    # 1. Temporal Prior (LB004)
    t['lb004_pred'] = np.clip(
        t['w_exp'] * t['persistence'] + (1 - t['w_exp']) * t['analog'], 0, 1)
        
    # 2. Spatial Prior (Drift-Diffusion)
    t['ddm_prior'] = apply_ddm_prior(t, morn_feats)
    
    # 3. Hybrid Prior: Blend Time & Space Physics (50/50 split)
    t['hybrid_prior'] = 0.5 * t['lb004_pred'] + 0.5 * t['ddm_prior']
    
    t['analog_minus_persist'] = t['analog'] - t['persistence']

    # Neighbor features 
    t = t.merge(g5_stats, left_on='geohash5', right_index=True, how='left')
    t['neighbor_mean']  = t['neighbor_mean'].fillna(g5_stats['neighbor_mean'].mean())
    t['neighbor_std']   = t['neighbor_std'].fillna(0)
    t['neighbor_count'] = t['neighbor_count'].fillna(0)

    t = t.merge(g4_stats, left_on='geohash4', right_index=True, how='left')
    t['area_mean'] = t['area_mean'].fillna(g4_stats['area_mean'].mean())
    t['area_std']  = t['area_std'].fillna(0)
    t['local_vs_neighbor'] = t['analog'] / (t['neighbor_mean'] + 1e-6)

    return t

# ==========================================================================
# SECTION 7: FEATURE LIST
# ==========================================================================
FEATURES = [
    'hour', 'minute', 'sin_tmin', 'cos_tmin', 'is_rush', 'is_night',
    'lat', 'lon',
    'NumberofLanes', 'LargeVehicles_bin', 'Landmarks_bin',
    'RoadType_enc', 'Weather_enc',
    'temp_missing', 'Temperature', 'lanes_x_large',
    'analog', 'persistence', 'morning_std',
    'horizon', 'w_exp', 'analog_minus_persist',
    'neighbor_mean', 'neighbor_std', 'neighbor_count', 
    'area_mean', 'area_std', 'local_vs_neighbor',
    'lb004_pred', 'ddm_prior' # Inject priors directly into features for ML
]

# ==========================================================================
# SECTION 8 & 9: BUILD FRAMES
# ==========================================================================
print('\n' + '='*60)
print('BUILDING TRAINING & TEST FRAMES')
print('='*60)

tr_tgt = d48[(d48.tmin >= TGT_LO) & (d48.tmin <= TGT_HI)].copy().reset_index(drop=True)
morn_rows = d48[d48.tmin <= ANCHOR_END]
morn48 = morning_features(d48)

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

# IMPORTANT: ML MODELS NOW TARGET THE RESIDUAL OF THE HYBRID PRIOR
hybrid_oof = tr_frame['hybrid_prior'].values
residual_oof = ytr - hybrid_oof

Xtr = tr_frame[FEATURES].copy().fillna(0.0)

# TEST FRAME
fn48_full = build_denoised_analog(d48)
morn49 = morning_features(d49)
test_frame = assemble_features(test, fn48_full, morn49, d48_g5_stats, d48_g4_stats)
hybrid_test = test_frame['hybrid_prior'].values
Xte = test_frame[FEATURES].copy().fillna(0.0)

print(f'Training: X={Xtr.shape} | Hybrid Prior R^2 = {r2_score(ytr, hybrid_oof):.6f}')
print(f'Residual Mean = {residual_oof.mean():.6f} | Std = {residual_oof.std():.4f}')

# ==========================================================================
# SECTION 10: MODEL CONFIGS
# ==========================================================================
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03, 
    'num_leaves': 31, 'min_child_samples': 50, 'reg_alpha': 1.0, 
    'reg_lambda': 10.0, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'n_estimators': 800, 'verbose': -1, 'random_state': SEED,
}
xgb_params = {
    'objective': 'reg:squarederror', 'learning_rate': 0.03, 'max_depth': 5, 
    'min_child_weight': 50, 'reg_alpha': 1.0, 'reg_lambda': 10.0,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'n_estimators': 800, 
    'verbosity': 0, 'random_state': SEED,
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
mlp_params = {
    'hidden_layer_sizes': (128, 64, 32), 'activation': 'relu', 'solver': 'adam',
    'alpha': 0.01, 'learning_rate': 'adaptive', 'learning_rate_init': 0.001,
    'max_iter': 500, 'early_stopping': True, 'validation_fraction': 0.1,
    'n_iter_no_change': 15, 'random_state': SEED, 'verbose': False,
}

MODEL_NAMES = []
if HAS_LGB: MODEL_NAMES.append('LGB')
if HAS_XGB: MODEL_NAMES.append('XGB')
if HAS_CB:  MODEL_NAMES.append('CB')
MODEL_NAMES.extend(['HGB', 'MLP'])

# ==========================================================================
# SECTION 11: 5-FOLD CV (DIRECT + RESIDUAL on HYBRID PRIOR)
# ==========================================================================
print('\n' + '='*60)
print(f'{N_FOLDS}-FOLD CV: DIRECT + RESIDUAL (Hybrid DDM)')
print('='*60)

kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
Xtr_np = Xtr.values.astype(np.float32)

oof_direct = {n: np.zeros(len(ytr)) for n in MODEL_NAMES}
oof_resid  = {n: np.zeros(len(ytr)) for n in MODEL_NAMES}
fmodels_dir, fmodels_res = {n: [] for n in MODEL_NAMES}, {n: [] for n in MODEL_NAMES}
mlp_scalers = {'dir': [], 'res': []}

for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr_np)):
    X_f_tr, X_f_va = Xtr_np[tr_idx], Xtr_np[va_idx]
    y_f_tr, y_f_va = ytr[tr_idx], ytr[va_idx]
    
    # Train target is now the residual from the Hybrid Prior
    r_f_tr = residual_oof[tr_idx] 
    hybrid_f_va = hybrid_oof[va_idx]

    dir_r2, res_r2 = {}, {}

    # Trees
    for name, cls, params in [
        ('LGB', lgb.LGBMRegressor if HAS_LGB else None, lgb_params),
        ('XGB', xgb.XGBRegressor if HAS_XGB else None, xgb_params),
        ('CB', CatBoostRegressor if HAS_CB else None, cb_params),
        ('HGB', HistGradientBoostingRegressor, hgb_params),
    ]:
        if cls is None: continue
        # Direct
        m_d = cls(**params); m_d.fit(X_f_tr, y_f_tr)
        p_d = np.clip(m_d.predict(X_f_va), 0, 1)
        oof_direct[name][va_idx] = p_d
        dir_r2[name] = r2_score(y_f_va, p_d)
        fmodels_dir[name].append(m_d)
        
        # Residual (Predicts the error of the PDE + LB004 blend)
        m_r = cls(**params); m_r.fit(X_f_tr, r_f_tr)
        p_r = np.clip(hybrid_f_va + m_r.predict(X_f_va), 0, 1)
        oof_resid[name][va_idx] = p_r
        res_r2[name] = r2_score(y_f_va, p_r)
        fmodels_res[name].append(m_r)

    # MLP
    scaler_d = StandardScaler().fit(X_f_tr)
    mlp_d = MLPRegressor(**mlp_params)
    mlp_d.fit(scaler_d.transform(X_f_tr), y_f_tr)
    p_mlp_d = np.clip(mlp_d.predict(scaler_d.transform(X_f_va)), 0, 1)
    oof_direct['MLP'][va_idx] = p_mlp_d
    dir_r2['MLP'] = r2_score(y_f_va, p_mlp_d)
    fmodels_dir['MLP'].append(mlp_d)
    mlp_scalers['dir'].append(scaler_d)

    scaler_r = StandardScaler().fit(X_f_tr)
    mlp_r = MLPRegressor(**mlp_params)
    mlp_r.fit(scaler_r.transform(X_f_tr), r_f_tr)
    p_mlp_r = np.clip(hybrid_f_va + mlp_r.predict(scaler_r.transform(X_f_va)), 0, 1)
    oof_resid['MLP'][va_idx] = p_mlp_r
    res_r2['MLP'] = r2_score(y_f_va, p_mlp_r)
    fmodels_res['MLP'].append(mlp_r)
    mlp_scalers['res'].append(scaler_r)

    print(f'Fold {fold_idx+1}:  ' + '  '.join(f'{n}:d={dir_r2[n]:.4f}/r={res_r2[n]:.4f}' for n in MODEL_NAMES))

# ==========================================================================
# SECTION 12 & 13: STACKING ENSEMBLE
# ==========================================================================
oof_best, fmodels_best, approach_best = {}, {}, {}
for name in MODEL_NAMES:
    if r2_score(ytr, oof_resid[name]) > r2_score(ytr, oof_direct[name]):
        oof_best[name], fmodels_best[name], approach_best[name] = oof_resid[name], ('resid', fmodels_res[name]), 'resid'
    else:
        oof_best[name], fmodels_best[name], approach_best[name] = oof_direct[name], ('direct', fmodels_dir[name]), 'direct'

stack_train = np.column_stack([oof_best[n] for n in MODEL_NAMES])
ridge = Ridge(alpha=1.0)
ridge.fit(stack_train, ytr)
stacked_oof = np.clip(ridge.predict(stack_train), 0, 1)
stacked_r2  = r2_score(ytr, stacked_oof)

print(f'\nRidge Stack R^2: {stacked_r2:.6f} | Coefs: {dict(zip(MODEL_NAMES, ridge.coef_))}')

# ==========================================================================
# SECTION 14 & 15 & 16: TEST INFERENCE
# ==========================================================================
print('\n--- Running Test Inference ---')
Xte_np = Xte.values.astype(np.float32)
test_preds = {}

for name in MODEL_NAMES:
    approach, models = fmodels_best[name]
    preds = []
    for i, fm in enumerate(models):
        X_in = mlp_scalers['res' if approach == 'resid' else 'dir'][i].transform(Xte_np) if name == 'MLP' else Xte_np
        
        if approach == 'direct':
            preds.append(np.clip(fm.predict(X_in), 0, 1))
        else:
            preds.append(np.clip(hybrid_test + fm.predict(X_in), 0, 1))
            
    test_preds[name] = np.mean(preds, axis=0)

stack_test = np.column_stack([test_preds[n] for n in MODEL_NAMES])
model_test = np.clip(ridge.predict(stack_test), 0, 1)

# Generate final blends (blending hybrid prior + ML stack)
submissions = {}
for a_pct in [40, 45, 50]:
    a = a_pct / 100.0
    submissions[f'hybrid_a{a_pct:02d}'] = np.clip(a*hybrid_test + (1-a)*model_test, 0, 1)

pd.DataFrame({ID: test_frame[ID].values, TARGET: submissions['hybrid_a45']}).to_csv('x_submission_ddm.csv', index=False)
print(f'Saved: x_submission_ddm.csv (Mean Pred: {submissions["hybrid_a45"].mean():.4f})')