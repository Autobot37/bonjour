"""
===========================================================================
Gridlock 2.0 -- curr_best92_dd.py
===========================================================================
All proven techniques from v4 (LB=91.08) + curr_best91 additions
(MLP, neighbor geohash features, multi-seed averaging) PLUS:

  NEW: Drift-Diffusion Prior Module
  ----------------------------------
  LB004 is a special case of the Ornstein-Uhlenbeck process:
    E[X_{t+h} | X_t] = mu + (X_t - mu) * exp(-theta * h)
  with global theta=1/TAU fixed for ALL geohashes.

  This module:
    1. Estimates theta PER GEOHASH from d48 lag-1 autocorrelation
    2. Adds a drift term c (OLS slope on d48): dd_prior = ou_pred + (c/theta)*(1-exp(-theta*h))
    3. Computes spatial Laplacian: neighbor_demand - local_demand
    4. Computes ou_surprise: morning deviation from typical pattern
    5. Exposes dd_prior as an additional blend anchor

  New features (10): ou_theta, ou_sigma, ou_drift_rate, ou_pred, dd_prior,
                     ou_vs_lb004, velocity_d48, spatial_laplacian,
                     ou_surprise, surprise_decay

Architecture:
  FINAL = alpha * ANCHOR + (1-alpha) * RidgeStack(LGB, XGB, CB, HGB, MLP)
  ANCHOR = dd_prior (if OOF R2 >= lb004) else lb004
  Proven best alpha: 0.45

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
# PROVEN CONFIG
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
# SECTION 5: NEIGHBOR DEMAND FEATURES
# ==========================================================================
print('\n--- Computing neighbor features (Grab SEA AI) ---')
d48_g5_stats = d48.groupby('geohash5')[TARGET].agg(
    neighbor_mean='mean',
    neighbor_std='std',
    neighbor_count='count',
).fillna(0)

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
    t = t.merge(morn_feats, left_on='geohash', right_index=True, how='left')
    t['persistence'] = t['persistence'].fillna(t['analog'])
    t['morning_std'] = t['morning_std'].fillna(0.0)
    t['horizon'] = t['tmin'] - ANCHOR_END
    t['w_exp'] = np.exp(-t['horizon'] / TAU)
    t['lb004_pred'] = np.clip(
        t['w_exp'] * t['persistence'] + (1 - t['w_exp']) * t['analog'], 0, 1)
    t['analog_minus_persist'] = t['analog'] - t['persistence']
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
# SECTION 7: BASE FEATURE LIST (22 proven + 6 neighbor)
# ==========================================================================
FEATURES = [
    'hour', 'minute', 'sin_tmin', 'cos_tmin', 'is_rush', 'is_night',
    'lat', 'lon',
    'NumberofLanes', 'LargeVehicles_bin', 'Landmarks_bin',
    'RoadType_enc', 'Weather_enc',
    'temp_missing', 'Temperature', 'lanes_x_large',
    'analog', 'persistence', 'morning_std',
    'horizon', 'w_exp',
    'analog_minus_persist',
    'neighbor_mean', 'neighbor_std', 'neighbor_count',
    'area_mean', 'area_std', 'local_vs_neighbor',
]

print(f'\nBase features: {len(FEATURES)}')

# ==========================================================================
# SECTION 8: BUILD TRAINING FRAME
# ==========================================================================
print('\n' + '='*60)
print('BUILDING TRAINING FRAME')
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
lb004_oof = tr_frame['lb004_pred'].values
residual_oof = ytr - lb004_oof

Xtr = tr_frame[FEATURES].copy().fillna(0.0)
print(f'Training: X={Xtr.shape}')
print(f'LB004 OOF R^2 = {r2_score(ytr, lb004_oof):.6f}')

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

# ==========================================================================
# SECTION 9.5: DRIFT-DIFFUSION PRIOR MODULE
# ==========================================================================
print('\n' + '='*60)
print('SECTION 9.5: DRIFT-DIFFUSION PRIOR MODULE')
print('='*60)

# ---------------------------------------------------------------------------
# DD-1  Per-geohash OU parameter estimation from d48
#
# theta: -log(rho_lag1) / dt  where rho = lag-1 autocorrelation of demand
#        clipped to [1e-4, 0.05]  <=>  TAU range [20 min, 10000 min]
#        global 1/TAU = 0.00417 sits comfortably inside this range
# sigma: empirical std of demand on d48 (diffusion coefficient)
# drift_rate: OLS slope of demand vs tmin [demand/minute]
# ---------------------------------------------------------------------------
def _estimate_ou_params(day_df, tgt=TARGET, min_obs=3):
    rows = []
    for gh, g in day_df.sort_values(['geohash', 'tmin']).groupby('geohash'):
        g  = g.sort_values('tmin')
        v  = g[tgt].values.astype(float)
        t  = g['tmin'].values.astype(float)
        mu = float(np.nanmean(v))
        sig = float(np.std(v)) if np.std(v) > 1e-6 else 0.01
        if len(v) < min_obs:
            rows.append(dict(geohash=gh, ou_theta=1./TAU,
                             ou_sigma=sig, ou_drift_rate=0., ou_mu=mu))
            continue
        dts = np.diff(t); dts = dts[dts > 0]
        dt  = float(np.mean(dts)) if len(dts) else 15.
        ctr = v - mu
        var = float(np.mean(ctr ** 2))
        rho = float(np.clip(np.mean(ctr[:-1] * ctr[1:]) / var,
                            0.01, 0.9999)) if var > 1e-8 else 0.5
        th  = float(np.clip(-np.log(rho) / dt, 1e-4, 0.05))
        tc  = t - t.mean()
        denom = float(np.sum(tc ** 2))
        dr  = float(np.sum(tc * v) / denom) if denom > 0 else 0.
        rows.append(dict(geohash=gh, ou_theta=th, ou_sigma=sig,
                         ou_drift_rate=dr, ou_mu=mu))
    return pd.DataFrame(rows).set_index('geohash')

print('  [DD1] Estimating OU parameters per geohash from d48...')
_ou_params = _estimate_ou_params(d48)
print(f'        Geohashes fitted   : {len(_ou_params)}')
print(f'        theta mean         : {_ou_params.ou_theta.mean():.5f}  '
      f'(global 1/TAU={1/TAU:.5f})')
print(f'        pct faster global  : {(_ou_params.ou_theta > 1/TAU).mean()*100:.1f}%')
print(f'        sigma mean         : {_ou_params.ou_sigma.mean():.4f}')
print(f'        drift_rate mean    : {_ou_params.ou_drift_rate.mean():.7f} demand/min')

# ---------------------------------------------------------------------------
# DD-2  d48 temporal velocity field (central differences)
# ---------------------------------------------------------------------------
print('  [DD2] d48 velocity field...')
_dv = d48.sort_values(['geohash', 'tmin']).copy()
_gg = _dv.groupby('geohash', sort=False)
_dv['_dn'] = _gg[TARGET].shift(-1)
_dv['_dp'] = _gg[TARGET].shift(1)
_dv['_tn'] = _gg['tmin'].shift(-1)
_dv['_tp'] = _gg['tmin'].shift(1)
_both = _dv['_dn'].notna() & _dv['_dp'].notna()
_fwd  = ~_both & _dv['_dn'].notna()
_dv['_vel'] = np.nan
_dv.loc[_both, '_vel'] = (
    (_dv.loc[_both, '_dn'] - _dv.loc[_both, '_dp']) /
    (_dv.loc[_both, '_tn'] - _dv.loc[_both, '_tp']).clip(lower=1)
)
_dv.loc[_fwd, '_vel'] = (
    (_dv.loc[_fwd, '_dn'] - _dv.loc[_fwd, TARGET]) /
    (_dv.loc[_fwd, '_tn'] - _dv.loc[_fwd, 'tmin']).clip(lower=1)
)
_vel_gh_hr = _dv.groupby(['geohash', 'hour'])['_vel'].mean()
_vel_gh    = _dv.groupby('geohash')['_vel'].mean().fillna(0.)
print(f'        {len(_vel_gh_hr)} (geohash x hour) velocity entries')

# ---------------------------------------------------------------------------
# DD-3  Spatial Laplacian: mean_demand(geohash5 neighbors) - demand(g)
# ---------------------------------------------------------------------------
print('  [DD3] Spatial Laplacian from d48 geohash5 neighbor means...')
_g5h_demand = d48.groupby(['geohash5', 'hour'])[TARGET].mean()

# ---------------------------------------------------------------------------
# DD-4  Full-training morning reference for ou_surprise
#        Using all training days -> stable baseline, works for train and test
# ---------------------------------------------------------------------------
print('  [DD4] Full-training morning reference for ou_surprise...')
_morn_ref_all = (train[train.tmin <= ANCHOR_END]
                 .groupby('geohash')[TARGET].mean())

# ---------------------------------------------------------------------------
# DD-5  Feature assembly function
# ---------------------------------------------------------------------------
def add_dd_features(df):
    """
    Append 10 drift-diffusion features to a pre-assembled feature frame.
    Required columns: analog, persistence, horizon, lb004_pred,
                      geohash, geohash5, hour.
    """
    t = df.copy()

    # Map per-geohash OU params via .map() (preserves index, no reshape risk)
    for col, default in [('ou_theta',      1. / TAU),
                          ('ou_sigma',      0.05),
                          ('ou_drift_rate', 0.),
                          ('ou_mu',         np.nan)]:
        t[col] = t['geohash'].map(_ou_params[col]).fillna(default).astype(float)
    t['ou_mu'] = t['ou_mu'].fillna(t['analog'])

    h     = t['horizon'].values.astype(float)
    theta = t['ou_theta'].values
    mu    = t['analog'].values.astype(float)
    X0    = t['persistence'].values.astype(float)
    decay = np.exp(-theta * h)

    # ou_pred: OU with per-geohash theta (generalises LB004's global TAU)
    t['ou_pred']     = np.clip(mu + (X0 - mu) * decay, 0., 1.)

    # ou_vs_lb004: where local theta diverges from global TAU
    t['ou_vs_lb004'] = t['ou_pred'].values - t['lb004_pred'].values

    # dd_prior: ou_pred + integrated drift correction (c/theta)*(1-exp(-theta*h))
    sth              = np.maximum(theta, 1e-6)
    t['dd_prior']    = np.clip(
        t['ou_pred'].values + t['ou_drift_rate'].values * (1. - decay) / sth,
        0., 1.)

    # drift_extrap: pure linear extrapolation (ablation / reference feature)
    t['drift_extrap'] = np.clip(X0 + t['ou_drift_rate'].values * h, 0., 1.)

    # velocity_d48: d48 demand velocity at (geohash, hour), fallback to geohash mean
    idx_gh = pd.MultiIndex.from_arrays([t['geohash'], t['hour']])
    vel    = _vel_gh_hr.reindex(idx_gh).values.astype(float)
    nanm   = np.isnan(vel)
    if nanm.any():
        fallback      = t['geohash'].map(_vel_gh).fillna(0.).values
        vel           = np.where(nanm, fallback, vel)
    t['velocity_d48'] = vel

    # spatial_laplacian: geohash5 neighbor mean demand - local analog
    idx_g5 = pd.MultiIndex.from_arrays([t['geohash5'], t['hour']])
    nbr    = _g5h_demand.reindex(idx_g5).values.astype(float)
    t['spatial_laplacian'] = np.nan_to_num(nbr - mu)

    # ou_surprise: morning observation vs typical morning (full training reference)
    mref              = t['geohash'].map(_morn_ref_all).fillna(t['persistence']).values
    t['ou_surprise']  = X0 - mref

    # surprise_decay: shock attenuated by OU reversion over horizon
    t['surprise_decay'] = t['ou_surprise'].values * decay

    return t

# ---------------------------------------------------------------------------
# DD-6  Apply to tr_frame and test_frame
# ---------------------------------------------------------------------------
print('  [DD5/6] Applying DD features...')
tr_frame   = add_dd_features(tr_frame)
test_frame = add_dd_features(test_frame)

# Refresh downstream arrays
ytr          = tr_frame[TARGET].values
lb004_oof    = tr_frame['lb004_pred'].values
residual_oof = ytr - lb004_oof

# ---------------------------------------------------------------------------
# DD-7  Extend FEATURES and rebuild Xtr / Xte
# ---------------------------------------------------------------------------
DD_FEATURES = [
    'ou_theta',           # per-geohash OU reversion speed [1/min]
    'ou_sigma',           # demand volatility (diffusion coefficient)
    'ou_drift_rate',      # intraday drift from d48 OLS [demand/min]
    'ou_pred',            # OU prediction with local theta
    'dd_prior',           # OU + integrated drift correction
    'ou_vs_lb004',        # local theta vs global TAU divergence
    'velocity_d48',       # d48 demand velocity at (geohash, hour)
    'spatial_laplacian',  # demand pressure from geohash5 neighbors
    'ou_surprise',        # morning: today - typical pattern
    'surprise_decay',     # shock * exp(-theta*h)
]

FEATURES = FEATURES + DD_FEATURES
Xtr = tr_frame[FEATURES].copy().fillna(0.)
Xte = test_frame[FEATURES].copy().fillna(0.)
print(f'\n  Features: {len(FEATURES)} ({len(FEATURES) - 10} base + 10 DD)')
print(f'  Xtr: {Xtr.shape} | Xte: {Xte.shape}')

# ---------------------------------------------------------------------------
# DD-8  Diagnostics
# ---------------------------------------------------------------------------
_ou_r2 = r2_score(ytr, np.clip(tr_frame['ou_pred'],  0, 1))
_dd_r2 = r2_score(ytr, np.clip(tr_frame['dd_prior'], 0, 1))
_lb_r2 = r2_score(ytr, lb004_oof)

print(f'\n  Prior R2 comparison (OOF on d48 target window):')
print(f'    LB004  (global TAU={TAU:.0f}): {_lb_r2:.6f}  score={max(0,100*_lb_r2):.2f}')
print(f'    ou_pred (local theta):  {_ou_r2:.6f}  score={max(0,100*_ou_r2):.2f}  delta={100*(_ou_r2-_lb_r2):+.4f}')
print(f'    dd_prior (OU+drift):    {_dd_r2:.6f}  score={max(0,100*_dd_r2):.2f}  delta={100*(_dd_r2-_lb_r2):+.4f}')

print('\n  Distribution check (DD features):')
for f in DD_FEATURES:
    tm, te = Xtr[f].mean(), Xte[f].mean()
    ratio  = te / tm if abs(tm) > 1e-8 else float('nan')
    flag   = 'WARN' if not np.isnan(ratio) and abs(ratio - 1.) > 0.35 else '    '
    print(f'  {flag}  {f:22s}  train={tm:8.4f}  test={te:8.4f}  r={ratio:.2f}')

# Store DD priors and select best blend anchor
dd_prior_oof_arr  = np.clip(tr_frame['dd_prior'],  0, 1).values
dd_prior_test_arr = np.clip(test_frame['dd_prior'], 0, 1).values
ou_pred_test_arr  = np.clip(test_frame['ou_pred'],  0, 1).values

# Auto-select: use dd_prior as blend anchor if it's >= lb004 OOF
_dd_anchor_name = 'dd_prior' if _dd_r2 >= _lb_r2 else 'lb004'
_dd_anchor_oof  = dd_prior_oof_arr  if _dd_r2 >= _lb_r2 else lb004_oof
_dd_anchor_test = dd_prior_test_arr if _dd_r2 >= _lb_r2 else lb004_test
print(f'\n  DD blend anchor selected: {_dd_anchor_name}')
print('\n--- DRIFT-DIFFUSION MODULE COMPLETE ---\n')

# ==========================================================================
# SECTION 10: MODEL CONFIGS
# ==========================================================================
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
mlp_params = {
    'hidden_layer_sizes': (128, 64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 15,
    'random_state': SEED,
    'verbose': False,
}

MODEL_NAMES = []
if HAS_LGB: MODEL_NAMES.append('LGB')
if HAS_XGB: MODEL_NAMES.append('XGB')
if HAS_CB:  MODEL_NAMES.append('CB')
MODEL_NAMES.append('HGB')
MODEL_NAMES.append('MLP')

print(f'Models: {MODEL_NAMES}')

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
mlp_scalers = {'dir': [], 'res': []}

for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xtr_np)):
    X_f_tr, X_f_va = Xtr_np[tr_idx], Xtr_np[va_idx]
    y_f_tr, y_f_va = ytr[tr_idx], ytr[va_idx]
    r_f_tr = residual_oof[tr_idx]
    lb004_f_va = lb004_oof[va_idx]

    dir_r2, res_r2 = {}, {}

    for name, cls, params in [
        ('LGB', lgb.LGBMRegressor if HAS_LGB else None, lgb_params),
        ('XGB', xgb.XGBRegressor  if HAS_XGB else None, xgb_params),
        ('CB',  CatBoostRegressor  if HAS_CB  else None, cb_params),
        ('HGB', HistGradientBoostingRegressor,            hgb_params),
    ]:
        if cls is None:
            continue
        m_d = cls(**params); m_d.fit(X_f_tr, y_f_tr)
        p_d = np.clip(m_d.predict(X_f_va), 0, 1)
        oof_direct[name][va_idx] = p_d
        dir_r2[name] = r2_score(y_f_va, p_d)
        fmodels_dir[name].append(m_d)

        m_r = cls(**params); m_r.fit(X_f_tr, r_f_tr)
        p_r = np.clip(lb004_f_va + m_r.predict(X_f_va), 0, 1)
        oof_resid[name][va_idx] = p_r
        res_r2[name] = r2_score(y_f_va, p_r)
        fmodels_res[name].append(m_r)

    scaler_d = StandardScaler().fit(X_f_tr)
    X_f_tr_sc = scaler_d.transform(X_f_tr)
    X_f_va_sc = scaler_d.transform(X_f_va)

    mlp_d = MLPRegressor(**mlp_params)
    mlp_d.fit(X_f_tr_sc, y_f_tr)
    p_mlp_d = np.clip(mlp_d.predict(X_f_va_sc), 0, 1)
    oof_direct['MLP'][va_idx] = p_mlp_d
    dir_r2['MLP'] = r2_score(y_f_va, p_mlp_d)
    fmodels_dir['MLP'].append(mlp_d)
    mlp_scalers['dir'].append(scaler_d)

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

ridge = Ridge(alpha=1.0)
ridge.fit(stack_train, ytr)
stacked_oof = np.clip(ridge.predict(stack_train), 0, 1)
stacked_r2  = r2_score(ytr, stacked_oof)

best_w_r2, best_weights = -1e9, None
n_m = len(MODEL_NAMES)
if n_m == 5:
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

tree_names = [n for n in MODEL_NAMES if n != 'MLP']
stack_trees = np.column_stack([oof_best[n] for n in tree_names])
ridge_trees = Ridge(alpha=1.0)
ridge_trees.fit(stack_trees, ytr)
trees_oof = np.clip(ridge_trees.predict(stack_trees), 0, 1)
trees_r2 = r2_score(ytr, trees_oof)
print(f'\nTrees-only Stack R^2: {trees_r2:.6f}  Score: {max(0,100*trees_r2):.2f}')
print(f'MLP added:       R^2: {stacked_r2:.6f}  Score: {max(0,100*stacked_r2):.2f}')
print(f'MLP delta: {(stacked_r2 - trees_r2)*100:+.4f} points')

ensemble_methods = {'Ridge_Stack': stacked_r2, 'Best_Wt': best_w_r2, 'Trees_Only': trees_r2}
best_method = max(ensemble_methods, key=ensemble_methods.get)
print(f'\nBest method: {best_method}')

if best_method == 'Trees_Only':
    model_oof = trees_oof
elif best_method == 'Ridge_Stack':
    model_oof = stacked_oof
else:
    model_oof = np.clip(stack_train @ best_weights, 0, 1)

# ==========================================================================
# SECTION 14: TEST PREDICTIONS
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

print('\n--- Model Correlation Matrix ---')
corr_matrix = np.corrcoef([test_preds[n] for n in MODEL_NAMES])
print(f'{"":8s}  ' + '  '.join(f'{n:6s}' for n in MODEL_NAMES))
for i, n in enumerate(MODEL_NAMES):
    print(f'{n:8s}  ' + '  '.join(f'{corr_matrix[i,j]:.4f}' for j in range(len(MODEL_NAMES))))

stack_test       = np.column_stack([test_preds[n] for n in MODEL_NAMES])
stack_test_trees = np.column_stack([test_preds[n] for n in tree_names])

if best_method == 'Trees_Only':
    model_test = np.clip(ridge_trees.predict(stack_test_trees), 0, 1)
elif best_method == 'Ridge_Stack':
    model_test = np.clip(ridge.predict(stack_test), 0, 1)
else:
    model_test = np.clip(stack_test @ best_weights, 0, 1)

print(f'\nEnsemble test ({best_method}): mean={model_test.mean():.4f}')

# ==========================================================================
# SECTION 15: MULTI-SEED AVERAGING
# ==========================================================================
print('\n' + '='*60)
print('MULTI-SEED AVERAGING')
print('='*60)

seed_preds = [model_test.copy()]

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
            ('XGB', xgb.XGBRegressor  if HAS_XGB else None, xgb_params),
            ('CB',  CatBoostRegressor  if HAS_CB  else None, cb_params),
            ('HGB', HistGradientBoostingRegressor,            hgb_params),
        ]:
            if cls is None or name not in tree_names:
                continue
            p_copy = dict(params)
            if 'random_state' in p_copy: p_copy['random_state'] = extra_seed
            if 'random_seed'  in p_copy: p_copy['random_seed']  = extra_seed

            approach = approach_best.get(name, 'direct')
            m = cls(**p_copy)
            if approach == 'resid':
                m.fit(X_f_tr, r_f_tr)
                oof_s[name][va_idx] = np.clip(lb004_f_va + m.predict(X_f_va), 0, 1)
            else:
                m.fit(X_f_tr, y_f_tr)
                oof_s[name][va_idx] = np.clip(m.predict(X_f_va), 0, 1)
            fmod_s[name].append(m)

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
    pred_s  = np.clip(ridge_trees.predict(stack_s), 0, 1)
    seed_preds.append(pred_s)
    print(f'  Seed {extra_seed}: mean={pred_s.mean():.4f}')

model_test_multiseed = np.mean(seed_preds, axis=0)
print(f'\nMulti-seed average ({len(seed_preds)} seeds): mean={model_test_multiseed.mean():.4f}')

# ==========================================================================
# SECTION 16: GENERATE ALL SUBMISSIONS
# ==========================================================================
print('\n' + '='*60)
print('GENERATING SUBMISSIONS')
print('='*60)

submissions = {}

# ---- Proven lb004-blended (single seed + multi-seed) ----
for a_pct in [40, 43, 44, 45, 46, 47, 50]:
    a = a_pct / 100.0
    submissions[f'best_a{a_pct:02d}']  = np.clip(a*lb004_test + (1-a)*model_test, 0, 1)
    submissions[f'multi_a{a_pct:02d}'] = np.clip(a*lb004_test + (1-a)*model_test_multiseed, 0, 1)

# ---- DD-prior blended (uses best anchor: dd_prior or lb004) ----
for a_pct in [40, 43, 45, 47, 50]:
    a = a_pct / 100.0
    submissions[f'dd_a{a_pct:02d}']       = np.clip(a*_dd_anchor_test + (1-a)*model_test,           0, 1)
    submissions[f'dd_multi_a{a_pct:02d}'] = np.clip(a*_dd_anchor_test + (1-a)*model_test_multiseed, 0, 1)

# ---- Pure model ----
submissions['best_pure']  = model_test
submissions['multi_pure'] = model_test_multiseed

for name, pred in submissions.items():
    sub = pd.DataFrame({ID: test_frame[ID].values, TARGET: pred})
    sub.to_csv(f'x_{name}.csv', index=False)
    print(f'  {name:25s}  mean={pred.mean():.4f}  min={pred.min():.4f}  max={pred.max():.4f}')

# Default submissions
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

print('--- Per-Model OOF ---')
for name in MODEL_NAMES:
    r2 = r2_score(ytr, oof_best[name])
    print(f'{name:6s}  R^2={r2:.6f}  Score={max(0,100*r2):.2f}  ({approach_best[name]})')

print('\n--- DD Prior vs LB004 ---')
print(f'  LB004   : {_lb_r2:.6f}  score={max(0,100*_lb_r2):.2f}')
print(f'  ou_pred : {_ou_r2:.6f}  score={max(0,100*_ou_r2):.2f}  delta={100*(_ou_r2-_lb_r2):+.4f}')
print(f'  dd_prior: {_dd_r2:.6f}  score={max(0,100*_dd_r2):.2f}  delta={100*(_dd_r2-_lb_r2):+.4f}')
print(f'  Blend anchor used: {_dd_anchor_name}')

print('\n--- Ensemble OOF ---')
print(f'Ridge Stack (all): {stacked_r2:.6f}  Score: {max(0,100*stacked_r2):.2f}')
print(f'Trees only:        {trees_r2:.6f}  Score: {max(0,100*trees_r2):.2f}')
print(f'Best method:       {best_method}')

print('\n--- Blend Grid (lb004 anchor) ---')
for a_pct in [40, 43, 44, 45, 46, 47, 50]:
    a = a_pct / 100.0
    r2 = r2_score(ytr, np.clip(a*lb004_oof + (1-a)*model_oof, 0, 1))
    print(f'  a={a:.2f}  OOF={max(0,100*r2):.2f}  -> best_a{a_pct:02d}.csv / multi_a{a_pct:02d}.csv')

print('\n--- Blend Grid (DD anchor) ---')
for a_pct in [40, 43, 45, 47, 50]:
    a = a_pct / 100.0
    r2 = r2_score(ytr, np.clip(a*_dd_anchor_oof + (1-a)*model_oof, 0, 1))
    print(f'  a={a:.2f}  OOF={max(0,100*r2):.2f}  -> dd_a{a_pct:02d}.csv / dd_multi_a{a_pct:02d}.csv')

if HAS_LGB and fmodels_best.get('LGB'):
    print('\n--- Top Features (LGB, top 20) ---')
    imp = fmodels_best['LGB'][1][0].feature_importances_
    for fname, importance in sorted(zip(FEATURES, imp), key=lambda x: x[1], reverse=True)[:20]:
        tag = ' [DD]' if fname in DD_FEATURES else ''
        print(f'  {fname:25s}  {importance:8.0f}{tag}')

results_file = 'results_utk.csv'
new_rows = []
for name in MODEL_NAMES:
    new_rows.append({
        'model': f'best92_dd_{name}',
        'final_r2': r2_score(ytr, oof_best[name]),
        'mean_r2': r2_score(ytr, oof_best[name]),
        'fold_1': 0, 'fold_2': 0, 'fold_3': 0, 'fold_4': 0, 'fold_5': 0,
    })
new_rows.append({'model': 'best92_dd_stack', 'final_r2': stacked_r2, 'mean_r2': stacked_r2,
                 'fold_1': 0, 'fold_2': 0, 'fold_3': 0, 'fold_4': 0, 'fold_5': 0})
new_rows.append({'model': 'best92_dd_prior', 'final_r2': _dd_r2, 'mean_r2': _dd_r2,
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
print('  best_a45.csv      = proven lb004 blend (single seed)')
print('  multi_a45.csv     = proven lb004 blend (multi-seed)')
print(f'  dd_a45.csv        = {_dd_anchor_name} blend (single seed) [NEW]')
print(f'  dd_multi_a45.csv  = {_dd_anchor_name} blend (multi-seed) [NEW]')
print(f'  dd_a43.csv        = {_dd_anchor_name} more model weight [NEW]')
print(f'  dd_a47.csv        = {_dd_anchor_name} more anchor weight [NEW]')
print(f'{"="*60}')
print('DONE!')