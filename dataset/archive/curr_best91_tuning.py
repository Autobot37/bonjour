import os, warnings, time, random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize

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

SEED = 42
TARGET = 'demand'
ID = 'Index'

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

def compute_encoders(history_df, target=TARGET, m=20.0):
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
def build_denoised_analog(analog_day_df, roll_win, w_roll, smooth_m):
    d_unique = analog_day_df.drop_duplicates(subset=['geohash', 'tmin'])
    d = d_unique.sort_values(['geohash', 'tmin'])
    roll = d.groupby('geohash')[TARGET].transform(
        lambda s: s.rolling(roll_win, center=True, min_periods=1).mean()
    )
    roll_series = pd.Series(
        roll.values,
        index=pd.MultiIndex.from_arrays([d.geohash, d.tmin])
    ).sort_index()
    enc = compute_encoders(d_unique, target=TARGET, m=smooth_m)
    def apply_fn(df):
        ar = roll_series.reindex(
            pd.MultiIndex.from_arrays([df.geohash, df.tmin])
        ).to_numpy()
        gh = geohash_hour_encoded(enc, df)
        return np.where(np.isnan(ar), gh, w_roll * ar + (1 - w_roll) * gh)
    return apply_fn

# ==========================================================================
# SECTION 4: MORNING FEATURES
# ==========================================================================
def morning_features(morning_df, anchor_end):
    m = morning_df[morning_df.tmin <= anchor_end].sort_values(['geohash', 'tmin'])
    if len(m) == 0:
        return pd.DataFrame(columns=['persistence', 'morning_std'])
    g = m.groupby('geohash')[TARGET]
    last = m.loc[m.groupby('geohash')['tmin'].idxmax()].set_index('geohash')[TARGET]
    return pd.DataFrame({'persistence': last, 'morning_std': g.std().fillna(0.0)})

# ==========================================================================
# SECTION 5: FEATURE ASSEMBLY
# ==========================================================================
def assemble_features(target_df, analog_fn, morn_feats, g5_stats, g4_stats, anchor_end, tau):
    t = target_df.copy()
    t['analog'] = analog_fn(t)

    # Morning features
    if not morn_feats.empty:
        t = t.merge(morn_feats, left_on='geohash', right_index=True, how='left')
    else:
        t['persistence'] = np.nan
        t['morning_std'] = 0.0
    t['persistence'] = t['persistence'].fillna(t['analog'])
    t['morning_std'] = t['morning_std'].fillna(0.0)

    # Horizon + LB004
    t['horizon'] = t['tmin'] - anchor_end
    t['w_exp'] = np.exp(-t['horizon'] / tau)
    t['lb004_pred'] = np.clip(
        t['w_exp'] * t['persistence'] + (1 - t['w_exp']) * t['analog'], 0, 1)
    t['analog_minus_persist'] = t['analog'] - t['persistence']

    # Neighbor features
    t = t.merge(g5_stats, left_on='geohash5', right_index=True, how='left')
    t['neighbor_mean']  = t['neighbor_mean'].fillna(g5_stats['neighbor_mean'].mean() if not g5_stats.empty else 0)
    t['neighbor_std']   = t['neighbor_std'].fillna(0)
    t['neighbor_count'] = t['neighbor_count'].fillna(0)

    t = t.merge(g4_stats, left_on='geohash4', right_index=True, how='left')
    t['area_mean'] = t['area_mean'].fillna(g4_stats['area_mean'].mean() if not g4_stats.empty else 0)
    t['area_std']  = t['area_std'].fillna(0)

    # Ratio of this location vs its neighborhood
    t['local_vs_neighbor'] = t['analog'] / (t['neighbor_mean'] + 1e-6)

    return t

FEATURES = [
    'hour', 'minute', 'sin_tmin', 'cos_tmin', 'is_rush', 'is_night',
    'lat', 'lon',
    'NumberofLanes', 'LargeVehicles_bin', 'Landmarks_bin',
    'RoadType_enc', 'Weather_enc',
    'temp_missing', 'Temperature', 'lanes_x_large',
    'analog', 'persistence', 'morning_std',
    'horizon', 'w_exp',
    'analog_minus_persist',
    'neighbor_mean',        
    'neighbor_std',         
    'neighbor_count',       
    'area_mean',            
    'area_std',             
    'local_vs_neighbor',    
]

# ==========================================================================
# SECTION 6: HYPERPARAMETER ENSEMBLING HELPERS
# ==========================================================================
def find_best_blend(preds_dict, lb004_pred, y_target):
    # Optimize weights for a list of predictions + lb004 to maximize R2 score
    keys = list(preds_dict.keys())
    X = np.column_stack([preds_dict[k] for k in keys] + [lb004_pred])
    
    # Try SLSQP first
    def loss(w):
        pred = np.clip(X @ w, 0, 1)
        return -r2_score(y_target, pred)
    
    bounds = [(0, 1) for _ in range(X.shape[1])]
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    w0 = np.ones(X.shape[1]) / X.shape[1]
    
    try:
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons)
        pred_slsqp = np.clip(X @ res.x, 0, 1)
        r2_slsqp = -res.fun
        best_w = res.x
        best_pred = pred_slsqp
        best_r2 = r2_slsqp
        method = 'slsqp'
    except Exception:
        # Fallback to Ridge
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(X, y_target)
        pred_ridge = np.clip(ridge.predict(X), 0, 1)
        r2_ridge = r2_score(y_target, pred_ridge)
        best_w = ridge.coef_
        best_pred = pred_ridge
        best_r2 = r2_ridge
        method = 'ridge'
        
    return method, best_w, best_r2, best_pred


def train_proxy_split(n_rows, seed=SEED):
    """One deterministic held-out split from train.csv-derived rows only."""
    kf = KFold(5, shuffle=True, random_state=seed)
    return next(kf.split(np.arange(n_rows)))


def score_config_on_train_proxy(
    Xtr,
    ytr,
    lb004_oof,
    config_models,
    include_cb=False,
    include_mlp=False,
):
    """Train on 80% of d48 target rows and score on held-out d48 rows."""
    tr_idx, va_idx = train_proxy_split(len(ytr))
    X_fit, X_val = Xtr[tr_idx], Xtr[va_idx]
    y_fit, y_val = ytr[tr_idx], ytr[va_idx]
    lb_fit, lb_val = lb004_oof[tr_idx], lb004_oof[va_idx]
    residual_fit = y_fit - lb_fit
    preds = {}

    lgb_cfg = config_models.get('lgb')
    if HAS_LGB and lgb_cfg is not None:
        m = lgb.LGBMRegressor(objective='regression', **lgb_cfg, verbose=-1, random_state=SEED)
        m.fit(X_fit, y_fit)
        preds['LGB_dir'] = np.clip(m.predict(X_val), 0, 1)
        mr = lgb.LGBMRegressor(objective='regression', **lgb_cfg, verbose=-1, random_state=SEED)
        mr.fit(X_fit, residual_fit)
        preds['LGB_res'] = np.clip(lb_val + mr.predict(X_val), 0, 1)

    xgb_cfg = config_models.get('xgb')
    if HAS_XGB and xgb_cfg is not None:
        m = xgb.XGBRegressor(objective='reg:squarederror', **xgb_cfg, verbosity=0, random_state=SEED)
        m.fit(X_fit, y_fit)
        preds['XGB_dir'] = np.clip(m.predict(X_val), 0, 1)
        mr = xgb.XGBRegressor(objective='reg:squarederror', **xgb_cfg, verbosity=0, random_state=SEED)
        mr.fit(X_fit, residual_fit)
        preds['XGB_res'] = np.clip(lb_val + mr.predict(X_val), 0, 1)

    hgb_cfg = config_models.get('hgb')
    if hgb_cfg is not None:
        m = HistGradientBoostingRegressor(**hgb_cfg, early_stopping=False, random_state=SEED)
        m.fit(X_fit, y_fit)
        preds['HGB_dir'] = np.clip(m.predict(X_val), 0, 1)
        mr = HistGradientBoostingRegressor(**hgb_cfg, early_stopping=False, random_state=SEED)
        mr.fit(X_fit, residual_fit)
        preds['HGB_res'] = np.clip(lb_val + mr.predict(X_val), 0, 1)

    if include_cb and HAS_CB:
        cb_cfg = config_models.get('cb', {'iterations': 600, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 8.0})
        m = CatBoostRegressor(**cb_cfg, random_seed=SEED, verbose=0)
        m.fit(X_fit, y_fit)
        preds['CB_dir'] = np.clip(m.predict(X_val), 0, 1)
        mr = CatBoostRegressor(**cb_cfg, random_seed=SEED, verbose=0)
        mr.fit(X_fit, residual_fit)
        preds['CB_res'] = np.clip(lb_val + mr.predict(X_val), 0, 1)

    if include_mlp:
        mlp_cfg = config_models.get('mlp', {'hidden_layer_sizes': (128, 64), 'alpha': 0.01, 'learning_rate_init': 0.001})
        scaler = StandardScaler().fit(X_fit)
        X_fit_sc, X_val_sc = scaler.transform(X_fit), scaler.transform(X_val)
        m = MLPRegressor(**mlp_cfg, activation='relu', solver='adam', max_iter=300,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=12, random_state=SEED, verbose=False)
        m.fit(X_fit_sc, y_fit)
        preds['MLP_dir'] = np.clip(m.predict(X_val_sc), 0, 1)
        mr = MLPRegressor(**mlp_cfg, activation='relu', solver='adam', max_iter=300,
                          early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=12, random_state=SEED, verbose=False)
        mr.fit(X_fit_sc, residual_fit)
        preds['MLP_res'] = np.clip(lb_val + mr.predict(X_val_sc), 0, 1)

    return find_best_blend(preds, lb_val, y_val)

# ==========================================================================
# SECTION 7: MAIN SEARCH PIPELINE
# ==========================================================================
def main():
    print("Loading datasets...")
    train_raw = pd.read_csv('train.csv')
    test_raw  = pd.read_csv('test.csv')

    print("Extracting base features...")
    train = add_base_features(train_raw)
    test  = add_base_features(test_raw)
    
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
    
    d48_g5_stats = d48.groupby('geohash5')[TARGET].agg(
        neighbor_mean='mean', neighbor_std='std', neighbor_count='count').fillna(0)
    d48_g4_stats = d48.groupby('geohash4')[TARGET].agg(
        area_mean='mean', area_std='std').fillna(0)

    # ----------------------------------------------------------------------
    # STAGE 1: COARSE RANDOM SEARCH (Tree models only, cheap estimators)
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 1: COARSE RANDOM SEARCH (40 ITERATIONS)")
    print("="*70)
    
    stage1_results = []
    
    for iter_idx in range(40):
        # Sample feature parameters
        smooth_m = random.choice([5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
        roll_win = random.choice([3, 5, 7])
        tau = random.choice([150.0, 180.0, 210.0, 240.0, 270.0, 300.0])
        w_roll = random.choice([0.1, 0.2, 0.25, 0.3, 0.4])
        anchor_end = random.choice([90, 120, 150])
        
        # Sample cheap model parameters
        lgb_lr = random.choice([0.02, 0.03, 0.05, 0.08])
        lgb_leaves = random.choice([15, 31, 63])
        lgb_min_child = random.choice([20, 50, 100])
        lgb_reg_alpha = random.choice([0.1, 1.0, 5.0])
        lgb_reg_lambda = random.choice([0.1, 1.0, 10.0])
        
        xgb_lr = random.choice([0.02, 0.03, 0.05, 0.08])
        xgb_depth = random.choice([3, 5, 7])
        xgb_min_child = random.choice([10, 50, 100])
        xgb_reg_alpha = random.choice([0.1, 1.0, 5.0])
        xgb_reg_lambda = random.choice([0.1, 1.0, 10.0])
        
        hgb_lr = random.choice([0.02, 0.03, 0.05, 0.08])
        hgb_leaves = random.choice([15, 31, 63])
        hgb_min_child = random.choice([20, 50, 100])
        hgb_l2 = random.choice([0.1, 1.0, 10.0])
        
        # Build features for day 48 and day 49
        morn48 = morning_features(d48, anchor_end)
        tr_tgt = d48[(d48.tmin >= 135) & (d48.tmin <= 825)].copy().reset_index(drop=True)
        morn_rows = d48[d48.tmin <= anchor_end]
        
        # OOF Analog calculation for training
        oof_analog = np.full(len(tr_tgt), np.nan)
        kf_analog = KFold(5, shuffle=True, random_state=SEED)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf_analog.split(tr_tgt)):
            src = pd.concat([morn_rows, tr_tgt.iloc[tr_idx]], ignore_index=True)
            fn_fold = build_denoised_analog(src, roll_win, w_roll, smooth_m)
            oof_analog[va_idx] = fn_fold(tr_tgt.iloc[va_idx])
            
        _oof_s = pd.Series(oof_analog, index=tr_tgt.index)
        oof_fn = lambda df: _oof_s.reindex(df.index).values
        tr_frame = assemble_features(tr_tgt, oof_fn, morn48, d48_g5_stats, d48_g4_stats, anchor_end, tau)
        
        Xtr = tr_frame[FEATURES].copy().fillna(0.0).values.astype(np.float32)
        ytr = tr_frame[TARGET].values
        lb004_oof = tr_frame['lb004_pred'].values
        model_cfg = {
            'lgb': {'learning_rate': lgb_lr, 'num_leaves': lgb_leaves,
                    'min_child_samples': lgb_min_child, 'reg_alpha': lgb_reg_alpha,
                    'reg_lambda': lgb_reg_lambda, 'n_estimators': 150},
            'xgb': {'learning_rate': xgb_lr, 'max_depth': xgb_depth,
                    'min_child_weight': xgb_min_child, 'reg_alpha': xgb_reg_alpha,
                    'reg_lambda': xgb_reg_lambda, 'n_estimators': 150},
            'hgb': {'learning_rate': hgb_lr, 'max_leaf_nodes': hgb_leaves,
                    'min_samples_leaf': hgb_min_child, 'l2_regularization': hgb_l2,
                    'max_iter': 150}
        }

        method, weights, r2_score_val, pred_ensemble = score_config_on_train_proxy(
            Xtr, ytr, lb004_oof, model_cfg, include_cb=False, include_mlp=False
        )
        
        config = {
            'feature': {'smooth_m': smooth_m, 'roll_win': roll_win, 'tau': tau, 'w_roll': w_roll, 'anchor_end': anchor_end},
            'lgb': model_cfg['lgb'],
            'xgb': model_cfg['xgb'],
            'hgb': model_cfg['hgb']
        }
        
        stage1_results.append((r2_score_val, config))
        print(f"Iter {iter_idx+1:2d}/40 | smooth={smooth_m:4.1f}, roll={roll_win}, tau={tau:5.1f}, w_roll={w_roll:4.2f}, anchor={anchor_end:3d} | R2 = {r2_score_val*100:8.5f}%")
        
    # Sort and pick top 5
    stage1_results.sort(key=lambda x: x[0], reverse=True)
    top_5 = stage1_results[:5]
    print(f"\nTop 5 Coarse R2 Scores: {[round(x[0]*100, 4) for x in top_5]}")

    # ----------------------------------------------------------------------
    # STAGE 2: FINE TUNING (Add CatBoost + MLP, use larger estimators)
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 2: FINE TUNING TOP 5 CONFIGURATIONS (TRAIN-PROXY VALIDATION)")
    print("="*70)
    
    best_overall_r2 = -1e9
    best_config = None
    best_model_weights = None
    best_method = None
    best_preds = None
    
    for i, (coarse_r2, config) in enumerate(top_5):
        print(f"\nEvaluating Top Config {i+1}/5 (Coarse R2: {coarse_r2*100:.4f}%)")
        
        f_cfg = config['feature']
        smooth_m, roll_win, tau, w_roll, anchor_end = f_cfg['smooth_m'], f_cfg['roll_win'], f_cfg['tau'], f_cfg['w_roll'], f_cfg['anchor_end']
        
        # Build train-only proxy features for hyperparameter validation.
        morn48 = morning_features(d48, anchor_end)
        tr_tgt = d48[(d48.tmin >= 135) & (d48.tmin <= 825)].copy().reset_index(drop=True)
        morn_rows = d48[d48.tmin <= anchor_end]
        
        oof_analog = np.full(len(tr_tgt), np.nan)
        kf_analog = KFold(5, shuffle=True, random_state=SEED)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf_analog.split(tr_tgt)):
            src = pd.concat([morn_rows, tr_tgt.iloc[tr_idx]], ignore_index=True)
            fn_fold = build_denoised_analog(src, roll_win, w_roll, smooth_m)
            oof_analog[va_idx] = fn_fold(tr_tgt.iloc[va_idx])
            
        _oof_s = pd.Series(oof_analog, index=tr_tgt.index)
        oof_fn = lambda df: _oof_s.reindex(df.index).values
        tr_frame = assemble_features(tr_tgt, oof_fn, morn48, d48_g5_stats, d48_g4_stats, anchor_end, tau)

        Xtr = tr_frame[FEATURES].copy().fillna(0.0).values.astype(np.float32)
        ytr = tr_frame[TARGET].values
        lb004_oof = tr_frame['lb004_pred'].values

        # Add heavier optional model families, then score on a held-out split of train.csv.
        if HAS_CB:
            cb_lr = random.choice([0.02, 0.03, 0.05])
            cb_depth = random.choice([4, 5, 6])
            cb_l2 = random.choice([3.0, 5.0, 10.0])
            config['cb'] = {'iterations': 600, 'learning_rate': cb_lr, 'depth': cb_depth, 'l2_leaf_reg': cb_l2}

        mlp_h = random.choice([(128, 64, 32), (128, 64)])
        mlp_alpha = random.choice([0.005, 0.01, 0.05])
        mlp_lr_init = random.choice([0.001, 0.002])
        config['mlp'] = {'hidden_layer_sizes': mlp_h, 'alpha': mlp_alpha, 'learning_rate_init': mlp_lr_init}

        model_cfg = {
            'lgb': dict(config['lgb'], n_estimators=600),
            'xgb': dict(config['xgb'], n_estimators=600),
            'hgb': dict(config['hgb'], max_iter=600),
            'cb': config.get('cb'),
            'mlp': config['mlp'],
        }
        method, weights, r2_score_val, pred_ensemble = score_config_on_train_proxy(
            Xtr, ytr, lb004_oof, model_cfg, include_cb=HAS_CB, include_mlp=True
        )

        print(f"  Train-proxy R2 Score: {r2_score_val*100:.5f}% (Method: {method})")
        
        if r2_score_val > best_overall_r2:
            best_overall_r2 = r2_score_val
            best_config = config.copy()
            best_model_weights = weights
            best_method = method
            best_preds = pred_ensemble
            
    print("\n" + "="*70)
    print(f"STAGE 2 COMPLETE. HIGHEST TRAIN-PROXY R2: {best_overall_r2*100:.5f}%")
    print("="*70)
    print("Best Hyperparameters:")
    print(best_config)

    # ----------------------------------------------------------------------
    # STAGE 3: FINAL FULL ENSEMBLE WITH MULTI-SEED 5-FOLD CV
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 3: RUNNING FULL 5-FOLD CV + MULTI-SEED AVERAGING ON OPTIMAL HYP")
    print("="*70)
    
    # We retrieve the best hyperparameters
    f_cfg = best_config['feature']
    smooth_m, roll_win, tau, w_roll, anchor_end = f_cfg['smooth_m'], f_cfg['roll_win'], f_cfg['tau'], f_cfg['w_roll'], f_cfg['anchor_end']
    
    # Model parameters
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse',
        **best_config['lgb'], 'n_estimators': 800, 'verbose': -1, 'random_state': SEED
    }
    xgb_params = {
        'objective': 'reg:squarederror',
        **best_config['xgb'], 'n_estimators': 800, 'verbosity': 0, 'random_state': SEED
    }
    cb_params = {
        **best_config.get('cb', {'iterations': 800, 'learning_rate': 0.03, 'depth': 5, 'l2_leaf_reg': 10.0}),
        'iterations': 800, 'random_seed': SEED, 'verbose': 0
    }
    hgb_params = {
        **best_config['hgb'], 'max_iter': 800, 'early_stopping': False, 'random_state': SEED
    }
    mlp_params = {
        **best_config.get('mlp', {'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.01, 'learning_rate_init': 0.001}),
        'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'early_stopping': True,
        'validation_fraction': 0.1, 'n_iter_no_change': 15, 'random_state': SEED, 'verbose': False
    }
    
    morn48 = morning_features(d48, anchor_end)
    tr_tgt = d48[(d48.tmin >= 135) & (d48.tmin <= 825)].copy().reset_index(drop=True)
    morn_rows = d48[d48.tmin <= anchor_end]
    
    oof_analog = np.full(len(tr_tgt), np.nan)
    kf_analog = KFold(5, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf_analog.split(tr_tgt)):
        src = pd.concat([morn_rows, tr_tgt.iloc[tr_idx]], ignore_index=True)
        fn_fold = build_denoised_analog(src, roll_win, w_roll, smooth_m)
        oof_analog[va_idx] = fn_fold(tr_tgt.iloc[va_idx])
        
    _oof_s = pd.Series(oof_analog, index=tr_tgt.index)
    oof_fn = lambda df: _oof_s.reindex(df.index).values
    tr_frame = assemble_features(tr_tgt, oof_fn, morn48, d48_g5_stats, d48_g4_stats, anchor_end, tau)
    ytr = tr_frame[TARGET].values
    lb004_oof = tr_frame['lb004_pred'].values
    residual_oof = ytr - lb004_oof
    
    Xtr = tr_frame[FEATURES].copy().fillna(0.0)
    Xtr_np = Xtr.values.astype(np.float32)
    
    morn49 = morning_features(d49, anchor_end)
    fn_te = build_denoised_analog(d48, roll_win, w_roll, smooth_m)
    test_frame = assemble_features(test, fn_te, morn49, d48_g5_stats, d48_g4_stats, anchor_end, tau)
    lb004_test = test_frame['lb004_pred'].values
    Xte = test_frame[FEATURES].copy().fillna(0.0)
    Xte_np = Xte.values.astype(np.float32)
    
    MODEL_NAMES = []
    if HAS_LGB: MODEL_NAMES.append('LGB')
    if HAS_XGB: MODEL_NAMES.append('XGB')
    if HAS_CB:  MODEL_NAMES.append('CB')
    MODEL_NAMES.append('HGB')
    MODEL_NAMES.append('MLP')
    
    # Direct + Residual 5-Fold training on main Seed (42)
    print("\n--- Training Main Seed 5-Fold models ---")
    kf = KFold(5, shuffle=True, random_state=SEED)
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
        
        # Tree models
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
            fmodels_dir[name].append(m_d)
            
            # Residual
            m_r = cls(**params); m_r.fit(X_f_tr, r_f_tr)
            p_r = np.clip(lb004_f_va + m_r.predict(X_f_va), 0, 1)
            oof_resid[name][va_idx] = p_r
            fmodels_res[name].append(m_r)
            
        # MLP scaling and training
        scaler_d = StandardScaler().fit(X_f_tr)
        X_f_tr_sc = scaler_d.transform(X_f_tr)
        X_f_va_sc = scaler_d.transform(X_f_va)
        
        # MLP Direct
        mlp_d = MLPRegressor(**mlp_params)
        mlp_d.fit(X_f_tr_sc, y_f_tr)
        p_mlp_d = np.clip(mlp_d.predict(X_f_va_sc), 0, 1)
        oof_direct['MLP'][va_idx] = p_mlp_d
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
        fmodels_res['MLP'].append(mlp_r)
        mlp_scalers['res'].append(scaler_r)
        
    # Select best approach (Winner) per model based on OOF R2
    print(f'\n{"Model":6s}  {"Direct":>10s}  {"Residual":>10s}  {"Winner":>8s}')
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
        
    # Test Predictions for main Seed
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
        
    stack_train = np.column_stack([oof_best[n] for n in MODEL_NAMES])
    
    # Ridge stack ensembling
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack_train, ytr)
    stacked_oof = np.clip(ridge.predict(stack_train), 0, 1)
    stacked_r2 = r2_score(ytr, stacked_oof)
    print(f'\nRidge Stack OOF R^2: {stacked_r2:.6f}')
    
    # Try weighted average search on OOF
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
        print(f'Best Wt Avg OOF R^2: {best_w_r2:.6f}')
        
    # Stacking ensemble for test set
    stack_test = np.column_stack([test_preds[n] for n in MODEL_NAMES])
    tree_names = [n for n in MODEL_NAMES if n != 'MLP']
    stack_trees = np.column_stack([oof_best[n] for n in tree_names])
    ridge_trees = Ridge(alpha=1.0)
    ridge_trees.fit(stack_trees, ytr)
    trees_oof = np.clip(ridge_trees.predict(stack_trees), 0, 1)
    trees_r2 = r2_score(ytr, trees_oof)
    
    ensemble_methods = {'Ridge_Stack': stacked_r2, 'Best_Wt': best_w_r2, 'Trees_Only': trees_r2}
    best_ensemble_method = max(ensemble_methods, key=ensemble_methods.get)
    print(f'Best stacking ensemble method: {best_ensemble_method}')
    
    if best_ensemble_method == 'Trees_Only':
        model_test = np.clip(ridge_trees.predict(np.column_stack([test_preds[n] for n in tree_names])), 0, 1)
        model_oof = trees_oof
    elif best_ensemble_method == 'Ridge_Stack':
        model_test = np.clip(ridge.predict(stack_test), 0, 1)
        model_oof = stacked_oof
    else:
        model_test = np.clip(stack_test @ best_weights, 0, 1)
        model_oof = np.clip(stack_train @ best_weights, 0, 1)
        
    # Multi-Seed Averaging (variance reduction) over Seeds 123 and 777
    print("\n--- Running Multi-Seed Averaging (Seeds 123, 777) ---")
    seed_preds = [model_test.copy()]
    
    for extra_seed in [123, 777]:
        print(f"  Training Seed {extra_seed} 5-Fold models...")
        kf_s = KFold(5, shuffle=True, random_state=extra_seed)
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
                if cls is None or name not in tree_names: continue
                p_copy = dict(params)
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
        
    model_test_multiseed = np.mean(seed_preds, axis=0)
    print(f"Multi-seed average computed over {len(seed_preds)} seeds.")

    # Generate all submissions
    print("\n--- Generating final submission CSVs ---")
    submissions = {}
    
    # Blends for single seed
    for a_pct in [40, 43, 44, 45, 46, 47, 50]:
        a = a_pct / 100.0
        submissions[f'best_a{a_pct:02d}'] = np.clip(a*lb004_test + (1-a)*model_test, 0, 1)
        
    # Blends for multi seed
    for a_pct in [40, 43, 44, 45, 46, 47, 50]:
        a = a_pct / 100.0
        submissions[f'multi_a{a_pct:02d}'] = np.clip(a*lb004_test + (1-a)*model_test_multiseed, 0, 1)
        
    submissions['best_pure'] = model_test
    submissions['multi_pure'] = model_test_multiseed
    
    # Write all candidate submissions. Selection is deterministic, not answer-key based.
    for name, pred in submissions.items():
        print(f"  Submission {name:20s} | mean={pred.mean():.6f} min={pred.min():.6f} max={pred.max():.6f}")
        sub_df = pd.DataFrame({ID: test_frame[ID].values, TARGET: pred})
        sub_df.to_csv(f'x_{name}.csv', index=False)

    default_name = 'best_a45'
    default_pred = submissions[default_name]
    pd.DataFrame({ID: test_frame[ID].values, TARGET: default_pred}).to_csv('x_submission.csv', index=False)
    pd.DataFrame({ID: test_frame[ID].values, TARGET: default_pred}).to_csv('x_heavy_submission.csv', index=False)
    pd.DataFrame({ID: test_frame[ID].values, TARGET: default_pred}).to_csv('curr_best91_tuning_submission.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL OUTPUTS")
    print("="*70)
    print(f"Default submission selected without answer-key scoring: {default_name}")
    print("Saved: x_submission.csv, x_heavy_submission.csv, curr_best91_tuning_submission.csv")
    print("Done!")

if __name__ == '__main__':
    main()
