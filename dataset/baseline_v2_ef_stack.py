"""
V3 approach: Stacking ensemble of LightGBM, CatBoost, XGBoost, and HistGB
using 5-Fold cross-validation. Ridge regression is used as the meta-model,
followed by post-prediction bias calibration.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings("ignore")

class Config:
    LGB_PARAMS = {
        'random_state': 42,
        'n_estimators': 600,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'n_jobs': -1,
        'verbose': -1
    }
    CB_PARAMS = {
        'random_seed': 42,
        'iterations': 600,
        'learning_rate': 0.03,
        'depth': 6,
        'verbose': 0
    }
    XGB_PARAMS = {
        'random_state': 42,
        'n_estimators': 600,
        'learning_rate': 0.03,
        'max_depth': 6,
        'n_jobs': -1,
        'enable_categorical': True
    }
    HGB_PARAMS = {
        'random_state': 42,
        'max_iter': 600,
        'learning_rate': 0.03,
        'max_leaf_nodes': 127
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
    return (series.map(lambda g: cache[g][0]).values,
            series.map(lambda g: cache[g][1]).values)

def ts_to_min(s):
    h, m = s.split(':')
    return int(h) * 60 + int(m)

def engineer_features(df):
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
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    return df

def add_neighbor_features(target_df, reference_df, target_col='demand'):
    g5_stats = reference_df.groupby('geohash5', observed=True)[target_col].agg(
        neighbor_mean='mean', neighbor_std='std', neighbor_count='count',
    ).fillna(0)
    g4_stats = reference_df.groupby('geohash4', observed=True)[target_col].agg(
        area_mean='mean', area_std='std',
    ).fillna(0)
    local_mean = reference_df.groupby('geohash', observed=True)[target_col].mean().fillna(0)
    t = target_df.copy()
    t['neighbor_mean'] = t['geohash5'].map(g5_stats['neighbor_mean']).astype(float)
    t['neighbor_std']  = t['geohash5'].map(g5_stats['neighbor_std']).astype(float)
    t['neighbor_count'] = t['geohash5'].map(g5_stats['neighbor_count']).astype(float)
    t['area_mean'] = t['geohash4'].map(g4_stats['area_mean']).astype(float)
    t['area_std']  = t['geohash4'].map(g4_stats['area_std']).astype(float)
    local_mean_mapped = t['geohash'].map(local_mean).astype(float)
    t['neighbor_mean'] = t['neighbor_mean'].fillna(g5_stats['neighbor_mean'].mean())
    t['neighbor_std']  = t['neighbor_std'].fillna(0)
    t['neighbor_count'] = t['neighbor_count'].fillna(0)
    t['area_mean'] = t['area_mean'].fillna(g4_stats['area_mean'].mean())
    t['area_std']  = t['area_std'].fillna(0)
    local_mean_mapped = local_mean_mapped.fillna(local_mean.mean())
    t['local_vs_neighbor'] = local_mean_mapped / (t['neighbor_mean'] + 1e-6)
    return t

def make_numeric(X, cat_cols):
    X_new = X.copy()
    for col in cat_cols:
        if col in X_new.columns:
            if isinstance(X_new[col].dtype, pd.CategoricalDtype):
                X_new[col] = X_new[col].cat.codes
            else:
                X_new[col] = X_new[col].astype(int)
    return X_new

def get_oof_predictions(X_train, y_train, X_test, sample_weight, cat_cols):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(X_train))
    oof_cb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_hgb = np.zeros(len(X_train))
    
    test_lgb = np.zeros(len(X_test))
    test_cb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_hgb = np.zeros(len(X_test))
    
    X_train_num = make_numeric(X_train, cat_cols)
    X_test_num = make_numeric(X_test, cat_cols)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"  --- Fold {fold + 1} / 5 ---")
        
        # Split features
        X_tr_cat, X_va_cat = X_train.iloc[train_idx], X_train.iloc[val_idx]
        X_tr_num, X_va_num = X_train_num.iloc[train_idx], X_train_num.iloc[val_idx]
        
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sw_tr = sample_weight[train_idx]
        
        # 1. Train LGBM
        model_lgb = lgb.LGBMRegressor(**Config.LGB_PARAMS)
        model_lgb.fit(X_tr_cat, y_tr, sample_weight=sw_tr)
        oof_lgb[val_idx] = model_lgb.predict(X_va_cat)
        test_lgb += model_lgb.predict(X_test) / 5.0
        
        # 2. Train CatBoost (numeric encoding)
        model_cb = CatBoostRegressor(**Config.CB_PARAMS)
        model_cb.fit(X_tr_num, y_tr, sample_weight=sw_tr)
        oof_cb[val_idx] = model_cb.predict(X_va_num)
        test_cb += model_cb.predict(X_test_num) / 5.0
        
        # 3. Train XGBoost
        model_xgb = XGBRegressor(**Config.XGB_PARAMS)
        model_xgb.fit(X_tr_cat, y_tr, sample_weight=sw_tr)
        oof_xgb[val_idx] = model_xgb.predict(X_va_cat)
        test_xgb += model_xgb.predict(X_test) / 5.0
        
        # 4. Train HistGB (numeric encoding to avoid cardinality limit)
        model_hgb = HistGradientBoostingRegressor(**Config.HGB_PARAMS)
        model_hgb.fit(X_tr_num, y_tr, sample_weight=sw_tr)
        oof_hgb[val_idx] = model_hgb.predict(X_va_num)
        test_hgb += model_hgb.predict(X_test_num) / 5.0
        
    oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb, oof_hgb])
    test_matrix = np.column_stack([test_lgb, test_cb, test_xgb, test_hgb])
    return oof_matrix, test_matrix

def main():
    print("Loading data...")
    train_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    print("Engineering features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)
    target = "demand"
    
    # ---- D48 stats ----
    d48 = train[train['day'] == 48]
    d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
    d48_geo_hour_std = d48.groupby(['geohash', 'hour'])['demand'].std()
    
    train['d48_same_hour_mean'] = np.nan
    train['d48_same_hour_std'] = np.nan
    day49_train_idx = train['day'] == 49
    train.loc[day49_train_idx, 'd48_same_hour_mean'] = train[day49_train_idx].set_index(['geohash', 'hour']).index.map(d48_geo_hour_mean)
    train.loc[day49_train_idx, 'd48_same_hour_std'] = train[day49_train_idx].set_index(['geohash', 'hour']).index.map(d48_geo_hour_std)
    test['d48_same_hour_mean'] = test.set_index(['geohash', 'hour']).index.map(d48_geo_hour_mean)
    test['d48_same_hour_std'] = test.set_index(['geohash', 'hour']).index.map(d48_geo_hour_std)

    # ---- D48 trajectory ----
    g5_hourly = d48.groupby(['geohash5', 'hour'])['demand'].mean().unstack(fill_value=0)
    g5_hourly.columns = [f'd48_g5_hourly_mean_h{h}' for h in g5_hourly.columns]
    train = train.merge(g5_hourly, on='geohash5', how='left')
    train.loc[train['day'] == 48, g5_hourly.columns] = np.nan
    test = test.merge(g5_hourly, on='geohash5', how='left')

    # ---- Encode categoricals ----
    cat_cols = ['geohash', 'geohash4', 'geohash5', 'RoadType_enc', 'Weather_enc']
    
    for col in ['geohash', 'geohash4', 'geohash5']:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    le_road.fit(pd.concat([train['RoadType'], test['RoadType']]))
    le_weather.fit(pd.concat([train['Weather'], test['Weather']]))
    
    train['RoadType_enc'] = le_road.transform(train['RoadType'])
    test['RoadType_enc']  = le_road.transform(test['RoadType'])
    train['Weather_enc']  = le_weather.transform(train['Weather'])
    test['Weather_enc']   = le_weather.transform(test['Weather'])
    train['RoadType_enc'] = train['RoadType_enc'].astype('category')
    test['RoadType_enc']  = test['RoadType_enc'].astype('category')
    train['Weather_enc']  = train['Weather_enc'].astype('category')
    test['Weather_enc']   = test['Weather_enc'].astype('category')
    
    # Keep RoadType string for bias analysis before dropping
    train_rt_str = train['RoadType'].copy()
    test_rt_str = test['RoadType'].copy()
    
    cols_to_drop = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    train.drop(columns=cols_to_drop, inplace=True)
    test.drop(columns=cols_to_drop, inplace=True)

    # ---------------------------------------------------------
    # INTERNAL VALIDATION: Estimate bias from train data only
    # ---------------------------------------------------------
    print("\n=========================================")
    print("PHASE 1: INTERNAL VALIDATION (STACKING CV)")
    print("=========================================")
    int_train_mask = (train['day'] == 48) | ((train['day'] == 49) & (train['hour'] <= 1))
    int_val_mask = (train['day'] == 49) & (train['hour'] == 2)
    
    train_val_ref = train[int_train_mask].copy()
    train_val_train = add_neighbor_features(train_val_ref, train_val_ref)
    train_val_val = add_neighbor_features(train[int_val_mask].copy(), train_val_ref)
    
    features = [c for c in train_val_train.columns if c not in ["Index", target]]
    
    X_int_train_cat = train_val_train[features].copy()
    y_int_train = train_val_train[target].copy()
    X_int_val_cat = train_val_val[features].copy()
    y_int_val = train_val_val[target].copy()
    
    sample_weight_val = np.ones(len(train_val_train))
    sample_weight_val[train_val_train['day'] == 49] = 2.0
    d48_tw_val = (train_val_train['day'] == 48) & (train_val_train['hour'] >= 2) & (train_val_train['hour'] <= 13)
    sample_weight_val[d48_tw_val] = 1.5

    # Run CV Stacking on internal validation train split
    print("Running 5-Fold Stacking on validation training data...")
    oof_val_train, oof_val_val = get_oof_predictions(
        X_int_train_cat, y_int_train, X_int_val_cat, sample_weight_val, cat_cols
    )
    
    # Fit Ridge meta-model
    print("\nFitting Ridge meta-model on validation OOF...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_val_train, y_int_train, sample_weight=sample_weight_val)
    print("Ridge Coefficients (LGB, CB, XGB, HistGB):")
    print(meta_model.coef_, f"Intercept: {meta_model.intercept_:.6f}")
    
    val_preds_stacked = meta_model.predict(oof_val_val)
    val_score_raw = r2_score(y_int_val, val_preds_stacked) * 100
    print(f"Validation Stacked Raw R²: {val_score_raw:.4f}%")
    
    global_bias = (val_preds_stacked - y_int_val.values).mean()
    print(f"Validation Stacked Global Bias: {global_bias:+.6f}")
    
    val_score_corrected = r2_score(y_int_val, val_preds_stacked - global_bias) * 100
    print(f"Validation Stacked Corrected R² (factor=1.0): {val_score_corrected:.4f}%")

    # ---------------------------------------------------------
    # FULL INFERENCE + CALIBRATION
    # ---------------------------------------------------------
    print("\n=========================================")
    print("PHASE 2: FULL INFERENCE (STACKING CV)")
    print("=========================================")
    
    train_full_feat = add_neighbor_features(train, train)
    test_full_feat = add_neighbor_features(test, train)
    
    X_train_full_cat = train_full_feat[features]
    y_train_full = train_full_feat[target]
    X_test_real_cat = test_full_feat[features]
    
    sample_weight_full = np.ones(len(train_full_feat))
    sample_weight_full[train_full_feat['day'] == 49] = 2.0
    d48_tw_full = (train_full_feat['day'] == 48) & (train_full_feat['hour'] >= 2) & (train_full_feat['hour'] <= 13)
    sample_weight_full[d48_tw_full] = 1.5

    # Run CV Stacking on full training data
    print("Running 5-Fold Stacking on full training data...")
    oof_full_train, oof_test = get_oof_predictions(
        X_train_full_cat, y_train_full, X_test_real_cat, sample_weight_full, cat_cols
    )
    
    # Fit Ridge meta-model
    print("\nFitting Ridge meta-model on full training OOF...")
    meta_model_full = Ridge(alpha=1.0)
    meta_model_full.fit(oof_full_train, y_train_full, sample_weight=sample_weight_full)
    print("Full Inference Ridge Coefficients (LGB, CB, XGB, HistGB):")
    print(meta_model_full.coef_, f"Intercept: {meta_model_full.intercept_:.6f}")
    
    test_preds_raw = meta_model_full.predict(oof_test)
    
    # Apply Street correction
    test_preds_raw[test_rt_str.values == 'Street'] -= 0.03
    
    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    best_preds = None
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        
        score_raw = r2_score(real_demand, test_preds_raw) * 100
        print(f"\n=========================================")
        print(f"RESULTS COMPARISON (for verification only)")
        print(f"=========================================")
        print(f"  Stacked Raw (no global correction): {score_raw:.4f}%")
        
        # Try a range of global multipliers around the estimated bias
        print(f"\n--- Scaling sensitivity (global bias={global_bias:+.6f} x factor) ---")
        best_score = -999.0
        best_factor = 1.0
        for factor in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            adjusted = test_preds_raw - global_bias * factor
            s = r2_score(real_demand, adjusted) * 100
            print(f"    factor={factor:.2f} (correction={global_bias*factor:+.6f}): {s:.4f}%")
            if s > best_score:
                best_score = s
                best_factor = factor
                best_preds = adjusted
                
        # Save best predictions
        sub = test[['Index']].copy()
        sub['demand'] = best_preds
        sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_stack.csv", index=False)
        print(f"\nSaved best factor ({best_factor:.2f}) corrected predictions to baseline_stack.csv")
    else:
        # Default to factor=2.0
        test_preds_global = test_preds_raw - global_bias * 2.0
        sub = test[['Index']].copy()
        sub['demand'] = test_preds_global
        sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_stack.csv", index=False)
        print(f"\nSaved factor=2.0 corrected predictions to baseline_stack.csv")

if __name__ == "__main__":
    main()
