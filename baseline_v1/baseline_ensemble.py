import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
    
    CAT_PARAMS = {
        'random_state': 42,
        'iterations': 800,
        'learning_rate': 0.05,
        'depth': 8,
        'verbose': 0,
        'task_type': 'CPU',
        'l2_leaf_reg': 3
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

def get_oof(model, X, y, X_test):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    
    X_model = X.copy()
    X_test_model = X_test.copy()
    
    # Reset index for KFold slicing
    X_model = X_model.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    X_test_model = X_test_model.reset_index(drop=True)
            
    for train_idx, val_idx in kf.split(X_model):
        X_tr, y_tr = X_model.iloc[train_idx], y_reset.iloc[train_idx]
        X_va = X_model.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_va)
        
    # Fit on all training data for test inference
    model.fit(X_model, y_reset)
    test_pred_final = model.predict(X_test_model)
    
    return oof, test_pred_final

def main():
    print("Loading data...")
    train_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    print("Engineering Moonshot features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)

    target = "demand"
    
    # 1. Encode high-cardinality string columns to category natively
    for col in ['geohash', 'geohash4', 'geohash5']:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    # 2. Encode low-cardinality categoricals
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
    
    # 3. Drop raw string columns that have numeric/category encodings
    cols_to_drop = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    train.drop(columns=cols_to_drop, inplace=True)
    test.drop(columns=cols_to_drop, inplace=True)
    
    features = [c for c in train.columns if c not in ["Index", target]]
    cat_cols = ['geohash', 'geohash4', 'geohash5', 'RoadType_enc', 'Weather_enc']

    # ---------------------------------------------------------
    # INTERNAL TIME-BASED VALIDATION
    # ---------------------------------------------------------
    print("\n=========================================")
    print("INTERNAL TIME-BASED VALIDATION (MANUAL STACKING)")
    print("=========================================")
    int_train_mask = (train['day'] == 48) & (train['hour'] <= 12)
    int_val_mask = (train['day'] == 48) & (train['hour'] > 12) & (train['hour'] < 14)
    
    X_int_train = train[int_train_mask][features].copy()
    y_int_train = train[int_train_mask][target].copy()
    X_int_val = train[int_val_mask][features].copy()
    y_int_val = train[int_val_mask][target].copy()
    
    models = {
        'lgb': lgb.LGBMRegressor(**Config.LGB_PARAMS),
        'cat': CatBoostRegressor(**Config.CAT_PARAMS, cat_features=cat_cols)
    }
    
    oof_train = np.zeros((len(X_int_train), 2))
    oof_val = np.zeros((len(X_int_val), 2))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name} for Internal Validation...")
        oof_train[:, i], oof_val[:, i] = get_oof(model, X_int_train, y_int_train, X_int_val)
        
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_train, y_int_train)
    val_preds = ridge.predict(oof_val)
    
    val_score = r2_score(y_int_val, val_preds) * 100
    print(f"INTERNAL VALIDATION R2 SCORE: {val_score:.6f}")

    # ---------------------------------------------------------
    # FULL INFERENCE FOR REAL LEADERBOARD
    # ---------------------------------------------------------
    print("\n=========================================")
    print("FULL INFERENCE FOR REAL LEADERBOARD")
    print("=========================================")
    X_train_full = train[features]
    y_train_full = train[target]
    X_test_real = test[features]

    oof_train_full = np.zeros((len(X_train_full), 2))
    oof_test = np.zeros((len(X_test_real), 2))
    
    models_full = {
        'lgb': lgb.LGBMRegressor(**Config.LGB_PARAMS),
        'cat': CatBoostRegressor(**Config.CAT_PARAMS, cat_features=cat_cols)
    }
    
    for i, (name, model) in enumerate(models_full.items()):
        print(f"Training {name} for Full Inference...")
        oof_train_full[:, i], oof_test[:, i] = get_oof(model, X_train_full, y_train_full, X_test_real)
        
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(oof_train_full, y_train_full)
    preds_ens = ridge_full.predict(oof_test)
    
    sub = test[['Index']].copy()
    sub['demand'] = preds_ens
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_stack.csv", index=False)
    print("Saved baseline_stack.csv to dataset/")

    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        
        score_ens = r2_score(real_demand, preds_ens) * 100
        
        print(f"\n=========================================")
        print(f"STACKING ENSEMBLE REAL LEADERBOARD RESULTS")
        for i, name in enumerate(models_full.keys()):
            b_score = r2_score(real_demand, oof_test[:, i]) * 100
            print(f"{name.upper()} Score: {b_score:.6f}")
            
        print(f"Final Ridge Stacker Score : {score_ens:.6f}")
        
        print("\nDynamic Ridge Metalearner Weights:")
        for name, coef in zip(models_full.keys(), ridge_full.coef_):
            print(f"  {name}: {coef:.4f}")
        print(f"  Intercept: {ridge_full.intercept_:.4f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()
