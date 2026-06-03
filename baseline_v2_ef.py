import pandas as pd
import numpy as np
import lightgbm as lgb
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
    # Neighbor stats per geohash5
    g5_stats = reference_df.groupby('geohash5', observed=True)[target_col].agg(
        neighbor_mean='mean',
        neighbor_std='std',
        neighbor_count='count',
    ).fillna(0)
    
    # Area stats per geohash4
    g4_stats = reference_df.groupby('geohash4', observed=True)[target_col].agg(
        area_mean='mean',
        area_std='std',
    ).fillna(0)
    
    # Local mean per geohash
    local_mean = reference_df.groupby('geohash', observed=True)[target_col].mean().fillna(0)
    t = target_df.copy()
    # Use map and explicitly cast to float to avoid Categorical dtype mapping behaviors
    t['neighbor_mean'] = t['geohash5'].map(g5_stats['neighbor_mean']).astype(float)
    t['neighbor_std']  = t['geohash5'].map(g5_stats['neighbor_std']).astype(float)
    t['neighbor_count'] = t['geohash5'].map(g5_stats['neighbor_count']).astype(float)
    
    t['area_mean'] = t['geohash4'].map(g4_stats['area_mean']).astype(float)
    t['area_std']  = t['geohash4'].map(g4_stats['area_std']).astype(float)
    
    local_mean_mapped = t['geohash'].map(local_mean).astype(float)
    
    # Fill NAs with mean / defaults
    t['neighbor_mean'] = t['neighbor_mean'].fillna(g5_stats['neighbor_mean'].mean())
    t['neighbor_std']  = t['neighbor_std'].fillna(0)
    t['neighbor_count'] = t['neighbor_count'].fillna(0)
    
    t['area_mean'] = t['area_mean'].fillna(g4_stats['area_mean'].mean())
    t['area_std']  = t['area_std'].fillna(0)
    
    local_mean_mapped = local_mean_mapped.fillna(local_mean.mean())
    
    t['local_vs_neighbor'] = local_mean_mapped / (t['neighbor_mean'] + 1e-6)
    
    return t


def main():
    print("Loading data...")
    train_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    print("Engineering Moonshot features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)

    target = "demand"
    
    # Compute Day 48 stats on raw geohashes BEFORE they are LabelEncoded
    d48 = train[train['day'] == 48]
    d48_geo_hour_mean = d48.groupby(['geohash', 'hour'])['demand'].mean()
    d48_geo_hour_std = d48.groupby(['geohash', 'hour'])['demand'].std()
    
    # Map to train (only for day 49)
    train['d48_same_hour_mean'] = np.nan
    train['d48_same_hour_std'] = np.nan
    
    day49_train_idx = train['day'] == 49
    train.loc[day49_train_idx, 'd48_same_hour_mean'] = train[day49_train_idx].set_index(['geohash', 'hour']).index.map(d48_geo_hour_mean)
    train.loc[day49_train_idx, 'd48_same_hour_std'] = train[day49_train_idx].set_index(['geohash', 'hour']).index.map(d48_geo_hour_std)
    
    # Map to test
    test['d48_same_hour_mean'] = test.set_index(['geohash', 'hour']).index.map(d48_geo_hour_mean)
    test['d48_same_hour_std'] = test.set_index(['geohash', 'hour']).index.map(d48_geo_hour_std)

    # Compute d48 demand trajectory per geohash5 at each hour
    g5_hourly = d48.groupby(['geohash5', 'hour'])['demand'].mean().unstack(fill_value=0)
    g5_hourly.columns = [f'd48_g5_hourly_mean_h{h}' for h in g5_hourly.columns]
    
    # Merge and mask for train (leak-free: set d48 rows to NaN)
    train = train.merge(g5_hourly, on='geohash5', how='left')
    train.loc[train['day'] == 48, g5_hourly.columns] = np.nan
    
    # Merge for test
    test = test.merge(g5_hourly, on='geohash5', how='left')
    
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

    # ---------------------------------------------------------
    # INTERNAL TIME-BASED VALIDATION
    # ---------------------------------------------------------
    print("\n=========================================")
    print("INTERNAL TIME-BASED VALIDATION")
    print("=========================================")
    int_train_mask = (train['day'] == 48) | ((train['day'] == 49) & (train['hour'] <= 1))
    int_val_mask = (train['day'] == 49) & (train['hour'] == 2)
    
    # Compute neighbor features on validation training set and merge
    train_val_ref = train[int_train_mask].copy()
    train_val_train = add_neighbor_features(train_val_ref, train_val_ref)
    train_val_val = add_neighbor_features(train[int_val_mask].copy(), train_val_ref)
    
    features = [c for c in train_val_train.columns if c not in ["Index", target]]
    
    X_int_train = train_val_train[features].copy()
    y_int_train = train_val_train[target].copy()
    X_int_val = train_val_val[features].copy()
    y_int_val = train_val_val[target].copy()
    
    model = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    print("Training LightGBM for Internal Validation...")
    sample_weight_val = np.ones(len(train_val_train))
    sample_weight_val[train_val_train['day'] == 49] = 2.0
    d48_test_window_val = (train_val_train['day'] == 48) & (train_val_train['hour'] >= 2) & (train_val_train['hour'] <= 13)
    sample_weight_val[d48_test_window_val] = 1.5
    model.fit(X_int_train, y_int_train, sample_weight=sample_weight_val)
    val_preds = model.predict(X_int_val)
    
    val_score = r2_score(y_int_val, val_preds) * 100
    print(f"INTERNAL VALIDATION R2 SCORE: {val_score:.6f}")

    # ---------------------------------------------------------
    # FULL INFERENCE FOR REAL LEADERBOARD
    # ---------------------------------------------------------
    print("\n=========================================")
    print("FULL INFERENCE FOR REAL LEADERBOARD")
    print("=========================================")
    # Compute neighbor features on full training set and merge
    train_filtered = train
    test_filtered = test
    
    train_full_feat = add_neighbor_features(train_filtered, train_filtered)
    test_full_feat = add_neighbor_features(test_filtered, train_filtered)
    
    X_train_full = train_full_feat[features]
    y_train_full = train_full_feat[target]
    X_test_real = test_full_feat[features]
    
    model_full = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    print("Training LightGBM for Full Inference...")
    sample_weight_full = np.ones(len(train_full_feat))
    sample_weight_full[train_full_feat['day'] == 49] = 2.0
    d48_test_window_full = (train_full_feat['day'] == 48) & (train_full_feat['hour'] >= 2) & (train_full_feat['hour'] <= 13)
    sample_weight_full[d48_test_window_full] = 1.5
    model_full.fit(X_train_full, y_train_full, sample_weight=sample_weight_full)
    test_preds = model_full.predict(X_test_real)
    
    sub = test_filtered[['Index']].copy()
    sub['demand'] = test_preds
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_stack.csv", index=False)
    print("Saved baseline_stack.csv to dataset/")

    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        
        score = r2_score(real_demand, test_preds) * 100
        print(f"\n=========================================")
        print(f"LIGHTGBM REAL LEADERBOARD Score: {score:.6f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()
