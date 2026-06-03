import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings("ignore")

def engineer_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    df['minute'] = df['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    df['geohash4_hour'] = df['geohash_4'] + "_" + df['hour'].astype(str)
    df['Weather_RoadType'] = df['Weather'].astype(str) + "_" + df['RoadType'].astype(str)
    df['RoadType_Lanes'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
    
    return df

def apply_hierarchical_te(tr, va, target, l1_col, l2_col, smoothing=10.0):
    global_m = tr[target].mean()
    
    stats_l2 = tr.groupby(l2_col)[target].agg(['mean', 'count'])
    te_l2 = (stats_l2['mean'] * stats_l2['count'] + global_m * smoothing) / (stats_l2['count'] + smoothing)
    
    stats_l1 = tr.groupby(l1_col)[target].agg(['mean', 'count'])
    
    l1_to_l2 = tr.groupby(l1_col)[l2_col].first()
    l1_prior = l1_to_l2.map(te_l2).fillna(global_m)
    
    te_l1 = (stats_l1['mean'] * stats_l1['count'] + l1_prior * smoothing) / (stats_l1['count'] + smoothing)
    
    res = va[l1_col].map(te_l1)
    res = res.fillna(0)
    return res

def kfold_hierarchical_te(train, test, target, mappings, n_splits=5, smoothing=10.0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_encoded = pd.DataFrame(index=train.index)
    for l1_col in mappings.keys():
        train_encoded[l1_col + '_te'] = np.nan
        
    for train_idx, val_idx in kf.split(train):
        tr = train.iloc[train_idx]
        va = train.iloc[val_idx]
        
        for l1_col, l2_col in mappings.items():
            train_encoded.loc[va.index, l1_col + '_te'] = apply_hierarchical_te(tr, va, target, l1_col, l2_col, smoothing)
            
    test_encoded = pd.DataFrame(index=test.index)
    for l1_col, l2_col in mappings.items():
        test_encoded[l1_col + '_te'] = apply_hierarchical_te(train, test, target, l1_col, l2_col, smoothing)
        
    return train_encoded, test_encoded

def main():
    print("Loading data...")
    train_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    print("Engineering base features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)
    
    print("Computing Morning Features (Leakage-Free)...")
    morning_mask = train['hour'] <= 2
    morning_df = train[morning_mask].copy()
    
    # Sort by time so 'last' gets the latest persistence value correctly
    morning_df = morning_df.sort_values(by=['day', 'geohash', 'hour', 'minute'])
    
    morning_stats = morning_df.groupby(['day', 'geohash']).agg(
        morning_mean=('demand', 'mean'),
        morning_std=('demand', 'std'),
        persistence=('demand', 'last')
    ).reset_index()
    
    train = train.merge(morning_stats, on=['day', 'geohash'], how='left')
    test = test.merge(morning_stats, on=['day', 'geohash'], how='left')
    
    for col in ['morning_mean', 'morning_std', 'persistence']:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)
        
    # Prevent direct leakage where persistence equals target
    train = train[train['hour'] > 2].copy()
    
    if 'timestamp' in train.columns:
        train.drop(columns=['timestamp'], inplace=True)
        test.drop(columns=['timestamp'], inplace=True)

    target = "demand"
    
    mappings = {
        'geohash4_hour': 'geohash_4',
        'Weather_RoadType': 'RoadType',
        'RoadType_Lanes': 'RoadType'
    }
    
    print("Performing Hierarchical Target Encoding (Leak-Free Out-Of-Fold)...")
    train_te, test_te = kfold_hierarchical_te(train, test, target, mappings)
    
    train = pd.concat([train, train_te], axis=1)
    test = pd.concat([test, test_te], axis=1)
    
    # Drop the string features
    train.drop(columns=list(mappings.keys()), inplace=True)
    test.drop(columns=list(mappings.keys()), inplace=True)

    features = [c for c in train.columns if c not in ["Index", target]]
    
    cat_cols = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    
    print("Encoding base categorical features...")
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # Internal Validation
    print("\n=========================================")
    print("INTERNAL TIME-BASED VALIDATION")
    print("=========================================")
    int_train_mask = (train['day'] == 48) & (train['hour'] <= 12)
    int_val_mask = (train['day'] == 48) & (train['hour'] > 12) & (train['hour'] < 14)
    
    X_int_train = train[int_train_mask][features]
    y_int_train = train[int_train_mask][target]
    X_int_val = train[int_val_mask][features]
    y_int_val = train[int_val_mask][target]
    
    lgb_params = {
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
    
    model_val = lgb.LGBMRegressor(**lgb_params)
    model_val.fit(X_int_train, y_int_train)
    val_preds = model_val.predict(X_int_val)
    val_score = r2_score(y_int_val, val_preds) * 100
    print(f"INTERNAL VALIDATION R2 SCORE: {val_score:.6f}")
    
    # Full Inference
    print("\n=========================================")
    print("FULL INFERENCE FOR REAL LEADERBOARD")
    print("=========================================")
    X_train_full = train[features]
    y_train_full = train[target]
    X_test_real = test[features]
    
    model_full = lgb.LGBMRegressor(**lgb_params)
    model_full.fit(X_train_full, y_train_full)
    preds = model_full.predict(X_test_real)
    
    sub = test[['Index']].copy()
    sub['demand'] = preds
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_te.csv", index=False)
    print("Saved baseline_te.csv to dataset/")

    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        score = r2_score(real_demand, preds) * 100
        print(f"\n=========================================")
        print(f"REAL LEADERBOARD RESULT")
        print(f"LightGBM + TE + MorningStats Score : {score:.6f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()
