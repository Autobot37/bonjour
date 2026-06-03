from numpy.testing import verbose
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import os
import numpy as np

def main():
    train = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    target = "demand"
    
    # Extract hour and minute before dropping timestamp
    train['hour'] = train['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    train['minute'] = train['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    test['hour'] = test['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    test['minute'] = test['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    
    # Define our internal split masks
    int_train_mask = (train['day'] == 48) & (train['hour'] <= 12)
    int_val_mask = (train['day'] == 48) & (train['hour'] > 12) & (train['hour'] < 14)
    
    # Extract geohash 4 and 5
    train['geohash_4'] = train['geohash'].str[:4]
    train['geohash_5'] = train['geohash'].str[:5]
    test['geohash_4'] = test['geohash'].str[:4]
    test['geohash_5'] = test['geohash'].str[:5]

    train.drop(columns=['timestamp'], inplace=True)
    test.drop(columns=['timestamp'], inplace=True)

    features = [c for c in train.columns if c not in ["Index", target]]
    
    cat_cols = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

    # Label encode and convert to category for LightGBM
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # Full Train / Test
    X_train_full = train[features]
    y_train_full = train[target]
    X_test_real = test[features]

    X_int_train = train[int_train_mask][features]
    y_int_train = train[int_train_mask][target]
    X_int_val = train[int_val_mask][features]
    y_int_val = train[int_val_mask][target]

    print("=========================================")
    print("INTERNAL TIME-BASED VALIDATION (Tuned LightGBM)")
    print(f"Train samples: {len(X_int_train)}")
    print(f"Val samples  : {len(X_int_val)}")
    
    # Tuned hyperparameters
    params = {
        'random_state': 42,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model_val = lgb.LGBMRegressor(**params)
    model_val.fit(X_int_train, y_int_train)
    val_preds = model_val.predict(X_int_val)
    val_score = r2_score(y_int_val, val_preds) * 100
    print(f"INTERNAL VALIDATION R2 SCORE: {val_score:.6f}")
    print("=========================================\n")

    print("Training full LightGBM baseline for real test...")
    # Add more estimators for the full dataset since it has more rows
    params['n_estimators'] = 500
    model_full = lgb.LGBMRegressor(**params)
    model_full.fit(X_train_full, y_train_full)

    print("Predicting...")
    preds = model_full.predict(X_test_real)

    sub = test[['Index']].copy()
    sub['demand'] = preds
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_lgb.csv", index=False)
    print("Saved baseline_lgb.csv to dataset/")

    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        pred_demand = pd.to_numeric(pd.Series(preds), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        score = r2_score(real_demand, pred_demand) * 100
        print(f"\n=========================================")
        print(f"REAL LEADERBOARD R2 SCORE (0-100): {score:.6f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()