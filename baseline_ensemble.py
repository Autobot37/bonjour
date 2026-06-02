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
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    return df

def main():
    print("Loading data...")
    train_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test_raw = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")

    print("Engineering Moonshot features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)

    target = "demand"
    features = [c for c in train.columns if c not in ["Index", target]]
    
    cat_cols = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 
                'Landmarks', 'Weather', 'geohash4_hour', 'Weather_RoadType', 'RoadType_Lanes']
    
    print("Encoding categorical features natively...")
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # ---------------------------------------------------------
    # INTERNAL TIME-BASED VALIDATION (DIRECT FIT BLENDING)
    # ---------------------------------------------------------
    print("\n=========================================")
    print("INTERNAL TIME-BASED VALIDATION (DIRECT FIT BLENDING)")
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
    
    val_preds_dict = {}
    for name, model in models.items():
        print(f"Training {name} on X_int_train...")
        model.fit(X_int_train, y_int_train)
        val_preds_dict[name] = model.predict(X_int_val)
        
    df_val_preds = pd.DataFrame(val_preds_dict)
    
    # Train Ridge metalearner on validation predictions (blending)
    ridge = Ridge(alpha=1.0)
    ridge.fit(df_val_preds, y_int_val)
    val_preds = ridge.predict(df_val_preds)
    
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

    test_preds_dict = {}
    models_full = {
        'lgb': lgb.LGBMRegressor(**Config.LGB_PARAMS),
        'cat': CatBoostRegressor(**Config.CAT_PARAMS, cat_features=cat_cols)
    }
    
    for name, model in models_full.items():
        print(f"Training {name} on full train data...")
        model.fit(X_train_full, y_train_full)
        test_preds_dict[name] = model.predict(X_test_real)
        
    df_test_preds = pd.DataFrame(test_preds_dict)
    preds_ens = ridge.predict(df_test_preds)
    
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
        for name in models_full.keys():
            b_score = r2_score(real_demand, df_test_preds[name]) * 100
            print(f"{name.upper()} Score: {b_score:.6f}")
            
        print(f"Final Ridge Stacker Score : {score_ens:.6f}")
        
        print("\nDynamic Ridge Metalearner Weights:")
        for name, coef in zip(models_full.keys(), ridge.coef_):
            print(f"  {name}: {coef:.4f}")
        print(f"  Intercept: {ridge.intercept_:.4f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()
