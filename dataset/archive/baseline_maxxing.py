import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import optuna
import os
import warnings
warnings.filterwarnings("ignore")

# Optuna configuration
N_TRIALS = 30

def engineer_features(df):
    df = df.copy()
    
    # Base Time Features
    df['hour'] = df['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    df['minute'] = df['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    
    # Base Spatial
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    # HIGH-ORDER INTERACTIONS
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

    print("Engineering advanced features...")
    train = engineer_features(train_raw)
    test = engineer_features(test_raw)
    
    # Validation Masks
    int_train_mask = (train['day'] == 48) & (train['hour'] <= 12)
    int_val_mask = (train['day'] == 48) & (train['hour'] > 12) & (train['hour'] < 14)

    target = "demand"
    features = [c for c in train.columns if c not in ["Index", target]]
    
    cat_cols = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 
                'Landmarks', 'Weather', 'geohash4_hour', 'Weather_RoadType', 'RoadType_Lanes']

    print("Encoding categorical features...")
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    # Splits
    X_train_full = train[features]
    y_train_full = train[target]
    X_test_real = test[features]

    X_int_train = train[int_train_mask][features].copy()
    y_int_train = train[int_train_mask][target].copy()
    X_int_val = train[int_val_mask][features].copy()
    y_int_val = train[int_val_mask][target].copy()

    print("\n=========================================")
    print("STARTING OPTUNA HYPERPARAMETER TUNING")
    print(f"Trials: {N_TRIALS}")
    print("=========================================\n")

    def objective(trial):
        # 1. Sample LightGBM Hyperparameters
        lgb_params = {
            'random_state': 42,
            'n_estimators': trial.suggest_int('lgb_n_estimators', 200, 600, step=100),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 63, 255),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('lgb_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_lambda', 0.1, 10.0, log=True),
            'n_jobs': -1,
            'verbose': -1
        }
        
        # 2. Sample CatBoost Hyperparameters
        cat_params = {
            'random_state': 42,
            'iterations': trial.suggest_int('cat_iterations', 300, 800, step=100),
            'learning_rate': trial.suggest_float('cat_lr', 0.02, 0.1, log=True),
            'depth': trial.suggest_int('cat_depth', 6, 10),
            'l2_leaf_reg': trial.suggest_float('cat_l2', 1, 10),
            'verbose': 0,
            'task_type': 'CPU'
        }
        
        # 3. Sample Ensemble Blend Weight
        lgb_weight = trial.suggest_float('lgb_weight', 0.4, 0.9)
        cat_weight = 1.0 - lgb_weight
        
        # Train & Predict LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_int_train, y_int_train)
        preds_lgb = m_lgb.predict(X_int_val)
        
        # Train & Predict CatBoost
        m_cat = CatBoostRegressor(**cat_params, cat_features=cat_cols)
        m_cat.fit(X_int_train, y_int_train)
        preds_cat = m_cat.predict(X_int_val)
        
        # Blend
        preds_ens = (preds_lgb * lgb_weight) + (preds_cat * cat_weight)
        
        score = r2_score(y_int_val, preds_ens) * 100
        return score

    # Run Optuna Study
    optuna.logging.set_verbosity(optuna.logging.INFO) # Show progress
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=========================================")
    print("OPTUNA TUNING COMPLETE")
    print(f"Best Internal Validation R2 Score: {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=========================================\n")

    # ---------------------------------------------------------
    # FULL LEADERBOARD TRAINING WITH BEST PARAMS
    # ---------------------------------------------------------
    print("Training Full LightGBM + CatBoost Ensembles with Best Params...")
    
    best = study.best_params
    
    final_lgb_params = {
        'random_state': 42,
        'n_estimators': best['lgb_n_estimators'] + 100, # More estimators for full train
        'learning_rate': best['lgb_lr'],
        'num_leaves': best['lgb_num_leaves'],
        'colsample_bytree': best['lgb_colsample'],
        'subsample': best['lgb_subsample'],
        'reg_alpha': best['lgb_alpha'],
        'reg_lambda': best['lgb_lambda'],
        'n_jobs': -1,
        'verbose': -1
    }
    
    final_cat_params = {
        'random_state': 42,
        'iterations': best['cat_iterations'] + 100, # More estimators for full train
        'learning_rate': best['cat_lr'],
        'depth': best['cat_depth'],
        'l2_leaf_reg': best['cat_l2'],
        'verbose': 0,
        'task_type': 'CPU'
    }
    
    final_lgb_weight = best['lgb_weight']
    final_cat_weight = 1.0 - final_lgb_weight
    
    m_lgb = lgb.LGBMRegressor(**final_lgb_params)
    m_lgb.fit(X_train_full, y_train_full)
    test_preds_lgb = m_lgb.predict(X_test_real)
    
    m_cat = CatBoostRegressor(**final_cat_params, cat_features=cat_cols)
    m_cat.fit(X_train_full, y_train_full)
    test_preds_cat = m_cat.predict(X_test_real)
    
    test_preds_ens = (test_preds_lgb * final_lgb_weight) + (test_preds_cat * final_cat_weight)
    
    sub = test[['Index']].copy()
    sub['demand'] = test_preds_ens
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_maxxing.csv", index=False)
    print("Saved baseline_maxxing.csv to dataset/")
    
    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        score = r2_score(real_demand, test_preds_ens) * 100
        print(f"\n=========================================")
        print(f"OPTUNA ENSEMBLE REAL LEADERBOARD R2 SCORE: {score:.6f}")
        print(f"=========================================\n")

if __name__ == "__main__":
    main()
