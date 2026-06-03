import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import os

# =================================================================================
# CONFIGURATION & HYPERPARAMETERS
# =================================================================================
class Config:
    # --- Clustering Settings ---
    # Features used to group data into experts. Must be present in the dataset.
    CLUSTER_FEATURES = ['geohash_4', 'hour', 'Temperature']
    N_CLUSTERS = 4
    KMEANS_RANDOM_STATE = 42
    
    # --- Internal Validation Time Split ---
    # We predict the future based on the past to prevent target leakage.
    VAL_TRAIN_DAY = 48
    VAL_TRAIN_MAX_HOUR = 12
    VAL_TEST_DAY = 48
    VAL_TEST_HOUR_MIN = 12
    VAL_TEST_HOUR_MAX = 14
    
    # --- LightGBM Hyperparameters (Internal Validation) ---
    LGB_VAL_PARAMS = {
        'random_state': 42,
        'n_estimators': 100, 
        'learning_rate': 0.05,
        'num_leaves': 127,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # --- LightGBM Hyperparameters (Full Train for Inference) ---
    LGB_FULL_PARAMS = {
        'random_state': 42,
        'n_estimators': 300, # Higher since we train on 100% of the train data
        'learning_rate': 0.05,
        'num_leaves': 127,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1
    }

# =================================================================================

def load_and_engineer_data():
    """Loads CSVs, engineers base time/spatial features, and returns splits."""
    train = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\train.csv")
    test = pd.read_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\test.csv")
    
    # Time Features
    train['hour'] = train['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    train['minute'] = train['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    test['hour'] = test['timestamp'].astype(str).apply(lambda x: int(x.split(':')[0]) if ':' in x else -1)
    test['minute'] = test['timestamp'].astype(str).apply(lambda x: int(x.split(':')[1]) if ':' in x else -1)
    
    # Internal Validation Masks
    int_train_mask = (train['day'] == Config.VAL_TRAIN_DAY) & (train['hour'] <= Config.VAL_TRAIN_MAX_HOUR)
    int_val_mask = (train['day'] == Config.VAL_TEST_DAY) & (train['hour'] > Config.VAL_TEST_HOUR_MIN) & (train['hour'] < Config.VAL_TEST_HOUR_MAX)
    
    # Spatial Features
    train['geohash_4'] = train['geohash'].str[:4]
    train['geohash_5'] = train['geohash'].str[:5]
    test['geohash_4'] = test['geohash'].str[:4]
    test['geohash_5'] = test['geohash'].str[:5]

    # Clean up obsolete columns
    train.drop(columns=['timestamp'], inplace=True)
    test.drop(columns=['timestamp'], inplace=True)
    
    return train, test, int_train_mask, int_val_mask

def preprocess_categoricals(train, test, cat_cols):
    """Safely label-encodes categorical variables across train and test sets."""
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    return train, test

def main():
    print("Loading and engineering data...")
    train, test, int_train_mask, int_val_mask = load_and_engineer_data()

    target = "demand"
    features = [c for c in train.columns if c not in ["Index", target]]
    cat_cols = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

    print("Encoding categorical features...")
    train, test = preprocess_categoricals(train, test, cat_cols)

    # ---------------------------------------------------------
    # 1. SETUP DATASETS
    # ---------------------------------------------------------
    X_train_full = train[features]
    y_train_full = train[target]
    X_test_real = test[features]

    X_int_train = train[int_train_mask][features].copy()
    y_int_train = train[int_train_mask][target].copy()
    X_int_val = train[int_val_mask][features].copy()
    y_int_val = train[int_val_mask][target].copy()

    # ---------------------------------------------------------
    # 2. INTERNAL VALIDATION (Train/Val Split)
    # ---------------------------------------------------------
    print("\n=========================================")
    print("APPROACH A: K-MEANS MIXTURE OF EXPERTS (INTERNAL VALIDATION)")
    print(f"Clustering Features: {Config.CLUSTER_FEATURES}")
    print(f"Train samples: {len(X_int_train)}")
    print(f"Val samples  : {len(X_int_val)}")
    
    # Safely handle missing values in cluster features (e.g., Temperature)
    X_train_full_cf = X_train_full[Config.CLUSTER_FEATURES].fillna(0)
    X_int_train_cf = X_int_train[Config.CLUSTER_FEATURES].fillna(0)
    X_int_val_cf = X_int_val[Config.CLUSTER_FEATURES].fillna(0)
    
    scaler = StandardScaler()
    scaler.fit(X_train_full_cf) # Fit scaler globally to keep representations consistent
    
    kmeans = KMeans(n_clusters=Config.N_CLUSTERS, random_state=Config.KMEANS_RANDOM_STATE, n_init=10)
    
    X_int_train_scaled = scaler.transform(X_int_train_cf)
    kmeans.fit(X_int_train_scaled)
    train_clusters = kmeans.predict(X_int_train_scaled)
    
    X_int_val_scaled = scaler.transform(X_int_val_cf)
    val_clusters = kmeans.predict(X_int_val_scaled)
    
    val_models = {}
    for c in range(Config.N_CLUSTERS):
        mask = (train_clusters == c)
        if mask.sum() > 0:
            m = lgb.LGBMRegressor(**Config.LGB_VAL_PARAMS)
            m.fit(X_int_train[mask], y_int_train[mask])
            val_models[c] = m
            
    val_preds = np.zeros(len(X_int_val))
    for c in range(Config.N_CLUSTERS):
        mask = (val_clusters == c)
        if mask.sum() > 0 and c in val_models:
            val_preds[mask] = val_models[c].predict(X_int_val[mask])
            
    val_score = r2_score(y_int_val, val_preds) * 100
    print(f"INTERNAL VALIDATION R2 SCORE: {val_score:.6f}")
    print("=========================================\n")

    # ---------------------------------------------------------
    # 3. FULL LEADERBOARD TRAINING (Train on all data)
    # ---------------------------------------------------------
    print("Training Full K-Means Mixture Models for Real Inference...")
    X_test_real_cf = X_test_real[Config.CLUSTER_FEATURES].fillna(0)
    
    X_train_scaled = scaler.transform(X_train_full_cf)
    kmeans.fit(X_train_scaled)
    full_train_clusters = kmeans.predict(X_train_scaled)
    
    X_test_scaled = scaler.transform(X_test_real_cf)
    test_clusters = kmeans.predict(X_test_scaled)
    
    full_models = {}
    for c in range(Config.N_CLUSTERS):
        mask = (full_train_clusters == c)
        if mask.sum() > 0:
            m = lgb.LGBMRegressor(**Config.LGB_FULL_PARAMS)
            m.fit(X_train_full[mask], y_train_full[mask])
            full_models[c] = m
            
    test_preds = np.zeros(len(X_test_real))
    for c in range(Config.N_CLUSTERS):
        mask = (test_clusters == c)
        if mask.sum() > 0 and c in full_models:
            test_preds[mask] = full_models[c].predict(X_test_real[mask])
            
    sub = test[['Index']].copy()
    sub['demand'] = test_preds
    sub.to_csv(r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\baseline_kmeans_A.csv", index=False)
    
    # ---------------------------------------------------------
    # 4. LEADERBOARD EVALUATION
    # ---------------------------------------------------------
    real_test_path = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset\real_test.csv"
    if os.path.exists(real_test_path):
        df_real = pd.read_csv(real_test_path)
        real_demand = df_real["demand"].to_numpy(dtype=np.float64)
        score = r2_score(real_demand, test_preds) * 100
        print(f"REAL LEADERBOARD R2 SCORE (0-100): {score:.6f}")
        print("=========================================\n")

if __name__ == "__main__":
    main()
