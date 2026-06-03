# Gridlock 2.0 Experiment Log

A record of internal validation and real leaderboard (test) R² scores across different modeling approaches.

## Experiment Summary

| # | Model / Approach | Configuration / Key Features | Internal R² (%) | Leaderboard R² (%) |
|---|---|---|---|---|
| 1 | **LightGBM Baseline** | Default baseline | — | 90.191526 |
| 2 | **LightGBM + Regularization** | Same hours as test + regularizer | 89.375723 | 90.490212 |
| 3 | **LightGBM + Extra Features** | Geohash + timestamp + label encoding | 94.400000 | 90.930000 |
| 4 | **LightGBM + Target Encoding** | Target encoding on same features | 92.750000 | 90.530000 |
| 5 | **LightGBM + Categorical Tuning** | Revert to categorical + some tuning | 94.560000 | 91.020000 |
| 6 | **LightGBM + RevIN** | Tuned LightGBM + RevIN | 93.772538 | 89.444272 |
| 7 | **Approach A (K-Means MoE)** | Clustering on: `['geohash_4', 'hour', 'Temperature']` | 93.554542 | 91.114997 |
| 8 | **Approach A (K-Means MoE)** | Clustering on: `['geohash_4', 'hour']` (`ans_A.py`) | **94.405234** | 91.048802 |
| 9 | **Approach A (K-Means MoE)** | Clustering on: `['geohash_4', 'hour', 'NumberofLanes']` | 93.422939 | 90.354813 |
| 10 | **Fixed Equation (LB004)** | Non-ML formulation from `curr_best91.py` | 82.425673 | 87.360876 |
| 11 | **LightGBM + Adv Features & TE** | Advanced features + target encoding (`baseline_extrafeatures.py`) | 45.160533 | 89.498005 |
| 12 | **Ridge Stacking Ensemble (LGB+CAT)** | LGB, CAT stacked with Ridge, OOF KFold (`baseline_ensemble.py`) | 91.398836 | 91.288871 |
| 13 | **Ridge Blending (LGB+CAT)** | LGB, CAT stacked with Ridge, direct fit blending (`baseline_ensemble.py`) | 93.029782 | 87.969808 |
| 14 | **LightGBM (Single, Hour < 14 Filter)** | Single LGBM directly fit, train filtered to < 14 (`baseline_extrafeatures.py`) | 91.211799 | 90.795229 |
| 15 | **LightGBM (Single, Unfiltered)** | Single LGBM directly fit, all hours used (`baseline_extrafeatures.py`) | 91.211799 | 90.922008 |
| 16 | **LightGBM (Single, Decoded Geohash & Cyclical Time)** | Single LGBM directly fit, decoded geohash lat/lon + sin/cos time features (`baseline_extrafeatures.py`) | 94.549121 | 91.087517 |
| 17 | **LightGBM (Single, Extra Features + Neighbors)** | Single LightGBM with extra features and neighbor stats (`baseline_extrafeatures.py`) | 94.707473 | 91.273864 |
| 18 | **Ridge Stacking Ensemble (LGB+CAT) + Extra Features** | LGB, CAT stacked with Ridge, using extra features (`baseline_ensemble.py`) | 94.609539 | **91.458834** |
| 19 | **Ridge Stacking Ensemble (LGB+CAT) + Neighbors** | LGB, CAT stacked with Ridge, using neighbor features (`baseline_ensemble.py`) | 94.958240 | 91.297008 |


---

## Detailed Experiment Configurations

### 1. LightGBM Baselines & Tuning
- **Baseline + Regularization**: Applying regularizer and matching training hours to test hour window.
- **Engineered Features**: Added base geohash representation, timestamp minutes, and label encoding.
- **Target Encoding**: Attempted target encoding but observed a drop in leaderboard score (90.93 $\rightarrow$ 90.53) likely due to overfitting.
- **Categorical Handling**: Reverting back to native LightGBM categorical features with tuning improved the leaderboard score to **91.02%**.

### 2. K-Means Mixture of Experts (Approach A)
Dividing the data into $K=4$ clusters using spatial and temporal features, training an independent expert model per cluster.
- **With Temperature**: High validation (93.55%) and good leaderboard score (**91.11%**).
- **Without Temperature**: High internal validation (**94.41%**) but slightly lower on the leaderboard (**91.05%**).
- **With Number of Lanes**: Decreased performance both internally and on the leaderboard.

### 3. Non-ML Formulations (Fixed Equation)
- **LB004**: Computes target demand using a combination of morning persistence and denoised daily analog without any ML model. Good baseline baseline score of **87.36%** on the leaderboard.

### 4. Advanced Features + Target Encoding
- **Advanced Features & Target Encoding (`baseline_extrafeatures.py`)**: Included target encodings (geohash/hour, road type/hour), neighborhood stats, morning statistics, and 24h hourly lag features. Achieved an internal validation score of **45.16%** and a leaderboard score of **89.50%**.

### 5. Stacking Ensemble
Ensembling LightGBM and CatBoost regressors using a Ridge metalearner.
- **Ridge Stacker (LGB + CatBoost OOF) (`baseline_ensemble.py`)**: Achieved a leaderboard score of **91.29%**.
  - **Individual LB Scores**: CatBoost (**91.01%**), LGB (**90.92%**)
  - **Metalearner weights**: LGB: 0.6466, CatBoost: 0.3633 (Intercept: -0.0004)
- **Ridge Blending (LGB + CatBoost Direct Fit Blending)**: Leaderboard score dropped to **87.97%**.
  - **Why it hurt**: Without KFold out-of-fold (OOF) cross-validation, the Ridge meta-learner had to be trained on the validation set predictions (which only cover hours 13-14). Because it optimized weights for this narrow time window, it failed to generalize to the test set's broader temporal distribution.
- **Ridge Stacker with Extra Features (`baseline_ensemble.py`)**: Incorporated the advanced features from `baseline_extrafeatures.py` into the stacker. This yielded the current best leaderboard score of **91.46%**!
  - **Individual LB Scores**: LGB (**91.22%**), CatBoost (**91.18%**)
  - **Metalearner weights**: LGB: 0.6729, CatBoost: 0.3371 (Intercept: -0.0006)
- **Ridge Stacker with Extra Features & Neighbors (`baseline_ensemble.py`)**: Added rush-hour/night cyclical features and spatial neighborhood/area demand statistics to the stacker. Achieved an internal validation score of **94.96%** and a leaderboard score of **91.30%**.
  - **Individual LB Scores**: LGB (**91.27%**), CatBoost (**90.82%**)
  - **Metalearner weights**: LGB: 0.5632, CatBoost: 0.4454 (Intercept: -0.0007)

### 6. Single LightGBM (No Stacking/OOF)
- **Single LightGBM with Hour < 14 Filter (`baseline_extrafeatures.py`)**: Trained on training data filtered to hours < 14 (matching the test set temporal window). Achieved internal validation R² of **91.21%** and leaderboard score of **90.80%**.
- **Single LightGBM Unfiltered (`baseline_extrafeatures.py`)**: Trained on all hours. Achieved internal validation R² of **91.21%** and leaderboard score of **90.92%**.
  - **Key Observation**: Restricting the training data to only matching test hours (< 14) decreases the data size and actually decreases the leaderboard score (from 90.92% down to 90.80%). The model benefits more from the extra training data (hours 14–23) despite the temporal range mismatch.
- **Single LightGBM with Decoded Geohash & Cyclical Time (`baseline_extrafeatures.py`)**: Added geohash decoding (lat/lon coordinates), cyclical time representation (`sin_tmin`, `cos_tmin`), and other engineered indicators (`is_rush`, `is_night`, `temp_missing`, `lanes_x_large`). Achieved an internal validation score of **94.55%** and a leaderboard score of **91.09%**.
- **Single LightGBM with Extra Features (`baseline_extrafeatures.py`)**: Improved feature engineering and corrected column data types. Achieved an internal validation score of **94.62%** and a leaderboard score of **91.22%**.
- **Single LightGBM with Extra Features & Neighbors (`baseline_extrafeatures.py`)**: Added rush-hour/night cyclical features and spatial neighborhood/area demand statistics from `curr_best91.py`. Achieved an internal validation score of **94.71%** and a leaderboard score of **91.27%**.

---

## V2 Experiments (Temporal Validation Restructuring & Cross-Day Feature Engineering)

A series of experiments focusing on aligning the internal validation split with the real test set's temporal boundary (predicting day 49 using day 48 + day 49 morning), and developing leak-free cross-day features.

### Experiment Summary Table

| # | Model / Approach | Key Changes / Features | Split Config | Internal Val R² (%) | Leaderboard R² (%) |
|---|---|---|---|:---:|:---:|
| 1 | **LightGBM Baseline** | Default baseline | Original (d48 h<=12 / h13) | 94.5657 | 91.0237 |
| 2 | **LightGBM Baseline** | Default baseline | New (d48 full + d49 h0-1 / d49 h2) | 93.1822 | 91.0237 |
| 3 | **LightGBM Extra Features** | + Neighbor & Cyclical features | Original (d48 h<=12 / h13) | 94.7075 | 91.2739 |
| 4 | **LightGBM Extra Features** | + Neighbor & Cyclical features | New (d48 full + d49 h0-1 / d49 h2) | 93.4454 | 91.4713 |
| 5 | **Shift Ratio (Geohash)** | Stats + geohash-level morning shift ratio | New (d48 full + d49 h0-1 / d49 h2) | 90.6373 | 91.0580 |
| 6 | **Shift Ratio (Geohash5)** | Stats + geohash5-level morning shift ratio | New (d48 full + d49 h0-1 / d49 h2) | 92.7154 | 91.3223 |
| 7 | **Shift Ratio (Geohash4)** | Stats + geohash4-level morning shift ratio | New (d48 full + d49 h0-1 / d49 h2) | 93.1891 | 91.4661 |
| 8 | **Shift Ratio (Global)** | Stats + global morning shift ratio | New (d48 full + d49 h0-1 / d49 h2) | 93.6720 | 91.2845 |
| 9 | **Trajectory Features** | Stats + Day 48 hourly regional trajectory | New (d48 full + d49 h0-1 / d49 h2) | 93.5844 | 91.5812 |
| 10 | **Trajectory + Scale** | Trajectory + morning regional scale factor | New (d48 full + d49 h0-1 / d49 h2) | 93.9722 | 91.4822 |
| 11 | **Trajectory + RoadType TE** | Trajectory + RoadType x Hour TE | New (d48 full + d49 h0-1 / d49 h2) | 93.8777 | 91.5353 |
| 12 | **Sample Weighting (Optimal)** | Trajectory + Temporal Sample Weights | New (d48 full + d49 h0-1 / d49 h2) | **93.8539** | **91.7226** |

### Detailed Analysis of V2 Phase

1. **Validation Realignment**: Realigned the internal validation target to Day 49 hour 2, with training on Day 48 (all) and Day 49 (hours 0 and 1). This mimics the real test set structure and guarantees our validation metrics match leaderboard improvements.
2. **Leak-Free Cross-Day Features**: Implemented Day 48 same-hour statistics (`d48_same_hour_mean` / `std`) and Day 48 regional trajectory profiles (`d48_g5_hourly_mean_h0` to `h23`) mapped exclusively onto Day 49 rows. Day 48 training rows are masked with `NaN` to completely eliminate target leakage.
3. **Noisy Scaling Ratio Issues**: Tested various spatial aggregations for the morning shift ratio (`d49_morning / d48_morning`). The geohash-level ratio degraded performance to 91.05% due to high noise (only ~6 morning samples per geohash). Larger regional groups (like `geohash4`) and no scale factor perform more robustly.
4. **Optimal Sample Weighting**: Achieved our current best score (**91.7226%**) by combining trajectory features with sample weights (2x on Day 49 morning rows, 1.5x on Day 48 test window hours 2-13, 1.0 otherwise), focusing the LightGBM objective function directly on the target test window.