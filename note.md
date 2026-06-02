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
| 12 | **Ridge Stacking Ensemble (LGB+CAT)** | LGB, CAT stacked with Ridge, OOF KFold (`baseline_ensemble.py`) | 91.398836 | **91.288871** |
| 13 | **Ridge Blending (LGB+CAT)** | LGB, CAT stacked with Ridge, direct fit blending (`baseline_ensemble.py`) | 93.029782 | 87.969808 |

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
- **Ridge Stacker (LGB + CatBoost OOF) (`baseline_ensemble.py`)**: Achieved the highest leaderboard score so far (**91.29%**).
  - **Individual LB Scores**: CatBoost (**91.01%**), LGB (**90.92%**)
  - **Metalearner weights**: LGB: 0.6466, CatBoost: 0.3633 (Intercept: -0.0004)
- **Ridge Blending (LGB + CatBoost Direct Fit Blending)**: Leaderboard score dropped to **87.97%**.
  - **Why it hurt**: Without KFold out-of-fold (OOF) cross-validation, the Ridge meta-learner had to be trained on the validation set predictions (which only cover hours 13-14). Because it optimized weights for this narrow time window, it failed to generalize to the test set's broader temporal distribution.