"""
Systematic analysis: For EVERY categorical variable, compute the
d49/d48 morning shift ratio per category vs the global shift.

Categories where shift_ratio deviates from global = model will
over/under-predict for that category on d49.

All from train.csv only.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv(r"dataset\train.csv")
train['hour'] = train['timestamp'].str.split(':').str[0].astype(int)
train['geohash4'] = train['geohash'].str[:4]
train['geohash5'] = train['geohash'].str[:5]
train['RoadType'] = train['RoadType'].fillna('Missing').astype(str)
train['Weather'] = train['Weather'].fillna('Missing').astype(str)
train['LargeVehicles'] = train['LargeVehicles'].fillna('Missing').astype(str)
train['Landmarks'] = train['Landmarks'].fillna('Missing').astype(str)
train['NumberofLanes'] = pd.to_numeric(train['NumberofLanes'], errors='coerce').fillna(0).astype(int).astype(str)

d48 = train[train.day == 48]
d49 = train[train.day == 49]
d48_morning = d48[d48.hour <= 2]
d49_morning = d49  # d49 only has h0-2 in train

global_d48 = d48_morning.demand.mean()
global_d49 = d49_morning.demand.mean()
global_shift = global_d49 / global_d48
print(f"Global d49/d48 morning shift: {global_shift:.4f}")
print(f"  d48 morning mean: {global_d48:.5f}")
print(f"  d49 morning mean: {global_d49:.5f}")
print(f"\nCategories with shift BELOW global ({global_shift:.3f}) = model OVERPREDICTS for them")
print(f"Categories with shift ABOVE global ({global_shift:.3f}) = model UNDERPREDICTS for them")

categorical_vars = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 
                    'NumberofLanes', 'geohash4', 'geohash5', 'hour']

print("\n" + "="*80)
for var in categorical_vars:
    print(f"\n--- {var} ---")
    d48_g = d48_morning.groupby(var).demand.agg(['mean', 'count'])
    d49_g = d49_morning.groupby(var).demand.agg(['mean', 'count'])
    
    all_cats = d48_g.index.union(d49_g.index)
    rows = []
    for cat in all_cats:
        d48_m = d48_g.loc[cat, 'mean'] if cat in d48_g.index else np.nan
        d49_m = d49_g.loc[cat, 'mean'] if cat in d49_g.index else np.nan
        d48_n = d48_g.loc[cat, 'count'] if cat in d48_g.index else 0
        d49_n = d49_g.loc[cat, 'count'] if cat in d49_g.index else 0
        
        if pd.isna(d48_m) or pd.isna(d49_m) or d48_m < 1e-6:
            shift = np.nan
        else:
            shift = d49_m / d48_m
        
        deviation = shift - global_shift if not np.nan else np.nan
        rows.append({'cat': cat, 'd48_mean': d48_m, 'd49_mean': d49_m,
                     'shift': shift, 'deviation': shift - global_shift if shift == shift else np.nan,
                     'd48_n': d48_n, 'd49_n': d49_n})
    
    df_r = pd.DataFrame(rows).sort_values('deviation')
    for _, r in df_r.iterrows():
        if pd.isna(r['shift']):
            flag = "  [NO DATA]"
        elif abs(r['deviation']) > 0.2:
            flag = "  <-- LARGE DEVIATION"
        elif abs(r['deviation']) > 0.1:
            flag = "  <- moderate"
        else:
            flag = ""
        print(f"  {str(r['cat']):15s}: d48={r['d48_mean']:.4f}  d49={r['d49_mean']:.4f}  "
              f"shift={r['shift']:.3f}  dev_from_global={r['deviation']:+.3f}  "
              f"(n_d48={int(r['d48_n'])}, n_d49={int(r['d49_n'])}){flag}")
