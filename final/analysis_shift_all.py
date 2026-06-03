"""
Cross-Day Shift Analysis (train.csv only)

Computes d49/d48 morning demand shift ratios for every categorical variable.
Used to identify which categories deviate from the global shift pattern,
indicating potential systematic over/under-prediction by the model.

Key finding: Street RoadType has shift=0.994 (flat) vs global=1.460,
making it the primary candidate for post-prediction correction.
"""
import pandas as pd
import numpy as np
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

# Morning subsets: d48 h0-2 vs d49 h0-2 (all d49 in train is morning)
d48_morning = train[(train.day == 48) & (train.hour <= 2)]
d49_morning = train[train.day == 49]

global_d48 = d48_morning.demand.mean()
global_d49 = d49_morning.demand.mean()
global_shift = global_d49 / global_d48

print(f"Global d49/d48 morning shift: {global_shift:.4f}")
print(f"  d48 morning mean: {global_d48:.5f}")
print(f"  d49 morning mean: {global_d49:.5f}")
print(f"\nCategories with shift BELOW global -> model OVERPREDICTS for them")
print(f"Categories with shift ABOVE global -> model UNDERPREDICTS for them")

# Analyze each categorical variable
categorical_vars = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
                    'NumberofLanes', 'geohash4', 'hour']

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
        shift = d49_m / d48_m if d48_m and d48_m > 1e-6 else np.nan
        dev = shift - global_shift if shift == shift else np.nan
        rows.append({'cat': cat, 'd48_mean': d48_m, 'd49_mean': d49_m,
                     'shift': shift, 'deviation': dev,
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
              f"(n={int(r['d48_n'])}+{int(r['d49_n'])}){flag}")
