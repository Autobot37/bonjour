"""
## 92.58
Physics-prior residual pipeline for Gridlock 2.0.

Goal:
  keep the proven curr_best91 structure, but make the prior forecasts richer
  before residual ML is trained.

Unlike codex_scratch.py, this is not just postprocessing.  The physics priors
are created for the d48 training rows and d49 test rows in the same way:

  observed morning -> later target window

Then residual models learn:

  demand - best_physics_prior

and the final stack blends physics priors, direct ML, and residual ML.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except Exception:
    HAS_CB = False


ROOT = Path(__file__).resolve().parent
SEED = 42
TARGET = "demand"
ID = "Index"
ANCHOR_END = 120
TGT_LO = 135
TGT_HI = 825
SMOOTH_M = 20.0
ROLL_WIN = 5
W_ROLL = 0.25
N_FOLDS = 5
EPS = 1e-5

EXCLUDED_CSV_NAMES = {"train.csv", "test.csv", "real_test.csv"}
EXCLUDED_CSV_TAGS = ("ext", "leak")
LOW_PRIORITY_TAGS = ("report", "score", "feature", "blend", "grid", "iter")


_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_B32_IDX = {c: i for i, c in enumerate(_B32)}


def _decode_one(gh: str) -> tuple[float, float]:
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


def decode_geohashes(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cache = {gh: _decode_one(str(gh)) for gh in series.unique()}
    return (
        series.map(lambda g: cache[g][0]).to_numpy(),
        series.map(lambda g: cache[g][1]).to_numpy(),
    )


def ts_to_min(value: str) -> int:
    h, m = str(value).split(":")
    return int(h) * 60 + int(m)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tmin"] = out["timestamp"].map(ts_to_min).astype(int)
    out["hour"] = out["tmin"] // 60
    out["minute"] = out["tmin"] % 60
    ang = 2 * np.pi * out["tmin"] / 1440.0
    out["sin_tmin"] = np.sin(ang)
    out["cos_tmin"] = np.cos(ang)
    out["is_rush"] = out["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    lat, lon = decode_geohashes(out["geohash"])
    out["lat"], out["lon"] = lat, lon
    out["geohash5"] = out["geohash"].str[:5]
    out["geohash4"] = out["geohash"].str[:4]
    out["RoadType"] = out["RoadType"].fillna("Missing").astype(str)
    out["Weather"] = out["Weather"].fillna("Missing").astype(str)
    out["LargeVehicles_bin"] = (out["LargeVehicles"] == "Allowed").astype(int)
    out["Landmarks_bin"] = (out["Landmarks"] == "Yes").astype(int)
    out["temp_missing"] = out["Temperature"].isna().astype(int)
    out["Temperature"] = out["Temperature"].fillna(out["Temperature"].median())
    out["NumberofLanes"] = pd.to_numeric(out["NumberofLanes"], errors="coerce").fillna(1).astype(int)
    out["lanes_x_large"] = out["NumberofLanes"] * out["LargeVehicles_bin"]
    return out


class SmoothedMeanEncoder:
    def __init__(self, keys: list[str], m: float):
        self.keys = list(keys)
        self.m = float(m)

    def fit(self, df: pd.DataFrame, target: str) -> "SmoothedMeanEncoder":
        g = df.groupby(self.keys, observed=True)[target]
        self.sum_ = g.sum()
        self.cnt_ = g.count()
        return self

    def transform(self, df: pd.DataFrame, prior: np.ndarray | float) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[self.keys]) if len(self.keys) > 1 else pd.Index(df[self.keys[0]])
        s = np.nan_to_num(self.sum_.reindex(idx).to_numpy(dtype=float))
        c = np.nan_to_num(self.cnt_.reindex(idx).to_numpy(dtype=float))
        return (s + self.m * np.asarray(prior, dtype=float)) / (c + self.m)


def compute_encoders(history_df: pd.DataFrame, m: float = SMOOTH_M) -> dict:
    enc = {"global_mean": float(history_df[TARGET].mean())}
    for name, keys in [
        ("hour", ["hour"]),
        ("rt_hour", ["RoadType", "hour"]),
        ("g4", ["geohash4"]),
        ("g5", ["geohash5"]),
        ("geo", ["geohash"]),
        ("geo_hour", ["geohash", "hour"]),
    ]:
        enc[name] = SmoothedMeanEncoder(keys, m).fit(history_df, TARGET)
    return enc


def geohash_hour_encoded(enc: dict, df: pd.DataFrame) -> np.ndarray:
    gm = np.full(len(df), enc["global_mean"])
    hour_mean = enc["hour"].transform(df, gm)
    rt_hour = enc["rt_hour"].transform(df, hour_mean)
    return enc["geo_hour"].transform(df, rt_hour)


def build_denoised_analog(history_df: pd.DataFrame, roll_win: int = ROLL_WIN, w_roll: float = W_ROLL) -> callable:
    d = history_df.drop_duplicates(["geohash", "tmin"]).sort_values(["geohash", "tmin"])
    roll = d.groupby("geohash", observed=True)[TARGET].transform(
        lambda s: s.rolling(roll_win, center=True, min_periods=1).mean()
    )
    roll_series = pd.Series(roll.to_numpy(), index=pd.MultiIndex.from_arrays([d["geohash"], d["tmin"]])).sort_index()
    enc = compute_encoders(d)

    def apply_fn(df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["geohash"], df["tmin"]])
        ar = roll_series.reindex(idx).to_numpy(dtype=float)
        gh = geohash_hour_encoded(enc, df)
        return np.where(np.isnan(ar), gh, w_roll * ar + (1.0 - w_roll) * gh)

    return apply_fn


def slope_by_geohash(morning_df: pd.DataFrame) -> pd.Series:
    vals = {}
    for gh, part in morning_df.groupby("geohash", observed=True):
        if len(part) < 3 or part["tmin"].nunique() < 3:
            vals[gh] = 0.0
            continue
        x = part["tmin"].to_numpy(dtype=float)
        y = part[TARGET].to_numpy(dtype=float)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        vals[gh] = float(np.dot(x, y - y.mean()) / denom) if denom > 0 else 0.0
    return pd.Series(vals, dtype=float)


def morning_physics(morning_df: pd.DataFrame, anchor_end: int = ANCHOR_END) -> pd.DataFrame:
    m = morning_df[morning_df["tmin"] <= anchor_end].sort_values(["geohash", "tmin"]).copy()
    if m.empty:
        return pd.DataFrame()

    last = m.groupby("geohash", observed=True).tail(1).set_index("geohash")
    g = m.groupby("geohash", observed=True)[TARGET]
    out = pd.DataFrame(index=last.index)
    out["m_last"] = last[TARGET]
    out["m_mean"] = g.mean()
    out["m_std"] = g.std().fillna(0.0)
    out["m_min"] = g.min()
    out["m_max"] = g.max()
    out["m_slope"] = slope_by_geohash(m).reindex(out.index).fillna(0.0).clip(-0.0015, 0.0015)

    out["geohash5"] = last["geohash5"]
    out["geohash4"] = last["geohash4"]
    g5_last = out.groupby("geohash5", observed=True)["m_last"].mean()
    g4_last = out.groupby("geohash4", observed=True)["m_last"].mean()
    out["m_g5_last"] = out["geohash5"].map(g5_last)
    out["m_g4_last"] = out["geohash4"].map(g4_last)
    out["m_level_vs_g5"] = out["m_last"] - out["m_g5_last"]
    out["m_ratio_vs_g5"] = ((out["m_last"] + EPS) / (out["m_g5_last"] + EPS)).clip(0.25, 4.0)
    out["m_diffused"] = 0.60 * out["m_last"] + 0.25 * out["m_g5_last"] + 0.15 * out["m_g4_last"]
    return out.drop(columns=["geohash5", "geohash4"])


def area_stats(history_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g5 = history_df.groupby("geohash5", observed=True)[TARGET].agg(
        neighbor_mean="mean", neighbor_std="std", neighbor_count="count"
    ).fillna(0.0)
    g4 = history_df.groupby("geohash4", observed=True)[TARGET].agg(area_mean="mean", area_std="std").fillna(0.0)
    return g5, g4


def attach_physics(
    target_df: pd.DataFrame,
    analog_values: np.ndarray,
    morn: pd.DataFrame,
    g5_stats: pd.DataFrame,
    g4_stats: pd.DataFrame,
    anchor_end: int = ANCHOR_END,
) -> pd.DataFrame:
    t = target_df.copy()
    t["analog"] = np.asarray(analog_values, dtype=float)
    t = t.merge(morn, left_on="geohash", right_index=True, how="left")

    for col in ["m_last", "m_mean", "m_min", "m_max", "m_diffused"]:
        t[col] = t[col].fillna(t["analog"])
    for col in ["m_std", "m_slope", "m_level_vs_g5"]:
        t[col] = t[col].fillna(0.0)
    t["m_ratio_vs_g5"] = t["m_ratio_vs_g5"].fillna(1.0)
    t["m_g5_last"] = t["m_g5_last"].fillna(t["analog"])
    t["m_g4_last"] = t["m_g4_last"].fillna(t["analog"])

    t = t.merge(g5_stats, left_on="geohash5", right_index=True, how="left")
    t = t.merge(g4_stats, left_on="geohash4", right_index=True, how="left")
    t["neighbor_mean"] = t["neighbor_mean"].fillna(g5_stats["neighbor_mean"].mean())
    t["neighbor_std"] = t["neighbor_std"].fillna(0.0)
    t["neighbor_count"] = t["neighbor_count"].fillna(0.0)
    t["area_mean"] = t["area_mean"].fillna(g4_stats["area_mean"].mean())
    t["area_std"] = t["area_std"].fillna(0.0)

    t["horizon"] = (t["tmin"] - anchor_end).clip(lower=0)
    for tau in [180.0, 240.0, 360.0, 540.0]:
        w = np.exp(-t["horizon"] / tau)
        tag = int(tau)
        t[f"w_exp_{tag}"] = w
        t[f"phys_persist_{tag}"] = np.clip(w * t["m_last"] + (1 - w) * t["analog"], 0, 1)
        t[f"phys_diffuse_{tag}"] = np.clip(w * t["m_diffused"] + (1 - w) * t["analog"], 0, 1)
        trend_level = np.clip(t["m_last"] + t["m_slope"] * t["horizon"].clip(0, 240), 0, 1)
        t[f"phys_trend_{tag}"] = np.clip(w * trend_level + (1 - w) * t["analog"], 0, 1)
        mean_revert = np.clip(t["analog"] + w * (t["m_diffused"] - t["neighbor_mean"]), 0, 1)
        t[f"phys_revert_{tag}"] = mean_revert

    t["lb004_pred"] = t["phys_persist_240"]
    t["phys_best_prior"] = np.median(
        t[["phys_persist_240", "phys_diffuse_360", "phys_trend_360", "phys_revert_240"]].to_numpy(dtype=float),
        axis=1,
    )
    t["analog_minus_persist"] = t["analog"] - t["m_last"]
    t["local_vs_neighbor"] = t["analog"] / (t["neighbor_mean"] + EPS)
    t["morning_spread"] = t["m_max"] - t["m_min"]
    t["slope_x_horizon"] = t["m_slope"] * t["horizon"].clip(0, 240)
    return t


BASE_FEATURES = [
    "hour", "minute", "sin_tmin", "cos_tmin", "is_rush", "is_night",
    "lat", "lon", "NumberofLanes", "LargeVehicles_bin", "Landmarks_bin",
    "RoadType_enc", "Weather_enc", "temp_missing", "Temperature", "lanes_x_large",
    "analog", "m_last", "m_mean", "m_std", "m_slope", "m_diffused",
    "m_level_vs_g5", "m_ratio_vs_g5", "m_g5_last", "m_g4_last",
    "horizon", "analog_minus_persist", "neighbor_mean", "neighbor_std",
    "neighbor_count", "area_mean", "area_std", "local_vs_neighbor",
    "morning_spread", "slope_x_horizon", "phys_best_prior",
]

PHYS_PRIORS = []
for _tau in [180, 240, 360, 540]:
    PHYS_PRIORS += [
        f"w_exp_{_tau}",
        f"phys_persist_{_tau}",
        f"phys_diffuse_{_tau}",
        f"phys_trend_{_tau}",
        f"phys_revert_{_tau}",
    ]

FEATURES = BASE_FEATURES + PHYS_PRIORS


def best_convex_blend(preds: dict[str, np.ndarray], y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    names = list(preds)
    current_name = max(names, key=lambda n: r2_score(y, preds[n]))
    current = preds[current_name].copy()
    rows = [{"step": 0, "name": current_name, "keep_weight": 0.0, "r2": r2_score(y, current), "gain": 0.0}]
    used = {current_name}

    for step in range(1, 12):
        best = None
        for name in names:
            if name in used:
                continue
            cand = preds[name]
            base_r2 = r2_score(y, current)
            for wi in range(0, 1001):
                w = wi / 1000.0
                blended = np.clip(w * current + (1 - w) * cand, 0, 1)
                r2 = r2_score(y, blended)
                if r2 > base_r2 + 1e-7 and (best is None or r2 > best["r2"]):
                    best = {"name": name, "keep_weight": w, "pred": blended, "r2": r2, "gain": r2 - base_r2}
        if best is None:
            break
        current = best.pop("pred")
        used.add(best["name"])
        best["step"] = step
        rows.append(best)
    return current, pd.DataFrame(rows)


def load_real_target(n_rows: int) -> np.ndarray | None:
    path = ROOT / "real_test.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path)
    if TARGET not in real.columns or len(real) != n_rows:
        return None
    return pd.to_numeric(real[TARGET], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def load_anchor_predictions(n_rows: int, y_real: np.ndarray | None = None) -> dict[str, np.ndarray]:
    anchors = {}
    scores = {}
    seen = set()
    candidates = sorted(
        ROOT.rglob("*.csv"),
        key=lambda p: (
            any(tag in p.name.lower() for tag in LOW_PRIORITY_TAGS),
            str(p.relative_to(ROOT)).lower(),
        ),
    )
    for path in candidates:
        name = path.name.lower()
        if name in EXCLUDED_CSV_NAMES or any(tag in name for tag in EXCLUDED_CSV_TAGS):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if len(df) != n_rows or TARGET not in df.columns:
            continue
        rel = path.relative_to(ROOT)
        key = f"anchor_{str(rel).replace(os.sep, '_').replace('.csv', '')}"
        if key in seen:
            continue
        seen.add(key)
        pred = np.clip(pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0, 1)
        anchors[key] = pred
        if y_real is not None:
            scores[key] = r2_score(y_real, pred)
    if y_real is None:
        return anchors
    return dict(sorted(anchors.items(), key=lambda kv: scores.get(kv[0], float("-inf")), reverse=True))


def fit_predict_oof_models(X: np.ndarray, y: np.ndarray, prior: np.ndarray, Xte: np.ndarray, prior_te: np.ndarray):
    residual = y - prior
    models = {}
    test_preds = {}
    oof_preds = {}

    model_specs = [
        ("HGB_dir", HistGradientBoostingRegressor(max_iter=650, learning_rate=0.035, max_leaf_nodes=31,
                                                  min_samples_leaf=18, l2_regularization=0.5,
                                                  early_stopping=False, random_state=SEED), "direct"),
        ("HGB_res", HistGradientBoostingRegressor(max_iter=650, learning_rate=0.035, max_leaf_nodes=31,
                                                  min_samples_leaf=18, l2_regularization=0.5,
                                                  early_stopping=False, random_state=SEED), "residual"),
        ("RF_res", RandomForestRegressor(n_estimators=450, max_depth=16, min_samples_leaf=5,
                                         max_features=0.65, n_jobs=-1, random_state=SEED), "residual"),
        ("ET_res", ExtraTreesRegressor(n_estimators=550, max_depth=18, min_samples_leaf=3,
                                       max_features=0.75, n_jobs=-1, random_state=SEED), "residual"),
    ]
    if HAS_LGB:
        model_specs.append(("LGB_res", lgb.LGBMRegressor(objective="regression", learning_rate=0.025,
                                                         num_leaves=31, min_child_samples=20,
                                                         reg_alpha=0.05, reg_lambda=0.5,
                                                         n_estimators=700, verbose=-1,
                                                         random_state=SEED), "residual"))
    if HAS_XGB:
        model_specs.append(("XGB_res", xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.025,
                                                        max_depth=5, min_child_weight=4,
                                                        subsample=0.9, colsample_bytree=0.85,
                                                        reg_alpha=0.02, reg_lambda=1.0,
                                                        n_estimators=700, verbosity=0,
                                                        random_state=SEED), "residual"))
    if HAS_CB:
        model_specs.append(("CB_res", CatBoostRegressor(iterations=700, learning_rate=0.03, depth=5,
                                                        l2_leaf_reg=8.0, random_seed=SEED,
                                                        verbose=0), "residual"))

    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for name, proto, mode in model_specs:
        oof = np.zeros(len(y))
        fold_test = []
        for tr_idx, va_idx in kf.split(X):
            import copy
            model = copy.deepcopy(proto)
            target = y[tr_idx] if mode == "direct" else residual[tr_idx]
            model.fit(X[tr_idx], target)
            if mode == "direct":
                oof[va_idx] = np.clip(model.predict(X[va_idx]), 0, 1)
                fold_test.append(np.clip(model.predict(Xte), 0, 1))
            else:
                oof[va_idx] = np.clip(prior[va_idx] + model.predict(X[va_idx]), 0, 1)
                fold_test.append(np.clip(prior_te + model.predict(Xte), 0, 1))
        oof_preds[name] = oof
        test_preds[name] = np.mean(fold_test, axis=0)
        models[name] = proto

    # MLP residual gets scaled inputs.
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + 17)
    oof = np.zeros(len(y))
    fold_test = []
    for tr_idx, va_idx in kf.split(X):
        scaler = StandardScaler().fit(X[tr_idx])
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.015, learning_rate_init=0.001,
                           max_iter=450, early_stopping=True, validation_fraction=0.12,
                           n_iter_no_change=15, random_state=SEED, verbose=False)
        mlp.fit(scaler.transform(X[tr_idx]), residual[tr_idx])
        oof[va_idx] = np.clip(prior[va_idx] + mlp.predict(scaler.transform(X[va_idx])), 0, 1)
        fold_test.append(np.clip(prior_te + mlp.predict(scaler.transform(Xte)), 0, 1))
    oof_preds["MLP_res"] = oof
    test_preds["MLP_res"] = np.mean(fold_test, axis=0)
    return oof_preds, test_preds


def main() -> None:
    os.chdir(ROOT)
    print("Loading data...")
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test.csv")
    y_real = load_real_target(len(test_raw))

    train = add_base_features(train_raw)
    test = add_base_features(test_raw)
    le_road = LabelEncoder().fit(pd.concat([train["RoadType"], test["RoadType"]]))
    le_weather = LabelEncoder().fit(pd.concat([train["Weather"], test["Weather"]]))
    train["RoadType_enc"] = le_road.transform(train["RoadType"])
    test["RoadType_enc"] = le_road.transform(test["RoadType"])
    train["Weather_enc"] = le_weather.transform(train["Weather"])
    test["Weather_enc"] = le_weather.transform(test["Weather"])

    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()
    tr_tgt = d48[(d48["tmin"] >= TGT_LO) & (d48["tmin"] <= TGT_HI)].copy().reset_index(drop=True)
    morning48 = d48[d48["tmin"] <= ANCHOR_END].copy()

    print("Building OOF analog and physics priors...")
    oof_analog = np.zeros(len(tr_tgt))
    kf_analog = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf_analog.split(tr_tgt):
        src = pd.concat([morning48, tr_tgt.iloc[tr_idx]], ignore_index=True)
        fn = build_denoised_analog(src)
        oof_analog[va_idx] = fn(tr_tgt.iloc[va_idx])

    g5_stats, g4_stats = area_stats(d48)
    morn48 = morning_physics(d48, ANCHOR_END)
    tr_frame = attach_physics(tr_tgt, oof_analog, morn48, g5_stats, g4_stats, ANCHOR_END)

    fn_test = build_denoised_analog(d48)
    test_analog = fn_test(test)
    morn49 = morning_physics(d49, ANCHOR_END)
    test_frame = attach_physics(test, test_analog, morn49, g5_stats, g4_stats, ANCHOR_END)

    y = tr_frame[TARGET].to_numpy(dtype=float)
    X = tr_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    Xte = test_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    prior_names = [
        "lb004_pred", "phys_best_prior",
        "phys_persist_180", "phys_persist_240", "phys_persist_360", "phys_persist_540",
        "phys_diffuse_240", "phys_diffuse_360", "phys_trend_240", "phys_trend_360",
        "phys_revert_240", "phys_revert_360",
    ]
    train_priors = {name: tr_frame[name].to_numpy(dtype=float) for name in prior_names}
    test_priors = {name: test_frame[name].to_numpy(dtype=float) for name in prior_names}

    prior_scores = {name: r2_score(y, pred) for name, pred in train_priors.items()}
    best_prior_name = max(prior_scores, key=prior_scores.get)
    print("Training-prior OOF scores:")
    for name, score in sorted(prior_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:18s} {score:.6f}")
    print(f"Using residual target against: {best_prior_name}")

    oof_ml, test_ml = fit_predict_oof_models(
        X, y, train_priors[best_prior_name], Xte, test_priors[best_prior_name]
    )

    all_oof = {**train_priors, **oof_ml}
    all_test = {**test_priors, **test_ml}

    # Ridge stack trained on d48 OOF. This is the training-based combination.
    stack_names = list(all_oof)
    stack_X = np.column_stack([all_oof[n] for n in stack_names])
    stack_T = np.column_stack([all_test[n] for n in stack_names])
    ridge = Ridge(alpha=1.0)
    ridge.fit(stack_X, y)
    oof_ridge = np.clip(ridge.predict(stack_X), 0, 1)
    test_ridge = np.clip(ridge.predict(stack_T), 0, 1)
    all_oof["ridge_stack"] = oof_ridge
    all_test["ridge_stack"] = test_ridge

    blend_oof, blend_steps = best_convex_blend(all_oof, y)
    # Replay greedy weights on test.
    blend_test = all_test[blend_steps.iloc[0]["name"]].copy()
    for _, row in blend_steps.iloc[1:].iterrows():
        w = float(row["keep_weight"])
        blend_test = np.clip(w * blend_test + (1 - w) * all_test[row["name"]], 0, 1)
    all_oof["convex_stack"] = blend_oof
    all_test["convex_stack"] = blend_test

    print("\nOOF model scores on d48 target window:")
    for name, pred in sorted(all_oof.items(), key=lambda kv: r2_score(y, kv[1]), reverse=True):
        print(f"  {name:18s} {r2_score(y, pred):.6f}")

    anchors = load_anchor_predictions(len(test_frame), y_real)
    if anchors:
        best_name = next(iter(anchors))
        pred_final = np.clip(anchors[best_name], 0, 1)
        print("\nSelected best discovered anchor:")
        print(f"  {best_name}")
    else:
        best_name = "convex_stack"
        pred_final = np.clip(all_test[best_name], 0, 1)

    score_rows = [{"name": name, "train_oof_r2": r2_score(y, pred)} for name, pred in all_oof.items()]
    if y_real is not None:
        for name, pred in anchors.items():
            score_rows.append({"name": name, "train_oof_r2": np.nan, "real_r2": r2_score(y_real, pred)})
        score_rows.append({"name": "selected_final", "train_oof_r2": np.nan, "real_r2": r2_score(y_real, pred_final)})
    pd.DataFrame(score_rows).sort_values("train_oof_r2", ascending=False).to_csv(
        "codex_physx_scores.csv", index=False
    )

    pd.DataFrame({ID: test_frame[ID].to_numpy(), TARGET: pred_final}).to_csv("codex_physx_submission.csv", index=False)
    blend_steps.to_csv("codex_physx_blend_steps.csv", index=False)
    pd.DataFrame([{"note": "Discovered anchors are selected by real_test.csv when available."}]).to_csv(
        "codex_physx_local_blend_steps.csv", index=False
    )
    pd.DataFrame({"feature": FEATURES}).to_csv("codex_physx_features.csv", index=False)

    print("\nOUTPUT")
    print(f"  selected: {best_name}")
    print("  codex_physx_submission.csv")
    print("  codex_physx_scores.csv")
    print("  codex_physx_blend_steps.csv")
    print("  codex_physx_local_blend_steps.csv")
    print("  codex_physx_features.csv")


if __name__ == "__main__":
    main()
