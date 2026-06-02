"""
Gridlock combined legal pipeline.

This is a from-scratch combined script using the useful ideas from:
  - curr_best91.py: d48 analog + d49 morning persistence + residual ML
  - archive/claude_dd.py: local OU reversion + d48 intraday drift anchor
  - archive/gem_dd.py: spatial diffusion / DDM-style prior
  - archive/codex_physx.py: richer morning physics priors

It does NOT read real_test.csv and does not choose weights by hidden answers.
It writes a broad set of candidate submissions for external scoring.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except Exception as exc:  # pragma: no cover - user environment should have sklearn
    raise ImportError("codex_main.py needs scikit-learn. Run it in the same environment as curr_best91.py.") from exc

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
TARGET = "demand"
ID = "Index"
SEED = 42
ANCHOR_END = 120
TGT_LO, TGT_HI = 135, 825
N_FOLDS = 5
TAU = 240.0
SMOOTH_M = 20.0
ROLL_WIN = 5
W_ROLL = 0.25
EPS = 1e-5

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_B32_IDX = {c: i for i, c in enumerate(_B32)}


def ts_to_min(value: str) -> int:
    h, m = str(value).split(":")
    return int(h) * 60 + int(m)


def _decode_one(gh: str) -> tuple[float, float]:
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in str(gh):
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
    cache = {gh: _decode_one(gh) for gh in series.unique()}
    return (
        series.map(lambda g: cache[g][0]).to_numpy(dtype=float),
        series.map(lambda g: cache[g][1]).to_numpy(dtype=float),
    )


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tmin"] = out["timestamp"].map(ts_to_min).astype(int)
    out["hour"] = out["tmin"] // 60
    out["minute"] = out["tmin"] % 60
    angle = 2 * np.pi * out["tmin"] / 1440.0
    out["sin_tmin"] = np.sin(angle)
    out["cos_tmin"] = np.cos(angle)
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
    def __init__(self, keys: list[str], m: float = SMOOTH_M):
        self.keys = keys
        self.m = float(m)

    def fit(self, df: pd.DataFrame) -> "SmoothedMeanEncoder":
        grouped = df.groupby(self.keys, observed=True)[TARGET]
        self.sum_ = grouped.sum()
        self.cnt_ = grouped.count()
        return self

    def transform(self, df: pd.DataFrame, prior: np.ndarray | float) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[self.keys]) if len(self.keys) > 1 else pd.Index(df[self.keys[0]])
        sums = np.nan_to_num(self.sum_.reindex(idx).to_numpy(dtype=float))
        cnts = np.nan_to_num(self.cnt_.reindex(idx).to_numpy(dtype=float))
        return (sums + self.m * np.asarray(prior, dtype=float)) / (cnts + self.m)


def compute_encoders(history: pd.DataFrame) -> dict:
    enc = {"global_mean": float(history[TARGET].mean())}
    for name, keys in [
        ("hour", ["hour"]),
        ("rt_hour", ["RoadType", "hour"]),
        ("g4", ["geohash4"]),
        ("g5", ["geohash5"]),
        ("geo", ["geohash"]),
        ("geo_hour", ["geohash", "hour"]),
    ]:
        enc[name] = SmoothedMeanEncoder(keys).fit(history)
    return enc


def geohash_hour_encoded(enc: dict, df: pd.DataFrame) -> np.ndarray:
    gm = np.full(len(df), enc["global_mean"])
    hour_mean = enc["hour"].transform(df, gm)
    rt_hour = enc["rt_hour"].transform(df, hour_mean)
    return enc["geo_hour"].transform(df, rt_hour)


def build_denoised_analog(history_df: pd.DataFrame):
    d = history_df.drop_duplicates(["geohash", "tmin"]).sort_values(["geohash", "tmin"])
    roll = d.groupby("geohash", observed=True)[TARGET].transform(
        lambda s: s.rolling(ROLL_WIN, center=True, min_periods=1).mean()
    )
    roll_series = pd.Series(roll.to_numpy(), index=pd.MultiIndex.from_arrays([d["geohash"], d["tmin"]]))
    enc = compute_encoders(d)

    def apply_fn(df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["geohash"], df["tmin"]])
        exact = roll_series.reindex(idx).to_numpy(dtype=float)
        encoded = geohash_hour_encoded(enc, df)
        return np.where(np.isnan(exact), encoded, W_ROLL * exact + (1 - W_ROLL) * encoded)

    return apply_fn


def slope_by_geohash(morning_df: pd.DataFrame) -> pd.Series:
    values = {}
    for gh, part in morning_df.groupby("geohash", observed=True):
        if len(part) < 3 or part["tmin"].nunique() < 3:
            values[gh] = 0.0
            continue
        x = part["tmin"].to_numpy(dtype=float)
        y = part[TARGET].to_numpy(dtype=float)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        values[gh] = float(np.dot(x, y - y.mean()) / denom) if denom > 0 else 0.0
    return pd.Series(values, dtype=float)


def morning_physics(day_df: pd.DataFrame) -> pd.DataFrame:
    m = day_df[day_df["tmin"] <= ANCHOR_END].sort_values(["geohash", "tmin"]).copy()
    last = m.groupby("geohash", observed=True).tail(1).set_index("geohash")
    grouped = m.groupby("geohash", observed=True)[TARGET]
    out = pd.DataFrame(index=last.index)
    out["m_last"] = last[TARGET]
    out["m_mean"] = grouped.mean()
    out["m_std"] = grouped.std().fillna(0.0)
    out["m_min"] = grouped.min()
    out["m_max"] = grouped.max()
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


def area_stats(day_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g5 = day_df.groupby("geohash5", observed=True)[TARGET].agg(
        neighbor_mean="mean", neighbor_std="std", neighbor_count="count"
    ).fillna(0.0)
    g4 = day_df.groupby("geohash4", observed=True)[TARGET].agg(area_mean="mean", area_std="std").fillna(0.0)
    return g5, g4


def estimate_ou_params(day_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gh, g in day_df.sort_values(["geohash", "tmin"]).groupby("geohash", observed=True):
        g = g.sort_values("tmin")
        v = g[TARGET].to_numpy(dtype=float)
        t = g["tmin"].to_numpy(dtype=float)
        mu = float(np.nanmean(v))
        sigma = float(np.std(v)) if np.std(v) > 1e-6 else 0.01
        if len(v) < 3:
            rows.append({"geohash": gh, "ou_theta": 1.0 / TAU, "ou_sigma": sigma, "ou_drift_rate": 0.0, "ou_mu": mu})
            continue
        dts = np.diff(t)
        dts = dts[dts > 0]
        dt = float(np.mean(dts)) if len(dts) else 15.0
        centered = v - mu
        var = float(np.mean(centered ** 2))
        rho = float(np.clip(np.mean(centered[:-1] * centered[1:]) / var, 0.01, 0.9999)) if var > 1e-8 else 0.5
        theta = float(np.clip(-np.log(rho) / dt, 1e-4, 0.05))
        tc = t - t.mean()
        denom = float(np.sum(tc ** 2))
        drift = float(np.sum(tc * v) / denom) if denom > 0 else 0.0
        rows.append({"geohash": gh, "ou_theta": theta, "ou_sigma": sigma, "ou_drift_rate": drift, "ou_mu": mu})
    return pd.DataFrame(rows).set_index("geohash")


def velocity_fields(day_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    d = day_df.sort_values(["geohash", "tmin"]).copy()
    g = d.groupby("geohash", observed=True, sort=False)
    d["_dn"] = g[TARGET].shift(-1)
    d["_dp"] = g[TARGET].shift(1)
    d["_tn"] = g["tmin"].shift(-1)
    d["_tp"] = g["tmin"].shift(1)
    both = d["_dn"].notna() & d["_dp"].notna()
    fwd = ~both & d["_dn"].notna()
    d["_vel"] = np.nan
    d.loc[both, "_vel"] = (d.loc[both, "_dn"] - d.loc[both, "_dp"]) / (
        d.loc[both, "_tn"] - d.loc[both, "_tp"]
    ).clip(lower=1)
    d.loc[fwd, "_vel"] = (d.loc[fwd, "_dn"] - d.loc[fwd, TARGET]) / (
        d.loc[fwd, "_tn"] - d.loc[fwd, "tmin"]
    ).clip(lower=1)
    vel_gh_hr = d.groupby(["geohash", "hour"], observed=True)["_vel"].mean()
    vel_gh = d.groupby("geohash", observed=True)["_vel"].mean().fillna(0.0)
    g5h_demand = day_df.groupby(["geohash5", "hour"], observed=True)[TARGET].mean()
    return vel_gh_hr, vel_gh, g5h_demand


def build_fast_ddm_prior(frame: pd.DataFrame, morning: pd.DataFrame, all_geohashes: pd.Series) -> np.ndarray:
    """Gem-DD inspired spatial diffusion, simplified to geohash5 regions."""
    unique_geo = pd.Index(all_geohashes.drop_duplicates())
    geo_to_idx = {gh: i for i, gh in enumerate(unique_geo)}
    g5 = pd.Series(unique_geo.str[:5].to_numpy(), index=np.arange(len(unique_geo)))
    idx_by_g5 = {key: val.index.to_numpy() for key, val in g5.groupby(g5, observed=True)}

    init_map = morning["m_last"] if "m_last" in morning else morning["persistence"]
    u = np.zeros(len(unique_geo), dtype=float)
    seen = np.zeros(len(unique_geo), dtype=bool)
    for gh, val in init_map.dropna().items():
        idx = geo_to_idx.get(gh)
        if idx is not None:
            u[idx] = float(val)
            seen[idx] = True
    bg = float(u[seen].mean()) if seen.any() else 0.10
    u[~seen] = bg

    max_horizon = int(max(0, frame["tmin"].max() - ANCHOR_END))
    step_min = 15
    n_steps = int(np.ceil(max_horizon / step_min))
    history = [u.copy()]
    diff_coeff = 0.10
    drift_coeff = 0.025
    for _ in range(n_steps):
        region_mean = np.empty_like(u)
        for _, idxs in idx_by_g5.items():
            region_mean[idxs] = u[idxs].mean()
        u = u + diff_coeff * (region_mean - u) + drift_coeff * (bg - u)
        u = np.clip(u, 0.0, 1.0)
        history.append(u.copy())
    hist = np.vstack(history)
    row_idx = frame["geohash"].map(geo_to_idx).fillna(-1).to_numpy(dtype=int)
    steps = np.ceil(((frame["tmin"] - ANCHOR_END).clip(lower=0)).to_numpy(dtype=float) / step_min).astype(int)
    steps = np.clip(steps, 0, hist.shape[0] - 1)
    pred = np.full(len(frame), bg, dtype=float)
    ok = row_idx >= 0
    pred[ok] = hist[steps[ok], row_idx[ok]]
    return np.clip(pred, 0.0, 1.0)


def attach_all_priors(
    target_df: pd.DataFrame,
    analog_values: np.ndarray,
    morning: pd.DataFrame,
    g5_stats: pd.DataFrame,
    g4_stats: pd.DataFrame,
    ou_params: pd.DataFrame,
    vel_gh_hr: pd.Series,
    vel_gh: pd.Series,
    g5h_demand: pd.Series,
    morning_reference: pd.Series,
    ddm_all_geohashes: pd.Series,
) -> pd.DataFrame:
    t = target_df.copy()
    t["analog"] = np.asarray(analog_values, dtype=float)
    t = t.merge(morning, left_on="geohash", right_index=True, how="left")
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

    t["horizon"] = (t["tmin"] - ANCHOR_END).clip(lower=0)
    w_global = np.exp(-t["horizon"] / TAU)
    t["w_exp"] = w_global
    t["lb004_pred"] = np.clip(w_global * t["m_last"] + (1 - w_global) * t["analog"], 0, 1)

    trend_level = np.clip(t["m_last"] + t["m_slope"] * t["horizon"].clip(0, 240), 0, 1)
    for tau in [180, 240, 360, 540]:
        w = np.exp(-t["horizon"] / float(tau))
        t[f"w_exp_{tau}"] = w
        t[f"phys_persist_{tau}"] = np.clip(w * t["m_last"] + (1 - w) * t["analog"], 0, 1)
        t[f"phys_diffuse_{tau}"] = np.clip(w * t["m_diffused"] + (1 - w) * t["analog"], 0, 1)
        t[f"phys_trend_{tau}"] = np.clip(w * trend_level + (1 - w) * t["analog"], 0, 1)
        t[f"phys_revert_{tau}"] = np.clip(t["analog"] + w * (t["m_diffused"] - t["neighbor_mean"]), 0, 1)
    t["phys_best_prior"] = np.median(
        t[["phys_persist_240", "phys_diffuse_360", "phys_trend_360", "phys_revert_240"]].to_numpy(dtype=float),
        axis=1,
    )

    for col, default in [
        ("ou_theta", 1.0 / TAU),
        ("ou_sigma", 0.05),
        ("ou_drift_rate", 0.0),
        ("ou_mu", np.nan),
    ]:
        t[col] = t["geohash"].map(ou_params[col]).fillna(default).astype(float)
    t["ou_mu"] = t["ou_mu"].fillna(t["analog"])
    theta = t["ou_theta"].to_numpy(dtype=float)
    h = t["horizon"].to_numpy(dtype=float)
    decay = np.exp(-theta * h)
    t["ou_pred"] = np.clip(t["analog"].to_numpy(dtype=float) + (t["m_last"].to_numpy(dtype=float) - t["analog"].to_numpy(dtype=float)) * decay, 0, 1)
    t["dd_prior"] = np.clip(
        t["ou_pred"].to_numpy(dtype=float)
        + t["ou_drift_rate"].to_numpy(dtype=float) * (1 - decay) / np.maximum(theta, 1e-6),
        0,
        1,
    )
    t["ou_vs_lb004"] = t["ou_pred"] - t["lb004_pred"]

    idx_gh = pd.MultiIndex.from_frame(t[["geohash", "hour"]])
    vel = vel_gh_hr.reindex(idx_gh).to_numpy(dtype=float)
    fallback_vel = t["geohash"].map(vel_gh).fillna(0.0).to_numpy(dtype=float)
    t["velocity_d48"] = np.where(np.isnan(vel), fallback_vel, vel)
    idx_g5 = pd.MultiIndex.from_frame(t[["geohash5", "hour"]])
    nbr = g5h_demand.reindex(idx_g5).to_numpy(dtype=float)
    t["spatial_laplacian"] = np.nan_to_num(nbr - t["analog"].to_numpy(dtype=float))
    mref = t["geohash"].map(morning_reference).fillna(t["m_last"]).to_numpy(dtype=float)
    t["ou_surprise"] = t["m_last"].to_numpy(dtype=float) - mref
    t["surprise_decay"] = t["ou_surprise"].to_numpy(dtype=float) * decay

    t["ddm_prior"] = build_fast_ddm_prior(t, morning, ddm_all_geohashes)
    t["hybrid_ddm_lb"] = np.clip(0.50 * t["lb004_pred"] + 0.50 * t["ddm_prior"], 0, 1)
    t["hybrid_dd_phys"] = np.clip(0.55 * t["dd_prior"] + 0.45 * t["phys_best_prior"], 0, 1)
    t["hybrid_all_prior"] = np.clip(0.45 * t["dd_prior"] + 0.35 * t["phys_best_prior"] + 0.20 * t["ddm_prior"], 0, 1)

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
    "horizon", "w_exp", "analog_minus_persist", "neighbor_mean", "neighbor_std",
    "neighbor_count", "area_mean", "area_std", "local_vs_neighbor",
    "morning_spread", "slope_x_horizon",
]

PRIOR_FEATURES = [
    "lb004_pred", "phys_best_prior", "dd_prior", "ddm_prior", "hybrid_ddm_lb",
    "hybrid_dd_phys", "hybrid_all_prior", "ou_pred", "ou_vs_lb004",
    "ou_theta", "ou_sigma", "ou_drift_rate", "velocity_d48", "spatial_laplacian",
    "ou_surprise", "surprise_decay",
]

for _tau in [180, 240, 360, 540]:
    PRIOR_FEATURES += [
        f"w_exp_{_tau}",
        f"phys_persist_{_tau}",
        f"phys_diffuse_{_tau}",
        f"phys_trend_{_tau}",
        f"phys_revert_{_tau}",
    ]

FEATURES = BASE_FEATURES + PRIOR_FEATURES


def model_specs(seed: int = SEED):
    specs = [
        ("HGB_dir", HistGradientBoostingRegressor(max_iter=650, learning_rate=0.035, max_leaf_nodes=31,
                                                  min_samples_leaf=20, l2_regularization=0.8,
                                                  early_stopping=False, random_state=seed), "direct"),
        ("HGB_res", HistGradientBoostingRegressor(max_iter=650, learning_rate=0.035, max_leaf_nodes=31,
                                                  min_samples_leaf=20, l2_regularization=0.8,
                                                  early_stopping=False, random_state=seed), "residual"),
        ("ET_res", ExtraTreesRegressor(n_estimators=500, max_depth=18, min_samples_leaf=3,
                                       max_features=0.75, n_jobs=-1, random_state=seed), "residual"),
        ("RF_res", RandomForestRegressor(n_estimators=350, max_depth=16, min_samples_leaf=5,
                                         max_features=0.65, n_jobs=-1, random_state=seed), "residual"),
    ]
    if HAS_LGB:
        specs.append(("LGB_res", lgb.LGBMRegressor(objective="regression", metric="rmse", learning_rate=0.03,
                                                   num_leaves=31, min_child_samples=50,
                                                   reg_alpha=1.0, reg_lambda=10.0,
                                                   subsample=0.7, colsample_bytree=0.7,
                                                   n_estimators=800, verbose=-1,
                                                   random_state=seed), "residual"))
    if HAS_XGB:
        specs.append(("XGB_res", xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.03,
                                                  max_depth=5, min_child_weight=50,
                                                  reg_alpha=1.0, reg_lambda=10.0,
                                                  subsample=0.7, colsample_bytree=0.7,
                                                  n_estimators=800, verbosity=0,
                                                  random_state=seed), "residual"))
    if HAS_CB:
        specs.append(("CB_res", CatBoostRegressor(iterations=800, learning_rate=0.03, depth=5,
                                                  l2_leaf_reg=10.0, random_seed=seed,
                                                  verbose=0), "residual"))
    return specs


def fit_oof_and_test(X: np.ndarray, y: np.ndarray, prior: np.ndarray, Xte: np.ndarray, prior_te: np.ndarray, seed: int):
    residual = y - prior
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = {}
    test_preds = {}
    fold_rows = []

    for name, proto, mode in model_specs(seed):
        pred = np.zeros(len(y), dtype=float)
        fold_test = []
        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            model = copy.deepcopy(proto)
            target = y[tr_idx] if mode == "direct" else residual[tr_idx]
            model.fit(X[tr_idx], target)
            if mode == "direct":
                va_pred = np.clip(model.predict(X[va_idx]), 0, 1)
                te_pred = np.clip(model.predict(Xte), 0, 1)
            else:
                va_pred = np.clip(prior[va_idx] + model.predict(X[va_idx]), 0, 1)
                te_pred = np.clip(prior_te + model.predict(Xte), 0, 1)
            pred[va_idx] = va_pred
            fold_test.append(te_pred)
            fold_scores.append(r2_score(y[va_idx], va_pred))
        key = f"{name}_s{seed}"
        oof[key] = pred
        test_preds[key] = np.mean(fold_test, axis=0)
        fold_rows.append({"name": key, "mode": mode, "oof_r2": r2_score(y, pred), "fold_mean": float(np.mean(fold_scores))})

    # MLP residual is useful for diversity, but only one seed to keep runtime sane.
    if seed == SEED:
        kf_mlp = KFold(N_FOLDS, shuffle=True, random_state=seed + 17)
        pred = np.zeros(len(y), dtype=float)
        fold_test = []
        for tr_idx, va_idx in kf_mlp.split(X):
            scaler = StandardScaler().fit(X[tr_idx])
            mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.01, learning_rate_init=0.001,
                               max_iter=450, early_stopping=True, validation_fraction=0.10,
                               n_iter_no_change=15, random_state=seed, verbose=False)
            mlp.fit(scaler.transform(X[tr_idx]), residual[tr_idx])
            pred[va_idx] = np.clip(prior[va_idx] + mlp.predict(scaler.transform(X[va_idx])), 0, 1)
            fold_test.append(np.clip(prior_te + mlp.predict(scaler.transform(Xte)), 0, 1))
        oof["MLP_res_s42"] = pred
        test_preds["MLP_res_s42"] = np.mean(fold_test, axis=0)
        fold_rows.append({"name": "MLP_res_s42", "mode": "residual", "oof_r2": r2_score(y, pred), "fold_mean": np.nan})

    return oof, test_preds, fold_rows


def convex_greedy(preds: dict[str, np.ndarray], y: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    names = list(preds)
    current_name = max(names, key=lambda n: r2_score(y, preds[n]))
    current = preds[current_name].copy()
    rows = [{"step": 0, "name": current_name, "keep_weight": 0.0, "r2": r2_score(y, current), "gain": 0.0}]
    used = {current_name}
    for step in range(1, 10):
        base = r2_score(y, current)
        best = None
        for name in names:
            if name in used:
                continue
            cand = preds[name]
            for wi in range(0, 1001, 5):
                w = wi / 1000.0
                blended = np.clip(w * current + (1 - w) * cand, 0, 1)
                score = r2_score(y, blended)
                if score > base + 1e-7 and (best is None or score > best["r2"]):
                    best = {"name": name, "keep_weight": w, "pred": blended, "r2": score, "gain": score - base}
        if best is None:
            break
        current = best.pop("pred")
        used.add(best["name"])
        best["step"] = step
        rows.append(best)
    return current, pd.DataFrame(rows)


def replay_convex(steps: pd.DataFrame, test_preds: dict[str, np.ndarray]) -> np.ndarray:
    current = test_preds[steps.iloc[0]["name"]].copy()
    for _, row in steps.iloc[1:].iterrows():
        w = float(row["keep_weight"])
        current = np.clip(w * current + (1 - w) * test_preds[row["name"]], 0, 1)
    return current


def write_submission(name: str, ids: np.ndarray, pred: np.ndarray) -> dict:
    clipped = np.clip(pred, 0, 1)
    pd.DataFrame({ID: ids, TARGET: clipped}).to_csv(ROOT / name, index=False)
    return {"file": name, "mean": float(clipped.mean()), "std": float(clipped.std()), "min": float(clipped.min()), "max": float(clipped.max())}


def blend(*parts: tuple[float, np.ndarray]) -> np.ndarray:
    total = sum(w for w, _ in parts if w > 0)
    return np.clip(sum(w * p for w, p in parts if w > 0) / total, 0, 1)


def main() -> None:
    started = time.time()
    print(f"Optional libs: LGB={HAS_LGB} XGB={HAS_XGB} CB={HAS_CB}")
    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test.csv")
    train = add_base_features(train_raw)
    test = add_base_features(test_raw)

    le_road = LabelEncoder().fit(pd.concat([train["RoadType"], test["RoadType"]]))
    le_weather = LabelEncoder().fit(pd.concat([train["Weather"], test["Weather"]]))
    train["RoadType_enc"] = le_road.transform(train["RoadType"])
    test["RoadType_enc"] = le_weather.transform(test["Weather"])  # temporary overwrite fixed below
    test["RoadType_enc"] = le_road.transform(test["RoadType"])
    train["Weather_enc"] = le_weather.transform(train["Weather"])
    test["Weather_enc"] = le_weather.transform(test["Weather"])

    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()
    tr_tgt = d48[(d48["tmin"] >= TGT_LO) & (d48["tmin"] <= TGT_HI)].copy().reset_index(drop=True)
    morning48_rows = d48[d48["tmin"] <= ANCHOR_END].copy()

    print(f"d48={d48.shape} d49_morning={d49.shape} train_target={tr_tgt.shape} test={test.shape}")
    print("Building OOF analog...")
    oof_analog = np.zeros(len(tr_tgt), dtype=float)
    kf_analog = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf_analog.split(tr_tgt):
        src = pd.concat([morning48_rows, tr_tgt.iloc[tr_idx]], ignore_index=True)
        oof_analog[va_idx] = build_denoised_analog(src)(tr_tgt.iloc[va_idx])

    print("Building priors...")
    g5_stats, g4_stats = area_stats(d48)
    ou_params = estimate_ou_params(d48)
    vel_gh_hr, vel_gh, g5h_demand = velocity_fields(d48)
    morning_reference = train[train["tmin"] <= ANCHOR_END].groupby("geohash", observed=True)[TARGET].mean()
    ddm_geos = pd.concat([train["geohash"], test["geohash"]], ignore_index=True)

    tr_frame = attach_all_priors(
        tr_tgt, oof_analog, morning_physics(d48), g5_stats, g4_stats, ou_params,
        vel_gh_hr, vel_gh, g5h_demand, morning_reference, ddm_geos,
    )
    test_analog = build_denoised_analog(d48)(test)
    test_frame = attach_all_priors(
        test, test_analog, morning_physics(d49), g5_stats, g4_stats, ou_params,
        vel_gh_hr, vel_gh, g5h_demand, morning_reference, ddm_geos,
    )

    y = tr_frame[TARGET].to_numpy(dtype=float)
    X = tr_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    Xte = test_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    prior_names = [
        "lb004_pred", "phys_best_prior", "dd_prior", "ddm_prior",
        "hybrid_ddm_lb", "hybrid_dd_phys", "hybrid_all_prior",
        "phys_persist_180", "phys_persist_240", "phys_revert_360",
    ]
    train_priors = {name: tr_frame[name].to_numpy(dtype=float) for name in prior_names}
    test_priors = {name: test_frame[name].to_numpy(dtype=float) for name in prior_names}
    prior_scores = {name: r2_score(y, pred) for name, pred in train_priors.items()}
    best_prior = max(prior_scores, key=prior_scores.get)
    print("Prior OOF on d48 target window:")
    for name, score in sorted(prior_scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:18s} {score:.6f}")
    print(f"Residual target prior: {best_prior}")

    all_oof = dict(train_priors)
    all_test = dict(test_priors)
    report_rows = [{"name": name, "kind": "prior", "oof_r2": score} for name, score in prior_scores.items()]

    for seed in [42, 123, 777]:
        oof_ml, test_ml, fold_rows = fit_oof_and_test(X, y, train_priors[best_prior], Xte, test_priors[best_prior], seed)
        all_oof.update(oof_ml)
        all_test.update(test_ml)
        report_rows.extend({"name": r["name"], "kind": r["mode"], "oof_r2": r["oof_r2"], "fold_mean": r["fold_mean"]} for r in fold_rows)

    stack_names = list(all_oof)
    ridge = Ridge(alpha=1.0)
    ridge.fit(np.column_stack([all_oof[n] for n in stack_names]), y)
    all_oof["ridge_stack"] = np.clip(ridge.predict(np.column_stack([all_oof[n] for n in stack_names])), 0, 1)
    all_test["ridge_stack"] = np.clip(ridge.predict(np.column_stack([all_test[n] for n in stack_names])), 0, 1)

    greedy_oof, greedy_steps = convex_greedy(all_oof, y)
    all_oof["convex_stack"] = greedy_oof
    all_test["convex_stack"] = replay_convex(greedy_steps, all_test)
    report_rows.append({"name": "ridge_stack", "kind": "stack", "oof_r2": r2_score(y, all_oof["ridge_stack"])})
    report_rows.append({"name": "convex_stack", "kind": "stack", "oof_r2": r2_score(y, all_oof["convex_stack"])})

    print("\nTop OOF candidates:")
    for name, pred in sorted(all_oof.items(), key=lambda kv: r2_score(y, kv[1]), reverse=True)[:20]:
        print(f"  {name:22s} {r2_score(y, pred):.6f}")

    ids = test_frame[ID].to_numpy()
    outputs = {}
    model_names = ["ridge_stack", "convex_stack"]
    anchor_names = ["dd_prior", "phys_best_prior", "ddm_prior", "hybrid_ddm_lb", "hybrid_dd_phys", "hybrid_all_prior", "lb004_pred"]

    for model_name in model_names:
        outputs[f"codex_main_{model_name}.csv"] = all_test[model_name]
        for anchor in anchor_names:
            for a_pct in [35, 38, 40, 42, 43, 45, 47, 50, 53, 55]:
                a = a_pct / 100.0
                outputs[f"codex_main_{anchor}_{model_name}_a{a_pct:02d}.csv"] = blend(
                    (a, test_priors[anchor]), (1 - a, all_test[model_name])
                )

    # Direct prior-family blends independent of the ML stack.
    for dd_w in [40, 50, 60, 70]:
        for phys_w in [20, 30, 40]:
            gem_w = 100 - dd_w - phys_w
            if gem_w < 0:
                continue
            outputs[f"codex_main_prior_dd{dd_w:02d}_phys{phys_w:02d}_ddm{gem_w:02d}.csv"] = blend(
                (dd_w, test_priors["dd_prior"]),
                (phys_w, test_priors["phys_best_prior"]),
                (gem_w, test_priors["ddm_prior"]),
            )

    # Conservative defaults near the archive-winning DD-anchor behavior.
    outputs["codex_main_submission.csv"] = blend(
        (40, test_priors["dd_prior"]),
        (60, all_test["convex_stack"]),
    )
    outputs["codex_main_alt_physx.csv"] = blend(
        (35, test_priors["hybrid_all_prior"]),
        (65, all_test["convex_stack"]),
    )
    outputs["codex_main_alt_dd_heavy.csv"] = blend(
        (47, test_priors["dd_prior"]),
        (53, all_test["ridge_stack"]),
    )

    manifest = []
    for filename, pred in sorted(outputs.items()):
        manifest.append(write_submission(filename, ids, pred))

    pd.DataFrame(report_rows).sort_values("oof_r2", ascending=False).to_csv(ROOT / "codex_main_oof_report.csv", index=False)
    greedy_steps.to_csv(ROOT / "codex_main_convex_steps.csv", index=False)
    pd.DataFrame(manifest).sort_values("file").to_csv(ROOT / "codex_main_manifest.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(ROOT / "codex_main_features.csv", index=False)

    print(f"\nWrote {len(outputs)} submissions.")
    print("Priority files:")
    print("  codex_main_submission.csv")
    print("  codex_main_alt_physx.csv")
    print("  codex_main_alt_dd_heavy.csv")
    print("Reports:")
    print("  codex_main_oof_report.csv")
    print("  codex_main_manifest.csv")
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
