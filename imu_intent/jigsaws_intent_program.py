from __future__ import annotations

import argparse
import importlib.util
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if importlib.util.find_spec("xgboost") is not None:
    from xgboost import XGBClassifier
else:
    XGBClassifier = None

CLASS_NAME_MAP = {0: "NO_LOCK_REQUIRED", 1: "LOCK_REQUIRED"}
INV_CLASS_NAME_MAP = {v: k for k, v in CLASS_NAME_MAP.items()}


@dataclass
class TrialRecord:
    task: str
    trial_id: str
    surgeon: str
    signals: np.ndarray  # [T, D]
    gestures: np.ndarray  # [T]
    sample_rate_hz: float


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def parse_surgeon(trial_id: str) -> str:
    parts = trial_id.split("_")
    if len(parts) < 2:
        return "UNK"
    token = parts[-1]
    return token[0].upper() if token else "UNK"


def parse_transcription(path: Path, length: int) -> np.ndarray:
    labels = np.array(["G0"] * length, dtype=object)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 3:
                continue
            try:
                start = int(float(toks[0])) - 1
                end = int(float(toks[1])) - 1
            except ValueError:
                continue
            gesture = toks[2].strip()
            s = max(0, start)
            e = min(length - 1, end)
            if s <= e:
                labels[s : e + 1] = gesture
    return labels


def load_jigsaws_trials(cfg: dict) -> list[TrialRecord]:
    ds_cfg = cfg.get("dataset", {})
    root = Path(ds_cfg.get("root", ".")).resolve()
    tasks = list(ds_cfg.get("tasks", ["Knot_Tying", "Needle_Passing", "Suturing"]))
    kin_subdir = str(ds_cfg.get("kinematics_subdir", "kinematics/AllGestures"))
    trans_subdir = str(ds_cfg.get("transcriptions_subdir", "transcriptions"))
    sample_rate_hz = float(ds_cfg.get("sample_rate_hz", 30.0))

    records: list[TrialRecord] = []
    for task in tasks:
        kin_dir = root / task / kin_subdir
        trans_dir = root / task / trans_subdir
        if not kin_dir.exists() or not trans_dir.exists():
            continue
        for kin_path in sorted(kin_dir.glob("*.txt")):
            trial_id = kin_path.stem
            trans_path = trans_dir / f"{trial_id}.txt"
            if not trans_path.exists():
                continue
            try:
                signals = np.loadtxt(kin_path, dtype=float)
            except Exception:
                continue
            if signals.ndim == 1:
                signals = signals.reshape(-1, 1)
            if signals.shape[0] < 16:
                continue
            gestures = parse_transcription(trans_path, signals.shape[0])
            surgeon = parse_surgeon(trial_id)
            records.append(
                TrialRecord(
                    task=task,
                    trial_id=trial_id,
                    surgeon=surgeon,
                    signals=signals.astype(np.float32),
                    gestures=gestures,
                    sample_rate_hz=sample_rate_hz,
                )
            )
    return records


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


BASE_FEATURE_ORDER = [
    "global_vel_mean",
    "global_vel_std",
    "global_vel_p90",
    "global_acc_mean",
    "global_acc_p90",
    "posture_std_mean",
    "posture_range_mean",
    "posture_abs_mean",
    "micro_motion_ratio",
    "large_motion_ratio",
    "left_vel_mean",
    "right_vel_mean",
    "hand_vel_asymmetry",
    "hand_sync_corr",
    "vel_acc_ratio",
    "stability_margin",
    "velocity_balance_ratio",
    "sync_weighted_micro",
    "range_to_std_ratio",
    "p90_to_mean_vel",
    "bilateral_vel_sum",
    "hand_energy_ratio",
]

DISTRIBUTION_STAT_KEYS = [
    "mean",
    "std",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "max",
    "iqr",
]

DISTRIBUTION_FEATURE_PREFIXES = [
    "vel_abs",
    "acc_abs",
    "jerk_abs",
    "dim_vel_mean",
    "dim_acc_mean",
    "dim_range",
    "dim_std",
    "left_vel_abs",
    "left_acc_abs",
    "right_vel_abs",
    "right_acc_abs",
]

SHAPE_FEATURE_ORDER = [
    "vel_energy_entropy",
    "acc_energy_entropy",
    "range_entropy",
    "active_dim_ratio_vel_gt_p75",
    "vel_early_mean",
    "vel_mid_mean",
    "vel_late_mean",
    "vel_late_minus_early",
    "vel_mid_peak_ratio",
    "vel_trend_slope",
    "vel_temporal_std",
    "vel_temporal_entropy",
    "vel_burst_ratio",
    "left_range_entropy",
    "right_range_entropy",
    "hand_vel_corr",
    "hand_vel_absdiff_mean",
    "hand_vel_lead_lag1_corr",
    "hand_vel_lead_lagm1_corr",
]

FEATURE_ORDER = (
    BASE_FEATURE_ORDER
    + [f"{prefix}_{key}" for prefix in DISTRIBUTION_FEATURE_PREFIXES for key in DISTRIBUTION_STAT_KEYS]
    + SHAPE_FEATURE_ORDER
)


def _motion_entropy(values: np.ndarray) -> float:
    x = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    total = float(np.sum(x))
    if x.size <= 1 or total <= 1e-12:
        return 0.0
    p = x / total
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(float(x.size)))


def _add_distribution_stats(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        stats = dict.fromkeys(DISTRIBUTION_STAT_KEYS, 0.0)
    else:
        q10, q25, q50, q75, q90, q95 = np.percentile(x, [10, 25, 50, 75, 90, 95])
        stats = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p10": float(q10),
            "p25": float(q25),
            "p50": float(q50),
            "p75": float(q75),
            "p90": float(q90),
            "p95": float(q95),
            "max": float(np.max(x)),
            "iqr": float(q75 - q25),
        }
    for key in DISTRIBUTION_STAT_KEYS:
        features[f"{prefix}_{key}"] = float(stats[key])


def extract_window_feature_dict(window: np.ndarray) -> dict[str, float]:
    # kinematics window: [W, D]
    d1 = np.diff(window, axis=0)
    d2 = np.diff(d1, axis=0) if d1.shape[0] > 1 else np.zeros_like(d1)
    abs_d1 = np.abs(d1)
    abs_d2 = np.abs(d2)

    d = window.shape[1]
    split = d // 2 if d >= 2 else 1
    left = d1[:, :split] if d1.size else np.zeros((0, split), dtype=np.float32)
    right = d1[:, split:] if d1.size else np.zeros((0, d - split), dtype=np.float32)
    left_mag = np.mean(np.abs(left), axis=1) if left.size else np.zeros((0,), dtype=np.float32)
    right_mag = np.mean(np.abs(right), axis=1) if right.size else np.zeros((0,), dtype=np.float32)

    global_vel = float(abs_d1.mean()) if abs_d1.size else 0.0
    left_vel_mean = float(np.mean(np.abs(left))) if left.size else 0.0
    right_vel_mean = float(np.mean(np.abs(right))) if right.size else 0.0
    features = {
        "global_vel_mean": global_vel,
        "global_vel_std": float(abs_d1.std()) if abs_d1.size else 0.0,
        "global_vel_p90": float(np.percentile(abs_d1, 90)) if abs_d1.size else 0.0,
        "global_acc_mean": float(abs_d2.mean()) if abs_d2.size else 0.0,
        "global_acc_p90": float(np.percentile(abs_d2, 90)) if abs_d2.size else 0.0,
        "posture_std_mean": float(np.std(window, axis=0).mean()),
        "posture_range_mean": float(np.ptp(window, axis=0).mean()),
        "posture_abs_mean": float(np.mean(np.abs(window))),
        "micro_motion_ratio": float(np.mean(abs_d1 < 0.01)) if abs_d1.size else 0.0,
        "large_motion_ratio": float(np.mean(abs_d1 > 0.04)) if abs_d1.size else 0.0,
        "left_vel_mean": left_vel_mean,
        "right_vel_mean": right_vel_mean,
        "hand_vel_asymmetry": abs(left_vel_mean - right_vel_mean),
        "hand_sync_corr": _safe_corr(left_mag, right_mag),
    }
    eps = 1e-6
    features.update(
        {
            "vel_acc_ratio": features["global_vel_mean"] / (features["global_acc_mean"] + eps),
            "stability_margin": features["micro_motion_ratio"] - features["large_motion_ratio"],
            "velocity_balance_ratio": left_vel_mean / (right_vel_mean + eps),
            "sync_weighted_micro": features["hand_sync_corr"] * features["micro_motion_ratio"],
            "range_to_std_ratio": features["posture_range_mean"] / (features["posture_std_mean"] + eps),
            "p90_to_mean_vel": features["global_vel_p90"] / (features["global_vel_mean"] + eps),
            "bilateral_vel_sum": left_vel_mean + right_vel_mean,
            "hand_energy_ratio": (left_vel_mean**2 + eps) / (right_vel_mean**2 + eps),
        }
    )

    d3 = np.diff(d2, axis=0) if d2.shape[0] > 1 else np.zeros_like(d2)
    abs_d3 = np.abs(d3)
    _add_distribution_stats(features, "vel_abs", abs_d1)
    _add_distribution_stats(features, "acc_abs", abs_d2)
    _add_distribution_stats(features, "jerk_abs", abs_d3)

    per_dim_vel = np.mean(abs_d1, axis=0) if abs_d1.size else np.zeros((window.shape[1],), dtype=np.float32)
    per_dim_acc = np.mean(abs_d2, axis=0) if abs_d2.size else np.zeros((window.shape[1],), dtype=np.float32)
    per_dim_range = np.ptp(window, axis=0)
    per_dim_std = np.std(window, axis=0)
    _add_distribution_stats(features, "dim_vel_mean", per_dim_vel)
    _add_distribution_stats(features, "dim_acc_mean", per_dim_acc)
    _add_distribution_stats(features, "dim_range", per_dim_range)
    _add_distribution_stats(features, "dim_std", per_dim_std)

    features["vel_energy_entropy"] = _motion_entropy(np.square(per_dim_vel))
    features["acc_energy_entropy"] = _motion_entropy(np.square(per_dim_acc))
    features["range_entropy"] = _motion_entropy(per_dim_range)
    features["active_dim_ratio_vel_gt_p75"] = (
        float(np.mean(per_dim_vel > np.percentile(per_dim_vel, 75))) if per_dim_vel.size else 0.0
    )

    if abs_d1.size:
        n = abs_d1.shape[0]
        third = max(1, n // 3)
        early = float(np.mean(abs_d1[:third]))
        mid = float(np.mean(abs_d1[third : 2 * third])) if n >= 3 else early
        late = float(np.mean(abs_d1[-third:]))
        velocity_series = np.mean(abs_d1, axis=1)
        time_index = np.arange(velocity_series.size, dtype=np.float64)
        features.update(
            {
                "vel_early_mean": early,
                "vel_mid_mean": mid,
                "vel_late_mean": late,
                "vel_late_minus_early": late - early,
                "vel_mid_peak_ratio": mid / (max(early, late) + eps),
                "vel_trend_slope": (
                    float(np.polyfit(time_index, velocity_series, 1)[0]) if velocity_series.size > 1 else 0.0
                ),
                "vel_temporal_std": float(np.std(velocity_series)),
                "vel_temporal_entropy": _motion_entropy(velocity_series),
                "vel_burst_ratio": float(np.mean(velocity_series > np.percentile(velocity_series, 80))),
            }
        )
    else:
        features.update(
            {
                "vel_early_mean": 0.0,
                "vel_mid_mean": 0.0,
                "vel_late_mean": 0.0,
                "vel_late_minus_early": 0.0,
                "vel_mid_peak_ratio": 0.0,
                "vel_trend_slope": 0.0,
                "vel_temporal_std": 0.0,
                "vel_temporal_entropy": 0.0,
                "vel_burst_ratio": 0.0,
            }
        )

    for hand_name, hand_slice in (("left", slice(0, split)), ("right", slice(split, d))):
        hand_window = window[:, hand_slice]
        hand_d1 = np.diff(hand_window, axis=0) if hand_window.size else np.zeros((0, 0), dtype=np.float32)
        hand_d2 = np.diff(hand_d1, axis=0) if hand_d1.shape[0] > 1 else np.zeros_like(hand_d1)
        _add_distribution_stats(features, f"{hand_name}_vel_abs", np.abs(hand_d1))
        _add_distribution_stats(features, f"{hand_name}_acc_abs", np.abs(hand_d2))
        features[f"{hand_name}_range_entropy"] = (
            _motion_entropy(np.ptp(hand_window, axis=0)) if hand_window.size else 0.0
        )

    if abs_d1.size:
        left_velocity = np.mean(abs_d1[:, :split], axis=1) if split > 0 else np.zeros((abs_d1.shape[0],))
        right_velocity = (
            np.mean(abs_d1[:, split:], axis=1) if split < d else np.zeros((abs_d1.shape[0],), dtype=np.float32)
        )
        features["hand_vel_corr"] = _safe_corr(left_velocity, right_velocity)
        features["hand_vel_absdiff_mean"] = float(np.mean(np.abs(left_velocity - right_velocity)))
        features["hand_vel_lead_lag1_corr"] = (
            _safe_corr(left_velocity[:-1], right_velocity[1:]) if left_velocity.size > 2 else 0.0
        )
        features["hand_vel_lead_lagm1_corr"] = (
            _safe_corr(left_velocity[1:], right_velocity[:-1]) if left_velocity.size > 2 else 0.0
        )
    else:
        features["hand_vel_corr"] = 0.0
        features["hand_vel_absdiff_mean"] = 0.0
        features["hand_vel_lead_lag1_corr"] = 0.0
        features["hand_vel_lead_lagm1_corr"] = 0.0

    return features


def build_window_table(records: list[TrialRecord], cfg: dict) -> pd.DataFrame:
    w_cfg = cfg.get("window", {})
    window_size = int(w_cfg.get("size", 90))
    stride = int(w_cfg.get("stride", 30))
    majority_ratio = float(w_cfg.get("majority_ratio", 0.6))

    rows: list[dict[str, Any]] = []
    for rec in records:
        t = rec.signals.shape[0]
        if t < window_size:
            continue
        for start in range(0, t - window_size + 1, stride):
            end = start + window_size
            seg_g = rec.gestures[start:end]
            non_idle = [g for g in seg_g.tolist() if g != "G0"]
            if not non_idle:
                continue
            cnt = Counter(non_idle)
            major_gesture, major_count = cnt.most_common(1)[0]
            if major_count / float(window_size) < majority_ratio:
                continue

            feats = extract_window_feature_dict(rec.signals[start:end])
            row = {
                "task": rec.task,
                "trial_id": rec.trial_id,
                "surgeon": rec.surgeon,
                "start": start,
                "end": end,
                "major_gesture": major_gesture,
                "majority_ratio": major_count / float(window_size),
            }
            row.update(feats)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def discover_lock_gestures(
    df: pd.DataFrame,
    lock_quantile: float,
    min_windows_per_gesture: int,
    manual_lock: list[str],
    manual_no_lock: list[str],
    auto_discover: bool,
) -> tuple[set[str], pd.DataFrame]:
    stats = (
        df.groupby("major_gesture")
        .agg(
            windows=("major_gesture", "size"),
            median_global_vel=("global_vel_mean", "median"),
            p75_global_vel=("global_vel_mean", lambda x: float(np.percentile(x, 75))),
            mean_micro_ratio=("micro_motion_ratio", "mean"),
            mean_large_ratio=("large_motion_ratio", "mean"),
        )
        .sort_values("median_global_vel", ascending=True)
    )
    stats = stats.reset_index()
    valid = stats[stats["windows"] >= int(min_windows_per_gesture)].copy()

    if not auto_discover:
        lock_set = {g for g in manual_lock}
    else:
        if valid.empty:
            lock_set = {g for g in manual_lock}
        else:
            cutoff = float(np.quantile(valid["median_global_vel"].to_numpy(), lock_quantile))
            lock_set = set(valid.loc[valid["median_global_vel"] <= cutoff, "major_gesture"].astype(str).tolist())
            if manual_lock:
                lock_set.update(manual_lock)

    for g in manual_no_lock:
        if g in lock_set:
            lock_set.remove(g)

    return lock_set, stats


def apply_lock_label(df: pd.DataFrame, lock_gestures: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["lock_label"] = np.where(out["major_gesture"].isin(lock_gestures), "LOCK_REQUIRED", "NO_LOCK_REQUIRED")
    return out


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols)
    ]
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _build_candidate_models(cfg: dict) -> dict[str, Any]:
    m_cfg = cfg.get("model", {})
    random_seed = int(m_cfg.get("random_seed", 42))
    candidates: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=int(m_cfg.get("rf_estimators", 500)),
            max_depth=None if str(m_cfg.get("rf_max_depth", "20")).lower() == "none" else int(m_cfg.get("rf_max_depth", 20)),
            min_samples_leaf=int(m_cfg.get("rf_min_samples_leaf", 2)),
            max_features=m_cfg.get("rf_max_features", "sqrt"),
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=int(m_cfg.get("et_estimators", 800)),
            max_depth=None if str(m_cfg.get("et_max_depth", "none")).lower() == "none" else int(m_cfg.get("et_max_depth")),
            min_samples_leaf=int(m_cfg.get("et_min_samples_leaf", 1)),
            max_features=m_cfg.get("et_max_features", "sqrt"),
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=int(m_cfg.get("xgb_estimators", 800)),
            max_depth=int(m_cfg.get("xgb_max_depth", 6)),
            learning_rate=float(m_cfg.get("xgb_learning_rate", 0.04)),
            subsample=float(m_cfg.get("xgb_subsample", 0.9)),
            colsample_bytree=float(m_cfg.get("xgb_colsample_bytree", 0.8)),
            min_child_weight=float(m_cfg.get("xgb_min_child_weight", 1.0)),
            reg_lambda=float(m_cfg.get("xgb_reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )
    return candidates


def _make_pipeline(estimator: Any, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", estimator),
        ]
    )


def _score_probabilities(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[np.ndarray, dict, np.ndarray]:
    pred_num = (probabilities >= threshold).astype(np.int32)
    pred_labels = np.array([CLASS_NAME_MAP[int(v)] for v in pred_num], dtype=object)
    report = classification_report(
        y_true,
        pred_labels,
        labels=["LOCK_REQUIRED", "NO_LOCK_REQUIRED"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, pred_labels, labels=["LOCK_REQUIRED", "NO_LOCK_REQUIRED"])
    return pred_labels, report, cm


def _mean_probability(models: list[Pipeline], weights: list[float], x_df: pd.DataFrame) -> np.ndarray:
    total = np.zeros((len(x_df),), dtype=np.float64)
    denom = max(float(sum(weights)), 1e-6)
    for model, weight in zip(models, weights):
        total += float(weight) * model.predict_proba(x_df)[:, 1]
    return total / denom


def predict_bundle_probabilities(bundle: dict[str, Any], x_df: pd.DataFrame) -> np.ndarray:
    if str(bundle.get("model_kind", "single")) == "ensemble":
        models = list(bundle.get("models", []))
        weights = [float(x) for x in bundle.get("ensemble_weights", [])]
        return _mean_probability(models, weights, x_df)
    model = bundle["model"]
    return model.predict_proba(x_df)[:, int(bundle.get("lock_class_index", 1))]


def _trialwise_ema(probabilities: np.ndarray, trial_ids: np.ndarray, order_values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.asarray(probabilities, dtype=np.float64).copy()
    temp = pd.DataFrame(
        {
            "trial_id": trial_ids.astype(str),
            "order_value": order_values.astype(np.float64),
            "row_id": np.arange(len(out), dtype=np.int32),
        }
    ).sort_values(["trial_id", "order_value", "row_id"], kind="mergesort")
    for _, group in temp.groupby("trial_id", sort=False):
        row_ids = group["row_id"].to_numpy(dtype=np.int32)
        if row_ids.size == 0:
            continue
        values = out[row_ids]
        ema_values = np.empty_like(values, dtype=np.float64)
        ema_values[0] = values[0]
        for idx in range(1, len(values)):
            ema_values[idx] = float(alpha) * values[idx] + (1.0 - float(alpha)) * ema_values[idx - 1]
        out[row_ids] = ema_values
    return out


def apply_temporal_postprocess(
    probabilities: np.ndarray,
    trial_ids: np.ndarray,
    order_values: np.ndarray,
    temporal_cfg: dict[str, Any] | None,
) -> np.ndarray:
    cfg = dict(temporal_cfg or {})
    method = str(cfg.get("method", "none")).lower()
    raw = np.asarray(probabilities, dtype=np.float64)
    if method == "ema":
        alpha = float(cfg.get("alpha", 1.0))
        if not (0.0 < alpha <= 1.0):
            alpha = 1.0
        return _trialwise_ema(raw, np.asarray(trial_ids), np.asarray(order_values), alpha)
    return raw


def predict_bundle_labels(
    bundle: dict[str, Any],
    x_df: pd.DataFrame,
    trial_ids: np.ndarray | None = None,
    order_values: np.ndarray | None = None,
) -> np.ndarray:
    probabilities = predict_bundle_probabilities(bundle, x_df)
    temporal_cfg = dict(bundle.get("temporal_postprocess", {}))
    if temporal_cfg and str(temporal_cfg.get("method", "none")).lower() != "none":
        if trial_ids is None:
            trial_ids = np.array(["default_trial"] * len(probabilities), dtype=object)
        if order_values is None:
            order_values = np.arange(len(probabilities), dtype=np.float64)
        probabilities = apply_temporal_postprocess(probabilities, trial_ids, order_values, temporal_cfg)
    threshold = float(bundle.get("classification_threshold", 0.5))
    pred_num = (probabilities >= threshold).astype(np.int32)
    return np.array([CLASS_NAME_MAP[int(v)] for v in pred_num], dtype=object)


def _aggregate_feature_importance(models: list[Pipeline], weights: list[float]) -> tuple[list[str], dict[str, float]]:
    feature_names = models[0].named_steps["preprocessor"].get_feature_names_out().tolist()
    total = np.zeros((len(feature_names),), dtype=np.float64)
    denom = max(float(sum(weights)), 1e-6)
    for model, weight in zip(models, weights):
        clf = model.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            total += float(weight) * np.asarray(clf.feature_importances_, dtype=np.float64)
    total = total / denom
    fi = dict(sorted(zip(feature_names, total.tolist()), key=lambda kv: kv[1], reverse=True))
    return feature_names, fi


def _threshold_grid(cfg: dict) -> np.ndarray:
    cv_cfg = cfg.get("cross_validation", {})
    t_min = float(cv_cfg.get("threshold_min", 0.30))
    t_max = float(cv_cfg.get("threshold_max", 0.70))
    t_points = int(cv_cfg.get("threshold_points", 81))
    return np.linspace(t_min, t_max, t_points, dtype=np.float64)


def _build_cv_splitter(cfg: dict, groups: np.ndarray):
    cv_cfg = cfg.get("cross_validation", {})
    strategy = str(cv_cfg.get("strategy", "leave_one_surgeon_out")).lower()
    unique_groups = np.unique(groups)
    if strategy == "group_kfold":
        n_splits = int(cv_cfg.get("n_splits", min(5, len(unique_groups))))
        n_splits = max(2, min(n_splits, len(unique_groups)))
        return GroupKFold(n_splits=n_splits)
    return LeaveOneGroupOut()


def _optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray, threshold_grid: np.ndarray) -> tuple[float, np.ndarray, dict, np.ndarray]:
    best_threshold = 0.5
    best_pred: np.ndarray | None = None
    best_report: dict | None = None
    best_cm: np.ndarray | None = None
    best_score = (-1.0, -1.0)
    for threshold in threshold_grid:
        pred_labels, report, cm = _score_probabilities(y_true, probabilities, threshold=float(threshold))
        macro_f1 = float(report["macro avg"]["f1-score"])
        acc = float(report["accuracy"])
        score = (macro_f1, acc)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_pred = pred_labels
            best_report = report
            best_cm = cm
    if best_pred is None or best_report is None or best_cm is None:
        raise RuntimeError("Threshold optimization failed to produce predictions.")
    return best_threshold, best_pred, best_report, best_cm


def _task_metrics(task_values: np.ndarray, y_true: np.ndarray, pred_labels: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    accuracy_by_task: dict[str, float] = {}
    auc_by_task: dict[str, float] = {}
    for task_name in sorted(set(task_values.tolist())):
        mask = task_values == task_name
        accuracy_by_task[str(task_name)] = float(np.mean(pred_labels[mask] == y_true[mask]))
        y_task_num = np.array([INV_CLASS_NAME_MAP[str(x)] for x in y_true[mask]], dtype=np.int32)
        if len(np.unique(y_task_num)) >= 2:
            auc_by_task[str(task_name)] = float(roc_auc_score(y_task_num, probabilities[mask]))
    return accuracy_by_task, auc_by_task


def _search_spaces(cfg: dict, random_seed: int) -> dict[str, tuple[dict[str, list[Any]], int, dict[str, Any]]]:
    m_cfg = cfg.get("model", {})
    s_cfg = cfg.get("search", {})
    return {
        "random_forest": (
            {
                "n_estimators": [400, 600, 800, 1000],
                "max_depth": [None, 16, 24, 32],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 4, 8],
                "max_features": ["sqrt", 0.6, 0.8],
                "random_state": [11, 23, random_seed, 77],
            },
            int(s_cfg.get("rf_trials", 12)),
            {
                "n_estimators": int(m_cfg.get("rf_estimators", 500)),
                "max_depth": None if str(m_cfg.get("rf_max_depth", "20")).lower() == "none" else int(m_cfg.get("rf_max_depth", 20)),
                "min_samples_leaf": int(m_cfg.get("rf_min_samples_leaf", 2)),
                "min_samples_split": 2,
                "max_features": m_cfg.get("rf_max_features", "sqrt"),
                "random_state": random_seed,
            },
        ),
        "extra_trees": (
            {
                "n_estimators": [500, 800, 1000, 1200],
                "max_depth": [None, 24, 40],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 4, 8],
                "max_features": ["sqrt", "log2", 0.6, 0.8],
                "random_state": [13, 29, random_seed, 101],
            },
            int(s_cfg.get("et_trials", 16)),
            {
                "n_estimators": int(m_cfg.get("et_estimators", 900)),
                "max_depth": None if str(m_cfg.get("et_max_depth", "none")).lower() == "none" else int(m_cfg.get("et_max_depth")),
                "min_samples_leaf": int(m_cfg.get("et_min_samples_leaf", 1)),
                "min_samples_split": 2,
                "max_features": m_cfg.get("et_max_features", "sqrt"),
                "random_state": random_seed,
            },
        ),
    }


def _xgb_defaults_and_space(cfg: dict, random_seed: int) -> tuple[dict[str, list[Any]], int, dict[str, Any]] | None:
    if XGBClassifier is None:
        return None
    m_cfg = cfg.get("model", {})
    s_cfg = cfg.get("search", {})
    return (
        {
            "n_estimators": [300, 500, 700, 900],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.02, 0.03, 0.05, 0.07],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1.0, 3.0, 5.0],
            "gamma": [0.0, 0.1, 0.3],
            "reg_lambda": [1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "random_state": [17, 31, random_seed, 73],
        },
        int(s_cfg.get("xgb_trials", 18)),
        {
            "n_estimators": int(m_cfg.get("xgb_estimators", 800)),
            "max_depth": int(m_cfg.get("xgb_max_depth", 6)),
            "learning_rate": float(m_cfg.get("xgb_learning_rate", 0.04)),
            "subsample": float(m_cfg.get("xgb_subsample", 0.9)),
            "colsample_bytree": float(m_cfg.get("xgb_colsample_bytree", 0.8)),
            "min_child_weight": float(m_cfg.get("xgb_min_child_weight", 1.0)),
            "gamma": 0.0,
            "reg_lambda": float(m_cfg.get("xgb_reg_lambda", 1.0)),
            "reg_alpha": 0.0,
            "random_state": random_seed,
        },
    )


def _candidate_param_sets(cfg: dict) -> list[dict[str, Any]]:
    m_cfg = cfg.get("model", {})
    random_seed = int(m_cfg.get("random_seed", 42))
    search_spaces = _search_spaces(cfg, random_seed)
    xgb_space = _xgb_defaults_and_space(cfg, random_seed)
    if xgb_space is not None:
        search_spaces["xgboost"] = xgb_space

    requested = [str(x) for x in m_cfg.get("candidate_models", ["random_forest", "extra_trees", "xgboost"])]
    results: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    sampler_seed = int(cfg.get("search", {}).get("random_seed", random_seed))
    for model_name in requested:
        if model_name not in search_spaces:
            continue
        space, n_iter, defaults = search_spaces[model_name]
        candidates = [defaults]
        if n_iter > 0:
            sampled = list(ParameterSampler(space, n_iter=n_iter, random_state=sampler_seed + len(results) * 11 + 7))
            candidates.extend(sampled)
        for params in candidates:
            key = (model_name, tuple(sorted((str(k), str(v)) for k, v in params.items())))
            if key in seen:
                continue
            seen.add(key)
            results.append({"model_name": model_name, "params": dict(params)})
    return results


def _build_estimator(model_name: str, params: dict[str, Any], y_train_num: np.ndarray | None = None) -> Any:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=int(params["random_state"]),
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=int(params["random_state"]),
            n_jobs=-1,
        )
    if model_name == "xgboost" and XGBClassifier is not None:
        scale_pos_weight = 1.0
        if y_train_num is not None:
            positives = float(np.sum(y_train_num == 1))
            negatives = float(np.sum(y_train_num == 0))
            if positives > 0:
                scale_pos_weight = negatives / positives
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params.get("gamma", 0.0)),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_bin=256,
            random_state=int(params["random_state"]),
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def _evaluate_candidate_oof(
    candidate: dict[str, Any],
    x_df: pd.DataFrame,
    y_labels: np.ndarray,
    y_num: np.ndarray,
    groups: np.ndarray,
    tasks: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict,
) -> dict[str, Any]:
    splitter = _build_cv_splitter(cfg, groups)
    oof_probabilities = np.zeros((len(x_df),), dtype=np.float64)
    fold_assign = np.full((len(x_df),), -1, dtype=np.int32)
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x_df, y_labels, groups=groups)):
        estimator = _build_estimator(candidate["model_name"], candidate["params"], y_train_num=y_num[train_idx])
        pipeline = _make_pipeline(estimator, numeric_cols, categorical_cols)
        pipeline.fit(x_df.iloc[train_idx], y_num[train_idx])
        oof_probabilities[valid_idx] = pipeline.predict_proba(x_df.iloc[valid_idx])[:, 1]
        fold_assign[valid_idx] = fold_idx

    threshold, pred_labels, report, cm = _optimize_threshold(y_labels, oof_probabilities, _threshold_grid(cfg))
    auc = float(roc_auc_score(y_num, oof_probabilities)) if len(np.unique(y_num)) >= 2 else 0.0
    task_accuracy, task_auc = _task_metrics(tasks, y_labels, pred_labels, oof_probabilities)
    return {
        "candidate_id": f"{candidate['model_name']}__{abs(hash(tuple(sorted(candidate['params'].items())))) % 10_000_000}",
        "model_name": candidate["model_name"],
        "params": dict(candidate["params"]),
        "threshold": float(threshold),
        "pred_labels": pred_labels,
        "probabilities": oof_probabilities,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "task_accuracy": task_accuracy,
        "task_auc": task_auc,
        "fold_assign": fold_assign,
        "score_tuple": (
            float(report["macro avg"]["f1-score"]),
            auc,
            float(report["accuracy"]),
        ),
    }


def _greedy_ensemble_selection(
    candidate_results: list[dict[str, Any]],
    y_labels: np.ndarray,
    y_num: np.ndarray,
    tasks: np.ndarray,
    trial_ids: np.ndarray,
    order_values: np.ndarray,
    cfg: dict,
) -> dict[str, Any] | None:
    if not candidate_results:
        return None
    s_cfg = cfg.get("search", {})
    top_k = int(s_cfg.get("ensemble_top_k", 8))
    max_members = int(s_cfg.get("ensemble_max_members", 6))
    min_gain = float(s_cfg.get("ensemble_min_gain", 1e-4))
    ranked = sorted(candidate_results, key=lambda x: x["score_tuple"], reverse=True)[:top_k]
    threshold_grid = _threshold_grid(cfg)

    selected_ids: list[str] = []
    current_probabilities: np.ndarray | None = None
    current_eval: dict[str, Any] | None = None

    for _ in range(max_members):
        step_best: dict[str, Any] | None = None
        for cand in ranked:
            cand_probs = cand["probabilities"]
            if current_probabilities is None:
                trial_probs = cand_probs
            else:
                member_count = max(len(selected_ids), 1)
                trial_probs = (current_probabilities * member_count + cand_probs) / float(member_count + 1)
            postprocessed = _select_temporal_postprocess(
                probabilities=trial_probs,
                y_labels=y_labels,
                y_num=y_num,
                tasks=tasks,
                trial_ids=trial_ids,
                order_values=order_values,
                cfg=cfg,
            )
            score_tuple = postprocessed["score_tuple"]
            candidate_eval = {
                "added_candidate_id": cand["candidate_id"],
                "raw_probabilities": trial_probs,
                "probabilities": postprocessed["probabilities"],
                "threshold": float(postprocessed["threshold"]),
                "pred_labels": postprocessed["pred_labels"],
                "classification_report": postprocessed["classification_report"],
                "confusion_matrix": postprocessed["confusion_matrix"],
                "roc_auc": postprocessed["roc_auc"],
                "task_accuracy": postprocessed["task_accuracy"],
                "task_auc": postprocessed["task_auc"],
                "temporal_postprocess": dict(postprocessed["temporal_postprocess"]),
                "score_tuple": score_tuple,
            }
            if step_best is None or candidate_eval["score_tuple"] > step_best["score_tuple"]:
                step_best = candidate_eval
        if step_best is None:
            break
        if current_eval is not None and (step_best["score_tuple"][0] - current_eval["score_tuple"][0]) < min_gain:
            break
        selected_ids.append(step_best["added_candidate_id"])
        current_probabilities = step_best["probabilities"]
        current_eval = step_best

    if current_eval is None:
        return None
    weight_counter = Counter(selected_ids)
    return {
        "selected_candidate_ids": selected_ids,
        "weights": {cid: float(weight_counter[cid]) for cid in weight_counter},
        "threshold": current_eval["threshold"],
        "pred_labels": current_eval["pred_labels"],
        "probabilities": current_eval["probabilities"],
        "classification_report": current_eval["classification_report"],
        "confusion_matrix": current_eval["confusion_matrix"],
        "roc_auc": current_eval["roc_auc"],
        "task_accuracy": current_eval["task_accuracy"],
        "task_auc": current_eval["task_auc"],
        "temporal_postprocess": current_eval["temporal_postprocess"],
        "score_tuple": current_eval["score_tuple"],
    }


def _postprocess_candidates(cfg: dict) -> list[dict[str, Any]]:
    post_cfg = cfg.get("postprocess", {})
    if not bool(post_cfg.get("enable", True)):
        return [{"method": "none"}]
    methods = [str(x).lower() for x in post_cfg.get("methods", ["none", "ema"])]
    candidates: list[dict[str, Any]] = [{"method": "none"}]
    if "ema" in methods:
        alpha_min = float(post_cfg.get("ema_alpha_min", 0.45))
        alpha_max = float(post_cfg.get("ema_alpha_max", 0.75))
        alpha_points = int(post_cfg.get("ema_alpha_points", 13))
        for alpha in np.linspace(alpha_min, alpha_max, alpha_points, dtype=np.float64):
            candidates.append({"method": "ema", "alpha": float(alpha)})
    return candidates


def _select_temporal_postprocess(
    probabilities: np.ndarray,
    y_labels: np.ndarray,
    y_num: np.ndarray,
    tasks: np.ndarray,
    trial_ids: np.ndarray,
    order_values: np.ndarray,
    cfg: dict,
) -> dict[str, Any]:
    threshold_grid = _threshold_grid(cfg)
    best_result: dict[str, Any] | None = None
    for candidate in _postprocess_candidates(cfg):
        processed_probabilities = apply_temporal_postprocess(probabilities, trial_ids, order_values, candidate)
        threshold, pred_labels, report, cm = _optimize_threshold(y_labels, processed_probabilities, threshold_grid)
        auc = float(roc_auc_score(y_num, processed_probabilities)) if len(np.unique(y_num)) >= 2 else 0.0
        task_accuracy, task_auc = _task_metrics(tasks, y_labels, pred_labels, processed_probabilities)
        result = {
            "temporal_postprocess": candidate,
            "probabilities": processed_probabilities,
            "threshold": float(threshold),
            "pred_labels": pred_labels,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
            "task_accuracy": task_accuracy,
            "task_auc": task_auc,
            "score_tuple": (
                float(report["macro avg"]["f1-score"]),
                float(report["accuracy"]),
                auc,
            ),
        }
        if best_result is None or result["score_tuple"] > best_result["score_tuple"]:
            best_result = result
    if best_result is None:
        raise RuntimeError("Temporal postprocess search failed.")
    return best_result

def train_model(df: pd.DataFrame, cfg: dict) -> tuple[dict[str, Any], dict, pd.DataFrame, str, dict[str, float]]:
    m_cfg = cfg.get("model", {})
    use_task_context = bool(m_cfg.get("use_task_context", True))
    numeric_cols = list(FEATURE_ORDER)
    categorical_cols = ["task"] if use_task_context else []
    feature_cols = numeric_cols + categorical_cols

    x_df = df[feature_cols].copy()
    y_labels = df["lock_label"].to_numpy(dtype=object)
    y_num = np.array([INV_CLASS_NAME_MAP[str(label)] for label in y_labels], dtype=np.int32)
    groups = df["surgeon"].to_numpy(dtype=object)
    tasks = df["task"].astype(str).to_numpy()
    trial_ids = df["trial_id"].astype(str).to_numpy()
    order_values = df["start"].to_numpy(dtype=np.float64)

    candidate_specs = _candidate_param_sets(cfg)
    if not candidate_specs:
        raise RuntimeError("No candidate parameter sets were generated.")

    candidate_results: list[dict[str, Any]] = []
    candidate_post_results: dict[str, dict[str, Any]] = {}
    candidate_scores: dict[str, dict[str, Any]] = {}
    total_candidates = len(candidate_specs)
    print(f"Training-only grouped CV search started. Candidates: {total_candidates}", flush=True)
    for idx, candidate in enumerate(candidate_specs, start=1):
        print(f"[{idx}/{total_candidates}] evaluating {candidate['model_name']} with params={candidate['params']}", flush=True)
        result = _evaluate_candidate_oof(
            candidate=candidate,
            x_df=x_df,
            y_labels=y_labels,
            y_num=y_num,
            groups=groups,
            tasks=tasks,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            cfg=cfg,
        )
        candidate_results.append(result)
        postprocessed = _select_temporal_postprocess(
            probabilities=result["probabilities"],
            y_labels=y_labels,
            y_num=y_num,
            tasks=tasks,
            trial_ids=trial_ids,
            order_values=order_values,
            cfg=cfg,
        )
        candidate_post_results[result["candidate_id"]] = postprocessed
        candidate_scores[result["candidate_id"]] = {
            "model_name": result["model_name"],
            "raw_threshold": result["threshold"],
            "raw_macro_f1": float(result["classification_report"]["macro avg"]["f1-score"]),
            "raw_accuracy": float(result["classification_report"]["accuracy"]),
            "raw_roc_auc": float(result["roc_auc"]),
            "threshold": float(postprocessed["threshold"]),
            "macro_f1": float(postprocessed["classification_report"]["macro avg"]["f1-score"]),
            "accuracy": float(postprocessed["classification_report"]["accuracy"]),
            "roc_auc": float(postprocessed["roc_auc"]),
            "temporal_postprocess": dict(postprocessed["temporal_postprocess"]),
            "params": result["params"],
        }
        print(
            f"    -> macro_f1={candidate_scores[result['candidate_id']]['macro_f1']:.4f}, "
            f"auc={candidate_scores[result['candidate_id']]['roc_auc']:.4f}, "
            f"threshold={candidate_scores[result['candidate_id']]['threshold']:.3f}",
            flush=True,
        )

    best_single = max(candidate_results, key=lambda x: candidate_post_results[x["candidate_id"]]["score_tuple"])
    best_single_post = candidate_post_results[best_single["candidate_id"]]
    ensemble_result = _greedy_ensemble_selection(
        candidate_results,
        y_labels,
        y_num,
        tasks,
        trial_ids,
        order_values,
        cfg,
    )

    selected_kind = "single"
    selected_name = best_single["model_name"]
    selected_result = best_single_post
    selected_candidate_ids = [best_single["candidate_id"]]
    selected_weight_map = {best_single["candidate_id"]: 1.0}
    if ensemble_result is not None and ensemble_result["score_tuple"] > best_single_post["score_tuple"]:
        selected_kind = "ensemble"
        selected_name = "group_cv_greedy_ensemble"
        selected_result = ensemble_result
        selected_candidate_ids = list(ensemble_result["selected_candidate_ids"])
        selected_weight_map = dict(ensemble_result["weights"])
        candidate_scores[selected_name] = {
            "model_name": "ensemble",
            "threshold": float(ensemble_result["threshold"]),
            "macro_f1": float(ensemble_result["classification_report"]["macro avg"]["f1-score"]),
            "accuracy": float(ensemble_result["classification_report"]["accuracy"]),
            "roc_auc": float(ensemble_result["roc_auc"]),
            "temporal_postprocess": dict(ensemble_result["temporal_postprocess"]),
            "members": dict(selected_weight_map),
        }

    selected_threshold = float(selected_result["threshold"])
    selected_probabilities = np.asarray(selected_result["probabilities"], dtype=np.float64)
    selected_pred_labels = np.asarray(selected_result["pred_labels"], dtype=object)
    selected_report = dict(selected_result["classification_report"])
    selected_cm = np.asarray(selected_result["confusion_matrix"])
    selected_task_accuracy = dict(selected_result["task_accuracy"])
    selected_task_auc = dict(selected_result["task_auc"])
    temporal_cfg = dict(selected_result["temporal_postprocess"])

    candidate_lookup = {result["candidate_id"]: result for result in candidate_results}
    if selected_kind == "ensemble":
        final_models: list[Pipeline] = []
        final_weights: list[float] = []
        for candidate_id, weight in selected_weight_map.items():
            candidate = candidate_lookup[candidate_id]
            estimator = _build_estimator(candidate["model_name"], candidate["params"], y_train_num=y_num)
            pipeline = _make_pipeline(estimator, numeric_cols, categorical_cols)
            pipeline.fit(x_df, y_num)
            final_models.append(pipeline)
            final_weights.append(float(weight))
        final_bundle = {
            "model_kind": "ensemble",
            "models": final_models,
            "ensemble_weights": final_weights,
            "ensemble_members": selected_candidate_ids,
            "classification_threshold": selected_threshold,
            "lock_class_index": 1,
            "context_columns": categorical_cols,
            "input_feature_columns": feature_cols,
            "raw_feature_names": list(FEATURE_ORDER),
            "classes": ["NO_LOCK_REQUIRED", "LOCK_REQUIRED"],
            "class_name_map": CLASS_NAME_MAP,
            "temporal_postprocess": temporal_cfg,
        }
        feature_names, feature_importance = _aggregate_feature_importance(final_models, final_weights)
    else:
        estimator = _build_estimator(best_single["model_name"], best_single["params"], y_train_num=y_num)
        final_model = _make_pipeline(estimator, numeric_cols, categorical_cols)
        final_model.fit(x_df, y_num)
        final_bundle = {
            "model_kind": "single",
            "model": final_model,
            "classification_threshold": selected_threshold,
            "lock_class_index": 1,
            "context_columns": categorical_cols,
            "input_feature_columns": feature_cols,
            "raw_feature_names": list(FEATURE_ORDER),
            "classes": ["NO_LOCK_REQUIRED", "LOCK_REQUIRED"],
            "class_name_map": CLASS_NAME_MAP,
            "temporal_postprocess": temporal_cfg,
        }
        feature_names, feature_importance = _aggregate_feature_importance([final_model], [1.0])

    df_out = df.copy()
    df_out["split"] = "cv_oof"
    df_out["pred_label"] = selected_pred_labels
    df_out["proba_lock"] = selected_probabilities
    df_out["holdout_pred_label"] = ""
    df_out["holdout_proba_lock"] = np.nan

    fold_meta = {
        "train_windows": int(len(df_out)),
        "test_windows": 0,
        "validation_windows": int(len(df_out)),
        "validation_surgeons": sorted(set(df_out["surgeon"].tolist())),
        "labels": ["LOCK_REQUIRED", "NO_LOCK_REQUIRED"],
        "classification_report": selected_report,
        "confusion_matrix": selected_cm.tolist(),
        "candidate_scores": candidate_scores,
        "best_model_name": selected_name,
        "best_single_model_name": best_single["model_name"],
        "selection_strategy": "train_only_group_cv_oof_then_full_refit_no_test_peek",
        "validation_task_accuracy": selected_task_accuracy,
        "test_task_accuracy": selected_task_accuracy,
        "roc_auc_overall": float(selected_result["roc_auc"]),
        "validation_roc_auc_overall": float(selected_result["roc_auc"]),
        "roc_auc_by_task": selected_task_auc,
        "validation_roc_auc_by_task": selected_task_auc,
        "classification_threshold": selected_threshold,
        "selected_candidate_ids": selected_candidate_ids,
        "temporal_postprocess": temporal_cfg,
    }
    final_bundle["feature_names"] = feature_names
    final_bundle["selection_strategy"] = fold_meta["selection_strategy"]
    return final_bundle, fold_meta, df_out, selected_name, feature_importance


def add_decision_logic(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    d_cfg = cfg.get("decision", {})
    proba_thr = float(d_cfg.get("proba_lock_threshold", 0.58))
    pctl = float(d_cfg.get("stable_velocity_percentile_from_lock", 75))

    lock_df = df[df["lock_label"] == "LOCK_REQUIRED"]
    if lock_df.empty:
        stable_vel_thr = float(df["global_vel_mean"].quantile(0.4))
    else:
        stable_vel_thr = float(np.percentile(lock_df["global_vel_mean"].to_numpy(), pctl))

    out = df.copy()
    out["lock_decision"] = np.where(
        (out["proba_lock"] >= proba_thr) & (out["global_vel_mean"] <= stable_vel_thr),
        "SUGGEST_LOCK",
        "SUGGEST_NO_LOCK",
    )
    out["decision_reason"] = np.where(
        out["lock_decision"] == "SUGGEST_LOCK",
        "高锁定概率且全局速度低（精细稳态）",
        "锁定概率不足或全局速度偏高（重定位/大运动）",
    )

    decision_stats = {
        "proba_lock_threshold": proba_thr,
        "stable_velocity_threshold": stable_vel_thr,
        "suggest_lock_ratio": float(np.mean(out["lock_decision"] == "SUGGEST_LOCK")),
    }
    return out, decision_stats


def build_policy_summary(df: pd.DataFrame, decision_stats: dict) -> dict:
    lock_rows = df[df["lock_decision"] == "SUGGEST_LOCK"]
    nolock_rows = df[df["lock_decision"] == "SUGGEST_NO_LOCK"]

    def _safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if not s.empty else 0.0

    policy = {
        "need_lock_when": {
            "global_vel_mean_leq": decision_stats["stable_velocity_threshold"],
            "proba_lock_geq": decision_stats["proba_lock_threshold"],
            "typical_micro_motion_ratio": _safe_mean(lock_rows["micro_motion_ratio"]),
            "typical_large_motion_ratio": _safe_mean(lock_rows["large_motion_ratio"]),
            "clinical_interpretation": "精细操作阶段、姿态已收敛、动作以微调为主，建议锁定辅助支撑。",
        },
        "no_lock_when": {
            "global_vel_mean_gt": decision_stats["stable_velocity_threshold"],
            "or_proba_lock_lt": decision_stats["proba_lock_threshold"],
            "typical_micro_motion_ratio": _safe_mean(nolock_rows["micro_motion_ratio"]),
            "typical_large_motion_ratio": _safe_mean(nolock_rows["large_motion_ratio"]),
            "clinical_interpretation": "大幅重定位、手眼协同换位或动作不稳定阶段，不建议锁定。",
        },
    }
    return policy


def task_accuracy(df: pd.DataFrame, split: str = "test") -> dict[str, float]:
    subset = df[df["split"] == split]
    out: dict[str, float] = {}
    for task, g in subset.groupby("task"):
        out[str(task)] = float(np.mean(g["lock_label"] == g["pred_label"]))
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train JIGSAWS surgical intent model and output lock/no-lock decisions."
    )
    parser.add_argument("--config", default="config/jigsaws_intent.toml")
    parser.add_argument("--output-dir", default="logs/jigsaws_intent")
    parser.add_argument("--model-out", default="models/jigsaws_intent_model.joblib")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--predictions-out", default="")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    model_out = Path(args.model_out)
    ensure_dir(model_out.parent)

    metrics_out = Path(args.metrics_out) if args.metrics_out else out_dir / "metrics.json"
    preds_out = Path(args.predictions_out) if args.predictions_out else out_dir / "window_predictions.csv"

    trials = load_jigsaws_trials(cfg)
    if not trials:
        raise RuntimeError("No JIGSAWS trials loaded. Check dataset root and folder layout.")

    df = build_window_table(trials, cfg)
    if df.empty:
        raise RuntimeError("No valid windows generated from JIGSAWS.")

    l_cfg = cfg.get("labeling", {})
    lock_set, gesture_stats = discover_lock_gestures(
        df=df,
        lock_quantile=float(l_cfg.get("lock_quantile", 0.45)),
        min_windows_per_gesture=int(l_cfg.get("min_windows_per_gesture", 50)),
        manual_lock=[str(x) for x in l_cfg.get("manual_lock_gestures", [])],
        manual_no_lock=[str(x) for x in l_cfg.get("manual_no_lock_gestures", [])],
        auto_discover=bool(l_cfg.get("auto_discover_lock_gestures", True)),
    )
    df = apply_lock_label(df, lock_set)

    bundle, fold_meta, df_pred, best_model_name, feature_importance = train_model(df, cfg)
    df_pred, decision_stats = add_decision_logic(df_pred, cfg)
    policy = build_policy_summary(df_pred, decision_stats)
    bundle["decision_stats"] = decision_stats
    bundle["lock_gestures"] = sorted(lock_set)
    bundle["best_model_name"] = best_model_name
    bundle["default_task"] = str(cfg.get("dataset", {}).get("tasks", ["Suturing"])[0])
    bundle["config"] = cfg
    joblib.dump(bundle, model_out)

    df_pred.to_csv(preds_out, index=False)

    metrics = {
        "loaded_trials": int(len(trials)),
        "window_count": int(len(df_pred)),
        "tasks": sorted(df_pred["task"].unique().tolist()),
        "surgeons": sorted(df_pred["surgeon"].unique().tolist()),
        "lock_gestures": sorted(lock_set),
        "gesture_motion_stats": gesture_stats.to_dict(orient="records"),
        "lock_label_distribution": df_pred["lock_label"].value_counts().to_dict(),
        "split_meta": fold_meta,
        "test_task_accuracy": fold_meta.get("test_task_accuracy", {}),
        "decision_stats": decision_stats,
        "policy_summary": policy,
        "feature_importance": feature_importance,
        "best_model_name": best_model_name,
        "example_rules": [
            "当模型锁定概率高且全局运动速度低，判为建议锁定。",
            "当模型锁定概率不足或全局运动速度高，判为不建议锁定。",
            "建议锁定多见于精细操作稳态；不建议锁定多见于换位与大运动阶段。",
        ],
    }

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("JIGSAWS intent training finished.")
    print(f"Trials loaded: {len(trials)}")
    print(f"Windows: {len(df_pred)}")
    print("Lock gestures:", ", ".join(sorted(lock_set)))
    print(f"Model saved: {model_out}")
    print(f"Metrics saved: {metrics_out}")
    print(f"Predictions saved: {preds_out}")
    print("Policy summary:")
    print(json.dumps(policy, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
