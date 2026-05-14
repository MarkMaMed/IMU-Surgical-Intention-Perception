from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .features import feature_names
from .loaders import CANONICAL_CHANNELS, load_all_enabled_records
from .synthetic import make_synthetic_records
from .windowing import build_window_dataset


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("rb") as f:
        return tomllib.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train surgical intent model from JIGSAWS/Opportunity/NinaPro/PAMAP2.")
    parser.add_argument("--config", default="config/imu_multidataset.toml")
    parser.add_argument("--use-synthetic-if-empty", action="store_true")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--metrics-out", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    global_cfg = cfg.get("global", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})

    records = load_all_enabled_records(cfg)
    if not records and args.use_synthetic_if_empty:
        records = make_synthetic_records()

    if not records:
        raise RuntimeError(
            "No dataset records loaded. Please check dataset paths in config/imu_multidataset.toml "
            "or run with --use-synthetic-if-empty for a quick pipeline check."
        )

    X, y, meta = build_window_dataset(
        records=records,
        window_size=int(global_cfg.get("window_size", 128)),
        stride=int(global_cfg.get("stride", 32)),
        majority_ratio=float(global_cfg.get("majority_ratio", 0.6)),
        max_windows_per_sequence=int(global_cfg.get("max_windows_per_sequence", 0)),
    )
    if X.size == 0:
        raise RuntimeError("No valid windows generated. Adjust window/stride/majority_ratio or check label quality.")

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(global_cfg.get("test_size", 0.2)),
        random_state=int(global_cfg.get("random_seed", 42)),
        stratify=y,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 320)),
        max_depth=int(model_cfg.get("max_depth", 18)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        class_weight="balanced_subsample",
        random_state=int(global_cfg.get("random_seed", 42)),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels_sorted = sorted(list(set(y.tolist())))
    report = classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()

    dataset_perf = defaultdict(lambda: {"n": 0, "correct": 0})
    for i in test_idx.tolist():
        ds = meta[i]["dataset"]
        dataset_perf[ds]["n"] += 1
        dataset_perf[ds]["correct"] += int(y[i] == clf.predict(X[i : i + 1])[0])

    dataset_acc = {}
    for ds, info in dataset_perf.items():
        dataset_acc[ds] = float(info["correct"]) / max(1, int(info["n"]))

    source_counter = Counter(str(m.get("signal_source", "unknown")) for m in meta)
    dataset_source_counter = Counter(
        f"{m.get('dataset', 'NA')}::{m.get('signal_source', 'unknown')}" for m in meta
    )

    metrics = {
        "loaded_sequences": int(len(records)),
        "window_count": int(len(y)),
        "class_distribution": dict(Counter(y.tolist())),
        "labels": labels_sorted,
        "classification_report": report,
        "confusion_matrix": cm,
        "dataset_accuracy": dataset_acc,
        "signal_source_distribution": dict(source_counter),
        "dataset_signal_source_distribution": dict(dataset_source_counter),
    }

    model_path = Path(args.model_out or out_cfg.get("model_path", "models/imu_intent_multidataset.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "window_size": int(global_cfg.get("window_size", 128)),
        "stride": int(global_cfg.get("stride", 32)),
        "channels": CANONICAL_CHANNELS,
        "feature_names": feature_names(CANONICAL_CHANNELS),
        "labels": labels_sorted,
        "source_datasets": sorted(list({r.dataset for r in records})),
    }
    joblib.dump(bundle, model_path)

    metrics_path = Path(args.metrics_out or out_cfg.get("metrics_path", "logs/imu_intent_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("Dataset coverage:", ", ".join(bundle["source_datasets"]))
    print("Class distribution:", metrics["class_distribution"])
    print("Signal source distribution:", metrics["signal_source_distribution"])


if __name__ == "__main__":
    main()
