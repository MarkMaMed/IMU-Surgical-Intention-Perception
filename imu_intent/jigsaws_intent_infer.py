from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .jigsaws_intent_program import (
    FEATURE_ORDER,
    apply_temporal_postprocess,
    extract_window_feature_dict,
    parse_transcription,
    predict_bundle_labels,
    predict_bundle_probabilities,
)


def load_trial(kinematics: Path, transcriptions: Path | None) -> tuple[np.ndarray, np.ndarray]:
    signals = np.loadtxt(kinematics, dtype=float)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    if transcriptions is None or not transcriptions.exists():
        gestures = np.array(["G0"] * signals.shape[0], dtype=object)
    else:
        gestures = parse_transcription(transcriptions, signals.shape[0])
    return signals.astype(np.float32), gestures


def infer_task_name(kinematics: Path, fallback: str) -> str:
    stem = kinematics.stem
    for task_name in ["Knot_Tying", "Needle_Passing", "Suturing"]:
        if task_name in stem:
            return task_name
    for part in kinematics.parts:
        if part in {"Knot_Tying", "Needle_Passing", "Suturing"}:
            return part
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for JIGSAWS lock/no-lock intent model.")
    parser.add_argument("--model", default="models/jigsaws_intent_model.joblib")
    parser.add_argument("--kinematics", required=True)
    parser.add_argument("--transcriptions", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--output", default="logs/jigsaws_intent_infer.csv")
    parser.add_argument("--summary", default="logs/jigsaws_intent_infer_summary.json")
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model))
    decision_stats = dict(bundle.get("decision_stats", {}))
    raw_feature_names = list(bundle.get("raw_feature_names", FEATURE_ORDER))
    input_feature_columns = list(bundle.get("input_feature_columns", raw_feature_names))
    context_columns = list(bundle.get("context_columns", []))
    bundle_window_cfg = dict(bundle.get("config", {}).get("window", {}))
    window_size = int(args.window_size or bundle_window_cfg.get("size", 90))
    stride = int(args.stride or bundle_window_cfg.get("stride", 30))

    kin_path = Path(args.kinematics)
    trans_path = Path(args.transcriptions) if args.transcriptions else None
    signals, gestures = load_trial(kin_path, trans_path)
    task_name = args.task or infer_task_name(kin_path, str(bundle.get("default_task", "Suturing")))

    rows: list[dict] = []
    feature_rows: list[dict] = []
    for start in range(0, signals.shape[0] - window_size + 1, stride):
        end = start + window_size
        feat = extract_window_feature_dict(signals[start:end])
        row_features = {name: feat.get(name, 0.0) for name in raw_feature_names}
        for column in context_columns:
            if column == "task":
                row_features[column] = task_name
        feature_rows.append(row_features)
        g = gestures[start:end]
        g = [x for x in g.tolist() if x != "G0"]
        major_gesture = str(pd.Series(g).mode().iloc[0]) if g else "G0"
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "task": task_name,
                "major_gesture": major_gesture,
                "pred_label": "",
                "proba_lock": 0.0,
                "global_vel_mean": feat["global_vel_mean"],
                "micro_motion_ratio": feat["micro_motion_ratio"],
                "large_motion_ratio": feat["large_motion_ratio"],
                "lock_decision": "",
                "decision_reason": "",
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        x_df = pd.DataFrame(feature_rows, columns=input_feature_columns)
        trial_ids = np.array([kin_path.stem] * len(out_df), dtype=object)
        order_values = out_df["window_start"].to_numpy(dtype=np.float64)
        raw_probas = predict_bundle_probabilities(bundle, x_df)
        probas = apply_temporal_postprocess(
            raw_probas,
            trial_ids=trial_ids,
            order_values=order_values,
            temporal_cfg=dict(bundle.get("temporal_postprocess", {})),
        )
        pred_labels = predict_bundle_labels(bundle, x_df, trial_ids=trial_ids, order_values=order_values)
        stable_velocity_threshold = float(decision_stats.get("stable_velocity_threshold", 0.02))
        proba_threshold = float(decision_stats.get("proba_lock_threshold", 0.58))
        suggest_lock = (probas >= proba_threshold) & (
            out_df["global_vel_mean"].to_numpy(dtype=float) <= stable_velocity_threshold
        )
        out_df["pred_label"] = pred_labels
        out_df["proba_lock"] = probas
        out_df["lock_decision"] = np.where(suggest_lock, "SUGGEST_LOCK", "SUGGEST_NO_LOCK")
        out_df["decision_reason"] = np.where(
            suggest_lock,
            "高锁定概率且速度低，建议锁定",
            "锁定概率不足或速度偏高，不建议锁定",
        )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = {
        "kinematics": str(kin_path),
        "task": task_name,
        "windows": int(len(out_df)),
        "decision_counts": out_df["lock_decision"].value_counts().to_dict() if not out_df.empty else {},
        "avg_proba_lock": float(out_df["proba_lock"].mean()) if not out_df.empty else 0.0,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Inference saved: {out_path}")
    print(f"Summary saved: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
