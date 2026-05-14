from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import extract_window_features

ALIASES = {
    "acc_x": ["acc_x", "ax", "body_acc_x", "IMU_hand_acceleration_16g_1"],
    "acc_y": ["acc_y", "ay", "body_acc_y", "IMU_hand_acceleration_16g_2"],
    "acc_z": ["acc_z", "az", "body_acc_z", "IMU_hand_acceleration_16g_3"],
    "gyro_x": ["gyro_x", "gx", "body_gyro_x", "IMU_hand_gyroscope_1"],
    "gyro_y": ["gyro_y", "gy", "body_gyro_y", "IMU_hand_gyroscope_2"],
    "gyro_z": ["gyro_z", "gz", "body_gyro_z", "IMU_hand_gyroscope_3"],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _to_canonical_signals(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        col = _find_column(df, ALIASES[key])
        if col is None:
            raise ValueError(f"Missing required column for {key}. Accepted aliases: {ALIASES[key]}")
        cols.append(col)
    return df[cols].to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer surgical intent from IMU CSV.")
    parser.add_argument("--model", default="models/imu_intent_multidataset.joblib")
    parser.add_argument("--input", required=True, help="CSV with IMU channels (acc_x/y/z, gyro_x/y/z or aliases).")
    parser.add_argument("--output", default="logs/imu_intent_predictions.csv")
    parser.add_argument("--summary", default="logs/imu_intent_predictions_summary.json")
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    default_ws = int(bundle.get("window_size", 128))
    default_stride = int(bundle.get("stride", 32))
    ws = int(args.window_size) if args.window_size > 0 else default_ws
    stride = int(args.stride) if args.stride > 0 else default_stride

    df = pd.read_csv(args.input)
    signals = _to_canonical_signals(df)
    if signals.shape[0] < ws:
        raise ValueError(f"Input length {signals.shape[0]} is smaller than window_size={ws}")

    rows = []
    starts = []
    feats = []
    counts: dict[str, int] = {}
    for start in range(0, signals.shape[0] - ws + 1, stride):
        end = start + ws
        starts.append((start, end))
        feats.append(extract_window_features(signals[start:end]))

    if not feats:
        raise ValueError("No windows generated from input.")

    feat_mat = np.vstack(feats)
    preds = model.predict(feat_mat)
    probas = model.predict_proba(feat_mat)

    for i, (start, end) in enumerate(starts):
        pred = str(preds[i])
        conf = float(np.max(probas[i]))
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "pred_intent": pred,
                "confidence": conf,
                "candidate_lock": int(pred == "PREPARE_LOCK"),
                "candidate_unlock": int(pred == "PREPARE_UNLOCK"),
            }
        )
        counts[pred] = counts.get(pred, 0) + 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    summary = {
        "input": args.input,
        "window_size": ws,
        "stride": stride,
        "prediction_count": len(rows),
        "intent_count": counts,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {out_path}")
    print(f"Summary saved to: {summary_path}")
    print("Intent distribution:", counts)


if __name__ == "__main__":
    main()
