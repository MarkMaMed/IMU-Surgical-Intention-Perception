from __future__ import annotations

import argparse
import time

import joblib
import pandas as pd

from .features import extract_window_features
from .infer_from_csv import _to_canonical_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo realtime intent stream demo from IMU CSV.")
    parser.add_argument("--model", default="models/imu_intent_multidataset.joblib")
    parser.add_argument("--input", default="demo/mock_imu_stream.csv")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--sleep-ms", type=int, default=120)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]

    df = pd.read_csv(args.input)
    signals = _to_canonical_signals(df)
    for start in range(0, signals.shape[0] - args.window_size + 1, args.stride):
        end = start + args.window_size
        feat = extract_window_features(signals[start:end]).reshape(1, -1)
        pred = str(model.predict(feat)[0])
        conf = float(model.predict_proba(feat).max())
        msg = (
            f"[{start:04d}:{end:04d}] intent={pred:<15} conf={conf:.3f} "
            f"candidate_lock={int(pred == 'PREPARE_LOCK')} "
            f"candidate_unlock={int(pred == 'PREPARE_UNLOCK')}"
        )
        print(msg)
        time.sleep(max(0, args.sleep_ms) / 1000.0)


if __name__ == "__main__":
    main()

