from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timeline from infer_from_csv prediction output.")
    parser.add_argument("--predictions", required=True, help="Path to *_predictions.csv")
    parser.add_argument("--output", default="plots/imu_timeline.png")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    df = pd.read_csv(pred_path)
    required = {"window_start", "window_end", "pred_intent", "confidence"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Expected at least: {sorted(required)}")

    x = ((df["window_start"].to_numpy() + df["window_end"].to_numpy()) / 2.0).astype(float)
    intents = df["pred_intent"].astype(str)
    confidence = df["confidence"].astype(float)

    intent_names = sorted(intents.unique().tolist())
    y_map = {k: i for i, k in enumerate(intent_names)}
    y = intents.map(y_map).to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(x, y, drawstyle="steps-post", linewidth=1.8, color="#4C78A8")
    ax1.set_yticks(list(y_map.values()))
    ax1.set_yticklabels(intent_names)
    ax1.set_title("Predicted Intent Timeline")
    ax1.set_ylabel("intent")
    ax1.grid(alpha=0.2)

    if "candidate_lock" in df.columns:
        lock_pts = df["candidate_lock"].fillna(0).astype(int).to_numpy() > 0
        ax1.scatter(x[lock_pts], y[lock_pts], s=10, color="#54A24B", label="candidate_lock", alpha=0.7)
    if "candidate_unlock" in df.columns:
        unlock_pts = df["candidate_unlock"].fillna(0).astype(int).to_numpy() > 0
        ax1.scatter(x[unlock_pts], y[unlock_pts], s=10, color="#E45756", label="candidate_unlock", alpha=0.7)
    if "candidate_lock" in df.columns or "candidate_unlock" in df.columns:
        ax1.legend(loc="upper right")

    ax2.plot(x, confidence.to_numpy(), linewidth=1.2, color="#F58518")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Model Confidence")
    ax2.set_ylabel("confidence")
    ax2.set_xlabel("window center index")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Saved timeline: {out}")


if __name__ == "__main__":
    main()
