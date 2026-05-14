from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import pandas as pd


def _build_transitions(labels: list[str]) -> Counter[tuple[str, str]]:
    c: Counter[tuple[str, str]] = Counter()
    if not labels:
        return c
    prev = labels[0]
    for cur in labels[1:]:
        if cur != prev:
            c[(prev, cur)] += 1
        prev = cur
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot intent transition graph from prediction csv.")
    parser.add_argument("--predictions", required=True, help="Path to *_predictions.csv")
    parser.add_argument("--output", default="plots/imu_transition_graph.png")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    df = pd.read_csv(pred_path)
    if "pred_intent" not in df.columns:
        raise ValueError("Missing column: pred_intent")

    labels = df["pred_intent"].astype(str).tolist()
    trans = _build_transitions(labels)
    nodes = sorted(set(labels))
    if not nodes:
        raise RuntimeError("No intents found in predictions.")

    n = len(nodes)
    radius = 1.0
    pos = {}
    for i, name in enumerate(nodes):
        a = 2.0 * math.pi * i / float(max(1, n))
        pos[name] = (radius * math.cos(a), radius * math.sin(a))

    max_w = max(trans.values()) if trans else 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Intent Transition Graph")
    ax.axis("off")

    for name, (x, y) in pos.items():
        ax.scatter([x], [y], s=1800, color="#4C78A8", alpha=0.9)
        ax.text(x, y, name, color="white", ha="center", va="center", fontsize=9)

    for (src, dst), w in trans.items():
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        lw = 1.0 + 5.0 * (float(w) / float(max_w))
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=lw, color="#F58518", alpha=0.65),
        )
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(mx, my, str(w), fontsize=8, color="#333333")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"Saved transition graph: {out}")


if __name__ == "__main__":
    main()
