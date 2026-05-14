from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import numpy as np


def _safe_dict(x: object) -> dict:
    return x if isinstance(x, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IMU training dashboard from metrics json.")
    parser.add_argument("--metrics", required=True, help="Path to imu_intent_metrics*.json")
    parser.add_argument("--output", default="plots/imu_dashboard.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    class_dist = _safe_dict(metrics.get("class_distribution"))
    dataset_acc = _safe_dict(metrics.get("dataset_accuracy"))
    source_dist = _safe_dict(metrics.get("signal_source_distribution"))
    labels = list(metrics.get("labels", []))
    cm = np.array(metrics.get("confusion_matrix", []), dtype=float)
    report = _safe_dict(metrics.get("classification_report"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    if class_dist:
        names = list(class_dist.keys())
        vals = [class_dist[k] for k in names]
        ax1.bar(names, vals, color="#4C78A8")
        ax1.set_title("Class Distribution")
        ax1.set_ylabel("window count")
        ax1.tick_params(axis="x", rotation=15)
    else:
        ax1.text(0.5, 0.5, "No class_distribution", ha="center", va="center")
        ax1.set_axis_off()

    if dataset_acc:
        names = list(dataset_acc.keys())
        vals = [dataset_acc[k] for k in names]
        ax2.bar(names, vals, color="#F58518")
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Dataset Accuracy")
        ax2.set_ylabel("accuracy")
        ax2.tick_params(axis="x", rotation=15)
    else:
        ax2.text(0.5, 0.5, "No dataset_accuracy", ha="center", va="center")
        ax2.set_axis_off()

    if cm.size > 0 and labels:
        im = ax3.imshow(cm, cmap="Blues")
        ax3.set_title("Confusion Matrix")
        ax3.set_xticks(np.arange(len(labels)))
        ax3.set_xticklabels(labels, rotation=20, ha="right")
        ax3.set_yticks(np.arange(len(labels)))
        ax3.set_yticklabels(labels)
        ax3.set_xlabel("Pred")
        ax3.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "No confusion_matrix", ha="center", va="center")
        ax3.set_axis_off()

    summary_lines = [
        f"loaded_sequences: {metrics.get('loaded_sequences', 'NA')}",
        f"window_count: {metrics.get('window_count', 'NA')}",
        f"accuracy: {report.get('accuracy', 'NA')}",
        f"macro_f1: {_safe_dict(report.get('macro avg')).get('f1-score', 'NA')}",
    ]
    if source_dist:
        summary_lines.append("signal_source_distribution:")
        for k, v in source_dist.items():
            summary_lines.append(f"  - {k}: {v}")

    ax4.axis("off")
    ax4.set_title("Summary")
    ax4.text(0.02, 0.98, "\n".join(summary_lines), ha="left", va="top", family="monospace")

    fig.suptitle("IMU Intent Training Dashboard", fontsize=14)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Saved dashboard: {out}")


if __name__ == "__main__":
    main()
