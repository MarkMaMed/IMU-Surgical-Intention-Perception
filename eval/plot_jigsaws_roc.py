from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _binary_truth(series: pd.Series) -> np.ndarray:
    return (series.astype(str).to_numpy() == "LOCK_REQUIRED").astype(np.int32)


def _roc_data(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, float(auc(fpr, tpr))


def _styled_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.2)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#94a3b8", alpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC curves for JIGSAWS holdout predictions.")
    parser.add_argument("--predictions", default="logs/jigsaws_intent/window_predictions.csv")
    parser.add_argument("--metrics", default="logs/jigsaws_intent/metrics.json")
    parser.add_argument("--output-dir", default="plots/jigsaws_roc")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    metrics_path = Path(args.metrics)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions not found: {pred_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(pred_path)
    if "holdout_proba_lock" not in df.columns:
        raise ValueError("Missing holdout_proba_lock in predictions file. Please retrain the model first.")

    holdout = df[(df["split"] == "test") & df["holdout_proba_lock"].notna()].copy()
    if holdout.empty:
        raise RuntimeError("No holdout predictions available for ROC plotting.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = _binary_truth(holdout["lock_label"])
    scores = holdout["holdout_proba_lock"].to_numpy(dtype=float)
    fpr, tpr, auc_overall = _roc_data(y_true, scores)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#ffffff")
    _styled_axes(ax, "Overall ROC - Holdout Set")
    ax.fill_between(fpr, 0, tpr, color="#93c5fd", alpha=0.25)
    ax.plot(fpr, tpr, color="#1d4ed8", lw=3.0, label=f"Overall ROC (AUC={auc_overall:.3f})")
    ax.scatter([fpr[np.argmax(tpr - fpr)]], [tpr[np.argmax(tpr - fpr)]], s=90, color="#f59e0b", edgecolors="#ffffff", zorder=5)
    ax.legend(loc="lower right", frameon=True)
    ax.text(
        0.52,
        0.15,
        f"Reported holdout AUC = {metrics.get('split_meta', {}).get('roc_auc_overall', auc_overall):.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#60a5fa", alpha=0.95),
    )
    _save(fig, out_dir / "jigsaws_roc_overall.png")

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#ffffff")
    _styled_axes(ax, "Task-wise ROC Comparison")
    palette = {
        "Knot_Tying": "#2563eb",
        "Needle_Passing": "#db2777",
        "Suturing": "#16a34a",
    }
    for task_name, task_df in holdout.groupby("task"):
        yt = _binary_truth(task_df["lock_label"])
        if len(np.unique(yt)) < 2:
            continue
        task_scores = task_df["holdout_proba_lock"].to_numpy(dtype=float)
        task_fpr, task_tpr, task_auc = _roc_data(yt, task_scores)
        ax.plot(
            task_fpr,
            task_tpr,
            lw=2.6,
            color=palette.get(str(task_name), None),
            label=f"{task_name} (AUC={task_auc:.3f})",
        )
    ax.legend(loc="lower right", frameon=True)
    _save(fig, out_dir / "jigsaws_roc_by_task.png")

    task_names = [str(x) for x in sorted(holdout["task"].astype(str).unique().tolist())]
    fig, axes = plt.subplots(1, max(1, len(task_names)), figsize=(5.2 * max(1, len(task_names)), 4.8), facecolor="#ffffff")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, task_name in zip(axes, task_names):
        task_df = holdout[holdout["task"].astype(str) == task_name].copy()
        yt = _binary_truth(task_df["lock_label"])
        _styled_axes(ax, task_name)
        if len(np.unique(yt)) < 2:
            ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
            continue
        task_scores = task_df["holdout_proba_lock"].to_numpy(dtype=float)
        task_fpr, task_tpr, task_auc = _roc_data(yt, task_scores)
        color = palette.get(task_name, "#7c3aed")
        ax.fill_between(task_fpr, 0, task_tpr, color=color, alpha=0.18)
        ax.plot(task_fpr, task_tpr, lw=2.8, color=color)
        ax.text(
            0.52,
            0.12,
            f"AUC={task_auc:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec=color, alpha=0.95),
        )
    _save(fig, out_dir / "jigsaws_roc_panels.png")


if __name__ == "__main__":
    main()
