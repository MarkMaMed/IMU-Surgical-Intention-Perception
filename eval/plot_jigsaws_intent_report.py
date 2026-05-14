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


def _get_cm(metrics: dict) -> tuple[np.ndarray, list[str]]:
    split_meta = metrics.get("split_meta", {})
    labels = list(split_meta.get("labels", ["LOCK_REQUIRED", "NO_LOCK_REQUIRED"]))
    cm = np.array(split_meta.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=float)
    return cm, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create high-end visual report for JIGSAWS surgical intent.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="plots/jigsaws_intent_report.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    pred_path = Path(args.predictions)
    if not metrics_path.exists() or not pred_path.exists():
        raise FileNotFoundError("metrics/predictions file not found.")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    df = pd.read_csv(pred_path)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # A: Task lock ratio
    ax_a = fig.add_subplot(gs[0, 0])
    task_stats = (
        df.groupby(["task", "lock_decision"]).size().unstack(fill_value=0)
        if not df.empty
        else pd.DataFrame()
    )
    if not task_stats.empty:
        total = task_stats.sum(axis=1).replace(0, 1)
        ratio = task_stats.div(total, axis=0)
        x = np.arange(len(ratio.index))
        ax_a.bar(x, ratio.get("SUGGEST_LOCK", 0), label="SUGGEST_LOCK", color="#2ca02c")
        ax_a.bar(
            x,
            ratio.get("SUGGEST_NO_LOCK", 0),
            bottom=ratio.get("SUGGEST_LOCK", 0),
            label="SUGGEST_NO_LOCK",
            color="#d62728",
        )
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(ratio.index.tolist(), rotation=15)
        ax_a.set_ylim(0, 1.0)
        ax_a.set_title("Task-wise Lock Decision Ratio")
        ax_a.legend(fontsize=8)
    else:
        ax_a.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_a.set_axis_off()

    # B: confusion matrix
    ax_b = fig.add_subplot(gs[0, 1])
    cm, labels = _get_cm(metrics)
    if cm.size > 0:
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax_b.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax_b.set_xticks(np.arange(len(labels)))
        ax_b.set_xticklabels(labels, rotation=20, ha="right")
        ax_b.set_yticks(np.arange(len(labels)))
        ax_b.set_yticklabels(labels)
        ax_b.set_title("Normalized Confusion Matrix (test)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_b.text(j, i, f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    else:
        ax_b.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
        ax_b.set_axis_off()

    # C: top feature importance
    ax_c = fig.add_subplot(gs[0, 2])
    fi = metrics.get("feature_importance", {})
    if fi:
        top = list(fi.items())[:10]
        names = [k for k, _ in top][::-1]
        vals = [v for _, v in top][::-1]
        ax_c.barh(names, vals, color="#9467bd")
        ax_c.set_title("Top Feature Importance")
    else:
        ax_c.text(0.5, 0.5, "No feature importance", ha="center", va="center")
        ax_c.set_axis_off()

    # D: timeline (choose a long test trial)
    ax_d = fig.add_subplot(gs[1, :2])
    if not df.empty:
        candidate = (
            df[df["split"] == "test"].groupby("trial_id").size().sort_values(ascending=False).index.tolist()
            or df.groupby("trial_id").size().sort_values(ascending=False).index.tolist()
        )
        trial = candidate[0]
        td = df[df["trial_id"] == trial].sort_values("start")
        x = (td["start"].to_numpy() + td["end"].to_numpy()) / 2.0
        y = td["proba_lock"].to_numpy()
        dec = td["lock_decision"].to_numpy()
        ax_d.plot(x, y, color="#1f77b4", lw=1.8, label="P(lock)")
        ax_d.fill_between(x, 0, y, color="#1f77b4", alpha=0.15)
        lock_mask = dec == "SUGGEST_LOCK"
        ax_d.scatter(x[lock_mask], y[lock_mask], s=14, color="#2ca02c", label="SUGGEST_LOCK", alpha=0.8)
        ax_d.scatter(x[~lock_mask], y[~lock_mask], s=10, color="#d62728", label="SUGGEST_NO_LOCK", alpha=0.5)
        ax_d.axhline(metrics.get("decision_stats", {}).get("proba_lock_threshold", 0.58), ls="--", color="#ff7f0e")
        ax_d.set_ylim(0, 1.05)
        ax_d.set_title(f"Lock Probability Timeline ({trial})")
        ax_d.set_xlabel("frame index")
        ax_d.set_ylabel("probability")
        ax_d.legend(fontsize=8, ncol=3)
    else:
        ax_d.text(0.5, 0.5, "No timeline data", ha="center", va="center")
        ax_d.set_axis_off()

    # E: velocity vs probability
    ax_e = fig.add_subplot(gs[1, 2])
    if not df.empty:
        sample = df.sample(min(len(df), 6000), random_state=42)
        color = np.where(sample["lock_decision"] == "SUGGEST_LOCK", "#2ca02c", "#d62728")
        ax_e.scatter(sample["global_vel_mean"], sample["proba_lock"], c=color, s=8, alpha=0.25, edgecolors="none")
        vthr = float(metrics.get("decision_stats", {}).get("stable_velocity_threshold", 0.02))
        pthr = float(metrics.get("decision_stats", {}).get("proba_lock_threshold", 0.58))
        ax_e.axvline(vthr, ls="--", color="#ff7f0e", lw=1.2)
        ax_e.axhline(pthr, ls="--", color="#ff7f0e", lw=1.2)
        ax_e.set_title("Decision Phase Space")
        ax_e.set_xlabel("global_vel_mean")
        ax_e.set_ylabel("P(lock)")
    else:
        ax_e.text(0.5, 0.5, "No scatter data", ha="center", va="center")
        ax_e.set_axis_off()

    # F: summary block
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.axis("off")
    split_meta = metrics.get("split_meta", {})
    report = split_meta.get("classification_report", {})
    acc = report.get("accuracy", None)
    macro_f1 = report.get("macro avg", {}).get("f1-score", None)
    lines = [
        "JIGSAWS Surgical Intent Summary",
        f"Trials: {metrics.get('loaded_trials', 'NA')} | Windows: {metrics.get('window_count', 'NA')}",
        f"Test Accuracy: {acc:.4f}" if isinstance(acc, (float, int)) else "Test Accuracy: NA",
        f"Macro F1: {macro_f1:.4f}" if isinstance(macro_f1, (float, int)) else "Macro F1: NA",
        f"Lock Gestures: {', '.join(metrics.get('lock_gestures', []))}",
        "Rule: High P(lock) + low global velocity => suggest lock.",
        "Interpretation: lock in fine stable phases; avoid lock in relocation/high-motion phases.",
    ]
    ax_f.text(
        0.01,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0f172a", ec="#334155", alpha=0.95),
        color="#e2e8f0",
    )

    fig.suptitle("JIGSAWS Surgical Intent Intelligence Report", fontsize=18, fontweight="bold")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved report figure: {out}")


if __name__ == "__main__":
    main()
