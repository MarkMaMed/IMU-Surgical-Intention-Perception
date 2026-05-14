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
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions csv: {missing}")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _glow_threshold_line(ax: plt.Axes, *, orientation: str, value: float, color: str, label: str) -> None:
    if orientation == "v":
        ax.axvline(value, ls="--", lw=5.2, color="white", alpha=0.12, zorder=3)
        ax.axvline(value, ls="--", lw=2.2, color=color, alpha=0.98, zorder=4, label=label)
    else:
        ax.axhline(value, ls="--", lw=5.2, color="white", alpha=0.12, zorder=3)
        ax.axhline(value, ls="--", lw=2.2, color=color, alpha=0.98, zorder=4, label=label)


def plot_phase_map(df: pd.DataFrame, decision_stats: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1220")
    ax.set_facecolor("#0f172a")

    hb = ax.hexbin(
        df["global_vel_mean"].to_numpy(),
        df["proba_lock"].to_numpy(),
        gridsize=60,
        mincnt=1,
        cmap="magma",
        linewidths=0.0,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("window density")
    cbar.ax.tick_params(colors="#e2e8f0")
    cbar.outline.set_edgecolor("#94a3b8")

    v_thr = float(decision_stats.get("stable_velocity_threshold", 0.06))
    p_thr = float(decision_stats.get("proba_lock_threshold", 0.58))
    x_max = max(0.17, float(df["global_vel_mean"].quantile(0.995)))

    # Highlight decision regions with bright color blocks so they read well in PPT screenshots.
    ax.add_patch(
        Rectangle((0.0, p_thr), v_thr, 1.05 - p_thr, facecolor="#22c55e", alpha=0.12, edgecolor="#86efac", lw=2.0, zorder=1)
    )
    ax.add_patch(
        Rectangle((v_thr, 0.0), x_max - v_thr, p_thr, facecolor="#ef4444", alpha=0.09, edgecolor="#fca5a5", lw=2.0, zorder=1)
    )
    ax.add_patch(
        Rectangle((0.0, 0.0), v_thr, p_thr, facecolor="#f59e0b", alpha=0.06, edgecolor="#fde68a", lw=1.2, zorder=1)
    )
    ax.add_patch(
        Rectangle((v_thr, p_thr), x_max - v_thr, 1.05 - p_thr, facecolor="#8b5cf6", alpha=0.05, edgecolor="#c4b5fd", lw=1.2, zorder=1)
    )

    _glow_threshold_line(
        ax,
        orientation="v",
        value=v_thr,
        color="#22d3ee",
        label=f"velocity threshold={v_thr:.3f}",
    )
    _glow_threshold_line(
        ax,
        orientation="h",
        value=p_thr,
        color="#2dd4bf",
        label=f"lock prob threshold={p_thr:.2f}",
    )

    lock_df = df[(df["global_vel_mean"] <= v_thr) & (df["proba_lock"] >= p_thr)]
    no_lock_df = df[(df["global_vel_mean"] > v_thr) & (df["proba_lock"] < p_thr)]
    if not lock_df.empty:
        ax.scatter(
            [float(lock_df["global_vel_mean"].median())],
            [float(lock_df["proba_lock"].median())],
            s=280,
            marker="*",
            color="#fef08a",
            edgecolors="#ffffff",
            linewidths=1.3,
            zorder=6,
        )
    if not no_lock_df.empty:
        ax.scatter(
            [float(no_lock_df["global_vel_mean"].median())],
            [float(no_lock_df["proba_lock"].median())],
            s=220,
            marker="D",
            color="#fb7185",
            edgecolors="#ffffff",
            linewidths=1.1,
            zorder=6,
        )

    ax.text(
        v_thr * 0.52,
        0.86,
        "Lock Suggestion Zone",
        color="#dcfce7",
        fontsize=13,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#14532d", ec="#86efac", alpha=0.85),
    )
    ax.text(
        v_thr * 1.05,
        0.22,
        "No-Lock Zone",
        color="#fee2e2",
        fontsize=13,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#7f1d1d", ec="#fca5a5", alpha=0.82),
    )
    ax.text(
        0.012,
        1.02,
        "Bright overlays mark the two main decision regions",
        color="#fde68a",
        fontsize=10,
        weight="bold",
        transform=ax.transAxes,
    )

    ax.set_title("Surgical Intent Decision Phase Map", fontsize=16, color="#e2e8f0", weight="bold")
    ax.set_xlabel("Global Motion Velocity (lower means more stable)", color="#cbd5e1")
    ax.set_ylabel("Predicted Lock Probability", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.set_xlim(-0.002, x_max)
    ax.legend(facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#e2e8f0")

    _save(fig, out_dir / "showcase_phase_map.png")


def plot_task_gesture_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot_count = df.groupby("major_gesture").size().sort_values(ascending=False)
    top_gestures = pivot_count.head(12).index.tolist()
    sub = df[df["major_gesture"].isin(top_gestures)].copy()

    mean_proba = (
        sub.groupby(["task", "major_gesture"])["proba_lock"].mean().unstack(fill_value=0.0).reindex(columns=top_gestures)
    )
    lock_ratio = (
        sub.assign(is_lock=(sub["lock_decision"] == "SUGGEST_LOCK").astype(float))
        .groupby(["task", "major_gesture"])["is_lock"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=top_gestures)
    )

    fig = plt.figure(figsize=(15, 9), facecolor="#ffffff")
    gs = fig.add_gridspec(1, 2, wspace=0.2)

    for i, (title, mat, cmap) in enumerate(
        [
            ("Mean P(lock) by Task x Gesture", mean_proba.to_numpy(), "viridis"),
            ("Lock Decision Ratio by Task x Gesture", lock_ratio.to_numpy(), "plasma"),
        ]
    ):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(top_gestures)))
        ax.set_xticklabels(top_gestures, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(mean_proba.index)))
        ax.set_yticklabels(mean_proba.index.tolist())
        ax.set_title(title, fontsize=13, weight="bold")
        row_best = mat.argmax(axis=1) if mat.size else np.array([], dtype=int)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                text_color = "#fff7ed" if mat[r, c] < 0.8 else "#1f2937"
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=8, color=text_color, weight="bold")
                if c == int(row_best[r]):
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#fde047",
                            linewidth=2.8,
                            zorder=4,
                        )
                    )
                elif mat[r, c] >= 0.75:
                    ax.add_patch(
                        Rectangle(
                            (c - 0.5, r - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#f97316",
                            linewidth=1.8,
                            zorder=4,
                        )
                    )
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.suptitle("Task-Gesture Fine-Grained Surgical Intent Landscape", fontsize=16, weight="bold")
    _save(fig, out_dir / "showcase_task_gesture_heatmap.png")


def plot_surgeon_generalization(df: pd.DataFrame, out_dir: Path) -> None:
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        test_df = df.copy()

    acc = (
        test_df.assign(correct=(test_df["pred_label"] == test_df["lock_label"]).astype(float))
        .groupby(["surgeon", "task"])["correct"]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index()
    )
    counts = test_df.groupby("surgeon").size().reindex(acc.index).fillna(0).astype(int)

    fig = plt.figure(figsize=(14, 8), facecolor="#f8fafc")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    mat = np.nan_to_num(acc.to_numpy(), nan=0.0)
    im = ax1.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks(np.arange(len(acc.columns)))
    ax1.set_xticklabels(acc.columns.tolist())
    ax1.set_yticks(np.arange(len(acc.index)))
    ax1.set_yticklabels(acc.index.tolist())
    ax1.set_title("Generalization Accuracy (Surgeon x Task)", fontsize=13, weight="bold")
    max_pos = np.unravel_index(np.argmax(mat), mat.shape)
    min_pos = np.unravel_index(np.argmin(mat), mat.shape)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            value = float(mat[r, c])
            text_color = "#f8fafc" if value >= 0.55 else "#111827"
            stroke_color = "#0f172a" if value >= 0.55 else "#ffffff"
            ax1.text(
                c,
                r,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                weight="bold",
                path_effects=[pe.withStroke(linewidth=2.2, foreground=stroke_color, alpha=0.9)],
            )
            if (r, c) == max_pos:
                ax1.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, edgecolor="#fde047", linewidth=3.0))
            elif (r, c) == min_pos:
                ax1.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, edgecolor="#fb7185", linewidth=3.0))
    fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)

    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = ["#2563eb"] * len(counts)
    if len(counts) > 0:
        bar_colors[int(np.argmax(counts.to_numpy()))] = "#f59e0b"
    ax2.barh(acc.index.astype(str).tolist(), counts.tolist(), color=bar_colors)
    ax2.set_title("Test Windows per Surgeon", fontsize=13, weight="bold")
    ax2.set_xlabel("count")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Cross-Surgeon Robustness View", fontsize=16, weight="bold")
    _save(fig, out_dir / "showcase_surgeon_generalization.png")


def plot_trial_storyboard(df: pd.DataFrame, decision_stats: dict, out_dir: Path, trial: str | None) -> None:
    if trial and trial in set(df["trial_id"].astype(str).tolist()):
        trial_id = trial
    else:
        candidate = (
            df[df["split"] == "test"].groupby("trial_id").size().sort_values(ascending=False).index.tolist()
            or df.groupby("trial_id").size().sort_values(ascending=False).index.tolist()
        )
        if not candidate:
            return
        trial_id = str(candidate[0])

    td = df[df["trial_id"] == trial_id].sort_values("start").copy()
    if td.empty:
        return

    x = ((td["start"].to_numpy() + td["end"].to_numpy()) / 2.0).astype(float)
    p = td["proba_lock"].to_numpy(dtype=float)
    v = td["global_vel_mean"].to_numpy(dtype=float)
    dec = td["lock_decision"].astype(str).to_numpy()
    gest = td["major_gesture"].astype(str).to_numpy()

    gesture_order = sorted(set(gest))
    g2i = {g: i for i, g in enumerate(gesture_order)}
    gi = np.array([g2i[g] for g in gest], dtype=float)

    p_thr = float(decision_stats.get("proba_lock_threshold", 0.58))
    v_thr = float(decision_stats.get("stable_velocity_threshold", 0.06))

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"hspace": 0.1}, facecolor="#ffffff")

    ax0, ax1, ax2 = axes
    for xv, is_lock in zip(x, lock_mask := (dec == "SUGGEST_LOCK")):
        color = "#86efac" if is_lock else "#fecaca"
        alpha = 0.08 if is_lock else 0.05
        ax0.axvspan(xv - 15, xv + 15, color=color, alpha=alpha, zorder=0)
        ax1.axvspan(xv - 15, xv + 15, color=color, alpha=alpha * 0.9, zorder=0)
        ax2.axvspan(xv - 15, xv + 15, color=color, alpha=alpha * 0.9, zorder=0)
    ax0.plot(x, p, color="#1d4ed8", lw=1.8)
    ax0.fill_between(x, 0, p, color="#93c5fd", alpha=0.35)
    _glow_threshold_line(ax0, orientation="h", value=p_thr, color="#10b981", label="lock threshold")
    ax0.scatter(x[lock_mask], p[lock_mask], s=18, color="#059669", alpha=0.9, label="SUGGEST_LOCK")
    ax0.scatter(x[~lock_mask], p[~lock_mask], s=14, color="#dc2626", alpha=0.65, label="SUGGEST_NO_LOCK")
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("P(lock)")
    ax0.legend(loc="upper right", ncol=2, fontsize=9)
    ax0.set_title(f"Trial Storyboard - {trial_id}", fontsize=15, weight="bold")

    ax1.plot(x, v, color="#7c3aed", lw=1.6)
    _glow_threshold_line(ax1, orientation="h", value=v_thr, color="#0ea5e9", label="velocity threshold")
    ax1.fill_between(x, 0, np.minimum(v, v_thr), color="#c4b5fd", alpha=0.25)
    ax1.set_ylabel("velocity")
    ax1.text(0.01, 0.9, "Green spans = lock-favorable windows", transform=ax1.transAxes, color="#166534", fontsize=10, weight="bold")

    ax2.scatter(x, gi, c=np.where(lock_mask, "#16a34a", "#ef4444"), s=14, alpha=0.8)
    ax2.set_yticks(np.arange(len(gesture_order)))
    ax2.set_yticklabels(gesture_order)
    ax2.set_ylabel("gesture")
    ax2.set_xlabel("frame index")
    ax2.grid(alpha=0.2)

    _save(fig, out_dir / "showcase_trial_storyboard.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate showcase figures for JIGSAWS intent model.")
    parser.add_argument("--metrics", default="logs/jigsaws_intent/metrics.json")
    parser.add_argument("--predictions", default="logs/jigsaws_intent/window_predictions.csv")
    parser.add_argument("--output-dir", default="plots/jigsaws_showcase")
    parser.add_argument("--trial", default="")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    pred_path = Path(args.predictions)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions not found: {pred_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    df = pd.read_csv(pred_path)
    _ensure_columns(
        df,
        [
            "task",
            "trial_id",
            "surgeon",
            "major_gesture",
            "global_vel_mean",
            "proba_lock",
            "lock_decision",
            "split",
            "pred_label",
            "lock_label",
            "start",
            "end",
        ],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_stats = metrics.get("decision_stats", {})

    plot_phase_map(df, decision_stats, out_dir)
    plot_task_gesture_heatmap(df, out_dir)
    plot_surgeon_generalization(df, out_dir)
    plot_trial_storyboard(df, decision_stats, out_dir, args.trial or None)
    print(f"Showcase figures exported to: {out_dir}")


if __name__ == "__main__":
    main()
