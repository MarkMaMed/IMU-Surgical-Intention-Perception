from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path("plots/midterm_defense")


def _pick_chinese_font() -> str:
    candidates = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "STSong",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


FONT_NAME = _pick_chinese_font()


def _setup_style() -> None:
    plt.rcParams["font.family"] = FONT_NAME
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f8fafc"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["savefig.facecolor"] = "#f8fafc"


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def _load_data(metrics_path: Path, predictions_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(predictions_path)
    holdout = df[df["split"] == "test"].copy()
    if "holdout_pred_label" in holdout.columns:
        holdout["eval_pred"] = holdout["holdout_pred_label"].fillna(holdout["pred_label"])
        holdout["eval_prob"] = holdout["holdout_proba_lock"].fillna(holdout["proba_lock"])
    else:
        holdout["eval_pred"] = holdout["pred_label"]
        holdout["eval_prob"] = holdout["proba_lock"]
    holdout["is_correct"] = (holdout["eval_pred"] == holdout["lock_label"]).astype(float)
    return metrics, df, holdout


def _panel(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str], accent: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        facecolor="#ffffff",
        edgecolor=accent,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y + h - 0.08), w, 0.08, transform=ax.transAxes, facecolor=accent, edgecolor=accent))
    ax.text(x + 0.02, y + h - 0.04, title, transform=ax.transAxes, fontsize=15, color="white", va="center", weight="bold")
    for idx, line in enumerate(lines):
        ax.text(
            x + 0.03,
            y + h - 0.13 - idx * 0.075,
            f"- {line}",
            transform=ax.transAxes,
            fontsize=12,
            color="#0f172a",
            va="top",
        )


def plot_progress_board(metrics: dict) -> Path:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    acc = metrics["split_meta"]["classification_report"]["accuracy"]
    macro_f1 = metrics["split_meta"]["classification_report"]["macro avg"]["f1-score"]
    auc = metrics["split_meta"]["roc_auc_overall"]
    prob_thr = metrics["decision_stats"]["proba_lock_threshold"]
    vel_thr = metrics["decision_stats"]["stable_velocity_threshold"]

    ax.text(0.05, 0.94, "中期项目进展总览", fontsize=28, weight="bold", color="#0f172a", transform=ax.transAxes)
    ax.text(
        0.72,
        0.94,
        "当前阶段：可运行、可评估、可转化",
        fontsize=13,
        color="#14532d",
        weight="bold",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7", ec="#86efac"),
    )
    ax.text(
        0.05,
        0.885,
        "围绕结构设计、共享控制、IMU意态感知和临床前验证四条线同步推进，已形成原型基础、算法支撑与验证路径。",
        fontsize=13,
        color="#334155",
        transform=ax.transAxes,
    )

    _panel(
        ax,
        0.05,
        0.50,
        0.42,
        0.30,
        "结构与系统方案",
        [
            "核心部件三维建模与装配关系校核完成",
            "关键机构样机验证完成，可展示结构总装与局部设计",
            "围绕舒适性和稳定性补充绑缚改良材料与装配辅料",
        ],
        "#2563eb",
    )
    _panel(
        ax,
        0.53,
        0.50,
        0.42,
        0.30,
        "控制与软件",
        [
            "双模式随动与姿态保持控制逻辑已明确",
            "模式切换状态机与脚踏确认链路完成",
            "形成“候选提出—人工确认—执行切换”的共享控制思路",
        ],
        "#7c3aed",
    )
    _panel(
        ax,
        0.05,
        0.14,
        0.42,
        0.30,
        "数据与算法",
        [
            "JIGSAWS 意态感知流程已跑通，共 103 条试验数据，7957 个滑窗样本",
            f"离线结果：准确率 {acc:.4f}，宏平均 F1 {macro_f1:.4f}，整体 AUC {auc:.3f}",
            f"锁定判据：锁定概率不低于 {prob_thr:.2f}，全局速度不高于 {vel_thr:.3f}",
        ],
        "#059669",
    )
    _panel(
        ax,
        0.53,
        0.14,
        0.42,
        0.30,
        "临床前验证准备",
        [
            "性能验证实验方案完成，覆盖静态、动态、耐力、精细操作四类任务",
            "标准化数据记录表完成，客观指标与主观量表已配套",
            "下一阶段进入伦理与小样本临床前验证，接入真实 IMU 与本体传感",
        ],
        "#ea580c",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.04),
            0.90,
            0.06,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.5,
            facecolor="#0f172a",
            edgecolor="#1e293b",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.07,
        "当前项目已形成：原型基础、共享控制逻辑、IMU意态感知程序、临床验证方法学",
        fontsize=14,
        color="#f8fafc",
        transform=ax.transAxes,
        ha="center",
        va="center",
        weight="bold",
    )

    return _save(fig, "midterm_progress_board.png")


def plot_workload_budget() -> Path:
    categories = ["文献与需求", "机械设计", "控制软件", "数据训练"]
    hours = np.array([80, 130, 140, 80])
    colors = ["#2563eb", "#7c3aed", "#059669", "#ea580c"]

    budget_labels = ["绑缚材料", "紧固辅料", "IMU与固定件", "问卷打印", "预留经费"]
    budget_values = np.array([300, 460, 700, 40, 3000])
    budget_colors = ["#3b82f6", "#60a5fa", "#14b8a6", "#f59e0b", "#cbd5e1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.suptitle("项目实施投入工作量与经费启用情况", fontsize=24, weight="bold", y=0.98)

    y = np.arange(len(categories))
    ax1.barh(y, hours, color=colors, height=0.56)
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories, fontsize=12)
    ax1.set_xlabel("投入人时  单位 h", fontsize=12)
    ax1.set_title("累计投入约 430 人时", fontsize=16, weight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.18)
    ax1.set_xlim(0, 160)
    for i, v in enumerate(hours):
        ax1.text(v + 4, i, f"{v} h", va="center", fontsize=12, weight="bold", color="#0f172a")
        ax1.text(v - 6, i, f"{v / hours.sum():.1%}", va="center", ha="right", fontsize=10, color="white", weight="bold")

    wedges, _ = ax2.pie(
        budget_values,
        colors=budget_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
    )
    ax2.text(0, 0.08, "4500 元", ha="center", va="center", fontsize=24, weight="bold", color="#0f172a")
    ax2.text(0, -0.10, "中期经费", ha="center", va="center", fontsize=12, color="#475569")
    ax2.set_title("已启用 1500 元，预留 3000 元用于临床前验证", fontsize=16, weight="bold")
    legend_labels = [f"{k}：{v} 元" for k, v in zip(budget_labels, budget_values)]
    ax2.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11)

    fig.text(
        0.72,
        0.12,
        "经费已覆盖：\n1. 结构穿戴改良与装配辅料\n2. 真实采集前的 IMU 模块与固定件\n3. 记录表、问卷与答辩材料准备\n4. 剩余经费用于伦理后的小样本临床前实验",
        fontsize=11.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#93c5fd"),
    )

    return _save(fig, "midterm_workload_budget.png")


def _draw_kpi_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, value: str, note: str, accent: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="white",
            edgecolor=accent,
            linewidth=1.6,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.02, y + h - 0.06, title, transform=ax.transAxes, fontsize=12, color="#475569", weight="bold")
    ax.text(x + 0.02, y + h - 0.17, value, transform=ax.transAxes, fontsize=24, color=accent, weight="bold")
    ax.text(x + 0.02, y + 0.03, note, transform=ax.transAxes, fontsize=10.5, color="#334155")


def plot_stage_results_dashboard(metrics: dict, df: pd.DataFrame, holdout: pd.DataFrame) -> Path:
    split = metrics["split_meta"]
    task_counts = df.groupby("task").size().reindex(["Knot_Tying", "Needle_Passing", "Suturing"])
    task_acc = holdout.groupby("task")["is_correct"].mean().reindex(["Knot_Tying", "Needle_Passing", "Suturing"])
    candidates = split["candidate_scores"]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.18], hspace=0.26, wspace=0.25)
    top_ax = fig.add_subplot(gs[0, :])
    top_ax.axis("off")

    fig.suptitle("阶段性成果与关键指标", fontsize=24, weight="bold", y=0.98)

    kpis = [
        ("试验条目", str(metrics["loaded_trials"]), "JIGSAWS 原始 trial 数", "#2563eb"),
        ("滑窗样本", str(metrics["window_count"]), "公开手术运动窗口", "#7c3aed"),
        ("测试准确率", f"{split['classification_report']['accuracy']:.4f}", "group holdout 结果", "#059669"),
        ("宏平均 F1", f"{split['classification_report']['macro avg']['f1-score']:.4f}", "类别均衡表现", "#ea580c"),
        ("整体 AUC", f"{split['roc_auc_overall']:.3f}", "ROC 判别能力", "#dc2626"),
        ("专利申请", "4 项", "2 项新型，2 项机型相关", "#0891b2"),
    ]
    card_w = 0.145
    gap = 0.015
    for idx, (title, value, note, accent) in enumerate(kpis):
        _draw_kpi_card(top_ax, 0.02 + idx * (card_w + gap), 0.13, card_w, 0.72, title, value, note, accent)

    ax1 = fig.add_subplot(gs[1, 0])
    bars = ax1.bar(task_counts.index, task_counts.values, color=["#60a5fa", "#34d399", "#f59e0b"])
    ax1.set_title("公开手术数据窗口分布", fontsize=15, weight="bold")
    ax1.set_ylabel("窗口数")
    ax1.grid(axis="y", alpha=0.16)
    for bar, value in zip(bars, task_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{int(value)}", ha="center", fontsize=11, weight="bold")
    ax1.tick_params(axis="x", rotation=12)

    ax2 = fig.add_subplot(gs[1, 1])
    bars = ax2.bar(task_acc.index, task_acc.values, color=["#1d4ed8", "#0f766e", "#b45309"])
    ax2.set_ylim(0, 1.05)
    ax2.set_title("按任务划分的 holdout 准确率", fontsize=15, weight="bold")
    ax2.set_ylabel("准确率")
    ax2.grid(axis="y", alpha=0.16)
    for bar, value in zip(bars, task_acc.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=11, weight="bold")
    ax2.tick_params(axis="x", rotation=12)

    ax3 = fig.add_subplot(gs[1, 2])
    model_names = ["RandomForest", "ExtraTrees", "XGBoost", "随机森林与极端树集成"]
    model_scores = [
        candidates["random_forest"]["macro_f1"],
        candidates["extra_trees"]["macro_f1"],
        candidates["xgboost"]["macro_f1"],
        candidates["rf_et_seed_ensemble"]["macro_f1"],
    ]
    colors = ["#93c5fd", "#60a5fa", "#fca5a5", "#22c55e"]
    bars = ax3.barh(model_names, model_scores, color=colors)
    ax3.set_xlim(0.70, 0.85)
    ax3.set_title("候选模型 宏平均 F1 对比", fontsize=15, weight="bold")
    ax3.grid(axis="x", alpha=0.16)
    for bar, value in zip(bars, model_scores):
        ax3.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=11, weight="bold")

    fig.text(
        0.02,
        0.03,
        "核心结论：当前程序已能较稳定地区分“精细稳态阶段，建议锁定”与“重定位和大运动阶段，不建议锁定”，并且可直接输出相图、热图、时间分镜与 ROC 图用于答辩展示。",
        fontsize=12,
        color="#334155",
    )

    return _save(fig, "midterm_stage_results_dashboard.png")


def plot_stage_results_table(metrics: dict) -> Path:
    fig = plt.figure(figsize=(16, 9.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.5, 0.965, "阶段性成果汇总表", fontsize=24, weight="bold", ha="center", va="top")

    columns = ["成果模块", "当前完成内容", "关键证据", "答辩可展示"]
    rows = [
        [
            "结构原型",
            "核心部件建模、装配校核、关键机构样机验证完成",
            "结构方案已具备 CAD 与总装展示基础",
            "结构设计图、总装图、关键机构局部图",
        ],
        [
            "共享控制",
            "双模式随动与姿态保持逻辑及脚踏确认链路完成",
            "形成“候选提出—人工确认—执行切换”状态机",
            "控制逻辑图、模式切换说明",
        ],
        [
            "IMU 意态感知",
            "JIGSAWS 清洗、滑窗、特征、训练、推理与可视化流程完成",
            f"103 条试验数据，7957 个窗口，准确率 {metrics['split_meta']['classification_report']['accuracy']:.4f}",
            "总体结果图、相图、热图、时间分镜、ROC",
        ],
        [
            "临床前验证",
            "实验方案与标准数据记录表完成，任务与指标体系明确",
            "4 类任务，3 类条件，客观终点和主观终点齐备",
            "任务矩阵图、验证流程图、记录表样例",
        ],
        [
            "知识产权",
            "已申请 4 项相关专利",
            "项目创新点具备归纳与保护基础",
            "成果概述页、创新点总结页",
        ],
    ]

    col_widths = [0.14, 0.36, 0.20, 0.23]
    start_x = 0.04
    total_width = sum(col_widths)
    y_top = 0.82
    row_h = 0.125

    x = start_x
    for col, width in zip(columns, col_widths):
        ax.add_patch(Rectangle((x, y_top), width, 0.08, transform=ax.transAxes, facecolor="#0f172a", edgecolor="white"))
        ax.text(x + 0.01, y_top + 0.04, col, transform=ax.transAxes, va="center", fontsize=13, color="white", weight="bold")
        x += width

    for r, row in enumerate(rows):
        y = y_top - (r + 1) * row_h
        row_color = "#ffffff" if r % 2 == 0 else "#f8fafc"
        x = start_x
        for c, (text, width) in enumerate(zip(row, col_widths)):
            ax.add_patch(Rectangle((x, y), width, row_h, transform=ax.transAxes, facecolor=row_color, edgecolor="#cbd5e1", linewidth=1.2))
            ax.text(
                x + 0.01,
                y + row_h / 2,
                text,
                transform=ax.transAxes,
                va="center",
                fontsize=12,
                color="#0f172a",
                wrap=True,
                weight="bold" if c == 0 else None,
            )
            x += width

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.08),
            total_width,
            0.07,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#eff6ff",
            edgecolor="#93c5fd",
        )
    )
    ax.text(
        0.045,
        0.115,
        "结论：项目已从“设计构想”进入“可运行原型、可解释算法、可执行验证方案”的中期阶段，具备继续开展小样本临床前验证的基础。",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#1e3a8a",
        weight="bold",
    )

    return _save(fig, "midterm_stage_results_table.png")


def plot_clinical_validation_matrix() -> Path:
    fig = plt.figure(figsize=(18, 9.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.5, 0.965, "临床前性能验证任务矩阵", fontsize=24, weight="bold", ha="center", va="top")

    columns = ["任务场景", "对照条件", "主要客观指标", "主观量表", "主要传感器配置"]
    rows = [
        [
            "静态姿态保持",
            "无外骨骼对照锁定支撑",
            "末端漂移量\n8-12 Hz 震颤能量\n三角肌与斜方肌 sEMG RMS",
            "Borg 与 RPE\n舒适度反馈",
            "上臂、前臂、手背 IMU\nEMG\n肩肘关节力矩",
        ],
        [
            "动态轨迹跟随",
            "无外骨骼对照随动模式",
            "完成时间\nRMSE 与最大偏差\n交互力与响应延迟",
            "NASA TLX\n控制感与稳定性评分",
            "上臂、前臂、末端 IMU\nEMG\n交互力与力矩传感",
        ],
        [
            "重复操作耐力",
            "无外骨骼对照外骨骼辅助",
            "RPE 六到二十\nsEMG MPF 下降趋势\n心率与总完成次数",
            "Borg 与 RPE\n局部酸痛记录",
            "IMU\nEMG\n心率\n关节力矩",
        ],
        [
            "精细操作模拟",
            "无外骨骼对照随动模式",
            "总耗时\n掉落次数与误触次数\nFLS 评分与末端抖动",
            "NASA TLX\nSUS\n自然度与受限感",
            "含手背与器械末端的 IMU\nEMG 近端 可加前臂\n交互力与力矩",
        ],
    ]

    col_widths = [0.13, 0.15, 0.25, 0.16, 0.22]
    start_x = 0.035
    y_top = 0.79
    header_h = 0.08
    row_h = 0.16

    x = start_x
    for col, width in zip(columns, col_widths):
        ax.add_patch(Rectangle((x, y_top), width, header_h, transform=ax.transAxes, facecolor="#0f172a", edgecolor="white"))
        ax.text(x + 0.01, y_top + header_h / 2, col, transform=ax.transAxes, va="center", fontsize=13, color="white", weight="bold")
        x += width

    row_colors = ["#f8fafc", "#ffffff"]
    for r, row in enumerate(rows):
        y = y_top - (r + 1) * row_h
        x = start_x
        for c, (text, width) in enumerate(zip(row, col_widths)):
            face = row_colors[r % 2]
            edge = "#cbd5e1"
            if c == 0:
                face = ["#dbeafe", "#dcfce7", "#fef3c7", "#fee2e2"][r]
                edge = ["#60a5fa", "#4ade80", "#f59e0b", "#fb7185"][r]
            ax.add_patch(Rectangle((x, y), width, row_h, transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.5))
            ax.text(
                x + 0.01,
                y + row_h / 2,
                text,
                transform=ax.transAxes,
                va="center",
                fontsize=12,
                color="#0f172a",
                weight="bold" if c == 0 else None,
            )
            x += width

    ax.text(
        0.035,
        0.08,
        "说明：所有任务均以“受试者自身前后对照”为主，实验环境包含实验室仿真与模拟手术室两类场景；"
        "指标同时覆盖减负、稳定性、透明度、耐力、可用性与安全性。",
        transform=ax.transAxes,
        fontsize=12,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.45", fc="#eff6ff", ec="#93c5fd"),
    )

    return _save(fig, "clinical_validation_task_matrix.png")


def _flow_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
            transform=ax.transAxes,
        )
    )
    ax.text(x + w / 2, y + h * 0.68, title, transform=ax.transAxes, ha="center", va="center", fontsize=14, weight="bold", color="#0f172a")
    ax.text(x + w / 2, y + h * 0.33, body, transform=ax.transAxes, ha="center", va="center", fontsize=11.5, color="#334155")


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color,
        )
    )


def plot_clinical_validation_flow() -> Path:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.suptitle("临床前验证流程与终点闭环", fontsize=24, weight="bold", y=0.97)

    _flow_box(ax, 0.05, 0.60, 0.18, 0.22, "受试者分组", "健康志愿者不少于10名\n外科医生不少于10名", "#dbeafe", "#60a5fa")
    _flow_box(ax, 0.30, 0.60, 0.18, 0.22, "实验条件", "无外骨骼\n姿态保持模式\n随动模式", "#ede9fe", "#8b5cf6")
    _flow_box(ax, 0.55, 0.52, 0.16, 0.30, "任务设置", "静态姿态保持\n动态轨迹跟随\n重复操作耐力\n精细操作模拟", "#dcfce7", "#22c55e")
    _flow_box(ax, 0.78, 0.58, 0.17, 0.24, "核心终点", "减负\n稳定性与精度\n疲劳发展\n可用性与安全性", "#fef3c7", "#f59e0b")

    _arrow(ax, (0.23, 0.71), (0.30, 0.71), "#64748b")
    _arrow(ax, (0.48, 0.71), (0.55, 0.67), "#64748b")
    _arrow(ax, (0.71, 0.67), (0.78, 0.70), "#64748b")

    _flow_box(ax, 0.12, 0.24, 0.20, 0.16, "实验环境", "实验室仿真\n模拟手术室", "#ffffff", "#cbd5e1")
    _flow_box(ax, 0.40, 0.24, 0.20, 0.16, "客观采集", "IMU\nEMG\n心率\n交互力与力矩", "#ffffff", "#cbd5e1")
    _flow_box(ax, 0.68, 0.24, 0.20, 0.16, "主观评价", "Borg 与 RPE\nNASA TLX\nSUS\n舒适度与安全反馈", "#ffffff", "#cbd5e1")

    _arrow(ax, (0.22, 0.60), (0.22, 0.40), "#94a3b8")
    _arrow(ax, (0.50, 0.52), (0.50, 0.40), "#94a3b8")
    _arrow(ax, (0.86, 0.58), (0.78, 0.40), "#94a3b8")

    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.05),
            0.90,
            0.09,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#0f172a",
            edgecolor="#1e293b",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.50,
        0.095,
        "最终形成：无外骨骼、随动、锁定三条件对照数据，验证肌电负荷、轨迹精度、疲劳发展、可用性与安全性。",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#f8fafc",
        weight="bold",
    )

    return _save(fig, "clinical_validation_flow.png")


def write_index(paths: list[Path]) -> Path:
    doc = Path("docs/中期答辩图表清单.md")
    lines = [
        "# 中期答辩图表清单",
        "",
        "以下图表基于中期答辩文案、临床前验证实验方案、标准数据记录表以及 `logs/jigsaws_intent/metrics.json` 的真实结果生成。",
        "",
        "## 新生成图表",
        "",
        f"1. [项目进展总览](/Users/maziang/Documents/方案一/{paths[0].as_posix()})：适合放在“概述项目进展情况”页。",
        f"2. [工作量与经费使用](/Users/maziang/Documents/方案一/{paths[1].as_posix()})：适合放在“投入工作量 / 经费”页。",
        f"3. [阶段性成果与关键指标](/Users/maziang/Documents/方案一/{paths[2].as_posix()})：适合放在“阶段性成果”页。",
        f"4. [阶段性成果汇总表](/Users/maziang/Documents/方案一/{paths[3].as_posix()})：适合放在“阶段性成果总结”页。",
        f"5. [临床验证任务矩阵](/Users/maziang/Documents/方案一/{paths[4].as_posix()})：适合放在“临床验证实验设计”页。",
        f"6. [临床验证流程图](/Users/maziang/Documents/方案一/{paths[5].as_posix()})：适合放在“验证闭环 / 下一阶段计划”页。",
        "",
        "## 现有可直接沿用图",
        "",
        "1. [总体结果报告](/Users/maziang/Documents/方案一/plots/jigsaws_intent_report.png)",
        "2. [锁定决策相图](/Users/maziang/Documents/方案一/plots/jigsaws_showcase/showcase_phase_map.png)",
        "3. [跨术者鲁棒性图](/Users/maziang/Documents/方案一/plots/jigsaws_showcase/showcase_surgeon_generalization.png)",
        "4. [任务-手势热图](/Users/maziang/Documents/方案一/plots/jigsaws_showcase/showcase_task_gesture_heatmap.png)",
        "5. [时间分镜图](/Users/maziang/Documents/方案一/plots/jigsaws_showcase/showcase_trial_storyboard.png)",
        "6. [ROC 总图](/Users/maziang/Documents/方案一/plots/jigsaws_roc/jigsaws_roc_overall.png)",
        "",
        "## 推荐答辩顺序",
        "",
        "1. 先用“项目进展总览”讲目前完成了什么。",
        "2. 再用“工作量与经费使用”说明投入与执行情况。",
        "3. 接着用“阶段性成果与关键指标”过渡到算法结果。",
        "4. 然后补充已有的相图、热图、分镜图、ROC 图展示模型细节。",
        "5. 最后用“临床验证任务矩阵”和“临床验证流程图”收束到下一阶段计划。",
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {doc}")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate midterm defense figures and tables.")
    parser.add_argument("--metrics", default="logs/jigsaws_intent/metrics.json")
    parser.add_argument("--predictions", default="logs/jigsaws_intent/window_predictions.csv")
    args = parser.parse_args()

    _setup_style()
    metrics, df, holdout = _load_data(Path(args.metrics), Path(args.predictions))

    paths = [
        plot_progress_board(metrics),
        plot_workload_budget(),
        plot_stage_results_dashboard(metrics, df, holdout),
        plot_stage_results_table(metrics),
        plot_clinical_validation_matrix(),
        plot_clinical_validation_flow(),
    ]
    write_index(paths)


if __name__ == "__main__":
    main()
