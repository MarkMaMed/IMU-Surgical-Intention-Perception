# 方案一（IMU 手术意态感知）

本仓库现已清理视觉识别意图相关模块，当前仅保留 IMU 多数据集意态感知流程（JIGSAWS、Opportunity、NinaPro、PAMAP2）。

## 目录

```text
config/
  imu_multidataset.toml
imu_intent/
  loaders.py
  train_multidataset.py
  infer_from_csv.py
  stream_demo.py
  verify_jigsaws_layout.py
  fetch_jigsaws_official.py
eval/
  plot_imu_dashboard.py
  plot_imu_timeline.py
  plot_intent_transition_graph.py
  build_visual_gallery.py
docs/
  IMU意态感知程序说明.md
  IMU可视化展示方案.md
  JIGSAWS官方下载接入说明.md
```

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 训练

```bash
python3 -m imu_intent.train_multidataset \
  --config config/imu_multidataset.toml \
  --model-out models/imu_intent_multidataset_real.joblib \
  --metrics-out logs/imu_intent_metrics_real.json
```

## 推理

```bash
python3 -m imu_intent.infer_from_csv \
  --model models/imu_intent_multidataset_real.joblib \
  --input data/public_imu/PAMAP2_Dataset/Protocol/subject101_small.csv \
  --output logs/imu_intent_subject101_small_predictions.csv \
  --summary logs/imu_intent_subject101_small_summary.json
```

## 可视化

```bash
python3 -m eval.plot_imu_dashboard --metrics logs/imu_intent_metrics_real.json --output plots/imu_dashboard.png
python3 -m eval.plot_imu_timeline --predictions logs/imu_intent_subject101_small_predictions.csv --output plots/imu_timeline.png
python3 -m eval.plot_intent_transition_graph --predictions logs/imu_intent_subject101_small_predictions.csv --output plots/imu_transition_graph.png
python3 -m eval.build_visual_gallery --plots-dir plots --output plots/gallery.html
```

## JIGSAWS 手术意态感知（锁定/不锁定）

增强版说明：
- 使用 151 个可解释运动学特征（速度/加速度/jerk 分布、双手协同、时间形态与能量熵）
- 引入手术任务上下文（`Knot_Tying` / `Needle_Passing` / `Suturing`）
- 自动比较 `RandomForest`、`ExtraTrees`、`XGBoost`，并做分组交叉验证下的贪心集成
- 在不看测试集的前提下，仅用训练集做术者分组 OOF 选择，再用全数据重拟合部署模型
- 当前严格 OOF 结果：Accuracy `0.8417`，Macro-F1 `0.8266`，ROC-AUC `0.8958`
- 当前自动选中方案：`XGBoost + ExtraTrees` 贪心集成，叠加因果 `EMA(alpha=0.75)` 时序平滑

```bash
python3 -m imu_intent.jigsaws_intent_program \
  --config config/jigsaws_intent.toml \
  --output-dir logs/jigsaws_intent \
  --model-out models/jigsaws_intent_model.joblib

python3 -m eval.plot_jigsaws_intent_report \
  --metrics logs/jigsaws_intent/metrics.json \
  --predictions logs/jigsaws_intent/window_predictions.csv \
  --output plots/jigsaws_intent_report.png

python3 -m eval.plot_jigsaws_showcase \
  --metrics logs/jigsaws_intent/metrics.json \
  --predictions logs/jigsaws_intent/window_predictions.csv \
  --output-dir plots/jigsaws_showcase

python3 -m eval.plot_jigsaws_roc \
  --predictions logs/jigsaws_intent/window_predictions.csv \
  --metrics logs/jigsaws_intent/metrics.json \
  --output-dir plots/jigsaws_roc
```

详细说明见：
- `docs/JIGSAWS手术意态感知程序.md`

## JIGSAWS 原始 kinematics 接入

官方入口：
- [https://www.cs.jhu.edu/~los/jigsaws/info.php](https://www.cs.jhu.edu/~los/jigsaws/info.php)

收到官方邮件下载链接后：

```bash
python3 -m imu_intent.fetch_jigsaws_official \
  --suturing-url "<官方邮件链接1>" \
  --knot-url "<官方邮件链接2>" \
  --needle-url "<官方邮件链接3>"

python3 -m imu_intent.verify_jigsaws_layout \
  --config config/imu_multidataset.toml \
  --output logs/jigsaws_layout_check.json
```
