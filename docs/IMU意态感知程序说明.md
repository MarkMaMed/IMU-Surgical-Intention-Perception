# IMU 手术意态感知程序说明（JIGSAWS + Opportunity + NinaPro + PAMAP2）

## 1. 目标

本模块用于根据公开数据集中的上肢动作时序信号，训练并推理术者意态：
- `PREPARE_LOCK`：准备稳定持位（可对应候选锁定）
- `PREPARE_UNLOCK`：准备重新移动（可对应候选解锁）
- `FINE_OPERATE`：精细操作过程
- `IDLE`：空闲/过渡状态

## 2. 模块结构

```text
imu_intent/
  loaders.py                 # 4个公开数据集加载器
  verify_jigsaws_layout.py   # JIGSAWS目录结构检查（kinematics/fallback判定）
  fetch_jigsaws_official.py  # 官方邮件链接下载与解压
  windowing.py               # 滑窗与标签多数投票
  features.py                # 可解释特征提取
  train_multidataset.py      # 多数据集联合训练
  infer_from_csv.py          # CSV离线推理
  stream_demo.py             # 类实时推理演示
  generate_mock_imu_csv.py   # 生成模拟IMU流
  synthetic.py               # 合成数据（无数据时快速自测）
config/
  imu_multidataset.toml      # 数据路径、映射和训练参数
```

## 3. 数据准备

请将公开数据集放到以下目录（可在配置文件中修改）：
- `data/public_imu/JIGSAWS`
- `data/public_imu/OpportunityUCIDataset/dataset`
- `data/public_imu/NinaPro`
- `data/public_imu/PAMAP2_Dataset/Protocol`

说明：
- 各数据集原始标签并非“手术意图标签”，本程序通过配置文件映射到统一意图空间。
- `config/imu_multidataset.toml` 已给出默认映射，建议根据你的实验定义再细化。
- 当公开镜像缺失 JIGSAWS 的 kinematics/transcriptions 时，程序会自动启用视频运动学兜底特征（仅用于流程训练与工程验证）。

### 3.1 JIGSAWS 原始 kinematics 的官方接入路径

JIGSAWS 官方入口为表单 + reCAPTCHA + 邮件链接，入口：
- `https://www.cs.jhu.edu/~los/jigsaws/info.php`

拿到邮件下载链接后，执行：

```bash
python3 -m imu_intent.fetch_jigsaws_official \
  --suturing-url "<官方邮件链接1>" \
  --knot-url "<官方邮件链接2>" \
  --needle-url "<官方邮件链接3>"
```

验证当前是否仍在 fallback：

```bash
python3 -m imu_intent.verify_jigsaws_layout \
  --config config/imu_multidataset.toml \
  --output logs/jigsaws_layout_check.json
```

## 4. 训练

```bash
python3 -m imu_intent.train_multidataset --config config/imu_multidataset.toml
```

若当前机器没有准备公开数据，可先做流水线自检：

```bash
python3 -m imu_intent.train_multidataset \
  --config config/imu_multidataset.toml \
  --use-synthetic-if-empty
```

输出：
- 模型：`models/imu_intent_multidataset.joblib`
- 指标：`logs/imu_intent_metrics.json`

训练日志会额外输出：
- `signal_source_distribution`：窗口级来源统计（`raw` / `kinematics` / `video_fallback`）

## 5. 推理

输入 CSV 需要包含 6 轴 IMU（支持别名）：
- `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

运行：

```bash
python3 -m imu_intent.infer_from_csv \
  --model models/imu_intent_multidataset.joblib \
  --input demo/mock_imu_stream.csv
```

输出：
- `logs/imu_intent_predictions.csv`
- `logs/imu_intent_predictions_summary.json`

## 6. 类实时演示

```bash
python3 -m imu_intent.stream_demo \
  --model models/imu_intent_multidataset.joblib \
  --input demo/mock_imu_stream.csv
```

终端会持续输出窗口级意图和置信度，可用于与你现有状态机做联调。

## 7. 与现有共享控制系统对接建议

可将推理结果映射为：
- `PREPARE_LOCK` -> `lock_candidate_signal = True`
- `PREPARE_UNLOCK` -> `unlock_candidate_signal = True`
- 其余状态 -> 不触发候选

再继续沿用你现有的脚踏确认和状态机执行逻辑，实现“IMU候选 + 脚踏确认”的共享控制。

## 8. 可视化展示建议（答辩可直接用）

### 方案A：训练总览仪表板
- 展示类别分布、混淆矩阵、跨数据集精度、来源统计

```bash
python3 -m eval.plot_imu_dashboard \
  --metrics logs/imu_intent_metrics_real.json \
  --output plots/imu_dashboard.png
```

### 方案B：单次试验意图时间轴
- 展示窗口级意图标签、置信度、候选锁定/候选解锁点位

```bash
python3 -m eval.plot_imu_timeline \
  --predictions logs/imu_intent_subject101_small_predictions.csv \
  --output plots/imu_timeline.png
```

### 方案C：意图转移网络图
- 展示不同意图间的切换方向和频次，适合讲“状态演化”

```bash
python3 -m eval.plot_intent_transition_graph \
  --predictions logs/imu_intent_subject101_small_predictions.csv \
  --output plots/imu_transition_graph.png
```

### 方案D：一页式图库
- 把多张图整合到网页，答辩现场一屏展示

```bash
python3 -m eval.build_visual_gallery \
  --plots-dir plots \
  --output plots/gallery.html
```
