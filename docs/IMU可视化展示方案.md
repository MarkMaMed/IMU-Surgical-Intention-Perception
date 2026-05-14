# IMU 手术意态感知可视化展示方案（中期答辩）

## 1. 目标

用于回答三类问题：
- 模型是否可用（精度、鲁棒性）
- 模型如何工作（意图如何随时间变化）
- 系统如何落地（如何接入共享控制状态机）

## 2. 方案一：训练总览仪表板

图名：`plots/imu_dashboard.png`

内容：
- 类别窗口分布（是否偏置）
- 混淆矩阵（易混淆类别）
- 数据集分项精度（跨域泛化）
- 来源统计（`raw/kinematics/video_fallback`）

命令：

```bash
python3 -m eval.plot_imu_dashboard \
  --metrics logs/imu_intent_metrics_real.json \
  --output plots/imu_dashboard.png
```

建议讲解：
- “我们不仅看总体准确率，还看跨库表现和误分类结构。”

## 3. 方案二：单序列意图时间轴

图名：`plots/imu_timeline.png`

内容：
- 上图：窗口级 `pred_intent` 随时间变化
- 下图：模型置信度曲线
- 可叠加候选锁定/候选解锁标记点

命令：

```bash
python3 -m eval.plot_imu_timeline \
  --predictions logs/imu_intent_subject101_small_predictions.csv \
  --output plots/imu_timeline.png
```

建议讲解：
- “候选意图在时间上是连续出现的，不是孤立噪声点。”

## 4. 方案三：意图转移图

图名：`plots/imu_transition_graph.png`

内容：
- 节点：意图状态
- 有向边：状态切换方向
- 边权：切换频次

命令：

```bash
python3 -m eval.plot_intent_transition_graph \
  --predictions logs/imu_intent_subject101_small_predictions.csv \
  --output plots/imu_transition_graph.png
```

建议讲解：
- “从图上可以直观看到‘准备锁定 -> 精细操作 -> 准备解锁’的路径结构。”

## 5. 方案四：一页式答辩图库

页面：`plots/gallery.html`

命令：

```bash
python3 -m eval.build_visual_gallery \
  --plots-dir plots \
  --output plots/gallery.html
```

建议讲解：
- “答辩时可以一页浏览全部图，先看概览，再放大细节。”

## 6. 推荐答辩顺序（3分钟）

1. 仪表板：先讲“可用性”  
2. 时间轴：再讲“时序逻辑”  
3. 转移图：最后讲“状态机可对接性”  
4. 图库页：现场快速切换辅助问答

