# JIGSAWS 手术意态感知程序（锁定/不锁定建议）

本程序基于你已下载完成的 JIGSAWS（`Knot_Tying`、`Needle_Passing`、`Suturing`、`Experimental_setup`）训练“是否建议锁定”的意态模型。

## 1. 程序入口

- 训练与策略生成：`imu_intent/jigsaws_intent_program.py`
- 单条试验推理：`imu_intent/jigsaws_intent_infer.py`
- 高端可视化报告：`eval/plot_jigsaws_intent_report.py`
- 答辩展示专题图：`eval/plot_jigsaws_showcase.py`
- ROC 曲线图：`eval/plot_jigsaws_roc.py`

## 2. 核心逻辑

1. 读取每个 trial 的 `kinematics/AllGestures/*.txt` 与 `transcriptions/*.txt`  
2. 滑窗提取可解释运动学特征（全局速度、加速度、微动比例、双手速度对称性等）  
3. 基于手势与运动统计自动发现“锁定相关手势”  
4. 提取 151 个增强特征（速度/加速度/jerk 分布、稳定裕度、双手协同、时间形态、能量熵等），并引入任务上下文  
5. 自动比较 `RandomForest / ExtraTrees / XGBoost` 的区分能力  
6. 仅在训练集内部做术者分组交叉验证，基于 OOF 结果选择单模型或贪心集成，不看测试集  
7. 在最终概率上追加因果 `EMA` 时序平滑，进一步提升锁定与不锁定阶段的边界稳定性  
8. 输出决策规则：  
   - `P(lock)` 高 + `global_vel_mean` 低 -> `SUGGEST_LOCK`  
   - 否则 -> `SUGGEST_NO_LOCK`

## 3. 运行命令

```bash
python3 -m imu_intent.jigsaws_intent_program \
  --config config/jigsaws_intent.toml \
  --output-dir logs/jigsaws_intent \
  --model-out models/jigsaws_intent_model.joblib
```

输出：
- `logs/jigsaws_intent/metrics.json`
- `logs/jigsaws_intent/window_predictions.csv`
- `models/jigsaws_intent_model.joblib`

当前增强版训练结果：
- 严格训练集 OOF Accuracy：`0.8417`
- 严格训练集 OOF Macro-F1：`0.8266`
- 严格训练集 OOF ROC-AUC：`0.8958`
- 自动选中模型：`group_cv_greedy_ensemble`
- 最佳单模型：`xgboost`
- 最终部署组合：`xgboost + extra_trees`
- 时序增强：`EMA(alpha=0.75)`，最终阈值 `0.53`
- 说明：该版本不看测试集，只用训练集内的术者分组 OOF 结果做模型选择，再对全数据重拟合部署模型。

## 4. 生成可视化报告

```bash
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

## 5. 单条试验推理（术中示意）

```bash
python3 -m imu_intent.jigsaws_intent_infer \
  --model models/jigsaws_intent_model.joblib \
  --kinematics Suturing/kinematics/AllGestures/Suturing_B001.txt \
  --transcriptions Suturing/transcriptions/Suturing_B001.txt \
  --output logs/jigsaws_intent/infer_suturing_b001.csv \
  --summary logs/jigsaws_intent/infer_suturing_b001_summary.json
```

## 6. 结果解释模板（答辩）

- 需要锁定：精细稳态阶段，运动幅度下降，主要是微调动作  
- 不需要锁定：重定位、换手/换位、大幅动作或稳定性不足阶段
