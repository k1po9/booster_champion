# Analysis v1

本目录保存由本地不可变 `dataset/v1/*.jsonl` 生成的最新派生结果。原始数据体积约 330 MB，不提交 Git；报告记录输入文件、筛选方法、留出方式和模型参数，工具位于 `tools/`。

## 当前有效报告

1. `match_data_analysis.md/json`：五场比赛的整体质量、踢球和机器人运动概览。
2. `ball_motion_cross_validation.md/json`：按整场比赛留一的固定/线性/平方阻力模型比较。
3. `ball_motion_online_replay.md`：正式 `src/tactics/ball_prediction.py` 对87条干净自由滚动轨迹的实现级回放。
4. `robot_eta_cross_validation.md/json`：机器人巡航和转向响应实验；保留为基线，不提升为正式 ETA 能力。

## 最终采用结论

- 球模型候选：`a(v)=0.3791+0.1399v²`。
- 0.35 秒在线回放：median 0.537 m，p75 1.063 m。
- 0.50 秒在线回放：median 0.470 m，p75 0.902 m。
- 球模型只作为带置信度和不确定半径的 advisory 能力。
- ETA 实验得到 `translation_gain=0.8945`、`yaw_gain=0.8789`，但启动样本不足，因此最新策略改用动作看门狗与保守敌方压力时间。

## 复现命令

```bash
python3 tools/analyze_match_dataset.py --input dataset/v1 --output analysis/v1
python3 tools/cross_validate_ball_motion.py --input dataset/v1 --output analysis/v1
python3 tools/cross_validate_robot_eta.py --input dataset/v1 --output analysis/v1
```
