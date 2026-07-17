# 球路径与动态 ETA 接入说明

> 状态（2026-07-18）：两个模型已封装为纯 Python 公共接口，并以置信度门控方式接入 `ChampionPlaybook`。安全树、裁判状态、定位球和最终命令覆盖关系不变。

## 1. 公共接口

球路径位于 `src/tactics/ball_trajectory.py`：

```text
BallTrajectoryObservation
EventBallTrajectoryPredictor
BallTrajectoryPrediction
predict_launched_ball_path
```

预测需要检测到一次启动并收满五个新观测。输出提供相对
`observed_at_sec` 的 `position_at`、`velocity_at`、`confidence_at`、
`uncertainty_at`、有效时域、分段编号和失效原因。模型未覆盖最终自由停止，
因此不会伪造 `stop_point`。

机器人 ETA 位于 `src/tactics/robot_eta.py`：

```text
RobotArrivalQuery
RobotArrivalEstimate
DynamicRobotArrivalEstimator
RobotMotionTracker
InterceptEstimate
estimate_earliest_intercept
```

查询显式携带机器人位姿、原始/绕障目标、路径长度、公开位姿估计速度、
到达阈值、速度上限、运行/跌倒/踢球状态、目标年龄和目标是否变化。输出包含
ETA、最早/最晚区间、置信度、可达性、失效原因、所选模型、备用模型预测、
切换原因和可解释的运动组成。

## 2. 两个 ETA 参数集的动态选择

运行时不读取 JSON，也不在线训练。两个 ridge 参数集均固化在
`src/tactics/robot_eta_model_data.py`：

- 普通直达运动默认使用旧五局模型，它在新三局公平验证中的综合分数更好；
- 低速上限、大转角或明确绕障/加长路径，且 team2 域距离不高于 4.0、同时
  比旧模型域距离至少近 0.20 时，才使用八局 team2 模型；
- team2 特征域不匹配时立即回到旧模型；
- 两个模型的预测差异不会被隐藏，而是直接扩大 `uncertainty_sec` 和最晚到达界；
- 跌倒、恢复、踢球、非有限输入和非法路径明确返回不可达。

这不是逐帧投票。Handler 角色已有切换冷却和优势阈值，ETA 又以目标年龄和
目标位移判定是否换目标，因此模型变化不会直接造成角色抖动。

保守门控在新三局整场留出预测中只切换 51/2,892 个查询：中位误差
0.497→0.496 秒、P75 0.871→0.869 秒、P90 2.003→1.997 秒，综合分数
1.797→1.794。收益很小，所以仍按 advisory 使用，不把它解释成已达正式 ETA 门槛。

## 3. 当前 Champion 组合策略

每个合法 PLAYING tick 的决策顺序为：

1. 安全和数据层先过滤裁判阶段、处罚、过期球员和球观测；
2. 同时更新旧滚动球预测和 schema-v4 事件球路径；
3. 对最多两名场上 Handler 候选，以 `0.10/0.25/0.50 s` 三个固定球路点执行
   有界拦截搜索；
4. 有可信相遇点时，将当前球位向相遇点做置信度加权；否则依次回退到
   schema-v4 短期领先点、旧滚动球领先点、当前球位；
5. Handler 排序使用 `ETA + 0.20 × uncertainty + slot_bias`，再经过原有
   0.35 秒优势阈值和 0.60 秒切换冷却；
6. 选出的追球目标现在真正进入 `ChampionChaserRole` 移动闭环，不再只用于
   shadow 角色排序；
7. 敌方压力、AttackWatchdog、射门/传球/带球/解围候选、Outlet 和 Cover
   继续按冠军策略 V2 工作；
8. SafetyOverrides 最后仍可覆盖普通战术命令。

诊断输出新增 `ball_path`、`handler_eta` 和 `intercept`，可以直接看到模型名、
备用 ETA、置信度、不确定区间、相遇点和回退原因。

## 4. 当前精度与使用边界

八局 team2 挑战者的整局留出结果为：4,726 个查询，中位绝对误差
0.486 秒、P90 1.584 秒；原始目标起点中位误差 0.688 秒、P90 2.439 秒。
它在新三局上降低了全查询 P90，但综合分数没有击败旧模型。

因此当前 ETA 只允许参与：

- Handler 候选排序；
- 短时球路相遇点的软目标；
- shadow 诊断和后续数据验收。

它不得单独决定门将长距离出击、定位球是否合法、承诺必然先触球或绕过当前
球位回退。关闭 `enable_prediction`、`enable_robot_eta` 或
`enable_intercept` 后，各层可分别退回既有逻辑。
