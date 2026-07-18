# Champion 自定义策略交接入口

本目录是 agent/champion-strategy 分支的最小完整交接文档。新对话按下面顺序阅读，即可了解当前模型、公共接口、策略接入方式、验证结论和已知限制，无需再查历史计划稿。

## 当前结论

- schema-v4 球路径模型已封装并接入 Champion 的追球目标，但只做置信度加权的短期引导。
- 机器人 ETA 已封装基线与 team2 两套参数，并用保守门控动态选择；ETA 只用于 Handler 排序和短期拦截，不是硬实时保证。
- Champion 已增加统一团队状态与排他 Ball Ownership。正常进攻是 Keeper + Handler + Outlet，防守切换为 Keeper + Pressurer + Marker；门将紧急接管时外场 Chaser 同帧撤销并转为 SecondBall。
- 所有模型使用固化的 Python 常量，比赛运行时不读取 JSON、数据集或第三方机器学习库。
- 预测无效、置信度不足、超出校准时域或机器人不可达时会立即退回几何基线；GameController、定位球、处罚、跌倒恢复和最终安全覆盖仍由既有行为树负责。
- 离线误差支持把模型作为软决策信号，但尚不足以证明比赛胜率提升。下一步应在仿真环境做 A/B 回放或对局验证。

## 阅读顺序

1. [模型说明](模型说明.md)：训练数据、算法、留出误差、动态门控和使用边界。
2. [预测接口说明](预测接口说明.md)：输入输出、调用方式、无效原因、时间语义和回退契约。
3. [当前冠军策略说明](当前冠军策略说明.md)：模型如何进入角色分配、追球、动作选择、诊断和安全链。

## 代码入口

| 目的 | 文件 |
| --- | --- |
| 球路径接口 | src/tactics/ball_trajectory.py |
| 球模型固化参数 | src/tactics/ball_event_model_data.py |
| ETA 与拦截接口 | src/tactics/robot_eta.py |
| ETA 双模型固化参数 | src/tactics/robot_eta_model_data.py |
| 球权事务与 Marker 纯逻辑 | src/tactics/team_coordination.py |
| tactics 公共导出 | src/tactics/__init__.py |
| Champion 策略 | src/play/champion.py |
| 球接口测试 | tests/test_ball_trajectory_v4.py |
| ETA 接口测试 | tests/test_robot_eta_interface.py |
| 团队协调纯测试 | tests/test_team_coordination.py |
| Champion 接入测试 | tests/test_champion_playbook.py |

模型训练报告保存在 analysis/v4/。原始 dataset/ 不属于比赛运行时依赖，也不应因为整理文档而提交。

## 验证基线

提交前至少运行：

~~~bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q src tests
git diff --check
~~~

本地若缺少项目依赖 py_trees，Champion 集成测试会显式跳过；纯模型和接口测试仍应通过。完整仿真环境不在本机，不能把本地单元测试等同于比赛效果验证。
