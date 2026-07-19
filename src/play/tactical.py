"""Central tactical orchestration for the dynamic 3v3 triangle.

================================================================================
战术编排设计文档 (Tactical Orchestration Design Document)
================================================================================

【整体架构】
本文件实现了动态三角战术系统 (Dynamic Triangle Tactics)，是机器人足球 3v3 比赛
的核心决策引擎。系统采用分层架构：

  1. Playbook 层 (playbook.py): 最高层策略入口，负责角色注册和分配
  2. Tactical 层 (本文件): 战术协调器，每 tick 生成全局战术计划
  3. Role 层 (dynamic_roles.py): 具体角色行为，根据战术计划执行动作
  4. Tactics 层 (src/tactics/): 底层算法库（球预测、压力评估、进攻监控等）
  5. Soccer Framework 层 (src/soccer_framework/): 基础数据结构和配置

【数据流】
  PlayContext (每tick输入)
       ↓
  DynamicTriangleCoordinator.update()
       ↓
  TacticalContext (每tick输出，不可变快照)
       ↓
  PrimaryRole / SecondaryRole / SafetyRole (读取并执行)

【战术模式 (TacticalMode)】
  - ATTACK: 控球进攻，我方距离球最近且有优势
  - CONTEST: 争夺球权，双方距离相近
  - DEFEND: 防守站位，对方有优势
  - EMERGENCY_DEFEND: 紧急防守，球已进入危险区域
  - RESTART: 重新开始（定位球、犯规后等）

【角色分配 (Role Assignment)】
  - PRIMARY (主要): 负责追球/射门/传球的核心球员
  - SECONDARY (次要): 接应/支援球员，准备接传球
  - SAFETY (安全): 防守保护球员，防止对方反击

【Primary 选择机制】
  使用"成本评估 + 防抖动"机制：
  1. 计算每个球员到达目标点的成本（距离/速度 + 转向时间）
  2. 选择成本最低的球员作为 Primary
  3. 切换需要连续 N 个 tick 确认（防抖动），避免频繁切换

【进攻意图 (PrimaryIntent)】
  - APPROACH: 接近球
  - CHALLENGE: 争抢球
  - INTERCEPT: 拦截滚动中的球
  - SHOOT: 射门（球门方向清晰时）
  - PASS: 传球（给 Secondary 的接应点）
  - DRIBBLE: 带球推进
  - PROGRESSIVE_TOUCH: 轻触推进（紧急避压或边线情况）
  - PRESS: 压迫对方
  - CLEAR: 解围（紧急防守时）

【涉及的文件】
  - src/play/playbook.py: DynamicTrianglePlaybook 调用 coordinator.update()
  - src/play/dynamic_roles.py: PrimaryRole/SecondaryRole/SafetyRole 读取 TacticalContext
  - src/play/role.py: RoleStrategy 基类
  - src/tactics/ball_prediction.py: SlidingWindowBallPredictor 球运动预测
  - src/tactics/attack_watchdog.py: AttackWatchdog 监控进攻是否停滞
  - src/tactics/opponent_pressure.py: OpponentPressureEstimator 评估对方压力
  - src/tactics/geometry.py: clamp, normalize_angle 几何工具函数
  - src/tactics/targeting.py: Targeting 提供射门/传球/带球目标计算
  - src/tactics/motion.py: MotionController 运动控制
  - src/soccer_framework/types.py: BallState, Pose2D, PlayContext 等数据结构
  - src/soccer_framework/config.py: SoccerConfig, SoccerStrategyTuning 配置参数
  - src/runtime.py: SoccerKit 组装所有组件
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from ..soccer_framework import BallState, Pose2D, ReadySlot, SetPlay, PlayContext
from ..tactics import (
    AttackAttemptObservation,  # 进攻尝试观测数据
    AttackPhase,               # 进攻阶段（对齐/执行）
    AttackWatchdog,            # 进攻监控器，检测进攻是否停滞
    BallMotionPrediction,      # 球运动预测结果
    BallObservation,           # 单次球观测数据
    OpponentPressureEstimator, # 对方压力评估器
    SlidingWindowBallPredictor,# 滑动窗口球预测器
)
from ..tactics.geometry import clamp, normalize_angle  # 几何工具：数值限制、角度归一化

if TYPE_CHECKING:
    from ..runtime import SoccerKit  # 仅在类型检查时导入，避免循环依赖


# ============================================================================
# 枚举定义
# ============================================================================

class TacticalMode(str, Enum):
    """战术模式：定义当前团队整体态势。
    
    模式选择基于球位置、双方距离比较、比赛状态等因素。
    每个模式会影响角色目标点选择和 Primary 的行为意图。
    """
    ATTACK = "attack"              # 进攻模式：我方控球，组织进攻
    CONTEST = "contest"            # 争夺模式：双方距离相近，争夺球权
    DEFEND = "defend"              # 防守模式：对方有优势，组织防守
    EMERGENCY_DEFEND = "emergency_defend"  # 紧急防守：球已进入我方危险区域
    RESTART = "restart"            # 重新开始：定位球、犯规后等暂停状态


class PrimaryIntent(str, Enum):
    """Primary 球员的行动意图：决定当前应该做什么。
    
    意图选择是战术决策的核心，综合考虑：
    - 当前战术模式
    - 球门方向是否清晰
    - 传球路线是否畅通
    - 对方压力大小
    - 球是否在边线附近
    """
    APPROACH = "approach"                    # 接近球
    CHALLENGE = "challenge"                  # 争抢球
    INTERCEPT = "intercept"                  # 拦截滚动中的球
    SHOOT = "shoot"                          # 射门
    PASS = "pass"                            # 传球
    DRIBBLE = "dribble"                      # 带球推进
    PROGRESSIVE_TOUCH = "progressive_touch"  # 轻触推进（小幅度踢球前进）
    PRESS = "press"                          # 压迫对方
    CLEAR = "clear"                          # 解围（大脚踢出危险区）


# ============================================================================
# 角色常量
# ============================================================================

ROLE_PRIMARY = "primary"      # 主要角色：负责追球和执行进攻动作
ROLE_SECONDARY = "secondary"  # 次要角色：接应传球，提供支援
ROLE_SAFETY = "safety"        # 安全角色：防守保护，防止反击


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass(frozen=True)
class TacticalContext:
    """战术上下文：每个 tick 生成的不可变团队战术计划快照。
    
    这是一个冻结的数据类（frozen=True），一旦创建就不能修改，确保
    所有角色读取到一致的战术视图。
    
    属性说明：
    - mode: 当前战术模式
    - primary_id/secondary_id/safety_id: 各角色的球员 ID
    - primary_intent: Primary 球员的行动意图
    - primary_target: Primary 应该移动到的目标位置
    - action_target: 实际动作目标（射门点/传球点/带球方向）
    - secondary_target: Secondary 的目标位置
    - safety_target: Safety 的目标位置
    - receive_target: Secondary 的接球点（用于传球配合）
    - secondary_ready: Secondary 是否已到达接球位置
    - counterattack_risk: 反击风险评分 (0.0-1.0)
    - ball_prediction: 球的运动预测
    - pressure_time_sec: 对方施加压力的时间（秒），越小越危险
    """
    mode: TacticalMode                           # 当前战术模式
    primary_id: int | None                       # Primary 球员 ID
    secondary_id: int | None                     # Secondary 球员 ID
    safety_id: int | None                        # Safety 球员 ID
    primary_intent: PrimaryIntent                # Primary 行动意图
    primary_target: Pose2D                       # Primary 移动目标点
    action_target: Pose2D                        # 动作执行目标点
    secondary_target: Pose2D                     # Secondary 移动目标点
    safety_target: Pose2D                        # Safety 移动目标点
    receive_target: Pose2D | None = None         # Secondary 接球点
    secondary_ready: bool = False                # Secondary 是否就位
    counterattack_risk: float = 0.0              # 反击风险 (0-1)
    ball_prediction: BallMotionPrediction | None = None  # 球预测
    pressure_time_sec: float | None = None       # 压力时间（秒）

    def role_of(self, player_id: int) -> str:
        """查询指定球员 ID 对应的角色。
        
        Args:
            player_id: 球员 ID
            
        Returns:
            角色字符串：ROLE_PRIMARY / ROLE_SECONDARY / ROLE_SAFETY / "none"
        """
        if player_id == self.primary_id:
            return ROLE_PRIMARY
        if player_id == self.secondary_id:
            return ROLE_SECONDARY
        if player_id == self.safety_id:
            return ROLE_SAFETY
        return "none"


@dataclass
class _SwitchState:
    """Primary 切换状态：用于实现防抖动机制。
    
    为避免 Primary 角色在球员之间频繁切换，系统使用确认机制：
    1. 当发现更好的候选者时，记录为 challenger
    2. 连续 N 个 tick 都确认该候选者更优，才执行切换
    3. N 由 strategy.role_switch_confirm_ticks 配置
    
    属性说明：
    - current: 当前 Primary 球员 ID
    - challenger: 挑战者球员 ID（可能成为新的 Primary）
    - ticks: 挑战者连续确认的 tick 数
    """
    current: int | None = None      # 当前 Primary
    challenger: int | None = None   # 挑战者
    ticks: int = 0                  # 连续确认计数


# ============================================================================
# 核心协调器
# ============================================================================

class DynamicTriangleCoordinator:
    """动态三角战术协调器：每 tick 生成全局战术计划并稳定 Primary 切换。
    
    职责：
    1. 更新球运动预测（滑动窗口）
    2. 选择当前战术模式
    3. 选择 Primary 球员（带防抖动）
    4. 分配 Secondary 和 Safety 角色
    5. 评估对方压力
    6. 选择 Primary 的行动意图
    7. 计算所有角色的目标位置
    
    输出：
    - TacticalContext：不可变的战术计划快照
    - 供 PrimaryRole/SecondaryRole/SafetyRole 读取执行
    
    设计原则：
    - 每 tick 重新计算，但通过防抖动保持稳定
    - 所有计算基于当前 PlayContext，无隐藏状态（除 SwitchState）
    - 目标点计算考虑场地边界和障碍物
    """

    def __init__(self, kit: "SoccerKit"):
        """初始化协调器。
        
        Args:
            kit: SoccerKit 实例，提供配置、场地信息、目标计算工具等
        """
        self.kit = kit
        # 球预测器：使用 0.45 秒滑动窗口平滑观测
        self._ball = SlidingWindowBallPredictor(window_sec=0.45)
        # 压力评估器：评估对方球员对球的威胁
        self._pressure = OpponentPressureEstimator()
        # 进攻监控器：检测进攻是否停滞（长时间未进展）
        self._watchdog = AttackWatchdog()
        # Primary 切换状态：防抖动
        self._primary_switch = _SwitchState()
        # 上一次生成的战术上下文
        self.last_context: TacticalContext | None = None
        # 我方中圈开球第一脚状态
        self._kickoff_start_ball: Pose2D | None = None
        self._kickoff_first_touch_done: bool = False

    def update(self, context: PlayContext, now_sec: float) -> TacticalContext:
        """主更新方法：每 tick 调用，生成新的战术计划。
        
        执行流程：
        1. 更新球观测和预测
        2. 选择战术模式
        3. 获取可用球员列表
        4. 选择 Primary
        5. 分配 Secondary 和 Safety
        6. 评估对方压力
        7. 计算接球点
        8. 选择 Primary 行动意图
        9. 检测进攻是否停滞（如停滞则改为轻触推进）
        10. 计算所有目标点
        11. 计算反击风险
        12. 返回 TacticalContext
        
        Args:
            context: 当前比赛上下文（包含球、队友、对手、比赛状态等）
            now_sec: 当前时间（秒）
            
        Returns:
            TacticalContext: 不可变的战术计划快照
        """
        # --- 步骤 1: 更新球预测 ---
        ball = context.known_ball
        # 使用时间戳：如果有最近观测用观测时间，否则用当前时间
        stamp = ball.last_seen_at if ball.last_seen_at > 0.0 else now_sec
        self._ball.add_observation(
            BallObservation(stamp, ball.x, ball.y, ball.confidence)
        )
        prediction = self._ball.predict()  # 获取当前球运动预测

        # 更新我方中圈开球第一脚状态
        self._update_kickoff_state(context)

        # --- 步骤 2: 选择战术模式 ---
        mode = self._select_mode(context)
        
        # --- 步骤 3: 获取可用球员 ---
        active = self._active_players(context)
        
        # --- 步骤 4: 选择 Primary ---
        primary = self._select_primary(active, context, mode, prediction)
        
        # --- 步骤 5: 分配支援角色 ---
        secondary, safety = self._assign_support_slots(active, primary, context)

        # --- 步骤 6: 评估对方压力 ---
        pressure = self._pressure.estimate(
            now_sec=now_sec,
            opponents=context.opponents,
            ball_prediction=prediction,
            target=None if prediction is not None else Pose2D(ball.x, ball.y),
            game_state=context.game_state,
            opponent_team_id=self.kit.config.opponent_team_id(),
        )
        pressure_time = pressure.pressure_time_sec  # 压力时间，越小越危险
        
        # --- 步骤 7: 计算接球点 ---
        receive = self._receive_target(secondary, context, mode)
        ready = self._is_ready(secondary, receive, context)  # Secondary 是否就位
        
        # --- 步骤 8: 选择 Primary 行动意图 ---
        intent, action_target = self._select_primary_action(
            secondary, mode, context, receive, pressure_time
        )
        
        # --- 步骤 9: 检测进攻停滞 ---
        if (
            mode != TacticalMode.RESTART
            and self._attack_is_stalled(
                primary,
                intent,
                action_target,
                context,
                prediction,
                now_sec,
            )
        ):
            # 进攻停滞：改为轻触推进，尝试打破僵局
            intent = PrimaryIntent.PROGRESSIVE_TOUCH
            action_target = self.kit.targeting.dribble_target(ball)
            
        # --- 步骤 10: 计算目标点 ---
        primary_target = self._primary_claim_target(
            intent, action_target, context, prediction
        )
        secondary_target = self._secondary_target(
            secondary, mode, intent, action_target, receive, context, prediction
        )
        
        # --- 步骤 11: 计算反击风险 ---
        risk = self._counterattack_risk(mode, ball, pressure_time)
        safety_target = self._safety_target(mode, risk, context)

        # --- 步骤 12: 生成快照 ---
        snapshot = TacticalContext(
            mode=mode,
            primary_id=primary,
            secondary_id=secondary,
            safety_id=safety,
            primary_intent=intent,
            primary_target=primary_target,
            action_target=action_target,
            secondary_target=secondary_target,
            safety_target=safety_target,
            receive_target=receive,
            secondary_ready=ready,
            counterattack_risk=risk,
            ball_prediction=prediction,
            pressure_time_sec=pressure_time,
        )
        self.last_context = snapshot
        return snapshot

    def _active_players(self, context: PlayContext) -> list[int]:
        """获取当前可用的球员 ID 列表。
        
        优先返回有姿态数据（pose 不为 None）的球员，
        如果没有则返回所有允许的球员。
        
        Args:
            context: 比赛上下文
            
        Returns:
            可用球员 ID 列表
        """
        game = context.known_game
        # 有姿态数据的球员（可以精确定位）
        visible = [
            player_id
            for player_id in self.kit.config.player_ids
            if self.kit.is_player_allowed(game, player_id)  # 允许上场
            and context.teammates.get(player_id) is not None  # 有队友数据
            and context.teammates[player_id].pose is not None  # 有姿态数据
        ]
        if visible:
            return visible
        # 降级：返回所有允许上场的球员
        return [
            player_id
            for player_id in self.kit.config.player_ids
            if self.kit.is_player_allowed(game, player_id)
        ]

    def _select_mode(self, context: PlayContext) -> TacticalMode:
        """选择当前战术模式。
        
        决策逻辑（按优先级）：
        1. 如果有定位球 -> RESTART
        2. 如果球 x 坐标 <= 紧急防守线 -> EMERGENCY_DEFEND
        3. 如果对方距离 + 优势 < 我方距离 -> DEFEND（对方控球）
        4. 如果我方距离 <= 控制半径 且 我方距离 + 优势 < 对方距离 -> ATTACK（我方控球）
        5. 如果球 x 坐标 <= 防守线 -> DEFEND
        6. 否则 -> CONTEST（争夺）
        
        Args:
            context: 比赛上下文
            
        Returns:
            TacticalMode: 选定的战术模式
        """
        game = context.known_game
        ball = context.known_ball
        
        # 定位球状态 -> 重新开始
        if game.set_play != SetPlay.NONE:
            is_our_kickoff = (
                game.set_play == SetPlay.KICKOFF
                and game.is_kickoff_for_team(self.kit.config.team_id)
            )

            # 我方中圈开球：
            # 只有第一脚完成前保持 RESTART。
            if is_our_kickoff:
                if not self._kickoff_first_touch_done:
                    return TacticalMode.RESTART

                # 第一脚已经完成：
                # 不再返回 RESTART，继续下面的常规态势判断。
            else:
                # 其他定位球暂时保持原来的 RESTART 处理
                return TacticalMode.RESTART
            
        tuning = self.kit.config.strategy
        
        # 球进入危险区域 -> 紧急防守
        if ball.x <= tuning.emergency_defend_x_m:
            return TacticalMode.EMERGENCY_DEFEND
            
        # 计算双方到球的最近距离
        our_distance = self._nearest_distance(context.teammates, ball)
        their_distance = self._nearest_distance(context.opponents, ball)
        
        # 对方有明显优势 -> 防守
        if their_distance + tuning.possession_advantage_m < our_distance:
            return TacticalMode.DEFEND
            
        # 我方有控球优势 -> 进攻
        if (
            our_distance <= tuning.possession_control_radius_m
            and our_distance + tuning.possession_advantage_m < their_distance
        ):
            return TacticalMode.ATTACK
            
        # 球在我方半场 -> 防守
        if ball.x <= tuning.defend_x_m:
            return TacticalMode.DEFEND
            
        # 默认：争夺
        return TacticalMode.CONTEST

    @staticmethod
    def _nearest_distance(robots: dict, ball: BallState) -> float:
        """计算一群机器人到球的最短距离。
        
        Args:
            robots: 机器人字典 {player_id: RobotState}
            ball: 球状态
            
        Returns:
            最短距离，如果没有有效机器人则返回 inf
        """
        distances = [
            math.hypot(robot.pose.x - ball.x, robot.pose.y - ball.y)
            for robot in robots.values()
            if robot.pose is not None
        ]
        return min(distances, default=math.inf)

    def _select_primary(
        self,
        active: list[int],
        context: PlayContext,
        mode: TacticalMode,
        prediction: BallMotionPrediction | None,
    ) -> int | None:
        """选择 Primary 球员（带防抖动机制）。
        
        选择流程：
        1. 计算拦截点（球预测位置）
        2. 对每个活跃球员计算到达成本
        3. 选择成本最低的球员
        4. 防抖动：如果挑战者连续 N tick 更优，才切换
        
        Args:
            active: 可用球员 ID 列表
            context: 比赛上下文
            mode: 当前战术模式
            prediction: 球运动预测
            
        Returns:
            Primary 球员 ID，如果没有可用球员则返回 None
        """
        if not active:
            self._primary_switch = _SwitchState()
            return None
            
        # 计算拦截点（球预计到达的位置）
        point = self._intercept_point(context.known_ball, prediction)
        
        # 按成本排序（成本越低越好）
        scored = sorted(
            (self._claim_cost(player_id, point, context, mode), player_id)
            for player_id in active
        )
        best_cost, best = scored[0]  # 最佳候选
        
        state = self._primary_switch
        
        # 如果当前 Primary 不在活跃列表中，直接选择最佳
        if state.current not in active:
            self._primary_switch = _SwitchState(current=best)
            return best
            
        # 获取当前 Primary 的成本
        current_cost = next(cost for cost, pid in scored if pid == state.current)
        tuning = self.kit.config.strategy
        
        # 如果当前 Primary 仍然最优，或者优势不够大，保持当前
        if (
            best == state.current
            or best_cost + tuning.role_switch_advantage_sec >= current_cost
        ):
            state.challenger = None
            state.ticks = 0
            return state.current
            
        # 挑战者确认计数
        if state.challenger == best:
            state.ticks += 1  # 连续确认
        else:
            state.challenger = best  # 新挑战者
            state.ticks = 1
            
        # 达到确认阈值，执行切换
        if state.ticks >= max(1, tuning.role_switch_confirm_ticks):
            state.current = best
            state.challenger = None
            state.ticks = 0
            
        return state.current

    def _claim_cost(
        self,
        player_id: int,
        point: Pose2D,
        context: PlayContext,
        mode: TacticalMode,
    ) -> float:
        """计算球员到达目标点的成本。
        
        成本 = 移动时间 + 转向时间 * 0.35 + 守门员惩罚
        
        Args:
            player_id: 球员 ID
            point: 目标点
            context: 比赛上下文
            mode: 战术模式
            
        Returns:
            成本值（秒），越低越好
        """
        robot = context.teammates.get(player_id)
        if robot is None or robot.pose is None:
            return math.inf  # 无效球员，成本无限大
            
        pose = robot.pose
        tuning = self.kit.config.strategy
        
        # 计算速度参数
        speed = max(0.1, tuning.max_linear_speed * tuning.robot_translation_gain)
        yaw_speed = max(0.1, tuning.max_angular_speed * tuning.robot_yaw_gain)
        
        # 计算到目标点的方位角
        bearing = math.atan2(point.y - pose.y, point.x - pose.x)
        
        # 移动时间成本
        cost = math.hypot(point.x - pose.x, point.y - pose.y) / speed
        
        # 转向时间成本（权重 0.35）
        cost += (
            abs(normalize_angle(bearing - pose.theta)) / yaw_speed * 0.35
        )
        
        # 守门员惩罚：非防守模式下，守门员追球成本更高
        is_keeper = (
            self.kit.config.ready_slot_for_player(player_id) == ReadySlot.KEEPER
        )
        if is_keeper and mode not in {
            TacticalMode.EMERGENCY_DEFEND,
            TacticalMode.DEFEND,
        }:
            cost += 2.0  # 守门员惩罚，鼓励其他球员追球
            
        return cost

    def _assign_support_slots(
        self,
        active: list[int],
        primary: int | None,
        context: PlayContext,
    ) -> tuple[int | None, int | None]:
        """分配 Secondary 和 Safety 角色。
        
        分配策略：
        - 3 人：Primary + Secondary + Safety（完整三角）
        - 2 人：Primary + Safety（Secondary 合并到 Safety）
        - 1 人：Primary + Safety（Safety 由 Primary 兼任）
        
        Safety 选择：
        - 优先选择配置的守门员
        - 否则选择 x 坐标最小（最靠后）的球员
        
        Args:
            active: 可用球员 ID 列表
            primary: Primary 球员 ID
            context: 比赛上下文
            
        Returns:
            (secondary_id, safety_id) 元组
        """
        remaining = [player_id for player_id in active if player_id != primary]
        
        # 1 人情况：Primary 兼任 Safety
        if not remaining:
            return None, primary
            
        # 2 人情况：只有 Primary + Safety
        if len(remaining) == 1:
            return None, remaining[0]
            
        # 3 人情况：完整三角
        keeper = self.kit.config.goalkeeper_player_id()
        # 选择 Safety：优先守门员，否则选择最靠后的球员
        safety = keeper if keeper in remaining else min(
            remaining,
            key=lambda player_id: (
                context.teammates[player_id].pose.x
                if context.teammates.get(player_id)
                and context.teammates[player_id].pose
                else math.inf
            ),
        )
        # 剩余的就是 Secondary
        secondary = next(player_id for player_id in remaining if player_id != safety)
        return secondary, safety

    def _intercept_point(
        self,
        ball: BallState,
        prediction: BallMotionPrediction | None,
    ) -> Pose2D:
        """计算拦截点：球预计到达的位置。
        
        如果预测不可用或置信度低，返回球当前位置。
        
        Args:
            ball: 球状态
            prediction: 球运动预测
            
        Returns:
            拦截点坐标（已限制在场地内）
        """
        if (
            prediction is None
            or prediction.confidence < 0.35  # 置信度阈值
            or prediction.motion_state != "rolling"  # 只预测滚动中的球
        ):
            return Pose2D(ball.x, ball.y)
            
        # 预测未来位置
        point = prediction.position_at(
            self.kit.config.strategy.intercept_prediction_horizon_sec
        )
        return self.kit.field.clamp_inside_field(point)  # 限制在场地内

    def _receive_target(
        self,
        secondary: int | None,
        context: PlayContext,
        mode: TacticalMode,
    ) -> Pose2D | None:
        """计算 Secondary 的接球点。
        
        接球点设计：
        - 在球前方（对方球门方向）
        - 偏向一侧（根据球员 ID 奇偶性）
        - 限制在场地边界内
        
        Args:
            secondary: Secondary 球员 ID
            context: 比赛上下文
            mode: 战术模式
            
        Returns:
            接球点坐标，如果不需要接球则返回 None
        """
        if secondary is None or mode not in {
            TacticalMode.ATTACK,
            TacticalMode.RESTART,
        }:
            return None
            
        ball = context.known_ball
        robot = context.teammates.get(secondary)
        
        # 计算偏移方向（奇偶性决定左右）
        side = 1.0 if secondary % 2 == 0 else -1.0
        base_y = (
            robot.pose.y
            if robot is not None and robot.pose is not None
            else ball.y + side
        )
        
        # 接球点 x：在球前方，靠近对方球门
        x = min(self.kit.field.opponent_goal_x() - 0.8, ball.x + 1.4)
        # 接球点 y：基于球员位置，加上侧向偏移
        y = clamp(
            base_y + side * 0.35,
            -self.kit.config.field_width / 2 + 0.55,
            self.kit.config.field_width / 2 - 0.55,
        )
        
        # 计算朝向：面向球
        return self.kit.field.clamp_inside_field(
            Pose2D(x, y, math.atan2(ball.y - y, ball.x - x))
        )

    def _is_ready(
        self,
        secondary: int | None,
        target: Pose2D | None,
        context: PlayContext,
    ) -> bool:
        """检查 Secondary 是否已到达接球点。
        
        Args:
            secondary: Secondary 球员 ID
            target: 接球点
            context: 比赛上下文
            
        Returns:
            是否已到达（距离 <= 接球半径）
        """
        if secondary is None or target is None:
            return False
            
        robot = context.teammates.get(secondary)
        return (
            robot is not None
            and robot.pose is not None
            and math.hypot(robot.pose.x - target.x, robot.pose.y - target.y)
            <= self.kit.config.strategy.secondary_receive_radius_m
        )

    def _update_kickoff_state(self, context: PlayContext) -> None:
        """跟踪我方中圈开球第一脚是否已经完成。"""
        game = context.known_game
        ball = context.known_ball

        is_our_kickoff = (
            game.set_play == SetPlay.KICKOFF
            and game.is_kickoff_for_team(self.kit.config.team_id)
        )

        # 不再处于我方开球，清空状态，等待下一次开球
        if not is_our_kickoff:
            self._kickoff_start_ball = None
            self._kickoff_first_touch_done = False
            return

        # 刚进入一次新的我方开球
        if self._kickoff_start_ball is None:
            self._kickoff_start_ball = Pose2D(ball.x, ball.y)
            self._kickoff_first_touch_done = False
            return

        # 第一脚完成后保持 done，直到本次 KICKOFF 结束
        if self._kickoff_first_touch_done:
            return

        moved = math.hypot(
            ball.x - self._kickoff_start_ball.x,
            ball.y - self._kickoff_start_ball.y,
        )

        if moved >= self.kit.config.strategy.restart_touch_distance:
            self._kickoff_first_touch_done = True

    def _select_kickoff_target(
        self,
        context: PlayContext,
    ) -> Pose2D:
        """选择我方中圈开球第一脚的斜向推进目标。

        只比较左前和右前两个方向，选择对手阻挡更少的一侧。
        """
        ball = context.known_ball
        opponents = self.kit.obstacles.opponent_obstacles(context)

        candidates = [
            Pose2D(
                ball.x + 1.2,
                ball.y + 0.7,
                0.0,
            ),
            Pose2D(
                ball.x + 1.2,
                ball.y - 0.7,
                0.0,
            ),
        ]

        def score(target: Pose2D) -> float:
            return self.kit.targeting.lane_clear_score(
                ball.x,
                ball.y,
                target.x,
                target.y,
                opponents,
            )

        return max(candidates, key=score)

    def _select_clear_target(
        self,
        context: PlayContext,
    ) -> Pose2D:
        """选择最安全的解围目标点。
        
        从多个候选解围方向中评分，选择对手最少、最向前、最远离边线的方向。
        
        评分公式：
            score = 0.60 * lane_score
                  + 0.20 * forward_score
                  + 0.15 * space_score
                  + 0.05 * sideline_score
        
        Args:
            context: 比赛上下文
            
        Returns:
            最佳解围目标点 Pose2D
        """
        ball = context.known_ball
        opponents = self.kit.obstacles.opponent_obstacles(context)
        
        # 候选解围角度（相对于正 x 轴，即向对方球门方向）
        # 覆盖从 -55° 到 +55° 的范围
        candidate_angles_deg = [-55, -40, -25, -10, 0, 10, 25, 40, 55]
        
        # 解围距离：确保球踢出足够远
        clear_distance = 4.0
        
        best_score = -1.0
        best_target = Pose2D(
            min(1.5, ball.x + clear_distance), -0.35 * ball.y
        )
        
        for angle_deg in candidate_angles_deg:
            angle_rad = math.radians(angle_deg)
            
            # 候选目标点
            target_x = ball.x + clear_distance * math.cos(angle_rad)
            target_y = ball.y + clear_distance * math.sin(angle_rad)
            
            # 1. lane_score: 踢球线路是否有对手（最重要）
            lane_score = self.kit.targeting.lane_clear_score(
                ball.x, ball.y, target_x, target_y, opponents
            )
            
            # 2. forward_score: 前进距离评分
            # 越向前分数越高，归一化到 [0, 1]
            forward_score = clamp(
                (target_x - ball.x) / clear_distance, 0.0, 1.0
            )
            
            # 3. space_score: 落点附近是否有对手
            # 计算落点附近最近对手的距离，距离越远分数越高
            min_opp_dist = float("inf")
            for opp in opponents:
                dist = math.hypot(opp.x - target_x, opp.y - target_y)
                if dist < min_opp_dist:
                    min_opp_dist = dist
            # 归一化：2m 以上为安全
            space_score = clamp(min_opp_dist / 2.0, 0.0, 1.0)
            
            # 4. sideline_score: 是否过于靠近边线
            half_width = self.kit.config.field_width / 2
            sideline_margin = 0.5  # 期望的最小边线距离
            y_abs = abs(target_y)
            if y_abs <= half_width - sideline_margin:
                sideline_score = 1.0
            elif y_abs >= half_width:
                sideline_score = 0.0
            else:
                sideline_score = clamp(
                    (half_width - y_abs) / sideline_margin, 0.0, 1.0
                )
            
            # 综合评分
            score = (
                0.60 * lane_score
                + 0.20 * forward_score
                + 0.15 * space_score
                + 0.05 * sideline_score
            )
            
            if score > best_score:
                best_score = score
                best_target = Pose2D(
                    clamp(target_x, -self.kit.config.field_length / 2 + 0.3,
                          self.kit.config.field_length / 2 - 0.3),
                    clamp(target_y, -half_width + 0.3, half_width - 0.3),
                    angle_rad,
                )
        
        return self.kit.field.clamp_inside_field(best_target)

    def _select_primary_action(
        self,
        secondary: int | None,
        mode: TacticalMode,
        context: PlayContext,
        receive: Pose2D | None,
        pressure_time: float | None,
    ) -> tuple[PrimaryIntent, Pose2D]:
        """选择 Primary 的行动意图和目标。
        
        决策优先级（从高到低）：
        1. 紧急防守 -> CLEAR（解围）
        2. 防守 -> CLEAR 或 PRESS
        3. 争夺 -> INTERCEPT
        4. 球在边线 -> PROGRESSIVE_TOUCH（恢复位置）
        5. 射门条件满足 -> SHOOT
        6. 传球条件满足 -> PASS
        7. 压力大 -> PROGRESSIVE_TOUCH（避压）
        8. 默认 -> DRIBBLE（带球）
        
        Args:
            secondary: Secondary 球员 ID
            mode: 战术模式
            context: 比赛上下文
            receive: 接球点
            pressure_time: 压力时间
            
        Returns:
            (意图, 目标点) 元组
        """
        ball = context.known_ball
        goal = Pose2D(self.kit.field.opponent_goal_x(), 0.0)  # 球门中心

        # 我方中圈开球第一脚：斜向安全推进，不等待 Secondary
        if mode == TacticalMode.RESTART:
            game = context.known_game

            if (
                game.set_play == SetPlay.KICKOFF
                and game.is_kickoff_for_team(self.kit.config.team_id)
            ):
                return (
                    PrimaryIntent.PROGRESSIVE_TOUCH,
                    self._select_kickoff_target(context),
                )

        # 紧急防守：大脚解围（使用安全解围目标选择）
        if mode == TacticalMode.EMERGENCY_DEFEND:
            return PrimaryIntent.CLEAR, self._select_clear_target(context)
            
        # 防守：根据球位置决定解围或压迫
        if mode == TacticalMode.DEFEND:
            if ball.x < -self.kit.config.field_length * 0.18:
                # 球在我方半场，解围（使用安全解围目标选择）
                return PrimaryIntent.CLEAR, self._select_clear_target(context)
            return PrimaryIntent.PRESS, goal
            
        # 争夺：拦截球
        if mode == TacticalMode.CONTEST:
            return PrimaryIntent.INTERCEPT, goal
            
        # 球在边线附近：恢复位置
        if self.kit.targeting.ball_near_sideline(ball):
            return (
                PrimaryIntent.PROGRESSIVE_TOUCH,
                self.kit.targeting.sideline_recovery_target(ball),
            )
            
        # 射门条件：x 坐标足够 且 射门路线清晰
        if (
            ball.x >= self.kit.config.strategy.shot_min_x_m
            and self.kit.targeting.shot_lane_is_clear(context)
        ):
            return PrimaryIntent.SHOOT, goal
            
        # 传球条件：Secondary 就位 且 传球路线清晰 且 向前距离足够
        if secondary is not None and receive is not None:
            lane = self.kit.targeting.lane_clear_score(
                ball.x,
                ball.y,
                receive.x,
                receive.y,
                self.kit.obstacles.opponent_obstacles(context),
            )
            if (
                lane >= 0.55  # 路线清晰度阈值
                and receive.x - ball.x
                >= self.kit.config.strategy.pass_min_forward_m  # 最小向前距离
            ):
                return PrimaryIntent.PASS, receive
                
        # 压力大：轻触推进避压
        if pressure_time is not None and pressure_time <= 0.8:
            return (
                PrimaryIntent.PROGRESSIVE_TOUCH,
                self.kit.targeting.dribble_target(ball),
            )
            
        # 默认：带球推进
        return PrimaryIntent.DRIBBLE, self.kit.targeting.dribble_target(ball)

    def _attack_is_stalled(
        self,
        primary: int | None,
        intent: PrimaryIntent,
        action: Pose2D,
        context: PlayContext,
        prediction: BallMotionPrediction | None,
        now_sec: float,
    ) -> bool:
        """检测进攻是否停滞。
        
        使用 AttackWatchdog 监控 Primary 的进攻进展：
        - 如果 Primary 长时间在球附近但没有有效进展，认为停滞
        - 停滞时改为 PROGRESSIVE_TOUCH 尝试打破僵局
        
        Args:
            primary: Primary 球员 ID
            intent: 当前意图
            action: 动作目标
            context: 比赛上下文
            prediction: 球预测
            now_sec: 当前时间
            
        Returns:
            是否应该逃离（进攻停滞）
        """
        if primary is None:
            return False
            
        robot = context.teammates.get(primary)
        if robot is None or robot.pose is None:
            self._watchdog.reset(primary)
            return False
            
        ball = context.known_ball
        distance = math.hypot(robot.pose.x - ball.x, robot.pose.y - ball.y)
        
        # 判断是否在执行有效动作
        actionable = intent in {
            PrimaryIntent.SHOOT,
            PrimaryIntent.PASS,
            PrimaryIntent.DRIBBLE,
            PrimaryIntent.PROGRESSIVE_TOUCH,
            PrimaryIntent.CLEAR,
        }
        active = (
            actionable
            and distance <= self.kit.config.strategy.possession_control_radius_m
        )
        
        # 计算角度误差
        desired = math.atan2(action.y - ball.y, action.x - ball.x)
        angle_error = abs(normalize_angle(desired - robot.pose.theta))
        
        # 确定进攻阶段
        phase = (
            AttackPhase.EXECUTE
            if active and angle_error <= 0.15  # 角度误差小，执行阶段
            else AttackPhase.ALIGN  # 否则对齐阶段
        )
        
        # 更新监控器
        status = self._watchdog.update(
            primary,
            AttackAttemptObservation(
                t_sec=now_sec,
                phase=phase,
                ball_distance_m=distance,
                angle_error_rad=angle_error,
                ball_speed_mps=(
                    prediction.speed_mps if prediction is not None else 0.0
                ),
                active=active,
            ),
        )
        return status.should_escape and intent != PrimaryIntent.CLEAR

    def _primary_claim_target(
        self,
        intent: PrimaryIntent,
        action: Pose2D,
        context: PlayContext,
        prediction: BallMotionPrediction | None,
    ) -> Pose2D:
        """计算 Primary 的移动目标点。
        
        对于拦截类意图（INTERCEPT/CHALLENGE/PRESS），目标是预测的球位置；
        对于其他意图，目标是球当前位置。
        
        目标点会添加接近距离（0.35m），避免直接撞到球。
        
        Args:
            intent: 行动意图
            action: 动作目标
            context: 比赛上下文
            prediction: 球预测
            
        Returns:
            Primary 应该移动到的目标点
        """
        ball = context.known_ball
        # 拦截类意图：目标是预测的球位置
        claim = (
            self._intercept_point(ball, prediction)
            if intent
            in {
                PrimaryIntent.INTERCEPT,
                PrimaryIntent.CHALLENGE,
                PrimaryIntent.PRESS,
            }
            else Pose2D(ball.x, ball.y)  # 其他意图：目标是球当前位置
        )
        # 计算朝向：面向动作目标
        theta = math.atan2(action.y - claim.y, action.x - claim.x)
        # 使用 MotionController 计算接近目标点（带接近距离）
        return self.kit.motion.approach_target(
            BallState(claim.x, claim.y), theta, 0.35
        )

    def _secondary_target(
        self,
        secondary: int | None,
        mode: TacticalMode,
        intent: PrimaryIntent,
        action: Pose2D,
        receive: Pose2D | None,
        context: PlayContext,
        prediction: BallMotionPrediction | None,
    ) -> Pose2D:
        """计算 Secondary 的移动目标点。
        
        目标选择策略：
        - 如果 Primary 要传球，Secondary 移动到接球点
        - 如果 Primary 要射门，Secondary 移动到补位点
        - 争夺模式：Secondary 移动到球预测停止点
        - 防守模式：Secondary 移动到防守位置
        - 其他：移动到支援点或接球点
        
        Args:
            secondary: Secondary 球员 ID
            mode: 战术模式
            intent: Primary 的行动意图
            action: 动作目标
            receive: 接球点
            context: 比赛上下文
            prediction: 球预测
            
        Returns:
            Secondary 应该移动到的目标点
        """
        ball = context.known_ball

        # 开球第一脚完成前，Secondary 留在己方半场，不参与提前接应
        if mode == TacticalMode.RESTART:
            robot = context.teammates.get(secondary)

            if robot is not None and robot.pose is not None:
                x = self.kit.field.own_half_x(
                    robot.pose.x,
                    margin=0.15,
                )
                y = robot.pose.y

                return Pose2D(
                    x,
                    y,
                    self.kit.field.face_ball_theta(x, y, ball),
                )

            # 无有效 pose 时的安全兜底
            return Pose2D(
                -0.5,
                0.0,
                self.kit.field.face_ball_theta(-0.5, 0.0, ball),
            )

        # 没有 Secondary：返回球后方位置
        if secondary is None:
            return Pose2D(ball.x - 1.2, ball.y)
            
        # Primary 要传球：Secondary 移动到接球点
        if intent == PrimaryIntent.PASS and receive is not None:
            return receive
            
        # Primary 要射门：Secondary 移动到补位点
        if intent == PrimaryIntent.SHOOT:
            sign = (
                ball.y
                if abs(ball.y) > 0.05
                else (1.0 if secondary % 2 else -1.0)
            )
            y = -math.copysign(min(1.4, abs(ball.y) + 0.6), sign)
            return self.kit.field.clamp_inside_field(
                Pose2D(min(action.x - 1.0, ball.x + 1.6), y, 0.0)
            )
            
        # 争夺模式：移动到球预测停止点
        if mode == TacticalMode.CONTEST:
            point = (
                prediction.stop
                if prediction is not None and prediction.confidence >= 0.35
                else Pose2D(ball.x + 0.4, -ball.y * 0.35)
            )
            return self.kit.field.clamp_inside_field(
                Pose2D(
                    point.x,
                    point.y,
                    self.kit.field.face_ball_theta(point.x, point.y, ball),
                )
            )
            
        # 防守模式：移动到防守位置
        if mode in {TacticalMode.DEFEND, TacticalMode.EMERGENCY_DEFEND}:
            x = max(self.kit.field.own_goal_x() + 1.4, ball.x - 1.0)
            y = clamp(ball.y * 0.55, -1.7, 1.7)
            return Pose2D(
                x, y, self.kit.field.face_ball_theta(x, y, ball)
            )
            
        # 默认：移动到支援点或接球点
        return receive or self.kit.targeting.support_target(
            secondary, context, self.kit.is_player_allowed
        )

    def _counterattack_risk(
        self,
        mode: TacticalMode,
        ball: BallState,
        pressure_time: float | None,
    ) -> float:
        """计算反击风险评分 (0.0-1.0)。
        
        风险因素：
        - 战术模式：不同模式有基础风险值
        - 压力时间：压力越大（时间越小），风险越高
        - 球位置：球在我方半场，风险更高
        
        Args:
            mode: 战术模式
            ball: 球状态
            pressure_time: 压力时间
            
        Returns:
            风险评分，0.0 最低，1.0 最高
        """
        # 基础风险值（按模式）
        risk = {
            TacticalMode.ATTACK: 0.35,
            TacticalMode.CONTEST: 0.60,
            TacticalMode.DEFEND: 0.80,
            TacticalMode.EMERGENCY_DEFEND: 1.0,
            TacticalMode.RESTART: 0.45,
        }[mode]
        
        # 压力时间修正：压力越大风险越高
        if pressure_time is not None:
            risk += clamp((1.2 - pressure_time) / 2.0, 0.0, 0.35)
            
        # 球位置修正：球在我方半场风险更高
        if ball.x < 0.0:
            risk += 0.10
            
        return clamp(risk, 0.0, 1.0)

    def _safety_target(
        self,
        mode: TacticalMode,
        risk: float,
        context: PlayContext,
    ) -> Pose2D:
        """计算 Safety 的移动目标点。
        
        Safety 位置设计：
        - 紧急防守：守在球门前
        - 其他模式：根据风险动态调整位置
          - x 坐标：球位置 - 防守深度，受风险影响
          - y 坐标：跟随球的 y 坐标，但幅度减小
        
        Args:
            mode: 战术模式
            risk: 反击风险评分
            context: 比赛上下文
            
        Returns:
            Safety 应该移动到的目标点
        """
        ball = context.known_ball
        tuning = self.kit.config.strategy
        own_goal = self.kit.field.own_goal_x()
        
        if mode == TacticalMode.EMERGENCY_DEFEND:
            # 紧急防守：守在球门中心附近
            x = own_goal + 0.35
            y = clamp(
                ball.y * 0.70,
                -self.kit.config.goal_width / 2 + 0.2,
                self.kit.config.goal_width / 2 - 0.2,
            )
        else:
            # 动态防守位置
            desired_x = ball.x - tuning.rest_defense_depth_m
            deepest_x = own_goal + tuning.safety_goal_offset_m
            x = max(deepest_x, min(-0.25, desired_x - risk * 0.8))
            y = clamp(
                ball.y * 0.35,
                -tuning.safety_lateral_limit_m,
                tuning.safety_lateral_limit_m,
            )
            
        return self.kit.field.clamp_inside_field(
            Pose2D(x, y, self.kit.field.face_ball_theta(x, y, ball)),
            margin=0.3,
        )
