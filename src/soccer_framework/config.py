"""团队与策略配置，从默认值或环境变量加载。

配置设计文档 (Configuration Design Document)

【整体架构】
本文件定义了机器人足球系统的所有可调参数，分为三个层次：

  1. SoccerConfig: 团队级配置（队伍ID、机器人名称、场地尺寸等）
  2. SoccerStrategyTuning: 策略调优参数（速度、踢球、避障、传球、带球等）
  3. SoccerDebugConfig: 调试开关（行为树追踪等）

【配置加载方式】
  - 环境变量: 仅加载每场比赛必须更改的身份字段（SOCCER_TEAM_ID, SOCCER_ROBOT_NAMES）
  - 数据类默认值: 策略调优参数通过默认值设置，调试时可覆盖
  - ROS 参数: 部分调试参数可通过 ROS 参数服务器覆盖

【SoccerStrategyTuning 参数分组】
  - 速度限制: max_linear_speed, max_angular_speed
  - 踢球滞后: soccer_kick_enter_distance, soccer_kick_exit_distance 等
  - 定位球: restart_touch_distance, opponent_restart_avoid_distance_m
  - 路径绕行: opponent_obstacle_radius, teammate_obstacle_radius 等
  - 偏航避障: yaw_avoid_horizon_sec, yaw_avoid_min_distance_m, yaw_avoid_bias_max
  - 球权仲裁: teammate_challenge_tie_margin_m
  - 动态三角: role_switch_advantage_sec, role_switch_confirm_ticks 等
  - 传球: pass_enabled, pass_min_score, pass_min_forward_m, pass_lane_clearance
  - 带球: dribble_advance_m, dribble_center_pull
  - 支援: support_depth_m, support_lateral_m, support_min_spacing_m
  - 守门: goalkeeper_challenge_margin_m
  - 边线恢复: sideline_recovery_margin_m, sideline_recovery_infield_m 等

【涉及的文件】
  - src/runtime.py: SoccerKit 使用 SoccerConfig 组装所有组件
  - src/play/tactical.py: DynamicTriangleCoordinator 读取 strategy 参数
  - src/tactics/targeting.py: Targeting 使用传球/带球/支援参数
  - src/tactics/motion.py: MotionController 使用速度限制和避障参数
  - src/soccer_framework/types.py: FieldDimensions, ReadySlot 等基础类型
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .types import (
    ADULT_FIELD_DIMENSIONS,
    FieldDimensions,
    ReadySlot,
)


__all__ = [
    "SoccerConfig",
    "SoccerDebugConfig",
    "SoccerStrategyTuning",
]


DEFAULT_READY_SLOT_SEQUENCE = (
    ReadySlot.CENTER,
    ReadySlot.SIDE,
    ReadySlot.KEEPER,
)


@dataclass
class SoccerDebugConfig:
    """调试专用开关，用于可选的诊断功能。
    
    这些控制项故意不暴露为环境变量，以保持正常比赛启动时的配置简洁。
    本地调试时可通过修改默认值或传递自定义实例给 SoccerConfig 来启用。
    
    Attributes:
        bt_trace_ticks: 行为树追踪模式，"off" 表示关闭，"on" 表示开启
        bt_trace_sample_sec: 行为树追踪采样间隔（秒），默认 0.5 秒
    """

    bt_trace_ticks: str = "off"  # 行为树追踪开关："off"=关闭, "on"=开启
    bt_trace_sample_sec: float = 0.5  # 行为树追踪采样间隔（秒）


@dataclass
class SoccerStrategyTuning:
    """策略行为调优参数集合。
    
    这些是"如何踢球"的参数，按功能分组，默认值与硬件能力或比赛规则绑定。
    
    【参数分组说明】
    1. 速度限制: 运动层的硬性输出限制，与底盘稳定性和场地摩擦力相关
    2. 踢球滞后: 使用进入/退出阈值+延迟，防止在距离边界附近频繁切换
    3. 定位球: 重新开始比赛时的距离阈值
    4. 路径绕行: 第一层避障，通过计算绕行点绕过障碍物
    5. 偏航避障: 第二层避障，通过微调偏航角速度避开近距离邻居
    6. 球权仲裁: 队友间球权争夺的平局判定边界
    7. 动态三角: 角色切换和模式选择的滞后参数
    8. 传球/带球/支援: 进攻配合的阈值和几何参数
    9. 守门/边线恢复: 特殊场景的恢复策略
    
    这些参数不作为单独的环境变量暴露；调试时修改默认值或传递自定义配置。
    """

    # ==========================================================================
    # 速度限制 (Speed Limits)
    # 运动层的硬性输出限制，与底盘稳定性和场地摩擦力相关
    # ==========================================================================
    max_linear_speed: float = 0.8  # 线速度上限 (m/s)
    max_angular_speed: float = 1.0  # 角速度上限 (rad/s)

    # ==========================================================================
    # 踢球滞后 (Kick Hysteresis)
    # 使用进入/退出阈值+延迟，防止在距离边界附近频繁切换（防抖动）
    # ==========================================================================
    soccer_kick_enter_distance: float = 2.5  # 进入踢球模式的距离阈值 (m)，低于此值进入
    soccer_kick_exit_distance: float = 3.0  # 退出踢球模式的距离阈值 (m)，必须大于 enter
    soccer_kick_power: float = 1.5  # 踢球力度
    soccer_kick_min_active_sec: float = 1.0  # 最小踢球持续时间 (s)，避免瞬间切换
    soccer_kick_exit_delay_sec: float = 1.5  # 退出条件满足后延迟时间 (s)，才真正离开踢球模式

    # ==========================================================================
    # 定位球与重新开始 (Set Plays and Restarts)
    # ==========================================================================
    restart_touch_distance: float = 0.45  # "触球"距离阈值 (m)
    opponent_restart_avoid_distance_m: float = (
        1.6  # 规则要求 1.45m，默认加 0.15m 缓冲，设为 1.60m
    )

    # ==========================================================================
    # 路径绕行 (Path Detour via Points)
    # 第一层避障：从当前位置到目标画一条线，如果障碍物、对方、队友或球门
    # 在这条走廊内，计算绕行点并重写目标，让机器人绕过去。
    # 触发条件基于走廊阻塞，而非纯距离：远处的阻挡者会触发，近处非阻挡者不会。
    # 球门尺寸由规则固定在 navigation.goal_structure_obstacles，不在此调优。
    # 对方半径大于队友半径，因为对方有竞争性且不可预测。
    # ==========================================================================
    opponent_obstacle_radius: float = 0.55  # 对方绕行圆形半径 (m)
    teammate_obstacle_radius: float = (
        0.48  # 队友绕行圆形半径 (m)，较小因为队友行为可预测
    )
    obstacle_safety_margin: float = 0.22  # 障碍物半径外的额外安全边距 (m)，所有障碍物共用
    obstacle_start_ignore_distance: float = 0.35  # 忽略距离起点这么近的障碍物，避免近距离抖动
    obstacle_target_ignore_distance: float = 0.35  # 忽略距离目标这么近的障碍物，避免到达时被阻挡

    # ==========================================================================
    # 偏航避障 (Yaw Avoidance Bias)
    # 第二层避障：保持目标不变，检查附近机器人（当前距离或预测最短距离低于阈值），
    # 然后给 vyaw 添加 +/-bias_max，让机器人在通过时稍微转向。
    # 对于双足底盘，稳定的 set_velocity 组合是 vx + vyaw；vy 横向移动来自步态合成，
    # 所以这一层只改变 vyaw，从不改变 vy。
    # 队友始终算作邻居；对方由 BT 阶段通过 move_to 决定是否包含。
    # PLAY 模式排除对方，避免追球者被推开；READY/恢复模式包含对方。
    # ==========================================================================
    yaw_avoid_horizon_sec: float = 1.0  # 附近邻居轨迹的预测时间窗口 (s)
    yaw_avoid_min_distance_m: float = 0.78  # 仅当当前或预测距离低于此值时才施加偏航偏置
    yaw_avoid_bias_max: float = (
        0.6  # 每个邻居的最大 vyaw 偏置 (rad/s)，会被缩放因子减小
    )

    # ==========================================================================
    # 球权仲裁 (Ball-Claim Arbitration)
    # ==========================================================================
    teammate_challenge_tie_margin_m: float = (
        0.15  # 队友球权争夺的平局判定边界 (m)，防止频繁切换
    )

    # ==========================================================================
    # 动态三角角色切换与模式选择 (Dynamic Triangle Role Auction and Mode Selection)
    # ==========================================================================
    role_switch_advantage_sec: float = 0.30  # 角色切换优势时间 (s)，新角色需比当前角色快这么多才触发切换
    role_switch_confirm_ticks: int = 4  # 角色切换确认 tick 数，需连续这么多 tick 确认才真正切换
    robot_translation_gain: float = 0.8945  # 机器人平移速度增益系数
    robot_yaw_gain: float = 0.8789  # 机器人偏航速度增益系数
    possession_control_radius_m: float = 1.10  # 控球控制半径 (m)，在此范围内认为可以控球
    possession_advantage_m: float = 0.30  # 控球优势距离 (m)，我方比对方近这么多才算有优势
    emergency_defend_x_m: float = -4.20  # 紧急防守触发 x 坐标 (m)，球低于此值触发紧急防守
    defend_x_m: float = -0.35  # 防守触发 x 坐标 (m)，球低于此值触发防守模式

    # ==========================================================================
    # 动态三角几何参数 (Dynamic Triangle Geometry)
    # ==========================================================================
    secondary_receive_radius_m: float = 0.65  # Secondary 接球准备半径 (m)，在此范围内认为已准备好接球
    safety_goal_offset_m: float = 1.15  # Safety 距离球门的偏移量 (m)
    safety_lateral_limit_m: float = 1.50  # Safety 横向移动限制 (m)
    rest_defense_depth_m: float = 2.25  # 防守深度 (m)，Safety 在球后方的距离
    intercept_prediction_horizon_sec: float = 0.65  # 拦截预测时间窗口 (s)
    shot_min_x_m: float = 1.20  # 最小射门 x 坐标 (m)，球必须超过此值才考虑射门

    # ==========================================================================
    # 传球 (Passing)
    # ==========================================================================
    pass_enabled: bool = True  # 传球总开关
    pass_min_score: float = 0.52  # 最小传球得分，低于此值改为带球
    pass_min_forward_m: float = 0.35  # 最小向前推进距离 (m)，避免横传/回传
    pass_lane_clearance: float = 0.75  # 传球通道所需净空 (m)，避免被拦截

    # ==========================================================================
    # 带球 (Dribbling)
    # ==========================================================================
    dribble_advance_m: float = 1.15  # 单次带球推进距离 (m)
    dribble_center_pull: float = 0.65  # 带球时向中线拉回的力度，避免贴边线

    # ==========================================================================
    # 支援站位 (Support Positioning)
    # ==========================================================================
    support_depth_m: float = 1.05  # 支援者在持球者后方的深度 (m)
    support_lateral_m: float = 1.25  # 支援者横向间距 (m)
    support_min_spacing_m: float = 1.15  # 最小队友间距 (m)，避免扎堆

    # ==========================================================================
    # 守门与挑战 (Goalkeeping and Challenges)
    # ==========================================================================
    goalkeeper_challenge_margin_m: float = 0.70  # 触发守门员挑战的边界距离 (m)

    # ==========================================================================
    # 边线与底线恢复 (Sideline and Goal-Line Recovery)
    # ==========================================================================
    sideline_recovery_margin_m: float = 0.90  # 边线恢复距离阈值 (m)
    sideline_recovery_infield_m: float = 1.60  # 恢复时向场内拉回的深度 (m)
    sideline_recovery_advance_m: float = 0.75  # 恢复时向前推进距离 (m)
    goal_line_recovery_margin_m: float = 0.08  # 底线恢复边界 (m)，较小以防止越线


@dataclass
class SoccerConfig:
    """团队级完整配置。
    
    字段大致分为两层：
    
    1. "环境身份": team_id 和 robot_names，这些是唯一通过 from_env() 从环境变量加载的字段
    2. "构造函数/ROS 参数配置": opponent_robot_names, control_hz, game_controller_topic,
       场地尺寸、初始 ready_slots, SoccerStrategyTuning, SoccerDebugConfig
       这些通过默认值、构造函数参数或 ROS 调试参数修改，而非公开环境变量
    
    Attributes:
        team_id: 队伍 ID (1 或 2)
        robot_names: 我方机器人名称元组
        opponent_robot_names: 对方机器人名称元组
        ready_slots: 球员初始站位映射 {player_id: ReadySlot}
        control_hz: 控制频率 (Hz)
        game_controller_topic: 游戏控制器 ROS 话题
        field_length: 场地长度 (m)
        field_width: 场地宽度 (m)
        penalty_dist: 点球点距离 (m)
        goal_width: 球门宽度 (m)
        center_circle_radius: 中圈半径 (m)
        penalty_area_length: 罚球区长度 (m)
        penalty_area_width: 罚球区宽度 (m)
        goal_area_length: 球门区长度 (m)
        goal_area_width: 球门区宽度 (m)
        strategy: 策略调优参数
        debug: 调试配置
    """

    team_id: int = 1
    robot_names: tuple[str, ...] = ("robot1", "robot2", "robot3")
    opponent_robot_names: tuple[str, ...] = ()
    ready_slots: dict[int, ReadySlot] = field(
        default_factory=lambda: {
            1: ReadySlot.CENTER,
            2: ReadySlot.SIDE,
            3: ReadySlot.KEEPER,
        }
    )
    control_hz: float = 30.0
    game_controller_topic: str = "/soccer/game_controller"
    field_length: float = ADULT_FIELD_DIMENSIONS.length
    field_width: float = ADULT_FIELD_DIMENSIONS.width
    penalty_dist: float = ADULT_FIELD_DIMENSIONS.penalty_dist
    goal_width: float = ADULT_FIELD_DIMENSIONS.goal_width
    center_circle_radius: float = ADULT_FIELD_DIMENSIONS.circle_radius
    penalty_area_length: float = ADULT_FIELD_DIMENSIONS.penalty_area_length
    penalty_area_width: float = ADULT_FIELD_DIMENSIONS.penalty_area_width
    goal_area_length: float = ADULT_FIELD_DIMENSIONS.goal_area_length
    goal_area_width: float = ADULT_FIELD_DIMENSIONS.goal_area_width
    strategy: SoccerStrategyTuning = field(default_factory=SoccerStrategyTuning)
    debug: SoccerDebugConfig = field(default_factory=SoccerDebugConfig)

    def __post_init__(self) -> None:
        """初始化后处理：设置默认对方名称和补全站位配置。"""
        if not self.opponent_robot_names:
            self.opponent_robot_names = _default_opponent_robot_names(self.team_id)
        self.ready_slots = _complete_ready_slots(
            self.ready_slots,
            player_count=len(self.robot_names),
        )

    @property
    def player_ids(self) -> tuple[int, ...]:
        """返回我方球员 ID 元组 (1, 2, 3, ...)。"""
        return tuple(range(1, len(self.robot_names) + 1))

    def ready_slot_for_player(self, player_id: int) -> ReadySlot:
        """获取指定球员的初始站位槽位。
        
        Args:
            player_id: 球员 ID
            
        Returns:
            球员的初始站位槽位，未配置时返回 SIDE
        """
        return self.ready_slots.get(player_id, ReadySlot.SIDE)

    def goalkeeper_player_id(self) -> int | None:
        """查找守门员球员 ID。
        
        Returns:
            守门员的球员 ID，如果没有球员分配为 KEEPER 则返回 None
        """
        for player_id in self.player_ids:
            if self.ready_slot_for_player(player_id) == ReadySlot.KEEPER:
                return player_id
        return None

    def opponent_team_id(self) -> int:
        """返回对方队伍 ID。
        
        Returns:
            如果我方是 1 则返回 2，否则返回 1
        """
        return 2 if self.team_id == 1 else 1

    def field_dimensions(self) -> FieldDimensions:
        """构建场地尺寸对象。
        
        Returns:
            FieldDimensions 实例，包含所有场地几何参数
        """
        return FieldDimensions(
            length=self.field_length,
            width=self.field_width,
            penalty_dist=self.penalty_dist,
            goal_width=self.goal_width,
            circle_radius=self.center_circle_radius,
            penalty_area_length=self.penalty_area_length,
            penalty_area_width=self.penalty_area_width,
            goal_area_length=self.goal_area_length,
            goal_area_width=self.goal_area_width,
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SoccerConfig":
        """从环境变量加载每场比赛的字段，其余字段保持数据类默认值。
        
        公开环境变量仅限于每场比赛的身份信息：
        - SOCCER_TEAM_ID: 队伍 ID
        - SOCCER_ROBOT_NAMES: 机器人名称（逗号分隔）
        
        速度限制、踢球滞后等策略调优参数在 SoccerStrategyTuning 中定义，
        不再作为环境变量；通过修改默认值或传递自定义配置来调整。
        
        Args:
            environ: 环境变量映射，默认使用 os.environ
            
        Returns:
            SoccerConfig 实例，包含从环境变量加载的身份信息和默认策略参数
        """
        env = os.environ if environ is None else environ
        base = cls()
        team_id = _parse_int(env.get("SOCCER_TEAM_ID"), base.team_id)
        robot_names = _parse_robot_names(
            env.get("SOCCER_ROBOT_NAMES"),
            default=base.robot_names,
        )
        return cls(
            team_id=team_id,
            robot_names=robot_names,
        )


def _parse_robot_names(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    names = tuple(_normalize_robot_name(item) for item in _split_csv(value))
    if not names:
        return default
    return names


def _normalize_robot_name(value: str) -> str:
    normalized = value.strip()
    if normalized.lower() in {"default", "<default>", "none", "null"}:
        return ""
    return normalized


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _default_opponent_robot_names(team_id: int) -> tuple[str, ...]:
    if team_id == 1:
        return ("robot4", "robot5", "robot6")
    return ("robot1", "robot2", "robot3")


def _complete_ready_slots(
    ready_slots: Mapping[int, ReadySlot],
    player_count: int,
) -> dict[int, ReadySlot]:
    completed: dict[int, ReadySlot] = {}
    for player_id in range(1, player_count + 1):
        default_slot = (
            DEFAULT_READY_SLOT_SEQUENCE[player_id - 1]
            if player_id <= len(DEFAULT_READY_SLOT_SEQUENCE)
            else ReadySlot.SIDE
        )
        completed[player_id] = ready_slots.get(player_id, default_slot)
    return completed
