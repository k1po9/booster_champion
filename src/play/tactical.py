"""Central tactical orchestration for the dynamic 3v3 triangle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from ..soccer_framework import BallState, Pose2D, ReadySlot, SetPlay, PlayContext
from ..tactics import (
    AttackAttemptObservation,
    AttackPhase,
    AttackWatchdog,
    BallMotionPrediction,
    BallObservation,
    OpponentPressureEstimator,
    SlidingWindowBallPredictor,
)
from ..tactics.geometry import clamp, normalize_angle

if TYPE_CHECKING:
    from ..runtime import SoccerKit


class TacticalMode(str, Enum):
    ATTACK = "attack"
    CONTEST = "contest"
    DEFEND = "defend"
    EMERGENCY_DEFEND = "emergency_defend"
    RESTART = "restart"


class PrimaryIntent(str, Enum):
    APPROACH = "approach"
    CHALLENGE = "challenge"
    INTERCEPT = "intercept"
    SHOOT = "shoot"
    PASS = "pass"
    DRIBBLE = "dribble"
    PROGRESSIVE_TOUCH = "progressive_touch"
    PRESS = "press"
    CLEAR = "clear"


ROLE_PRIMARY = "primary"
ROLE_SECONDARY = "secondary"
ROLE_SAFETY = "safety"


@dataclass(frozen=True)
class TacticalContext:
    mode: TacticalMode
    primary_id: int | None
    secondary_id: int | None
    safety_id: int | None
    primary_intent: PrimaryIntent
    primary_target: Pose2D
    action_target: Pose2D
    secondary_target: Pose2D
    safety_target: Pose2D
    receive_target: Pose2D | None = None
    secondary_ready: bool = False
    counterattack_risk: float = 0.0
    ball_prediction: BallMotionPrediction | None = None
    pressure_time_sec: float | None = None

    def role_of(self, player_id: int) -> str:
        if player_id == self.primary_id:
            return ROLE_PRIMARY
        if player_id == self.secondary_id:
            return ROLE_SECONDARY
        if player_id == self.safety_id:
            return ROLE_SAFETY
        return "none"


@dataclass
class _SwitchState:
    current: int | None = None
    challenger: int | None = None
    ticks: int = 0


class DynamicTriangleCoordinator:
    """Build one immutable team plan per tick and stabilize Primary handoffs."""

    def __init__(self, kit: "SoccerKit"):
        self.kit = kit
        self._ball = SlidingWindowBallPredictor(window_sec=0.45)
        self._pressure = OpponentPressureEstimator()
        self._watchdog = AttackWatchdog()
        self._primary_switch = _SwitchState()
        self.last_context: TacticalContext | None = None

    def update(self, context: PlayContext, now_sec: float) -> TacticalContext:
        ball = context.known_ball
        stamp = ball.last_seen_at if ball.last_seen_at > 0.0 else now_sec
        self._ball.add_observation(
            BallObservation(stamp, ball.x, ball.y, ball.confidence)
        )
        prediction = self._ball.predict()
        mode = self._select_mode(context)
        active = self._active_players(context)
        primary = self._select_primary(active, context, mode, prediction)
        secondary, safety = self._assign_support_slots(active, primary, context)

        pressure = self._pressure.estimate(
            now_sec=now_sec,
            opponents=context.opponents,
            ball_prediction=prediction,
            target=None if prediction is not None else Pose2D(ball.x, ball.y),
            game_state=context.game_state,
            opponent_team_id=self.kit.config.opponent_team_id(),
        )
        pressure_time = pressure.pressure_time_sec
        receive = self._receive_target(secondary, context, mode)
        ready = self._is_ready(secondary, receive, context)
        intent, action_target = self._select_primary_action(
            secondary, mode, context, receive, pressure_time
        )
        if self._attack_is_stalled(
            primary,
            intent,
            action_target,
            context,
            prediction,
            now_sec,
        ):
            intent = PrimaryIntent.PROGRESSIVE_TOUCH
            action_target = self.kit.targeting.dribble_target(ball)
        primary_target = self._primary_claim_target(
            intent, action_target, context, prediction
        )
        secondary_target = self._secondary_target(
            secondary, mode, intent, action_target, receive, context, prediction
        )
        risk = self._counterattack_risk(mode, ball, pressure_time)
        safety_target = self._safety_target(mode, risk, context)

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
        game = context.known_game
        visible = [
            player_id
            for player_id in self.kit.config.player_ids
            if self.kit.is_player_allowed(game, player_id)
            and context.teammates.get(player_id) is not None
            and context.teammates[player_id].pose is not None
        ]
        if visible:
            return visible
        return [
            player_id
            for player_id in self.kit.config.player_ids
            if self.kit.is_player_allowed(game, player_id)
        ]

    def _select_mode(self, context: PlayContext) -> TacticalMode:
        game = context.known_game
        ball = context.known_ball
        if game.set_play != SetPlay.NONE:
            return TacticalMode.RESTART
        tuning = self.kit.config.strategy
        if ball.x <= tuning.emergency_defend_x_m:
            return TacticalMode.EMERGENCY_DEFEND
        our_distance = self._nearest_distance(context.teammates, ball)
        their_distance = self._nearest_distance(context.opponents, ball)
        if their_distance + tuning.possession_advantage_m < our_distance:
            return TacticalMode.DEFEND
        if (
            our_distance <= tuning.possession_control_radius_m
            and our_distance + tuning.possession_advantage_m < their_distance
        ):
            return TacticalMode.ATTACK
        if ball.x <= tuning.defend_x_m:
            return TacticalMode.DEFEND
        return TacticalMode.CONTEST

    @staticmethod
    def _nearest_distance(robots: dict, ball: BallState) -> float:
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
        if not active:
            self._primary_switch = _SwitchState()
            return None
        point = self._intercept_point(context.known_ball, prediction)
        scored = sorted(
            (self._claim_cost(player_id, point, context, mode), player_id)
            for player_id in active
        )
        best_cost, best = scored[0]
        state = self._primary_switch
        if state.current not in active:
            self._primary_switch = _SwitchState(current=best)
            return best
        current_cost = next(cost for cost, pid in scored if pid == state.current)
        tuning = self.kit.config.strategy
        if (
            best == state.current
            or best_cost + tuning.role_switch_advantage_sec >= current_cost
        ):
            state.challenger = None
            state.ticks = 0
            return state.current
        if state.challenger == best:
            state.ticks += 1
        else:
            state.challenger = best
            state.ticks = 1
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
        robot = context.teammates.get(player_id)
        if robot is None or robot.pose is None:
            return math.inf
        pose = robot.pose
        tuning = self.kit.config.strategy
        speed = max(0.1, tuning.max_linear_speed * tuning.robot_translation_gain)
        yaw_speed = max(0.1, tuning.max_angular_speed * tuning.robot_yaw_gain)
        bearing = math.atan2(point.y - pose.y, point.x - pose.x)
        cost = math.hypot(point.x - pose.x, point.y - pose.y) / speed
        cost += (
            abs(normalize_angle(bearing - pose.theta)) / yaw_speed * 0.35
        )
        is_keeper = (
            self.kit.config.ready_slot_for_player(player_id) == ReadySlot.KEEPER
        )
        if is_keeper and mode not in {
            TacticalMode.EMERGENCY_DEFEND,
            TacticalMode.DEFEND,
        }:
            cost += 2.0
        return cost

    def _assign_support_slots(
        self,
        active: list[int],
        primary: int | None,
        context: PlayContext,
    ) -> tuple[int | None, int | None]:
        remaining = [player_id for player_id in active if player_id != primary]
        if not remaining:
            # One-player degradation merges Primary and Safety.
            return None, primary
        if len(remaining) == 1:
            # Two-player degradation is Primary + Safety.
            return None, remaining[0]
        keeper = self.kit.config.goalkeeper_player_id()
        safety = keeper if keeper in remaining else min(
            remaining,
            key=lambda player_id: (
                context.teammates[player_id].pose.x
                if context.teammates.get(player_id)
                and context.teammates[player_id].pose
                else math.inf
            ),
        )
        secondary = next(player_id for player_id in remaining if player_id != safety)
        return secondary, safety

    def _intercept_point(
        self,
        ball: BallState,
        prediction: BallMotionPrediction | None,
    ) -> Pose2D:
        if (
            prediction is None
            or prediction.confidence < 0.35
            or prediction.motion_state != "rolling"
        ):
            return Pose2D(ball.x, ball.y)
        point = prediction.position_at(
            self.kit.config.strategy.intercept_prediction_horizon_sec
        )
        return self.kit.field.clamp_inside_field(point)

    def _receive_target(
        self,
        secondary: int | None,
        context: PlayContext,
        mode: TacticalMode,
    ) -> Pose2D | None:
        if secondary is None or mode not in {
            TacticalMode.ATTACK,
            TacticalMode.RESTART,
        }:
            return None
        ball = context.known_ball
        robot = context.teammates.get(secondary)
        side = 1.0 if secondary % 2 == 0 else -1.0
        base_y = (
            robot.pose.y
            if robot is not None and robot.pose is not None
            else ball.y + side
        )
        x = min(self.kit.field.opponent_goal_x() - 0.8, ball.x + 1.4)
        y = clamp(
            base_y + side * 0.35,
            -self.kit.config.field_width / 2 + 0.55,
            self.kit.config.field_width / 2 - 0.55,
        )
        return self.kit.field.clamp_inside_field(
            Pose2D(x, y, math.atan2(ball.y - y, ball.x - x))
        )

    def _is_ready(
        self,
        secondary: int | None,
        target: Pose2D | None,
        context: PlayContext,
    ) -> bool:
        if secondary is None or target is None:
            return False
        robot = context.teammates.get(secondary)
        return (
            robot is not None
            and robot.pose is not None
            and math.hypot(robot.pose.x - target.x, robot.pose.y - target.y)
            <= self.kit.config.strategy.secondary_receive_radius_m
        )

    def _select_primary_action(
        self,
        secondary: int | None,
        mode: TacticalMode,
        context: PlayContext,
        receive: Pose2D | None,
        pressure_time: float | None,
    ) -> tuple[PrimaryIntent, Pose2D]:
        ball = context.known_ball
        goal = Pose2D(self.kit.field.opponent_goal_x(), 0.0)
        if mode == TacticalMode.EMERGENCY_DEFEND:
            return PrimaryIntent.CLEAR, Pose2D(
                min(1.5, ball.x + 4.0), -0.35 * ball.y
            )
        if mode == TacticalMode.DEFEND:
            if ball.x < -self.kit.config.field_length * 0.18:
                return PrimaryIntent.CLEAR, Pose2D(1.0, -0.25 * ball.y)
            return PrimaryIntent.PRESS, goal
        if mode == TacticalMode.CONTEST:
            return PrimaryIntent.INTERCEPT, goal
        if self.kit.targeting.ball_near_sideline(ball):
            return (
                PrimaryIntent.PROGRESSIVE_TOUCH,
                self.kit.targeting.sideline_recovery_target(ball),
            )
        if (
            ball.x >= self.kit.config.strategy.shot_min_x_m
            and self.kit.targeting.shot_lane_is_clear(context)
        ):
            return PrimaryIntent.SHOOT, goal
        if secondary is not None and receive is not None:
            lane = self.kit.targeting.lane_clear_score(
                ball.x,
                ball.y,
                receive.x,
                receive.y,
                self.kit.obstacles.opponent_obstacles(context),
            )
            if (
                lane >= 0.55
                and receive.x - ball.x
                >= self.kit.config.strategy.pass_min_forward_m
            ):
                return PrimaryIntent.PASS, receive
        if pressure_time is not None and pressure_time <= 0.8:
            return (
                PrimaryIntent.PROGRESSIVE_TOUCH,
                self.kit.targeting.dribble_target(ball),
            )
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
        """Use the old progress watchdog only for a genuine near-ball attempt."""

        if primary is None:
            return False
        robot = context.teammates.get(primary)
        if robot is None or robot.pose is None:
            self._watchdog.reset(primary)
            return False
        ball = context.known_ball
        distance = math.hypot(robot.pose.x - ball.x, robot.pose.y - ball.y)
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
        desired = math.atan2(action.y - ball.y, action.x - ball.x)
        angle_error = abs(normalize_angle(desired - robot.pose.theta))
        phase = (
            AttackPhase.EXECUTE
            if active and angle_error <= 0.15
            else AttackPhase.ALIGN
        )
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
        ball = context.known_ball
        claim = (
            self._intercept_point(ball, prediction)
            if intent
            in {
                PrimaryIntent.INTERCEPT,
                PrimaryIntent.CHALLENGE,
                PrimaryIntent.PRESS,
            }
            else Pose2D(ball.x, ball.y)
        )
        theta = math.atan2(action.y - claim.y, action.x - claim.x)
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
        ball = context.known_ball
        if secondary is None:
            return Pose2D(ball.x - 1.2, ball.y)
        if intent == PrimaryIntent.PASS and receive is not None:
            return receive
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
        if mode in {TacticalMode.DEFEND, TacticalMode.EMERGENCY_DEFEND}:
            x = max(self.kit.field.own_goal_x() + 1.4, ball.x - 1.0)
            y = clamp(ball.y * 0.55, -1.7, 1.7)
            return Pose2D(
                x, y, self.kit.field.face_ball_theta(x, y, ball)
            )
        return receive or self.kit.targeting.support_target(
            secondary, context, self.kit.is_player_allowed
        )

    def _counterattack_risk(
        self,
        mode: TacticalMode,
        ball: BallState,
        pressure_time: float | None,
    ) -> float:
        risk = {
            TacticalMode.ATTACK: 0.35,
            TacticalMode.CONTEST: 0.60,
            TacticalMode.DEFEND: 0.80,
            TacticalMode.EMERGENCY_DEFEND: 1.0,
            TacticalMode.RESTART: 0.45,
        }[mode]
        if pressure_time is not None:
            risk += clamp((1.2 - pressure_time) / 2.0, 0.0, 0.35)
        if ball.x < 0.0:
            risk += 0.10
        return clamp(risk, 0.0, 1.0)

    def _safety_target(
        self,
        mode: TacticalMode,
        risk: float,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball
        tuning = self.kit.config.strategy
        own_goal = self.kit.field.own_goal_x()
        if mode == TacticalMode.EMERGENCY_DEFEND:
            x = own_goal + 0.35
            y = clamp(
                ball.y * 0.70,
                -self.kit.config.goal_width / 2 + 0.2,
                self.kit.config.goal_width / 2 - 0.2,
            )
        else:
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
