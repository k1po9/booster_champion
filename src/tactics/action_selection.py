"""Bounded, explainable kick-target candidates for ChampionPlaybook.

This module is deliberately pure and small: at most five shots, two passes,
three dribbles and three defensive clears are scored, then only the best of
each action kind enters the final comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..soccer_framework import PlayContext, Pose2D, SoccerConfig
from .geometry import TeamFieldFrame, clamp, normalize_angle
from .navigation import ObstacleCollector
from .targeting.attack import lane_clear_score
from .targeting.predicates import ball_in_own_defensive_area, ball_near_sideline


class ActionKind(str, Enum):
    RECOVERY = "recovery"
    SHOOT = "shoot"
    PASS = "pass"
    DRIBBLE = "dribble"
    CLEAR = "clear"


@dataclass(frozen=True)
class ActionCandidate:
    kind: ActionKind
    target: Pose2D
    utility: float
    lane_score: float
    turn_cost: float
    reason: str
    receiver_id: int | None = None
    kick_power: float | None = None


@dataclass(frozen=True)
class ActionSelection:
    selected: ActionCandidate
    finalists: tuple[ActionCandidate, ...]
    generated_count: int
    fallback_target: Pose2D


@dataclass(frozen=True)
class ActionSelectionTuning:
    shot_min_utility: float = 0.55
    pass_min_utility: float = 0.40
    dribble_min_utility: float = 0.10
    clear_min_utility: float = 0.0
    own_turn_weight: float = 0.32
    lane_weight: float = 1.65
    close_shot_power: float = 1.45
    normal_shot_power: float = 1.90
    power_shot_power: float = 2.25
    power_shot_min_distance_m: float = 3.0
    power_shot_min_lane: float = 0.68


class BoundedActionSelector:
    """Generate no more than thirteen candidates with no dynamic search."""

    def __init__(
        self,
        config: SoccerConfig,
        field: TeamFieldFrame,
        obstacles: ObstacleCollector,
        tuning: ActionSelectionTuning | None = None,
    ):
        self.config = config
        self.field = field
        self.obstacles = obstacles
        self.tuning = ActionSelectionTuning() if tuning is None else tuning

    def select(
        self,
        *,
        handler_id: int,
        context: PlayContext,
        pressure_level: str,
        fallback_target: Pose2D,
        risk_value: float = 0.5,
    ) -> ActionSelection:
        ball = context.known_ball
        if ball_near_sideline(self.config, ball):
            recovery = ActionCandidate(
                ActionKind.RECOVERY,
                fallback_target,
                utility=10.0,
                lane_score=1.0,
                turn_cost=0.0,
                reason="hard sideline recovery",
            )
            return ActionSelection(recovery, (recovery,), 1, fallback_target)

        robot = context.teammates.get(handler_id)
        if robot is None or robot.pose is None:
            fallback = ActionCandidate(
                ActionKind.DRIBBLE,
                fallback_target,
                utility=-10.0,
                lane_score=0.0,
                turn_cost=0.0,
                reason="missing Handler pose fallback",
            )
            return ActionSelection(fallback, (fallback,), 0, fallback_target)

        opponents = self.obstacles.opponent_obstacles(context)
        generated: list[ActionCandidate] = []
        risk = clamp(risk_value, 0.0, 1.0)
        generated.extend(self._shots(robot.pose, context, opponents, pressure_level, risk))
        generated.extend(self._passes(handler_id, robot.pose, context, opponents, pressure_level, risk))
        generated.extend(self._dribbles(robot.pose, context, opponents, pressure_level, risk))
        if ball_in_own_defensive_area(self.config, ball):
            generated.extend(self._clears(robot.pose, context, opponents, pressure_level, risk))

        finalists = self._best_per_kind(generated)
        if not finalists:
            fallback = ActionCandidate(
                ActionKind.DRIBBLE,
                fallback_target,
                utility=-1.0,
                lane_score=0.0,
                turn_cost=0.0,
                reason="no legal candidate fallback",
            )
            return ActionSelection(fallback, (fallback,), len(generated), fallback_target)
        selected = max(finalists, key=lambda item: (item.utility, -_kind_order(item.kind)))
        return ActionSelection(selected, finalists, len(generated), fallback_target)

    def _shots(self, pose, context, opponents, pressure_level, risk):
        ball = context.known_ball
        goal_x = self.field.opponent_goal_x()
        safe_half_width = max(0.10, self.config.goal_width / 2.0 - 0.22)
        candidates: list[ActionCandidate] = []
        for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
            target = Pose2D(goal_x, fraction * safe_half_width, 0.0)
            lane = lane_clear_score(
                self.config, ball.x, ball.y, target.x, target.y, opponents
            )
            turn = self._turn_cost(pose, ball.x, ball.y, target)
            distance = math.hypot(target.x - ball.x, target.y - ball.y)
            distance_penalty = clamp(distance / max(1.0, self.config.field_length), 0.0, 1.0)
            pressure_penalty = 0.45 if pressure_level == "immediate" and turn > 0.30 else 0.0
            utility = (
                self.tuning.lane_weight * lane
                + 0.85
                + 0.30 * clamp(ball.x / (self.config.field_length / 2.0), -1.0, 1.0)
                - 0.35 * distance_penalty
                - self.tuning.own_turn_weight * turn
                - pressure_penalty
                + (risk - 0.5) * 0.45
            )
            if utility >= self.tuning.shot_min_utility:
                if distance < 2.0:
                    kick_power = self.tuning.close_shot_power
                elif (
                    distance >= self.tuning.power_shot_min_distance_m
                    and lane >= self.tuning.power_shot_min_lane
                ):
                    kick_power = self.tuning.power_shot_power
                else:
                    kick_power = self.tuning.normal_shot_power
                candidates.append(
                    ActionCandidate(
                        ActionKind.SHOOT, target, utility, lane, turn,
                        f"shot lane={lane:.2f}", kick_power=kick_power,
                    )
                )
        return candidates

    def _passes(self, handler_id, pose, context, opponents, pressure_level, risk):
        ball = context.known_ball
        game = context.known_game
        scored: list[ActionCandidate] = []
        for receiver_id, teammate in sorted(context.teammates.items()):
            if receiver_id == handler_id or teammate.pose is None:
                continue
            if not game.is_active_player(self.config.team_id, receiver_id):
                continue
            target = teammate.pose
            lane = lane_clear_score(
                self.config, ball.x, ball.y, target.x, target.y, opponents
            )
            turn = self._turn_cost(pose, ball.x, ball.y, target)
            forward = clamp(
                (target.x - ball.x) / max(1.0, self.config.field_length / 2.0),
                -0.35,
                1.0,
            )
            distance = math.hypot(target.x - ball.x, target.y - ball.y)
            pressure_bonus = 0.30 if pressure_level in {"immediate", "urgent"} else 0.0
            utility = (
                self.tuning.lane_weight * lane
                + 0.70 * forward
                + pressure_bonus
                - 0.20 * clamp(distance / 8.0, 0.0, 1.0)
                - self.tuning.own_turn_weight * turn
                + (risk - 0.5) * max(0.0, forward) * 0.25
            )
            if utility >= self.tuning.pass_min_utility:
                scored.append(
                    ActionCandidate(
                        ActionKind.PASS, target, utility, lane, turn,
                        f"pass p{receiver_id} lane={lane:.2f}", receiver_id,
                        min(1.65, 1.05 + 0.10 * distance),
                    )
                )
        scored.sort(key=lambda item: (-item.utility, item.receiver_id or 0))
        return scored[:2]

    def _dribbles(self, pose, context, opponents, pressure_level, risk):
        if pressure_level == "immediate":
            return []
        ball = context.known_ball
        advance = self.config.strategy.dribble_advance_m
        candidates: list[ActionCandidate] = []
        for lateral in (-0.65, 0.0, 0.65):
            target = self.field.clamp_inside_field(
                Pose2D(ball.x + advance, ball.y + lateral, 0.0)
            )
            lane = lane_clear_score(
                self.config, ball.x, ball.y, target.x, target.y, opponents
            )
            turn = self._turn_cost(pose, ball.x, ball.y, target)
            center_safety = 1.0 - clamp(
                abs(target.y) / max(0.1, self.config.field_width / 2.0), 0.0, 1.0
            )
            utility = (
                1.25 * lane
                + 0.45
                + 0.15 * center_safety
                - self.tuning.own_turn_weight * turn
                - (0.20 if pressure_level == "urgent" else 0.0)
                + (risk - 0.5) * 0.20
            )
            if utility >= self.tuning.dribble_min_utility:
                candidates.append(
                    ActionCandidate(
                        ActionKind.DRIBBLE, target, utility, lane, turn,
                        f"dribble lane={lane:.2f}", kick_power=0.85,
                    )
                )
        return candidates

    def _clears(self, pose, context, opponents, pressure_level, risk):
        ball = context.known_ball
        forward_x = min(self.field.opponent_goal_x(), ball.x + 4.0)
        half_width = self.config.field_width / 2.0 - 0.30
        targets = (
            Pose2D(forward_x, -half_width, 0.0),
            Pose2D(forward_x, 0.0, 0.0),
            Pose2D(forward_x, half_width, 0.0),
        )
        candidates: list[ActionCandidate] = []
        for target in targets:
            lane = lane_clear_score(
                self.config, ball.x, ball.y, target.x, target.y, opponents
            )
            turn = self._turn_cost(pose, ball.x, ball.y, target)
            pressure_bonus = 1.0 if pressure_level in {"immediate", "urgent"} else 0.35
            center_risk = 0.35 if abs(target.y) < 0.2 else 0.0
            utility = (
                1.40 * lane + pressure_bonus - center_risk
                - self.tuning.own_turn_weight * turn
                + (0.5 - risk) * 0.55
            )
            if utility >= self.tuning.clear_min_utility:
                candidates.append(
                    ActionCandidate(
                        ActionKind.CLEAR, target, utility, lane, turn,
                        f"defensive clear lane={lane:.2f}", kick_power=2.20,
                    )
                )
        return candidates

    @staticmethod
    def _turn_cost(pose, ball_x: float, ball_y: float, target: Pose2D) -> float:
        desired = math.atan2(target.y - ball_y, target.x - ball_x)
        return abs(normalize_angle(desired - pose.theta)) / math.pi

    @staticmethod
    def _best_per_kind(candidates: list[ActionCandidate]) -> tuple[ActionCandidate, ...]:
        best: dict[ActionKind, ActionCandidate] = {}
        for candidate in candidates:
            previous = best.get(candidate.kind)
            if previous is None or candidate.utility > previous.utility:
                best[candidate.kind] = candidate
        return tuple(best[kind] for kind in ActionKind if kind in best)


def _kind_order(kind: ActionKind) -> int:
    return {
        ActionKind.RECOVERY: 0,
        ActionKind.SHOOT: 1,
        ActionKind.PASS: 2,
        ActionKind.CLEAR: 3,
        ActionKind.DRIBBLE: 4,
    }[kind]


__all__ = [
    "ActionCandidate",
    "ActionKind",
    "ActionSelection",
    "ActionSelectionTuning",
    "BoundedActionSelector",
]
