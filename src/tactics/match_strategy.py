"""Bounded match-management models for set plays and low-frequency adaptation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math

from ..soccer_framework import GameState, PlayContext, Pose2D, SetPlay, SoccerConfig
from .geometry import TeamFieldFrame, clamp


class KickoffPhase(str, Enum):
    IDLE = "idle"
    FIRST_TOUCH_ACTIVE = "first_touch_active"
    VERIFY_FIRST_TOUCH = "verify_first_touch"
    SECOND_PLAYER_ACQUIRE = "second_player_acquire"
    SECOND_KICK_ACTIVE = "second_kick_active"
    COMPLETE = "complete"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class KickoffStatus:
    epoch: int
    phase: KickoffPhase
    active: bool
    first_player_id: int | None
    second_player_id: int | None
    first_target: Pose2D | None
    second_target: Pose2D | None
    receiver_target: Pose2D | None
    first_player_hold_target: Pose2D | None
    ball_speed_mps: float
    elapsed_sec: float
    reason: str


class KickoffTransaction:
    """Persistent two-touch kickoff state inferred only from public observations."""

    def __init__(self, config: SoccerConfig, field: TeamFieldFrame):
        self.config = config
        self.field = field
        self._epoch = 0
        self._phase = KickoffPhase.IDLE
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._initial_ball: Pose2D | None = None
        self._second_ball: Pose2D | None = None
        self._first_id: int | None = None
        self._second_id: int | None = None
        self._saw_own_kickoff = False
        self._last_ball: Pose2D | None = None
        self._last_ball_at: float | None = None
        self._ball_velocity = (0.0, 0.0)

    def update(
        self,
        *,
        now_sec: float,
        context: PlayContext,
        first_player_id: int | None,
        second_player_id: int | None,
    ) -> KickoffStatus:
        game = context.known_game
        ball = context.known_ball
        own_kickoff = game.is_kickoff_for_team(self.config.team_id)
        new_edge = own_kickoff and not self._saw_own_kickoff
        self._saw_own_kickoff = own_kickoff

        if new_edge and game.state is GameState.PLAYING:
            self._epoch += 1
            self._phase = KickoffPhase.FIRST_TOUCH_ACTIVE
            self._started_at = now_sec
            self._phase_started_at = now_sec
            self._initial_ball = Pose2D(ball.x, ball.y, 0.0)
            self._second_ball = None
            self._first_id = first_player_id
            self._second_id = second_player_id
            self._last_ball = Pose2D(ball.x, ball.y, 0.0)
            self._last_ball_at = now_sec
            self._ball_velocity = (0.0, 0.0)

        self._update_ball_velocity(now_sec, ball.x, ball.y)

        if self._phase in {KickoffPhase.IDLE, KickoffPhase.COMPLETE, KickoffPhase.FALLBACK}:
            return self._status(now_sec)
        if now_sec - self._started_at >= 10.0:
            self._phase = KickoffPhase.FALLBACK
            return self._status(now_sec, "kickoff transaction timeout")
        if self._first_id is None or self._second_id is None:
            self._phase = KickoffPhase.FALLBACK
            return self._status(now_sec, "two field players unavailable")

        initial_move = self._distance_from(ball.x, ball.y, self._initial_ball)
        first_near = self._player_near(context, self._first_id, ball.x, ball.y, 1.0)
        opponent_clear = all(
            opponent.pose is None
            or math.hypot(ball.x - opponent.pose.x, ball.y - opponent.pose.y) > 0.35
            for opponent in context.opponents.values()
        )
        if (
            self._phase is KickoffPhase.FIRST_TOUCH_ACTIVE
            and initial_move >= 0.10
            and first_near
            and opponent_clear
        ):
            self._enter(KickoffPhase.VERIFY_FIRST_TOUCH, now_sec)
        elif self._phase is KickoffPhase.VERIFY_FIRST_TOUCH:
            if initial_move >= 0.16:
                self._enter(KickoffPhase.SECOND_PLAYER_ACQUIRE, now_sec)
            elif now_sec - self._phase_started_at >= 1.0:
                self._enter(KickoffPhase.FIRST_TOUCH_ACTIVE, now_sec)
        elif self._phase is KickoffPhase.SECOND_PLAYER_ACQUIRE:
            second = context.teammates.get(self._second_id)
            if second is not None and second.pose is not None:
                distance = math.hypot(ball.x - second.pose.x, ball.y - second.pose.y)
                if distance <= 0.62 and self._ball_speed() <= 0.22:
                    self._second_ball = Pose2D(ball.x, ball.y, 0.0)
                    self._enter(KickoffPhase.SECOND_KICK_ACTIVE, now_sec)
        elif self._phase is KickoffPhase.SECOND_KICK_ACTIVE:
            if (
                self._distance_from(ball.x, ball.y, self._second_ball) >= 0.30
                and self._player_near(context, self._second_id, ball.x, ball.y, 1.0)
            ):
                self._enter(KickoffPhase.COMPLETE, now_sec)
        return self._status(now_sec)

    def _update_ball_velocity(self, now_sec: float, x: float, y: float) -> None:
        previous = self._last_ball
        previous_at = self._last_ball_at
        if previous is not None and previous_at is not None:
            dt = now_sec - previous_at
            if 0.02 <= dt <= 0.50:
                measured = ((x - previous.x) / dt, (y - previous.y) / dt)
                self._ball_velocity = (
                    0.55 * measured[0] + 0.45 * self._ball_velocity[0],
                    0.55 * measured[1] + 0.45 * self._ball_velocity[1],
                )
        self._last_ball = Pose2D(x, y, 0.0)
        self._last_ball_at = now_sec

    def _ball_speed(self) -> float:
        return math.hypot(*self._ball_velocity)

    def _enter(self, phase: KickoffPhase, now_sec: float) -> None:
        self._phase = phase
        self._phase_started_at = now_sec

    @staticmethod
    def _distance_from(x: float, y: float, origin: Pose2D | None) -> float:
        return 0.0 if origin is None else math.hypot(x - origin.x, y - origin.y)

    @staticmethod
    def _player_near(
        context: PlayContext,
        player_id: int | None,
        ball_x: float,
        ball_y: float,
        distance_m: float,
    ) -> bool:
        if player_id is None:
            return False
        robot = context.teammates.get(player_id)
        return (
            robot is not None
            and robot.pose is not None
            and math.hypot(ball_x - robot.pose.x, ball_y - robot.pose.y) <= distance_m
        )

    def _status(self, now_sec: float, reason: str | None = None) -> KickoffStatus:
        first_target = None
        second_target = None
        receiver_target = None
        first_player_hold_target = None
        if self._initial_ball is not None:
            # Keep the first touch short so the receiver collects a settled ball.
            first_target = self.field.clamp_inside_field(
                Pose2D(self._initial_ball.x + 0.24, self._initial_ball.y + 0.10, 0.0)
            )
            lead_sec = min(0.28, 0.08 + 0.10 * self._ball_speed())
            current = self._last_ball or self._initial_ball
            receiver_target = self.field.clamp_inside_field(
                Pose2D(
                    current.x + self._ball_velocity[0] * lead_sec,
                    current.y + self._ball_velocity[1] * lead_sec,
                    0.0,
                )
            )
            first_player_hold_target = self.field.clamp_inside_field(
                Pose2D(
                    min(-0.45, self._initial_ball.x - 0.65),
                    self._initial_ball.y - 0.75,
                    0.0,
                )
            )
            second_target = Pose2D(
                self.field.opponent_goal_x(),
                -min(0.75, self.config.goal_width / 2.0 - 0.25),
                0.0,
            )
        phase_reason = reason or self._phase.value.replace("_", " ")
        return KickoffStatus(
            epoch=self._epoch,
            phase=self._phase,
            active=self._phase not in {
                KickoffPhase.IDLE, KickoffPhase.COMPLETE, KickoffPhase.FALLBACK
            },
            first_player_id=self._first_id,
            second_player_id=self._second_id,
            first_target=first_target,
            second_target=second_target,
            receiver_target=receiver_target,
            first_player_hold_target=first_player_hold_target,
            ball_speed_mps=self._ball_speed(),
            elapsed_sec=max(0.0, now_sec - self._started_at) if self._epoch else 0.0,
            reason=phase_reason,
        )


@dataclass(frozen=True)
class MatchRisk:
    value: float
    score_difference: int
    own_active: int
    opponent_active: int
    reason: str


def calculate_match_risk(
    config: SoccerConfig,
    context: PlayContext,
    pressure_level: str,
) -> MatchRisk:
    """Return continuous attacking risk appetite in [0, 1]."""

    game = context.known_game
    own = game.get_team_state(config.team_id)
    opponent = game.get_team_state(config.opponent_team_id())
    own_score = 0 if own is None else own.score
    opponent_score = 0 if opponent is None else opponent.score
    score_difference = own_score - opponent_score
    own_active = sum(
        game.is_active_player(config.team_id, pid) for pid in config.player_ids
    )
    opponent_active = sum(
        game.is_active_player(config.opponent_team_id(), pid)
        for pid in config.player_ids
    )
    late = 1.0 - clamp(game.secs_remaining / 600.0, 0.0, 1.0)
    value = 0.50
    value += clamp(-score_difference * 0.16, -0.32, 0.32) * (0.45 + 0.55 * late)
    value += clamp((opponent_active - own_active) * -0.12, -0.24, 0.24)
    value += 0.08 if context.known_ball.x > 0.0 else -0.05
    value -= 0.12 if pressure_level == "immediate" else 0.0
    value = clamp(value, 0.0, 1.0)
    return MatchRisk(
        value=value,
        score_difference=score_difference,
        own_active=own_active,
        opponent_active=opponent_active,
        reason=(
            f"score={score_difference:+d} time={game.secs_remaining}s "
            f"players={own_active}v{opponent_active}"
        ),
    )


@dataclass(frozen=True)
class OpponentShape:
    ball_density: float
    deepest_x: float | None
    lateral_compactness: float
    samples: int


class OpponentShapeTracker:
    """Maintain three public-position features at no more than 2 Hz."""

    def __init__(self, max_samples: int = 20, sample_period_sec: float = 0.5):
        self._samples: deque[tuple[float, float, float]] = deque(
            maxlen=max(1, max_samples)
        )
        self._period = max(0.5, sample_period_sec)
        self._last_at = -math.inf

    def update(self, now_sec: float, context: PlayContext) -> OpponentShape:
        poses = tuple(
            robot.pose for robot in context.opponents.values() if robot.pose is not None
        )
        if now_sec - self._last_at >= self._period and poses:
            ball = context.known_ball
            density = sum(
                math.hypot(pose.x - ball.x, pose.y - ball.y) <= 1.5
                for pose in poses
            ) / len(poses)
            deepest = max(pose.x for pose in poses)
            ys = [pose.y for pose in poses]
            compactness = max(ys) - min(ys) if len(ys) > 1 else 0.0
            self._samples.append((density, deepest, compactness))
            self._last_at = now_sec
        if not self._samples:
            return OpponentShape(0.0, None, 0.0, 0)
        count = len(self._samples)
        return OpponentShape(
            ball_density=sum(item[0] for item in self._samples) / count,
            deepest_x=sum(item[1] for item in self._samples) / count,
            lateral_compactness=sum(item[2] for item in self._samples) / count,
            samples=count,
        )


def own_restart_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    context: PlayContext,
    fallback: Pose2D,
    *,
    corner_lane_open: bool = True,
) -> Pose2D:
    """Deterministic one-touch target for non-kickoff own restarts."""

    game = context.known_game
    ball = context.known_ball
    if not game.is_restart_for_team(config.team_id) or game.set_play is SetPlay.NONE:
        return fallback
    if game.set_play is SetPlay.THROW_IN:
        return field.clamp_inside_field(
            Pose2D(ball.x + 1.7, ball.y * 0.55, 0.0)
        )
    if game.set_play is SetPlay.GOAL_KICK:
        side = 1.0 if ball.y <= 0.0 else -1.0
        return field.clamp_inside_field(Pose2D(ball.x + 3.2, side * 2.2, 0.0))
    if game.set_play is SetPlay.CORNER_KICK:
        if not corner_lane_open:
            return fallback
        far_post_y = -0.75 if ball.y >= 0.0 else 0.75
        return Pose2D(field.opponent_goal_x(), far_post_y, 0.0)
    return fallback


__all__ = [
    "KickoffPhase",
    "KickoffStatus",
    "KickoffTransaction",
    "MatchRisk",
    "OpponentShape",
    "OpponentShapeTracker",
    "calculate_match_risk",
    "own_restart_target",
]
