"""Conservative opponent pressure-time estimation.

Only public opponent poses are available.  The estimator may derive recent
velocity for diagnostics, but pressure time deliberately assumes each active
opponent can immediately pursue at the configured maximum speed.  This gives a
conservative earliest-arrival bound and never relies on hidden intent/state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..soccer_framework import GameControlState, Pose2D, RobotState
from .ball_prediction import BallMotionPrediction


__all__ = [
    "OpponentMotion",
    "OpponentMotionTracker",
    "OpponentPressureConfig",
    "OpponentPressureEstimate",
    "OpponentPressureReport",
    "OpponentPressureEstimator",
]


@dataclass(frozen=True)
class OpponentPressureConfig:
    opponent_max_speed_mps: float = 0.80
    control_radius_m: float = 0.55
    prediction_horizon_sec: float = 3.0
    search_step_sec: float = 0.05
    max_pose_age_sec: float = 0.50
    velocity_window_sec: float = 0.45


@dataclass(frozen=True)
class OpponentMotion:
    player_id: int
    velocity_x: float
    velocity_y: float
    speed_mps: float
    sample_count: int
    window_sec: float


@dataclass(frozen=True)
class _PoseObservation:
    t_sec: float
    x: float
    y: float


class OpponentMotionTracker:
    """Estimate observed opponent velocity from public pose history."""

    def __init__(self, window_sec: float = 0.45, max_samples: int = 30):
        self.window_sec = window_sec
        self.max_samples = max_samples
        self._history: dict[int, list[_PoseObservation]] = {}

    def reset(self, player_id: int | None = None) -> None:
        if player_id is None:
            self._history.clear()
        else:
            self._history.pop(player_id, None)

    def update(self, opponents: dict[int, RobotState], now_sec: float) -> None:
        for player_id, opponent in opponents.items():
            if opponent.pose is None:
                continue
            stamp = opponent.last_seen_at if opponent.last_seen_at > 0.0 else now_sec
            history = self._history.setdefault(player_id, [])
            if history and stamp <= history[-1].t_sec + 1e-4:
                continue
            if history and stamp - history[-1].t_sec > max(0.25, self.window_sec):
                history.clear()
            history.append(_PoseObservation(stamp, opponent.pose.x, opponent.pose.y))
            earliest = stamp - self.window_sec
            self._history[player_id] = [item for item in history if item.t_sec >= earliest][
                -self.max_samples :
            ]

    def motion(self, player_id: int) -> OpponentMotion | None:
        history = self._history.get(player_id) or []
        if len(history) < 2:
            return None
        start = history[0].t_sec
        times = [item.t_sec - start for item in history]
        vx = _linear_slope(times, [item.x for item in history])
        vy = _linear_slope(times, [item.y for item in history])
        if vx is None or vy is None:
            return None
        return OpponentMotion(
            player_id=player_id,
            velocity_x=vx,
            velocity_y=vy,
            speed_mps=math.hypot(vx, vy),
            sample_count=len(history),
            window_sec=max(0.0, history[-1].t_sec - history[0].t_sec),
        )


@dataclass(frozen=True)
class OpponentPressureEstimate:
    player_id: int
    time_to_pressure_sec: float | None
    contest_point: Pose2D | None
    current_distance_m: float
    pose_age_sec: float
    confidence: float
    closing_speed_mps: float | None
    observed_motion: OpponentMotion | None


@dataclass(frozen=True)
class OpponentPressureReport:
    earliest: OpponentPressureEstimate | None
    estimates: tuple[OpponentPressureEstimate, ...]
    considered_opponents: int
    stale_opponents: int
    inactive_opponents: int
    horizon_sec: float

    @property
    def pressure_time_sec(self) -> float | None:
        return None if self.earliest is None else self.earliest.time_to_pressure_sec

    def is_urgent(self, threshold_sec: float = 1.0) -> bool:
        pressure = self.pressure_time_sec
        return pressure is not None and pressure <= threshold_sec

    @property
    def has_unknown_pressure(self) -> bool:
        return self.stale_opponents > 0


class OpponentPressureEstimator:
    """Estimate the earliest active opponent arrival at the ball/target."""

    def __init__(
        self,
        config: OpponentPressureConfig | None = None,
        tracker: OpponentMotionTracker | None = None,
    ):
        self.config = OpponentPressureConfig() if config is None else config
        self.tracker = tracker or OpponentMotionTracker(self.config.velocity_window_sec)

    def estimate(
        self,
        *,
        now_sec: float,
        opponents: dict[int, RobotState],
        target: Pose2D | None = None,
        ball_prediction: BallMotionPrediction | None = None,
        game_state: GameControlState | None = None,
        opponent_team_id: int | None = None,
    ) -> OpponentPressureReport:
        """Return pressure against a stationary target or predicted ball path."""

        if target is None and ball_prediction is None:
            raise ValueError("target or ball_prediction is required")
        self.tracker.update(opponents, now_sec)
        estimates: list[OpponentPressureEstimate] = []
        stale = 0
        inactive = 0

        for player_id, opponent in sorted(opponents.items()):
            if (
                game_state is not None
                and opponent_team_id is not None
                and not game_state.is_active_player(opponent_team_id, player_id)
            ):
                inactive += 1
                continue
            if opponent.pose is None:
                stale += 1
                continue
            pose_age = (
                max(0.0, now_sec - opponent.last_seen_at)
                if opponent.last_seen_at > 0.0
                else 0.0
            )
            if pose_age > self.config.max_pose_age_sec:
                stale += 1
                continue

            current_target = ball_prediction.current if ball_prediction is not None else target
            assert current_target is not None
            current_distance = math.hypot(
                current_target.x - opponent.pose.x,
                current_target.y - opponent.pose.y,
            )
            pressure_time, contest_point = self._earliest_arrival(
                opponent.pose,
                target,
                ball_prediction,
            )
            motion = self.tracker.motion(player_id)
            closing_speed = self._closing_speed(opponent.pose, current_target, motion, ball_prediction)
            age_score = max(0.0, 1.0 - pose_age / max(1e-6, self.config.max_pose_age_sec))
            ball_score = 1.0 if ball_prediction is None else ball_prediction.confidence
            confidence = max(0.0, min(1.0, age_score * (0.5 + 0.5 * ball_score)))
            estimates.append(
                OpponentPressureEstimate(
                    player_id=player_id,
                    time_to_pressure_sec=pressure_time,
                    contest_point=contest_point,
                    current_distance_m=current_distance,
                    pose_age_sec=pose_age,
                    confidence=confidence,
                    closing_speed_mps=closing_speed,
                    observed_motion=motion,
                )
            )

        reachable = [item for item in estimates if item.time_to_pressure_sec is not None]
        earliest = min(reachable, key=lambda item: item.time_to_pressure_sec or 0.0) if reachable else None
        return OpponentPressureReport(
            earliest=earliest,
            estimates=tuple(
                sorted(
                    estimates,
                    key=lambda item: (
                        math.inf if item.time_to_pressure_sec is None else item.time_to_pressure_sec,
                        item.player_id,
                    ),
                )
            ),
            considered_opponents=len(estimates),
            stale_opponents=stale,
            inactive_opponents=inactive,
            horizon_sec=self.config.prediction_horizon_sec,
        )

    def _earliest_arrival(
        self,
        opponent_pose: Pose2D,
        target: Pose2D | None,
        ball_prediction: BallMotionPrediction | None,
    ) -> tuple[float | None, Pose2D | None]:
        speed = max(0.05, self.config.opponent_max_speed_mps)
        radius = max(0.0, self.config.control_radius_m)
        if ball_prediction is None:
            assert target is not None
            distance = math.hypot(target.x - opponent_pose.x, target.y - opponent_pose.y)
            arrival = max(0.0, distance - radius) / speed
            if arrival > self.config.prediction_horizon_sec:
                return None, None
            return arrival, target

        horizon = max(0.0, self.config.prediction_horizon_sec)
        step = max(0.01, self.config.search_step_sec)
        previous_t = 0.0
        previous_gap = self._reach_gap(opponent_pose, ball_prediction.position_at(0.0), 0.0, speed, radius)
        if previous_gap <= 0.0:
            return 0.0, ball_prediction.position_at(0.0)
        t = step
        while t <= horizon + 1e-9:
            point = ball_prediction.position_at(t)
            gap = self._reach_gap(opponent_pose, point, t, speed, radius)
            if gap <= 0.0:
                ratio = previous_gap / max(1e-9, previous_gap - gap)
                arrival = previous_t + ratio * (t - previous_t)
                return arrival, ball_prediction.position_at(arrival)
            previous_t = t
            previous_gap = gap
            t += step
        return None, None

    @staticmethod
    def _reach_gap(
        opponent_pose: Pose2D,
        point: Pose2D,
        t_sec: float,
        speed_mps: float,
        radius_m: float,
    ) -> float:
        distance = math.hypot(point.x - opponent_pose.x, point.y - opponent_pose.y)
        return distance - radius_m - speed_mps * t_sec

    @staticmethod
    def _closing_speed(
        opponent_pose: Pose2D,
        target: Pose2D,
        motion: OpponentMotion | None,
        ball_prediction: BallMotionPrediction | None,
    ) -> float | None:
        if motion is None:
            return None
        dx = target.x - opponent_pose.x
        dy = target.y - opponent_pose.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return 0.0
        target_vx = 0.0 if ball_prediction is None else ball_prediction.velocity_x
        target_vy = 0.0 if ball_prediction is None else ball_prediction.velocity_y
        relative_vx = target_vx - motion.velocity_x
        relative_vy = target_vy - motion.velocity_y
        return -(dx * relative_vx + dy * relative_vy) / distance


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 1e-10:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
