"""Bounded schema-v4 ball path prediction for strategy consumers.

The runtime is deliberately standard-library only.  It detects a sudden ball
acceleration, consumes the first five post-launch observations, and performs one
shared nearest-neighbour search for all calibrated future horizons.  Environment
interactions remain a caller concern: this module predicts the free path and
explicitly exposes uncertainty instead of assigning roles or choosing actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from ..soccer_framework import BallState, Pose2D
from .ball_event_model_data import (
    FEATURE_MEANS,
    FEATURE_SCALES,
    HELD_OUT_ERRORS_M,
    HORIZONS_SEC,
    INPUT_POINTS,
    MODEL_NAME,
    MODEL_NEIGHBORS,
    MODEL_ROWS,
)


__all__ = [
    "BallObservation",
    "BallPathKnot",
    "BallTrajectoryPrediction",
    "EventBallTrajectoryPredictor",
    "predict_launched_ball_path",
]


MAX_STRATEGY_HORIZON_SEC = 0.50
MAX_MODEL_HORIZON_SEC = HORIZONS_SEC[-1]
MAX_OBSERVATION_GAP_SEC = 0.25
MAX_PLAUSIBLE_SPEED_MPS = 7.0
MIN_LAUNCH_SPEED_MPS = 0.35
MIN_LAUNCH_SPEED_DELTA_MPS = 0.35
_EPS = 1e-9


@dataclass(frozen=True)
class BallObservation:
    """Timestamped public ball position in team-view field coordinates."""

    t_sec: float
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class BallPathKnot:
    """One calibrated future point relative to ``observed_at_sec``."""

    seconds: float
    position: Pose2D
    velocity_x: float
    velocity_y: float
    uncertainty_m: float
    confidence: float


@dataclass(frozen=True)
class BallTrajectoryPrediction:
    """Immutable free-path prediction with explicit validity diagnostics."""

    observed_at_sec: float
    current: Pose2D
    velocity_x: float
    velocity_y: float
    speed_mps: float
    motion_state: str
    sample_count: int
    window_sec: float
    valid_horizon_sec: float
    recommended_horizon_sec: float
    segment_id: int
    model_name: str
    invalid_reason: str | None
    knots: tuple[BallPathKnot, ...]

    @property
    def usable(self) -> bool:
        return self.invalid_reason is None and bool(self.knots)

    @property
    def stop_point(self) -> None:
        """The v4 event model does not claim a calibrated final stop point."""

        return None

    @property
    def time_to_stop_sec(self) -> None:
        return None

    def position_at(self, seconds: float) -> Pose2D | None:
        """Return a position relative to ``observed_at_sec`` without extrapolation."""

        pair = self._bracket(seconds)
        if pair is None:
            return None
        before, after, weight = pair
        return Pose2D(
            before.position.x + (after.position.x - before.position.x) * weight,
            before.position.y + (after.position.y - before.position.y) * weight,
            0.0,
        )

    def velocity_at(self, seconds: float) -> tuple[float, float] | None:
        pair = self._bracket(seconds)
        if pair is None:
            return None
        before, after, weight = pair
        return (
            before.velocity_x + (after.velocity_x - before.velocity_x) * weight,
            before.velocity_y + (after.velocity_y - before.velocity_y) * weight,
        )

    def uncertainty_at(self, seconds: float) -> float | None:
        pair = self._bracket(seconds)
        if pair is None:
            return None
        before, after, weight = pair
        return before.uncertainty_m + (after.uncertainty_m - before.uncertainty_m) * weight

    def confidence_at(self, seconds: float) -> float:
        pair = self._bracket(seconds)
        if pair is None:
            return 0.0
        before, after, weight = pair
        return max(0.0, min(1.0, before.confidence + (after.confidence - before.confidence) * weight))

    def position_ahead(self, now_sec: float, seconds_ahead: float) -> Pose2D | None:
        """Convenience query relative to now, accounting for cached prediction age."""

        return self.position_at(max(0.0, now_sec - self.observed_at_sec) + max(0.0, seconds_ahead))

    def is_strategy_usable(
        self,
        now_sec: float,
        seconds_ahead: float,
        *,
        min_confidence: float = 0.35,
        max_uncertainty_m: float = 0.45,
    ) -> bool:
        relative = max(0.0, now_sec - self.observed_at_sec) + max(0.0, seconds_ahead)
        uncertainty = self.uncertainty_at(relative)
        return (
            self.usable
            and relative <= self.recommended_horizon_sec + 1e-9
            and uncertainty is not None
            and uncertainty <= max_uncertainty_m
            and self.confidence_at(relative) >= min_confidence
        )

    def _bracket(self, seconds: float) -> tuple[BallPathKnot, BallPathKnot, float] | None:
        if not self.usable or seconds < -1e-9 or seconds > self.valid_horizon_sec + 1e-9:
            return None
        rows = self.knots
        for before, after in zip(rows, rows[1:]):
            if before.seconds - 1e-9 <= seconds <= after.seconds + 1e-9:
                span = after.seconds - before.seconds
                return before, after, 0.0 if span <= _EPS else (seconds - before.seconds) / span
        if rows and abs(seconds - rows[-1].seconds) <= 1e-9:
            return rows[-1], rows[-1], 0.0
        return None

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        observation: BallObservation | None,
        sample_count: int,
        segment_id: int,
    ) -> "BallTrajectoryPrediction":
        current = Pose2D(0.0, 0.0, 0.0) if observation is None else Pose2D(observation.x, observation.y, 0.0)
        return cls(
            observed_at_sec=0.0 if observation is None else observation.t_sec,
            current=current,
            velocity_x=0.0,
            velocity_y=0.0,
            speed_mps=0.0,
            motion_state="unavailable",
            sample_count=sample_count,
            window_sec=0.0,
            valid_horizon_sec=0.0,
            recommended_horizon_sec=0.0,
            segment_id=segment_id,
            model_name=MODEL_NAME,
            invalid_reason=reason,
            knots=(),
        )


def predict_launched_ball_path(
    observations: list[BallObservation] | tuple[BallObservation, ...],
    *,
    segment_id: int = 0,
) -> BallTrajectoryPrediction:
    """Predict from exactly the first five post-launch observations."""

    points = _unique_observations(observations)
    last = points[-1] if points else None
    if len(points) < INPUT_POINTS:
        return BallTrajectoryPrediction.unavailable(
            "warming_up", observation=last, sample_count=len(points), segment_id=segment_id
        )
    points = points[:INPUT_POINTS]
    if points[-1].t_sec - points[0].t_sec > 0.30:
        return BallTrajectoryPrediction.unavailable(
            "observation_window_too_long", observation=points[-1], sample_count=len(points), segment_id=segment_id
        )
    features, direction_x, direction_y, velocity_x, velocity_y, residual = _extract_features(points)
    speed = math.hypot(velocity_x, velocity_y)
    if speed < 0.05:
        return BallTrajectoryPrediction.unavailable(
            "settled_or_unobservable", observation=points[-1], sample_count=len(points), segment_id=segment_id
        )
    scaled = tuple((value - mean) / scale for value, mean, scale in zip(features, FEATURE_MEANS, FEATURE_SCALES))
    ranked = sorted(
        (
            sum((left - right) ** 2 for left, right in zip(scaled, row)),
            targets,
        )
        for row, targets in MODEL_ROWS
    )
    origin = points[-1]
    predictions: list[tuple[float, float, float, float, float]] = []
    for horizon in HORIZONS_SEC:
        nearest = []
        for distance, targets in ranked:
            target = next(((longitudinal, lateral) for seconds, longitudinal, lateral in targets if seconds == horizon), None)
            if target is not None:
                nearest.append((distance, target))
                if len(nearest) >= MODEL_NEIGHBORS:
                    break
        if not nearest:
            continue
        weights = [1.0 / (0.04 + distance) for distance, _ in nearest]
        total_weight = sum(weights)
        longitudinal = sum(weight * target[0] for weight, (_, target) in zip(weights, nearest)) / total_weight
        lateral = sum(weight * target[1] for weight, (_, target) in zip(weights, nearest)) / total_weight
        spread = math.sqrt(
            sum(
                weight * ((target[0] - longitudinal) ** 2 + (target[1] - lateral) ** 2)
                for weight, (_, target) in zip(weights, nearest)
            )
            / total_weight
        )
        nearest_distance = math.sqrt(max(0.0, nearest[0][0]))
        held_out_p75 = HELD_OUT_ERRORS_M[horizon][1]
        uncertainty = max(held_out_p75, 1.20 * spread) * (1.0 + 0.08 * nearest_distance)
        quality = 1.0 / (1.0 + 0.12 * nearest_distance + 8.0 * residual)
        confidence = quality * math.exp(-uncertainty / 0.55)
        x = origin.x + longitudinal * direction_x - lateral * direction_y
        y = origin.y + longitudinal * direction_y + lateral * direction_x
        predictions.append((horizon, x, y, uncertainty, confidence))
    if not predictions:
        return BallTrajectoryPrediction.unavailable(
            "model_has_no_supported_horizon", observation=origin, sample_count=len(points), segment_id=segment_id
        )
    knots = _build_knots(origin, velocity_x, velocity_y, residual, predictions)
    return BallTrajectoryPrediction(
        observed_at_sec=origin.t_sec,
        current=Pose2D(origin.x, origin.y, 0.0),
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed_mps=speed,
        motion_state="rolling",
        sample_count=len(points),
        window_sec=origin.t_sec - points[0].t_sec,
        valid_horizon_sec=knots[-1].seconds,
        recommended_horizon_sec=min(MAX_STRATEGY_HORIZON_SEC, knots[-1].seconds),
        segment_id=segment_id,
        model_name=MODEL_NAME,
        invalid_reason=None,
        knots=knots,
    )


class EventBallTrajectoryPredictor:
    """Detect launch segments and retain one bounded immutable prediction."""

    def __init__(self) -> None:
        self._recent: list[BallObservation] = []
        self._segment: list[BallObservation] = []
        self._prediction: BallTrajectoryPrediction | None = None
        self._segment_id = 0
        self._last_reset_reason = "not_started"

    def reset(self, reason: str = "manual_reset") -> None:
        self._segment.clear()
        self._prediction = None
        self._last_reset_reason = reason

    def add_ball(self, ball: BallState, now_sec: float | None = None) -> None:
        timestamp = ball.last_seen_at if now_sec is None else now_sec
        self.add_observation(BallObservation(timestamp, ball.x, ball.y, ball.confidence))

    def add_observation(self, observation: BallObservation) -> None:
        if not _finite_observation(observation):
            self.reset("invalid_observation")
            return
        if self._recent:
            previous = self._recent[-1]
            dt = observation.t_sec - previous.t_sec
            if dt <= 1e-5:
                return
            speed = _speed(previous, observation)
            if dt > MAX_OBSERVATION_GAP_SEC:
                self.reset("observation_gap")
                self._recent = [observation]
                return
            if speed > MAX_PLAUSIBLE_SPEED_MPS:
                self.reset("position_jump")
                self._recent = [observation]
                return
            previous_speed = _speed(self._recent[-2], previous) if len(self._recent) >= 2 else 0.0
            if self._segment and _interaction_changed_direction(self._recent, observation):
                self.reset("direction_change")
            if self._segment and len(self._segment) >= INPUT_POINTS and previous_speed >= 0.35 and (
                speed <= 0.12 or previous_speed - speed > 1.00
            ):
                self.reset("speed_drop")
            if self._segment and len(self._segment) >= INPUT_POINTS and speed - previous_speed > 0.90:
                self.reset("new_acceleration")
            if not self._segment and speed >= MIN_LAUNCH_SPEED_MPS and (
                previous_speed <= 0.12 or speed - previous_speed >= MIN_LAUNCH_SPEED_DELTA_MPS
            ):
                self._segment_id += 1
                self._segment = [observation]
                self._prediction = None
                self._last_reset_reason = "automatic_acceleration"
            elif self._segment and len(self._segment) < INPUT_POINTS:
                self._segment.append(observation)
                if len(self._segment) == INPUT_POINTS:
                    self._prediction = predict_launched_ball_path(self._segment, segment_id=self._segment_id)
        self._recent.append(observation)
        self._recent = self._recent[-4:]

    def predict(self) -> BallTrajectoryPrediction:
        if self._prediction is not None:
            return self._prediction
        last = self._recent[-1] if self._recent else None
        reason = "warming_up" if self._segment else self._last_reset_reason
        return BallTrajectoryPrediction.unavailable(
            reason, observation=last, sample_count=len(self._segment), segment_id=self._segment_id
        )

    def position_ahead(self, now_sec: float, seconds_ahead: float) -> Pose2D | None:
        return self.predict().position_ahead(now_sec, seconds_ahead)


def _unique_observations(observations: list[BallObservation] | tuple[BallObservation, ...]) -> list[BallObservation]:
    result: list[BallObservation] = []
    for observation in sorted(observations, key=lambda row: row.t_sec):
        if result and observation.t_sec - result[-1].t_sec <= 1e-5:
            result[-1] = observation
        else:
            result.append(observation)
    return result


def _linear_fit(points: list[BallObservation], component: str) -> tuple[float, float]:
    end = points[-1].t_sec
    rows = []
    for index, point in enumerate(points):
        offset = point.t_sec - end
        value = point.x if component == "x" else point.y
        weight = max(0.05, point.confidence) * (1.0 + index)
        rows.append((offset, value, weight))
    sw = sum(row[2] for row in rows)
    st = sum(offset * weight for offset, _, weight in rows)
    sy = sum(value * weight for _, value, weight in rows)
    stt = sum(offset * offset * weight for offset, _, weight in rows)
    sty = sum(offset * value * weight for offset, value, weight in rows)
    denominator = sw * stt - st * st
    slope = 0.0 if abs(denominator) <= _EPS else (sw * sty - st * sy) / denominator
    intercept = (sy - slope * st) / max(_EPS, sw)
    residual = math.sqrt(
        sum(weight * (value - intercept - slope * offset) ** 2 for offset, value, weight in rows)
        / max(_EPS, sw)
    )
    return slope, residual


def _extract_features(
    points: list[BallObservation],
) -> tuple[tuple[float, ...], float, float, float, float, float]:
    velocity_x, residual_x = _linear_fit(points, "x")
    velocity_y, residual_y = _linear_fit(points, "y")
    speed = math.hypot(velocity_x, velocity_y)
    if speed <= 0.03:
        dx, dy = points[-1].x - points[0].x, points[-1].y - points[0].y
        norm = math.hypot(dx, dy)
        direction_x, direction_y = ((1.0, 0.0) if norm <= _EPS else (dx / norm, dy / norm))
    else:
        direction_x, direction_y = velocity_x / speed, velocity_y / speed
    velocities = []
    for before, after in zip(points, points[1:]):
        dt = after.t_sec - before.t_sec
        if dt > 1e-4:
            velocities.append(((after.x - before.x) / dt, (after.y - before.y) / dt))
    projected = [vx * direction_x + vy * direction_y for vx, vy in velocities]
    lateral = [-vx * direction_y + vy * direction_x for vx, vy in velocities]
    recent_speed = statistics.median(projected[-2:])
    early_speed = statistics.median(projected[:2])
    span = max(1e-4, points[-1].t_sec - points[0].t_sec)
    acceleration = (recent_speed - early_speed) / span
    median_speed = statistics.median(projected)
    features = (
        speed,
        speed * speed,
        recent_speed,
        early_speed,
        acceleration,
        max(-12.0, min(12.0, acceleration)) * speed,
        statistics.median(lateral[-2:]),
        statistics.median(abs(value - median_speed) for value in projected),
        math.hypot(residual_x, residual_y),
        span,
        statistics.median(point.confidence for point in points),
    )
    return features, direction_x, direction_y, velocity_x, velocity_y, math.hypot(residual_x, residual_y)


def _build_knots(
    origin: BallObservation,
    velocity_x: float,
    velocity_y: float,
    residual: float,
    predictions: list[tuple[float, float, float, float, float]],
) -> tuple[BallPathKnot, ...]:
    raw = [(0.0, origin.x, origin.y, max(0.01, residual), max(0.0, 1.0 - 8.0 * residual))] + predictions
    knots = []
    for index, (seconds, x, y, uncertainty, confidence) in enumerate(raw):
        if index == 0:
            vx, vy = velocity_x, velocity_y
        elif index == len(raw) - 1:
            before = raw[index - 1]
            dt = seconds - before[0]
            vx, vy = (x - before[1]) / dt, (y - before[2]) / dt
        else:
            before, after = raw[index - 1], raw[index + 1]
            dt = after[0] - before[0]
            vx, vy = (after[1] - before[1]) / dt, (after[2] - before[2]) / dt
        knots.append(BallPathKnot(seconds, Pose2D(x, y, 0.0), vx, vy, uncertainty, confidence))
    return tuple(knots)


def _speed(before: BallObservation, after: BallObservation) -> float:
    dt = after.t_sec - before.t_sec
    return 0.0 if dt <= 1e-5 else math.hypot(after.x - before.x, after.y - before.y) / dt


def _interaction_changed_direction(recent: list[BallObservation], current: BallObservation) -> bool:
    if len(recent) < 2:
        return False
    before, previous = recent[-2], recent[-1]
    old_x, old_y = previous.x - before.x, previous.y - before.y
    new_x, new_y = current.x - previous.x, current.y - previous.y
    old_length, new_length = math.hypot(old_x, old_y), math.hypot(new_x, new_y)
    if old_length < 0.01 or new_length < 0.01:
        return False
    return (old_x * new_x + old_y * new_y) / (old_length * new_length) < 0.20


def _finite_observation(observation: BallObservation) -> bool:
    return all(math.isfinite(value) for value in (observation.t_sec, observation.x, observation.y, observation.confidence)) and observation.confidence > 0.0
