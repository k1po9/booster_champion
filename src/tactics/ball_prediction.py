"""Online rolling-ball state estimation for tactical decisions.

The module is pure: it owns no ROS, behavior-tree, or robot-control state.  It
estimates endpoint velocity from a short weighted observation window and uses a
cross-validated resistance model to expose future positions, stop region, and
confidence diagnostics.  Strategy callers must keep a current-ball fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..soccer_framework import BallState, Pose2D


__all__ = [
    "BallMotionPrediction",
    "BallObservation",
    "BallResistanceModel",
    "BallStopPrediction",
    "DEFAULT_BALL_RESISTANCE_MODEL",
    "SlidingWindowBallPredictor",
    "predict_ball_motion",
    "predict_ball_stop",
]


# The quadratic model was selected by leave-one-match-file-out validation over
# dataset/v1.  It is approved for confidence-weighted advisory use, not as an
# unconditional hard target.  See analysis/v1/ball_motion_cross_validation.md.
DEFAULT_BALL_RESISTANCE_MODEL: "BallResistanceModel"
DEFAULT_BALL_DECEL_MPS2 = 3.233  # Legacy explicit constant-model default.
DEFAULT_WINDOW_SEC = 0.45
DEFAULT_SETTLED_SPEED_MPS = 0.05
DEFAULT_MAX_OBSERVATION_GAP_SEC = 0.25
DEFAULT_MAX_PLAUSIBLE_SPEED_MPS = 6.0
_MIN_DT = 1e-4


@dataclass(frozen=True)
class BallObservation:
    """A timestamped ball position in team-view field coordinates."""

    t_sec: float
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class BallResistanceModel:
    """Empirical rolling resistance ``a(v) = a0 + k*v**speed_power``."""

    name: str
    a0_mps2: float
    k: float = 0.0
    speed_power: int = 0

    def deceleration(self, speed_mps: float) -> float:
        speed = max(0.0, speed_mps)
        return max(0.01, self.a0_mps2) + max(0.0, self.k) * speed**self.speed_power

    def stop_time_and_distance(self, speed_mps: float) -> tuple[float, float]:
        speed = max(0.0, speed_mps)
        a0 = max(0.01, self.a0_mps2)
        k = max(0.0, self.k)
        if speed <= 0.0:
            return 0.0, 0.0
        if self.speed_power == 0 or k <= 1e-8:
            return speed / a0, speed * speed / (2.0 * a0)
        if self.speed_power == 1:
            ratio = k * speed / a0
            stop_time = math.log1p(ratio) / k
            stop_distance = speed / k - a0 * math.log1p(ratio) / (k * k)
            return stop_time, max(0.0, stop_distance)
        if self.speed_power == 2:
            stop_time = math.atan(speed * math.sqrt(k / a0)) / math.sqrt(a0 * k)
            stop_distance = math.log1p(k * speed * speed / a0) / (2.0 * k)
            return stop_time, max(0.0, stop_distance)
        raise ValueError("speed_power must be 0, 1, or 2")

    def speed_and_distance_after(self, speed_mps: float, seconds: float) -> tuple[float, float]:
        """Return remaining speed and traveled distance after non-negative time."""

        speed = max(0.0, speed_mps)
        elapsed = max(0.0, seconds)
        stop_time, stop_distance = self.stop_time_and_distance(speed)
        if elapsed >= stop_time:
            return 0.0, stop_distance
        a0 = max(0.01, self.a0_mps2)
        k = max(0.0, self.k)
        if self.speed_power == 0 or k <= 1e-8:
            remaining = max(0.0, speed - a0 * elapsed)
            distance = speed * elapsed - 0.5 * a0 * elapsed * elapsed
            return remaining, max(0.0, distance)
        if self.speed_power == 1:
            remaining = (speed + a0 / k) * math.exp(-k * elapsed) - a0 / k
            remaining = max(0.0, remaining)
            distance = (
                (speed - remaining) / k
                - a0 / (k * k) * math.log((a0 + k * speed) / (a0 + k * remaining))
            )
            return remaining, max(0.0, distance)
        if self.speed_power == 2:
            angle = math.atan(speed * math.sqrt(k / a0)) - math.sqrt(a0 * k) * elapsed
            remaining = math.sqrt(a0 / k) * math.tan(max(0.0, angle))
            distance = math.log(
                (a0 + k * speed * speed) / (a0 + k * remaining * remaining)
            ) / (2.0 * k)
            return max(0.0, remaining), max(0.0, distance)
        raise ValueError("speed_power must be 0, 1, or 2")


DEFAULT_BALL_RESISTANCE_MODEL = BallResistanceModel(
    name="quadratic_v1_advisory",
    a0_mps2=0.37913599334485903,
    k=0.1399186020210285,
    speed_power=2,
)


@dataclass(frozen=True)
class BallMotionPrediction:
    """Predicted rolling state plus diagnostics and future-position queries."""

    current: Pose2D
    stop: Pose2D
    velocity_x: float
    velocity_y: float
    acceleration_x: float
    acceleration_y: float
    speed_mps: float
    decel_mps2: float
    time_to_stop_sec: float
    confidence: float
    uncertainty_radius_m: float
    fit_residual_m: float
    motion_state: str
    sample_count: int
    window_sec: float
    resistance_model: BallResistanceModel

    def speed_after(self, seconds: float) -> float:
        return self.resistance_model.speed_and_distance_after(self.speed_mps, seconds)[0]

    def position_at(self, seconds: float) -> Pose2D:
        """Predict position after ``seconds``, clamped naturally at the stop point."""

        if self.speed_mps <= DEFAULT_SETTLED_SPEED_MPS:
            return self.current
        _, distance = self.resistance_model.speed_and_distance_after(self.speed_mps, seconds)
        return Pose2D(
            self.current.x + self.velocity_x / self.speed_mps * distance,
            self.current.y + self.velocity_y / self.speed_mps * distance,
            0.0,
        )


# Backward-compatible public name used by the first implementation.
BallStopPrediction = BallMotionPrediction


def predict_ball_motion(
    observations: list[BallObservation],
    *,
    resistance_model: BallResistanceModel | None = None,
    settled_speed_mps: float = DEFAULT_SETTLED_SPEED_MPS,
) -> BallMotionPrediction | None:
    """Estimate endpoint motion and predict the free-roll trajectory.

    A weighted quadratic fit gives recent samples more influence and returns
    velocity at the end of the window, avoiding the old whole-window average
    velocity bias.  The result remains advisory: collisions and future touches
    are outside the free-roll model.
    """

    points = _unique_sorted_observations(observations)
    if len(points) < 2:
        return None
    model = DEFAULT_BALL_RESISTANCE_MODEL if resistance_model is None else resistance_model
    last = points[-1]
    window_sec = max(_MIN_DT, last.t_sec - points[0].t_sec)

    fit_x = _weighted_endpoint_fit(points, "x")
    fit_y = _weighted_endpoint_fit(points, "y")
    if fit_x is None or fit_y is None:
        return None
    vx, ax, residual_x = fit_x
    vy, ay, residual_y = fit_y
    speed = math.hypot(vx, vy)
    residual = math.hypot(residual_x, residual_y)
    current = Pose2D(last.x, last.y, 0.0)

    stability = _motion_stability(vx, vy, ax, ay, residual, speed)
    state = "settled" if speed <= settled_speed_mps else ("rolling" if stability >= 0.45 else "unstable")
    confidence = _confidence(points, window_sec, speed, settled_speed_mps, residual, stability)
    uncertainty = _uncertainty_radius(window_sec, residual, state)

    if speed <= settled_speed_mps:
        return BallMotionPrediction(
            current=current,
            stop=current,
            velocity_x=vx,
            velocity_y=vy,
            acceleration_x=ax,
            acceleration_y=ay,
            speed_mps=speed,
            decel_mps2=model.deceleration(speed),
            time_to_stop_sec=0.0,
            confidence=confidence,
            uncertainty_radius_m=uncertainty,
            fit_residual_m=residual,
            motion_state=state,
            sample_count=len(points),
            window_sec=window_sec,
            resistance_model=model,
        )

    time_to_stop, distance_to_stop = model.stop_time_and_distance(speed)
    stop = Pose2D(
        last.x + vx / speed * distance_to_stop,
        last.y + vy / speed * distance_to_stop,
        0.0,
    )
    return BallMotionPrediction(
        current=current,
        stop=stop,
        velocity_x=vx,
        velocity_y=vy,
        acceleration_x=ax,
        acceleration_y=ay,
        speed_mps=speed,
        decel_mps2=model.deceleration(speed),
        time_to_stop_sec=time_to_stop,
        confidence=confidence,
        uncertainty_radius_m=uncertainty,
        fit_residual_m=residual,
        motion_state=state,
        sample_count=len(points),
        window_sec=window_sec,
        resistance_model=model,
    )


def predict_ball_stop(
    observations: list[BallObservation],
    decel_mps2: float | None = None,
    settled_speed_mps: float = DEFAULT_SETTLED_SPEED_MPS,
    *,
    resistance_model: BallResistanceModel | None = None,
) -> BallStopPrediction | None:
    """Backward-compatible stop prediction entry point.

    Passing ``decel_mps2`` explicitly selects the legacy constant model.  With
    no explicit model, the cross-validated quadratic advisory model is used.
    """

    if decel_mps2 is not None:
        if resistance_model is not None:
            raise ValueError("pass either decel_mps2 or resistance_model, not both")
        resistance_model = BallResistanceModel("constant", max(0.01, decel_mps2))
    return predict_ball_motion(
        observations,
        resistance_model=resistance_model,
        settled_speed_mps=settled_speed_mps,
    )


class SlidingWindowBallPredictor:
    """Maintain recent observations, reject discontinuities, and predict motion."""

    def __init__(
        self,
        window_sec: float = DEFAULT_WINDOW_SEC,
        decel_mps2: float | None = None,
        settled_speed_mps: float = DEFAULT_SETTLED_SPEED_MPS,
        max_samples: int = 30,
        *,
        resistance_model: BallResistanceModel | None = None,
        max_observation_gap_sec: float = DEFAULT_MAX_OBSERVATION_GAP_SEC,
        max_plausible_speed_mps: float = DEFAULT_MAX_PLAUSIBLE_SPEED_MPS,
    ):
        if decel_mps2 is not None and resistance_model is not None:
            raise ValueError("pass either decel_mps2 or resistance_model, not both")
        self.window_sec = window_sec
        self.settled_speed_mps = settled_speed_mps
        self.max_samples = max_samples
        self.max_observation_gap_sec = max_observation_gap_sec
        self.max_plausible_speed_mps = max_plausible_speed_mps
        self.resistance_model = (
            BallResistanceModel("constant", max(0.01, decel_mps2))
            if decel_mps2 is not None
            else resistance_model or DEFAULT_BALL_RESISTANCE_MODEL
        )
        # Compatibility for callers that inspect the former attribute.
        self.decel_mps2 = self.resistance_model.a0_mps2
        self._observations: list[BallObservation] = []

    def reset(self) -> None:
        self._observations.clear()

    def add_observation(self, observation: BallObservation) -> None:
        if self._observations:
            previous = self._observations[-1]
            dt = observation.t_sec - previous.t_sec
            if abs(dt) <= _MIN_DT:
                return
            if dt < 0.0 or dt > self.max_observation_gap_sec:
                self.reset()
            elif math.hypot(observation.x - previous.x, observation.y - previous.y) / dt > self.max_plausible_speed_mps:
                self.reset()
            elif self._direction_reversed(previous, observation):
                # Keep the contact/bounce point as the first sample of the new segment.
                self._observations = [previous]
        self._observations.append(observation)
        self._trim(observation.t_sec)

    def add_ball(self, ball: BallState, now_sec: float | None = None) -> None:
        t_sec = ball.last_seen_at if now_sec is None else now_sec
        self.add_observation(BallObservation(t_sec, ball.x, ball.y, ball.confidence))

    def predict(self) -> BallMotionPrediction | None:
        return predict_ball_motion(
            self._observations,
            resistance_model=self.resistance_model,
            settled_speed_mps=self.settled_speed_mps,
        )

    def observations(self) -> tuple[BallObservation, ...]:
        return tuple(self._observations)

    def _trim(self, now_sec: float) -> None:
        earliest = now_sec - self.window_sec
        self._observations = [
            observation for observation in self._observations if observation.t_sec >= earliest
        ][-self.max_samples :]

    def _direction_reversed(self, previous: BallObservation, current: BallObservation) -> bool:
        if len(self._observations) < 2:
            return False
        before = self._observations[-2]
        old_dx = previous.x - before.x
        old_dy = previous.y - before.y
        new_dx = current.x - previous.x
        new_dy = current.y - previous.y
        old_length = math.hypot(old_dx, old_dy)
        new_length = math.hypot(new_dx, new_dy)
        if old_length < 0.015 or new_length < 0.015:
            return False
        cosine = (old_dx * new_dx + old_dy * new_dy) / (old_length * new_length)
        return cosine < -0.25


def _unique_sorted_observations(observations: list[BallObservation]) -> list[BallObservation]:
    result: list[BallObservation] = []
    for observation in sorted(observations, key=lambda item: item.t_sec):
        if result and observation.t_sec - result[-1].t_sec <= _MIN_DT:
            result[-1] = observation
        else:
            result.append(observation)
    return result


def _weighted_endpoint_fit(
    observations: list[BallObservation], component: str
) -> tuple[float, float, float] | None:
    end_t = observations[-1].t_sec
    offsets = [observation.t_sec - end_t for observation in observations]
    values = [observation.x if component == "x" else observation.y for observation in observations]
    weights = [
        math.exp(offset / 0.12) * max(0.05, min(1.0, observation.confidence))
        for offset, observation in zip(offsets, observations)
    ]
    degree = 2 if len(observations) >= 5 and end_t - observations[0].t_sec >= 0.12 else 1
    coefficients = _weighted_polynomial(offsets, values, weights, degree)
    if coefficients is None:
        return None
    predicted = [sum(coefficient * offset**power for power, coefficient in enumerate(coefficients)) for offset in offsets]
    weight_sum = sum(weights)
    residual = math.sqrt(
        sum(weight * (actual - fitted) ** 2 for actual, fitted, weight in zip(values, predicted, weights))
        / max(1e-9, weight_sum)
    )
    velocity = coefficients[1]
    acceleration = 2.0 * coefficients[2] if degree == 2 else 0.0
    return velocity, acceleration, residual


def _weighted_polynomial(
    xs: list[float], ys: list[float], weights: list[float], degree: int
) -> list[float] | None:
    size = degree + 1
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for x, y, weight in zip(xs, ys, weights):
        basis = [x**power for power in range(size)]
        for row in range(size):
            vector[row] += weight * basis[row] * y
            for column in range(size):
                matrix[row][column] += weight * basis[row] * basis[column]
    return _solve_linear_system(matrix, vector)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _motion_stability(
    vx: float, vy: float, ax: float, ay: float, residual: float, speed: float
) -> float:
    residual_score = math.exp(-residual / 0.08)
    if speed <= DEFAULT_SETTLED_SPEED_MPS:
        return residual_score
    direction_x = vx / speed
    direction_y = vy / speed
    parallel = ax * direction_x + ay * direction_y
    lateral = abs(-ax * direction_y + ay * direction_x)
    lateral_score = math.exp(-lateral / 2.0)
    acceleration_score = 1.0 if parallel <= 1.0 else math.exp(-(parallel - 1.0) / 2.0)
    return max(0.0, min(1.0, residual_score * lateral_score * acceleration_score))


def _confidence(
    observations: list[BallObservation],
    window_sec: float,
    speed_mps: float,
    settled_speed_mps: float,
    residual: float,
    stability: float,
) -> float:
    sample_score = min(1.0, len(observations) / 8.0)
    span_score = min(1.0, window_sec / 0.35)
    observation_score = sum(max(0.0, min(1.0, item.confidence)) for item in observations) / len(observations)
    speed_score = 0.75 if speed_mps <= settled_speed_mps else min(1.0, speed_mps / 0.60)
    fit_score = math.exp(-residual / 0.08)
    return max(
        0.0,
        min(1.0, sample_score * span_score * observation_score * speed_score * fit_score * stability),
    )


def _uncertainty_radius(window_sec: float, residual: float, motion_state: str) -> float:
    # p75 stop-error envelope from held-out dataset/v1 folds, interpolated by
    # observation span.  It is a conservative tactical radius, not a guarantee.
    knots = ((0.15, 1.320), (0.25, 1.252), (0.35, 1.063), (0.50, 0.902))
    if window_sec <= knots[0][0]:
        base = knots[0][1]
    elif window_sec >= knots[-1][0]:
        base = knots[-1][1]
    else:
        base = knots[-1][1]
        for (left_t, left_value), (right_t, right_value) in zip(knots, knots[1:]):
            if left_t <= window_sec <= right_t:
                ratio = (window_sec - left_t) / (right_t - left_t)
                base = left_value + ratio * (right_value - left_value)
                break
    if motion_state == "settled":
        base = min(base, 0.20)
    elif motion_state == "unstable":
        base = max(base, 1.20)
    return min(2.0, max(0.10, base, residual * 3.0))
