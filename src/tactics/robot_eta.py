"""Bounded robot-arrival interface with dynamic baseline/team2 selection.

Both ridge models are frozen Python constants.  The selector keeps the better
validated baseline for ordinary direct travel and uses the broader team2 model
for difficult, in-domain queries such as detours, low speed caps, and large
turns.  Estimates always expose disagreement and
held-out error as uncertainty; callers must retain a geometric fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..soccer_framework import Pose2D
from .ball_trajectory import BallTrajectoryPrediction
from .geometry import normalize_angle
from .robot_eta_model_data import BASELINE_MODEL, TEAM2_MODEL


TRANSLATION_COMMAND_GAIN = 0.8945
YAW_COMMAND_GAIN = 0.8789
_MIN_ETA_SEC = 0.05
_DOMAIN_FEATURES = (0, 1, 3, 5, 7, 8, 9, 10, 15, 17, 18, 21, 22)


@dataclass(frozen=True)
class RobotMotionEstimate:
    linear_speed_mps: float
    yaw_rate_radps: float
    sample_count: int
    window_sec: float


class RobotMotionTracker:
    """Small public-pose tracker used to populate optional ETA speed inputs."""

    def __init__(self, window_sec: float = 0.35, max_samples: int = 12):
        self.window_sec = max(0.10, window_sec)
        self.max_samples = max(2, max_samples)
        self._history: dict[int, list[tuple[float, Pose2D]]] = {}

    def reset(self, robot_id: int | None = None) -> None:
        if robot_id is None:
            self._history.clear()
        else:
            self._history.pop(robot_id, None)

    def update(self, robot_id: int, pose: Pose2D, observed_at_sec: float) -> None:
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.theta, observed_at_sec)):
            self.reset(robot_id)
            return
        rows = self._history.setdefault(robot_id, [])
        if rows and observed_at_sec <= rows[-1][0] + 1e-5:
            return
        if rows and observed_at_sec - rows[-1][0] > max(0.5, self.window_sec * 2.0):
            rows.clear()
        rows.append((observed_at_sec, pose))
        earliest = observed_at_sec - self.window_sec
        self._history[robot_id] = [row for row in rows if row[0] >= earliest][
            -self.max_samples :
        ]

    def motion(self, robot_id: int) -> RobotMotionEstimate | None:
        rows = self._history.get(robot_id) or []
        if len(rows) < 2:
            return None
        start_t, start = rows[0]
        end_t, end = rows[-1]
        elapsed = end_t - start_t
        if elapsed <= 1e-4:
            return None
        return RobotMotionEstimate(
            linear_speed_mps=math.hypot(end.x - start.x, end.y - start.y) / elapsed,
            yaw_rate_radps=abs(normalize_angle(end.theta - start.theta)) / elapsed,
            sample_count=len(rows),
            window_sec=elapsed,
        )


@dataclass(frozen=True)
class RobotArrivalQuery:
    now_sec: float
    robot_id: int
    pose: Pose2D
    raw_target: Pose2D
    adjusted_target: Pose2D | None = None
    planned_path_length_m: float | None = None
    observed_linear_speed_mps: float | None = None
    observed_yaw_rate_radps: float | None = None
    arrive_distance_m: float = 0.15
    arrive_angle_rad: float = 0.20
    linear_speed_limit_mps: float = 0.80
    command_speed_mps: float | None = None
    command_yaw_rate_radps: float | None = None
    runtime_mode: str | None = "walk"
    fall_state: str | None = "normal"
    kick_active: bool = False
    target_age_sec: float = 0.0
    target_changed: bool = True
    motion_phase: str | None = None


@dataclass(frozen=True)
class RobotArrivalEstimate:
    robot_id: int
    eta_sec: float | None
    earliest_sec: float | None
    latest_sec: float | None
    uncertainty_sec: float
    confidence: float
    reachable: bool
    invalid_reason: str | None
    model_name: str | None
    alternate_model_name: str | None
    alternate_eta_sec: float | None
    selection_reason: str
    turn_time_sec: float | None
    startup_time_sec: float | None
    travel_time_sec: float | None
    braking_time_sec: float | None
    final_align_time_sec: float | None
    path_length_m: float
    adjusted_target: Pose2D
    motion_phase: str


@dataclass(frozen=True)
class InterceptEstimate:
    robot_id: int
    intercept_point: Pose2D
    intercept_time_sec: float | None
    robot_eta_sec: float | None
    arrival_margin_sec: float | None
    confidence: float
    fallback_point: Pose2D
    reason: str
    arrival: RobotArrivalEstimate | None


class DynamicRobotArrivalEstimator:
    """Select one of two frozen ETA regressors from observable query context."""

    def estimate(self, query: RobotArrivalQuery) -> RobotArrivalEstimate:
        target = query.adjusted_target or query.raw_target
        straight = math.hypot(query.raw_target.x - query.pose.x, query.raw_target.y - query.pose.y)
        path_length = (
            straight
            if query.planned_path_length_m is None
            else query.planned_path_length_m
        )
        invalid = self._invalid_reason(query, target, path_length)
        if invalid is not None:
            return RobotArrivalEstimate(
                robot_id=query.robot_id,
                eta_sec=None,
                earliest_sec=None,
                latest_sec=None,
                uncertainty_sec=math.inf,
                confidence=0.0,
                reachable=False,
                invalid_reason=invalid,
                model_name=None,
                alternate_model_name=None,
                alternate_eta_sec=None,
                selection_reason="unreachable query",
                turn_time_sec=None,
                startup_time_sec=None,
                travel_time_sec=None,
                braking_time_sec=None,
                final_align_time_sec=None,
                path_length_m=max(0.0, path_length) if math.isfinite(path_length) else 0.0,
                adjusted_target=target,
                motion_phase="unavailable",
            )

        features, phase, heading_error, final_error = _features(query, target, path_length)
        baseline_eta = _predict(BASELINE_MODEL, features)
        team2_eta = _predict(TEAM2_MODEL, features)
        baseline_domain = _domain_distance(BASELINE_MODEL, features)
        team2_domain = _domain_distance(TEAM2_MODEL, features)
        detour = target != query.raw_target or path_length > straight + 0.10
        difficult = detour or query.linear_speed_limit_mps <= 0.65 or heading_error >= 0.75
        team2_in_domain = team2_domain <= 4.0 and team2_domain + 0.20 <= baseline_domain
        if difficult and team2_in_domain:
            selected, alternate = TEAM2_MODEL, BASELINE_MODEL
            eta, alternate_eta = team2_eta, baseline_eta
            selected_domain = team2_domain
            reason = "team2 model for difficult in-domain motion"
        else:
            selected, alternate = BASELINE_MODEL, TEAM2_MODEL
            eta, alternate_eta = baseline_eta, team2_eta
            selected_domain = baseline_domain
            reason = (
                "baseline model for ordinary motion"
                if not difficult
                else "baseline fallback outside team2 domain"
            )

        at_start = query.target_changed or query.target_age_sec < 0.50
        cap = max(0.10, query.linear_speed_limit_mps)
        turn_time = heading_error / YAW_COMMAND_GAIN
        travel_time = path_length / max(0.05, TRANSLATION_COMMAND_GAIN * cap)
        final_align = final_error / YAW_COMMAND_GAIN
        physical_lower = max(
            turn_time,
            max(0.0, path_length - query.arrive_distance_m)
            / max(0.05, TRANSLATION_COMMAND_GAIN * cap),
        )
        eta, alternate_eta = max(eta, physical_lower), max(alternate_eta, physical_lower)
        p75_key = "start_p75_sec" if at_start else "query_p75_sec"
        p90_key = "start_p90_sec" if at_start else "query_p90_sec"
        disagreement = abs(eta - alternate_eta)
        uncertainty = max(float(selected[p75_key]), disagreement)
        if query.observed_linear_speed_mps is None:
            uncertainty += 0.15
        if query.adjusted_target is None or query.planned_path_length_m is None:
            uncertainty += 0.12
        uncertainty *= 1.0 + 0.08 * max(0.0, selected_domain - 1.0)
        latest_radius = max(float(selected[p90_key]), disagreement, uncertainty)
        confidence = math.exp(-uncertainty / max(0.5, eta + 0.5)) * math.exp(
            -0.12 * max(0.0, selected_domain - 1.0)
        )

        return RobotArrivalEstimate(
            robot_id=query.robot_id,
            eta_sec=eta,
            earliest_sec=max(_MIN_ETA_SEC, eta - uncertainty),
            latest_sec=eta + latest_radius,
            uncertainty_sec=uncertainty,
            confidence=max(0.0, min(1.0, confidence)),
            reachable=True,
            invalid_reason=None,
            model_name=str(selected["name"]),
            alternate_model_name=str(alternate["name"]),
            alternate_eta_sec=alternate_eta,
            selection_reason=reason,
            turn_time_sec=turn_time,
            startup_time_sec=None,
            travel_time_sec=travel_time,
            braking_time_sec=None,
            final_align_time_sec=final_align,
            path_length_m=path_length,
            adjusted_target=target,
            motion_phase=phase,
        )

    @staticmethod
    def _invalid_reason(
        query: RobotArrivalQuery, target: Pose2D, path_length: float
    ) -> str | None:
        values = (
            query.now_sec,
            query.pose.x,
            query.pose.y,
            query.pose.theta,
            target.x,
            target.y,
            target.theta,
            path_length,
            query.linear_speed_limit_mps,
        )
        if not all(math.isfinite(value) for value in values):
            return "non_finite_input"
        if path_length < 0.0 or query.linear_speed_limit_mps <= 0.0:
            return "invalid_path_or_speed_limit"
        if query.kick_active:
            return "robot_kicking"
        if query.fall_state not in {None, "normal"}:
            return "robot_not_upright"
        if query.runtime_mode in {"damping", "get_up", "recovery", "kicking"}:
            return "runtime_mode_not_walkable"
        return None


def estimate_earliest_intercept(
    estimator: DynamicRobotArrivalEstimator,
    base_query: RobotArrivalQuery,
    trajectory: BallTrajectoryPrediction,
    *,
    candidate_times_sec: tuple[float, ...] = (0.10, 0.25, 0.50),
    safety_margin_sec: float = 0.25,
) -> InterceptEstimate:
    """Search a fixed set of ball times; never performs an unbounded solve."""

    fallback = trajectory.current
    if not trajectory.usable:
        return InterceptEstimate(
            base_query.robot_id, fallback, None, None, None, 0.0, fallback,
            trajectory.invalid_reason or "ball path unavailable", None,
        )
    best: tuple[float, Pose2D, RobotArrivalEstimate] | None = None
    prediction_age = max(0.0, base_query.now_sec - trajectory.observed_at_sec)
    for seconds_ahead in candidate_times_sec:
        relative = prediction_age + seconds_ahead
        point = trajectory.position_at(relative)
        if point is None:
            continue
        oriented_point = Pose2D(
            point.x,
            point.y,
            math.atan2(point.y - base_query.pose.y, point.x - base_query.pose.x),
        )
        query = replace(
            base_query,
            raw_target=oriented_point,
            adjusted_target=None,
            planned_path_length_m=None,
        )
        arrival = estimator.estimate(query)
        if not arrival.reachable or arrival.eta_sec is None:
            continue
        margin = seconds_ahead - arrival.eta_sec
        if best is None or margin > best[0]:
            best = (margin, point, arrival)
        if arrival.eta_sec <= seconds_ahead + safety_margin_sec:
            confidence = trajectory.confidence_at(relative) * arrival.confidence
            return InterceptEstimate(
                base_query.robot_id,
                point,
                seconds_ahead,
                arrival.eta_sec,
                margin,
                confidence,
                fallback,
                "earliest bounded reachable ball point",
                arrival,
            )
    if best is None:
        return InterceptEstimate(
            base_query.robot_id, fallback, None, None, None, 0.0, fallback,
            "no valid ball-path query", None,
        )
    margin, point, arrival = best
    return InterceptEstimate(
        base_query.robot_id,
        point,
        None,
        arrival.eta_sec,
        margin,
        0.0,
        fallback,
        "no confirmed intercept within calibrated horizon",
        arrival,
    )


def _features(
    query: RobotArrivalQuery, target: Pose2D, path_length: float
) -> tuple[tuple[float, ...], str, float, float]:
    dx = target.x - query.pose.x
    dy = target.y - query.pose.y
    desired = query.pose.theta if path_length <= 1e-6 else math.atan2(dy, dx)
    heading_signed = normalize_angle(desired - query.pose.theta)
    final_signed = normalize_angle(query.raw_target.theta - query.pose.theta)
    heading = abs(heading_signed)
    final = abs(final_signed)
    relative = abs(normalize_angle(final_signed - heading_signed))
    cap = max(0.10, query.linear_speed_limit_mps)
    speed = (
        0.0
        if query.observed_linear_speed_mps is None
        else max(0.0, query.observed_linear_speed_mps)
    )
    yaw = 0.0 if query.observed_yaw_rate_radps is None else abs(query.observed_yaw_rate_radps)
    phase = query.motion_phase or ("turn" if heading >= 0.50 else "run")
    control_distance = math.hypot(dx, dy)
    avoidance = target != query.raw_target or path_length > math.hypot(
        query.raw_target.x - query.pose.x, query.raw_target.y - query.pose.y
    ) + 0.10
    command_speed = query.command_speed_mps
    if command_speed is None:
        command_speed = 0.0 if phase == "turn" else cap
    command_yaw = query.command_yaw_rate_radps
    if command_yaw is None:
        command_yaw = min(1.0, heading) if phase == "turn" else min(0.30, heading)
    age = max(0.0, query.target_age_sec)
    features = (
        path_length,
        path_length / cap,
        math.sqrt(max(0.0, path_length)) / cap,
        heading,
        heading * heading,
        final,
        path_length * heading,
        cap,
        speed,
        speed / cap,
        yaw,
        1.0 if phase == "turn" else 0.0,
        1.0 if phase == "run" else 0.0,
        max(0.01, query.arrive_distance_m),
        1.0 if query.observed_linear_speed_mps is None else 0.0,
        relative,
        relative * relative,
        control_distance,
        1.0 if avoidance else 0.0,
        abs(command_speed),
        abs(command_yaw),
        age,
        min(1.0, age),
    )
    return features, phase, heading, final


def _predict(model: dict[str, object], features: tuple[float, ...]) -> float:
    means = model["means"]
    scales = model["scales"]
    coefficients = model["coefficients"]
    assert isinstance(means, tuple) and isinstance(scales, tuple)
    assert isinstance(coefficients, tuple)
    scaled = tuple((value - mean) / scale for value, mean, scale in zip(features, means, scales))
    value = coefficients[0] + sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients[1:], scaled)
    )
    return max(_MIN_ETA_SEC, value)


def _domain_distance(model: dict[str, object], features: tuple[float, ...]) -> float:
    means = model["means"]
    scales = model["scales"]
    assert isinstance(means, tuple) and isinstance(scales, tuple)
    values = [
        abs(features[index] - means[index]) / max(0.05, scales[index])
        for index in _DOMAIN_FEATURES
    ]
    return math.sqrt(sum(value * value for value in values) / len(values))


__all__ = [
    "DynamicRobotArrivalEstimator",
    "InterceptEstimate",
    "RobotArrivalEstimate",
    "RobotArrivalQuery",
    "RobotMotionEstimate",
    "RobotMotionTracker",
    "estimate_earliest_intercept",
]

