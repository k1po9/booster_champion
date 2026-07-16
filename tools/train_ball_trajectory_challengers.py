#!/usr/bin/env python3
"""Train and leave-one-match-out validate bounded rolling-ball challengers.

This script deliberately keeps inference implementations standard-library only.
It reuses the conservative free-roll extraction from
``cross_validate_ball_motion`` and compares the production-style quadratic
baseline with robust endpoint estimation, direction stabilisation, per-roll
drag adaptation, a monotone piecewise drag table, and a tiny ensemble.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cross_validate_ball_motion import (
    MAX_SPEED_MPS,
    MIN_SPEED_MPS,
    Point,
    ResistanceModel,
    Track,
    _endpoint_component,
    _solve_3x3,
    deceleration_observations,
    fit_models,
    load_tracks,
    percentile,
    stop_time_and_distance,
    summarize,
)


OBSERVATION_HORIZONS = (0.15, 0.25, 0.35, 0.50)
FUTURE_DELTAS = (0.10, 0.25, 0.50)
CANDIDATES = (
    "current_quadratic",
    "robust_quadratic",
    "stable_quadratic",
    "adaptive_quadratic",
    "piecewise_stable",
    "adaptive_piecewise_ensemble",
)
PIECEWISE_KNOTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
TABLE_STEP_MPS = 0.02


@dataclass(frozen=True)
class EndpointState:
    point: Point
    vx: float
    vy: float
    ax: float
    ay: float
    residual_m: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class PiecewiseResistance:
    knots_mps: tuple[float, ...]
    decel_mps2: tuple[float, ...]
    fit_samples: int

    def deceleration(self, speed: float) -> float:
        value = max(0.0, min(self.knots_mps[-1], speed))
        upper = bisect.bisect_right(self.knots_mps, value)
        if upper <= 0:
            return self.decel_mps2[0]
        if upper >= len(self.knots_mps):
            return self.decel_mps2[-1]
        lower = upper - 1
        span = self.knots_mps[upper] - self.knots_mps[lower]
        weight = 0.0 if span <= 1e-9 else (value - self.knots_mps[lower]) / span
        return max(
            0.01,
            self.decel_mps2[lower] * (1.0 - weight)
            + self.decel_mps2[upper] * weight,
        )


class PiecewiseLookup:
    """Cumulative time/distance integrals for constant-time online queries."""

    def __init__(self, model: PiecewiseResistance):
        self.model = model
        count = int(math.ceil(model.knots_mps[-1] / TABLE_STEP_MPS)) + 1
        self.speeds = [index * TABLE_STEP_MPS for index in range(count)]
        self.times = [0.0]
        self.distances = [0.0]
        for before, after in zip(self.speeds, self.speeds[1:]):
            a_before = model.deceleration(before)
            a_after = model.deceleration(after)
            dv = after - before
            self.times.append(
                self.times[-1] + 0.5 * (1.0 / a_before + 1.0 / a_after) * dv
            )
            self.distances.append(
                self.distances[-1]
                + 0.5 * (before / a_before + after / a_after) * dv
            )

    def cumulative(self, speed: float) -> tuple[float, float]:
        value = max(0.0, min(self.speeds[-1], speed))
        index = min(len(self.speeds) - 2, int(value / TABLE_STEP_MPS))
        before = self.speeds[index]
        weight = (value - before) / TABLE_STEP_MPS
        return (
            self.times[index] * (1.0 - weight) + self.times[index + 1] * weight,
            self.distances[index] * (1.0 - weight)
            + self.distances[index + 1] * weight,
        )

    def distance_after(self, speed: float, seconds: float | None) -> float:
        total_time, total_distance = self.cumulative(speed)
        if seconds is None or seconds >= total_time:
            return total_distance
        remaining_time = total_time - max(0.0, seconds)
        upper = bisect.bisect_left(self.times, remaining_time)
        if upper <= 0:
            remaining_speed = 0.0
        elif upper >= len(self.times):
            remaining_speed = self.speeds[-1]
        else:
            lower = upper - 1
            span = self.times[upper] - self.times[lower]
            weight = 0.0 if span <= 1e-12 else (remaining_time - self.times[lower]) / span
            remaining_speed = (
                self.speeds[lower] * (1.0 - weight) + self.speeds[upper] * weight
            )
        _, remaining_distance = self.cumulative(remaining_speed)
        return max(0.0, total_distance - remaining_distance)


def _weighted_median(rows: list[tuple[float, float]]) -> float | None:
    if not rows:
        return None
    ordered = sorted(rows)
    half = sum(weight for _, weight in ordered) * 0.5
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= half:
            return value
    return ordered[-1][0]


def _isotonic_non_decreasing(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([value, max(1e-6, weight), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[1] + right[1]
            blocks.append(
                [
                    (left[0] * left[1] + right[0] * right[1]) / total_weight,
                    total_weight,
                    left[2] + right[2],
                ]
            )
    result: list[float] = []
    for value, _, length in blocks:
        result.extend([value] * int(length))
    return result


def fit_piecewise(tracks: list[Track], fallback: ResistanceModel) -> PiecewiseResistance:
    observations = deceleration_observations(tracks)
    values: list[float] = []
    weights: list[float] = []
    for knot in PIECEWISE_KNOTS:
        rows: list[tuple[float, float]] = []
        for speed, decel in observations:
            distance = abs(speed - knot)
            if distance <= 0.65:
                rows.append((decel, math.exp(-0.5 * (distance / 0.32) ** 2)))
        local = _weighted_median(rows)
        prior = fallback.a0_mps2 + fallback.k * knot * knot
        evidence = sum(weight for _, weight in rows)
        blend = evidence / (evidence + 25.0)
        values.append(max(0.05, min(12.0, prior if local is None else prior * (1.0 - blend) + local * blend)))
        weights.append(max(1.0, evidence))
    values = _isotonic_non_decreasing(values, weights)
    return PiecewiseResistance(PIECEWISE_KNOTS, tuple(values), len(observations))


def _component_fit(
    points: list[Point], end_t: float, component: str, *, robust: bool
) -> tuple[float, float, float] | None:
    rows: list[tuple[float, float, float]] = []
    for point in points:
        offset = point.t - end_t
        if -0.28 <= offset <= 1e-6:
            value = point.x if component == "x" else point.y
            rows.append((offset, value, math.exp(offset / 0.12)))
    if len(rows) < 4:
        return None

    weights = [row[2] for row in rows]
    solution: list[float] | None = None
    for iteration in range(4 if robust else 1):
        matrix = [[0.0] * 3 for _ in range(3)]
        vector = [0.0] * 3
        for (offset, value, _), weight in zip(rows, weights):
            basis = (1.0, offset, offset * offset)
            for row in range(3):
                vector[row] += weight * basis[row] * value
                for column in range(3):
                    matrix[row][column] += weight * basis[row] * basis[column]
        solution = _solve_3x3(matrix, vector)
        if solution is None:
            return None
        if not robust or iteration == 3:
            break
        residuals = [
            value - (solution[0] + solution[1] * offset + solution[2] * offset * offset)
            for offset, value, _ in rows
        ]
        scale = max(0.005, 1.4826 * statistics.median(abs(value) for value in residuals))
        cutoff = 1.5 * scale
        weights = [
            base * (1.0 if abs(residual) <= cutoff else cutoff / abs(residual))
            for (_, _, base), residual in zip(rows, residuals)
        ]
    assert solution is not None
    residual = math.sqrt(
        sum(
            weight
            * (value - (solution[0] + solution[1] * offset + solution[2] * offset * offset)) ** 2
            for (offset, value, _), weight in zip(rows, weights)
        )
        / max(1e-9, sum(weights))
    )
    return solution[1], 2.0 * solution[2], residual


def estimate_state(track: Track, observation_horizon: float, kind: str) -> EndpointState | None:
    end_t = track.points[0].t + observation_horizon
    prefix = [point for point in track.points if point.t <= end_t + 1e-6]
    if len(prefix) < 4:
        return None
    last = prefix[-1]
    if kind == "current_quadratic":
        vx = _endpoint_component(prefix, last.t, "x")
        vy = _endpoint_component(prefix, last.t, "y")
        if vx is None or vy is None:
            return None
        return EndpointState(last, vx, vy, 0.0, 0.0, 0.0)

    fit_x = _component_fit(prefix, last.t, "x", robust=True)
    fit_y = _component_fit(prefix, last.t, "y", robust=True)
    if fit_x is None or fit_y is None:
        return None
    vx, ax, residual_x = fit_x
    vy, ay, residual_y = fit_y
    speed = math.hypot(vx, vy)
    if speed <= 1e-9:
        return None

    if kind not in {"robust_quadratic"}:
        dx = last.x - prefix[0].x
        dy = last.y - prefix[0].y
        displacement = math.hypot(dx, dy)
        if displacement > 1e-6:
            endpoint_x = vx / speed
            endpoint_y = vy / speed
            track_x = dx / displacement
            track_y = dy / displacement
            # Early endpoint derivatives are noisy; gradually hand authority back
            # to the endpoint estimate as the observation span grows.
            track_weight = max(0.12, min(0.38, 0.48 - 0.65 * observation_horizon))
            direction_x = endpoint_x * (1.0 - track_weight) + track_x * track_weight
            direction_y = endpoint_y * (1.0 - track_weight) + track_y * track_weight
            length = math.hypot(direction_x, direction_y)
            if length > 1e-9:
                vx = speed * direction_x / length
                vy = speed * direction_y / length
    return EndpointState(
        last,
        vx,
        vy,
        ax,
        ay,
        math.hypot(residual_x, residual_y),
    )


def _quadratic_distance_after(
    model: ResistanceModel, speed: float, seconds: float | None
) -> float:
    stop_time, stop_distance = stop_time_and_distance(model, speed)
    if seconds is None or seconds >= stop_time:
        return stop_distance
    elapsed = max(0.0, seconds)
    a0 = max(0.01, model.a0_mps2)
    k = max(0.0, model.k)
    if k <= 1e-8:
        return max(0.0, speed * elapsed - 0.5 * a0 * elapsed * elapsed)
    angle = math.atan(speed * math.sqrt(k / a0)) - math.sqrt(a0 * k) * elapsed
    remaining = math.sqrt(a0 / k) * math.tan(max(0.0, angle))
    remaining_distance = math.log1p(k * remaining * remaining / a0) / (2.0 * k)
    return max(0.0, stop_distance - remaining_distance)


def _adaptive_model(
    model: ResistanceModel, state: EndpointState, observation_horizon: float
) -> ResistanceModel:
    speed = state.speed
    global_decel = model.a0_mps2 + model.k * speed * speed
    local_decel = -(state.vx * state.ax + state.vy * state.ay) / max(1e-9, speed)
    if not (0.05 <= local_decel <= 12.0):
        return model
    span_weight = max(0.0, min(1.0, (observation_horizon - 0.12) / 0.38))
    fit_weight = math.exp(-state.residual_m / 0.08)
    ratio = max(0.55, min(1.75, local_decel / max(0.05, global_decel)))
    scale = 1.0 + 0.55 * span_weight * fit_weight * (ratio - 1.0)
    return ResistanceModel(
        "adaptive_quadratic",
        model.a0_mps2 * scale,
        model.k * scale,
        2,
        model.fit_samples,
    )


def predict(
    track: Track,
    quadratic: ResistanceModel,
    piecewise: PiecewiseLookup,
    observation_horizon: float,
    future_delta: float | None,
    kind: str,
) -> tuple[float, float] | None:
    state_kind = kind if kind in {"current_quadratic", "robust_quadratic"} else "stable_quadratic"
    state = estimate_state(track, observation_horizon, state_kind)
    if state is None or not (MIN_SPEED_MPS <= state.speed <= MAX_SPEED_MPS):
        return None

    if kind == "piecewise_stable":
        distance = piecewise.distance_after(state.speed, future_delta)
    elif kind == "adaptive_quadratic":
        distance = _quadratic_distance_after(
            _adaptive_model(quadratic, state, observation_horizon), state.speed, future_delta
        )
    elif kind == "adaptive_piecewise_ensemble":
        adaptive_distance = _quadratic_distance_after(
            _adaptive_model(quadratic, state, observation_horizon), state.speed, future_delta
        )
        piecewise_distance = piecewise.distance_after(state.speed, future_delta)
        distance = 0.5 * (adaptive_distance + piecewise_distance)
    else:
        distance = _quadratic_distance_after(quadratic, state.speed, future_delta)
    return (
        state.point.x + state.vx / state.speed * distance,
        state.point.y + state.vy / state.speed * distance,
    )


def actual_position(track: Track, target_t: float) -> tuple[float, float]:
    if target_t <= track.points[0].t:
        return track.points[0].x, track.points[0].y
    for before, after in zip(track.points, track.points[1:]):
        if target_t <= after.t:
            span = after.t - before.t
            weight = 0.0 if span <= 1e-9 else (target_t - before.t) / span
            return (
                before.x * (1.0 - weight) + after.x * weight,
                before.y * (1.0 - weight) + after.y * weight,
            )
    return track.points[-1].x, track.points[-1].y


def _quadratic(models: list[ResistanceModel]) -> ResistanceModel:
    return next(model for model in models if model.name == "quadratic")


def cross_validate(tracks: list[Track]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = sorted({track.dataset for track in tracks})
    stop_errors = {
        kind: {str(horizon): [] for horizon in OBSERVATION_HORIZONS}
        for kind in CANDIDATES
    }
    path_errors = {
        kind: {
            f"{observation}+{future}": []
            for observation in OBSERVATION_HORIZONS
            for future in FUTURE_DELTAS
        }
        for kind in CANDIDATES
    }
    folds: list[dict[str, Any]] = []

    for held_out in datasets:
        train = [track for track in tracks if track.dataset != held_out]
        validation = [track for track in tracks if track.dataset == held_out]
        quadratic = _quadratic(fit_models(train))
        piecewise_model = fit_piecewise(train, quadratic)
        lookup = PiecewiseLookup(piecewise_model)
        fold_stop = {kind: [] for kind in CANDIDATES}
        for track in validation:
            actual_stop = (track.points[-1].x, track.points[-1].y)
            for horizon in OBSERVATION_HORIZONS:
                observation_t = track.points[0].t + horizon
                for kind in CANDIDATES:
                    predicted_stop = predict(track, quadratic, lookup, horizon, None, kind)
                    if predicted_stop is not None:
                        error = math.hypot(
                            predicted_stop[0] - actual_stop[0],
                            predicted_stop[1] - actual_stop[1],
                        )
                        stop_errors[kind][str(horizon)].append(error)
                        if horizon == 0.35:
                            fold_stop[kind].append(error)
                    for future in FUTURE_DELTAS:
                        predicted_path = predict(track, quadratic, lookup, horizon, future, kind)
                        if predicted_path is None:
                            continue
                        actual = actual_position(track, observation_t + future)
                        path_errors[kind][f"{horizon}+{future}"].append(
                            math.hypot(predicted_path[0] - actual[0], predicted_path[1] - actual[1])
                        )
        folds.append(
            {
                "held_out": held_out,
                "train_tracks": len(train),
                "validation_tracks": len(validation),
                "quadratic": asdict(quadratic),
                "piecewise": asdict(piecewise_model),
                "stop_error_0.35_m": {
                    kind: summarize(values) for kind, values in fold_stop.items()
                },
            }
        )

    aggregate = {
        "stop_error_m": {
            kind: {horizon: summarize(values) for horizon, values in rows.items()}
            for kind, rows in stop_errors.items()
        },
        "future_position_error_m": {
            kind: {key: summarize(values) for key, values in rows.items()}
            for kind, rows in path_errors.items()
        },
    }
    return folds, aggregate


def selection_score(aggregate: dict[str, Any], kind: str) -> float:
    stop = aggregate["stop_error_m"][kind]["0.35"]
    path_a = aggregate["future_position_error_m"][kind]["0.25+0.25"]
    path_b = aggregate["future_position_error_m"][kind]["0.35+0.25"]
    if stop["median"] is None or path_a["median"] is None or path_b["median"] is None:
        return math.inf
    return (
        float(stop["median"])
        + 0.5 * float(stop["p75"])
        + float(path_a["median"])
        + float(path_b["median"])
    )


def benchmark(
    tracks: list[Track], quadratic: ResistanceModel, lookup: PiecewiseLookup
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    samples = tracks[: min(24, len(tracks))]
    for kind in CANDIDATES:
        durations: list[float] = []
        for _ in range(80):
            for track in samples:
                started = time.perf_counter_ns()
                predict(track, quadratic, lookup, 0.35, 0.25, kind)
                durations.append((time.perf_counter_ns() - started) / 1000.0)
        result[kind] = {
            "median_us": float(percentile(durations, 0.50) or 0.0),
            "p99_us": float(percentile(durations, 0.99) or 0.0),
            "max_us": max(durations, default=0.0),
        }
    return result


def _fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ball Trajectory Challenger Training",
        "",
        "All metrics are leave-one-match-file-out. Raw datasets are read-only.",
        "",
        f"- Clean free-roll tracks: {report['clean_track_count']}",
        f"- Selected challenger: **{report['selected_model']}**",
        f"- Selected score: {report['selection_scores'][report['selected_model']]:.3f}",
        "",
        "## Held-Out Stop-Point Error",
        "",
        "| Model | Observation | n | Median | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in CANDIDATES:
        for horizon, stats in report["aggregate"]["stop_error_m"][kind].items():
            lines.append(
                f"| {kind} | {horizon}s | {stats['count']} | {_fmt(stats['median'])}m | "
                f"{_fmt(stats['p75'])}m | {_fmt(stats['p90'])}m |"
            )
    lines.extend(
        [
            "",
            "## Held-Out Future-Position Error",
            "",
            "The label `0.25+0.50` means 0.25s of observation followed by a 0.50s prediction.",
            "",
            "| Model | Observation+Future | n | Median | p75 | p90 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind in CANDIDATES:
        for key, stats in report["aggregate"]["future_position_error_m"][kind].items():
            lines.append(
                f"| {kind} | {key}s | {stats['count']} | {_fmt(stats['median'])}m | "
                f"{_fmt(stats['p75'])}m | {_fmt(stats['p90'])}m |"
            )
    lines.extend(
        [
            "",
            "## Inference Microbenchmark",
            "",
            "| Model | Median | p99 | Max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for kind, stats in report["benchmark"].items():
        lines.append(
            f"| {kind} | {stats['median_us']:.1f}us | {stats['p99_us']:.1f}us | {stats['max_us']:.1f}us |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            "The score combines 0.35s stop median/p75 with two 0.25s-ahead path medians.",
            "Selection is evidence for the next experiment, not authorization for strategy takeover.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/v1")
    parser.add_argument("--output", default="analysis/v2")
    args = parser.parse_args()

    paths = sorted(Path(args.input).glob("*.jsonl"))
    if len(paths) < 2:
        raise SystemExit("at least two dataset files are required")
    tracks, counts = load_tracks(paths)
    if len({track.dataset for track in tracks}) < 2:
        raise SystemExit("not enough datasets contain clean free-roll tracks")

    folds, aggregate = cross_validate(tracks)
    scores = {kind: selection_score(aggregate, kind) for kind in CANDIDATES}
    selected = min(CANDIDATES, key=lambda kind: (scores[kind], kind))
    final_quadratic = _quadratic(fit_models(tracks))
    final_piecewise = fit_piecewise(tracks, final_quadratic)
    report = {
        "inputs": [str(path) for path in paths],
        "filter_counts": counts,
        "clean_track_count": len(tracks),
        "observation_horizons_sec": list(OBSERVATION_HORIZONS),
        "future_deltas_sec": list(FUTURE_DELTAS),
        "folds": folds,
        "aggregate": aggregate,
        "selection_scores": scores,
        "selected_model": selected,
        "final_quadratic": asdict(final_quadratic),
        "final_piecewise": asdict(final_piecewise),
        "benchmark": benchmark(tracks, final_quadratic, PiecewiseLookup(final_piecewise)),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "ball_trajectory_challengers.json"
    markdown_path = output / "ball_trajectory_challengers.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Selected {selected} score={scores[selected]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
