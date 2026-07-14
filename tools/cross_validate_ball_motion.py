#!/usr/bin/env python3
"""Cross-validate speed-dependent rolling-ball resistance models.

The tool reads immutable match JSONL files, extracts conservative free-roll
tracks, fits resistance from training files, and evaluates held-out files.
No third-party packages are required.

Models:

    constant:  dv/dt = -a0
    linear:    dv/dt = -(a0 + k*v)
    quadratic: dv/dt = -(a0 + k*v^2)

Validation is leave-one-dataset-file-out so samples from one simulated match
cannot appear in both training and validation for the same fold.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MIN_TRACK_DISTANCE_M = 0.40
MIN_STRAIGHTNESS = 0.90
MIN_SPEED_MPS = 0.10
MAX_SPEED_MPS = 6.0
MAX_DECEL_MPS2 = 15.0
VELOCITY_RADIUS_SEC = 0.14
ACCEL_PAIR_MIN_SEC = 0.10
ACCEL_PAIR_MAX_SEC = 0.28
DEFAULT_HORIZONS_SEC = (0.15, 0.25, 0.35, 0.50, 1.00)
MIN_RELATIVE_IMPROVEMENT = 0.10
MAX_MEDIAN_ERROR_035_M = 0.60
MAX_P75_ERROR_050_M = 0.90


@dataclass(frozen=True)
class Point:
    t: float
    x: float
    y: float


@dataclass(frozen=True)
class Track:
    dataset: str
    kick_id: int | None
    points: tuple[Point, ...]
    direction_x: float
    direction_y: float
    straightness: float
    distance_m: float


@dataclass(frozen=True)
class ResistanceModel:
    name: str
    a0_mps2: float
    k: float
    feature_power: int
    fit_samples: int


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _raw_points(record: dict[str, Any]) -> list[Point]:
    result: list[Point] = []
    last_t = -math.inf
    for item in record.get("trajectory") or []:
        if item.get("t_sec") is None or item.get("x") is None or item.get("y") is None:
            continue
        point = Point(float(item["t_sec"]), float(item["x"]), float(item["y"]))
        if point.t <= last_t + 1e-6:
            continue
        result.append(point)
        last_t = point.t
    return result


def _path_length(points: list[Point]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def _movement_start(points: list[Point]) -> int | None:
    """Find the first sustained displacement and skip the pre-kick static phase."""

    for index in range(len(points) - 3):
        first = points[index]
        later = points[index + 3]
        dt = later.t - first.t
        if dt <= 0.0:
            continue
        if math.hypot(later.x - first.x, later.y - first.y) / dt >= 0.20:
            return index
    return None


def extract_track(path: Path, record: dict[str, Any]) -> Track | None:
    if record.get("end_reason") != "ball_settled":
        return None
    points = _raw_points(record)
    if len(points) < 12:
        return None
    start_index = _movement_start(points)
    if start_index is None:
        return None
    points = points[start_index:]
    if len(points) < 10 or points[-1].t - points[0].t < 0.35:
        return None

    dx = points[-1].x - points[0].x
    dy = points[-1].y - points[0].y
    displacement = math.hypot(dx, dy)
    path_length = _path_length(points)
    if displacement < MIN_TRACK_DISTANCE_M or path_length <= 1e-6:
        return None
    straightness = displacement / path_length
    if straightness < MIN_STRAIGHTNESS:
        return None

    return Track(
        dataset=path.name,
        kick_id=record.get("kick_id"),
        points=tuple(points),
        direction_x=dx / displacement,
        direction_y=dy / displacement,
        straightness=straightness,
        distance_m=displacement,
    )


def load_tracks(paths: list[Path]) -> tuple[list[Track], dict[str, int]]:
    tracks: list[Track] = []
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        for record in load_jsonl(path):
            if record.get("record_type") != "kick":
                continue
            counts[f"{path.name}:all_kicks"] += 1
            track = extract_track(path, record)
            if track is not None:
                tracks.append(track)
                counts[f"{path.name}:clean_tracks"] += 1
    return tracks, dict(counts)


def _project(track: Track, point: Point) -> float:
    origin = track.points[0]
    return (point.x - origin.x) * track.direction_x + (point.y - origin.y) * track.direction_y


def _weighted_linear_velocity(track: Track, center_t: float, radius: float) -> float | None:
    rows: list[tuple[float, float, float]] = []
    for point in track.points:
        offset = point.t - center_t
        if abs(offset) <= radius:
            weight = math.exp(-abs(offset) / max(0.04, radius * 0.65))
            rows.append((offset, _project(track, point), weight))
    if len(rows) < 4:
        return None
    total_weight = sum(row[2] for row in rows)
    t_mean = sum(t * weight for t, _, weight in rows) / total_weight
    s_mean = sum(s * weight for _, s, weight in rows) / total_weight
    denom = sum(weight * (t - t_mean) ** 2 for t, _, weight in rows)
    if denom <= 1e-10:
        return None
    return sum(weight * (t - t_mean) * (s - s_mean) for t, s, weight in rows) / denom


def deceleration_observations(tracks: list[Track]) -> list[tuple[float, float]]:
    observations: list[tuple[float, float]] = []
    for track in tracks:
        candidates: list[tuple[float, float]] = []
        next_t = track.points[0].t + VELOCITY_RADIUS_SEC
        end_t = track.points[-1].t - VELOCITY_RADIUS_SEC
        while next_t <= end_t:
            velocity = _weighted_linear_velocity(track, next_t, VELOCITY_RADIUS_SEC)
            if velocity is not None:
                candidates.append((next_t, velocity))
            next_t += 0.08
        if len(candidates) < 3:
            continue

        # Start at the peak fitted speed to exclude the foot-contact acceleration phase.
        peak_index = max(range(len(candidates)), key=lambda index: candidates[index][1])
        candidates = candidates[peak_index:]
        for index, (before_t, before_v) in enumerate(candidates):
            for after_t, after_v in candidates[index + 1 :]:
                dt = after_t - before_t
                if dt < ACCEL_PAIR_MIN_SEC:
                    continue
                if dt > ACCEL_PAIR_MAX_SEC:
                    break
                speed = 0.5 * (before_v + after_v)
                decel = (before_v - after_v) / dt
                if (
                    MIN_SPEED_MPS <= speed <= MAX_SPEED_MPS
                    and 0.0 <= decel <= MAX_DECEL_MPS2
                ):
                    observations.append((speed, decel))
                break
    return observations


def _weighted_regression(
    xs: list[float], ys: list[float], weights: list[float]
) -> tuple[float, float]:
    total = sum(weights)
    if total <= 0.0:
        return 0.01, 0.0
    x_mean = sum(w * x for x, w in zip(xs, weights)) / total
    y_mean = sum(w * y for y, w in zip(ys, weights)) / total
    denom = sum(w * (x - x_mean) ** 2 for x, w in zip(xs, weights))
    if denom <= 1e-12:
        return max(0.01, y_mean), 0.0
    slope = sum(
        w * (x - x_mean) * (y - y_mean)
        for x, y, w in zip(xs, ys, weights)
    ) / denom
    intercept = y_mean - slope * x_mean
    return max(0.01, intercept), max(0.0, slope)


def _robust_fit(observations: list[tuple[float, float]], power: int) -> tuple[float, float]:
    if not observations:
        return 3.0, 0.0
    ys = [decel for _, decel in observations]
    if power == 0:
        return max(0.01, statistics.median(ys)), 0.0

    xs = [speed**power for speed, _ in observations]
    weights = [1.0] * len(xs)
    intercept, slope = _weighted_regression(xs, ys, weights)
    for _ in range(8):
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        scale = max(0.10, 1.4826 * statistics.median(abs(value) for value in residuals))
        cutoff = 1.5 * scale
        weights = [1.0 if abs(value) <= cutoff else cutoff / abs(value) for value in residuals]
        intercept, slope = _weighted_regression(xs, ys, weights)
    return intercept, slope


def fit_models(tracks: list[Track]) -> list[ResistanceModel]:
    observations = deceleration_observations(tracks)
    models: list[ResistanceModel] = []
    for name, power in (("constant", 0), ("linear", 1), ("quadratic", 2)):
        a0, k = _robust_fit(observations, power)
        models.append(ResistanceModel(name, a0, k, power, len(observations)))
    return models


def stop_time_and_distance(model: ResistanceModel, speed: float) -> tuple[float, float]:
    speed = max(0.0, speed)
    a0 = max(0.01, model.a0_mps2)
    k = max(0.0, model.k)
    if speed <= 0.0:
        return 0.0, 0.0
    if model.feature_power == 0 or k <= 1e-8:
        return speed / a0, speed * speed / (2.0 * a0)
    if model.feature_power == 1:
        ratio = k * speed / a0
        stop_time = math.log1p(ratio) / k
        stop_distance = speed / k - a0 * math.log1p(ratio) / (k * k)
        return stop_time, max(0.0, stop_distance)

    root = math.sqrt(k / a0)
    stop_time = math.atan(speed * root) / math.sqrt(a0 * k)
    stop_distance = math.log1p(k * speed * speed / a0) / (2.0 * k)
    return stop_time, max(0.0, stop_distance)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def _endpoint_component(points: list[Point], end_t: float, component: str) -> float | None:
    rows: list[tuple[float, float, float]] = []
    for point in points:
        offset = point.t - end_t
        if -0.28 <= offset <= 1e-6:
            value = point.x if component == "x" else point.y
            weight = math.exp(offset / 0.12)
            rows.append((offset, value, weight))
    if len(rows) < 4:
        return None

    # Weighted quadratic fit at t=end_t; coefficient c1 is endpoint velocity.
    matrix = [[0.0] * 3 for _ in range(3)]
    vector = [0.0] * 3
    for offset, value, weight in rows:
        basis = (1.0, offset, offset * offset)
        for row in range(3):
            vector[row] += weight * basis[row] * value
            for column in range(3):
                matrix[row][column] += weight * basis[row] * basis[column]
    solution = _solve_3x3(matrix, vector)
    return None if solution is None else solution[1]


def predict_stop(track: Track, model: ResistanceModel, horizon_sec: float) -> tuple[float, float] | None:
    start_t = track.points[0].t
    end_t = start_t + horizon_sec
    prefix = [point for point in track.points if point.t <= end_t + 1e-6]
    if len(prefix) < 4:
        return None
    last = prefix[-1]
    vx = _endpoint_component(prefix, last.t, "x")
    vy = _endpoint_component(prefix, last.t, "y")
    if vx is None or vy is None:
        return None
    speed = math.hypot(vx, vy)
    if not (MIN_SPEED_MPS <= speed <= MAX_SPEED_MPS):
        return None
    _, distance = stop_time_and_distance(model, speed)
    return last.x + vx / speed * distance, last.y + vy / speed * distance


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "mean": statistics.fmean(values) if values else None,
    }


def cross_validate(
    tracks: list[Track], horizons: tuple[float, ...]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float | int | None]]]]:
    datasets = sorted({track.dataset for track in tracks})
    folds: list[dict[str, Any]] = []
    aggregated: dict[str, dict[str, list[float]]] = {
        name: {str(horizon): [] for horizon in horizons}
        for name in ("constant", "linear", "quadratic")
    }

    for held_out in datasets:
        train = [track for track in tracks if track.dataset != held_out]
        validation = [track for track in tracks if track.dataset == held_out]
        models = fit_models(train)
        fold_errors: dict[str, dict[str, dict[str, float | int | None]]] = {}
        for model in models:
            by_horizon: dict[str, list[float]] = {str(horizon): [] for horizon in horizons}
            for track in validation:
                actual = track.points[-1]
                for horizon in horizons:
                    predicted = predict_stop(track, model, horizon)
                    if predicted is None:
                        continue
                    error = math.hypot(predicted[0] - actual.x, predicted[1] - actual.y)
                    by_horizon[str(horizon)].append(error)
                    aggregated[model.name][str(horizon)].append(error)
            fold_errors[model.name] = {
                horizon: summarize(errors) for horizon, errors in by_horizon.items()
            }
        folds.append(
            {
                "held_out": held_out,
                "train_tracks": len(train),
                "validation_tracks": len(validation),
                "models": [asdict(model) for model in models],
                "errors_m": fold_errors,
            }
        )

    aggregate_summary = {
        model: {horizon: summarize(errors) for horizon, errors in by_horizon.items()}
        for model, by_horizon in aggregated.items()
    }
    return folds, aggregate_summary


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def choose_model(aggregate: dict[str, dict[str, dict[str, Any]]]) -> str:
    target = "0.35"
    ranked = []
    for name, horizons in aggregate.items():
        stats = horizons.get(target) or {}
        median_error = stats.get("median")
        p75_error = stats.get("p75")
        if median_error is not None and p75_error is not None:
            ranked.append((float(median_error), float(p75_error), name))
    return min(ranked)[2] if ranked else "constant"


def deployment_decision(
    aggregate: dict[str, dict[str, dict[str, Any]]], candidate: str
) -> dict[str, Any]:
    baseline_035 = aggregate["constant"]["0.35"].get("median")
    candidate_035 = aggregate[candidate]["0.35"].get("median")
    candidate_050_p75 = aggregate[candidate]["0.5"].get("p75")
    improvement = None
    if baseline_035 not in (None, 0.0) and candidate_035 is not None:
        improvement = (float(baseline_035) - float(candidate_035)) / float(baseline_035)
    gates = {
        "relative_improvement_at_035": (
            improvement is not None and improvement >= MIN_RELATIVE_IMPROVEMENT
        ),
        "median_error_at_035": (
            candidate_035 is not None and float(candidate_035) <= MAX_MEDIAN_ERROR_035_M
        ),
        "p75_error_at_050": (
            candidate_050_p75 is not None
            and float(candidate_050_p75) <= MAX_P75_ERROR_050_M
        ),
    }
    approved = candidate != "constant" and all(gates.values())
    return {
        "candidate": candidate,
        "approved_for_hard_target": approved,
        "recommended_mode": "hard_target_candidate" if approved else "confidence_weighted_advisory",
        "relative_improvement_at_035": improvement,
        "thresholds": {
            "min_relative_improvement_at_035": MIN_RELATIVE_IMPROVEMENT,
            "max_median_error_at_035_m": MAX_MEDIAN_ERROR_035_M,
            "max_p75_error_at_050_m": MAX_P75_ERROR_050_M,
        },
        "gates": gates,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ball Motion Model Cross-Validation",
        "",
        "This report uses leave-one-dataset-file-out cross-validation. Raw JSONL files are read-only.",
        "",
        "## Data Filtering",
        "",
        f"- Clean free-roll tracks: {report['clean_track_count']}",
        f"- Minimum travel: {MIN_TRACK_DISTANCE_M:.2f} m",
        f"- Minimum straightness: {MIN_STRAIGHTNESS:.2f}",
        "- Only `ball_settled` kicks are used; the static pre-kick phase and fitted acceleration phase are excluded.",
        "- Robot contact is approximated by straightness and monotonic deceleration filters; future recorder versions should log contact flags explicitly.",
        "",
        "## Aggregate Held-Out Stop-Point Error",
        "",
        "| Model | Horizon | n | Median | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, horizons in report["aggregate_errors_m"].items():
        for horizon, stats in horizons.items():
            lines.append(
                f"| {model_name} | {horizon}s | {stats['count']} | "
                f"{_fmt(stats['median'])} m | {_fmt(stats['p75'])} m | {_fmt(stats['p90'])} m |"
            )

    lines.extend(["", "## Fold Parameters", ""])
    for fold in report["folds"]:
        lines.append(f"### Held out: `{fold['held_out']}`")
        lines.append("")
        lines.append(
            f"Train/validation tracks: {fold['train_tracks']} / {fold['validation_tracks']}"
        )
        lines.append("")
        for model in fold["models"]:
            lines.append(
                f"- {model['name']}: a0={model['a0_mps2']:.4f} m/s², "
                f"k={model['k']:.4f}, fit samples={model['fit_samples']}"
            )
        lines.append("")

    selected = report["selected_model"]
    decision = report["deployment_decision"]
    constant = report["aggregate_errors_m"]["constant"]["0.35"]
    selected_stats = report["aggregate_errors_m"][selected]["0.35"]
    base = constant.get("median")
    chosen = selected_stats.get("median")
    improvement = None
    if base not in (None, 0.0) and chosen is not None:
        improvement = (float(base) - float(chosen)) / float(base)

    lines.extend(
        [
            "## Decision",
            "",
            f"- Data-selected model at the 0.35s decision horizon: **{selected}**.",
            f"- 0.35s median error: {_fmt(chosen)} m; constant baseline: {_fmt(base)} m.",
            f"- Relative median improvement: {_fmt(None if improvement is None else improvement * 100.0, 1)}%.",
            f"- Hard-target gates: `{decision['gates']}`.",
            f"- Approved for hard-target use: **{decision['approved_for_hard_target']}**.",
            f"- Recommended integration mode: **{decision['recommended_mode']}**.",
            "- Selection in this report is evidence for implementation, not automatic authorization to change match behavior.",
            "- Before hard-target integration, require stable fold parameters, useful p75 error, and a confidence-gated fallback.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/v1", help="Directory containing JSONL files")
    parser.add_argument("--output", default="analysis/v1", help="Directory for derived reports")
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS_SEC),
        help="Comma-separated observation horizons in seconds",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    paths = sorted(input_dir.glob("*.jsonl"))
    if len(paths) < 2:
        raise SystemExit("Cross-validation requires at least two JSONL dataset files")
    horizons = tuple(float(item) for item in args.horizons.split(",") if item.strip())
    if 0.35 not in horizons:
        raise SystemExit("The model-selection horizon 0.35s must be included")

    tracks, counts = load_tracks(paths)
    if len({track.dataset for track in tracks}) < 2:
        raise SystemExit("Not enough dataset files contain clean free-roll tracks")
    folds, aggregate = cross_validate(tracks, horizons)
    selected_model = choose_model(aggregate)
    final_models = fit_models(tracks)
    report = {
        "inputs": [str(path) for path in paths],
        "filter_counts": counts,
        "clean_track_count": len(tracks),
        "horizons_sec": list(horizons),
        "folds": folds,
        "aggregate_errors_m": aggregate,
        "selected_model": selected_model,
        "final_models": [asdict(model) for model in final_models],
        "deployment_decision": deployment_decision(aggregate, selected_model),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ball_motion_cross_validation.json"
    markdown_path = output_dir / "ball_motion_cross_validation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
