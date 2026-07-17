#!/usr/bin/env python3
"""Train bounded ball path models from schema-v4 launch/censor events.

The extractor deliberately uses only observations at and after ``launch_event``
and never trains beyond ``free_path_valid_until_sec``.  Match files are the
validation unit, so adjacent points from one physical event cannot leak across
train and validation sets.  Every candidate and the eventual online model use
only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


INPUT_POINTS = 5
FUTURE_HORIZONS = (0.10, 0.25, 0.50, 0.75, 1.00)
LAUNCH_TYPES = ("own_kick_command", "robot_kick_candidate")
RIDGE_VALUES = (0.03, 0.30, 3.00)
KNN_VALUES = (8, 16, 32)
_EPS = 1e-9


@dataclass(frozen=True)
class Point:
    t: float
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Example:
    match_file: str
    match_id: int
    motion_id: int
    launch_type: str
    kick_power: float | None
    features: tuple[float, ...]
    origin_x: float
    origin_y: float
    direction_x: float
    direction_y: float
    targets: dict[float, tuple[float, float]]


@dataclass(frozen=True)
class Scaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales)
        )


@dataclass(frozen=True)
class RidgeModel:
    scaler: Scaler
    longitudinal: tuple[float, ...]
    lateral: tuple[float, ...]

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        row = (1.0,) + self.scaler.transform(features)
        return _dot(row, self.longitudinal), _dot(row, self.lateral)


@dataclass(frozen=True)
class KnnModel:
    scaler: Scaler
    rows: tuple[tuple[tuple[float, ...], float, float], ...]
    neighbors: int

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        query = self.scaler.transform(features)
        ranked = sorted(
            (
                sum((left - right) ** 2 for left, right in zip(query, row)),
                longitudinal,
                lateral,
            )
            for row, longitudinal, lateral in self.rows
        )[: self.neighbors]
        if not ranked:
            return 0.0, 0.0
        weights = [1.0 / (0.04 + distance) for distance, _, _ in ranked]
        total = sum(weights)
        return (
            sum(weight * row[1] for weight, row in zip(weights, ranked)) / total,
            sum(weight * row[2] for weight, row in zip(weights, ranked)) / total,
        )


@dataclass(frozen=True)
class SharedKnnModel:
    """One bounded neighbor search reused by every future-time query."""

    scaler: Scaler
    rows: tuple[tuple[tuple[float, ...], dict[float, tuple[float, float]]], ...]
    neighbors: int

    def predict_all(self, features: Sequence[float]) -> dict[float, tuple[float, float]]:
        query = self.scaler.transform(features)
        ranked = sorted(
            (sum((left - right) ** 2 for left, right in zip(query, row)), targets)
            for row, targets in self.rows
        )
        predictions = {}
        for horizon in FUTURE_HORIZONS:
            nearest = [(distance, targets[horizon]) for distance, targets in ranked if horizon in targets][: self.neighbors]
            if not nearest:
                continue
            weights = [1.0 / (0.04 + distance) for distance, _ in nearest]
            total = sum(weights)
            predictions[horizon] = (
                sum(weight * target[0] for weight, (_, target) in zip(weights, nearest)) / total,
                sum(weight * target[1] for weight, (_, target) in zip(weights, nearest)) / total,
            )
        return predictions


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * fraction
    lower = int(math.floor(index))
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.median(rows) if rows else 0.0


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _interpolate(points: Sequence[Point], target_t: float) -> Point | None:
    for before, after in zip(points, points[1:]):
        if before.t - 1e-8 <= target_t <= after.t + 1e-8:
            span = after.t - before.t
            weight = 0.0 if span <= _EPS else (target_t - before.t) / span
            return Point(
                target_t,
                before.x + (after.x - before.x) * weight,
                before.y + (after.y - before.y) * weight,
                min(before.confidence, after.confidence),
            )
    return None


def _linear_fit(points: Sequence[Point], component: str) -> tuple[float, float]:
    """Return endpoint velocity and RMS residual for a weighted line."""

    end = points[-1].t
    rows = []
    for index, point in enumerate(points):
        t = point.t - end
        value = point.x if component == "x" else point.y
        weight = max(0.05, point.confidence) * (1.0 + index)
        rows.append((t, value, weight))
    sw = sum(row[2] for row in rows)
    st = sum(t * weight for t, _, weight in rows)
    sy = sum(value * weight for _, value, weight in rows)
    stt = sum(t * t * weight for t, _, weight in rows)
    sty = sum(t * value * weight for t, value, weight in rows)
    denominator = sw * stt - st * st
    slope = 0.0 if abs(denominator) <= _EPS else (sw * sty - st * sy) / denominator
    intercept = (sy - slope * st) / max(_EPS, sw)
    residual = math.sqrt(
        sum(weight * (value - intercept - slope * t) ** 2 for t, value, weight in rows)
        / max(_EPS, sw)
    )
    return slope, residual


def _features(points: Sequence[Point]) -> tuple[tuple[float, ...], float, float]:
    vx, rx = _linear_fit(points, "x")
    vy, ry = _linear_fit(points, "y")
    speed = math.hypot(vx, vy)
    if speed <= 0.03:
        dx = points[-1].x - points[0].x
        dy = points[-1].y - points[0].y
        norm = math.hypot(dx, dy)
        ux, uy = ((1.0, 0.0) if norm <= _EPS else (dx / norm, dy / norm))
    else:
        ux, uy = vx / speed, vy / speed
    segment_velocities: list[tuple[float, float]] = []
    for before, after in zip(points, points[1:]):
        dt = after.t - before.t
        if dt > 1e-4:
            segment_velocities.append(((after.x - before.x) / dt, (after.y - before.y) / dt))
    projected = [vx0 * ux + vy0 * uy for vx0, vy0 in segment_velocities]
    lateral = [-vx0 * uy + vy0 * ux for vx0, vy0 in segment_velocities]
    recent_speed = _median(projected[-2:])
    early_speed = _median(projected[:2])
    span = max(1e-4, points[-1].t - points[0].t)
    accel = (recent_speed - early_speed) / span
    lateral_speed = _median(lateral[-2:])
    speed_jitter = _median(abs(value - _median(projected)) for value in projected)
    confidence = _median(point.confidence for point in points)
    features = (
        speed,
        speed * speed,
        recent_speed,
        early_speed,
        accel,
        max(-12.0, min(12.0, accel)) * speed,
        lateral_speed,
        speed_jitter,
        math.hypot(rx, ry),
        span,
        confidence,
    )
    return features, ux, uy


def _parse_points(raw: object) -> list[Point]:
    if not isinstance(raw, list):
        return []
    result = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            result.append(
                Point(float(row["t_sec"]), float(row["x"]), float(row["y"]), float(row.get("confidence", 1.0)))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def load_examples(dataset: Path) -> tuple[list[Example], dict[str, int]]:
    examples: list[Example] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for path in sorted(dataset.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("record_type") != "ball_motion":
                continue
            launch = record.get("launch_event") or {}
            launch_type = launch.get("type")
            if launch_type not in LAUNCH_TYPES:
                counts["rejected_launch_type"] += 1
                continue
            points = _parse_points(record.get("trajectory"))
            launch_index = launch.get("point_index")
            free_until = record.get("free_path_valid_until_sec")
            if not isinstance(launch_index, int) or not isinstance(free_until, (int, float)):
                counts["rejected_missing_boundary"] += 1
                continue
            prefix = points[launch_index : launch_index + INPUT_POINTS]
            if len(prefix) != INPUT_POINTS or prefix[-1].t - prefix[0].t > 0.30:
                counts["rejected_short_prefix"] += 1
                continue
            features, ux, uy = _features(prefix)
            origin = prefix[-1]
            targets: dict[float, tuple[float, float]] = {}
            for horizon in FUTURE_HORIZONS:
                target_t = origin.t + horizon
                if target_t > float(free_until) + 1e-8:
                    continue
                target = _interpolate(points, target_t)
                if target is None:
                    continue
                dx, dy = target.x - origin.x, target.y - origin.y
                targets[horizon] = (dx * ux + dy * uy, -dx * uy + dy * ux)
            if not targets:
                counts["rejected_no_target"] += 1
                continue
            examples.append(
                Example(
                    path.name,
                    int(record["match_id"]),
                    int(record["motion_id"]),
                    str(launch_type),
                    float(record["kick_power"]) if record.get("kick_power") is not None else None,
                    features,
                    origin.x,
                    origin.y,
                    ux,
                    uy,
                    targets,
                )
            )
            counts["accepted"] += 1
            counts[f"accepted_{launch_type}"] += 1
    counts["files"] = len(list(dataset.glob("*.jsonl")))
    return examples, dict(counts)


def _fit_scaler(rows: Sequence[Sequence[float]]) -> Scaler:
    columns = list(zip(*rows))
    means = tuple(statistics.fmean(column) for column in columns)
    scales = tuple(max(1e-6, statistics.pstdev(column)) for column in columns)
    return Scaler(means, scales)


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return tuple(0.0 for _ in vector)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        factor = augmented[column][column]
        augmented[column] = [value / factor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) > 1e-15:
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return tuple(augmented[row][-1] for row in range(size))


def fit_ridge(examples: Sequence[Example], horizon: float, ridge: float) -> RidgeModel:
    usable = [example for example in examples if horizon in example.targets]
    scaler = _fit_scaler([example.features for example in usable])
    rows = [(1.0,) + scaler.transform(example.features) for example in usable]
    size = len(rows[0])
    matrix = [[0.0] * size for _ in range(size)]
    vectors = [[0.0] * size for _ in range(2)]
    for row, example in zip(rows, usable):
        target = example.targets[horizon]
        for i in range(size):
            vectors[0][i] += row[i] * target[0]
            vectors[1][i] += row[i] * target[1]
            for j in range(size):
                matrix[i][j] += row[i] * row[j]
    for i in range(1, size):
        matrix[i][i] += ridge
    return RidgeModel(scaler, _solve(matrix, vectors[0]), _solve(matrix, vectors[1]))


def fit_knn(examples: Sequence[Example], horizon: float, neighbors: int) -> KnnModel:
    usable = [example for example in examples if horizon in example.targets]
    scaler = _fit_scaler([example.features for example in usable])
    rows = tuple(
        (scaler.transform(example.features), *example.targets[horizon]) for example in usable
    )
    return KnnModel(scaler, rows, neighbors)


def fit_shared_knn(examples: Sequence[Example], neighbors: int) -> SharedKnnModel:
    scaler = _fit_scaler([example.features for example in examples])
    rows = tuple((scaler.transform(example.features), example.targets) for example in examples)
    return SharedKnnModel(scaler, rows, neighbors)


def _predict_cv(example: Example, horizon: float) -> tuple[float, float]:
    return max(0.0, example.features[0] * horizon), 0.0


def _global_position(example: Example, local: tuple[float, float]) -> tuple[float, float]:
    longitudinal, lateral = local
    return (
        example.origin_x + longitudinal * example.direction_x - lateral * example.direction_y,
        example.origin_y + longitudinal * example.direction_y + lateral * example.direction_x,
    )


def validate(examples: Sequence[Example]) -> tuple[dict[str, object], str]:
    files = sorted({example.match_file for example in examples})
    errors: defaultdict[str, defaultdict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    fold_counts: dict[str, dict[str, int]] = {}
    for held_out in files:
        train = [example for example in examples if example.match_file != held_out]
        test = [example for example in examples if example.match_file == held_out]
        fold_counts[held_out] = {"train_events": len(train), "test_events": len(test)}
        shared_models = {value: fit_shared_knn(train, value) for value in KNN_VALUES}
        for example in test:
            for neighbors, model in shared_models.items():
                for horizon, local in model.predict_all(example.features).items():
                    if horizon not in example.targets:
                        continue
                    actual = _global_position(example, example.targets[horizon])
                    predicted = _global_position(example, local)
                    errors[f"shared_knn_{neighbors}"][horizon].append(math.hypot(predicted[0] - actual[0], predicted[1] - actual[1]))
        for horizon in FUTURE_HORIZONS:
            ridge_models = {value: fit_ridge(train, horizon, value) for value in RIDGE_VALUES}
            knn_models = {value: fit_knn(train, horizon, value) for value in KNN_VALUES}
            for example in test:
                if horizon not in example.targets:
                    continue
                actual = _global_position(example, example.targets[horizon])
                predictions = {"constant_velocity": _predict_cv(example, horizon)}
                predictions.update({f"ridge_{value:g}": model.predict(example.features) for value, model in ridge_models.items()})
                predictions.update({f"knn_{value}": model.predict(example.features) for value, model in knn_models.items()})
                for name, local in predictions.items():
                    predicted = _global_position(example, local)
                    errors[name][horizon].append(math.hypot(predicted[0] - actual[0], predicted[1] - actual[1]))
    metrics: dict[str, object] = {}
    scores: dict[str, float] = {}
    for name, by_horizon in errors.items():
        model_metrics = {}
        weighted = []
        for horizon in FUTURE_HORIZONS:
            values = by_horizon[horizon]
            row = {
                "n": len(values),
                "median_m": _percentile(values, 0.50),
                "p75_m": _percentile(values, 0.75),
                "p90_m": _percentile(values, 0.90),
                "mean_m": statistics.fmean(values),
            }
            model_metrics[str(horizon)] = row
            weight = {0.10: 0.8, 0.25: 1.2, 0.50: 1.4, 0.75: 1.0, 1.00: 0.8}[horizon]
            weighted.append(weight * (row["median_m"] + 0.55 * row["p75_m"] + 0.25 * row["p90_m"]))
        metrics[name] = model_metrics
        scores[name] = sum(weighted)
    raw_best = min(scores, key=scores.get)
    near_best = [name for name, score in scores.items() if score <= scores[raw_best] * 1.01]
    shared = [name for name in near_best if name.startswith("shared_knn_")]
    selected = min(shared, key=scores.get) if shared else raw_best
    selection = {
        "raw_accuracy_winner": raw_best,
        "near_best_tolerance": 0.01,
        "eligible_models": sorted(near_best),
        "reason": "reuse_one_neighbor_search_for_all_horizons" if shared else "lowest_validation_score",
    }
    return {"folds": fold_counts, "metrics": metrics, "scores": scores, "selection": selection}, selected


def fit_final(examples: Sequence[Example], selected: str) -> dict[str, object]:
    if selected.startswith("shared_knn_"):
        model = fit_shared_knn(examples, int(selected.rsplit("_", 1)[1]))
        return {
            "model_name": selected,
            "input_points": INPUT_POINTS,
            "future_horizons_sec": FUTURE_HORIZONS,
            "shared_model": {
                "kind": "shared_knn",
                "neighbors": model.neighbors,
                "feature_means": model.scaler.means,
                "feature_scales": model.scaler.scales,
                "rows": model.rows,
            },
        }
    models = {}
    for horizon in FUTURE_HORIZONS:
        if selected.startswith("ridge_"):
            model = fit_ridge(examples, horizon, float(selected.split("_", 1)[1]))
            models[str(horizon)] = {
                "kind": "ridge",
                "feature_means": model.scaler.means,
                "feature_scales": model.scaler.scales,
                "longitudinal_coefficients": model.longitudinal,
                "lateral_coefficients": model.lateral,
            }
        elif selected.startswith("knn_"):
            model = fit_knn(examples, horizon, int(selected.split("_", 1)[1]))
            models[str(horizon)] = {
                "kind": "knn",
                "neighbors": model.neighbors,
                "feature_means": model.scaler.means,
                "feature_scales": model.scaler.scales,
                "rows": model.rows,
            }
        else:
            models[str(horizon)] = {"kind": "constant_velocity"}
    return {"model_name": selected, "input_points": INPUT_POINTS, "future_horizons_sec": FUTURE_HORIZONS, "models": models}


def benchmark(examples: Sequence[Example], selected: str, repeats: int = 12000) -> dict[str, float]:
    horizon = 0.50
    if selected.startswith("shared_knn_"):
        model = fit_shared_knn(examples, int(selected.rsplit("_", 1)[1])); predict = model.predict_all
    elif selected.startswith("ridge_"):
        model = fit_ridge(examples, horizon, float(selected.split("_", 1)[1])); predict = model.predict
    elif selected.startswith("knn_"):
        model = fit_knn(examples, horizon, int(selected.split("_", 1)[1])); predict = model.predict
    else:
        predict = lambda row: (max(0.0, row[0] * horizon), 0.0)
    rows = [example.features for example in examples]
    timings = []
    for index in range(repeats):
        start = time.perf_counter_ns(); predict(rows[index % len(rows)]); timings.append((time.perf_counter_ns() - start) / 1000.0)
    return {"calls": repeats, "query_scope": "all_horizons" if selected.startswith("shared_knn_") else "one_horizon", "median_us": _percentile(timings, 0.50), "p99_us": _percentile(timings, 0.99), "max_us": max(timings)}


def _report(result: dict[str, object]) -> str:
    validation = result["validation"]
    selected = result["selected_model"]
    lines = [
        "# Schema v4 球启动事件路径模型训练报告",
        "",
        "本报告只使用启动事件之后、首个交互/终止事件之前的球轨迹前缀；验证按整场文件留出。旧 v1 自由停止轨迹未参与。",
        "",
        f"- 比赛文件：{result['extraction']['files']}",
        f"- 合法启动事件：{result['extraction']['accepted']}",
        f"- 选中候选：`{selected}`",
        f"- 全时距共享查询：median {result['benchmark']['median_us']:.1f} us，p99 {result['benchmark']['p99_us']:.1f} us",
        "",
        "## 整场留出误差",
        "",
        "| 模型 | 未来时距 | n | median | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(validation["metrics"]):
        for horizon in FUTURE_HORIZONS:
            row = validation["metrics"][name][str(horizon)]
            lines.append(f"| {name} | {horizon:.2f}s | {row['n']} | {row['median_m']:.3f}m | {row['p75_m']:.3f}m | {row['p90_m']:.3f}m |")
    lines.extend([
        "",
        "## 使用边界",
        "",
        "- 参数可进入离线审查和 shadow 回放；本工具不修改策略，也不代表已经授权预测结果接管追球。",
        "- 有效时域只覆盖训练过且事件前仍有效的未来时距；边界、门柱、二次触球等由上层场景判断截断。",
        "- 正式封装继续遵守 `BallTrajectoryPrediction` 接口原则，并根据整场留出误差校准不确定度。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset/v4/match_dataset.team"))
    parser.add_argument("--output", type=Path, default=Path("analysis/v4/ball_event_path_model.json"))
    parser.add_argument("--report", type=Path, default=Path("analysis/v4/ball_event_path_model.md"))
    args = parser.parse_args()
    examples, extraction = load_examples(args.dataset)
    if len({example.match_file for example in examples}) < 3:
        raise SystemExit("need at least three independent match files")
    validation, selected = validate(examples)
    result = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "extraction": extraction,
        "feature_names": ("speed", "speed_squared", "recent_speed", "early_speed", "acceleration", "acceleration_times_speed", "lateral_speed", "speed_jitter", "fit_residual", "observation_span", "confidence"),
        "validation": validation,
        "selected_model": selected,
        "benchmark": benchmark(examples, selected),
        "final_model": fit_final(examples, selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"extraction": extraction, "selected": selected, "benchmark": result["benchmark"], "scores": validation["scores"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
