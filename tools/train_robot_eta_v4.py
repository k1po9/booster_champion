#!/usr/bin/env python3
"""Train whole-match-held-out robot target ETA challengers from schema v4.

Only stable-target segments that actually reached the requested target supervise
total arrival time.  Censored segments remain useful diagnostics but are never
silently converted into completed ETA labels.  Runtime candidate mathematics is
standard-library only and bounded.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


MIN_TRAVEL_DISTANCE_M = 0.25
RIDGES = (0.1, 1.0, 10.0)
KNN_NEIGHBORS = (5, 9, 15)
_EPS = 1e-9


@dataclass(frozen=True)
class EtaExample:
    match_file: str
    match_id: int
    eta_id: int
    player_id: int
    distance_m: float
    path_heading_error_rad: float
    final_heading_error_rad: float
    speed_cap_mps: float
    arrive_distance_m: float
    observed_speed_mps: float | None
    observed_yaw_rate_radps: float | None
    start_phase: str
    relative_final_heading_error_rad: float
    control_distance_m: float
    avoidance_applied: bool
    command_speed_mps: float
    command_yaw_rate_radps: float
    target_age_sec: float
    duration_sec: float
    is_episode_start: bool

    def features(self) -> tuple[float, ...]:
        speed = 0.0 if self.observed_speed_mps is None else self.observed_speed_mps
        yaw = 0.0 if self.observed_yaw_rate_radps is None else self.observed_yaw_rate_radps
        distance = self.distance_m
        cap = max(0.1, self.speed_cap_mps)
        heading = self.path_heading_error_rad
        return (
            distance,
            distance / cap,
            math.sqrt(max(0.0, distance)) / cap,
            heading,
            heading * heading,
            self.final_heading_error_rad,
            distance * heading,
            cap,
            speed,
            speed / cap,
            yaw,
            1.0 if self.start_phase == "turn" else 0.0,
            1.0 if self.start_phase == "run" else 0.0,
            self.arrive_distance_m,
            1.0 if self.observed_speed_mps is None else 0.0,
            self.relative_final_heading_error_rad,
            self.relative_final_heading_error_rad * self.relative_final_heading_error_rad,
            self.control_distance_m,
            1.0 if self.avoidance_applied else 0.0,
            self.command_speed_mps,
            self.command_yaw_rate_radps,
            self.target_age_sec,
            min(1.0, self.target_age_sec),
        )


@dataclass(frozen=True)
class Scaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple((value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales))


@dataclass(frozen=True)
class RidgeModel:
    scaler: Scaler
    coefficients: tuple[float, ...]
    logarithmic: bool

    def predict(self, example: EtaExample) -> float:
        row = (1.0,) + self.scaler.transform(example.features())
        value = sum(left * right for left, right in zip(row, self.coefficients))
        return max(0.05, math.exp(value) if self.logarithmic else value)


@dataclass(frozen=True)
class KnnModel:
    scaler: Scaler
    rows: tuple[tuple[tuple[float, ...], float], ...]
    neighbors: int

    def predict(self, example: EtaExample) -> float:
        query = self.scaler.transform(example.features())
        nearest = sorted(
            (sum((a - b) ** 2 for a, b in zip(query, features)), duration)
            for features, duration in self.rows
        )[: self.neighbors]
        weights = [1.0 / (0.05 + distance) for distance, _ in nearest]
        return sum(weight * row[1] for weight, row in zip(weights, nearest)) / sum(weights)


def _angle_delta(after: float, before: float) -> float:
    return math.atan2(math.sin(after - before), math.cos(after - before))


def _load_file(path: Path) -> tuple[list[EtaExample], Counter]:
    frames: defaultdict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    eta_records = []
    counts = Counter()
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("record_type") == "frame":
            for key, robot in (record.get("teammates") or {}).items():
                pose = (robot or {}).get("pose")
                observed = (robot or {}).get("observed_monotonic_sec")
                if pose and observed is not None:
                    frames[int(key)].append((float(observed), float(pose["x"]), float(pose["y"]), float(pose["theta"])))
        elif record.get("record_type") == "robot_eta":
            eta_records.append(record)
            counts[f"end_{record.get('end_reason')}"] += 1
    examples = []
    for record in eta_records:
        if not record.get("eta_training_candidate"):
            counts["rejected_censored"] += 1
            continue
        trajectory = record.get("trajectory") or []
        if len(trajectory) < 3:
            counts["rejected_short"] += 1
            continue
        first = trajectory[0]
        distance = float(first.get("distance_to_requested_m") or 0.0)
        if distance < MIN_TRAVEL_DISTANCE_M:
            counts["complete_near_target"] += 1
            continue
        player_id = int(record["player_id"])
        speed, yaw_rate = _velocity_before(frames[player_id], float(record["started_monotonic_sec"]))
        queries = _arrival_queries(path.name, record, trajectory, speed, yaw_rate)
        examples.extend(queries)
        counts["accepted_travel"] += 1
        counts["accepted_queries"] += len(queries)
    return examples, counts


def _arrival_queries(
    match_file: str,
    record: dict,
    trajectory: list[dict],
    initial_speed: float | None,
    initial_yaw_rate: float | None,
) -> list[EtaExample]:
    """Create bounded, past-observable suffix queries from one completed trip."""

    selected = [0]
    next_t = 0.50
    for index, point in enumerate(trajectory[1:], 1):
        elapsed = float(point.get("t_sec") or 0.0)
        if elapsed + 1e-9 >= next_t:
            selected.append(index)
            next_t += 0.50
    result = []
    duration = float(record["duration_sec"])
    for index in selected:
        point = trajectory[index]
        elapsed = float(point.get("t_sec") or 0.0)
        remaining = duration - elapsed
        if remaining < 0.08:
            continue
        speed, yaw_rate = initial_speed, initial_yaw_rate
        if index > 0:
            before = trajectory[index - 1]
            before_pose = before.get("pose") or {}
            pose = point.get("pose") or {}
            dt = elapsed - float(before.get("t_sec") or 0.0)
            if dt > 1e-4:
                speed = math.hypot(
                    float(pose.get("x", 0.0)) - float(before_pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)) - float(before_pose.get("y", 0.0)),
                ) / dt
                yaw_rate = abs(_angle_delta(float(pose.get("theta", 0.0)), float(before_pose.get("theta", 0.0)))) / dt
        result.append(
            EtaExample(
                match_file,
                int(record["match_id"]),
                int(record["eta_id"]),
                int(record["player_id"]),
                float(point.get("distance_to_requested_m") or 0.0),
                abs(float(point.get("heading_to_path_error_rad") or 0.0)),
                abs(float(point.get("final_heading_error_rad") or 0.0)),
                float(point.get("linear_speed_limit_mps") or 0.8),
                float(point.get("arrive_distance_m") or 0.15),
                speed,
                yaw_rate,
                str(point.get("phase") or "unknown"),
                abs(_angle_delta(float(point.get("final_heading_error_rad") or 0.0), float(point.get("heading_to_path_error_rad") or 0.0))),
                _control_distance(point),
                bool(point.get("avoidance_applied")),
                abs(float((point.get("command") or {}).get("vx") or 0.0)),
                abs(float((point.get("command") or {}).get("vyaw") or 0.0)),
                elapsed,
                remaining,
                index == 0,
            )
        )
    return result


def _control_distance(point: dict) -> float:
    pose = point.get("pose") or {}
    target = point.get("control_target") or point.get("requested_target") or {}
    return math.hypot(
        float(target.get("x", pose.get("x", 0.0))) - float(pose.get("x", 0.0)),
        float(target.get("y", pose.get("y", 0.0))) - float(pose.get("y", 0.0)),
    )


def _velocity_before(
    frames: list[tuple[float, float, float, float]], started_at: float
) -> tuple[float | None, float | None]:
    if len(frames) < 2:
        return None, None
    times = [row[0] for row in frames]
    upper = bisect.bisect_right(times, started_at)
    if upper < 2:
        return None, None
    before, after = frames[upper - 2], frames[upper - 1]
    dt = after[0] - before[0]
    if dt < 0.02 or dt > 0.25 or started_at - after[0] > 0.25:
        return None, None
    return math.hypot(after[1] - before[1], after[2] - before[2]) / dt, abs(_angle_delta(after[3], before[3])) / dt


def load_examples(dataset: Path) -> tuple[list[EtaExample], dict[str, int]]:
    examples = []
    counts = Counter()
    paths = sorted(dataset.glob("*.jsonl"))
    for path in paths:
        rows, local = _load_file(path)
        examples.extend(rows)
        counts.update(local)
    counts["files"] = len(paths)
    counts["accepted_with_observed_speed"] = sum(row.observed_speed_mps is not None for row in examples)
    return examples, dict(counts)


def _fit_scaler(rows: Sequence[Sequence[float]]) -> Scaler:
    columns = list(zip(*rows))
    return Scaler(
        tuple(statistics.fmean(column) for column in columns),
        tuple(max(1e-6, statistics.pstdev(column)) for column in columns),
    )


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...]:
    size = len(vector)
    rows = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return tuple(0.0 for _ in vector)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for index in range(size):
            if index == column:
                continue
            scale = rows[index][column]
            rows[index] = [a - scale * b for a, b in zip(rows[index], rows[column])]
    return tuple(row[-1] for row in rows)


def fit_ridge(
    examples: Sequence[EtaExample],
    ridge: float,
    logarithmic: bool,
    *,
    balance_episode_starts: bool = False,
) -> RidgeModel:
    scaler = _fit_scaler([row.features() for row in examples])
    inputs = [(1.0,) + scaler.transform(row.features()) for row in examples]
    size = len(inputs[0])
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    episode_counts = Counter((row.match_file, row.match_id, row.eta_id) for row in examples)
    for features, example in zip(inputs, examples):
        target = math.log(example.duration_sec) if logarithmic else example.duration_sec
        weight = 1.0
        if balance_episode_starts:
            count = episode_counts[(example.match_file, example.match_id, example.eta_id)]
            weight = 0.5 / count + (0.5 if example.is_episode_start else 0.0)
        for i in range(size):
            vector[i] += weight * features[i] * target
            for j in range(size):
                matrix[i][j] += weight * features[i] * features[j]
    for i in range(1, size):
        matrix[i][i] += ridge
    return RidgeModel(scaler, _solve(matrix, vector), logarithmic)


def fit_knn(examples: Sequence[EtaExample], neighbors: int) -> KnnModel:
    scaler = _fit_scaler([row.features() for row in examples])
    return KnnModel(
        scaler,
        tuple((scaler.transform(row.features()), row.duration_sec) for row in examples),
        neighbors,
    )


def _kinematic(example: EtaExample, kind: str) -> float:
    travel = example.distance_m / max(0.1, 0.895 * example.speed_cap_mps)
    turn = example.path_heading_error_rad / 0.879
    final = example.final_heading_error_rad / 0.879
    if kind == "distance_only":
        return travel
    if kind == "serial_kinematic":
        return 0.20 + turn + travel + final
    return 0.20 + max(turn, travel) + 0.35 * final


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate(examples: Sequence[EtaExample]) -> tuple[dict[str, object], str]:
    files = sorted({row.match_file for row in examples})
    errors: defaultdict[str, list[float]] = defaultdict(list)
    signed: defaultdict[str, list[float]] = defaultdict(list)
    start_errors: defaultdict[str, list[float]] = defaultdict(list)
    start_signed: defaultdict[str, list[float]] = defaultdict(list)
    folds = {}
    for held_out in files:
        train = [row for row in examples if row.match_file != held_out]
        test = [row for row in examples if row.match_file == held_out]
        folds[held_out] = {"train": len(train), "test": len(test)}
        models = {}
        for ridge in RIDGES:
            models[f"ridge_{ridge:g}"] = fit_ridge(train, ridge, False)
            models[f"balanced_ridge_{ridge:g}"] = fit_ridge(
                train, ridge, False, balance_episode_starts=True
            )
            models[f"log_ridge_{ridge:g}"] = fit_ridge(train, ridge, True)
        for neighbors in KNN_NEIGHBORS:
            models[f"knn_{neighbors}"] = fit_knn(train, neighbors)
        for row in test:
            predictions = {name: _kinematic(row, name) for name in ("distance_only", "serial_kinematic", "parallel_kinematic")}
            predictions.update({name: model.predict(row) for name, model in models.items()})
            for name, prediction in predictions.items():
                delta = prediction - row.duration_sec
                errors[name].append(abs(delta))
                signed[name].append(delta)
                if row.is_episode_start:
                    start_errors[name].append(abs(delta))
                    start_signed[name].append(delta)
    metrics = {}
    start_metrics = {}
    scores = {}
    for name, values in errors.items():
        row = {
            "n": len(values),
            "median_abs_sec": _percentile(values, 0.50),
            "p75_abs_sec": _percentile(values, 0.75),
            "p90_abs_sec": _percentile(values, 0.90),
            "mean_abs_sec": statistics.fmean(values),
            "median_signed_sec": _percentile(signed[name], 0.50),
        }
        metrics[name] = row
        start_values = start_errors[name]
        start_metrics[name] = {
            "n": len(start_values),
            "median_abs_sec": _percentile(start_values, 0.50),
            "p75_abs_sec": _percentile(start_values, 0.75),
            "p90_abs_sec": _percentile(start_values, 0.90),
            "mean_abs_sec": statistics.fmean(start_values),
            "median_signed_sec": _percentile(start_signed[name], 0.50),
        }
        query_score = row["median_abs_sec"] + 0.65 * row["p75_abs_sec"] + 0.30 * row["p90_abs_sec"]
        start_row = start_metrics[name]
        start_score = start_row["median_abs_sec"] + 0.45 * start_row["p75_abs_sec"] + 0.20 * start_row["p90_abs_sec"]
        scores[name] = 0.55 * query_score + 0.45 * start_score
    raw_best = min(scores, key=scores.get)
    near_best = [name for name, score in scores.items() if score <= scores[raw_best] * 1.05]
    bounded = [name for name in near_best if "ridge_" in name and not name.startswith("log_")]
    selected = min(bounded, key=scores.get) if bounded else raw_best
    selection = {
        "raw_accuracy_winner": raw_best,
        "near_best_tolerance": 0.05,
        "eligible_models": sorted(near_best),
        "reason": "prefer_constant_time_model_for_many_eta_queries" if bounded else "lowest_validation_score",
    }
    return {"folds": folds, "metrics": metrics, "episode_start_metrics": start_metrics, "scores": scores, "selection": selection}, selected


def fit_final(examples: Sequence[EtaExample], selected: str) -> dict[str, object]:
    if selected.startswith("balanced_ridge_"):
        model = fit_ridge(
            examples, float(selected.rsplit("_", 1)[1]), False, balance_episode_starts=True
        )
        return {"kind": "ridge", "logarithmic": False, "balanced_episode_starts": True, "means": model.scaler.means, "scales": model.scaler.scales, "coefficients": model.coefficients}
    if selected.startswith("log_ridge_"):
        model = fit_ridge(examples, float(selected.rsplit("_", 1)[1]), True)
        return {"kind": "ridge", "logarithmic": True, "means": model.scaler.means, "scales": model.scaler.scales, "coefficients": model.coefficients}
    if selected.startswith("ridge_"):
        model = fit_ridge(examples, float(selected.rsplit("_", 1)[1]), False)
        return {"kind": "ridge", "logarithmic": False, "means": model.scaler.means, "scales": model.scaler.scales, "coefficients": model.coefficients}
    if selected.startswith("knn_"):
        model = fit_knn(examples, int(selected.rsplit("_", 1)[1]))
        return {"kind": "knn", "neighbors": model.neighbors, "means": model.scaler.means, "scales": model.scaler.scales, "rows": model.rows}
    return {"kind": selected}


def benchmark(examples: Sequence[EtaExample], selected: str) -> dict[str, float]:
    final = fit_final(examples, selected)
    if final["kind"] == "ridge":
        model = RidgeModel(Scaler(tuple(final["means"]), tuple(final["scales"])), tuple(final["coefficients"]), bool(final["logarithmic"])); predict = model.predict
    elif final["kind"] == "knn":
        model = KnnModel(Scaler(tuple(final["means"]), tuple(final["scales"])), tuple(final["rows"]), int(final["neighbors"])); predict = model.predict
    else:
        predict = lambda row: _kinematic(row, selected)
    samples = []
    for index in range(10000):
        start = time.perf_counter_ns(); predict(examples[index % len(examples)]); samples.append((time.perf_counter_ns() - start) / 1000.0)
    return {"median_us": _percentile(samples, 0.50), "p99_us": _percentile(samples, 0.99), "max_us": max(samples)}


def _report(result: dict[str, object]) -> str:
    lines = [
        "# Schema v4 Robot ETA Training",
        "",
        "Total ETA is supervised only by stable-target segments that actually arrived and started at least 0.25 m from the target. Validation holds out one whole match file.",
        "",
        f"- Travel arrivals: {result['extraction']['accepted_travel']}",
        f"- Fixed-interval suffix queries: {result['extraction']['accepted_queries']}",
        f"- Near-target arrivals kept out of travel model: {result['extraction']['complete_near_target']}",
        f"- Selected: `{result['selected_model']}`",
        f"- Runtime median/p99: {result['benchmark']['median_us']:.1f}/{result['benchmark']['p99_us']:.1f} us",
        "",
        "| Model | n | median abs | p75 abs | p90 abs | signed median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(result["validation"]["metrics"].items(), key=lambda item: result["validation"]["scores"][item[0]]):
        lines.append(f"| {name} | {row['n']} | {row['median_abs_sec']:.3f}s | {row['p75_abs_sec']:.3f}s | {row['p90_abs_sec']:.3f}s | {row['median_signed_sec']:.3f}s |")
    start = result["validation"]["episode_start_metrics"][result["selected_model"]]
    lines += [
        "",
        "## Original Episode Starts (selected model)",
        "",
        f"n={start['n']}, median={start['median_abs_sec']:.3f}s, p75={start['p75_abs_sec']:.3f}s, p90={start['p90_abs_sec']:.3f}s, signed median={start['median_signed_sec']:.3f}s.",
        "",
        "This result is not yet an interface approval. The small number of independent complete travel arrivals and the separate near-target alignment mode must be considered before strategy integration.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset/v4/match_dataset.team"))
    parser.add_argument("--output", type=Path, default=Path("analysis/v4/robot_eta_model.json"))
    parser.add_argument("--report", type=Path, default=Path("analysis/v4/robot_eta_model.md"))
    args = parser.parse_args()
    examples, extraction = load_examples(args.dataset)
    validation, selected = validate(examples)
    result = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "extraction": extraction,
        "feature_names": ("distance", "distance_over_cap", "sqrt_distance_over_cap", "path_heading_error", "path_heading_error_squared", "final_heading_error", "distance_times_heading", "speed_cap", "observed_speed", "observed_speed_over_cap", "observed_yaw_rate", "start_turn", "start_run", "arrive_distance", "speed_missing", "target_heading_relative_to_path", "target_heading_relative_to_path_squared", "control_distance", "avoidance_applied", "command_speed", "command_yaw_rate", "target_age", "target_age_capped_1s"),
        "validation": validation,
        "selected_model": selected,
        "benchmark": benchmark(examples, selected),
        "final_model": fit_final(examples, selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"extraction": extraction, "selected": selected, "benchmark": result["benchmark"], "scores": validation["scores"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
