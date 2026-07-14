#!/usr/bin/env python3
"""Analyze Booster 3v3 match dataset JSONL files.

The script keeps raw logs immutable and writes derived reports under analysis/.
It intentionally uses only the Python standard library so it can run in the
project container without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BALL_SETTLED_SPEED_MPS = 0.035
BALL_ROLLING_SPEED_MPS = 0.08
MIN_DT = 1e-4


def median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def mean(values: list[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def percentile(values: list[float], pct: float, default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def hypot2(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


@dataclass
class BallMotionSample:
    t_sec: float
    x: float
    y: float
    speed_mps: float


def trajectory_motion(trajectory: list[dict[str, Any]]) -> list[BallMotionSample]:
    samples: list[BallMotionSample] = []
    prev: dict[str, Any] | None = None
    for point in trajectory:
        if point.get("x") is None or point.get("y") is None or point.get("t_sec") is None:
            continue
        if prev is None:
            prev = point
            continue
        dt = float(point["t_sec"]) - float(prev["t_sec"])
        if dt <= MIN_DT:
            prev = point
            continue
        dx = float(point["x"]) - float(prev["x"])
        dy = float(point["y"]) - float(prev["y"])
        speed = hypot2(dx, dy) / dt
        samples.append(BallMotionSample(float(point["t_sec"]), float(point["x"]), float(point["y"]), speed))
        prev = point
    return samples


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if abs(denom) < 1e-9:
        return None
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom
    intercept = y_bar - slope * x_bar
    return slope, intercept


def estimate_stop_from_prefix(
    trajectory: list[dict[str, Any]],
    horizon_sec: float,
    fallback_decel_mps2: float,
) -> dict[str, float] | None:
    points = [
        p
        for p in trajectory
        if p.get("t_sec") is not None
        and p.get("x") is not None
        and p.get("y") is not None
        and float(p["t_sec"]) <= horizon_sec
    ]
    if len(points) < 4:
        return None

    # Ignore the short static phase before the ball really starts moving.
    motion = trajectory_motion(points)
    moving = [m for m in motion if m.speed_mps >= BALL_ROLLING_SPEED_MPS]
    if len(moving) < 3:
        return None

    last = moving[-1]
    prev = moving[-2]
    dt = max(MIN_DT, last.t_sec - prev.t_sec)
    vx = (last.x - prev.x) / dt
    vy = (last.y - prev.y) / dt
    speed = hypot2(vx, vy)
    if speed < BALL_SETTLED_SPEED_MPS:
        return {"x": last.x, "y": last.y, "t_stop": last.t_sec}

    reg = linear_regression([m.t_sec for m in moving], [m.speed_mps for m in moving])
    decel = fallback_decel_mps2
    if reg is not None:
        slope, _ = reg
        if slope < -0.01:
            decel = max(0.01, -slope)
    t_to_stop = speed / max(0.01, decel)
    distance = speed * speed / (2.0 * max(0.01, decel))
    return {
        "x": last.x + vx / speed * distance,
        "y": last.y + vy / speed * distance,
        "t_stop": last.t_sec + t_to_stop,
        "speed": speed,
        "decel": decel,
    }


def analyze_kick(record: dict[str, Any]) -> dict[str, Any] | None:
    trajectory = record.get("trajectory") or []
    if len(trajectory) < 3:
        return None
    motion = trajectory_motion(trajectory)
    speeds = [m.speed_mps for m in motion]
    moving = [m for m in motion if m.speed_mps >= BALL_ROLLING_SPEED_MPS]
    if not moving:
        return None
    first_point = trajectory[0]
    last_point = trajectory[-1]
    peak_speed = max(speeds) if speeds else 0.0
    first_moving_t = moving[0].t_sec
    initial_window_speeds = [
        m.speed_mps for m in moving if m.t_sec <= first_moving_t + 0.8
    ]
    estimated_initial_speed = max(initial_window_speeds) if initial_window_speeds else peak_speed
    distance = 0.0
    prev = None
    for point in trajectory:
        if point.get("x") is None or point.get("y") is None:
            continue
        if prev is not None:
            distance += hypot2(float(point["x"]) - float(prev["x"]), float(point["y"]) - float(prev["y"]))
        prev = point

    decel_samples: list[float] = []
    for before, after in zip(motion, motion[1:]):
        dt = after.t_sec - before.t_sec
        if dt <= MIN_DT:
            continue
        if before.speed_mps > after.speed_mps and before.speed_mps > BALL_ROLLING_SPEED_MPS:
            decel_samples.append((before.speed_mps - after.speed_mps) / dt)

    return {
        "dataset": "",
        "kick_id": record.get("kick_id"),
        "player_id": record.get("player_id"),
        "ready_slot": record.get("ready_slot"),
        "reason": record.get("reason"),
        "end_reason": record.get("end_reason"),
        "duration_sec": record.get("duration_sec"),
        "points": len(trajectory),
        "start": {"x": first_point.get("x"), "y": first_point.get("y")},
        "stop": {"x": last_point.get("x"), "y": last_point.get("y")},
        "path_distance_m": distance,
        "straight_distance_m": hypot2(
            float(last_point.get("x", 0.0)) - float(first_point.get("x", 0.0)),
            float(last_point.get("y", 0.0)) - float(first_point.get("y", 0.0)),
        ),
        "estimated_initial_speed_mps": estimated_initial_speed,
        "peak_speed_mps": peak_speed,
        "median_decel_mps2": median(decel_samples),
        "p75_decel_mps2": percentile(decel_samples, 0.75),
        "kick_power": (record.get("kick") or {}).get("power"),
        "kick_direction_rad": (record.get("kick") or {}).get("direction_rad"),
    }


def summarize_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    game_states: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    command_reasons: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    frame_gaps: list[float] = []
    move_speed_samples: list[float] = []
    yaw_speed_samples: list[float] = []
    command_tracking: list[float] = []
    ball_visible = 0
    frame_count = 0
    pose_total = 0
    pose_missing = 0
    elapsed_start = None
    elapsed_end = None
    prev_elapsed = None
    prev_robot_pose: dict[str, tuple[float, float, float, float]] = {}
    kick_rows: list[dict[str, Any]] = []

    for record in load_jsonl(path):
        record_type = record.get("record_type", "unknown")
        counts[record_type] += 1
        if record_type == "metadata":
            continue

        if record_type == "kick":
            kick = analyze_kick(record)
            if kick:
                kick["dataset"] = path.name
                kick_rows.append(kick)
            continue

        if record_type != "frame":
            continue

        frame_count += 1
        elapsed = record.get("elapsed_sec")
        if elapsed is not None:
            elapsed = float(elapsed)
            if elapsed_start is None:
                elapsed_start = elapsed
            elapsed_end = elapsed
            if prev_elapsed is not None:
                frame_gaps.append(elapsed - prev_elapsed)
            prev_elapsed = elapsed

        if record.get("ball"):
            ball_visible += 1

        game = record.get("game") or {}
        state = game.get("state") or "unknown"
        set_play = game.get("set_play") or "none"
        game_states[f"{state}/{set_play}"] += 1

        for role in (record.get("roles") or {}).values():
            roles[str(role)] += 1

        commands = record.get("commands") or {}
        for cmd in commands.values():
            intents[str(cmd.get("intent"))] += 1
            command_reasons[str(cmd.get("reason"))] += 1

        for side in ("teammates", "opponents"):
            for robot_id, robot in (record.get(side) or {}).items():
                pose_total += 1
                pose = robot.get("pose")
                if not pose:
                    pose_missing += 1
                    continue
                key = f"{side}:{robot_id}"
                t = float(record.get("monotonic_sec") or elapsed or 0.0)
                x = float(pose["x"])
                y = float(pose["y"])
                theta = float(pose["theta"])
                prev = prev_robot_pose.get(key)
                if prev is not None:
                    prev_t, prev_x, prev_y, prev_theta = prev
                    dt = t - prev_t
                    if 0.02 <= dt <= 0.3:
                        speed = hypot2(x - prev_x, y - prev_y) / dt
                        yaw_speed = abs(angle_diff(theta, prev_theta)) / dt
                        if speed < 3.0:
                            move_speed_samples.append(speed)
                        if yaw_speed < 8.0:
                            yaw_speed_samples.append(yaw_speed)
                prev_robot_pose[key] = (t, x, y, theta)

        for robot_id, cmd in commands.items():
            if cmd.get("intent") not in ("move", "kick"):
                continue
            vx = cmd.get("vx")
            vy = cmd.get("vy")
            if vx is None or vy is None:
                continue
            command_speed = hypot2(float(vx), float(vy))
            if command_speed > 0:
                command_tracking.append(command_speed)

    summary = {
        "dataset": path.name,
        "bytes": path.stat().st_size,
        "record_counts": dict(counts),
        "duration_sec": None if elapsed_start is None or elapsed_end is None else elapsed_end - elapsed_start,
        "frame_gap_median_sec": median(frame_gaps),
        "frame_gap_p95_sec": percentile(frame_gaps, 0.95),
        "frame_gap_max_sec": max(frame_gaps) if frame_gaps else 0.0,
        "ball_visible_ratio": ball_visible / frame_count if frame_count else 0.0,
        "pose_missing_ratio": pose_missing / pose_total if pose_total else 0.0,
        "top_game_states": game_states.most_common(8),
        "top_intents": intents.most_common(8),
        "top_command_reasons": command_reasons.most_common(10),
        "top_roles": roles.most_common(8),
        "robot_motion": {
            "speed_mps_median": median(move_speed_samples),
            "speed_mps_p75": percentile(move_speed_samples, 0.75),
            "speed_mps_p95": percentile(move_speed_samples, 0.95),
            "yaw_radps_median": median(yaw_speed_samples),
            "yaw_radps_p75": percentile(yaw_speed_samples, 0.75),
            "yaw_radps_p95": percentile(yaw_speed_samples, 0.95),
            "command_speed_mps_median": median(command_tracking),
        },
    }
    return summary, kick_rows, []


def summarize_power_model(kicks: list[dict[str, Any]]) -> dict[str, Any]:
    by_power: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kick in kicks:
        power = kick.get("kick_power")
        key = "unknown" if power is None else str(power)
        by_power[key].append(kick)
        by_reason[str(kick.get("reason"))].append(kick)

    power_rows: dict[str, dict[str, Any]] = {}
    for power, rows in sorted(by_power.items(), key=lambda item: item[0]):
        initial_speeds = [r["estimated_initial_speed_mps"] for r in rows]
        peak_speeds = [r["peak_speed_mps"] for r in rows]
        straight_distances = [r["straight_distance_m"] for r in rows]
        path_distances = [r["path_distance_m"] for r in rows]
        power_rows[power] = {
            "count": len(rows),
            "initial_speed_mps_median": median(initial_speeds),
            "initial_speed_mps_p25": percentile(initial_speeds, 0.25),
            "initial_speed_mps_p75": percentile(initial_speeds, 0.75),
            "peak_speed_mps_median": median(peak_speeds),
            "straight_distance_m_median": median(straight_distances),
            "path_distance_m_median": median(path_distances),
        }

    reason_rows: dict[str, dict[str, Any]] = {}
    for reason, rows in sorted(by_reason.items(), key=lambda item: len(item[1]), reverse=True):
        initial_speeds = [r["estimated_initial_speed_mps"] for r in rows]
        reason_rows[reason] = {
            "count": len(rows),
            "initial_speed_mps_median": median(initial_speeds),
            "initial_speed_mps_p25": percentile(initial_speeds, 0.25),
            "initial_speed_mps_p75": percentile(initial_speeds, 0.75),
        }

    usable_power_levels = [
        power for power, row in power_rows.items() if power != "unknown" and row["count"] >= 5
    ]
    return {
        "power_levels": power_rows,
        "by_reason": reason_rows,
        "can_fit_power_curve": len(usable_power_levels) >= 3,
        "notes": [
            "A power-to-speed curve needs at least three distinct kick_power levels with several samples each.",
            "Current data can calibrate the default kick power but cannot yet choose an arbitrary power for target distance.",
        ],
    }


def evaluate_stop_predictions(kicks: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    raw_kicks: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        for record in load_jsonl(path):
            if record.get("record_type") == "kick" and record.get("trajectory"):
                raw_kicks.append((path.name, record))

    global_decel = median([k["median_decel_mps2"] for k in kicks if k.get("median_decel_mps2")], 0.3)
    horizons = [0.5, 1.0, 1.5, 2.0]
    errors: dict[str, list[float]] = {str(h): [] for h in horizons}
    by_reason: dict[str, list[float]] = defaultdict(list)
    for dataset_name, record in raw_kicks:
        trajectory = record.get("trajectory") or []
        if len(trajectory) < 4:
            continue
        actual = trajectory[-1]
        if actual.get("x") is None or actual.get("y") is None:
            continue
        for horizon in horizons:
            prediction = estimate_stop_from_prefix(trajectory, horizon, global_decel)
            if not prediction:
                continue
            err = hypot2(prediction["x"] - float(actual["x"]), prediction["y"] - float(actual["y"]))
            errors[str(horizon)].append(err)
            if horizon == 1.0:
                by_reason[str(record.get("reason"))].append(err)

    return {
        "global_median_decel_mps2": global_decel,
        "stop_prediction_error_m": {
            horizon: {
                "count": len(vals),
                "median": median(vals),
                "p75": percentile(vals, 0.75),
                "p90": percentile(vals, 0.90),
            }
            for horizon, vals in errors.items()
        },
        "stop_prediction_error_by_reason_1s": {
            reason: {"count": len(vals), "median": median(vals), "p75": percentile(vals, 0.75)}
            for reason, vals in sorted(by_reason.items(), key=lambda item: len(item[1]), reverse=True)
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Match Dataset Analysis")
    lines.append("")
    lines.append("This report is generated from immutable raw JSONL logs under `dataset/`.")
    lines.append("")
    lines.append("## Dataset Quality")
    lines.append("")
    for item in report["datasets"]:
        lines.append(f"### {item['dataset']}")
        lines.append("")
        lines.append(f"- Records: {item['record_counts']}")
        lines.append(f"- Duration: {item['duration_sec']:.2f}s")
        lines.append(
            f"- Frame gap median/p95/max: {item['frame_gap_median_sec']:.3f}s / "
            f"{item['frame_gap_p95_sec']:.3f}s / {item['frame_gap_max_sec']:.3f}s"
        )
        lines.append(f"- Ball visible ratio: {item['ball_visible_ratio']:.3f}")
        lines.append(f"- Pose missing ratio: {item['pose_missing_ratio']:.5f}")
        lines.append(f"- Top game states: {item['top_game_states']}")
        lines.append(f"- Top command reasons: {item['top_command_reasons'][:5]}")
        lines.append("")

    ball = report["ball_model"]
    lines.append("## Ball Model")
    lines.append("")
    lines.append(f"- Kicks analyzed: {report['kick_count']}")
    lines.append(f"- Global median deceleration: {ball['global_median_decel_mps2']:.3f} m/s^2")
    lines.append("- Stop prediction error by observation horizon:")
    for horizon, stats in ball["stop_prediction_error_m"].items():
        lines.append(
            f"  - {horizon}s: n={stats['count']}, median={stats['median']:.3f}m, "
            f"p75={stats['p75']:.3f}m, p90={stats['p90']:.3f}m"
        )
    lines.append("")

    lines.append("## Kick Profile")
    lines.append("")
    lines.append(f"- Estimated initial speed median: {report['kick_profile']['initial_speed_mps_median']:.3f} m/s")
    lines.append(f"- Peak speed median: {report['kick_profile']['peak_speed_mps_median']:.3f} m/s")
    lines.append(f"- Straight distance median: {report['kick_profile']['straight_distance_m_median']:.3f} m")
    lines.append(f"- Path distance median: {report['kick_profile']['path_distance_m_median']:.3f} m")
    lines.append(f"- Kick end reasons: {report['kick_profile']['end_reasons']}")
    lines.append("")

    power_model = report["power_model"]
    lines.append("## Kick Power To Speed")
    lines.append("")
    for power, stats in power_model["power_levels"].items():
        lines.append(
            f"- power={power}: n={stats['count']}, initial speed median/p25/p75="
            f"{stats['initial_speed_mps_median']:.3f}/{stats['initial_speed_mps_p25']:.3f}/{stats['initial_speed_mps_p75']:.3f} m/s, "
            f"straight distance median={stats['straight_distance_m_median']:.3f} m"
        )
    if not power_model["can_fit_power_curve"]:
        lines.append(
            "- Power curve status: not enough distinct kick_power levels yet; collect multi-power calibration kicks before using this to choose arbitrary kick force."
        )
    lines.append("")

    lines.append("## Robot Motion Profile")
    lines.append("")
    for item in report["datasets"]:
        motion = item["robot_motion"]
        lines.append(
            f"- {item['dataset']}: speed median/p75/p95="
            f"{motion['speed_mps_median']:.3f}/{motion['speed_mps_p75']:.3f}/{motion['speed_mps_p95']:.3f} m/s, "
            f"yaw median/p75/p95={motion['yaw_radps_median']:.3f}/{motion['yaw_radps_p75']:.3f}/{motion['yaw_radps_p95']:.3f} rad/s"
        )
    lines.append("")

    lines.append("## First Tactical Conclusions")
    lines.append("")
    lines.append("- The data is suitable for a first ball stop-point predictor and a first robot ETA profile.")
    lines.append("- Use only PLAYING windows when training tactical decisions; FINISHED/READY/SET frames are useful for diagnostics but should not dominate strategy metrics.")
    lines.append("- The first deployable tactic should be a read-only predictor in `src/tactics/`, guarded by confidence and fallback to current direct-ball logic.")
    lines.append("- Before changing behavior, validate stop-point prediction on held-out kicks and require a median error below the robot control radius you are willing to trust.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset", help="Directory containing match_dataset*.jsonl")
    parser.add_argument("--output", default="analysis", help="Directory for derived reports")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No JSONL files found under {input_dir}")

    dataset_summaries: list[dict[str, Any]] = []
    all_kicks: list[dict[str, Any]] = []
    for path in paths:
        summary, kicks, _ = summarize_dataset(path)
        dataset_summaries.append(summary)
        all_kicks.extend(kicks)

    ball_model = evaluate_stop_predictions(all_kicks, paths)
    initial_speeds = [k["estimated_initial_speed_mps"] for k in all_kicks]
    peak_speeds = [k["peak_speed_mps"] for k in all_kicks]
    straight_distances = [k["straight_distance_m"] for k in all_kicks]
    path_distances = [k["path_distance_m"] for k in all_kicks]
    end_reasons = Counter(str(k["end_reason"]) for k in all_kicks)

    report = {
        "inputs": [str(p) for p in paths],
        "datasets": dataset_summaries,
        "kick_count": len(all_kicks),
        "kick_profile": {
            "initial_speed_mps_median": median(initial_speeds),
            "initial_speed_mps_p25": percentile(initial_speeds, 0.25),
            "initial_speed_mps_p75": percentile(initial_speeds, 0.75),
            "peak_speed_mps_median": median(peak_speeds),
            "peak_speed_mps_p75": percentile(peak_speeds, 0.75),
            "straight_distance_m_median": median(straight_distances),
            "path_distance_m_median": median(path_distances),
            "end_reasons": dict(end_reasons),
        },
        "ball_model": ball_model,
        "power_model": summarize_power_model(all_kicks),
        "kicks": all_kicks,
    }

    write_json(output_dir / "match_data_analysis.json", report)
    write_markdown(output_dir / "match_data_analysis.md", report)
    print(f"Wrote {output_dir / 'match_data_analysis.json'}")
    print(f"Wrote {output_dir / 'match_data_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
