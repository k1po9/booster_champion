#!/usr/bin/env python3
"""Cross-validate a first robot travel-time (ETA) calibration from match logs.

The current dataset records poses and executed velocity commands but not the
strategy's internal target point.  Therefore this experiment validates two
identifiable components only:

* time to accumulate 0.25/0.50/1.00 m during straight, sustained motion;
* time to accumulate 0.25/0.50/1.00 rad during sustained in-place turning.

Validation is leave-one-match-file-out.  Results are a locomotion calibration
for a future path-aware ETA model, not proof of arbitrary target ETA accuracy.
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


TRANSLATION_THRESHOLDS_M = (0.25, 0.50, 1.00)
TURN_THRESHOLDS_RAD = (0.25, 0.50, 1.00)
MAX_FRAME_GAP_SEC = 0.22
MIN_TRANSLATION_COMMAND_MPS = 0.25
MAX_STRAIGHT_YAW_COMMAND_RADPS = 0.25
MAX_TURN_TRANSLATION_COMMAND_MPS = 0.08
MIN_TURN_COMMAND_RADPS = 0.20
MIN_TRANSLATION_STRAIGHTNESS = 0.90


@dataclass(frozen=True)
class Sample:
    dataset: str
    player_id: str
    t: float
    x: float
    y: float
    theta: float
    command_speed: float
    command_yaw: float
    observed_speed: float
    observed_yaw_rate: float
    intent: str


@dataclass(frozen=True)
class TranslationEpisode:
    dataset: str
    player_id: str
    samples: tuple[Sample, ...]
    started_from_rest: bool
    straightness: float
    first_command_speed: float
    reach_times_sec: dict[str, float]


@dataclass(frozen=True)
class TurnEpisode:
    dataset: str
    player_id: str
    samples: tuple[Sample, ...]
    first_command_yaw: float
    reach_times_sec: dict[str, float]


@dataclass(frozen=True)
class EtaCalibration:
    translation_command_gain: float
    reaction_delay_sec: float
    acceleration_mps2: float
    yaw_command_gain: float
    translation_samples: int
    turn_samples: int
    reaction_episodes: int

    def cruise_speed(self, command_speed: float) -> float:
        return max(0.05, min(1.20, self.translation_command_gain * max(0.0, command_speed)))

    def yaw_rate(self, command_yaw: float) -> float:
        return max(0.05, min(2.00, self.yaw_command_gain * abs(command_yaw)))


@dataclass(frozen=True)
class EtaPrediction:
    baseline_sec: float
    empirical_cruise_sec: float
    dynamics_sec: float


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        'count': len(values),
        'median': percentile(values, 0.50),
        'p75': percentile(values, 0.75),
        'p90': percentile(values, 0.90),
        'mean': statistics.fmean(values) if values else None,
    }


def load_samples(path: Path) -> dict[str, list[Sample]]:
    result: dict[str, list[Sample]] = defaultdict(list)
    previous: dict[str, tuple[float, float, float, float]] = {}
    for record in load_jsonl(path):
        if record.get('record_type') != 'frame':
            continue
        game = record.get('game') or {}
        if game.get('state') != 'PLAYING' or game.get('stopped', False):
            previous.clear()
            continue
        t_value = record.get('monotonic_sec')
        if t_value is None:
            continue
        t = float(t_value)
        teammates = record.get('teammates') or {}
        commands = record.get('commands') or {}
        for player_id, robot in teammates.items():
            pose = (robot or {}).get('pose')
            command = commands.get(player_id) or {}
            if not pose:
                previous.pop(player_id, None)
                continue
            x = float(pose['x'])
            y = float(pose['y'])
            theta = float(pose['theta'])
            observed_speed = 0.0
            observed_yaw = 0.0
            before = previous.get(player_id)
            if before is not None:
                before_t, before_x, before_y, before_theta = before
                dt = t - before_t
                if 0.02 <= dt <= MAX_FRAME_GAP_SEC:
                    observed_speed = math.hypot(x - before_x, y - before_y) / dt
                    observed_yaw = abs(angle_diff(theta, before_theta)) / dt
            previous[player_id] = (t, x, y, theta)
            vx = float(command.get('vx') or 0.0)
            vy = float(command.get('vy') or 0.0)
            result[player_id].append(
                Sample(
                    dataset=path.name,
                    player_id=player_id,
                    t=t,
                    x=x,
                    y=y,
                    theta=theta,
                    command_speed=math.hypot(vx, vy),
                    command_yaw=float(command.get('vyaw') or 0.0),
                    observed_speed=observed_speed,
                    observed_yaw_rate=observed_yaw,
                    intent=str(command.get('intent') or 'unknown'),
                )
            )
    return result


def _segment(samples: list[Sample], predicate, sign_sensitive: bool = False) -> list[list[Sample]]:
    segments: list[list[Sample]] = []
    current: list[Sample] = []
    for sample in samples:
        continuous = not current or sample.t - current[-1].t <= MAX_FRAME_GAP_SEC
        same_sign = (
            not sign_sensitive
            or not current
            or sample.command_yaw * current[-1].command_yaw > 0.0
        )
        if predicate(sample) and continuous and same_sign:
            current.append(sample)
            continue
        if current:
            segments.append(current)
        current = [sample] if predicate(sample) else []
    if current:
        segments.append(current)
    return segments


def _reach_times_path(samples: list[Sample], thresholds: tuple[float, ...]) -> dict[str, float]:
    reached: dict[str, float] = {}
    distance = 0.0
    start_t = samples[0].t
    for before, after in zip(samples, samples[1:]):
        step = math.hypot(after.x - before.x, after.y - before.y)
        before_distance = distance
        distance += step
        for threshold in thresholds:
            key = str(threshold)
            if key in reached or distance < threshold or step <= 1e-9:
                continue
            ratio = (threshold - before_distance) / step
            reached[key] = before.t + ratio * (after.t - before.t) - start_t
    return reached


def _reach_times_turn(samples: list[Sample], thresholds: tuple[float, ...]) -> dict[str, float]:
    reached: dict[str, float] = {}
    rotation = 0.0
    start_t = samples[0].t
    for before, after in zip(samples, samples[1:]):
        step = abs(angle_diff(after.theta, before.theta))
        before_rotation = rotation
        rotation += step
        for threshold in thresholds:
            key = str(threshold)
            if key in reached or rotation < threshold or step <= 1e-9:
                continue
            ratio = (threshold - before_rotation) / step
            reached[key] = before.t + ratio * (after.t - before.t) - start_t
    return reached


def extract_episodes(paths: list[Path]) -> tuple[list[TranslationEpisode], list[TurnEpisode]]:
    translations: list[TranslationEpisode] = []
    turns: list[TurnEpisode] = []
    for path in paths:
        by_robot = load_samples(path)
        for samples in by_robot.values():
            straight_segments = _segment(
                samples,
                lambda sample: (
                    sample.intent == 'move'
                    and sample.command_speed >= MIN_TRANSLATION_COMMAND_MPS
                    and abs(sample.command_yaw) <= MAX_STRAIGHT_YAW_COMMAND_RADPS
                ),
            )
            for segment in straight_segments:
                if len(segment) < 5 or segment[-1].t - segment[0].t < 0.35:
                    continue
                path_length = sum(
                    math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(segment, segment[1:])
                )
                displacement = math.hypot(segment[-1].x - segment[0].x, segment[-1].y - segment[0].y)
                if path_length < TRANSLATION_THRESHOLDS_M[0]:
                    continue
                straightness = displacement / max(1e-9, path_length)
                if straightness < MIN_TRANSLATION_STRAIGHTNESS:
                    continue
                command_speeds = [sample.command_speed for sample in segment]
                if max(command_speeds) - min(command_speeds) > 0.18:
                    continue
                reach_times = _reach_times_path(segment, TRANSLATION_THRESHOLDS_M)
                if not reach_times:
                    continue
                translations.append(
                    TranslationEpisode(
                        dataset=path.name,
                        player_id=segment[0].player_id,
                        samples=tuple(segment),
                        started_from_rest=segment[0].observed_speed <= 0.12,
                        straightness=straightness,
                        first_command_speed=segment[0].command_speed,
                        reach_times_sec=reach_times,
                    )
                )

            turn_segments = _segment(
                samples,
                lambda sample: (
                    sample.intent == 'move'
                    and sample.command_speed <= MAX_TURN_TRANSLATION_COMMAND_MPS
                    and abs(sample.command_yaw) >= MIN_TURN_COMMAND_RADPS
                ),
                sign_sensitive=True,
            )
            for segment in turn_segments:
                if len(segment) < 4 or segment[-1].t - segment[0].t < 0.25:
                    continue
                reach_times = _reach_times_turn(segment, TURN_THRESHOLDS_RAD)
                if not reach_times:
                    continue
                turns.append(
                    TurnEpisode(
                        dataset=path.name,
                        player_id=segment[0].player_id,
                        samples=tuple(segment),
                        first_command_yaw=abs(segment[0].command_yaw),
                        reach_times_sec=reach_times,
                    )
                )
    return translations, turns


def fit_calibration(
    translations: list[TranslationEpisode], turns: list[TurnEpisode]
) -> EtaCalibration:
    speed_ratios: list[float] = []
    reaction_delays: list[float] = []
    accelerations: list[float] = []
    for episode in translations:
        start_t = episode.samples[0].t
        steady = [
            sample
            for sample in episode.samples
            if sample.t - start_t >= 0.35
            and sample.observed_speed >= 0.10
            and sample.observed_speed <= 1.50
            and sample.command_speed >= MIN_TRANSLATION_COMMAND_MPS
        ]
        speed_ratios.extend(sample.observed_speed / sample.command_speed for sample in steady)
        if not episode.started_from_rest:
            continue
        episode_delay = 0.0
        moving = [sample for sample in episode.samples if sample.observed_speed >= 0.15]
        if moving:
            delay = moving[0].t - start_t
            if 0.0 <= delay <= 0.80:
                episode_delay = delay
                reaction_delays.append(delay)
        if steady:
            cruise = statistics.median(sample.observed_speed for sample in steady)
            threshold = 0.80 * cruise
            reached = [sample for sample in episode.samples if sample.observed_speed >= threshold]
            if reached:
                rise_time = reached[0].t - start_t - episode_delay
                if rise_time > 0.05:
                    acceleration = cruise / rise_time
                    if 0.20 <= acceleration <= 5.0:
                        accelerations.append(acceleration)

    yaw_ratios: list[float] = []
    for episode in turns:
        start_t = episode.samples[0].t
        yaw_ratios.extend(
            sample.observed_yaw_rate / abs(sample.command_yaw)
            for sample in episode.samples
            if sample.t - start_t >= 0.20
            and abs(sample.command_yaw) >= MIN_TURN_COMMAND_RADPS
            and 0.05 <= sample.observed_yaw_rate <= 3.0
        )

    return EtaCalibration(
        translation_command_gain=max(0.20, min(1.50, statistics.median(speed_ratios) if speed_ratios else 0.90)),
        reaction_delay_sec=max(0.0, min(0.50, statistics.median(reaction_delays) if reaction_delays else 0.10)),
        acceleration_mps2=max(0.20, min(4.0, statistics.median(accelerations) if accelerations else 1.0)),
        yaw_command_gain=max(0.20, min(2.0, statistics.median(yaw_ratios) if yaw_ratios else 1.0)),
        translation_samples=len(speed_ratios),
        turn_samples=len(yaw_ratios),
        reaction_episodes=len(reaction_delays),
    )


def accelerated_travel_time(distance_m: float, cruise_speed_mps: float, acceleration_mps2: float) -> float:
    """Rest-to-distance time with constant acceleration capped at cruise speed."""

    distance = max(0.0, distance_m)
    cruise = max(0.05, cruise_speed_mps)
    acceleration = max(0.05, acceleration_mps2)
    accelerate_time = cruise / acceleration
    accelerate_distance = 0.5 * acceleration * accelerate_time * accelerate_time
    if distance <= accelerate_distance:
        return math.sqrt(2.0 * distance / acceleration)
    return accelerate_time + (distance - accelerate_distance) / cruise


def translation_predictions(
    calibration: EtaCalibration, distance_m: float, command_speed: float
) -> EtaPrediction:
    command = max(0.05, command_speed)
    cruise = calibration.cruise_speed(command)
    return EtaPrediction(
        baseline_sec=distance_m / command,
        empirical_cruise_sec=distance_m / cruise,
        dynamics_sec=(
            calibration.reaction_delay_sec
            + accelerated_travel_time(distance_m, cruise, calibration.acceleration_mps2)
        ),
    )


def cross_validate(
    translations: list[TranslationEpisode], turns: list[TurnEpisode]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = sorted({episode.dataset for episode in translations} | {episode.dataset for episode in turns})
    aggregate_translation: dict[str, dict[str, list[float]]] = {
        name: {str(threshold): [] for threshold in TRANSLATION_THRESHOLDS_M}
        for name in ('baseline', 'empirical_cruise', 'dynamics')
    }
    aggregate_turn: dict[str, dict[str, list[float]]] = {
        name: {str(threshold): [] for threshold in TURN_THRESHOLDS_RAD}
        for name in ('command_rate', 'empirical_rate')
    }
    folds: list[dict[str, Any]] = []

    for held_out in datasets:
        train_translation = [episode for episode in translations if episode.dataset != held_out]
        train_turn = [episode for episode in turns if episode.dataset != held_out]
        validation_translation = [
            episode for episode in translations if episode.dataset == held_out
        ]
        validation_rest_translation = [
            episode for episode in validation_translation if episode.started_from_rest
        ]
        validation_turn = [episode for episode in turns if episode.dataset == held_out]
        calibration = fit_calibration(train_translation, train_turn)
        fold_translation = {
            name: {str(threshold): [] for threshold in TRANSLATION_THRESHOLDS_M}
            for name in aggregate_translation
        }
        fold_turn = {
            name: {str(threshold): [] for threshold in TURN_THRESHOLDS_RAD}
            for name in aggregate_turn
        }

        for episode in validation_translation:
            for threshold in TRANSLATION_THRESHOLDS_M:
                actual = episode.reach_times_sec.get(str(threshold))
                if actual is None:
                    continue
                prediction = translation_predictions(calibration, threshold, episode.first_command_speed)
                predictions = [
                    ('baseline', prediction.baseline_sec),
                    ('empirical_cruise', prediction.empirical_cruise_sec),
                ]
                if episode.started_from_rest:
                    predictions.append(('dynamics', prediction.dynamics_sec))
                for name, predicted in predictions:
                    error = abs(predicted - actual)
                    fold_translation[name][str(threshold)].append(error)
                    aggregate_translation[name][str(threshold)].append(error)

        for episode in validation_turn:
            command_yaw = max(0.05, episode.first_command_yaw)
            for threshold in TURN_THRESHOLDS_RAD:
                actual = episode.reach_times_sec.get(str(threshold))
                if actual is None:
                    continue
                predicted = {
                    'command_rate': threshold / command_yaw,
                    'empirical_rate': threshold / calibration.yaw_rate(command_yaw),
                }
                for name, value in predicted.items():
                    error = abs(value - actual)
                    fold_turn[name][str(threshold)].append(error)
                    aggregate_turn[name][str(threshold)].append(error)

        folds.append(
            {
                'held_out': held_out,
                'calibration': asdict(calibration),
                'train_translation_episodes': len(train_translation),
                'validation_translation_episodes': len(validation_translation),
                'validation_rest_translation_episodes': len(validation_rest_translation),
                'train_turn_episodes': len(train_turn),
                'validation_turn_episodes': len(validation_turn),
                'translation_absolute_error_sec': {
                    name: {threshold: summarize(values) for threshold, values in rows.items()}
                    for name, rows in fold_translation.items()
                },
                'turn_absolute_error_sec': {
                    name: {threshold: summarize(values) for threshold, values in rows.items()}
                    for name, rows in fold_turn.items()
                },
            }
        )

    aggregate = {
        'translation_absolute_error_sec': {
            name: {threshold: summarize(values) for threshold, values in rows.items()}
            for name, rows in aggregate_translation.items()
        },
        'turn_absolute_error_sec': {
            name: {threshold: summarize(values) for threshold, values in rows.items()}
            for name, rows in aggregate_turn.items()
        },
    }
    return folds, aggregate


def fmt(value: float | int | None) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, int):
        return str(value)
    return f'{value:.3f}'


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        '# Robot ETA Cross-Validation',
        '',
        'This experiment validates locomotion components from recorded poses and executed commands. The logs do not contain the strategy internal target point, so this is not yet arbitrary-target ETA validation.',
        '',
        '## Dataset',
        '',
        f"- Translation episodes: {report['translation_episode_count']}",
        f"- From-rest translation episodes: {report['rest_translation_episode_count']}",
        f"- Turn episodes: {report['turn_episode_count']}",
        '- Cross-validation unit: one complete match JSONL file.',
        '- Cruise models use all straight sustained episodes; the dynamics model uses only episodes detected as starting from rest.',
        '- Translation validation uses actual accumulated path distance.',
        '',
        '## Held-Out Translation ETA Absolute Error',
        '',
        '| Model | Distance | n | Median | p75 | p90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model, thresholds in report['aggregate']['translation_absolute_error_sec'].items():
        for threshold, stats in thresholds.items():
            lines.append(
                f"| {model} | {threshold}m | {stats['count']} | {fmt(stats['median'])}s | {fmt(stats['p75'])}s | {fmt(stats['p90'])}s |"
            )
    lines.extend([
        '',
        '## Held-Out Turn ETA Absolute Error',
        '',
        '| Model | Rotation | n | Median | p75 | p90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for model, thresholds in report['aggregate']['turn_absolute_error_sec'].items():
        for threshold, stats in thresholds.items():
            lines.append(
                f"| {model} | {threshold}rad | {stats['count']} | {fmt(stats['median'])}s | {fmt(stats['p75'])}s | {fmt(stats['p90'])}s |"
            )
    lines.extend(['', '## Fold Calibration', ''])
    for fold in report['folds']:
        calibration = fold['calibration']
        lines.append(f"### Held out: `{fold['held_out']}`")
        lines.append('')
        lines.append(
            '- gain={:.3f}, reaction={:.3f}s, acceleration={:.3f}m/s², yaw_gain={:.3f}; validation translation/rest/turn={}/{}/{}'.format(
                calibration['translation_command_gain'],
                calibration['reaction_delay_sec'],
                calibration['acceleration_mps2'],
                calibration['yaw_command_gain'],
                fold['validation_translation_episodes'],
                fold['validation_rest_translation_episodes'],
                fold['validation_turn_episodes'],
            )
        )
        lines.append('')
    final = report['final_calibration']
    lines.extend([
        '## Full-Data Calibration',
        '',
        f"- Translation command gain: {final['translation_command_gain']:.4f}",
        f"- Start reaction delay: {final['reaction_delay_sec']:.4f}s",
        f"- Acceleration: {final['acceleration_mps2']:.4f}m/s²",
        f"- Yaw command gain: {final['yaw_command_gain']:.4f}",
        '',
        '## Scope Judgment',
        '',
        '- This calibration can support a first straight-line ETA prior and heading-turn cost.',
        '- A deployable target ETA still needs per-tick target coordinates, adjusted avoidance waypoint/path length, current velocity estimation, and arrival/slowdown labels.',
        '- Do not yet use this report alone to decide a hard ball-interception point.',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='dataset/v1')
    parser.add_argument('--output', default='analysis/v1')
    args = parser.parse_args()
    input_dir = Path(args.input)
    paths = sorted(input_dir.glob('*.jsonl'))
    if len(paths) < 2:
        raise SystemExit('ETA cross-validation requires at least two JSONL files')
    translations, turns = extract_episodes(paths)
    if len({episode.dataset for episode in translations}) < 2:
        raise SystemExit('Not enough match files contain usable translation episodes')
    folds, aggregate = cross_validate(translations, turns)
    final_calibration = fit_calibration(translations, turns)
    report = {
        'inputs': [str(path) for path in paths],
        'translation_episode_count': len(translations),
        'rest_translation_episode_count': sum(episode.started_from_rest for episode in translations),
        'turn_episode_count': len(turns),
        'folds': folds,
        'aggregate': aggregate,
        'final_calibration': asdict(final_calibration),
        'limitations': [
            'Internal target poses are not recorded.',
            'Avoidance-adjusted waypoints and planned path length are not recorded.',
            'Validation covers sustained straight travel and in-place turning components only.',
        ],
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'robot_eta_cross_validation.json'
    markdown_path = output_dir / 'robot_eta_cross_validation.md'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_markdown(markdown_path, report)
    print(f'Wrote {json_path}')
    print(f'Wrote {markdown_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
