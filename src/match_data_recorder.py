"""Unified, read-only match dataset recorder for offline model fitting.

The recorder samples the complete public match view and the Agent's executed
commands at a bounded rate. Kick trajectories additionally retain every new
ball observation seen by the 30 Hz control loop. It never changes strategy
state, simulator state, or outgoing robot commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Any, Mapping, TextIO

from .soccer_framework import (
    BallState,
    GameControlState,
    KickIntent,
    MoveIntent,
    NoopIntent,
    PlayContext,
    Pose2D,
    RobotCommand,
    RobotRuntimeStatus,
    RobotState,
    SoccerConfig,
)


_SCHEMA_VERSION = 3
_FRAME_HZ = 10.0
_MAX_TRAJECTORY_SEC = 8.0
_MAX_BALL_MOTION_SEC = 12.0
_MAX_ETA_SEC = 20.0
_BALL_STALE_SEC = 1.5
_SETTLE_WINDOW_SEC = 0.35
_SETTLE_MAX_TRAVEL_M = 0.025
_MIN_KICK_TRAVEL_M = 0.05
_MOTION_START_SPEED_MPS = 0.20
_MOTION_HISTORY_SEC = 0.30
_BALL_RADIUS_M = 0.11
_ROBOT_CONTACT_CANDIDATE_M = 0.45
_POST_PROXIMITY_CANDIDATE_M = 0.35
_ETA_TARGET_CHANGE_M = 0.25
_ETA_TARGET_CHANGE_RAD = 0.35


@dataclass
class _KickSample:
    kick_id: int
    started_at: float
    player_id: int
    command: KickIntent
    reason: str
    game: dict[str, object] | None
    player_pose: Pose2D | None
    trajectory: list[dict[str, object]] = field(default_factory=list)
    last_ball_stamp: float = -1.0
    motion_ids: list[int] = field(default_factory=list)


@dataclass
class _BallMotionSample:
    motion_id: int
    started_at: float
    source: str
    game: dict[str, object] | None
    trajectory: list[dict[str, object]] = field(default_factory=list)
    kick_id: int | None = None


@dataclass
class _EtaSample:
    eta_id: int
    started_at: float
    player_id: int
    reason: str
    role: str | None
    game: dict[str, object] | None
    requested_target: Pose2D
    trajectory: list[dict[str, object]] = field(default_factory=list)
    last_pose_stamp: float = -1.0


class MatchDataRecorder:
    """Write one time-aligned JSONL dataset for an entire simulation run."""

    def __init__(self, config: SoccerConfig, logger: Any):
        self._config = config
        self._logger = logger
        self._started_at = time.monotonic()
        self._frame_period = 1.0 / _FRAME_HZ
        self._next_frame_at = 0.0
        self._frame_id = 0
        self._kick_sample: _KickSample | None = None
        self._last_was_kick: dict[int, bool] = {}
        self._next_kick_id = 1
        self._completed_kicks = 0
        self._ball_history: deque[dict[str, object]] = deque(maxlen=24)
        self._last_ball_stamp = -1.0
        self._ball_motion: _BallMotionSample | None = None
        self._next_motion_id = 1
        self._completed_ball_motions = 0
        self._recent_motion_ends: deque[tuple[int, float, str]] = deque(maxlen=16)
        self._last_game_marker: dict[str, object] | None = None
        self._eta_samples: dict[int, _EtaSample] = {}
        self._next_eta_id = 1
        self._completed_eta_samples = 0
        self._closed = False
        self.path = self._resolve_path(config, logger)
        self._fp = self._open_file(self.path)
        self._write_queue: Queue[tuple[dict[str, object], bool] | None] = Queue()
        self._writer_error: str | None = None
        self._writer_thread: threading.Thread | None = None
        if self._fp is None:
            self._log(
                "Match dataset recorder could not open its output file",
                event="match_data_recorder_failed",
                path=str(self.path),
            )
            return
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="match_data_jsonl_writer",
            daemon=True,
        )
        self._writer_thread.start()
        self._write(self._metadata_record(), flush=True)
        self._log(
            "Match dataset path: " + str(self.path),
            event="match_data_recorder_ready",
            path=str(self.path),
            frame_hz=_FRAME_HZ,
        )

    def observe(
        self,
        now: float,
        context: PlayContext,
        commands: dict[int, RobotCommand],
        *,
        roles: Mapping[int, str] | None = None,
        robot_statuses: Mapping[int, RobotRuntimeStatus] | None = None,
    ) -> None:
        """Observe one completed strategy tick without mutating its inputs."""

        if self._closed or self._fp is None:
            return

        self._observe_game_event(now, context)
        self._observe_ball_motion(now, context)
        self._append_kick_ball(now, context)
        self._finish_kick_if_needed(now, context)
        self._detect_new_kicks(now, context, commands)
        self._observe_eta(now, context, commands, roles or {}, robot_statuses or {})

        if now + 1e-9 < self._next_frame_at:
            return
        self._frame_id += 1
        self._next_frame_at = now + self._frame_period
        self._write(
            self._frame_record(
                now,
                context,
                commands,
                roles or {},
                robot_statuses or {},
            )
        )

    def close(self, now: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        ended_at = time.monotonic() if now is None else now
        if self._kick_sample is not None:
            self._finish_kick("runtime_close", ended_at)
        if self._ball_motion is not None:
            self._finish_ball_motion("runtime_close", ended_at, "runtime")
        for player_id in list(self._eta_samples):
            self._finish_eta(player_id, "runtime_close", ended_at)
        if self._fp is None:
            return
        self._write(
            {
                "record_type": "session_end",
                "monotonic_sec": _round(ended_at),
                "elapsed_sec": _round(max(0.0, ended_at - self._started_at)),
                "frames": self._frame_id,
                "completed_kicks": self._completed_kicks,
                "completed_ball_motions": self._completed_ball_motions,
                "completed_eta_samples": self._completed_eta_samples,
            },
            flush=True,
        )
        self.flush()
        self._write_queue.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        try:
            self._fp.close()
        except OSError:
            pass
        self._fp = None
        self._writer_thread = None

    def _observe_game_event(self, now: float, context: PlayContext) -> None:
        current = _game_marker(context.game_state)
        if current is None:
            return
        previous = self._last_game_marker
        self._last_game_marker = current
        if previous is None or _game_signature(previous) == _game_signature(current):
            return
        label = _game_transition_label(previous, current)
        related: tuple[int, float, str] | None = None
        for candidate in reversed(self._recent_motion_ends):
            if 0.0 <= now - candidate[1] <= 2.5:
                related = candidate
                break
        motion_id = related[0] if related is not None else None
        self._write(
            {
                "record_type": "game_event",
                "monotonic_sec": _round(now),
                "event": label,
                "previous": previous,
                "current": current,
                "related_motion_id": motion_id,
                "association": (
                    "nearest_preceding_motion_within_2.5s"
                    if motion_id is not None else None
                ),
            },
            flush=True,
        )
        if motion_id is not None and label.startswith(("goal_", "referee_")):
            self._write(
                {
                    "record_type": "ball_motion_label",
                    "motion_id": motion_id,
                    "monotonic_sec": _round(now),
                    "official_label": label,
                    "association": "nearest_preceding_motion_within_2.5s",
                    "association_delay_sec": _round(now - related[1]),
                },
                flush=True,
            )

    def _observe_ball_motion(self, now: float, context: PlayContext) -> None:
        ball = context.ball
        if ball is None or ball.last_seen_at <= self._last_ball_stamp:
            self._finish_ball_motion_if_needed(now, context)
            return
        self._last_ball_stamp = ball.last_seen_at
        point = self._ball_motion_point(now, context)
        if point is None:
            return
        self._ball_history.append(point)

        sample = self._ball_motion
        if sample is None:
            game = context.game_state
            if game is None or game.state.value != "PLAYING" or game.stopped:
                self._ball_history.clear()
                self._ball_history.append(point)
                return
            start_index = self._motion_start_index()
            if start_index is not None:
                history = list(self._ball_history)[start_index:]
                started_at = float(history[0]["observed_monotonic_sec"])
                kick_id = self._kick_sample.kick_id if self._kick_sample else None
                source = "own_kick" if kick_id is not None else "observed_motion"
                sample = _BallMotionSample(
                    motion_id=self._next_motion_id,
                    started_at=started_at,
                    source=source,
                    game=_game_record(context.game_state, now),
                    kick_id=kick_id,
                )
                self._next_motion_id += 1
                sample.trajectory.extend(
                    _relative_ball_point(item, started_at) for item in history
                )
                self._ball_motion = sample
                if self._kick_sample is not None:
                    self._kick_sample.motion_ids.append(sample.motion_id)
        else:
            sample.trajectory.append(_relative_ball_point(point, sample.started_at))

        self._finish_ball_motion_if_needed(now, context)

    def _motion_start_index(self) -> int | None:
        history = list(self._ball_history)
        if len(history) < 2:
            return None
        last = history[-1]
        last_t = float(last["observed_monotonic_sec"])
        candidate: int | None = None
        for index in range(len(history) - 2, -1, -1):
            first = history[index]
            dt = last_t - float(first["observed_monotonic_sec"])
            if dt <= 0.0:
                continue
            if dt > _MOTION_HISTORY_SEC:
                break
            distance = math.hypot(
                float(last["x"]) - float(first["x"]),
                float(last["y"]) - float(first["y"]),
            )
            if distance >= 0.02 and distance / dt >= _MOTION_START_SPEED_MPS:
                candidate = index
        return candidate

    def _ball_motion_point(
        self,
        now: float,
        context: PlayContext,
    ) -> dict[str, object] | None:
        ball = context.ball
        if ball is None:
            return None
        return {
            "observed_monotonic_sec": _round(ball.last_seen_at),
            "x": _round(ball.x),
            "y": _round(ball.y),
            "confidence": _round(ball.confidence),
            "age_sec": _round(max(0.0, now - ball.last_seen_at)),
            "nearest_teammate": _nearest_robot(ball, context.teammates),
            "nearest_opponent": _nearest_robot(ball, context.opponents),
            "field_evidence": _field_evidence(self._config, ball),
            "game_marker": _game_marker(context.game_state),
        }

    def _finish_ball_motion_if_needed(
        self,
        now: float,
        context: PlayContext,
    ) -> None:
        sample = self._ball_motion
        if sample is None:
            return
        official_reason = _official_ball_end_reason(sample.game, context.game_state)
        if official_reason is not None:
            self._finish_ball_motion(official_reason, now, "official")
            return
        if now - sample.started_at >= _MAX_BALL_MOTION_SEC:
            self._finish_ball_motion("timeout", now, "runtime")
            return
        ball = context.ball
        if ball is None or now - ball.last_seen_at > _BALL_STALE_SEC:
            self._finish_ball_motion("ball_stale", now, "observation")
            return
        if sample.trajectory:
            evidence = sample.trajectory[-1].get("field_evidence")
            if isinstance(evidence, dict) and not evidence.get("whole_ball_in_field", True):
                reason = (
                    "goal_line_crossing_candidate"
                    if evidence.get("inside_goal_mouth")
                    else "boundary_crossing"
                )
                self._finish_ball_motion(reason, now, "geometry")
                return
        if _trajectory_has_settled(sample.trajectory):
            self._finish_ball_motion("natural_stop_candidate", now, "observation")

    def _finish_ball_motion(
        self,
        end_reason: str,
        now: float,
        label_source: str,
    ) -> None:
        sample = self._ball_motion
        self._ball_motion = None
        if sample is None:
            return
        flags = _ball_quality_flags(sample.trajectory)
        eligible = end_reason == "natural_stop_candidate" and not flags
        self._completed_ball_motions += 1
        self._recent_motion_ends.append((sample.motion_id, now, end_reason))
        self._ball_history.clear()
        if sample.trajectory:
            last_point = dict(sample.trajectory[-1])
            last_point.pop("t_sec", None)
            self._ball_history.append(last_point)
        self._write(
            {
                "record_type": "ball_motion",
                "motion_id": sample.motion_id,
                "source": sample.source,
                "kick_id": sample.kick_id,
                "started_monotonic_sec": _round(sample.started_at),
                "duration_sec": _round(max(0.0, now - sample.started_at)),
                "end_reason": end_reason,
                "label_source": label_source,
                "game_at_start": sample.game,
                "quality_flags": flags,
                "free_roll_terminal_candidate": eligible,
                "trajectory": sample.trajectory,
            },
            flush=True,
        )
        self._log(
            f"Ball motion {sample.motion_id} recorded ({end_reason})",
            event="ball_motion_recorded",
            console=False,
            motion_id=sample.motion_id,
            end_reason=end_reason,
            label_source=label_source,
            points=len(sample.trajectory),
            quality_flags=flags,
        )

    def _observe_eta(
        self,
        now: float,
        context: PlayContext,
        commands: Mapping[int, RobotCommand],
        roles: Mapping[int, str],
        robot_statuses: Mapping[int, RobotRuntimeStatus],
    ) -> None:
        for player_id in self._config.player_ids:
            command = commands.get(player_id)
            robot = context.teammates.get(player_id)
            status = robot_statuses.get(player_id)
            interruption = _eta_interruption_reason(
                self._config, context.game_state, player_id, status
            )
            if interruption is not None:
                self._finish_eta(player_id, interruption, now)
                continue
            trace = command.motion_target if command is not None else None
            if trace is None or robot is None or robot.pose is None:
                if player_id in self._eta_samples:
                    reason = _eta_command_end_reason(command)
                    self._finish_eta(player_id, reason, now)
                continue

            sample = self._eta_samples.get(player_id)
            if sample is not None and _target_changed(
                sample.requested_target, trace.requested_target
            ):
                self._finish_eta(player_id, "target_changed", now)
                sample = None

            if sample is None:
                if trace.phase == "arrived":
                    continue
                sample = _EtaSample(
                    eta_id=self._next_eta_id,
                    started_at=now,
                    player_id=player_id,
                    reason=command.reason,
                    role=str(roles[player_id]) if player_id in roles else None,
                    game=_game_record(context.game_state, now),
                    requested_target=trace.requested_target,
                )
                self._next_eta_id += 1
                self._eta_samples[player_id] = sample

            if robot.last_seen_at > sample.last_pose_stamp:
                sample.last_pose_stamp = robot.last_seen_at
                sample.trajectory.append(
                    _eta_point(
                        sample,
                        now,
                        robot,
                        command,
                        str(roles[player_id]) if player_id in roles else None,
                        status,
                        context.game_state,
                    )
                )

            if trace.phase == "arrived":
                self._finish_eta(player_id, "arrived", now)
            elif now - sample.started_at >= _MAX_ETA_SEC:
                self._finish_eta(player_id, "timeout", now)

    def _finish_eta(self, player_id: int, end_reason: str, now: float) -> None:
        sample = self._eta_samples.pop(player_id, None)
        if sample is None:
            return
        flags = _eta_quality_flags(sample.trajectory)
        arrived = end_reason == "arrived"
        self._completed_eta_samples += 1
        self._write(
            {
                "record_type": "robot_eta",
                "eta_id": sample.eta_id,
                "team_id": self._config.team_id,
                "player_id": sample.player_id,
                "ready_slot": self._config.ready_slot_for_player(player_id).value,
                "started_monotonic_sec": _round(sample.started_at),
                "duration_sec": _round(max(0.0, now - sample.started_at)),
                "end_reason": end_reason,
                "arrived": arrived,
                "censored": not arrived,
                "reason_at_start": sample.reason,
                "role_at_start": sample.role,
                "game_at_start": sample.game,
                "requested_target_at_start": _pose_record(sample.requested_target),
                "quality_flags": flags,
                "eta_training_candidate": (
                    arrived
                    and len(sample.trajectory) >= 3
                    and "stale_pose_sample" not in flags
                ),
                "trajectory": sample.trajectory,
            },
            flush=True,
        )
        self._log(
            f"Robot ETA {sample.eta_id} recorded ({end_reason})",
            event="robot_eta_recorded",
            console=False,
            eta_id=sample.eta_id,
            player_id=player_id,
            end_reason=end_reason,
            points=len(sample.trajectory),
            quality_flags=flags,
        )

    def _detect_new_kicks(
        self,
        now: float,
        context: PlayContext,
        commands: Mapping[int, RobotCommand],
    ) -> None:
        for player_id, command in sorted(commands.items()):
            is_kick = isinstance(command.intent, KickIntent)
            started = is_kick and not self._last_was_kick.get(player_id, False)
            self._last_was_kick[player_id] = is_kick
            if not started:
                continue
            if self._kick_sample is not None:
                self._finish_kick("next_kick", now)
            if self._ball_motion is not None:
                self._finish_ball_motion("own_kick_command", now, "command")
            assert isinstance(command.intent, KickIntent)
            robot = context.teammates.get(player_id)
            self._kick_sample = _KickSample(
                kick_id=self._next_kick_id,
                started_at=now,
                player_id=player_id,
                command=command.intent,
                reason=command.reason,
                game=_game_record(context.game_state, now),
                player_pose=robot.pose if robot is not None else None,
            )
            self._next_kick_id += 1
            self._append_kick_ball(now, context)

    def _append_kick_ball(self, now: float, context: PlayContext) -> None:
        sample = self._kick_sample
        ball = context.ball
        if sample is None or ball is None or ball.last_seen_at <= sample.last_ball_stamp:
            return
        sample.last_ball_stamp = ball.last_seen_at
        point = self._ball_motion_point(now, context)
        if point is not None:
            sample.trajectory.append(_relative_ball_point(point, sample.started_at))

    def _finish_kick_if_needed(self, now: float, context: PlayContext) -> None:
        sample = self._kick_sample
        if sample is None:
            return
        official_reason = _official_ball_end_reason(sample.game, context.game_state)
        if official_reason is not None:
            self._finish_kick(official_reason, now)
            return
        if now - sample.started_at >= _MAX_TRAJECTORY_SEC:
            self._finish_kick("timeout", now)
            return
        ball = context.ball
        if ball is None or now - ball.last_seen_at > _BALL_STALE_SEC:
            self._finish_kick("ball_stale", now)
            return
        if sample.trajectory:
            evidence = sample.trajectory[-1].get("field_evidence")
            if isinstance(evidence, dict) and not evidence.get("whole_ball_in_field", True):
                reason = (
                    "goal_line_crossing_candidate"
                    if evidence.get("inside_goal_mouth")
                    else "boundary_crossing"
                )
                self._finish_kick(reason, now)
                return
        if self._has_settled(sample):
            self._finish_kick("natural_stop_candidate", now)

    @staticmethod
    def _has_settled(sample: _KickSample) -> bool:
        return _trajectory_has_settled(sample.trajectory)

    def _finish_kick(self, end_reason: str, now: float) -> None:
        sample = self._kick_sample
        self._kick_sample = None
        if sample is None:
            return
        self._completed_kicks += 1
        quality_flags = _ball_quality_flags(sample.trajectory)
        self._write(
            {
                "record_type": "kick",
                "kick_id": sample.kick_id,
                "team_id": self._config.team_id,
                "player_id": sample.player_id,
                "ready_slot": self._config.ready_slot_for_player(sample.player_id).value,
                "started_monotonic_sec": _round(sample.started_at),
                "duration_sec": _round(max(0.0, now - sample.started_at)),
                "end_reason": end_reason,
                "reason": sample.reason,
                "motion_ids": sample.motion_ids,
                "quality_flags": quality_flags,
                "free_roll_terminal_candidate": (
                    end_reason == "natural_stop_candidate" and not quality_flags
                ),
                "game_at_start": sample.game,
                "kick": {
                    "direction_rad": _round(sample.command.direction),
                    "power": _round(sample.command.power),
                    "ball_x_robot": _round(sample.command.ball_x),
                    "ball_y_robot": _round(sample.command.ball_y),
                },
                "kicker_pose": _pose_record(sample.player_pose),
                "trajectory": sample.trajectory,
            },
            flush=True,
        )
        self._log(
            f"Match kick sample {sample.kick_id} recorded ({end_reason})",
            event="match_kick_recorded",
            console=False,
            kick_id=sample.kick_id,
            player_id=sample.player_id,
            end_reason=end_reason,
            trajectory_points=len(sample.trajectory),
        )

    def _metadata_record(self) -> dict[str, object]:
        config = self._config
        return {
            "record_type": "metadata",
            "created_monotonic_sec": _round(self._started_at),
            "team_id": config.team_id,
            "opponent_team_id": config.opponent_team_id(),
            "control_hz": config.control_hz,
            "frame_hz": _FRAME_HZ,
            "robots": [
                {
                    "player_id": player_id,
                    "name": name,
                    "ready_slot": config.ready_slot_for_player(player_id).value,
                }
                for player_id, name in enumerate(config.robot_names, start=1)
            ],
            "opponent_robot_names": list(config.opponent_robot_names),
            "field": {
                "length": config.field_length,
                "width": config.field_width,
                "goal_width": config.goal_width,
                "center_circle_radius": config.center_circle_radius,
                "penalty_area_length": config.penalty_area_length,
                "penalty_area_width": config.penalty_area_width,
                "goal_area_length": config.goal_area_length,
                "goal_area_width": config.goal_area_width,
                "ball_radius": _BALL_RADIUS_M,
            },
            "label_policy": {
                "official": "GameControl state, set play, stopped, score and penalty",
                "geometry": "whole-ball boundary tests from public field dimensions",
                "candidate_only": "robot proximity, post proximity and direction change",
                "robot_contact_candidate_m": _ROBOT_CONTACT_CANDIDATE_M,
                "post_proximity_candidate_m": _POST_PROXIMITY_CANDIDATE_M,
                "official_event_association_sec": 2.5,
            },
            "strategy_tuning": asdict(config.strategy),
            "sampling": {
                "world_frames": "fixed-rate latest public snapshot",
                "kick_trajectory": "every new public ball observation during an own kick",
                "ball_motion": "every new public ball observation in any detected motion segment",
                "robot_eta": "every new own-robot pose while a traced navigation target is active",
            },
        }

    def _frame_record(
        self,
        now: float,
        context: PlayContext,
        commands: Mapping[int, RobotCommand],
        roles: Mapping[int, str],
        robot_statuses: Mapping[int, RobotRuntimeStatus],
    ) -> dict[str, object]:
        return {
            "record_type": "frame",
            "frame_id": self._frame_id,
            "monotonic_sec": _round(now),
            "elapsed_sec": _round(max(0.0, now - self._started_at)),
            "game": _game_record(context.game_state, now),
            "ball": _ball_record(context.ball, now),
            "teammates": {
                str(player_id): _robot_record(robot, now)
                for player_id, robot in sorted(context.teammates.items())
            },
            "opponents": {
                str(player_id): _robot_record(robot, now)
                for player_id, robot in sorted(context.opponents.items())
            },
            "commands": {
                str(player_id): _command_record(command)
                for player_id, command in sorted(commands.items())
            },
            "roles": {
                str(player_id): str(role)
                for player_id, role in sorted(roles.items())
            },
            "robot_statuses": {
                str(player_id): _status_record(status, now)
                for player_id, status in sorted(robot_statuses.items())
            },
        }

    def _write(self, record: dict[str, object], *, flush: bool = False) -> None:
        if self._fp is None or self._writer_error is not None:
            return
        self._write_queue.put_nowait(
            ({"schema_version": _SCHEMA_VERSION, **record}, flush)
        )

    def flush(self) -> None:
        """Wait until queued records are durable; never called by the control loop."""
        if self._writer_thread is None:
            return
        self._write_queue.join()

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                record, flush = item
                if self._fp is None:
                    continue
                self._fp.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                if flush:
                    self._fp.flush()
            except Exception as exc:
                self._writer_error = f"{exc.__class__.__name__}: {exc}"
            finally:
                self._write_queue.task_done()

    @staticmethod
    def _resolve_path(config: SoccerConfig, logger: Any) -> Path:
        structured_path = getattr(logger, "path", None)
        if isinstance(structured_path, Path):
            return structured_path.with_name(
                f"match_dataset.team{config.team_id}.jsonl"
            )
        return Path(
            f"/tmp/booster_agent/soccer_logs/match_dataset.team{config.team_id}.jsonl"
        )

    @staticmethod
    def _open_file(path: Path) -> TextIO | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open("a", encoding="utf-8", buffering=64 * 1024)
        except OSError:
            return None

    def _log(self, message: str, **fields: object) -> None:
        info = getattr(self._logger, "info", None)
        if callable(info):
            info(message, **fields)


def _relative_ball_point(
    point: Mapping[str, object],
    started_at: float,
) -> dict[str, object]:
    record = dict(point)
    observed_at = float(record["observed_monotonic_sec"])
    record["t_sec"] = _round(max(0.0, observed_at - started_at))
    return record


def _nearest_robot(
    ball: BallState,
    robots: Mapping[int, RobotState],
) -> dict[str, object] | None:
    nearest: tuple[float, int] | None = None
    for player_id, robot in robots.items():
        if robot.pose is None:
            continue
        distance = math.hypot(ball.x - robot.pose.x, ball.y - robot.pose.y)
        if nearest is None or distance < nearest[0]:
            nearest = (distance, player_id)
    if nearest is None:
        return None
    return {"player_id": nearest[1], "distance_m": _round(nearest[0])}


def _field_evidence(config: SoccerConfig, ball: BallState) -> dict[str, object]:
    half_length = config.field_length / 2.0
    half_width = config.field_width / 2.0
    half_goal = config.goal_width / 2.0
    post_distance = min(
        math.hypot(ball.x - goal_x, ball.y - post_y)
        for goal_x in (-half_length, half_length)
        for post_y in (-half_goal, half_goal)
    )
    return {
        "whole_ball_in_field": (
            abs(ball.x) <= half_length + _BALL_RADIUS_M
            and abs(ball.y) <= half_width + _BALL_RADIUS_M
        ),
        "whole_ball_crossed_goal_line": abs(ball.x) > half_length + _BALL_RADIUS_M,
        "whole_ball_crossed_sideline": abs(ball.y) > half_width + _BALL_RADIUS_M,
        "inside_goal_mouth": abs(ball.y) + _BALL_RADIUS_M < half_goal,
        "x_boundary_margin_m": _round(half_length + _BALL_RADIUS_M - abs(ball.x)),
        "y_boundary_margin_m": _round(half_width + _BALL_RADIUS_M - abs(ball.y)),
        "nearest_post_center_distance_m": _round(post_distance),
    }


def _game_marker(game: GameControlState | None) -> dict[str, object] | None:
    if game is None:
        return None
    return {
        "packet_number": game.packet_number,
        "state": game.state.value,
        "set_play": game.set_play.value,
        "stopped": game.stopped,
        "kicking_team": game.kicking_team,
        "scores": {
            str(team.team_number): team.score for team in game.teams
        },
    }


def _game_signature(marker: Mapping[str, object]) -> tuple[object, ...]:
    scores = marker.get("scores")
    score_items = tuple(sorted(scores.items())) if isinstance(scores, dict) else ()
    return (
        marker.get("state"),
        marker.get("set_play"),
        marker.get("stopped"),
        marker.get("kicking_team"),
        score_items,
    )


def _game_transition_label(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> str:
    previous_scores = previous.get("scores")
    current_scores = current.get("scores")
    if isinstance(previous_scores, dict) and isinstance(current_scores, dict):
        for team_id, score in current_scores.items():
            old_score = previous_scores.get(team_id, score)
            if isinstance(score, int) and isinstance(old_score, int) and score > old_score:
                return "goal_confirmed_team_" + str(team_id)
    set_play = str(current.get("set_play", "NONE"))
    if bool(current.get("stopped")) and not bool(previous.get("stopped")):
        return "referee_" + (set_play.lower() if set_play != "NONE" else "stopped")
    if set_play != str(previous.get("set_play", "NONE")) and set_play != "NONE":
        return "referee_" + set_play.lower()
    state = str(current.get("state", ""))
    if state != str(previous.get("state", "")):
        return "referee_state_" + state.lower()
    if current.get("kicking_team") != previous.get("kicking_team"):
        return "kicking_team_changed"
    return "game_state_changed"


def _scores_from_record(record: Mapping[str, object] | None) -> dict[int, int]:
    if record is None:
        return {}
    result: dict[int, int] = {}
    teams = record.get("teams")
    if not isinstance(teams, list):
        return result
    for team in teams:
        if not isinstance(team, dict):
            continue
        number = team.get("team_number")
        score = team.get("score")
        if isinstance(number, int) and isinstance(score, int):
            result[number] = score
    return result


def _official_ball_end_reason(
    start: Mapping[str, object] | None,
    current: GameControlState | None,
) -> str | None:
    if start is None or current is None:
        return None
    start_scores = _scores_from_record(start)
    if any(team.score > start_scores.get(team.team_number, team.score) for team in current.teams):
        return "goal_confirmed"
    start_set_play = str(start.get("set_play", "NONE"))
    current_set_play = current.set_play.value
    if current.stopped and not bool(start.get("stopped", False)):
        if current_set_play != "NONE":
            return "referee_" + current_set_play.lower()
        return "referee_stopped"
    if current_set_play != start_set_play and current_set_play != "NONE":
        return "referee_" + current_set_play.lower()
    start_state = str(start.get("state", ""))
    if start_state == "PLAYING" and current.state.value != "PLAYING":
        return "referee_state_" + current.state.value.lower()
    return None


def _trajectory_has_settled(points: list[dict[str, object]]) -> bool:
    if len(points) < 3 or float(points[-1]["t_sec"]) < _SETTLE_WINDOW_SEC:
        return False
    first = points[0]
    last = points[-1]
    total_distance = math.hypot(
        float(last["x"]) - float(first["x"]),
        float(last["y"]) - float(first["y"]),
    )
    if total_distance < _MIN_KICK_TRAVEL_M:
        return False
    window = [
        point for point in points
        if float(last["t_sec"]) - float(point["t_sec"]) <= _SETTLE_WINDOW_SEC
    ]
    if len(window) < 2:
        return False
    travel = math.hypot(
        float(last["x"]) - float(window[0]["x"]),
        float(last["y"]) - float(window[0]["y"]),
    )
    return travel <= _SETTLE_MAX_TRAVEL_M


def _ball_quality_flags(points: list[dict[str, object]]) -> list[str]:
    flags: list[str] = []
    contact = False
    boundary = False
    post = False
    low_confidence = False
    for point in points:
        t_sec = float(point.get("t_sec", 0.0))
        if t_sec >= 0.25:
            for key in ("nearest_teammate", "nearest_opponent"):
                nearest = point.get(key)
                if isinstance(nearest, dict):
                    distance = nearest.get("distance_m")
                    if isinstance(distance, (int, float)) and distance <= _ROBOT_CONTACT_CANDIDATE_M:
                        contact = True
        evidence = point.get("field_evidence")
        if isinstance(evidence, dict):
            if not evidence.get("whole_ball_in_field", True):
                boundary = True
            distance = evidence.get("nearest_post_center_distance_m")
            if isinstance(distance, (int, float)) and distance <= _POST_PROXIMITY_CANDIDATE_M:
                post = True
        confidence = point.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < 0.8:
            low_confidence = True
    if contact:
        flags.append("robot_contact_candidate")
    if boundary:
        flags.append("boundary_crossing")
    if post:
        flags.append("goal_post_proximity_candidate")
    if _has_sharp_turn(points):
        flags.append("direction_change_candidate")
    if _has_speed_jump(points):
        flags.append("speed_jump_candidate")
    if low_confidence:
        flags.append("low_ball_confidence")
    return flags


def _has_sharp_turn(points: list[dict[str, object]]) -> bool:
    vectors: list[tuple[float, float]] = []
    anchor = 0
    for index in range(1, len(points)):
        dx = float(points[index]["x"]) - float(points[anchor]["x"])
        dy = float(points[index]["y"]) - float(points[anchor]["y"])
        if math.hypot(dx, dy) >= 0.08:
            vectors.append((dx, dy))
            anchor = index
    for first, second in zip(vectors, vectors[1:]):
        denominator = math.hypot(*first) * math.hypot(*second)
        if denominator <= 1e-9:
            continue
        cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
        if math.acos(cosine) >= math.radians(35.0):
            return True
    return False


def _has_speed_jump(points: list[dict[str, object]]) -> bool:
    speeds: list[tuple[float, float]] = []
    anchor = 0
    for index in range(1, len(points)):
        dt = float(points[index]["t_sec"]) - float(points[anchor]["t_sec"])
        if dt <= 0.0:
            continue
        dx = float(points[index]["x"]) - float(points[anchor]["x"])
        dy = float(points[index]["y"]) - float(points[anchor]["y"])
        distance = math.hypot(dx, dy)
        if distance >= 0.05 or dt >= 0.15:
            speeds.append((float(points[index]["t_sec"]), distance / dt))
            anchor = index
    for first, second in zip(speeds, speeds[1:]):
        if second[0] < 0.25:
            continue
        if second[1] > first[1] + 0.60 and second[1] > first[1] * 1.35:
            return True
    return False


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _target_changed(first: Pose2D, current: Pose2D) -> bool:
    return (
        math.hypot(current.x - first.x, current.y - first.y) > _ETA_TARGET_CHANGE_M
        or abs(_normalize_angle(current.theta - first.theta)) > _ETA_TARGET_CHANGE_RAD
    )


def _eta_interruption_reason(
    config: SoccerConfig,
    game: GameControlState | None,
    player_id: int,
    status: RobotRuntimeStatus | None,
) -> str | None:
    if status is not None and status.fall_down_state not in {None, "normal"}:
        return "robot_" + str(status.fall_down_state)
    if game is None:
        return None
    player = game.get_player_state(config.team_id, player_id)
    if player is not None and (
        player.penalty.value != "NONE" or player.secs_till_unpenalised > 0
    ):
        return "penalized_" + player.penalty.value.lower()
    if game.stopped:
        return "game_stopped"
    return None


def _eta_command_end_reason(command: RobotCommand | None) -> str:
    if command is None:
        return "command_missing"
    if isinstance(command.intent, KickIntent):
        return "kick_command"
    if isinstance(command.intent, NoopIntent):
        return "noop_command"
    return "navigation_ended"


def _eta_point(
    sample: _EtaSample,
    now: float,
    robot: RobotState,
    command: RobotCommand,
    role: str | None,
    status: RobotRuntimeStatus | None,
    game: GameControlState | None,
) -> dict[str, object]:
    assert robot.pose is not None
    assert command.motion_target is not None
    pose = robot.pose
    trace = command.motion_target
    dx = trace.requested_target.x - pose.x
    dy = trace.requested_target.y - pose.y
    line_heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else pose.theta
    return {
        "t_sec": _round(max(0.0, robot.last_seen_at - sample.started_at)),
        "observed_monotonic_sec": _round(robot.last_seen_at),
        "age_sec": _round(max(0.0, now - robot.last_seen_at)),
        "pose": _pose_record(pose),
        "requested_target": _pose_record(trace.requested_target),
        "control_target": _pose_record(trace.control_target),
        "arrive_distance_m": _round(trace.arrive_distance),
        "phase": trace.phase,
        "avoidance_applied": trace.avoidance_applied,
        "distance_to_requested_m": _round(math.hypot(dx, dy)),
        "heading_to_path_error_rad": _round(_normalize_angle(line_heading - pose.theta)),
        "final_heading_error_rad": _round(_normalize_angle(trace.requested_target.theta - pose.theta)),
        "command": _command_record(command, include_motion=False),
        "reason": command.reason,
        "role": role,
        "robot_status": _status_record(status, now) if status is not None else None,
        "game_marker": _game_marker(game),
    }


def _eta_quality_flags(points: list[dict[str, object]]) -> list[str]:
    flags: list[str] = []
    if any(bool(point.get("avoidance_applied")) for point in points):
        flags.append("avoidance_applied")
    if points:
        first_target = points[0].get("requested_target")
        if isinstance(first_target, dict):
            fx, fy = first_target.get("x"), first_target.get("y")
            if isinstance(fx, (int, float)) and isinstance(fy, (int, float)):
                for point in points[1:]:
                    target = point.get("requested_target")
                    if not isinstance(target, dict):
                        continue
                    x, y = target.get("x"), target.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        if math.hypot(x - fx, y - fy) > 0.10:
                            flags.append("target_revised")
                            break
    if any(float(point.get("age_sec", 0.0)) > 0.15 for point in points):
        flags.append("stale_pose_sample")
    return flags


def _round(value: float) -> float:
    return round(value, 6)


def _pose_record(pose: Pose2D | None) -> dict[str, float] | None:
    if pose is None:
        return None
    return {"x": _round(pose.x), "y": _round(pose.y), "theta": _round(pose.theta)}


def _robot_record(robot: RobotState, now: float) -> dict[str, object]:
    return {
        "player_id": robot.player_id,
        "pose": _pose_record(robot.pose),
        "observed_monotonic_sec": _round(robot.last_seen_at),
        "age_sec": (
            _round(max(0.0, now - robot.last_seen_at))
            if robot.last_seen_at > 0.0
            else None
        ),
    }


def _ball_record(ball: BallState | None, now: float) -> dict[str, object] | None:
    if ball is None:
        return None
    return {
        "x": _round(ball.x),
        "y": _round(ball.y),
        "confidence": _round(ball.confidence),
        "observed_monotonic_sec": _round(ball.last_seen_at),
        "age_sec": _round(max(0.0, now - ball.last_seen_at)),
    }


def _game_record(game: GameControlState | None, now: float) -> dict[str, object] | None:
    if game is None:
        return None
    return {
        "packet_number": game.packet_number,
        "state": game.state.value,
        "game_phase": game.game_phase.value,
        "set_play": game.set_play.value,
        "first_half": game.first_half,
        "stopped": game.stopped,
        "kicking_team": game.kicking_team,
        "secs_remaining": game.secs_remaining,
        "secondary_time": game.secondary_time,
        "observed_monotonic_sec": _round(game.last_seen_at),
        "age_sec": (
            _round(max(0.0, now - game.last_seen_at))
            if game.last_seen_at > 0.0
            else None
        ),
        "teams": [
            {
                "team_number": team.team_number,
                "score": team.score,
                "goalkeeper": team.goalkeeper,
                "penalty_shot": team.penalty_shot,
                "single_shots": team.single_shots,
                "message_budget": team.message_budget,
                "players": [
                    {
                        "player_id": player_id,
                        "penalty": player.penalty.value,
                        "secs_till_unpenalised": player.secs_till_unpenalised,
                        "warnings": player.warnings,
                        "cautions": player.cautions,
                    }
                    for player_id, player in enumerate(team.players, start=1)
                ],
            }
            for team in game.teams
        ],
    }


def _command_record(
    command: RobotCommand,
    *,
    include_motion: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {"reason": command.reason}
    intent = command.intent
    if isinstance(intent, MoveIntent):
        record.update(
            intent="move",
            vx=_round(intent.vx),
            vy=_round(intent.vy),
            vyaw=_round(intent.vyaw),
        )
    elif isinstance(intent, KickIntent):
        record.update(
            intent="kick",
            direction_rad=_round(intent.direction),
            power=_round(intent.power),
            ball_x_robot=_round(intent.ball_x),
            ball_y_robot=_round(intent.ball_y),
        )
    elif isinstance(intent, NoopIntent):
        record["intent"] = "noop"
    else:
        record["intent"] = "stop"
    if include_motion and command.motion_target is not None:
        trace = command.motion_target
        record["motion_target"] = {
            "requested": _pose_record(trace.requested_target),
            "control": _pose_record(trace.control_target),
            "arrive_distance_m": _round(trace.arrive_distance),
            "phase": trace.phase,
            "avoidance_applied": trace.avoidance_applied,
        }
    return record


def _status_record(status: RobotRuntimeStatus, now: float) -> dict[str, object]:
    return {
        "mode": status.mode,
        "fall_down_state": status.fall_down_state,
        "fall_down_recoverable": status.fall_down_recoverable,
        "updated_monotonic_sec": _round(status.updated_at),
        "age_sec": (
            _round(max(0.0, now - status.updated_at))
            if status.updated_at > 0.0
            else None
        ),
    }
