"""Unified, read-only match dataset recorder for offline model fitting.

The recorder samples the complete public match view and the Agent's executed
commands at a bounded rate. Kick trajectories additionally retain every new
ball observation seen by the 30 Hz control loop. It never changes strategy
state, simulator state, or outgoing robot commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
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


_SCHEMA_VERSION = 2
_FRAME_HZ = 10.0
_MAX_TRAJECTORY_SEC = 8.0
_BALL_STALE_SEC = 1.5
_SETTLE_WINDOW_SEC = 0.35
_SETTLE_MAX_TRAVEL_M = 0.025
_MIN_KICK_TRAVEL_M = 0.05


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
        self._closed = False
        self.path = self._resolve_path(config, logger)
        self._fp = self._open_file(self.path)
        if self._fp is None:
            self._log(
                "Match dataset recorder could not open its output file",
                event="match_data_recorder_failed",
                path=str(self.path),
            )
            return
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

        self._append_kick_ball(now, context)
        self._finish_kick_if_needed(now, context)
        self._detect_new_kicks(now, context, commands)

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
        if self._fp is None:
            return
        self._write(
            {
                "record_type": "session_end",
                "monotonic_sec": _round(ended_at),
                "elapsed_sec": _round(max(0.0, ended_at - self._started_at)),
                "frames": self._frame_id,
                "completed_kicks": self._completed_kicks,
            },
            flush=True,
        )
        try:
            self._fp.close()
        except OSError:
            pass
        self._fp = None

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
        sample.trajectory.append(
            {
                "t_sec": _round(max(0.0, ball.last_seen_at - sample.started_at)),
                "observed_monotonic_sec": _round(ball.last_seen_at),
                "x": _round(ball.x),
                "y": _round(ball.y),
                "confidence": _round(ball.confidence),
                "age_sec": _round(max(0.0, now - ball.last_seen_at)),
            }
        )

    def _finish_kick_if_needed(self, now: float, context: PlayContext) -> None:
        sample = self._kick_sample
        if sample is None:
            return
        if now - sample.started_at >= _MAX_TRAJECTORY_SEC:
            self._finish_kick("timeout", now)
            return
        ball = context.ball
        if ball is None or now - ball.last_seen_at > _BALL_STALE_SEC:
            self._finish_kick("ball_stale", now)
            return
        if self._has_settled(sample):
            self._finish_kick("ball_settled", now)

    @staticmethod
    def _has_settled(sample: _KickSample) -> bool:
        points = sample.trajectory
        if len(points) < 3 or float(points[-1]["t_sec"]) < _SETTLE_WINDOW_SEC:
            return False
        first = points[0]
        last = points[-1]
        total_dx = float(last["x"]) - float(first["x"])
        total_dy = float(last["y"]) - float(first["y"])
        if total_dx * total_dx + total_dy * total_dy < _MIN_KICK_TRAVEL_M**2:
            return False
        window = [
            point
            for point in points
            if float(last["t_sec"]) - float(point["t_sec"]) <= _SETTLE_WINDOW_SEC
        ]
        if len(window) < 2:
            return False
        dx = float(last["x"]) - float(window[0]["x"])
        dy = float(last["y"]) - float(window[0]["y"])
        return dx * dx + dy * dy <= _SETTLE_MAX_TRAVEL_M**2

    def _finish_kick(self, end_reason: str, now: float) -> None:
        sample = self._kick_sample
        self._kick_sample = None
        if sample is None:
            return
        self._completed_kicks += 1
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
            },
            "strategy_tuning": asdict(config.strategy),
            "sampling": {
                "world_frames": "fixed-rate latest public snapshot",
                "kick_trajectory": "every new public ball observation during an own kick",
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
        if self._fp is None:
            return
        record = {"schema_version": _SCHEMA_VERSION, **record}
        try:
            self._fp.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            if flush:
                self._fp.flush()
        except OSError:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None
            self._log(
                "Match dataset recorder disabled after write failure",
                event="match_data_recorder_failed",
            )

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


def _command_record(command: RobotCommand) -> dict[str, object]:
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
