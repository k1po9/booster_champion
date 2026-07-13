"""Development-only recorder for empirical ball-roll calibration.

The recorder observes only the strategy's own :class:`KickIntent` commands and
the public team-view ball positions already present in :class:`PlayContext`.
It never sends a command or changes the default strategy. One completed kick
is written as one JSON object to a JSONL file for offline trajectory fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, TextIO

from .soccer_framework import KickIntent, PlayContext, Pose2D, RobotCommand, SoccerConfig


_MAX_TRAJECTORY_SEC = 8.0
_BALL_STALE_SEC = 1.5
_SETTLE_WINDOW_SEC = 0.35
_SETTLE_MAX_TRAVEL_M = 0.025
_MIN_KICK_TRAVEL_M = 0.05


@dataclass
class _KickSample:
    """One kick plus the deduplicated public ball observations after it."""

    kick_id: int
    started_at: float
    player_id: int
    command: KickIntent
    game: dict[str, object]
    player_pose: Pose2D | None
    trajectory: list[dict[str, float]] = field(default_factory=list)
    last_ball_stamp: float = -1.0


class KickDataRecorder:
    """Record real default-strategy kicks without changing any command path.

    A sample starts only when a player transitions into ``KickIntent``. This
    prevents one automatic kick held across many ticks becoming many samples.
    The sample ends when the ball settles, becomes stale, is superseded by the
    next own kick, reaches the time cap, or the runtime closes.
    """

    def __init__(self, config: SoccerConfig, logger: Any):
        self._config = config
        self._logger = logger
        self._sample: _KickSample | None = None
        self._last_was_kick: dict[int, bool] = {}
        self._next_kick_id = 1
        self._closed = False
        self.path = self._resolve_path(config, logger)
        self._fp = self._open_file(self.path)
        if self._fp is not None:
            self._log(
                "Kick calibration data path: " + str(self.path),
                event="kick_data_recorder_ready",
                path=str(self.path),
            )

    def observe(
        self,
        now: float,
        context: PlayContext,
        commands: dict[int, RobotCommand],
    ) -> None:
        """Observe a completed strategy tick; never mutates commands or context."""

        if self._closed:
            return
        self._append_ball_sample(now, context)
        self._finish_if_needed(now, context)

        for player_id, command in sorted(commands.items()):
            is_kick = isinstance(command.intent, KickIntent)
            started = is_kick and not self._last_was_kick.get(player_id, False)
            self._last_was_kick[player_id] = is_kick
            if not started:
                continue
            if self._sample is not None:
                self._finish("next_kick", now)
            assert isinstance(command.intent, KickIntent)
            self._start(now, context, player_id, command.intent)

    def close(self, now: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sample is not None:
            self._finish("runtime_close", time.monotonic() if now is None else now)
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None

    def _start(
        self,
        now: float,
        context: PlayContext,
        player_id: int,
        command: KickIntent,
    ) -> None:
        robot = context.teammates.get(player_id)
        self._sample = _KickSample(
            kick_id=self._next_kick_id,
            started_at=now,
            player_id=player_id,
            command=command,
            game=self._game_record(context),
            player_pose=robot.pose if robot is not None else None,
        )
        self._next_kick_id += 1
        self._append_ball_sample(now, context)

    def _append_ball_sample(self, now: float, context: PlayContext) -> None:
        sample = self._sample
        ball = context.ball
        if sample is None or ball is None or ball.last_seen_at <= sample.last_ball_stamp:
            return
        sample.last_ball_stamp = ball.last_seen_at
        sample.trajectory.append(
            {
                "t_sec": round(max(0.0, ball.last_seen_at - sample.started_at), 6),
                "x": round(ball.x, 6),
                "y": round(ball.y, 6),
                "age_sec": round(max(0.0, now - ball.last_seen_at), 6),
            }
        )

    def _finish_if_needed(self, now: float, context: PlayContext) -> None:
        sample = self._sample
        if sample is None:
            return
        if now - sample.started_at >= _MAX_TRAJECTORY_SEC:
            self._finish("timeout", now)
            return
        ball = context.ball
        if ball is None or now - ball.last_seen_at > _BALL_STALE_SEC:
            self._finish("ball_stale", now)
            return
        if self._has_settled(sample):
            self._finish("ball_settled", now)

    @staticmethod
    def _has_settled(sample: _KickSample) -> bool:
        points = sample.trajectory
        if len(points) < 3 or points[-1]["t_sec"] < _SETTLE_WINDOW_SEC:
            return False
        last = points[-1]
        initial = points[0]
        total_dx = last["x"] - initial["x"]
        total_dy = last["y"] - initial["y"]
        if total_dx * total_dx + total_dy * total_dy < _MIN_KICK_TRAVEL_M * _MIN_KICK_TRAVEL_M:
            return False
        window = [
            point
            for point in points
            if last["t_sec"] - point["t_sec"] <= _SETTLE_WINDOW_SEC
        ]
        if len(window) < 2:
            return False
        start = window[0]
        dx = last["x"] - start["x"]
        dy = last["y"] - start["y"]
        return dx * dx + dy * dy <= _SETTLE_MAX_TRAVEL_M * _SETTLE_MAX_TRAVEL_M

    def _finish(self, reason: str, now: float) -> None:
        sample = self._sample
        self._sample = None
        if sample is None or self._fp is None:
            return
        record = {
            "schema_version": 1,
            "kick_id": sample.kick_id,
            "team_id": self._config.team_id,
            "player_id": sample.player_id,
            "ready_slot": self._config.ready_slot_for_player(sample.player_id).value,
            "started_monotonic_sec": round(sample.started_at, 6),
            "duration_sec": round(max(0.0, now - sample.started_at), 6),
            "end_reason": reason,
            "game": sample.game,
            "kick": {
                "direction_rad": round(sample.command.direction, 6),
                "power": round(sample.command.power, 6),
                "ball_x_robot": round(sample.command.ball_x, 6),
                "ball_y_robot": round(sample.command.ball_y, 6),
            },
            "kicker_pose": self._pose_record(sample.player_pose),
            "trajectory": sample.trajectory,
        }
        try:
            self._fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._fp.flush()
        except OSError:
            self._fp = None
            self._log(
                "Kick calibration recorder disabled after write failure",
                event="kick_data_recorder_failed",
            )
            return
        self._log(
            f"Kick calibration sample {sample.kick_id} recorded ({reason})",
            event="kick_data_recorded",
            console=False,
            kick_id=sample.kick_id,
            player_id=sample.player_id,
            end_reason=reason,
            trajectory_points=len(sample.trajectory),
        )

    @staticmethod
    def _resolve_path(config: SoccerConfig, logger: Any) -> Path:
        structured_path = getattr(logger, "path", None)
        if isinstance(structured_path, Path):
            return structured_path.with_name(f"kick_samples.team{config.team_id}.jsonl")
        return Path("/tmp/booster_agent/soccer_logs/kick_samples.jsonl")

    @staticmethod
    def _open_file(path: Path) -> TextIO | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open("a", encoding="utf-8", buffering=1)
        except OSError:
            return None

    @staticmethod
    def _pose_record(pose: Pose2D | None) -> dict[str, float] | None:
        if pose is None:
            return None
        return {
            "x": round(pose.x, 6),
            "y": round(pose.y, 6),
            "theta": round(pose.theta, 6),
        }

    @staticmethod
    def _game_record(context: PlayContext) -> dict[str, object]:
        game = context.game_state
        if game is None:
            return {}
        return {
            "state": game.state.value,
            "set_play": game.set_play.value,
            "kicking_team": game.kicking_team,
            "stopped": game.stopped,
            "secs_remaining": game.secs_remaining,
        }

    def _log(self, message: str, **fields: object) -> None:
        info = getattr(self._logger, "info", None)
        if callable(info):
            info(message, **fields)
