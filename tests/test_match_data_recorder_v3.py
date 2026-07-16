import json
from pathlib import Path
import tempfile
import unittest

from src.match_data_recorder import MatchDataRecorder
from src.tactics.motion import MotionController
from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    MotionTargetTrace,
    MoveIntent,
    PlayContext,
    Pose2D,
    RobotCommand,
    RobotState,
    SoccerConfig,
)


class _Logger:
    def __init__(self, path: Path):
        self.path = path

    def info(self, _message: str, **_fields: object) -> None:
        pass


def _game(now: float) -> GameControlState:
    game = GameControlState(state=GameState.PLAYING)
    game.last_seen_at = now
    return game


def _context(now: float, ball_x: float | None, robot_x: float = -2.0) -> PlayContext:
    ball = None if ball_x is None else BallState(x=ball_x, y=0.0, last_seen_at=now)
    return PlayContext(
        game_state=_game(now),
        teammates={1: RobotState(1, Pose2D(robot_x, 0.0, 0.0), now)},
        opponents={1: RobotState(1, Pose2D(3.0, 2.0, 0.0), now)},
        ball=ball,
    )


class _NoObstacles:
    def collect_all(self, _player_id: int, _context: PlayContext) -> list:
        return []


class MatchDataRecorderV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "match_dataset.team1.jsonl"
        self.recorder = MatchDataRecorder(
            SoccerConfig(robot_names=("robot1",), opponent_robot_names=("robot4",)),
            _Logger(Path(self.tempdir.name) / "events.jsonl"),
        )

    def tearDown(self) -> None:
        self.recorder.close(999.0)
        self.tempdir.cleanup()

    def records(self) -> list[dict]:
        self.recorder.flush()
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_motion_controller_preserves_requested_target(self) -> None:
        controller = MotionController(
            SoccerConfig(robot_names=("robot1",)),
            None,
            None,
            _NoObstacles(),
        )
        context = _context(50.0, None, robot_x=0.0)
        target = Pose2D(1.0, 0.0, 0.0)
        command = controller.move_to_target(1, context, target, "eta test")
        self.assertIsNotNone(command.motion_target)
        assert command.motion_target is not None
        self.assertEqual(command.motion_target.requested_target, target)
        self.assertEqual(command.motion_target.control_target, target)
        self.assertEqual(command.motion_target.phase, "run")

    def test_ready_ball_placement_is_not_a_motion_sample(self) -> None:
        for now, x in ((75.0, -4.0), (75.1, 0.0), (75.2, 0.5)):
            context = _context(now, x)
            assert context.game_state is not None
            context.game_state.state = GameState.READY
            self.recorder.observe(now, context, {1: RobotCommand.stop("ready")})
        self.recorder.flush()
        motions = [row for row in self.records() if row["record_type"] == "ball_motion"]
        self.assertEqual(motions, [])

    def test_clean_motion_stops_as_free_roll_candidate(self) -> None:
        for now, x in ((100.00, 0.00), (100.05, 0.04), (100.10, 0.10),
                       (100.20, 0.20), (100.35, 0.20), (100.55, 0.20)):
            self.recorder.observe(now, _context(now, x), {1: RobotCommand.stop("idle")})
        motions = [row for row in self.records() if row["record_type"] == "ball_motion"]
        self.assertEqual(motions[-1]["end_reason"], "natural_stop_candidate")
        self.assertTrue(motions[-1]["free_roll_terminal_candidate"])
        self.assertIn("nearest_teammate", motions[-1]["trajectory"][0])
        self.assertIn("field_evidence", motions[-1]["trajectory"][0])

    def test_boundary_crossing_is_not_natural_stop(self) -> None:
        for now, x in ((200.00, 6.90), (200.10, 7.00), (200.20, 7.13)):
            self.recorder.observe(now, _context(now, x), {1: RobotCommand.stop("idle")})
        motions = [row for row in self.records() if row["record_type"] == "ball_motion"]
        self.assertEqual(motions[-1]["end_reason"], "goal_line_crossing_candidate")
        self.assertEqual(motions[-1]["label_source"], "geometry")
        self.assertFalse(motions[-1]["free_roll_terminal_candidate"])
        self.assertIn("boundary_crossing", motions[-1]["quality_flags"])

        confirmed = _context(200.30, 7.13)
        assert confirmed.game_state is not None
        confirmed.game_state.teams[0].score = 1
        self.recorder.observe(200.30, confirmed, {1: RobotCommand.stop("goal reset")})
        labels = [row for row in self.records() if row["record_type"] == "ball_motion_label"]
        self.assertEqual(labels[-1]["motion_id"], motions[-1]["motion_id"])
        self.assertEqual(labels[-1]["official_label"], "goal_confirmed_team_1")

    def test_nearby_robot_marks_stop_as_contact_candidate(self) -> None:
        for now, x in ((250.00, 0.00), (250.05, 0.04), (250.10, 0.10),
                       (250.30, 0.20), (250.50, 0.20), (250.70, 0.20)):
            self.recorder.observe(
                now,
                _context(now, x, robot_x=x + 0.10),
                {1: RobotCommand.stop("idle")},
            )
        motions = [row for row in self.records() if row["record_type"] == "ball_motion"]
        self.assertEqual(motions[-1]["end_reason"], "natural_stop_candidate")
        self.assertIn("robot_contact_candidate", motions[-1]["quality_flags"])
        self.assertFalse(motions[-1]["free_roll_terminal_candidate"])

    def test_eta_target_change_creates_censored_segment(self) -> None:
        for now, target_x in ((280.0, 1.0), (280.1, 2.0)):
            target = Pose2D(target_x, 0.0, 0.0)
            command = RobotCommand(
                intent=MoveIntent(vx=0.5),
                reason="dynamic target",
                motion_target=MotionTargetTrace(target, target, 0.15, "run"),
            )
            self.recorder.observe(now, _context(now, None, robot_x=0.0), {1: command})
        rows = [row for row in self.records() if row["record_type"] == "robot_eta"]
        self.assertEqual(rows[-1]["end_reason"], "target_changed")
        self.assertTrue(rows[-1]["censored"])
        self.assertFalse(rows[-1]["eta_training_candidate"])

    def test_eta_record_preserves_target_and_phases(self) -> None:
        target = Pose2D(1.0, 0.0, 0.0)
        for now, x, phase, vx in (
            (300.00, 0.00, "turn", 0.0),
            (300.10, 0.05, "run", 0.5),
            (300.20, 0.60, "run", 0.5),
            (300.30, 0.96, "arrived", 0.0),
        ):
            command = RobotCommand(
                intent=MoveIntent(vx=vx),
                reason="ready target",
                motion_target=MotionTargetTrace(
                    requested_target=target,
                    control_target=target,
                    arrive_distance=0.15,
                    phase=phase,
                ),
            )
            self.recorder.observe(
                now,
                _context(now, None, robot_x=x),
                {1: command},
                roles={1: "supporter"},
            )
        rows = [row for row in self.records() if row["record_type"] == "robot_eta"]
        eta = rows[-1]
        self.assertEqual(eta["end_reason"], "arrived")
        self.assertTrue(eta["eta_training_candidate"])
        self.assertEqual(eta["requested_target_at_start"]["x"], 1.0)
        self.assertEqual([p["phase"] for p in eta["trajectory"]], ["turn", "run", "run", "arrived"])
        self.assertIn("heading_to_path_error_rad", eta["trajectory"][0])


if __name__ == "__main__":
    unittest.main()
