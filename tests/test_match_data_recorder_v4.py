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
    KickIntent,
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


class _Kicker:
    def mark_kicking(self, _player_id: int) -> None:
        pass


class _NoObstacles:
    def collect_all(self, _player_id: int, _context: PlayContext) -> list:
        return []


class MatchDataRecorderV4Tests(unittest.TestCase):
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
            _Kicker(),
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
        self.assertEqual(command.motion_target.linear_speed_limit_mps, 0.4)
        self.assertEqual(command.intent.vx, 0.4)

        next_target = Pose2D(2.0, 0.0, 0.0)
        next_command = controller.move_to_target(1, context, next_target, "eta test 2")
        assert next_command.motion_target is not None
        self.assertEqual(next_command.motion_target.linear_speed_limit_mps, 0.6)
        self.assertEqual(next_command.intent.vx, 0.6)

    def test_kick_power_is_stable_per_episode_and_rotates(self) -> None:
        controller = MotionController(
            SoccerConfig(robot_names=("robot1",)),
            None,
            _Kicker(),
            _NoObstacles(),
        )
        first = _context(60.00, 0.1, robot_x=0.0)
        same = _context(60.05, 0.1, robot_x=0.0)
        second = _context(61.00, 0.1, robot_x=0.0)
        first_command = controller.kick_command(1, first, 0.0, "kick")
        same_command = controller.kick_command(1, same, 0.0, "kick")
        second_command = controller.kick_command(1, second, 0.0, "kick")
        self.assertEqual(first_command.intent.power, 1.0)
        self.assertEqual(same_command.intent.power, 1.0)
        self.assertEqual(second_command.intent.power, 1.25)

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
        self.assertEqual(motions[-1]["schema_version"], 4)
        self.assertEqual(motions[-1]["terminal_event"]["type"], "natural_stop_candidate")
        self.assertEqual(
            motions[-1]["free_path_valid_until_sec"],
            motions[-1]["terminal_event"]["t_sec"],
        )
        self.assertIn(motions[-1]["launch_event"]["type"], {
            "unattributed_acceleration_candidate", "robot_kick_candidate"
        })

    def test_boundary_crossing_is_not_natural_stop(self) -> None:
        for now, x in ((200.00, 6.90), (200.10, 7.00), (200.20, 7.13)):
            self.recorder.observe(now, _context(now, x), {1: RobotCommand.stop("idle")})
        motions = [row for row in self.records() if row["record_type"] == "ball_motion"]
        self.assertEqual(motions[-1]["end_reason"], "goal_line_crossing_candidate")
        self.assertEqual(motions[-1]["label_source"], "geometry")
        self.assertFalse(motions[-1]["free_roll_terminal_candidate"])
        self.assertIn("boundary_crossing", motions[-1]["quality_flags"])
        self.assertEqual(motions[-1]["terminal_event"]["type"], "goal_line_exit_candidate")

        for now, x in ((200.23, 7.16), (200.26, 7.20)):
            self.recorder.observe(now, _context(now, x), {1: RobotCommand.stop("outside")})
        self.assertEqual(
            len([row for row in self.records() if row["record_type"] == "ball_motion"]),
            1,
        )

        confirmed = _context(200.30, 7.13)
        assert confirmed.game_state is not None
        confirmed.game_state.teams[0].score = 1
        self.recorder.observe(200.30, confirmed, {1: RobotCommand.stop("goal reset")})
        labels = [row for row in self.records() if row["record_type"] == "ball_motion_label"]
        self.assertEqual(labels[-1]["motion_id"], motions[-1]["motion_id"])
        self.assertEqual(labels[-1]["official_label"], "goal_confirmed_team_1")

    def test_later_robot_contact_limits_free_path_horizon(self) -> None:
        rows = (
            (230.00, 0.00, -2.0),
            (230.05, 0.05, -2.0),
            (230.10, 0.12, -2.0),
            (230.20, 0.22, 0.72),
            (230.30, 0.32, 0.80),
            (230.40, 0.24, 0.30),
            (230.60, 0.24, 0.30),
            (230.80, 0.24, 0.30),
        )
        for now, ball_x, robot_x in rows:
            self.recorder.observe(
                now,
                _context(now, ball_x, robot_x=robot_x),
                {1: RobotCommand.stop("contact test")},
            )
        motion = [row for row in self.records() if row["record_type"] == "ball_motion"][-1]
        contacts = [
            event for event in motion["event_candidates"]
            if event["type"] == "robot_contact_candidate"
        ]
        self.assertTrue(contacts)
        self.assertGreaterEqual(contacts[0]["confidence"], 0.75)
        self.assertLessEqual(
            motion["free_path_valid_until_sec"], contacts[0]["t_sec"]
        )

    def test_own_kick_command_labels_launch_event(self) -> None:
        kick = RobotCommand(
            intent=KickIntent(direction=0.0, power=1.25, ball_x=0.1, ball_y=0.0),
            reason="label kick",
        )
        self.recorder.observe(240.00, _context(240.00, 0.0, robot_x=-0.1), {1: kick})
        for now, x in ((240.05, 0.05), (240.10, 0.12), (240.20, 0.25),
                       (240.40, 0.25), (240.60, 0.25)):
            self.recorder.observe(now, _context(now, x, robot_x=-0.1), {1: kick})
        motion = [row for row in self.records() if row["record_type"] == "ball_motion"][-1]
        self.assertEqual(motion["source"], "own_kick")
        self.assertEqual(motion["kick_player_id"], 1)
        self.assertEqual(motion["kick_power"], 1.25)
        self.assertEqual(motion["launch_event"]["type"], "own_kick_command")
        self.assertEqual(motion["launch_event"]["confidence"], 1.0)

    def test_secondary_robot_acceleration_is_kick_candidate(self) -> None:
        rows = (
            (235.00, 0.00, -2.0),
            (235.10, 0.05, -2.0),
            (235.20, 0.10, -2.0),
            (235.30, 0.15, 0.70),
            (235.40, 0.40, 0.45),
            (235.50, 0.65, 0.70),
            (235.70, 0.65, 0.70),
            (235.90, 0.65, 0.70),
        )
        for now, ball_x, robot_x in rows:
            self.recorder.observe(
                now,
                _context(now, ball_x, robot_x=robot_x),
                {1: RobotCommand.stop("second kick test")},
            )
        motion = [row for row in self.records() if row["record_type"] == "ball_motion"][-1]
        kicks = [
            event for event in motion["event_candidates"]
            if event["type"] == "robot_kick_candidate"
        ]
        self.assertTrue(kicks)
        self.assertGreaterEqual(kicks[0]["speed_delta_mps"], 0.5)
        self.assertLessEqual(motion["free_path_valid_until_sec"], kicks[0]["t_sec"])

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

    def test_finished_state_stops_frames_and_allows_next_match(self) -> None:
        context = _context(270.0, None)
        assert context.game_state is not None
        context.game_state.state = GameState.FINISHED
        self.recorder.observe(270.0, context, {1: RobotCommand.stop("finished")})
        self.recorder.observe(271.0, context, {1: RobotCommand.stop("still finished")})
        records = self.records()
        rows = [row for row in records if row["record_type"] == "match_end"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game"]["state"], "FINISHED")
        self.assertEqual(rows[0]["match_id"], 1)
        self.assertEqual([row for row in records if row["record_type"] == "frame"], [])

        next_context = _context(272.0, None)
        assert next_context.game_state is not None
        next_context.game_state.state = GameState.READY
        self.recorder.observe(272.0, next_context, {1: RobotCommand.stop("next match")})
        records = self.records()
        starts = [row for row in records if row["record_type"] == "match_start"]
        self.assertEqual(starts[-1]["match_id"], 2)
        frames = [row for row in records if row["record_type"] == "frame"]
        self.assertEqual(frames[-1]["match_id"], 2)

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

    def test_eta_hold_finishes_once_and_does_not_restart(self) -> None:
        target = Pose2D(1.0, 0.0, 0.0)
        for now, x, phase, vx in (
            (290.00, 0.00, "run", 0.5),
            (290.10, 0.50, "run", 0.5),
            (290.20, 0.96, "hold", 0.0),
            (290.30, 0.96, "hold", 0.0),
        ):
            command = RobotCommand(
                intent=MoveIntent(vx=vx),
                reason="hold target",
                motion_target=MotionTargetTrace(target, target, 0.15, phase),
            )
            self.recorder.observe(now, _context(now, None, robot_x=x), {1: command})
        rows = [
            row for row in self.records()
            if row["record_type"] == "robot_eta"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["end_reason"], "arrived")
        self.assertEqual([point["phase"] for point in rows[0]["trajectory"]], ["run", "run", "hold"])

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
                reason=(
                    "eta_exp|scenario=short_straight|episode=1|active=1"
                    "|mode=rest_start|distance=0.60|path_error=0.000"
                    "|final_error=0.000"
                ),
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
        self.assertEqual(eta["experiment_at_start"]["scenario"], "short_straight")
        self.assertEqual(eta["experiment_at_start"]["episode"], 1)


if __name__ == "__main__":
    unittest.main()
