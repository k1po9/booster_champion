import importlib.util
import unittest

if importlib.util.find_spec("py_trees") is None:
    raise unittest.SkipTest("project dependency py_trees is not installed")

from src.play import (
    ChampionPlaybook,
    ChampionTuning,
    DefaultPlaybook,
    PLAYBOOKS,
    ROLE_CHASER,
    ROLE_GOALKEEPER,
    ROLE_MARKER,
    ROLE_NONE,
    ROLE_SECOND_BALL,
    ROLE_SUPPORTER,
)
from src.runtime import SoccerKit
from src.soccer_framework import (
    BallState,
    GameControlState,
    Penalty,
    PlayContext,
    Pose2D,
    RobotState,
    SoccerConfig,
    SetPlay,
)
from src.tactics import BallOwnerRole, TeamPhase


class FakeClock:
    def __init__(self, now=10.0):
        self.now = now

    def __call__(self):
        return self.now


def make_context(*, poses=None, ball_x=0.0, ball_stamp=10.0):
    poses = poses or {
        1: Pose2D(-0.2, 0.0, 0.0),
        2: Pose2D(-1.0, 0.5, 0.0),
        3: Pose2D(-5.0, 0.0, 0.0),
    }
    return PlayContext(
        game_state=GameControlState(),
        teammates={
            pid: RobotState(pid, pose, last_seen_at=ball_stamp)
            for pid, pose in poses.items()
        },
        ball=BallState(ball_x, 0.0, last_seen_at=ball_stamp),
    )


class ChampionPlaybookTest(unittest.TestCase):
    def setUp(self):
        self.kit = SoccerKit(SoccerConfig())
        self.clock = FakeClock()
        self.playbook = ChampionPlaybook(self.kit, clock=self.clock)

    def test_champion_is_registered_as_default_with_legacy_fallback(self):
        self.assertIn("champion", PLAYBOOKS.names())
        self.assertIsInstance(PLAYBOOKS.create("champion", self.kit), ChampionPlaybook)
        self.assertIsInstance(PLAYBOOKS.create_default(self.kit), ChampionPlaybook)
        self.assertIsInstance(PLAYBOOKS.create("default", self.kit), DefaultPlaybook)

    def test_full_team_has_handler_outlet_and_configured_cover(self):
        assignment = self.playbook.assign_roles(make_context())

        self.assertEqual(assignment.role_of(3), ROLE_GOALKEEPER)
        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertEqual(len(assignment.players_of(ROLE_SUPPORTER)), 1)

    def test_penalized_goalkeeper_gets_emergency_cover(self):
        context = make_context()
        context.known_game.teams[0].players[2].penalty = Penalty.PUSHING

        assignment = self.playbook.assign_roles(context)

        self.assertEqual(assignment.role_of(3), ROLE_NONE)
        self.assertEqual(assignment.role_of(2), ROLE_GOALKEEPER)
        self.assertEqual(assignment.role_of(1), ROLE_CHASER)

    def test_one_available_player_prioritizes_cover(self):
        context = make_context(poses={1: Pose2D(-1.0, 0.0, 0.0)})

        assignment = self.playbook.assign_roles(context)

        self.assertEqual(assignment.role_of(1), ROLE_GOALKEEPER)
        self.assertEqual(assignment.players_of(ROLE_CHASER), ())

    def test_keeper_blocks_line_and_leaves_interception_to_field_player(self):
        context = make_context(
            poses={
                1: Pose2D(-1.0, 0.0, 0.0),
                2: Pose2D(-2.0, 0.8, 0.0),
                3: Pose2D(-5.2, 0.0, 0.0),
            },
            ball_x=-5.0,
        )
        context.opponents = {
            1: RobotState(1, Pose2D(-4.8, 0.0, 0.0), self.clock.now),
            2: RobotState(2, Pose2D(-5.0, 1.0, 0.0), self.clock.now),
        }

        assignment = self.playbook.assign_roles(context)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None

        self.assertEqual(snapshot.team_phase, TeamPhase.KEEPER_BLOCK)
        self.assertEqual(snapshot.ball_owner_role, BallOwnerRole.HANDLER)
        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertEqual(len(assignment.players_of(ROLE_SECOND_BALL)), 0)
        self.assertEqual(len(assignment.players_of(ROLE_MARKER)), 1)
        self.assertFalse(self.playbook.goalkeeper_owns_ball())

    def test_keeper_clears_only_when_ball_is_close_slow_and_unpressured(self):
        context = make_context(
            poses={
                1: Pose2D(-1.0, 0.0, 0.0),
                2: Pose2D(-2.0, 0.8, 0.0),
                3: Pose2D(-5.2, 0.0, 0.0),
            },
            ball_x=-5.0,
        )

        assignment = self.playbook.assign_roles(context)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None

        self.assertEqual(snapshot.team_phase, TeamPhase.KEEPER_EMERGENCY)
        self.assertEqual(snapshot.ball_owner_role, BallOwnerRole.KEEPER)
        self.assertEqual(assignment.players_of(ROLE_CHASER), ())
        self.assertEqual(len(assignment.players_of(ROLE_SECOND_BALL)), 1)
        self.assertTrue(self.playbook.goalkeeper_owns_ball())

    def test_keeper_takeover_is_blocked_during_set_play(self):
        context = make_context(
            poses={
                1: Pose2D(-1.0, 0.0, 0.0),
                2: Pose2D(-2.0, 0.8, 0.0),
                3: Pose2D(-5.2, 0.0, 0.0),
            },
            ball_x=-5.0,
        )
        context.known_game.set_play = SetPlay.DIRECT_FREE_KICK

        assignment = self.playbook.assign_roles(context)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None

        self.assertNotEqual(snapshot.team_phase, TeamPhase.KEEPER_EMERGENCY)
        self.assertEqual(snapshot.ball_owner_role, BallOwnerRole.HANDLER)
        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertFalse(self.playbook.goalkeeper_owns_ball())

    def test_coordination_switch_restores_legacy_role_structure(self):
        playbook = ChampionPlaybook(
            self.kit,
            clock=self.clock,
            tuning=ChampionTuning(enable_team_coordination=False),
        )
        context = make_context(
            poses={
                1: Pose2D(-1.0, 0.0, 0.0),
                2: Pose2D(-2.0, 0.8, 0.0),
                3: Pose2D(-5.2, 0.0, 0.0),
            },
            ball_x=-5.0,
        )

        assignment = playbook.assign_roles(context)

        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertEqual(len(assignment.players_of(ROLE_SECOND_BALL)), 0)
        self.assertEqual(len(assignment.players_of(ROLE_MARKER)), 0)
        keeper_role = playbook.role_registry.get(ROLE_GOALKEEPER)
        self.assertIsNotNone(keeper_role)
        assert keeper_role is not None
        self.assertTrue(keeper_role.wants_to_kick(self.kit, context))

    def test_keeper_release_restores_one_field_handler(self):
        context = make_context(
            poses={
                1: Pose2D(-1.0, 0.0, 0.0),
                2: Pose2D(-2.0, 0.8, 0.0),
                3: Pose2D(-5.2, 0.0, 0.0),
            },
            ball_x=-5.0,
        )
        self.playbook.assign_roles(context)

        self.clock.now += 0.7
        released = make_context(ball_x=0.0, ball_stamp=self.clock.now)
        assignment = self.playbook.assign_roles(released)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None

        self.assertEqual(snapshot.team_phase, TeamPhase.KEEPER_RECOVER)
        self.assertEqual(snapshot.ball_owner_role, BallOwnerRole.HANDLER)
        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertFalse(self.playbook.goalkeeper_owns_ball())

    def test_defending_assigns_marker_to_dangerous_off_ball_opponent(self):
        context = make_context(ball_x=-1.0)
        context.opponents = {
            1: RobotState(1, Pose2D(-1.0, 0.1, 0.0), self.clock.now),
            2: RobotState(2, Pose2D(-4.5, 0.8, 0.0), self.clock.now),
            3: RobotState(3, Pose2D(1.0, -2.0, 0.0), self.clock.now),
        }

        assignment = self.playbook.assign_roles(context)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None

        self.assertEqual(snapshot.team_phase, TeamPhase.DEFEND)
        self.assertEqual(snapshot.marked_opponent_id, 2)
        self.assertEqual(len(assignment.players_of(ROLE_CHASER)), 1)
        self.assertEqual(len(assignment.players_of(ROLE_MARKER)), 1)

    def test_handler_hysteresis_rejects_small_and_early_switches(self):
        first = make_context(
            poses={
                1: Pose2D(-0.30, 0.0, 0.0),
                2: Pose2D(-0.40, 0.0, 0.0),
                3: Pose2D(-5.0, 0.0, 0.0),
            }
        )
        self.assertEqual(self.playbook.assign_roles(first).players_of(ROLE_CHASER), (1,))

        self.clock.now += 0.2
        early = make_context(
            poses={
                1: Pose2D(-1.5, 0.0, 0.0),
                2: Pose2D(-0.1, 0.0, 0.0),
                3: Pose2D(-5.0, 0.0, 0.0),
            },
            ball_stamp=10.2,
        )
        self.assertEqual(self.playbook.assign_roles(early).players_of(ROLE_CHASER), (1,))

        self.clock.now += 0.5
        self.assertEqual(self.playbook.assign_roles(early).players_of(ROLE_CHASER), (2,))

    def test_prediction_eta_pressure_and_watchdog_are_exposed(self):
        for index, x in enumerate((0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48)):
            stamp = 10.0 + index * 0.04
            self.clock.now = stamp
            context = make_context(
                poses={
                    1: Pose2D(x - 0.3, 0.0, 0.0),
                    2: Pose2D(-2.0, 1.0, 0.0),
                    3: Pose2D(-5.0, 0.0, 0.0),
                },
                ball_x=x,
                ball_stamp=stamp,
            )
            self.playbook.assign_roles(context)

        snapshot = self.playbook.last_snapshot
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.ball_prediction)
        self.assertIsNotNone(snapshot.ball_trajectory)
        assert snapshot.ball_trajectory is not None
        self.assertTrue(snapshot.ball_trajectory.usable)
        self.assertIsNotNone(snapshot.handler_eta)
        self.assertIsNotNone(snapshot.handler_id)
        self.assertEqual(
            self.playbook.handler_chase_target(snapshot.handler_id, context),
            snapshot.chase_target,
        )
        self.assertGreaterEqual(snapshot.chase_target.x, context.known_ball.x)
        self.assertIsNotNone(snapshot.pressure)
        self.assertIsNotNone(snapshot.watchdog)

    def test_capability_switches_return_to_current_ball_fallback(self):
        playbook = ChampionPlaybook(
            self.kit,
            clock=self.clock,
            tuning=ChampionTuning(
                enable_prediction=False,
                enable_pressure=False,
                enable_watchdog=False,
                enable_baseline_shadow=False,
            ),
        )

        playbook.assign_roles(make_context(ball_x=0.4))

        snapshot = playbook.last_snapshot
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNone(snapshot.ball_prediction)
        self.assertIsNone(snapshot.pressure)
        self.assertIsNone(snapshot.watchdog)
        self.assertIsNone(snapshot.baseline_assignment)
        self.assertEqual(snapshot.chase_target, Pose2D(0.4, 0.0, 0.0))

    def test_diagnostics_include_baseline_and_bounded_performance_window(self):
        playbook = ChampionPlaybook(
            self.kit,
            clock=self.clock,
            tuning=ChampionTuning(performance_window_size=3),
        )
        for index in range(5):
            self.clock.now += 0.1
            playbook.assign_roles(make_context(ball_stamp=self.clock.now))

        performance = playbook.performance()
        diagnostics = playbook.diagnostics()

        self.assertEqual(performance.sample_count, 3)
        self.assertGreaterEqual(performance.p99_ms, performance.average_ms)
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertIsNotNone(diagnostics["baseline_roles"])
        self.assertEqual(diagnostics["performance"]["samples"], 3)

    def test_action_execution_switch_uses_selected_candidate(self):
        playbook = ChampionPlaybook(
            self.kit,
            clock=self.clock,
            tuning=ChampionTuning(enable_action_execution=True),
        )
        context = make_context()
        assignment = playbook.assign_roles(context)
        handler_id = assignment.players_of(ROLE_CHASER)[0]
        snapshot = playbook.last_snapshot
        assert snapshot is not None and snapshot.action_selection is not None

        target = playbook.handler_kick_target(handler_id, context)

        self.assertEqual(target, snapshot.action_selection.selected.target)

    def test_kickoff_receiver_stays_behind_halfway_and_cannot_kick_early(self):
        context = make_context()
        context.known_game.kicking_team = self.kit.config.team_id
        assignment = self.playbook.assign_roles(context)
        snapshot = self.playbook.last_snapshot
        assert snapshot is not None and snapshot.kickoff is not None
        receiver_id = snapshot.kickoff.second_player_id
        assert receiver_id is not None

        target = self.playbook.outlet_target(receiver_id, context)

        self.assertLessEqual(
            target.x, -self.playbook.tuning.kickoff_receiver_halfway_margin_m
        )
        self.assertTrue(self.playbook.handler_wants_to_kick(
            assignment.players_of(ROLE_CHASER)[0], context
        ))

    def test_outlet_chooses_side_farther_from_opponent(self):
        context = make_context()
        context.opponents = {
            1: RobotState(1, Pose2D(1.5, 1.5, 0.0), self.clock.now),
        }

        target = self.playbook.outlet_target(2, context)

        self.assertLess(target.y, 0.0)

    def test_watchdog_stall_switches_to_a_different_action(self):
        playbook = ChampionPlaybook(
            self.kit,
            clock=self.clock,
            tuning=ChampionTuning(enable_action_execution=True),
        )
        context = make_context()
        playbook.assign_roles(context)
        first = playbook.last_snapshot
        assert first is not None and first.action_selection is not None

        self.clock.now += 1.1
        context.known_ball.last_seen_at = self.clock.now
        for robot in context.teammates.values():
            robot.last_seen_at = self.clock.now
        playbook.assign_roles(context)
        second = playbook.last_snapshot
        assert second is not None and second.action_selection is not None

        self.assertTrue(second.watchdog.should_escape)
        self.assertNotEqual(
            second.action_selection.selected.target,
            first.action_selection.selected.target,
        )

    def test_dangerous_ball_tightens_outlet_behind_play(self):
        context = make_context(ball_x=-5.0)
        self.playbook.assign_roles(context)

        target = self.playbook.outlet_target(2, context)

        self.assertLessEqual(target.x, self.kit.field.own_goal_x() + 2.4)
        self.assertLessEqual(abs(target.y), 1.05)


if __name__ == "__main__":
    unittest.main()
