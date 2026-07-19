import importlib.util
import unittest

if importlib.util.find_spec("py_trees") is None:
    raise unittest.SkipTest(
        "py_trees==2.4.0 is supplied by the packaged agent runtime"
    )

from src.play import (
    DynamicTrianglePlaybook,
    PrimaryIntent,
    ROLE_PRIMARY,
    ROLE_SAFETY,
    ROLE_SECONDARY,
    TacticalMode,
)
from src.tactics import ObstacleCollector, Targeting, TeamFieldFrame
from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    PlayContext,
    Pose2D,
    RobotState,
    SoccerConfig,
    SoccerStrategyTuning,
)

class _FakeMotion:
    @staticmethod
    def approach_target(ball, theta, offset):
        import math

        return Pose2D(
            ball.x - offset * math.cos(theta),
            ball.y - offset * math.sin(theta),
            theta,
        )


class SoccerKit:
    """ROS-free kit exposing only the pure services used by the coordinator."""

    def __init__(self, config):
        self.config = config
        self.field = TeamFieldFrame(config)
        self.obstacles = ObstacleCollector(config, self.field)
        self.targeting = Targeting(config, self.field, self.obstacles)
        self.motion = _FakeMotion()

    def is_player_allowed(self, game, player_id):
        return game.is_active_player(self.config.team_id, player_id)


def make_context(
    ball: tuple[float, float],
    teammates: dict[int, tuple[float, float]],
    opponents: dict[int, tuple[float, float]] | None = None,
    *,
    stamp: float = 1.0,
) -> PlayContext:
    game = GameControlState(state=GameState.PLAYING, last_seen_at=stamp)
    return PlayContext(
        game_state=game,
        ball=BallState(ball[0], ball[1], last_seen_at=stamp),
        teammates={
            player_id: RobotState(
                player_id,
                Pose2D(x, y, 0.0),
                last_seen_at=stamp,
            )
            for player_id, (x, y) in teammates.items()
        },
        opponents={
            player_id: RobotState(
                player_id,
                Pose2D(x, y, 0.0),
                last_seen_at=stamp,
            )
            for player_id, (x, y) in (opponents or {}).items()
        },
    )


class DynamicTrianglePlaybookTest(unittest.TestCase):
    def test_attack_assigns_three_distinct_task_slots(self):
        playbook = DynamicTrianglePlaybook(SoccerKit(SoccerConfig()))
        context = make_context(
            (1.0, 0.0),
            {1: (0.5, 0.0), 2: (-0.2, 1.0), 3: (-5.8, 0.0)},
            {1: (4.0, 0.0)},
        )

        playbook.prepare_tick(context, 1.0)
        assignment = playbook.assign_roles(context)
        tactical = playbook.tactical_context

        self.assertEqual(tactical.mode, TacticalMode.ATTACK)
        self.assertEqual(
            set(assignment.by_player.values()),
            {ROLE_PRIMARY, ROLE_SECONDARY, ROLE_SAFETY},
        )
        self.assertEqual(len(set(assignment.by_player)), 3)
        self.assertLess(tactical.safety_target.x, tactical.primary_target.x)

    def test_emergency_mode_prioritizes_clear_and_goal_line_cover(self):
        playbook = DynamicTrianglePlaybook(SoccerKit(SoccerConfig()))
        context = make_context(
            (-5.2, 0.8),
            {1: (-4.7, 0.8), 2: (-3.8, -0.5), 3: (-5.9, 0.0)},
            {1: (-5.0, 0.7)},
        )

        playbook.prepare_tick(context, 1.0)
        tactical = playbook.tactical_context

        self.assertEqual(tactical.mode, TacticalMode.EMERGENCY_DEFEND)
        self.assertEqual(tactical.primary_intent, PrimaryIntent.CLEAR)
        self.assertLess(tactical.safety_target.x, -5.5)
        self.assertLessEqual(abs(tactical.safety_target.y), 1.1)

    def test_primary_handoff_requires_sustained_advantage(self):
        tuning = SoccerStrategyTuning(
            role_switch_advantage_sec=0.1,
            role_switch_confirm_ticks=3,
        )
        playbook = DynamicTrianglePlaybook(
            SoccerKit(SoccerConfig(strategy=tuning))
        )
        initial = make_context(
            (0.0, 0.0),
            {1: (-0.2, 0.0), 2: (-2.0, 0.0), 3: (-5.8, 0.0)},
            {1: (3.0, 0.0)},
        )
        playbook.prepare_tick(initial, 1.0)
        first = playbook.tactical_context.primary_id

        challenger = make_context(
            (0.0, 0.0),
            {1: (-3.0, 0.0), 2: (-0.1, 0.0), 3: (-5.8, 0.0)},
            {1: (3.0, 0.0)},
            stamp=1.1,
        )
        playbook.prepare_tick(challenger, 1.1)
        self.assertEqual(playbook.tactical_context.primary_id, first)
        playbook.prepare_tick(challenger, 1.2)
        self.assertEqual(playbook.tactical_context.primary_id, first)
        playbook.prepare_tick(challenger, 1.3)
        self.assertEqual(playbook.tactical_context.primary_id, 2)

    def test_two_players_degrade_to_primary_and_safety(self):
        config = SoccerConfig(robot_names=("robot1", "robot2"))
        playbook = DynamicTrianglePlaybook(SoccerKit(config))
        context = make_context(
            (0.5, 0.0),
            {1: (0.0, 0.0), 2: (-3.0, 0.0)},
            {1: (4.0, 0.0)},
        )

        playbook.prepare_tick(context, 1.0)
        tactical = playbook.tactical_context
        assignment = playbook.assign_roles(context)

        self.assertIsNone(tactical.secondary_id)
        self.assertEqual(
            set(assignment.by_player.values()),
            {ROLE_PRIMARY, ROLE_SAFETY},
        )

    def test_pass_waits_until_secondary_reaches_receive_window(self):
        playbook = DynamicTrianglePlaybook(SoccerKit(SoccerConfig()))
        context = make_context(
            (0.5, 0.0),
            {1: (0.1, 0.0), 2: (0.0, 1.0), 3: (-5.8, 0.0)},
            {},
        )
        playbook.prepare_tick(context, 1.0)
        first = playbook.tactical_context

        self.assertEqual(first.primary_intent, PrimaryIntent.PASS)
        self.assertFalse(first.secondary_ready)
        self.assertIsNotNone(first.receive_target)
        target = first.receive_target
        context.teammates[first.secondary_id].pose = Pose2D(
            target.x, target.y, target.theta
        )
        context.ball.last_seen_at = 1.1
        playbook.prepare_tick(context, 1.1)

        self.assertEqual(playbook.tactical_context.primary_intent, PrimaryIntent.PASS)
        self.assertTrue(playbook.tactical_context.secondary_ready)


if __name__ == "__main__":
    unittest.main()
