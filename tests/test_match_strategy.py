import unittest

from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    Penalty,
    PlayContext,
    Pose2D,
    RobotState,
    SetPlay,
    SoccerConfig,
)
from src.tactics.geometry import TeamFieldFrame
from src.tactics.ready_stance import ReadyStance
from src.tactics.match_strategy import (
    KickoffPhase,
    KickoffTransaction,
    OpponentShapeTracker,
    calculate_match_risk,
    own_restart_target,
)


def context_at(x=0.0, y=0.0):
    game = GameControlState(state=GameState.PLAYING, secs_remaining=300)
    return PlayContext(
        game_state=game,
        teammates={
            1: RobotState(1, Pose2D(-0.3, 0.0, 0.0), 1.0),
            2: RobotState(2, Pose2D(0.4, 0.2, 0.0), 1.0),
            3: RobotState(3, Pose2D(-5.0, 0.0, 0.0), 1.0),
        },
        ball=BallState(x, y, 1.0),
    )


class KickoffTransactionTest(unittest.TestCase):
    def setUp(self):
        self.config = SoccerConfig()
        self.transaction = KickoffTransaction(
            self.config, TeamFieldFrame(self.config)
        )

    def test_two_touch_transaction_survives_gamecontroller_clear(self):
        context = context_at()
        context.known_game.kicking_team = self.config.team_id
        status = self.transaction.update(
            now_sec=1.0, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.FIRST_TOUCH_ACTIVE)
        self.assertIsNotNone(status.receiver_target)
        self.assertLess(status.first_target.x, 0.30)

        context.known_game.kicking_team = 0
        context.known_ball.x = 0.20
        status = self.transaction.update(
            now_sec=1.2, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.VERIFY_FIRST_TOUCH)

        context.known_ball.x = 0.30
        status = self.transaction.update(
            now_sec=1.3, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.SECOND_PLAYER_ACQUIRE)

        context.teammates[2].pose = Pose2D(0.4, 0.0, 0.0)
        status = self.transaction.update(
            now_sec=1.4, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.SECOND_PLAYER_ACQUIRE)
        self.assertGreater(status.ball_speed_mps, 0.22)

        status = self.transaction.update(
            now_sec=1.5, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.SECOND_KICK_ACTIVE)

        context.known_ball.x = 0.65
        status = self.transaction.update(
            now_sec=1.6, context=context, first_player_id=1, second_player_id=2
        )
        self.assertEqual(status.phase, KickoffPhase.COMPLETE)
        self.assertFalse(status.active)

    def test_kickoff_times_out_to_fallback(self):
        context = context_at()
        context.known_game.kicking_team = self.config.team_id
        self.transaction.update(
            now_sec=1.0, context=context, first_player_id=1, second_player_id=2
        )

        status = self.transaction.update(
            now_sec=11.1, context=context, first_player_id=1, second_player_id=2
        )

        self.assertEqual(status.phase, KickoffPhase.FALLBACK)


class MatchManagementTest(unittest.TestCase):
    def test_late_trailing_team_has_higher_risk_than_late_leader(self):
        config = SoccerConfig()
        trailing = context_at(1.0)
        trailing.known_game.secs_remaining = 40
        trailing.known_game.teams[0].score = 0
        trailing.known_game.teams[1].score = 2
        leading = context_at(1.0)
        leading.known_game.secs_remaining = 40
        leading.known_game.teams[0].score = 2

        attack = calculate_match_risk(config, trailing, "manageable")
        protect = calculate_match_risk(config, leading, "manageable")

        self.assertGreater(attack.value, protect.value)
        self.assertGreaterEqual(attack.value, 0.0)
        self.assertLessEqual(attack.value, 1.0)

    def test_player_disadvantage_reduces_risk(self):
        config = SoccerConfig()
        full = context_at()
        reduced = context_at()
        reduced.known_game.teams[0].players[1].penalty = Penalty.PUSHING

        full_risk = calculate_match_risk(config, full, "manageable")
        reduced_risk = calculate_match_risk(config, reduced, "manageable")

        self.assertLess(reduced_risk.value, full_risk.value)

    def test_goalkeeper_guard_projects_ball_to_goal_line_and_avoids_posts(self):
        config = SoccerConfig()
        field = TeamFieldFrame(config)
        stance = ReadyStance(config, field)
        ball = BallState(0.0, 2.5, 1.0)

        target = stance.goalkeeper_guard_target(ball)
        expected_y = ball.y * (target.x - field.own_goal_x()) / (
            ball.x - field.own_goal_x()
        )

        self.assertAlmostEqual(target.y, expected_y)
        self.assertLessEqual(abs(target.y), config.goal_width / 2.0 - 0.40)

    def test_restart_targets_are_deterministic_and_safe(self):
        config = SoccerConfig()
        field = TeamFieldFrame(config)
        context = context_at(0.0, 4.2)
        context.known_game.kicking_team = config.team_id
        context.known_game.set_play = SetPlay.THROW_IN

        target = own_restart_target(
            config, field, context, Pose2D(field.opponent_goal_x(), 0.0, 0.0)
        )

        self.assertGreater(target.x, context.known_ball.x)
        self.assertLess(abs(target.y), abs(context.known_ball.y))

    def test_opponent_shape_samples_at_two_hz_with_fixed_window(self):
        tracker = OpponentShapeTracker(max_samples=3)
        context = context_at()
        context.opponents = {
            1: RobotState(1, Pose2D(0.5, -0.5, 0.0), 1.0),
            2: RobotState(2, Pose2D(1.0, 0.5, 0.0), 1.0),
        }
        for now in (1.0, 1.1, 1.5, 2.0, 2.5):
            shape = tracker.update(now, context)

        self.assertEqual(shape.samples, 3)
        self.assertGreater(shape.ball_density, 0.0)
        self.assertAlmostEqual(shape.lateral_compactness, 1.0)


if __name__ == "__main__":
    unittest.main()
