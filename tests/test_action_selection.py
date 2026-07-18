import unittest
import time

from src.soccer_framework import (
    BallState,
    GameControlState,
    PlayContext,
    Pose2D,
    RobotState,
    SoccerConfig,
)
from src.tactics.action_selection import ActionKind, BoundedActionSelector
from src.tactics.geometry import TeamFieldFrame
from src.tactics.navigation import ObstacleCollector


def make_selector():
    config = SoccerConfig()
    field = TeamFieldFrame(config)
    return BoundedActionSelector(config, field, ObstacleCollector(config, field))


def make_context(ball=Pose2D(0.0, 0.0, 0.0), opponents=None):
    return PlayContext(
        game_state=GameControlState(),
        teammates={
            1: RobotState(1, Pose2D(ball.x - 0.3, ball.y, 0.0), 10.0),
            2: RobotState(2, Pose2D(ball.x + 1.5, ball.y + 0.8, 0.0), 10.0),
            3: RobotState(3, Pose2D(-5.0, 0.0, 0.0), 10.0),
        },
        opponents=opponents or {},
        ball=BallState(ball.x, ball.y, 10.0),
    )


class BoundedActionSelectorTest(unittest.TestCase):
    def test_candidate_count_and_finalists_have_fixed_bounds(self):
        selection = make_selector().select(
            handler_id=1,
            context=make_context(),
            pressure_level="manageable",
            fallback_target=Pose2D(2.0, 0.0, 0.0),
        )

        self.assertLessEqual(selection.generated_count, 10)
        self.assertLessEqual(len(selection.finalists), 3)
        self.assertIn(
            selection.selected.kind,
            {ActionKind.SHOOT, ActionKind.PASS, ActionKind.DRIBBLE},
        )

    def test_immediate_pressure_removes_dribble_candidates(self):
        selection = make_selector().select(
            handler_id=1,
            context=make_context(),
            pressure_level="immediate",
            fallback_target=Pose2D(2.0, 0.0, 0.0),
        )

        self.assertNotIn(ActionKind.DRIBBLE, {item.kind for item in selection.finalists})

    def test_defensive_area_adds_only_three_bounded_clear_candidates(self):
        selection = make_selector().select(
            handler_id=1,
            context=make_context(Pose2D(-5.0, 0.0, 0.0)),
            pressure_level="immediate",
            fallback_target=Pose2D(7.0, 0.0, 0.0),
        )

        self.assertLessEqual(selection.generated_count, 10)
        self.assertIn(ActionKind.CLEAR, {item.kind for item in selection.finalists})
        self.assertEqual(selection.selected.kind, ActionKind.CLEAR)

    def test_sideline_is_a_hard_recovery_scene(self):
        config = SoccerConfig()
        selector = make_selector()
        fallback = Pose2D(1.0, 0.0, 0.0)
        context = make_context(Pose2D(0.0, config.field_width / 2.0 - 0.05, 0.0))

        selection = selector.select(
            handler_id=1,
            context=context,
            pressure_level="urgent",
            fallback_target=fallback,
        )

        self.assertEqual(selection.generated_count, 1)
        self.assertEqual(selection.selected.kind, ActionKind.RECOVERY)
        self.assertEqual(selection.selected.target, fallback)

    def test_blocked_center_shot_can_choose_another_goal_point(self):
        opponents = {
            1: RobotState(1, Pose2D(3.0, 0.0, 0.0), 10.0),
        }
        selection = make_selector().select(
            handler_id=1,
            context=make_context(opponents=opponents),
            pressure_level="manageable",
            fallback_target=Pose2D(2.0, 0.0, 0.0),
        )
        shot = next(item for item in selection.finalists if item.kind is ActionKind.SHOOT)

        self.assertNotAlmostEqual(shot.target.y, 0.0)

    def test_open_long_shot_uses_power_kick(self):
        selection = make_selector().select(
            handler_id=1,
            context=make_context(Pose2D(0.0, 0.0, 0.0)),
            pressure_level="manageable",
            fallback_target=Pose2D(2.0, 0.0, 0.0),
        )
        shot = next(
            item for item in selection.finalists if item.kind is ActionKind.SHOOT
        )
        self.assertEqual(shot.kick_power, 2.25)

    def test_defensive_clear_uses_strong_kick(self):
        selection = make_selector().select(
            handler_id=1,
            context=make_context(Pose2D(-5.0, 0.0, 0.0)),
            pressure_level="immediate",
            fallback_target=Pose2D(2.0, 0.0, 0.0),
        )
        clear = next(
            item for item in selection.finalists if item.kind is ActionKind.CLEAR
        )
        self.assertEqual(clear.kick_power, 2.20)

    def test_average_selection_cost_stays_below_five_milliseconds(self):
        selector = make_selector()
        context = make_context(
            opponents={
                1: RobotState(1, Pose2D(1.0, -0.5, 0.0), 10.0),
                2: RobotState(2, Pose2D(2.0, 0.5, 0.0), 10.0),
                3: RobotState(3, Pose2D(3.0, 0.0, 0.0), 10.0),
            }
        )
        iterations = 1000

        started = time.perf_counter()
        for _ in range(iterations):
            selector.select(
                handler_id=1,
                context=context,
                pressure_level="urgent",
                fallback_target=Pose2D(2.0, 0.0, 0.0),
            )
        average_ms = (time.perf_counter() - started) * 1000.0 / iterations

        self.assertLess(average_ms, 5.0)

    def test_high_match_risk_increases_shot_utility(self):
        selector = make_selector()
        context = make_context(Pose2D(2.0, 0.0, 0.0))

        conservative = selector.select(
            handler_id=1,
            context=context,
            pressure_level="manageable",
            fallback_target=Pose2D(3.0, 0.0, 0.0),
            risk_value=0.1,
        )
        aggressive = selector.select(
            handler_id=1,
            context=context,
            pressure_level="manageable",
            fallback_target=Pose2D(3.0, 0.0, 0.0),
            risk_value=0.9,
        )
        low_shot = next(
            item for item in conservative.finalists if item.kind is ActionKind.SHOOT
        )
        high_shot = next(
            item for item in aggressive.finalists if item.kind is ActionKind.SHOOT
        )

        self.assertGreater(high_shot.utility, low_shot.utility)


if __name__ == "__main__":
    unittest.main()
