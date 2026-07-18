import unittest

from src.soccer_framework import Pose2D
from src.tactics.team_coordination import (
    KeeperTakeoverCoordinator,
    KeeperTakeoverEvidence,
    marker_target,
    own_goal_safe_target,
    select_dangerous_opponent,
)


class KeeperTakeoverCoordinatorTest(unittest.TestCase):
    def test_takeover_is_exclusive_until_minimum_hold_then_recovers(self):
        coordinator = KeeperTakeoverCoordinator(
            minimum_hold_sec=0.6,
            maximum_hold_sec=3.0,
            recover_hold_sec=0.8,
        )
        entered = coordinator.update(
            10.0,
            KeeperTakeoverEvidence(True, True, True, True, reason="threat"),
        )
        self.assertTrue(entered.active)

        held = coordinator.update(
            10.3,
            KeeperTakeoverEvidence(True, False, False, True),
        )
        self.assertTrue(held.active)

        released = coordinator.update(
            10.7,
            KeeperTakeoverEvidence(True, False, False, True),
        )
        self.assertFalse(released.active)
        self.assertTrue(released.recovering)

        normal = coordinator.update(
            11.6,
            KeeperTakeoverEvidence(True, False, False, True),
        )
        self.assertFalse(normal.active)
        self.assertFalse(normal.recovering)

    def test_maximum_duration_forces_recovery(self):
        coordinator = KeeperTakeoverCoordinator(
            minimum_hold_sec=0.1,
            maximum_hold_sec=1.0,
        )
        coordinator.update(
            1.0,
            KeeperTakeoverEvidence(True, True, True, True),
        )
        status = coordinator.update(
            2.1,
            KeeperTakeoverEvidence(True, True, True, True),
        )
        self.assertFalse(status.active)
        self.assertTrue(status.recovering)
        self.assertIn("maximum", status.reason)

    def test_recovery_hold_prevents_immediate_reentry(self):
        coordinator = KeeperTakeoverCoordinator(
            minimum_hold_sec=0.1,
            recover_hold_sec=0.8,
        )
        coordinator.update(
            1.0,
            KeeperTakeoverEvidence(True, True, True, True),
        )
        coordinator.update(
            1.2,
            KeeperTakeoverEvidence(True, False, False, True),
        )
        held = coordinator.update(
            1.3,
            KeeperTakeoverEvidence(True, True, True, True),
        )
        self.assertFalse(held.active)
        self.assertTrue(held.recovering)

        renewed = coordinator.update(
            2.1,
            KeeperTakeoverEvidence(True, True, True, True),
        )
        self.assertTrue(renewed.active)


class MarkerGeometryTest(unittest.TestCase):
    def test_selects_goal_side_off_ball_threat_and_excludes_carrier(self):
        threat = select_dangerous_opponent(
            (
                (1, Pose2D(-1.0, 0.0, 0.0)),
                (2, Pose2D(-5.0, 0.7, 0.0)),
                (3, Pose2D(1.0, -2.5, 0.0)),
            ),
            ball=Pose2D(-1.0, 0.0, 0.0),
            own_goal=Pose2D(-7.0, 0.0, 0.0),
            excluded_player_id=1,
        )
        self.assertIsNotNone(threat)
        assert threat is not None
        self.assertEqual(threat.player_id, 2)

    def test_previous_mark_is_retained_inside_switch_margin(self):
        threat = select_dangerous_opponent(
            (
                (2, Pose2D(-4.0, 0.5, 0.0)),
                (3, Pose2D(-4.05, -0.5, 0.0)),
            ),
            ball=Pose2D(-1.0, 0.0, 0.0),
            own_goal=Pose2D(-7.0, 0.0, 0.0),
            previous_player_id=2,
            switch_margin=0.5,
        )
        self.assertIsNotNone(threat)
        assert threat is not None
        self.assertEqual(threat.player_id, 2)

    def test_deep_marker_target_never_overshoots_own_goal(self):
        opponent = Pose2D(-6.8, 0.1, 0.0)
        own_goal = Pose2D(-7.0, 0.0, 0.0)
        target = marker_target(
            opponent,
            ball=Pose2D(-5.0, 0.0, 0.0),
            own_goal=own_goal,
            marking_distance_m=0.75,
        )
        self.assertGreater(target.x, own_goal.x)
        self.assertLess(target.x, opponent.x)

    def test_goal_safe_target_stays_clear_of_frame(self):
        target = own_goal_safe_target(
            Pose2D(-6.9, 1.0, 0.0), own_goal_x=-7.0
        )
        self.assertGreaterEqual(target.x, -6.45)

    def test_marker_target_is_between_opponent_and_goal(self):
        opponent = Pose2D(-3.0, 1.0, 0.0)
        target = marker_target(
            opponent,
            ball=Pose2D(-1.0, 0.0, 0.0),
            own_goal=Pose2D(-7.0, 0.0, 0.0),
            marking_distance_m=0.75,
        )
        self.assertLess(target.x, opponent.x)
        self.assertLess(abs(target.y), abs(opponent.y))


if __name__ == "__main__":
    unittest.main()
