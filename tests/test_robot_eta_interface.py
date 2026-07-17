import unittest

from src.soccer_framework import Pose2D
from src.tactics.ball_trajectory import BallObservation, predict_launched_ball_path
from src.tactics.robot_eta import (
    DynamicRobotArrivalEstimator,
    RobotArrivalQuery,
    RobotMotionTracker,
    estimate_earliest_intercept,
)


class RobotEtaInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = DynamicRobotArrivalEstimator()

    def test_dynamic_selector_keeps_baseline_for_ordinary_motion(self) -> None:
        estimate = self.estimator.estimate(
            RobotArrivalQuery(
                now_sec=10.0,
                robot_id=1,
                pose=Pose2D(0.0, 0.0, 0.0),
                raw_target=Pose2D(1.0, 0.0, 0.0),
                observed_linear_speed_mps=0.45,
                observed_yaw_rate_radps=0.05,
                target_age_sec=1.5,
                target_changed=False,
            )
        )

        self.assertTrue(estimate.reachable)
        self.assertEqual(estimate.model_name, "baseline_v4_balanced_ridge_0.1")
        self.assertLessEqual(estimate.earliest_sec, estimate.eta_sec)
        self.assertGreaterEqual(estimate.latest_sec, estimate.eta_sec)
        self.assertIsNotNone(estimate.alternate_eta_sec)

    def test_dynamic_selector_uses_team2_for_difficult_in_domain_motion(self) -> None:
        estimate = self.estimator.estimate(
            RobotArrivalQuery(
                now_sec=10.0,
                robot_id=2,
                pose=Pose2D(0.0, 0.0, 0.0),
                raw_target=Pose2D(1.4, 1.0, 1.0),
                adjusted_target=Pose2D(1.0, 0.8, 0.8),
                planned_path_length_m=2.0,
                observed_linear_speed_mps=0.20,
                observed_yaw_rate_radps=0.30,
                linear_speed_limit_mps=0.60,
                target_changed=True,
            )
        )

        self.assertEqual(estimate.model_name, "team2_v4_balanced_ridge_0.1")
        self.assertIn("difficult", estimate.selection_reason)
        self.assertGreater(estimate.uncertainty_sec, 0.0)

    def test_non_walkable_robot_is_explicitly_unreachable(self) -> None:
        estimate = self.estimator.estimate(
            RobotArrivalQuery(
                now_sec=10.0,
                robot_id=3,
                pose=Pose2D(),
                raw_target=Pose2D(1.0, 0.0, 0.0),
                fall_state="fallen",
            )
        )

        self.assertFalse(estimate.reachable)
        self.assertEqual(estimate.invalid_reason, "robot_not_upright")
        self.assertIsNone(estimate.eta_sec)

    def test_motion_tracker_estimates_public_pose_velocity(self) -> None:
        tracker = RobotMotionTracker()
        tracker.update(1, Pose2D(0.0, 0.0, 0.0), 1.0)
        tracker.update(1, Pose2D(0.2, 0.0, 0.1), 1.2)

        motion = tracker.motion(1)

        self.assertIsNotNone(motion)
        assert motion is not None
        self.assertAlmostEqual(motion.linear_speed_mps, 1.0)
        self.assertAlmostEqual(motion.yaw_rate_radps, 0.5)

    def test_intercept_search_is_bounded_and_keeps_fallback(self) -> None:
        trajectory = predict_launched_ball_path(
            [
                BallObservation(i / 30.0, 0.04 * i, 0.0)
                for i in range(5)
            ]
        )
        query = RobotArrivalQuery(
            now_sec=trajectory.observed_at_sec,
            robot_id=1,
            pose=Pose2D(trajectory.current.x - 0.05, 0.0, 0.0),
            raw_target=trajectory.current,
            observed_linear_speed_mps=0.2,
            observed_yaw_rate_radps=0.0,
        )

        intercept = estimate_earliest_intercept(
            self.estimator, query, trajectory
        )

        self.assertEqual(intercept.robot_id, 1)
        self.assertEqual(intercept.fallback_point, trajectory.current)
        self.assertIn(
            intercept.reason,
            {
                "earliest bounded reachable ball point",
                "no confirmed intercept within calibrated horizon",
            },
        )


if __name__ == "__main__":
    unittest.main()
