import unittest

from src.soccer_framework import Pose2D, RobotState
from src.tactics.ball_prediction import BallObservation, predict_ball_stop
from src.tactics.opponent_pressure import (
    OpponentMotionTracker,
    OpponentPressureConfig,
    OpponentPressureEstimator,
)


class OpponentPressureTest(unittest.TestCase):
    def test_stationary_target_uses_conservative_max_speed(self):
        estimator = OpponentPressureEstimator(
            OpponentPressureConfig(
                opponent_max_speed_mps=0.8,
                control_radius_m=0.5,
                prediction_horizon_sec=5.0,
            )
        )
        report = estimator.estimate(
            now_sec=10.0,
            opponents={1: RobotState(1, Pose2D(0.0, 0.0, 0.0), 10.0)},
            target=Pose2D(2.0, 0.0, 0.0),
        )
        self.assertIsNotNone(report.earliest)
        assert report.earliest is not None
        self.assertAlmostEqual(report.earliest.time_to_pressure_sec, 1.875)
        self.assertEqual(report.earliest.player_id, 1)

    def test_moving_ball_path_changes_contest_time_and_point(self):
        prediction = predict_ball_stop(
            [
                BallObservation(0.0, 2.0, 0.0),
                BallObservation(0.1, 2.1, 0.0),
                BallObservation(0.2, 2.2, 0.0),
                BallObservation(0.3, 2.3, 0.0),
            ],
            decel_mps2=2.0,
        )
        assert prediction is not None
        estimator = OpponentPressureEstimator(
            OpponentPressureConfig(
                opponent_max_speed_mps=0.8,
                control_radius_m=0.5,
                prediction_horizon_sec=5.0,
                search_step_sec=0.02,
            )
        )
        report = estimator.estimate(
            now_sec=1.0,
            opponents={1: RobotState(1, Pose2D(0.0, 0.0, 0.0), 1.0)},
            ball_prediction=prediction,
        )
        self.assertIsNotNone(report.earliest)
        assert report.earliest is not None
        self.assertGreater(report.earliest.time_to_pressure_sec, (2.3 - 0.5) / 0.8)
        assert report.earliest.contest_point is not None
        self.assertGreater(report.earliest.contest_point.x, 2.3)

    def test_stale_and_inactive_opponents_are_not_used(self):
        class Game:
            def is_active_player(self, team_id, player_id):
                return player_id != 2

        estimator = OpponentPressureEstimator(OpponentPressureConfig(max_pose_age_sec=0.5))
        report = estimator.estimate(
            now_sec=2.0,
            opponents={
                1: RobotState(1, Pose2D(0.0, 0.0, 0.0), 1.0),
                2: RobotState(2, Pose2D(0.1, 0.0, 0.0), 2.0),
            },
            target=Pose2D(1.0, 0.0, 0.0),
            game_state=Game(),
            opponent_team_id=2,
        )
        self.assertEqual(report.stale_opponents, 1)
        self.assertEqual(report.inactive_opponents, 1)
        self.assertEqual(report.considered_opponents, 0)
        self.assertTrue(report.has_unknown_pressure)

    def test_tracker_estimates_observed_velocity(self):
        tracker = OpponentMotionTracker(window_sec=0.5)
        tracker.update({1: RobotState(1, Pose2D(0.0, 0.0, 0.0), 1.0)}, 1.0)
        tracker.update({1: RobotState(1, Pose2D(0.1, 0.0, 0.0), 1.1)}, 1.1)
        tracker.update({1: RobotState(1, Pose2D(0.2, 0.0, 0.0), 1.2)}, 1.2)
        motion = tracker.motion(1)
        self.assertIsNotNone(motion)
        assert motion is not None
        self.assertAlmostEqual(motion.velocity_x, 1.0, delta=1e-6)
        self.assertAlmostEqual(motion.velocity_y, 0.0, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
