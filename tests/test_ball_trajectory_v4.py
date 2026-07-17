import unittest

from src.tactics.ball_event_model_data import MODEL_ROWS
from src.tactics.ball_trajectory import (
    BallObservation,
    EventBallTrajectoryPredictor,
    predict_launched_ball_path,
)


class BallTrajectoryV4Tests(unittest.TestCase):
    def test_embedded_model_is_bounded_and_complete(self) -> None:
        self.assertEqual(len(MODEL_ROWS), 338)
        self.assertTrue(all(len(features) == 11 for features, _ in MODEL_ROWS))

    def test_direct_prediction_exposes_bounded_queries(self) -> None:
        points = [
            BallObservation(index / 30.0, 0.065 * index - 0.0015 * index * index, 0.002 * index)
            for index in range(5)
        ]
        prediction = predict_launched_ball_path(points, segment_id=7)

        self.assertTrue(prediction.usable)
        self.assertEqual(prediction.segment_id, 7)
        self.assertIsNotNone(prediction.position_at(0.25))
        self.assertIsNotNone(prediction.position_at(0.50))
        self.assertIsNone(prediction.position_at(1.01))
        self.assertIsNone(prediction.stop_point)
        self.assertLess(prediction.recommended_horizon_sec, prediction.valid_horizon_sec)

    def test_streaming_predictor_detects_launch_and_warms_up(self) -> None:
        predictor = EventBallTrajectoryPredictor()
        predictor.add_observation(BallObservation(0.00, 0.0, 0.0))
        predictor.add_observation(BallObservation(0.04, 0.0, 0.0))
        self.assertFalse(predictor.predict().usable)

        for index, x in enumerate((0.08, 0.155, 0.225, 0.29, 0.35), start=2):
            predictor.add_observation(BallObservation(index * 0.04, x, 0.0))

        prediction = predictor.predict()
        self.assertTrue(prediction.usable)
        self.assertEqual(prediction.sample_count, 5)
        self.assertIsNotNone(prediction.position_ahead(prediction.observed_at_sec, 0.25))

    def test_gap_and_direction_change_invalidate_cached_path(self) -> None:
        predictor = EventBallTrajectoryPredictor()
        for index, x in enumerate((0.0, 0.0, 0.08, 0.155, 0.225, 0.29, 0.35)):
            predictor.add_observation(BallObservation(index * 0.04, x, 0.0))
        self.assertTrue(predictor.predict().usable)

        predictor.add_observation(BallObservation(0.28, 0.28, 0.0))
        self.assertFalse(predictor.predict().usable)
        self.assertEqual(predictor.predict().invalid_reason, "direction_change")


    def test_sharp_speed_drop_invalidates_cached_path(self) -> None:
        predictor = EventBallTrajectoryPredictor()
        for index, x in enumerate((0.0, 0.0, 0.08, 0.155, 0.225, 0.29, 0.35)):
            predictor.add_observation(BallObservation(index * 0.04, x, 0.0))
        self.assertTrue(predictor.predict().usable)

        predictor.add_observation(BallObservation(0.28, 0.351, 0.0))
        self.assertFalse(predictor.predict().usable)
        self.assertEqual(predictor.predict().invalid_reason, "speed_drop")


if __name__ == "__main__":
    unittest.main()
