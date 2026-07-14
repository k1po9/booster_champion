import unittest

from src.tactics.ball_prediction import (
    BallObservation,
    DEFAULT_BALL_RESISTANCE_MODEL,
    SlidingWindowBallPredictor,
    predict_ball_motion,
    predict_ball_stop,
)


class BallPredictionTest(unittest.TestCase):
    def test_static_ball_predicts_current_position(self):
        observations = [
            BallObservation(0.0, 1.0, -0.5),
            BallObservation(0.5, 1.01, -0.5),
            BallObservation(1.0, 1.0, -0.49),
        ]

        prediction = predict_ball_stop(observations)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertLess(prediction.speed_mps, 0.05)
        self.assertAlmostEqual(prediction.stop.x, 1.0, delta=0.03)
        self.assertAlmostEqual(prediction.stop.y, -0.49, delta=0.03)

    def test_rolling_ball_projects_forward(self):
        observations = [
            BallObservation(0.0, 0.0, 0.0),
            BallObservation(0.3, 0.3, 0.0),
            BallObservation(0.6, 0.6, 0.0),
            BallObservation(0.9, 0.9, 0.0),
        ]

        prediction = predict_ball_stop(observations, decel_mps2=2.0)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertGreater(prediction.stop.x, 1.1)
        self.assertAlmostEqual(prediction.stop.y, 0.0, delta=0.02)
        self.assertGreater(prediction.confidence, 0.4)

    def test_sliding_window_discards_old_observations(self):
        predictor = SlidingWindowBallPredictor(window_sec=0.5, decel_mps2=2.0)
        for i in range(10):
            predictor.add_observation(BallObservation(i * 0.2, float(i), 0.0))

        observations = predictor.observations()

        self.assertLessEqual(observations[-1].t_sec - observations[0].t_sec, 0.5)
        self.assertLess(len(observations), 10)
        self.assertIsNotNone(predictor.predict())

    def test_default_model_exposes_future_position_and_uncertainty(self):
        observations = [
            BallObservation(t, 2.0 * t - 0.5 * t * t, 0.0)
            for t in (0.0, 0.1, 0.2, 0.3, 0.4)
        ]

        prediction = predict_ball_motion(observations)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.resistance_model, DEFAULT_BALL_RESISTANCE_MODEL)
        self.assertAlmostEqual(prediction.speed_mps, 1.6, delta=0.08)
        self.assertGreater(prediction.position_at(0.2).x, prediction.current.x)
        self.assertLessEqual(prediction.position_at(10.0).x, prediction.stop.x + 1e-9)
        self.assertGreater(prediction.uncertainty_radius_m, 0.0)

    def test_predictor_resets_on_direction_reversal(self):
        predictor = SlidingWindowBallPredictor(window_sec=0.5)
        predictor.add_observation(BallObservation(0.0, 0.0, 0.0))
        predictor.add_observation(BallObservation(0.1, 0.1, 0.0))
        predictor.add_observation(BallObservation(0.2, 0.2, 0.0))
        predictor.add_observation(BallObservation(0.3, 0.05, 0.0))

        observations = predictor.observations()

        self.assertEqual(len(observations), 2)
        self.assertAlmostEqual(observations[0].x, 0.2)
        self.assertAlmostEqual(observations[1].x, 0.05)


if __name__ == "__main__":
    unittest.main()
