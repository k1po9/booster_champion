import unittest

from tools.cross_validate_robot_eta import (
    EtaCalibration,
    accelerated_travel_time,
    translation_predictions,
)


class RobotEtaExperimentTest(unittest.TestCase):
    def test_short_distance_uses_acceleration_triangle(self):
        self.assertAlmostEqual(accelerated_travel_time(0.25, 1.0, 2.0), 0.5)

    def test_long_distance_reaches_cruise_speed(self):
        self.assertAlmostEqual(accelerated_travel_time(1.0, 1.0, 2.0), 1.25)

    def test_dynamics_prediction_adds_reaction_delay(self):
        calibration = EtaCalibration(1.0, 0.1, 2.0, 1.0, 0, 0, 0)
        prediction = translation_predictions(calibration, 1.0, 1.0)
        self.assertAlmostEqual(prediction.baseline_sec, 1.0)
        self.assertAlmostEqual(prediction.empirical_cruise_sec, 1.0)
        self.assertAlmostEqual(prediction.dynamics_sec, 1.35)


if __name__ == '__main__':
    unittest.main()
