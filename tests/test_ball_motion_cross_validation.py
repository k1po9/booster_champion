import math
import unittest

from tools.cross_validate_ball_motion import ResistanceModel, stop_time_and_distance


class BallMotionFormulaTest(unittest.TestCase):
    def test_constant_resistance_matches_kinematics(self):
        model = ResistanceModel("constant", 2.0, 0.0, 0, 0)
        stop_time, stop_distance = stop_time_and_distance(model, 2.0)
        self.assertAlmostEqual(stop_time, 1.0)
        self.assertAlmostEqual(stop_distance, 1.0)

    def test_linear_resistance_matches_closed_form(self):
        model = ResistanceModel("linear", 1.0, 0.5, 1, 0)
        stop_time, stop_distance = stop_time_and_distance(model, 2.0)
        self.assertAlmostEqual(stop_time, math.log(2.0) / 0.5)
        self.assertAlmostEqual(stop_distance, 4.0 - 4.0 * math.log(2.0))

    def test_quadratic_resistance_matches_closed_form(self):
        model = ResistanceModel("quadratic", 1.0, 0.25, 2, 0)
        stop_time, stop_distance = stop_time_and_distance(model, 2.0)
        self.assertAlmostEqual(stop_time, math.pi / 2.0)
        self.assertAlmostEqual(stop_distance, 2.0 * math.log(2.0))


if __name__ == "__main__":
    unittest.main()
