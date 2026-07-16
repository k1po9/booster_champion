import unittest

from tools.cross_validate_ball_motion import Point
from tools.train_ball_trajectory_challengers import (
    PiecewiseLookup,
    PiecewiseResistance,
    actual_position,
)


class BallTrajectoryChallengerTest(unittest.TestCase):
    def test_piecewise_lookup_matches_constant_deceleration(self):
        model = PiecewiseResistance((0.0, 6.0), (2.0, 2.0), 10)
        lookup = PiecewiseLookup(model)

        self.assertAlmostEqual(lookup.distance_after(2.0, None), 1.0, delta=0.02)
        self.assertAlmostEqual(lookup.distance_after(2.0, 0.5), 0.75, delta=0.02)

    def test_actual_position_interpolates_without_future_leakage(self):
        class TrackStub:
            points = (Point(0.0, 0.0, 0.0), Point(1.0, 2.0, 1.0))

        x, y = actual_position(TrackStub(), 0.25)

        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.25)


if __name__ == "__main__":
    unittest.main()
