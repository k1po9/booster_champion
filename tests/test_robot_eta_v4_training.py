import math
import unittest

from tools.train_robot_eta_v4 import EtaExample, fit_ridge


def example(index: int, *, episode_start: bool) -> EtaExample:
    distance = 0.5 + 0.25 * index
    return EtaExample(
        match_file=f"match-{index % 3}",
        match_id=1,
        eta_id=index,
        player_id=1,
        distance_m=distance,
        path_heading_error_rad=0.2 + 0.05 * index,
        final_heading_error_rad=0.4,
        speed_cap_mps=0.4 + 0.2 * (index % 3),
        arrive_distance_m=0.15,
        observed_speed_mps=0.1,
        observed_yaw_rate_radps=0.05,
        start_phase="turn" if index % 2 else "run",
        relative_final_heading_error_rad=0.3,
        control_distance_m=distance,
        avoidance_applied=False,
        command_speed_mps=0.4,
        command_yaw_rate_radps=0.2,
        target_age_sec=0.0 if episode_start else 0.5,
        duration_sec=1.0 + distance * 2.0,
        is_episode_start=episode_start,
    )


class RobotEtaV4TrainingTests(unittest.TestCase):
    def test_feature_contract_has_fixed_size_and_finite_values(self) -> None:
        features = example(2, episode_start=True).features()
        self.assertEqual(len(features), 23)
        self.assertTrue(all(math.isfinite(value) for value in features))

    def test_balanced_ridge_is_bounded_and_predicts_positive_eta(self) -> None:
        rows = [example(index, episode_start=index % 2 == 0) for index in range(12)]
        model = fit_ridge(rows, 0.1, False, balance_episode_starts=True)

        prediction = model.predict(rows[3])

        self.assertGreaterEqual(prediction, 0.05)
        self.assertEqual(len(model.coefficients), 24)
        self.assertEqual(len(model.scaler.means), 23)


if __name__ == "__main__":
    unittest.main()
