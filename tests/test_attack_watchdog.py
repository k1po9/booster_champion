import unittest

from src.tactics.attack_watchdog import (
    AttackAttemptObservation,
    AttackPhase,
    AttackWatchdog,
    AttackWatchdogConfig,
)


class AttackWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.watchdog = AttackWatchdog(
            AttackWatchdogConfig(
                total_timeout_sec=5.0,
                no_progress_timeout_sec=1.0,
                approach_timeout_sec=3.0,
                align_timeout_sec=1.5,
                execute_timeout_sec=2.0,
            )
        )

    def observation(self, t, phase=AttackPhase.APPROACH, distance=1.0, angle=0.5, speed=0.0):
        return AttackAttemptObservation(t, phase, distance, angle, speed)

    def test_progress_refreshes_stall_deadline(self):
        self.watchdog.update(1, self.observation(0.0, distance=1.0))
        status = self.watchdog.update(1, self.observation(0.8, distance=0.90))
        self.assertFalse(status.stalled)

        status = self.watchdog.update(1, self.observation(1.7, distance=0.90))
        self.assertFalse(status.stalled)
        status = self.watchdog.update(1, self.observation(1.9, distance=0.90))
        self.assertTrue(status.stalled)
        self.assertTrue(status.should_escape)

    def test_phase_change_resets_phase_and_progress_clock(self):
        self.watchdog.update(1, self.observation(0.0))
        status = self.watchdog.update(1, self.observation(0.9, AttackPhase.ALIGN, 0.4, 0.4))
        self.assertEqual(status.phase, AttackPhase.ALIGN)
        self.assertAlmostEqual(status.phase_elapsed_sec, 0.0)
        self.assertFalse(status.should_escape)

    def test_phase_timeout_and_total_timeout_are_distinct(self):
        self.watchdog.update(1, self.observation(0.0, AttackPhase.ALIGN, 0.4, 0.4))
        status = self.watchdog.update(1, self.observation(1.6, AttackPhase.ALIGN, 0.3, 0.2))
        self.assertTrue(status.phase_timed_out)
        self.assertFalse(status.total_timed_out)

        self.watchdog.reset(1)
        self.watchdog.update(1, self.observation(0.0))
        status = self.watchdog.update(1, self.observation(5.1, distance=0.1))
        self.assertTrue(status.total_timed_out)

    def test_ball_release_completes_and_resets_attempt(self):
        self.watchdog.update(1, self.observation(0.0, AttackPhase.EXECUTE, 0.3, 0.1))
        status = self.watchdog.update(1, self.observation(0.3, AttackPhase.EXECUTE, 0.3, 0.1, 0.5))
        self.assertTrue(status.completed)
        self.assertFalse(status.should_escape)
        self.assertFalse(self.watchdog.status(1, 0.4).active)

    def test_pressure_limits_safe_action_budget(self):
        status = self.watchdog.update(1, self.observation(0.0))
        self.assertAlmostEqual(status.safe_action_budget(0.9, safety_margin_sec=0.2), 0.7)


if __name__ == '__main__':
    unittest.main()
