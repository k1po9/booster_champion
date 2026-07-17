import importlib.util
import math
import unittest
from collections import Counter

from src.match_data_recorder import _eta_experiment_record
from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    Penalty,
    PlayContext,
    Pose2D,
    RobotState,
    SetPlay,
    SoccerConfig,
)
from src.tactics.eta_experiment import (
    ETA_SCENARIOS,
    EtaExperimentCoordinator,
)
from src.tactics.geometry import TeamFieldFrame


class _Kit:
    def __init__(self) -> None:
        self.config = SoccerConfig()
        self.field = TeamFieldFrame(self.config)


def _context(
    now: float,
    p1: Pose2D = Pose2D(0.0, 0.0, 0.0),
    p2: Pose2D = Pose2D(0.0, 2.0, 0.0),
    p3: Pose2D = Pose2D(-5.75, 0.0, 0.0),
) -> PlayContext:
    game = GameControlState(state=GameState.PLAYING)
    game.last_seen_at = now
    return PlayContext(
        ball=BallState(x=0.0, y=0.0, last_seen_at=now),
        game_state=game,
        teammates={
            1: RobotState(1, p1, now),
            2: RobotState(2, p2, now),
            3: RobotState(3, p3, now),
        },
    )


class EtaExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kit = _Kit()
        self.coordinator = EtaExperimentCoordinator(self.kit)

    @unittest.skipUnless(
        importlib.util.find_spec("py_trees") is not None,
        "runtime dependency py_trees is not installed",
    )
    def test_collection_playbook_is_default_on_recorder_branch(self) -> None:
        from src.play import (
            PLAYBOOKS,
            ROLE_ETA_BALL_GUARD,
            ROLE_ETA_EXPERIMENT,
            ROLE_GOALKEEPER,
            EtaBallGuardRole,
            EtaExperimentPlaybook,
        )

        playbook = PLAYBOOKS.create_default(self.kit)
        self.assertIsInstance(playbook, EtaExperimentPlaybook)
        self.assertIn("normal-match", PLAYBOOKS.names())
        self.assertIn("eta-experiment", PLAYBOOKS.names())
        assignment = playbook.assign_roles(_context(5.0))
        self.assertEqual(assignment.role_of(1), ROLE_ETA_EXPERIMENT)
        self.assertEqual(assignment.role_of(2), ROLE_ETA_BALL_GUARD)
        self.assertEqual(assignment.role_of(3), ROLE_GOALKEEPER)
        guard_target = EtaBallGuardRole().kick_target(
            self.kit,
            2,
            _context(5.1),
        )
        self.assertGreater(guard_target.x, 4.0)
        self.assertGreater(abs(guard_target.y), 3.0)

    def test_target_is_held_through_arrival_confirmation(self) -> None:
        self.coordinator.update(_context(10.0))
        first = self.coordinator.target_for(1)
        self.assertEqual(self.coordinator.episode.active_player, 1)

        self.coordinator.update(_context(10.10, p1=first))
        self.assertEqual(self.coordinator.target_for(1), first)
        self.assertEqual(self.coordinator.scenario_index, 0)

        self.coordinator.update(_context(10.20, p1=first))
        self.assertEqual(self.coordinator.target_for(1), first)
        self.assertEqual(self.coordinator.scenario_index, 0)

        self.coordinator.update(_context(10.40, p1=first))
        self.assertEqual(self.coordinator.scenario_index, 1)
        self.assertEqual(self.coordinator.episode.active_player, 2)
        self.assertEqual(self.coordinator.episode.speed_limit_mps, 0.6)

    def test_first_matrix_balances_scenarios_speeds_and_players(self) -> None:
        context = _context(11.0)
        self.coordinator.update(context)
        observed: list[tuple[str, float, int]] = []
        total = len(ETA_SCENARIOS) * len(self.coordinator.speed_levels())
        for index in range(total):
            episode = self.coordinator.episode
            self.assertIsNotNone(episode)
            observed.append(
                (
                    episode.scenario.name,
                    episode.speed_limit_mps,
                    episode.active_player,
                )
            )
            self.coordinator._advance(
                context,
                11.1 + index,
                self.coordinator.available_players(context),
            )

        self.assertEqual(
            Counter(speed for _, speed, _ in observed),
            Counter({0.4: 14, 0.6: 14, 0.8: 14}),
        )
        self.assertEqual(
            Counter(player for _, _, player in observed),
            Counter({1: 21, 2: 21}),
        )
        for scenario in ETA_SCENARIOS:
            self.assertEqual(
                {
                    speed
                    for name, speed, _ in observed
                    if name == scenario.name
                },
                {0.4, 0.6, 0.8},
            )
        self.assertEqual(
            [(name, speed) for name, speed, _ in observed[:6]],
            [
                ("long_oblique_right", 0.4),
                ("medium_quarter_right", 0.6),
                ("long_final_reverse", 0.8),
                ("medium_quarter_left", 0.4),
                ("teammate_avoidance", 0.6),
                ("long_reverse", 0.8),
            ],
        )
        avoidance_index = next(
            index
            for index, (name, _, _) in enumerate(observed)
            if name == "teammate_avoidance"
        )
        self.assertLess(avoidance_index, 5)

    def test_collection_pauses_for_kickoff_and_opponent_restart(self) -> None:
        context = _context(11.0)
        self.coordinator.update(context)
        self.assertIsNotNone(self.coordinator.episode)

        game = context.known_game
        game.kicking_team = self.kit.config.team_id
        game.secondary_time = 10
        self.assertTrue(self.coordinator.collection_paused(context))
        self.coordinator.pause()
        self.assertIsNone(self.coordinator.episode)
        self.assertTrue(
            all("mode=parking" in reason for reason in self.coordinator.reasons.values())
        )

        game.secondary_time = 0
        game.set_play = SetPlay.CORNER_KICK
        game.kicking_team = self.kit.config.opponent_team_id()
        self.assertTrue(self.coordinator.collection_paused(context))

        game.kicking_team = self.kit.config.team_id
        self.assertFalse(self.coordinator.collection_paused(context))

        game.set_play = SetPlay.NONE
        self.assertFalse(self.coordinator.collection_paused(context))

    def test_observation_gap_restarts_same_scenario(self) -> None:
        self.coordinator.update(_context(12.0))
        first = self.coordinator.episode
        self.assertIsNotNone(first)

        self.coordinator.update(
            _context(13.5, p1=Pose2D(0.20, 0.0, 0.0))
        )
        self.assertGreater(self.coordinator.episode.number, first.number)
        self.assertEqual(self.coordinator.scenario_index, 0)
        self.assertEqual(self.coordinator.episode.start_pose.x, 0.20)

    def test_penalized_player_is_not_selected(self) -> None:
        context = _context(14.0)
        player = context.known_game.get_player_state(self.kit.config.team_id, 1)
        self.assertIsNotNone(player)
        player.penalty = Penalty.ILLEGAL_POSITIONING
        self.coordinator.update(context)
        self.assertEqual(self.coordinator.episode.active_player, 2)

    def test_avoidance_waits_for_two_available_field_players(self) -> None:
        self.coordinator.scenario_index = next(
            index
            for index, scenario in enumerate(ETA_SCENARIOS)
            if scenario.teammate_avoidance
        )
        context = _context(15.0)
        player = context.known_game.get_player_state(self.kit.config.team_id, 2)
        self.assertIsNotNone(player)
        player.penalty = Penalty.ILLEGAL_POSITIONING
        self.coordinator.update(context)
        self.assertIsNone(self.coordinator.episode)

    def test_all_relative_targets_remain_inside_safe_field(self) -> None:
        edge_poses = (
            Pose2D(6.45, 4.0, 0.0),
            Pose2D(-6.45, -4.0, math.pi),
            Pose2D(0.0, 0.0, 1.5),
        )
        for scenario in ETA_SCENARIOS:
            for pose in edge_poses:
                target = self.coordinator._relative_target(pose, scenario)
                self.assertLessEqual(abs(target.x), 6.55)
                self.assertLessEqual(abs(target.y), 4.05)

    def test_moving_retarget_changes_once_then_holds(self) -> None:
        self.coordinator.scenario_index = next(
            index
            for index, scenario in enumerate(ETA_SCENARIOS)
            if scenario.moving_retarget
        )
        self.coordinator.update(_context(20.0))
        warmup = self.coordinator.target_for(1)
        self.assertIn("mode=moving_warmup", self.coordinator.reason_for(1))

        moving_pose = Pose2D(0.20, 0.0, 0.0)
        self.coordinator.update(_context(21.0, p1=moving_pose))
        final = self.coordinator.target_for(1)
        self.assertNotEqual(final, warmup)
        self.assertIn("mode=moving_start", self.coordinator.reason_for(1))

        self.coordinator.update(_context(21.5, p1=Pose2D(0.30, 0.0, 0.0)))
        self.assertEqual(self.coordinator.target_for(1), final)

    def test_avoidance_scenario_positions_blocker_before_release(self) -> None:
        self.coordinator.scenario_index = next(
            index
            for index, scenario in enumerate(ETA_SCENARIOS)
            if scenario.teammate_avoidance
        )
        start = Pose2D(0.0, 0.0, 0.0)
        self.coordinator.update(_context(30.0, p1=start))
        episode = self.coordinator.episode
        self.assertEqual(episode.stage, "avoidance_prepare")
        self.assertEqual(self.coordinator.target_for(1), start)
        blocker_target = self.coordinator.target_for(2)
        self.assertIn("mode=blocker_prepare", self.coordinator.reason_for(2))

        self.coordinator.update(_context(31.0, p1=start, p2=blocker_target))
        self.assertEqual(self.coordinator.episode.stage, "active")
        self.assertNotEqual(self.coordinator.target_for(1), start)
        self.assertIn("mode=avoidance_active", self.coordinator.reason_for(1))

    def test_reason_is_parsed_into_typed_json_fields(self) -> None:
        record = _eta_experiment_record(
            "eta_exp|scenario=long_reverse|episode=7|active=2"
            "|mode=rest_start|distance=3.00|speed=0.60"
            "|path_error=3.142|final_error=0.000"
        )
        self.assertEqual(
            record,
            {
                "scenario": "long_reverse",
                "episode": 7,
                "active": 2,
                "mode": "rest_start",
                "distance": 3.0,
                "speed": 0.6,
                "path_error": 3.142,
                "final_error": 0.0,
            },
        )
        self.assertIsNone(_eta_experiment_record("supporter hold"))


if __name__ == "__main__":
    unittest.main()
