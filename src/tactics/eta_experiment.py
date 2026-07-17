"""Deterministic PLAY strategy for collecting complete robot ETA episodes.

This module deliberately optimizes dataset coverage rather than match score. It
keeps every requested target stable until the recorder has observed arrival,
then walks through a finite matrix of distance, turn, final-heading, moving
retarget, and teammate-avoidance cases. Only standard Python math and small
constant-size state are used in the 30 Hz path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from ..soccer_framework import GameState, PlayContext, Pose2D, SetPlay
from .geometry import normalize_angle

if TYPE_CHECKING:
    from ..runtime import SoccerKit


ROLE_ETA_EXPERIMENT = "eta_experiment"
_ARRIVAL_DISTANCE_M = 0.15
_ARRIVAL_ANGLE_RAD = 0.20
_ARRIVAL_HOLD_SEC = 0.25
_UPDATE_GAP_RESET_SEC = 1.00
_EPISODE_TIMEOUT_SEC = 20.5
_MOVING_RETARGET_MIN_SEC = 0.85
_MOVING_RETARGET_MAX_SEC = 1.80
_MOVING_RETARGET_MIN_TRAVEL_M = 0.12


@dataclass(frozen=True)
class EtaScenario:
    name: str
    distance_m: float
    path_error_rad: float
    final_heading_offset_rad: float
    moving_retarget: bool = False
    teammate_avoidance: bool = False


# A compact covering array, not a full Cartesian product. Repeated runs cycle
# deterministically while MotionController rotates speed caps independently.
ETA_SCENARIOS: tuple[EtaScenario, ...] = (
    EtaScenario("short_straight", 0.60, 0.00, 0.00),
    EtaScenario("short_turn_left", 0.80, 0.75, 0.00),
    EtaScenario("short_turn_right", 0.80, -0.75, 0.00),
    EtaScenario("medium_straight_align_left", 1.50, 0.00, 1.20),
    EtaScenario("medium_straight_align_right", 1.50, 0.00, -1.20),
    EtaScenario("medium_quarter_left", 1.80, 1.45, 0.00),
    EtaScenario("medium_quarter_right", 1.80, -1.45, 0.00),
    EtaScenario("long_oblique_left", 3.20, 0.70, 0.65),
    EtaScenario("long_oblique_right", 3.20, -0.70, -0.65),
    EtaScenario("long_reverse", 3.00, math.pi, 0.00),
    EtaScenario("long_final_reverse", 3.60, 0.00, math.pi),
    EtaScenario("moving_retarget_left", 2.60, 0.85, 0.50, moving_retarget=True),
    EtaScenario("moving_retarget_right", 2.60, -0.85, -0.50, moving_retarget=True),
    EtaScenario("teammate_avoidance", 3.00, 0.00, 0.00, teammate_avoidance=True),
)


@dataclass
class _Episode:
    number: int
    scenario: EtaScenario
    active_player: int
    target: Pose2D
    reason: str
    started_at: float
    stage: str = "active"
    start_pose: Pose2D | None = None
    final_target: Pose2D | None = None
    blocker_player: int | None = None
    arrived_at: float | None = None


class EtaExperimentCoordinator:
    """Small deterministic state machine shared by all ETA experiment roles."""

    def __init__(self, kit: "SoccerKit") -> None:
        self.kit = kit
        self.targets: dict[int, Pose2D] = {
            player_id: self._parking_target(player_id)
            for player_id in kit.config.player_ids
        }
        self.reasons: dict[int, str] = {
            player_id: f"eta_exp|mode=parking|player={player_id}"
            for player_id in kit.config.player_ids
        }
        self.scenario_index = 0
        self.active_index = 0
        self.episode_number = 0
        self.episode: _Episode | None = None
        self._last_update_at: float | None = None

    def collection_paused(self, context: PlayContext) -> bool:
        """Return whether rule guards temporarily own all experiment motion."""

        game = context.game_state
        if game is None or game.state != GameState.PLAYING or game.stopped:
            return True
        kickoff_active = (
            game.set_play == SetPlay.NONE
            and game.secondary_time > 0
            and game.has_kicking_team()
        )
        opponent_restart = (
            game.set_play != SetPlay.NONE
            and game.has_kicking_team()
            and game.kicking_team != self.kit.config.team_id
        )
        return kickoff_active or opponent_restart

    def pause(self) -> None:
        """Cancel the current episode without advancing its scenario.

        The behavior tree owns legal ready-position holding and opponent-restart
        avoidance while paused. Parking targets are restored so a resumed
        experiment cannot accidentally reuse a stale active or blocker target.
        """

        self.episode = None
        for player_id in self.kit.config.player_ids:
            self.targets[player_id] = self._parking_target(player_id)
            self.reasons[player_id] = (
                f"eta_exp|mode=parking|player={player_id}"
            )

    def update(self, context: PlayContext) -> None:
        now = self._observation_time(context)
        if (
            self._last_update_at is not None
            and now - self._last_update_at > _UPDATE_GAP_RESET_SEC
        ):
            self.pause()
        self._last_update_at = now
        candidates = self._experiment_players(context)
        if self.episode is not None:
            participants = {self.episode.active_player}
            if self.episode.blocker_player is not None:
                participants.add(self.episode.blocker_player)
            if not participants.issubset(candidates):
                self.pause()
        if not candidates:
            return
        if self.episode is None:
            self._start_episode(context, now, candidates)
            return

        episode = self.episode
        robot = context.teammates.get(episode.active_player)
        if robot is None or robot.pose is None:
            return

        if episode.stage == "moving_warmup":
            assert episode.start_pose is not None
            elapsed = now - episode.started_at
            travelled = math.hypot(
                robot.pose.x - episode.start_pose.x,
                robot.pose.y - episode.start_pose.y,
            )
            if (
                elapsed >= _MOVING_RETARGET_MAX_SEC
                or (
                    elapsed >= _MOVING_RETARGET_MIN_SEC
                    and travelled >= _MOVING_RETARGET_MIN_TRAVEL_M
                )
            ):
                target = self._relative_target(robot.pose, episode.scenario)
                episode.target = target
                episode.final_target = target
                episode.stage = "active"
                episode.started_at = now
                episode.arrived_at = None
                self.targets[episode.active_player] = target
                self.reasons[episode.active_player] = self._reason(episode, "moving_start")
            return

        if episode.stage == "avoidance_prepare":
            blocker = (
                context.teammates.get(episode.blocker_player)
                if episode.blocker_player is not None else None
            )
            blocker_target = (
                self.targets.get(episode.blocker_player)
                if episode.blocker_player is not None else None
            )
            blocker_ready = (
                blocker is not None
                and blocker.pose is not None
                and blocker_target is not None
                and self._arrived(blocker.pose, blocker_target)
            )
            if blocker_ready or now - episode.started_at >= 8.0:
                assert episode.final_target is not None
                episode.target = episode.final_target
                episode.stage = "active"
                episode.started_at = now
                episode.arrived_at = None
                self.targets[episode.active_player] = episode.target
                self.reasons[episode.active_player] = self._reason(
                    episode, "avoidance_active"
                )
            return

        if self._arrived(robot.pose, episode.target):
            if episode.arrived_at is None:
                # Keep issuing the same target so MatchDataRecorder receives an
                # explicit ``phase=arrived`` command before the next target.
                episode.arrived_at = now
                return
            if now - episode.arrived_at >= _ARRIVAL_HOLD_SEC:
                self._advance(context, now, candidates)
            return

        episode.arrived_at = None
        if now - episode.started_at >= _EPISODE_TIMEOUT_SEC:
            self._advance(context, now, candidates)

    def target_for(self, player_id: int) -> Pose2D:
        return self.targets[player_id]

    def reason_for(self, player_id: int) -> str:
        return self.reasons[player_id]

    def _advance(
        self,
        context: PlayContext,
        now: float,
        candidates: tuple[int, ...],
    ) -> None:
        old = self.episode
        if old is not None and old.blocker_player is not None:
            blocker = old.blocker_player
            self.targets[blocker] = self._parking_target(blocker)
            self.reasons[blocker] = f"eta_exp|mode=parking|player={blocker}"
        next_scenario = (self.scenario_index + 1) % len(ETA_SCENARIOS)
        if next_scenario == 0:
            # One robot completes the whole matrix before handing over. This
            # prevents the previous finisher becoming an uncontrolled obstacle
            # in every other episode.
            old_active = old.active_player if old is not None else None
            if old_active is not None:
                self.targets[old_active] = self._parking_target(old_active)
                self.reasons[old_active] = (
                    f"eta_exp|mode=parking_handover|player={old_active}"
                )
            self.active_index = (self.active_index + 1) % len(candidates)
        self.scenario_index = next_scenario
        self.episode = None
        self._start_episode(context, now, candidates)

    def _start_episode(
        self,
        context: PlayContext,
        now: float,
        candidates: tuple[int, ...],
    ) -> None:
        active = candidates[self.active_index % len(candidates)]
        robot = context.teammates.get(active)
        if robot is None or robot.pose is None:
            return
        scenario = ETA_SCENARIOS[self.scenario_index]
        if scenario.teammate_avoidance and len(candidates) < 2:
            return
        self.episode_number += 1
        target = self._relative_target(robot.pose, scenario)
        episode = _Episode(
            number=self.episode_number,
            scenario=scenario,
            active_player=active,
            target=target,
            reason="",
            started_at=now,
            start_pose=robot.pose,
            final_target=target,
        )

        if scenario.moving_retarget:
            warmup = self._target_from_heading(robot.pose, 1.70, robot.pose.theta)
            episode.target = warmup
            episode.stage = "moving_warmup"
            episode.reason = self._reason(episode, "moving_warmup")
        elif scenario.teammate_avoidance and len(candidates) >= 2:
            blocker = next(pid for pid in candidates if pid != active)
            final_target = self._relative_target(robot.pose, scenario)
            blocker_target = Pose2D(
                x=robot.pose.x + 0.48 * (final_target.x - robot.pose.x),
                y=robot.pose.y + 0.48 * (final_target.y - robot.pose.y),
                theta=math.atan2(
                    final_target.y - robot.pose.y,
                    final_target.x - robot.pose.x,
                ),
            )
            blocker_target = self.kit.field.clamp_inside_field(blocker_target, 0.45)
            hold = Pose2D(robot.pose.x, robot.pose.y, robot.pose.theta)
            episode.target = hold
            episode.final_target = final_target
            episode.blocker_player = blocker
            episode.stage = "avoidance_prepare"
            episode.reason = self._reason(episode, "avoidance_hold")
            self.targets[blocker] = blocker_target
            self.reasons[blocker] = (
                f"eta_exp|mode=blocker_prepare|episode={episode.number}"
                f"|player={blocker}|active={active}"
            )
        else:
            episode.reason = self._reason(episode, "rest_start")

        self.episode = episode
        self.targets[active] = episode.target
        self.reasons[active] = episode.reason

    def _relative_target(self, pose: Pose2D, scenario: EtaScenario) -> Pose2D:
        headings = [pose.theta + scenario.path_error_rad]
        if abs(scenario.path_error_rad) > 1e-6:
            headings.append(pose.theta - scenario.path_error_rad)
        headings.extend((headings[0] + math.pi, pose.theta))
        best: tuple[float, Pose2D] | None = None
        for heading in headings:
            candidate = self._target_from_heading(pose, scenario.distance_m, heading)
            achieved = math.hypot(candidate.x - pose.x, candidate.y - pose.y)
            if best is None or achieved > best[0]:
                best = (achieved, candidate)
            if achieved >= scenario.distance_m - 0.05:
                break
        assert best is not None
        path_heading = math.atan2(best[1].y - pose.y, best[1].x - pose.x)
        return Pose2D(
            best[1].x,
            best[1].y,
            normalize_angle(path_heading + scenario.final_heading_offset_rad),
        )

    def _target_from_heading(
        self,
        pose: Pose2D,
        distance_m: float,
        heading: float,
    ) -> Pose2D:
        return self.kit.field.clamp_inside_field(
            Pose2D(
                pose.x + distance_m * math.cos(heading),
                pose.y + distance_m * math.sin(heading),
                normalize_angle(heading),
            ),
            0.45,
        )

    def _parking_target(self, player_id: int) -> Pose2D:
        goalkeeper = self.kit.config.goalkeeper_player_id()
        if player_id == goalkeeper:
            return Pose2D(-5.75, 0.0, 0.0)
        field_players = [
            pid for pid in self.kit.config.player_ids if pid != goalkeeper
        ]
        index = field_players.index(player_id) if player_id in field_players else 0
        y = -3.15 if index % 2 == 0 else 3.15
        x = -4.20 + 0.55 * (index // 2)
        return self.kit.field.clamp_inside_field(Pose2D(x, y, 0.0), 0.45)

    def _experiment_players(self, context: PlayContext) -> tuple[int, ...]:
        goalkeeper = self.kit.config.goalkeeper_player_id()
        game = context.game_state
        now = self._observation_time(context)
        return tuple(
            player_id
            for player_id in self.kit.config.player_ids
            if player_id != goalkeeper
            and context.teammates.get(player_id) is not None
            and context.teammates[player_id].pose is not None
            and context.teammates[player_id].is_recent(now)
            and (game is None or game.is_active_player(self.kit.config.team_id, player_id))
        )

    @staticmethod
    def _observation_time(context: PlayContext) -> float:
        stamps = [robot.last_seen_at for robot in context.teammates.values()]
        if context.game_state is not None:
            stamps.append(context.game_state.last_seen_at)
        return max(stamps, default=0.0)

    @staticmethod
    def _arrived(pose: Pose2D, target: Pose2D) -> bool:
        return (
            math.hypot(target.x - pose.x, target.y - pose.y) < _ARRIVAL_DISTANCE_M
            and abs(normalize_angle(target.theta - pose.theta)) < _ARRIVAL_ANGLE_RAD
        )

    @staticmethod
    def _reason(episode: _Episode, mode: str) -> str:
        scenario = episode.scenario
        return (
            f"eta_exp|scenario={scenario.name}|episode={episode.number}"
            f"|active={episode.active_player}|mode={mode}"
            f"|distance={scenario.distance_m:.2f}"
            f"|path_error={scenario.path_error_rad:.3f}"
            f"|final_error={scenario.final_heading_offset_rad:.3f}"
        )
