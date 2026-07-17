"""Behavior-tree adapter for the deterministic robot ETA experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import py_trees

from ..soccer_framework import PlayContext, Pose2D
from ..tactics.eta_experiment import (
    ETA_SCENARIOS,
    ROLE_ETA_EXPERIMENT,
    EtaExperimentCoordinator,
    EtaScenario,
)
from .default_roles import ChaserRole, GoalkeeperRole
from .nodes import MoveToTarget
from .playbook import (
    ROLE_GOALKEEPER,
    Playbook,
    RoleAssignment,
)
from .role import RoleStrategy

if TYPE_CHECKING:
    from ..runtime import SoccerKit

ROLE_ETA_BALL_GUARD = "eta_ball_guard"


class EtaExperimentRole(RoleStrategy):
    name = ROLE_ETA_EXPERIMENT

    def __init__(self, coordinator: EtaExperimentCoordinator) -> None:
        self.coordinator = coordinator

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return MoveToTarget(
            kit,
            player_id,
            lambda _context: self.coordinator.target_for(player_id),
            reason_fn=lambda: self.coordinator.reason_for(player_id),
            linear_speed_limit_fn=lambda: self.coordinator.speed_for(
                player_id
            ),
        )


class EtaBallGuardRole(ChaserRole):
    """Contest the ball but clear laterally instead of creating goal pauses."""

    name = ROLE_ETA_BALL_GUARD

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        del player_id
        ball = context.known_ball
        lateral_y = 3.2 if ball.y >= 0.0 else -3.2
        deep_x = kit.field.opponent_goal_x() - 2.0
        return kit.field.clamp_inside_field(
            Pose2D(deep_x, lateral_y, 0.0),
            0.45,
        )


class EtaExperimentPlaybook(Playbook):
    """Default data-recorder Playbook; all PLAY robots follow the coordinator."""

    def __init__(self, kit: "SoccerKit") -> None:
        super().__init__(kit)
        self.coordinator = EtaExperimentCoordinator(kit)
        self.register_role(EtaExperimentRole(self.coordinator))
        self.register_role(EtaBallGuardRole())
        self.register_role(GoalkeeperRole())

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        if self.coordinator.collection_paused(context):
            self.coordinator.pause()
        else:
            self.coordinator.update(context)
        return self._role_assignment(context)

    def _role_assignment(self, context: PlayContext) -> RoleAssignment:
        """Keep the match alive while only experiment participants run ETA."""

        mapping: dict[int, str] = {}
        goalkeeper = self.kit.config.goalkeeper_player_id()
        if goalkeeper is not None:
            mapping[goalkeeper] = ROLE_GOALKEEPER

        eta_players = set(self.coordinator.eta_players())
        for player_id in eta_players:
            mapping[player_id] = ROLE_ETA_EXPERIMENT

        available = tuple(
            player_id
            for player_id in self.coordinator.available_players(context)
            if player_id not in eta_players
        )
        chaser = self._nearest_to_ball(context, available)
        if chaser is not None:
            mapping[chaser] = ROLE_ETA_BALL_GUARD
        return RoleAssignment(mapping)

    @staticmethod
    def _nearest_to_ball(
        context: PlayContext,
        candidates: tuple[int, ...],
    ) -> int | None:
        if not candidates:
            return None
        ball = context.known_ball
        return min(
            candidates,
            key=lambda player_id: (
                (context.teammates[player_id].pose.x - ball.x) ** 2
                + (context.teammates[player_id].pose.y - ball.y) ** 2,
                player_id,
            ),
        )

__all__ = [
    "ETA_SCENARIOS",
    "ROLE_ETA_BALL_GUARD",
    "ROLE_ETA_EXPERIMENT",
    "EtaBallGuardRole",
    "EtaExperimentCoordinator",
    "EtaExperimentPlaybook",
    "EtaExperimentRole",
    "EtaScenario",
]
