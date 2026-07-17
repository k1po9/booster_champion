"""Behavior-tree adapter for the deterministic robot ETA experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import py_trees

from ..soccer_framework import PlayContext
from ..tactics.eta_experiment import (
    ETA_SCENARIOS,
    ROLE_ETA_EXPERIMENT,
    EtaExperimentCoordinator,
    EtaScenario,
)
from .nodes import MoveToTarget
from .playbook import Playbook, RoleAssignment
from .role import RoleStrategy

if TYPE_CHECKING:
    from ..runtime import SoccerKit


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
        )


class EtaExperimentPlaybook(Playbook):
    """Default data-recorder Playbook; all PLAY robots follow the coordinator."""

    def __init__(self, kit: "SoccerKit") -> None:
        super().__init__(kit)
        self.coordinator = EtaExperimentCoordinator(kit)
        self.register_role(EtaExperimentRole(self.coordinator))

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        if self.coordinator.collection_paused(context):
            self.coordinator.pause()
            return RoleAssignment()
        self.coordinator.update(context)
        return RoleAssignment(
            {player_id: ROLE_ETA_EXPERIMENT for player_id in self.kit.config.player_ids}
        )


__all__ = [
    "ETA_SCENARIOS",
    "ROLE_ETA_EXPERIMENT",
    "EtaExperimentCoordinator",
    "EtaExperimentPlaybook",
    "EtaExperimentRole",
    "EtaScenario",
]
