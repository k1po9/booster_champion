"""Behaviour-tree roles for Primary, Secondary, and Safety task slots."""

from __future__ import annotations

import py_trees

from ..soccer_framework import PlayContext, Pose2D
from .nodes import AttackSubtreeConfig, MoveToTarget, build_attack_subtree
from .role import RoleStrategy
from .tactical import (
    DynamicTriangleCoordinator,
    PrimaryIntent,
    ROLE_PRIMARY,
    ROLE_SECONDARY,
    ROLE_SAFETY,
)


class PrimaryRole(RoleStrategy):
    """Solve the current-ball action selected by the team coordinator."""

    name = ROLE_PRIMARY

    def __init__(self, coordinator: DynamicTriangleCoordinator):
        self.coordinator = coordinator

    def _snapshot(self):
        snapshot = self.coordinator.last_context
        if snapshot is None:
            raise ValueError("tactical context is not ready")
        return snapshot

    def target(self, context: PlayContext) -> Pose2D:
        return self._snapshot().primary_target

    def kick_target(self, context: PlayContext) -> Pose2D:
        return self._snapshot().action_target

    def wants_to_kick(self, context: PlayContext) -> bool:
        snapshot = self._snapshot()
        if snapshot.primary_intent == PrimaryIntent.PASS:
            return snapshot.secondary_ready
        return snapshot.primary_intent in {
            PrimaryIntent.SHOOT,
            PrimaryIntent.DRIBBLE,
            PrimaryIntent.PROGRESSIVE_TOUCH,
            PrimaryIntent.CLEAR,
        }

    def build_subtree(self, kit, player_id: int) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=self.target,
                kick_target_fn=self.kick_target,
                wants_kick_fn=self.wants_to_kick,
                reason_fn=lambda: (
                    f"primary {self._snapshot().mode.value} "
                    f"{self._snapshot().primary_intent.value}"
                ),
                kick_reason_fn=lambda _target: (
                    f"primary {self._snapshot().primary_intent.value}"
                ),
            ),
        )


class SecondaryRole(RoleStrategy):
    """Occupy the predicted next-state position without duplicating Primary."""

    name = ROLE_SECONDARY

    def __init__(self, coordinator: DynamicTriangleCoordinator):
        self.coordinator = coordinator

    def target(self, _context: PlayContext) -> Pose2D:
        snapshot = self.coordinator.last_context
        if snapshot is None:
            raise ValueError("tactical context is not ready")
        return snapshot.secondary_target

    def build_subtree(self, kit, player_id: int) -> py_trees.behaviour.Behaviour:
        return MoveToTarget(
            kit,
            player_id,
            self.target,
            reason_fn=lambda: (
                f"secondary {self.coordinator.last_context.primary_intent.value}"
                if self.coordinator.last_context is not None
                else "secondary wait"
            ),
            hold_vyaw=0.12,
        )


class SafetyRole(RoleStrategy):
    """Maintain goal-side rest defence and cover the worst-case transition."""

    name = ROLE_SAFETY

    def __init__(self, coordinator: DynamicTriangleCoordinator):
        self.coordinator = coordinator

    def target(self, _context: PlayContext) -> Pose2D:
        snapshot = self.coordinator.last_context
        if snapshot is None:
            raise ValueError("tactical context is not ready")
        return snapshot.safety_target

    def build_subtree(self, kit, player_id: int) -> py_trees.behaviour.Behaviour:
        return MoveToTarget(
            kit,
            player_id,
            self.target,
            reason_fn=lambda: (
                f"safety cover risk={self.coordinator.last_context.counterattack_risk:.2f}"
                if self.coordinator.last_context is not None
                else "safety wait"
            ),
            hold_vyaw=0.12,
        )
