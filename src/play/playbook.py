"""PLAY-stage strategy entry point; competitors usually edit this file first.

:class:`RoleAssignment` snapshots "who does what" once per tick and stores a
``player_id -> role_name`` mapping that can hold custom roles. :class:`Playbook`
centralizes competitor-overridable PLAY decisions and explicitly registers roles
through :meth:`register_role`.

:class:`DefaultPlaybook` is the template default. It registers the chaser,
supporter, and goalkeeper roles; fixed starting slots use ``ReadySlot`` for
non-PLAY branches. To change tactics, override ``assign_roles``, customize
``select_chaser`` or ``kick_target``, or register a new role after
``super().__init__(kit)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..soccer_framework import PlayContext, ReadySlot, RobotCommand
from ..runtime import SoccerKit

if TYPE_CHECKING:
    from .role import RoleRegistry, RoleStrategy


# Default role labels kept as constants for ``assign_roles``; they no longer restrict the string set.
ROLE_CHASER = "chaser"
ROLE_SUPPORTER = "supporter"
ROLE_GOALKEEPER = "goalkeeper"
ROLE_NONE = "none"


@dataclass(frozen=True)
class RoleAssignment:
    """Snapshot of this tick's dynamic role assignment.

    The storage is ``by_player: Mapping[int, str]`` where keys are player IDs
    and values are role labels. ``role_of`` returns :data:`ROLE_NONE` when absent.

    Construction is direct: ``RoleAssignment({1: "chaser", 2: "supporter"})``.

    Use :meth:`players_of` for reverse lookup by role; specialized attributes
    such as chaser/supporters are intentionally not provided.
    """

    by_player: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze as a read-only view so external by_player edits cannot change ``role_of`` behavior.
        object.__setattr__(self, "by_player", MappingProxyType(dict(self.by_player)))

    def role_of(self, player_id: int) -> str:
        return self.by_player.get(player_id, ROLE_NONE)

    def players_of(self, name: str) -> tuple[int, ...]:
        return tuple(pid for pid, role in self.by_player.items() if role == name)


class Playbook:
    """Entry point for all PLAY-stage decisions competitors may override.

    Leaf nodes hold both :class:`SoccerKit` for tools and :class:`Playbook`
    for decisions. In the PLAY subtree, ``AssignRoles`` writes
    :meth:`assign_roles` output, per-player branches choose by
    :class:`RoleAssignment` plus ``role_registry``, kick leaves call
    :meth:`kick_target`, hold-style roles own their targets, and
    ``WaitForBall`` calls :meth:`waiting_command`.

    Subclasses add or remove roles in ``__init__`` with :meth:`register_role` and
    override :meth:`assign_roles` to choose this tick's assignment.
    """

    def __init__(self, kit: SoccerKit):
        # Delayed import breaks the role -> kit.blackboard -> kit -> play -> playbook cycle.
        from .role import RoleRegistry

        self.kit = kit
        self._registry = RoleRegistry()

    # Role registry

    def register_role(self, role: RoleStrategy) -> "Playbook":
        """Register a role on this Playbook and return self for chaining.

        Subclasses call this in ``__init__`` after ``super().__init__(kit)``.
        Registration order determines PLAY Selector branch priority.
        """

        self._registry.register(role)
        return self

    @property
    def role_registry(self) -> RoleRegistry:
        return self._registry

    # Role assignment, the core strategy node

    def assign_roles(
        self,
        context: PlayContext,
    ) -> RoleAssignment:
        """Return one :class:`RoleAssignment` per tick; subclasses decide the actual assignment."""

        raise NotImplementedError

    # Cross-role cooperative targets

    def waiting_command(
        self,
        player_id: int,
        context: PlayContext,
    ) -> RobotCommand:
        """Fallback command when no role is assigned.

        The default is a stop tagged by ReadySlot. Competitors can override this
        for custom idle positioning; the node has already cleared :class:`KickHysteresis`.
        """

        slot = self.kit.config.ready_slot_for_player(player_id)
        return RobotCommand.stop(f"{slot.value} waiting for ball")


# ----------------------------------------------------------------------
# Default implementation: fixed ReadySlot starts plus PLAY dynamic roles
# ----------------------------------------------------------------------


class DefaultPlaybook(Playbook):
    """Default SoccerSim playbook: chaser/supporter/goalkeeper dynamic roles plus Targeting scores.

    Subclasses can selectively override one method, for example:

    .. code-block:: python

    class AggressivePlaybook(DefaultPlaybook):
    def assign_roles(self, context):
    base = super().assign_roles(context)
    Move more players to supporters when trailing.
    """

    def __init__(self, kit: SoccerKit):
        super().__init__(kit)
        # Explicitly register default PLAY dynamic roles; competitor subclasses can
        # call register_role(...) after super().__init__(kit). DefenderRole is reserved for explicit custom Playbook registration.
        from .default_roles import (
            ChaserRole,
            GoalkeeperRole,
            SupporterRole,
        )

        self.register_role(ChaserRole())
        self.register_role(SupporterRole())
        self.register_role(GoalkeeperRole())

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        chaser_id = self.select_chaser(context)
        goalkeeper_id = self._configured_goalkeeper()

        mapping: dict[int, str] = {}
        for player_id in self.kit.config.player_ids:
            if player_id == goalkeeper_id:
                mapping[player_id] = ROLE_GOALKEEPER
            elif player_id == chaser_id:
                mapping[player_id] = ROLE_CHASER
            else:
                mapping[player_id] = ROLE_SUPPORTER

        return RoleAssignment(mapping)

    # Internals

    def _configured_goalkeeper(self) -> int | None:
        return self.kit.config.goalkeeper_player_id()

    def select_chaser(self, context: PlayContext) -> int:
        """Select this tick's chaser from our team.

        Decision priority:
        1. ReadySlot eligibility: keeper only joins dangerous balls, side only challenges when suitable.
        2. ``ball_claim_score``: cost based on distance to ball plus ReadySlot preference.
        3. Lowest player ID wins ties for predictable debugging.

        If no role is suitable, fall back to the smallest configured player ID so
        "nobody chases" does not become an extra ``None`` state. Override this method
        or call another score from :meth:`assign_roles` to change chase strategy.
        """
        config = self.kit.config
        targeting = self.kit.targeting
        ball = context.known_ball

        candidates: list[int] = []
        scored: list[tuple[float, int]] = []
        for player_id in config.player_ids:
            slot = config.ready_slot_for_player(player_id)
            if not self._slot_can_challenge(slot, context):
                continue
            candidates.append(player_id)
            robot = context.teammates.get(player_id)
            if robot is None or robot.pose is None:
                continue
            scored.append(
                (targeting.ball_claim_score(slot, robot.pose, ball), player_id)
            )

        if not candidates:
            return min(config.player_ids)
        if not scored:
            return min(candidates)

        tie_margin = config.strategy.teammate_challenge_tie_margin_m
        # Sort by score ascending; lower is better, and scores within tie_margin are tied.
        # Ties choose the smallest player ID for predictable debugging.
        ranked = sorted(scored, key=lambda item: item[0])
        best_score = ranked[0][0]
        tied_ids = [
            player_id for score, player_id in ranked if score <= best_score + tie_margin
        ]
        return min(tied_ids)

    def _slot_can_challenge(
        self,
        slot: ReadySlot,
        context: PlayContext,
    ) -> bool:
        targeting = self.kit.targeting
        ball = context.known_ball
        if slot == ReadySlot.KEEPER:
            return targeting.ball_in_own_defensive_area(ball)
        if slot == ReadySlot.SIDE:
            return targeting.side_should_challenge(context)
        return True
