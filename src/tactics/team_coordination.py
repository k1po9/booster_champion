"""Pure team-coordination primitives for exclusive ball ownership and marking.

The state machine does not assign behavior-tree roles or issue commands.  It
turns bounded, public-state evidence into an auditable goalkeeper takeover
status.  ChampionPlaybook owns the final atomic role assignment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import math

from ..soccer_framework import Pose2D
from .geometry import normalize_angle


class TeamPhase(str, Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    KEEPER_EMERGENCY = "keeper_emergency"
    KEEPER_RECOVER = "keeper_recover"


class BallOwnerRole(str, Enum):
    NONE = "none"
    HANDLER = "handler"
    KEEPER = "keeper"


@dataclass(frozen=True)
class KeeperTakeoverEvidence:
    """One tick of already-filtered public evidence for keeper arbitration."""

    eligible: bool
    ball_in_defensive_area: bool
    emergency_threat: bool
    keeper_advantage: bool
    release_confirmed: bool = False
    reason: str = ""


@dataclass(frozen=True)
class KeeperTakeoverStatus:
    active: bool
    recovering: bool
    phase_started_at_sec: float
    reason: str


class KeeperTakeoverCoordinator:
    """Bounded NORMAL -> EMERGENCY -> RECOVER transaction with hysteresis."""

    def __init__(
        self,
        *,
        minimum_hold_sec: float = 0.60,
        maximum_hold_sec: float = 3.00,
        recover_hold_sec: float = 0.80,
    ):
        self.minimum_hold_sec = max(0.0, minimum_hold_sec)
        self.maximum_hold_sec = max(self.minimum_hold_sec, maximum_hold_sec)
        self.recover_hold_sec = max(0.0, recover_hold_sec)
        self._state = "normal"
        self._started_at_sec = -math.inf
        self._reason = "normal field ownership"

    def reset(self) -> None:
        self._state = "normal"
        self._started_at_sec = -math.inf
        self._reason = "normal field ownership"

    def update(
        self,
        now_sec: float,
        evidence: KeeperTakeoverEvidence,
    ) -> KeeperTakeoverStatus:
        if not math.isfinite(self._started_at_sec):
            self._started_at_sec = now_sec
        takeover_required = (
            evidence.eligible
            and evidence.ball_in_defensive_area
            and evidence.emergency_threat
            and evidence.keeper_advantage
        )

        if self._state == "normal":
            if takeover_required:
                self._state = "emergency"
                self._started_at_sec = now_sec
                self._reason = evidence.reason or "keeper emergency accepted"
        elif self._state == "emergency":
            elapsed = max(0.0, now_sec - self._started_at_sec)
            may_release = elapsed >= self.minimum_hold_sec
            must_release = elapsed >= self.maximum_hold_sec or not evidence.eligible
            normal_release = (
                not evidence.ball_in_defensive_area
                or evidence.release_confirmed
                or not evidence.emergency_threat
            )
            if must_release or (may_release and normal_release):
                self._state = "recover"
                self._started_at_sec = now_sec
                self._reason = (
                    "keeper takeover maximum duration reached"
                    if elapsed >= self.maximum_hold_sec
                    else "keeper threat released"
                )
        else:
            # Recovery is a real, bounded transaction rather than a one-tick
            # label.  This prevents a persistent threat from immediately
            # re-entering emergency after the maximum-duration fail-safe.
            if now_sec - self._started_at_sec >= self.recover_hold_sec:
                if takeover_required:
                    self._state = "emergency"
                    self._reason = evidence.reason or "keeper emergency renewed"
                else:
                    self._state = "normal"
                    self._reason = "keeper recovery complete"
                self._started_at_sec = now_sec

        return KeeperTakeoverStatus(
            active=self._state == "emergency",
            recovering=self._state == "recover",
            phase_started_at_sec=self._started_at_sec,
            reason=self._reason,
        )


@dataclass(frozen=True)
class OpponentThreat:
    player_id: int
    pose: Pose2D
    score: float


def select_dangerous_opponent(
    opponents: Iterable[tuple[int, Pose2D]],
    *,
    ball: Pose2D,
    own_goal: Pose2D,
    defenders: Iterable[Pose2D] = (),
    excluded_player_id: int | None = None,
    previous_player_id: int | None = None,
    switch_margin: float = 0.30,
) -> OpponentThreat | None:
    """Select the most dangerous off-ball opponent with ID hysteresis."""

    defender_rows = tuple(defenders)
    scored: list[OpponentThreat] = []
    goal_span = max(1.0, math.hypot(ball.x - own_goal.x, ball.y - own_goal.y))
    for player_id, pose in opponents:
        if player_id == excluded_player_id:
            continue
        goal_distance = math.hypot(pose.x - own_goal.x, pose.y - own_goal.y)
        ball_distance = math.hypot(pose.x - ball.x, pose.y - ball.y)
        centrality = 1.0 / (1.0 + abs(pose.y - own_goal.y))
        pass_relevance = 1.0 / (0.75 + ball_distance)
        goal_proximity = goal_span / (1.0 + goal_distance)
        nearest_defender = min(
            (math.hypot(pose.x - item.x, pose.y - item.y) for item in defender_rows),
            default=2.0,
        )
        unmarked = min(2.0, nearest_defender) / 2.0
        score = (
            1.45 * goal_proximity
            + 0.70 * centrality
            + 0.55 * pass_relevance
            + 0.45 * unmarked
        )
        scored.append(OpponentThreat(player_id, pose, score))

    if not scored:
        return None
    best = max(scored, key=lambda item: (item.score, -item.player_id))
    previous = next(
        (item for item in scored if item.player_id == previous_player_id),
        None,
    )
    if previous is not None and best.score < previous.score + max(0.0, switch_margin):
        return previous
    return best


def marker_target(
    opponent: Pose2D,
    *,
    ball: Pose2D,
    own_goal: Pose2D,
    marking_distance_m: float = 0.75,
) -> Pose2D:
    """Stand goal-side of an opponent and face the ball."""

    dx = own_goal.x - opponent.x
    dy = own_goal.y - opponent.y
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        x, y = opponent.x, opponent.y
    else:
        # Never overshoot the goal when the opponent is already very deep.
        distance = max(0.05, min(max(0.25, marking_distance_m), length * 0.55))
        x = opponent.x + dx / length * distance
        y = opponent.y + dy / length * distance
    return Pose2D(x, y, normalize_angle(math.atan2(ball.y - y, ball.x - x)))


__all__ = [
    "BallOwnerRole",
    "KeeperTakeoverCoordinator",
    "KeeperTakeoverEvidence",
    "KeeperTakeoverStatus",
    "OpponentThreat",
    "TeamPhase",
    "marker_target",
    "select_dangerous_opponent",
]
