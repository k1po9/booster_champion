"""Progress-aware watchdog for one chaser's approach-align-kick attempt.

The watchdog is a pure tactical state model.  It does not choose a dribble direction or issue robot commands; it only reports whether an active attempt has
completed, exceeded a hard/phase deadline, or stopped making measurable
progress.  The play layer decides the recovery action and keeps rule/safety
branches authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


__all__ = [
    "AttackAttemptObservation",
    "AttackPhase",
    "AttackWatchdog",
    "AttackWatchdogConfig",
    "AttackWatchdogStatus",
]


class AttackPhase(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    ALIGN = "align"
    EXECUTE = "execute"
    ESCAPE = "escape"


@dataclass(frozen=True)
class AttackWatchdogConfig:
    """Timing and progress thresholds for an attack attempt."""

    total_timeout_sec: float = 4.0
    no_progress_timeout_sec: float = 1.0
    approach_timeout_sec: float = 2.5
    align_timeout_sec: float = 1.5
    execute_timeout_sec: float = 2.0
    distance_progress_m: float = 0.05
    angle_progress_rad: float = 0.08
    ball_release_speed_mps: float = 0.35

    def phase_timeout(self, phase: AttackPhase) -> float | None:
        return {
            AttackPhase.APPROACH: self.approach_timeout_sec,
            AttackPhase.ALIGN: self.align_timeout_sec,
            AttackPhase.EXECUTE: self.execute_timeout_sec,
        }.get(phase)


@dataclass(frozen=True)
class AttackAttemptObservation:
    """One control-tick observation supplied by the chaser play."""

    t_sec: float
    phase: AttackPhase
    ball_distance_m: float
    angle_error_rad: float | None = None
    ball_speed_mps: float = 0.0
    active: bool = True


@dataclass(frozen=True)
class AttackWatchdogStatus:
    player_id: int
    active: bool
    phase: AttackPhase
    elapsed_sec: float
    phase_elapsed_sec: float
    no_progress_sec: float
    remaining_budget_sec: float
    phase_remaining_sec: float | None
    completed: bool
    total_timed_out: bool
    phase_timed_out: bool
    stalled: bool
    should_escape: bool
    reason: str

    def safe_action_budget(
        self,
        opponent_pressure_sec: float | None,
        safety_margin_sec: float = 0.20,
    ) -> float:
        """Return time still safely available before own or opponent deadline."""

        own_budget = max(0.0, self.remaining_budget_sec)
        if opponent_pressure_sec is None or not math.isfinite(opponent_pressure_sec):
            return own_budget
        return max(0.0, min(own_budget, opponent_pressure_sec - safety_margin_sec))


@dataclass
class _AttemptState:
    started_at: float
    phase_started_at: float
    last_progress_at: float
    phase: AttackPhase
    best_distance_m: float
    best_angle_error_rad: float | None


class AttackWatchdog:
    """Track independent progress/deadlines for each player."""

    def __init__(self, config: AttackWatchdogConfig | None = None):
        self.config = AttackWatchdogConfig() if config is None else config
        self._states: dict[int, _AttemptState] = {}

    def reset(self, player_id: int | None = None) -> None:
        if player_id is None:
            self._states.clear()
        else:
            self._states.pop(player_id, None)

    def update(
        self,
        player_id: int,
        observation: AttackAttemptObservation,
    ) -> AttackWatchdogStatus:
        """Update one attempt and return a decision-ready status.

        Callers choose when an attempt becomes active.  In particular, the
        current 2.5 m kick-hysteresis entry distance is too broad to mean ball
        control and should not by itself start this watchdog.
        """

        if not observation.active or observation.phase is AttackPhase.IDLE:
            self.reset(player_id)
            return self._idle_status(player_id, "inactive")

        now = observation.t_sec
        state = self._states.get(player_id)
        if state is None or now < state.started_at:
            state = self._new_state(observation)
            self._states[player_id] = state

        phase_changed = observation.phase is not state.phase
        if phase_changed:
            state.phase = observation.phase
            state.phase_started_at = now
            state.last_progress_at = now
            state.best_distance_m = max(0.0, observation.ball_distance_m)
            state.best_angle_error_rad = self._absolute_angle(observation.angle_error_rad)
        else:
            self._update_progress(state, observation)

        elapsed = max(0.0, now - state.started_at)
        phase_elapsed = max(0.0, now - state.phase_started_at)
        no_progress = max(0.0, now - state.last_progress_at)
        total_timed_out = elapsed >= max(0.0, self.config.total_timeout_sec)
        phase_limit = self.config.phase_timeout(state.phase)
        phase_timed_out = phase_limit is not None and phase_elapsed >= max(0.0, phase_limit)
        stalled = no_progress >= max(0.0, self.config.no_progress_timeout_sec)
        completed = (
            state.phase is AttackPhase.EXECUTE
            and observation.ball_speed_mps >= self.config.ball_release_speed_mps
        )

        if completed:
            reason = "ball released"
        elif total_timed_out:
            reason = "attack total timeout"
        elif phase_timed_out:
            reason = f"{state.phase.value} phase timeout"
        elif stalled:
            reason = f"{state.phase.value} no progress"
        elif state.phase is AttackPhase.ESCAPE:
            reason = "escape active"
        elif phase_changed:
            reason = f"entered {state.phase.value}"
        else:
            reason = f"{state.phase.value} progressing"

        status = AttackWatchdogStatus(
            player_id=player_id,
            active=True,
            phase=state.phase,
            elapsed_sec=elapsed,
            phase_elapsed_sec=phase_elapsed,
            no_progress_sec=no_progress,
            remaining_budget_sec=max(0.0, self.config.total_timeout_sec - elapsed),
            phase_remaining_sec=(
                None if phase_limit is None else max(0.0, phase_limit - phase_elapsed)
            ),
            completed=completed,
            total_timed_out=total_timed_out,
            phase_timed_out=phase_timed_out,
            stalled=stalled,
            should_escape=(
                not completed
                and (
                    total_timed_out
                    or phase_timed_out
                    or stalled
                    or state.phase is AttackPhase.ESCAPE
                )
            ),
            reason=reason,
        )
        if completed:
            self.reset(player_id)
        return status

    def status(self, player_id: int, now_sec: float) -> AttackWatchdogStatus:
        state = self._states.get(player_id)
        if state is None:
            return self._idle_status(player_id, "not started")
        phase_limit = self.config.phase_timeout(state.phase)
        elapsed = max(0.0, now_sec - state.started_at)
        phase_elapsed = max(0.0, now_sec - state.phase_started_at)
        no_progress = max(0.0, now_sec - state.last_progress_at)
        total_timed_out = elapsed >= self.config.total_timeout_sec
        phase_timed_out = phase_limit is not None and phase_elapsed >= phase_limit
        stalled = no_progress >= self.config.no_progress_timeout_sec
        return AttackWatchdogStatus(
            player_id=player_id,
            active=True,
            phase=state.phase,
            elapsed_sec=elapsed,
            phase_elapsed_sec=phase_elapsed,
            no_progress_sec=no_progress,
            remaining_budget_sec=max(0.0, self.config.total_timeout_sec - elapsed),
            phase_remaining_sec=None if phase_limit is None else max(0.0, phase_limit - phase_elapsed),
            completed=False,
            total_timed_out=total_timed_out,
            phase_timed_out=phase_timed_out,
            stalled=stalled,
            should_escape=(
                total_timed_out
                or phase_timed_out
                or stalled
                or state.phase is AttackPhase.ESCAPE
            ),
            reason="status snapshot",
        )

    def _new_state(self, observation: AttackAttemptObservation) -> _AttemptState:
        return _AttemptState(
            started_at=observation.t_sec,
            phase_started_at=observation.t_sec,
            last_progress_at=observation.t_sec,
            phase=observation.phase,
            best_distance_m=max(0.0, observation.ball_distance_m),
            best_angle_error_rad=self._absolute_angle(observation.angle_error_rad),
        )

    def _update_progress(
        self,
        state: _AttemptState,
        observation: AttackAttemptObservation,
    ) -> None:
        distance = max(0.0, observation.ball_distance_m)
        angle = self._absolute_angle(observation.angle_error_rad)
        progressed = False
        if distance <= state.best_distance_m - self.config.distance_progress_m:
            state.best_distance_m = distance
            progressed = True
        if angle is not None and (
            state.best_angle_error_rad is None
            or angle <= state.best_angle_error_rad - self.config.angle_progress_rad
        ):
            state.best_angle_error_rad = angle
            progressed = True
        if progressed:
            state.last_progress_at = observation.t_sec

    @staticmethod
    def _absolute_angle(value: float | None) -> float | None:
        return None if value is None else abs(value)

    @staticmethod
    def _idle_status(player_id: int, reason: str) -> AttackWatchdogStatus:
        return AttackWatchdogStatus(
            player_id=player_id,
            active=False,
            phase=AttackPhase.IDLE,
            elapsed_sec=0.0,
            phase_elapsed_sec=0.0,
            no_progress_sec=0.0,
            remaining_budget_sec=0.0,
            phase_remaining_sec=None,
            completed=False,
            total_timed_out=False,
            phase_timed_out=False,
            stalled=False,
            should_escape=False,
            reason=reason,
        )
