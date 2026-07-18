"""Fast, conservative competition playbook built from the validated tactics.

The playbook changes role assignment immediately, while ball prediction,
opponent pressure and the attack watchdog remain advisory diagnostics until
their match behaviour has been validated.  Every calculation is bounded for
the 30 Hz control loop and uses only the Python standard library.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
import time

from ..soccer_framework import PlayContext, Pose2D, ReadySlot, SetPlay
from ..tactics import (
    ActionSelection,
    BallOwnerRole,
    AttackAttemptObservation,
    AttackPhase,
    AttackWatchdog,
    AttackWatchdogStatus,
    BallMotionPrediction,
    BallTrajectoryPrediction,
    BoundedActionSelector,
    DynamicRobotArrivalEstimator,
    EventBallTrajectoryPredictor,
    InterceptEstimate,
    KickoffPhase,
    KickoffStatus,
    KickoffTransaction,
    KeeperTakeoverCoordinator,
    KeeperTakeoverEvidence,
    KeeperTakeoverStatus,
    MatchRisk,
    OpponentShape,
    OpponentShapeTracker,
    OpponentPressureEstimator,
    OpponentPressureReport,
    RobotArrivalEstimate,
    RobotArrivalQuery,
    RobotMotionTracker,
    SlidingWindowBallPredictor,
    TeamPhase,
    calculate_match_risk,
    estimate_earliest_intercept,
    marker_target,
    own_restart_target,
    select_dangerous_opponent,
)
from .playbook import (
    DefaultPlaybook,
    ROLE_CHASER,
    ROLE_GOALKEEPER,
    ROLE_SUPPORTER,
    RoleAssignment,
)
from .default_roles import ChaserRole, GoalkeeperRole, SupporterRole
from .nodes import AttackSubtreeConfig, MoveToTarget, build_attack_subtree
from .role import RoleStrategy


ROLE_MARKER = "marker"
ROLE_SECOND_BALL = "second_ball"


@dataclass(frozen=True)
class ChampionTuning:
    """Small set of empirically grounded, cheap tactical thresholds."""

    translation_speed_mps: float = 0.716
    yaw_speed_radps: float = 0.879
    handler_switch_margin_sec: float = 0.35
    handler_switch_cooldown_sec: float = 0.60
    lead_horizon_sec: float = 0.25
    lead_min_confidence: float = 0.45
    lead_max_weight: float = 0.55
    lead_max_uncertainty_m: float = 1.10
    pressure_cache_sec: float = 0.10
    watchdog_activation_distance_m: float = 1.00
    watchdog_align_distance_m: float = 0.45
    watchdog_align_angle_rad: float = 0.35
    enable_prediction: bool = True
    enable_robot_eta: bool = True
    enable_intercept: bool = True
    intercept_safety_margin_sec: float = 0.35
    enable_pressure: bool = True
    enable_watchdog: bool = True
    enable_baseline_shadow: bool = True
    decision_budget_ms: float = 8.0
    performance_window_size: int = 180
    enable_action_selection: bool = True
    enable_action_execution: bool = True
    enable_outlet_positioning: bool = True
    enable_team_coordination: bool = True
    enable_marker: bool = True
    keeper_takeover_min_hold_sec: float = 0.60
    keeper_takeover_max_sec: float = 3.00
    keeper_recover_hold_sec: float = 0.80
    keeper_takeover_max_distance_m: float = 2.20
    keeper_takeover_advantage_sec: float = 0.35
    marker_distance_m: float = 0.75
    marker_switch_margin: float = 0.30
    action_switch_margin: float = 0.18
    action_min_hold_sec: float = 0.35
    enable_watchdog_escape: bool = True
    enable_set_pieces: bool = True
    enable_match_management: bool = True
    enable_opponent_adaptation: bool = True
    kickoff_first_touch_power: float = 0.35


@dataclass(frozen=True)
class ChampionSnapshot:
    """Read-only shadow diagnostics from the latest role-assignment tick."""

    handler_id: int | None
    cover_id: int | None
    ball_owner_id: int | None
    ball_owner_role: BallOwnerRole
    team_phase: TeamPhase
    coordination_since_sec: float
    coordination_reason: str
    field_candidate_id: int | None
    marker_id: int | None
    marked_opponent_id: int | None
    chase_target: Pose2D
    ball_prediction: BallMotionPrediction | None
    ball_trajectory: BallTrajectoryPrediction | None
    handler_eta: RobotArrivalEstimate | None
    intercept: InterceptEstimate | None
    pressure: OpponentPressureReport | None
    pressure_level: str
    watchdog: AttackWatchdogStatus | None
    baseline_assignment: RoleAssignment | None
    decision_duration_ms: float
    budget_exceeded: bool
    action_selection: ActionSelection | None
    kickoff: KickoffStatus | None
    match_risk: MatchRisk | None
    opponent_shape: OpponentShape | None


@dataclass(frozen=True)
class ChampionPerformance:
    sample_count: int
    latest_ms: float
    average_ms: float
    p99_ms: float
    maximum_ms: float
    budget_exceeded_count: int


class ChampionPlaybook(DefaultPlaybook):
    """Stable Handler/Outlet/Cover assignment with bounded advisory models."""

    def __init__(
        self,
        kit,
        *,
        tuning: ChampionTuning | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(kit)
        self.tuning = ChampionTuning() if tuning is None else tuning
        self._clock = clock
        self._ball_predictor = SlidingWindowBallPredictor()
        self._ball_trajectory_predictor = EventBallTrajectoryPredictor()
        self._arrival_estimator = DynamicRobotArrivalEstimator()
        self._robot_motion = RobotMotionTracker()
        self._pressure_estimator = OpponentPressureEstimator()
        self._attack_watchdog = AttackWatchdog()
        self._action_selector = BoundedActionSelector(
            kit.config, kit.field, kit.obstacles
        )
        self._kickoff = KickoffTransaction(kit.config, kit.field)
        self._opponent_shape = OpponentShapeTracker()
        self._last_ball_stamp: float | None = None
        self._handler_id: int | None = None
        self._handler_selected_at = -math.inf
        self._pressure_updated_at = -math.inf
        self._pressure: OpponentPressureReport | None = None
        self._last_snapshot: ChampionSnapshot | None = None
        self._last_assignment = RoleAssignment()
        self._decision_times_ms: deque[float] = deque(
            maxlen=max(1, self.tuning.performance_window_size)
        )
        self._budget_exceeded_count = 0
        self._action_selected_at = -math.inf
        self._eta_target_state: dict[int, tuple[Pose2D, float]] = {}
        self._keeper_takeover = KeeperTakeoverCoordinator(
            minimum_hold_sec=self.tuning.keeper_takeover_min_hold_sec,
            maximum_hold_sec=self.tuning.keeper_takeover_max_sec,
            recover_hold_sec=self.tuning.keeper_recover_hold_sec,
        )
        self._team_phase = TeamPhase.ATTACK
        self._team_phase_started_at = -math.inf
        self._marked_opponent_id: int | None = None
        self._marker_target: Pose2D | None = None
        self._second_ball_target: Pose2D | None = None
        self.role_registry.replace(ChampionChaserRole(self))
        self.role_registry.replace(ChampionSupporterRole(self))
        self.role_registry.replace(ChampionGoalkeeperRole(self))
        self.role_registry.register(ChampionMarkerRole(self))
        self.role_registry.register(ChampionSecondBallRole(self))

    @property
    def last_snapshot(self) -> ChampionSnapshot | None:
        return self._last_snapshot

    @property
    def last_assignment(self) -> RoleAssignment:
        return self._last_assignment

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        started_at = time.perf_counter()
        now = self._clock()
        active = self._active_players(context, now)
        prediction, trajectory = self._update_predictions(context)
        cover_id = self._select_cover(active, context)
        handler_candidates = tuple(pid for pid in active if pid != cover_id)
        chase_target, intercept = self._chase_target(
            context, prediction, trajectory, handler_candidates, now
        )
        field_candidate_id = self._select_handler(
            handler_candidates, chase_target, context, now
        )
        field_candidate_eta = (
            self._estimate_arrival(
                field_candidate_id, chase_target, context, now
            )
            if field_candidate_id is not None and self.tuning.enable_robot_eta
            else None
        )
        keeper_eta = (
            self._estimate_arrival(
                cover_id,
                Pose2D(context.known_ball.x, context.known_ball.y, 0.0),
                context,
                now,
                track_target=False,
            )
            if cover_id is not None and self.tuning.enable_robot_eta
            else None
        )
        takeover = self._update_keeper_takeover(
            context=context,
            cover_id=cover_id,
            field_candidate_id=field_candidate_id,
            keeper_eta=keeper_eta,
            handler_eta=field_candidate_eta,
            prediction=prediction,
            now=now,
        )
        defending = self._is_defending(context, active, now)
        if takeover.active:
            team_phase = TeamPhase.KEEPER_EMERGENCY
        elif takeover.recovering:
            team_phase = TeamPhase.KEEPER_RECOVER
        elif defending:
            team_phase = TeamPhase.DEFEND
        else:
            team_phase = TeamPhase.ATTACK
        if team_phase is not self._team_phase:
            self._team_phase = team_phase
            self._team_phase_started_at = now
        elif not math.isfinite(self._team_phase_started_at):
            self._team_phase_started_at = now

        mapping: dict[int, str] = {}
        if cover_id is not None:
            mapping[cover_id] = ROLE_GOALKEEPER

        handler_id = field_candidate_id
        ball_owner_id = handler_id
        ball_owner_role = (
            BallOwnerRole.HANDLER if handler_id is not None else BallOwnerRole.NONE
        )
        marker_id: int | None = None
        marked_opponent_id: int | None = None
        self._marker_target = None
        self._second_ball_target = None

        if team_phase is TeamPhase.KEEPER_EMERGENCY and cover_id is not None:
            handler_id = None
            ball_owner_id = cover_id
            ball_owner_role = BallOwnerRole.KEEPER
            if field_candidate_id is not None:
                mapping[field_candidate_id] = ROLE_SECOND_BALL
                self._second_ball_target = self._compute_second_ball_target(
                    field_candidate_id, context
                )
        elif handler_id is not None:
            mapping[handler_id] = ROLE_CHASER

        remaining = tuple(pid for pid in active if pid not in mapping)
        if (
            self.tuning.enable_team_coordination
            and self.tuning.enable_marker
            and team_phase in {
                TeamPhase.DEFEND,
                TeamPhase.KEEPER_EMERGENCY,
                TeamPhase.KEEPER_RECOVER,
            }
            and remaining
        ):
            marker_id = remaining[0]
            marked_opponent_id, self._marker_target = self._select_marker_target(
                marker_id, context, active, now
            )
            if self._marker_target is not None:
                mapping[marker_id] = ROLE_MARKER

        for player_id in active:
            if player_id not in mapping:
                mapping[player_id] = ROLE_SUPPORTER

        outlet_id = next(
            (pid for pid in active if pid not in {cover_id, handler_id}), None
        )
        kickoff = (
            self._kickoff.update(
                now_sec=now,
                context=context,
                first_player_id=handler_id,
                second_player_id=outlet_id,
            )
            if self.tuning.enable_set_pieces
            else None
        )
        if (
            team_phase is not TeamPhase.KEEPER_EMERGENCY
            and kickoff is not None
            and kickoff.phase in {
                KickoffPhase.SECOND_PLAYER_ACQUIRE,
                KickoffPhase.SECOND_KICK_ACTIVE,
            }
            and kickoff.second_player_id in active
        ):
            second_id = kickoff.second_player_id
            assert second_id is not None
            if handler_id is not None and handler_id != second_id:
                mapping[handler_id] = ROLE_SUPPORTER
            mapping[second_id] = ROLE_CHASER
            handler_id = second_id
            ball_owner_id = second_id
            ball_owner_role = BallOwnerRole.HANDLER
            if marker_id == second_id:
                marker_id = None
                marked_opponent_id = None
                self._marker_target = None

        handler_eta = (
            self._estimate_arrival(handler_id, chase_target, context, now)
            if handler_id is not None and self.tuning.enable_robot_eta
            else None
        )
        pressure = self._update_pressure(context, chase_target, prediction, now)
        risk = (
            calculate_match_risk(
                self.kit.config, context, self._pressure_level(pressure)
            )
            if self.tuning.enable_match_management
            else None
        )
        opponent_shape = (
            self._opponent_shape.update(now, context)
            if self.tuning.enable_opponent_adaptation
            else None
        )
        watchdog = self._update_watchdog(context, handler_id, prediction, now)
        action_selection = self._select_action(
            context, handler_id, self._pressure_level(pressure), watchdog, now,
            0.5 if risk is None else risk.value,
        )
        baseline = (
            super().assign_roles(context)
            if self.tuning.enable_baseline_shadow
            else None
        )
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        self._decision_times_ms.append(duration_ms)
        budget_exceeded = duration_ms > self.tuning.decision_budget_ms
        if budget_exceeded:
            self._budget_exceeded_count += 1
        self._last_snapshot = ChampionSnapshot(
            handler_id=handler_id,
            cover_id=cover_id,
            ball_owner_id=ball_owner_id,
            ball_owner_role=ball_owner_role,
            team_phase=team_phase,
            coordination_since_sec=self._team_phase_started_at,
            coordination_reason=(
                takeover.reason
                if team_phase in {
                    TeamPhase.KEEPER_EMERGENCY,
                    TeamPhase.KEEPER_RECOVER,
                }
                else (
                    "defensive press and mark structure"
                    if team_phase is TeamPhase.DEFEND
                    else "normal field ownership"
                )
            ),
            field_candidate_id=field_candidate_id,
            marker_id=marker_id,
            marked_opponent_id=marked_opponent_id,
            chase_target=chase_target,
            ball_prediction=prediction,
            ball_trajectory=trajectory,
            handler_eta=handler_eta,
            intercept=intercept,
            pressure=pressure,
            pressure_level=self._pressure_level(pressure),
            watchdog=watchdog,
            baseline_assignment=baseline,
            decision_duration_ms=duration_ms,
            budget_exceeded=budget_exceeded,
            action_selection=action_selection,
            kickoff=kickoff,
            match_risk=risk,
            opponent_shape=opponent_shape,
        )
        self._last_assignment = RoleAssignment(mapping)
        return self._last_assignment

    def performance(self) -> ChampionPerformance:
        values = tuple(self._decision_times_ms)
        if not values:
            return ChampionPerformance(0, 0.0, 0.0, 0.0, 0.0, 0)
        ordered = sorted(values)
        p99_index = max(0, math.ceil(len(ordered) * 0.99) - 1)
        return ChampionPerformance(
            sample_count=len(values),
            latest_ms=values[-1],
            average_ms=sum(values) / len(values),
            p99_ms=ordered[p99_index],
            maximum_ms=ordered[-1],
            budget_exceeded_count=self._budget_exceeded_count,
        )

    def diagnostics(self) -> Mapping[str, object] | None:
        snapshot = self._last_snapshot
        if snapshot is None:
            return None
        performance = self.performance()
        pressure_time = (
            None if snapshot.pressure is None else snapshot.pressure.pressure_time_sec
        )
        return {
            "name": "champion",
            "handler_id": snapshot.handler_id,
            "cover_id": snapshot.cover_id,
            "team_phase": snapshot.team_phase.value,
            "ball_ownership": {
                "owner_id": snapshot.ball_owner_id,
                "owner_role": snapshot.ball_owner_role.value,
                "phase_started_at_sec": snapshot.coordination_since_sec,
                "reason": snapshot.coordination_reason,
                "field_candidate_id": snapshot.field_candidate_id,
            },
            "marking": {
                "marker_id": snapshot.marker_id,
                "opponent_id": snapshot.marked_opponent_id,
            },
            "roles": dict(self._last_assignment.by_player),
            "baseline_roles": (
                None
                if snapshot.baseline_assignment is None
                else dict(snapshot.baseline_assignment.by_player)
            ),
            "chase_target": {
                "x": round(snapshot.chase_target.x, 3),
                "y": round(snapshot.chase_target.y, 3),
            },
            "prediction_state": (
                None
                if snapshot.ball_prediction is None
                else snapshot.ball_prediction.motion_state
            ),
            "ball_path": (
                None
                if snapshot.ball_trajectory is None
                else {
                    "usable": snapshot.ball_trajectory.usable,
                    "model": snapshot.ball_trajectory.model_name,
                    "segment": snapshot.ball_trajectory.segment_id,
                    "invalid_reason": snapshot.ball_trajectory.invalid_reason,
                }
            ),
            "handler_eta": (
                None
                if snapshot.handler_eta is None
                else {
                    "eta_sec": snapshot.handler_eta.eta_sec,
                    "earliest_sec": snapshot.handler_eta.earliest_sec,
                    "latest_sec": snapshot.handler_eta.latest_sec,
                    "confidence": round(snapshot.handler_eta.confidence, 3),
                    "model": snapshot.handler_eta.model_name,
                    "alternate_eta_sec": snapshot.handler_eta.alternate_eta_sec,
                    "selection_reason": snapshot.handler_eta.selection_reason,
                }
            ),
            "intercept": (
                None
                if snapshot.intercept is None
                else {
                    "point": {
                        "x": round(snapshot.intercept.intercept_point.x, 3),
                        "y": round(snapshot.intercept.intercept_point.y, 3),
                    },
                    "time_sec": snapshot.intercept.intercept_time_sec,
                    "robot_eta_sec": snapshot.intercept.robot_eta_sec,
                    "margin_sec": snapshot.intercept.arrival_margin_sec,
                    "confidence": round(snapshot.intercept.confidence, 3),
                    "reason": snapshot.intercept.reason,
                }
            ),
            "pressure_level": snapshot.pressure_level,
            "pressure_time_sec": (
                None if pressure_time is None else round(pressure_time, 3)
            ),
            "watchdog": (
                None
                if snapshot.watchdog is None
                else {
                    "phase": snapshot.watchdog.phase.value,
                    "escape": snapshot.watchdog.should_escape,
                    "reason": snapshot.watchdog.reason,
                }
            ),
            "action": (
                None
                if snapshot.action_selection is None
                else {
                    "kind": snapshot.action_selection.selected.kind.value,
                    "target": {
                        "x": round(snapshot.action_selection.selected.target.x, 3),
                        "y": round(snapshot.action_selection.selected.target.y, 3),
                    },
                    "utility": round(snapshot.action_selection.selected.utility, 3),
                    "reason": snapshot.action_selection.selected.reason,
                    "generated": snapshot.action_selection.generated_count,
                    "finalists": len(snapshot.action_selection.finalists),
                    "executing": self.tuning.enable_action_execution,
                }
            ),
            "kickoff": (
                None
                if snapshot.kickoff is None
                else {
                    "epoch": snapshot.kickoff.epoch,
                    "phase": snapshot.kickoff.phase.value,
                    "active": snapshot.kickoff.active,
                    "elapsed_sec": round(snapshot.kickoff.elapsed_sec, 3),
                    "reason": snapshot.kickoff.reason,
                }
            ),
            "match": (
                None
                if snapshot.match_risk is None
                else {
                    "risk": round(snapshot.match_risk.value, 3),
                    "score_difference": snapshot.match_risk.score_difference,
                    "players": (
                        f"{snapshot.match_risk.own_active}v"
                        f"{snapshot.match_risk.opponent_active}"
                    ),
                }
            ),
            "opponent_shape": (
                None
                if snapshot.opponent_shape is None
                else {
                    "samples": snapshot.opponent_shape.samples,
                    "ball_density": round(snapshot.opponent_shape.ball_density, 3),
                    "deepest_x": snapshot.opponent_shape.deepest_x,
                    "lateral_compactness": round(
                        snapshot.opponent_shape.lateral_compactness, 3
                    ),
                }
            ),
            "performance": {
                "samples": performance.sample_count,
                "latest_ms": round(performance.latest_ms, 3),
                "average_ms": round(performance.average_ms, 3),
                "p99_ms": round(performance.p99_ms, 3),
                "max_ms": round(performance.maximum_ms, 3),
                "budget_exceeded": performance.budget_exceeded_count,
            },
        }

    def handler_chase_target(self, player_id: int, context: PlayContext) -> Pose2D:
        """Return the stabilized predicted target only to the assigned Handler."""

        snapshot = self._last_snapshot
        if (
            snapshot is not None
            and snapshot.handler_id == player_id
            and self.tuning.enable_prediction
        ):
            return snapshot.chase_target
        ball = context.known_ball
        return Pose2D(ball.x, ball.y, 0.0)

    def handler_kick_target(self, player_id: int, context: PlayContext) -> Pose2D:
        fallback = self._default_kick_target(player_id, context)
        snapshot = self._last_snapshot
        if snapshot is not None and snapshot.kickoff is not None:
            kickoff = snapshot.kickoff
            if (
                kickoff.active
                and player_id == kickoff.first_player_id
                and kickoff.phase in {
                    KickoffPhase.FIRST_TOUCH_ACTIVE,
                    KickoffPhase.VERIFY_FIRST_TOUCH,
                }
                and kickoff.first_target is not None
            ):
                return kickoff.first_target
            if (
                kickoff.active
                and player_id == kickoff.second_player_id
                and kickoff.phase in {
                    KickoffPhase.SECOND_PLAYER_ACQUIRE,
                    KickoffPhase.SECOND_KICK_ACTIVE,
                }
                and kickoff.second_target is not None
            ):
                return kickoff.second_target
        if self.tuning.enable_set_pieces:
            restart = own_restart_target(
                self.kit.config,
                self.kit.field,
                context,
                fallback,
                corner_lane_open=self.kit.targeting.shot_lane_is_clear(context),
            )
            if restart != fallback:
                return restart
        if (
            not self.tuning.enable_action_execution
            or snapshot is None
            or snapshot.handler_id != player_id
            or snapshot.action_selection is None
        ):
            return fallback
        return snapshot.action_selection.selected.target

    def handler_kick_power(
        self, player_id: int, context: PlayContext
    ) -> float | None:
        snapshot = self._last_snapshot
        if snapshot is None or snapshot.kickoff is None:
            return None
        kickoff = snapshot.kickoff
        if (
            kickoff.active
            and player_id == kickoff.first_player_id
            and kickoff.phase in {
                KickoffPhase.FIRST_TOUCH_ACTIVE,
                KickoffPhase.VERIFY_FIRST_TOUCH,
            }
        ):
            return self.tuning.kickoff_first_touch_power
        return None

    def _select_action(
        self,
        context: PlayContext,
        handler_id: int | None,
        pressure_level: str,
        watchdog: AttackWatchdogStatus | None,
        now: float,
        risk_value: float,
    ) -> ActionSelection | None:
        if not self.tuning.enable_action_selection or handler_id is None:
            return None
        fallback = self._default_kick_target(handler_id, context)
        selection = self._action_selector.select(
            handler_id=handler_id,
            context=context,
            pressure_level=pressure_level,
            fallback_target=fallback,
            risk_value=risk_value,
        )
        return self._stabilize_action(selection, handler_id, watchdog, now)

    def _stabilize_action(
        self,
        selection: ActionSelection,
        handler_id: int,
        watchdog: AttackWatchdogStatus | None,
        now: float,
    ) -> ActionSelection:
        previous_snapshot = self._last_snapshot
        previous = (
            None
            if previous_snapshot is None
            or previous_snapshot.handler_id != handler_id
            or previous_snapshot.action_selection is None
            else previous_snapshot.action_selection.selected
        )
        if previous is None:
            self._action_selected_at = now
            return selection

        retained = next(
            (
                candidate
                for candidate in selection.finalists
                if _same_action(candidate, previous)
            ),
            None,
        )
        should_escape = (
            self.tuning.enable_watchdog_escape
            and watchdog is not None
            and watchdog.should_escape
        )
        if should_escape:
            alternatives = tuple(
                candidate
                for candidate in selection.finalists
                if not _same_action(candidate, previous)
            )
            if alternatives:
                escaped = min(
                    alternatives,
                    key=lambda candidate: (candidate.turn_cost, -candidate.utility),
                )
                self._action_selected_at = now
                self._attack_watchdog.reset(handler_id)
                return replace(selection, selected=escaped)

        if retained is None:
            self._action_selected_at = now
            return selection
        hold_complete = now - self._action_selected_at >= self.tuning.action_min_hold_sec
        challenger_wins = (
            selection.selected.utility
            >= retained.utility + self.tuning.action_switch_margin
        )
        if hold_complete and challenger_wins:
            self._action_selected_at = now
            return selection
        return replace(selection, selected=retained)

    def _default_kick_target(self, player_id: int, context: PlayContext) -> Pose2D:
        slot = self.kit.config.ready_slot_for_player(player_id)
        if slot == ReadySlot.SIDE:
            return self.kit.targeting.select_clear_or_pass_target(
                player_id, context, self.kit.is_player_allowed
            )
        return self.kit.targeting.select_kick_target(
            player_id, context, self.kit.is_player_allowed
        )

    def outlet_target(self, player_id: int, context: PlayContext) -> Pose2D:
        if not self.tuning.enable_outlet_positioning:
            return self.kit.targeting.support_target(
                player_id, context, self.kit.is_player_allowed
            )
        ball = context.known_ball
        half_length = self.kit.config.field_length / 2.0
        if ball.x < -half_length * 0.20:
            target_x = ball.x + 1.6
            lateral = 1.45
        elif ball.x < half_length * 0.35:
            target_x = ball.x + 1.9
            lateral = 1.65
        else:
            target_x = min(self.kit.field.opponent_goal_x() - 1.3, ball.x + 1.0)
            lateral = 1.25

        snapshot = self._last_snapshot
        if snapshot is not None and snapshot.match_risk is not None:
            target_x += (snapshot.match_risk.value - 0.5) * 0.8
        if (
            snapshot is not None
            and snapshot.opponent_shape is not None
            and snapshot.opponent_shape.samples > 0
            and (
                snapshot.opponent_shape.ball_density >= 0.66
                or snapshot.opponent_shape.lateral_compactness <= 1.8
            )
        ):
            lateral = min(self.kit.config.field_width / 2.0 - 0.4, lateral + 0.25)

        if (
            self.kit.targeting.ball_in_own_defensive_area(ball)
            and snapshot is not None
            and snapshot.cover_id is not None
        ):
            target_x = min(target_x, self.kit.field.own_goal_x() + 2.4)
            lateral = min(lateral, 1.05)

        candidates = (
            self.kit.field.clamp_inside_field(Pose2D(target_x, -lateral, 0.0)),
            self.kit.field.clamp_inside_field(Pose2D(target_x, lateral, 0.0)),
        )
        opponents = tuple(
            robot.pose for robot in context.opponents.values() if robot.pose is not None
        )

        def outlet_score(target: Pose2D) -> tuple[float, float]:
            nearest = min(
                (
                    math.hypot(target.x - opponent.x, target.y - opponent.y)
                    for opponent in opponents
                ),
                default=self.kit.config.field_width,
            )
            opposite_ball_side = 1.0 if target.y * ball.y <= 0.0 else 0.0
            return nearest, opposite_ball_side

        selected = max(candidates, key=outlet_score)
        return Pose2D(
            selected.x,
            selected.y,
            self.kit.field.face_ball_theta(selected.x, selected.y, ball),
        )

    def goalkeeper_owns_ball(self) -> bool:
        snapshot = self._last_snapshot
        return (
            snapshot is not None
            and snapshot.ball_owner_role is BallOwnerRole.KEEPER
            and snapshot.ball_owner_id == snapshot.cover_id
        )

    def marker_position_target(
        self, player_id: int, context: PlayContext
    ) -> Pose2D:
        snapshot = self._last_snapshot
        if (
            snapshot is not None
            and snapshot.marker_id == player_id
            and self._marker_target is not None
        ):
            return self._marker_target
        return self.outlet_target(player_id, context)

    def second_ball_position_target(
        self, player_id: int, context: PlayContext
    ) -> Pose2D:
        snapshot = self._last_snapshot
        if (
            snapshot is not None
            and snapshot.field_candidate_id == player_id
            and self._second_ball_target is not None
        ):
            return self._second_ball_target
        return self.outlet_target(player_id, context)

    def _update_keeper_takeover(
        self,
        *,
        context: PlayContext,
        cover_id: int | None,
        field_candidate_id: int | None,
        keeper_eta: RobotArrivalEstimate | None,
        handler_eta: RobotArrivalEstimate | None,
        prediction: BallMotionPrediction | None,
        now: float,
    ) -> KeeperTakeoverStatus:
        if not self.tuning.enable_team_coordination:
            self._keeper_takeover.reset()
            return self._keeper_takeover.update(
                now,
                KeeperTakeoverEvidence(False, False, False, False),
            )

        ball = context.known_ball
        defensive = self.kit.targeting.ball_in_own_defensive_area(ball)
        set_play_clear = context.known_game.set_play is SetPlay.NONE
        keeper_pose = (
            None
            if cover_id is None or cover_id not in context.teammates
            else context.teammates[cover_id].pose
        )
        handler_pose = (
            None
            if field_candidate_id is None
            or field_candidate_id not in context.teammates
            else context.teammates[field_candidate_id].pose
        )
        keeper_distance = (
            math.inf
            if keeper_pose is None
            else math.hypot(ball.x - keeper_pose.x, ball.y - keeper_pose.y)
        )
        handler_distance = (
            math.inf
            if handler_pose is None
            else math.hypot(ball.x - handler_pose.x, ball.y - handler_pose.y)
        )

        opponents = tuple(
            robot.pose
            for player_id, robot in context.opponents.items()
            if robot.pose is not None
            and robot.is_recent(now)
            and context.known_game.is_active_player(
                self.kit.config.opponent_team_id(), player_id
            )
        )
        nearest_opponent = min(
            (
                math.hypot(ball.x - pose.x, ball.y - pose.y)
                for pose in opponents
            ),
            default=math.inf,
        )
        own_goal_x = self.kit.field.own_goal_x()
        goal_corridor = (
            ball.x <= own_goal_x + min(2.60, self.kit.config.penalty_area_length)
            and abs(ball.y) <= self.kit.config.penalty_area_width / 2.0
        )
        rolling_toward_goal = (
            prediction is not None
            and prediction.motion_state == "rolling"
            and prediction.velocity_x <= -0.30
        )
        opponent_threat = nearest_opponent <= 1.25
        emergency_threat = goal_corridor or rolling_toward_goal or opponent_threat

        eta_advantage = False
        if field_candidate_id is None:
            eta_advantage = True
        elif (
            keeper_eta is not None
            and handler_eta is not None
            and keeper_eta.reachable
            and handler_eta.reachable
            and keeper_eta.eta_sec is not None
            and handler_eta.eta_sec is not None
        ):
            keeper_cost = keeper_eta.eta_sec + 0.20 * keeper_eta.uncertainty_sec
            handler_cost = handler_eta.eta_sec + 0.20 * handler_eta.uncertainty_sec
            eta_advantage = (
                keeper_cost + self.tuning.keeper_takeover_advantage_sec
                < handler_cost
            )
        distance_advantage = (
            keeper_distance + 0.65 < handler_distance
            or keeper_distance <= 0.80
        )
        keeper_advantage = eta_advantage or distance_advantage
        eligible = (
            cover_id is not None
            and keeper_pose is not None
            and set_play_clear
            and keeper_distance <= self.tuning.keeper_takeover_max_distance_m
        )
        release_confirmed = (
            prediction is not None
            and prediction.motion_state == "rolling"
            and prediction.velocity_x >= 0.45
            and keeper_distance >= 0.80
        )
        reasons = []
        if goal_corridor:
            reasons.append("deep goal corridor")
        if rolling_toward_goal:
            reasons.append("ball rolling toward own goal")
        if opponent_threat:
            reasons.append("opponent close to ball")
        if eta_advantage:
            reasons.append("keeper ETA advantage")
        elif distance_advantage:
            reasons.append("keeper distance advantage")
        return self._keeper_takeover.update(
            now,
            KeeperTakeoverEvidence(
                eligible=eligible,
                ball_in_defensive_area=defensive,
                emergency_threat=emergency_threat,
                keeper_advantage=keeper_advantage,
                release_confirmed=release_confirmed,
                reason=", ".join(reasons),
            ),
        )

    def _is_defending(
        self, context: PlayContext, active: tuple[int, ...], now: float
    ) -> bool:
        ball = context.known_ball
        if ball.x <= -0.25:
            return True
        own_distances = [
            math.hypot(ball.x - robot.pose.x, ball.y - robot.pose.y)
            for player_id in active
            if (robot := context.teammates.get(player_id)) is not None
            and robot.pose is not None
        ]
        opponent_team = self.kit.config.opponent_team_id()
        opponent_distances = [
            math.hypot(ball.x - robot.pose.x, ball.y - robot.pose.y)
            for player_id, robot in context.opponents.items()
            if robot.pose is not None
            and robot.is_recent(now)
            and context.known_game.is_active_player(opponent_team, player_id)
        ]
        return (
            ball.x < 1.0
            and bool(opponent_distances)
            and (
                not own_distances
                or min(opponent_distances) <= min(own_distances) + 0.35
            )
        )

    def _select_marker_target(
        self,
        marker_id: int,
        context: PlayContext,
        active: tuple[int, ...],
        now: float,
    ) -> tuple[int | None, Pose2D | None]:
        ball = context.known_ball
        opponent_team = self.kit.config.opponent_team_id()
        opponents = tuple(
            (player_id, robot.pose)
            for player_id, robot in context.opponents.items()
            if robot.pose is not None
            and robot.is_recent(now)
            and context.known_game.is_active_player(opponent_team, player_id)
        )
        if not opponents:
            self._marked_opponent_id = None
            return None, None

        carrier_id, carrier_pose = min(
            opponents,
            key=lambda item: math.hypot(
                item[1].x - ball.x, item[1].y - ball.y
            ),
        )
        if math.hypot(carrier_pose.x - ball.x, carrier_pose.y - ball.y) > 1.35:
            carrier_id = None
        defender_poses = tuple(
            robot.pose
            for player_id in active
            if player_id != marker_id
            and (robot := context.teammates.get(player_id)) is not None
            and robot.pose is not None
        )
        own_goal = Pose2D(self.kit.field.own_goal_x(), 0.0, 0.0)
        threat = select_dangerous_opponent(
            opponents,
            ball=Pose2D(ball.x, ball.y, 0.0),
            own_goal=own_goal,
            defenders=defender_poses,
            excluded_player_id=carrier_id,
            previous_player_id=self._marked_opponent_id,
            switch_margin=self.tuning.marker_switch_margin,
        )
        if threat is None:
            self._marked_opponent_id = None
            return None, None
        self._marked_opponent_id = threat.player_id
        target = marker_target(
            threat.pose,
            ball=Pose2D(ball.x, ball.y, 0.0),
            own_goal=own_goal,
            marking_distance_m=self.tuning.marker_distance_m,
        )
        return threat.player_id, self.kit.field.clamp_inside_field(target)

    def _compute_second_ball_target(
        self, player_id: int, context: PlayContext
    ) -> Pose2D:
        ball = context.known_ball
        own_goal_x = self.kit.field.own_goal_x()
        target_x = max(own_goal_x + 1.60, ball.x + 1.15)
        target_y = ball.y * 0.55
        target = self.kit.field.clamp_inside_field(
            Pose2D(target_x, target_y, 0.0)
        )
        return Pose2D(
            target.x,
            target.y,
            self.kit.field.face_ball_theta(target.x, target.y, ball),
        )

    def _active_players(self, context: PlayContext, now: float) -> tuple[int, ...]:
        game = context.known_game
        team_id = self.kit.config.team_id
        active = tuple(
            player_id
            for player_id in self.kit.config.player_ids
            if game.is_active_player(team_id, player_id)
            and (robot := context.teammates.get(player_id)) is not None
            and robot.pose is not None
            and robot.is_recent(now)
        )
        for player_id in active:
            robot = context.teammates[player_id]
            assert robot.pose is not None
            observed_at = robot.last_seen_at if robot.last_seen_at > 0.0 else now
            self._robot_motion.update(player_id, robot.pose, observed_at)
        return active

    def _select_cover(self, active: tuple[int, ...], context: PlayContext) -> int | None:
        if not active:
            return None
        configured = self.kit.config.goalkeeper_player_id()
        if configured in active:
            return configured

        # Team-view coordinates always defend toward -x.  In an emergency the
        # deepest active robot becomes cover; avoid pulling the current Handler
        # back when another robot is available.
        candidates = list(active)
        if len(candidates) > 1 and self._handler_id in candidates:
            candidates.remove(self._handler_id)
        def depth(player_id: int) -> tuple[float, int]:
            pose = context.teammates[player_id].pose
            assert pose is not None
            return pose.x, player_id

        return min(candidates, key=depth)

    def _select_handler(
        self,
        candidates: tuple[int, ...],
        target: Pose2D,
        context: PlayContext,
        now: float,
    ) -> int | None:
        if not candidates:
            if self._handler_id is not None:
                self._attack_watchdog.reset(self._handler_id)
            self._handler_id = None
            return None

        scores = {
            player_id: self._claim_cost(player_id, target, context, now)
            for player_id in candidates
        }
        challenger = min(candidates, key=lambda pid: (scores[pid], pid))
        current = self._handler_id
        if current not in scores:
            selected = challenger
        else:
            cooldown_done = (
                now - self._handler_selected_at
                >= self.tuning.handler_switch_cooldown_sec
            )
            clearly_better = (
                scores[challenger] + self.tuning.handler_switch_margin_sec < scores[current]
            )
            selected = challenger if cooldown_done and clearly_better else current

        if selected != self._handler_id:
            if self._handler_id is not None:
                self._attack_watchdog.reset(self._handler_id)
            self._handler_id = selected
            self._handler_selected_at = now
        return selected

    def _claim_cost(
        self,
        player_id: int,
        target: Pose2D,
        context: PlayContext,
        now: float,
    ) -> float:
        pose = context.teammates[player_id].pose
        assert pose is not None
        slot = self.kit.config.ready_slot_for_player(player_id)
        slot_bias = {
            ReadySlot.CENTER: -0.20,
            ReadySlot.SIDE: -0.10,
            ReadySlot.KEEPER: 0.30,
        }.get(slot, 0.0)
        if self.tuning.enable_robot_eta:
            estimate = self._estimate_arrival(player_id, target, context, now)
            if estimate.reachable and estimate.eta_sec is not None:
                return estimate.eta_sec + 0.20 * estimate.uncertainty_sec + slot_bias
        dx = target.x - pose.x
        dy = target.y - pose.y
        distance_cost = math.hypot(dx, dy) / max(0.01, self.tuning.translation_speed_mps)
        desired_heading = math.atan2(dy, dx)
        heading_error = abs(_wrap_angle(desired_heading - pose.theta))
        turn_cost = heading_error / max(0.01, self.tuning.yaw_speed_radps)
        return distance_cost + turn_cost + slot_bias

    def _estimate_arrival(
        self,
        player_id: int,
        target: Pose2D,
        context: PlayContext,
        now: float,
        *,
        track_target: bool = True,
    ) -> RobotArrivalEstimate:
        return self._arrival_estimator.estimate(
            self._arrival_query(
                player_id, target, context, now, track_target=track_target
            )
        )

    def _arrival_query(
        self,
        player_id: int,
        target: Pose2D,
        context: PlayContext,
        now: float,
        *,
        track_target: bool,
    ) -> RobotArrivalQuery:
        robot = context.teammates[player_id]
        assert robot.pose is not None
        pose = robot.pose
        desired = math.atan2(target.y - pose.y, target.x - pose.x)
        oriented_target = Pose2D(target.x, target.y, desired)
        changed = True
        age = 0.0
        if track_target:
            previous = self._eta_target_state.get(player_id)
            changed = (
                previous is None
                or math.hypot(target.x - previous[0].x, target.y - previous[0].y) > 0.35
            )
            if changed:
                self._eta_target_state[player_id] = (target, now)
            else:
                age = max(0.0, now - previous[1])
        motion = self._robot_motion.motion(player_id)
        return RobotArrivalQuery(
            now_sec=now,
            robot_id=player_id,
            pose=pose,
            raw_target=oriented_target,
            observed_linear_speed_mps=(
                None if motion is None else motion.linear_speed_mps
            ),
            observed_yaw_rate_radps=(
                None if motion is None else motion.yaw_rate_radps
            ),
            arrive_distance_m=0.15,
            arrive_angle_rad=0.20,
            linear_speed_limit_mps=self.kit.config.strategy.max_linear_speed,
            runtime_mode="walk",
            fall_state="normal",
            target_age_sec=age,
            target_changed=changed,
        )

    def _update_predictions(
        self, context: PlayContext
    ) -> tuple[BallMotionPrediction | None, BallTrajectoryPrediction | None]:
        if not self.tuning.enable_prediction:
            return None, None
        ball = context.known_ball
        stamp = ball.last_seen_at
        if stamp > 0.0 and stamp != self._last_ball_stamp:
            self._ball_predictor.add_ball(ball)
            self._ball_trajectory_predictor.add_ball(ball)
            self._last_ball_stamp = stamp
        return self._ball_predictor.predict(), self._ball_trajectory_predictor.predict()

    def _chase_target(
        self,
        context: PlayContext,
        prediction: BallMotionPrediction | None,
        trajectory: BallTrajectoryPrediction | None,
        candidates: tuple[int, ...],
        now: float,
    ) -> tuple[Pose2D, InterceptEstimate | None]:
        ball = context.known_ball
        current = Pose2D(ball.x, ball.y, 0.0)
        if (
            self.tuning.enable_intercept
            and self.tuning.enable_robot_eta
            and trajectory is not None
            and trajectory.usable
        ):
            intercepts = []
            for player_id in candidates:
                base_query = self._arrival_query(
                    player_id, current, context, now, track_target=False
                )
                estimate = estimate_earliest_intercept(
                    self._arrival_estimator,
                    base_query,
                    trajectory,
                    safety_margin_sec=self.tuning.intercept_safety_margin_sec,
                )
                if estimate.intercept_time_sec is not None:
                    intercepts.append(estimate)
            if intercepts:
                selected = min(
                    intercepts,
                    key=lambda item: (
                        item.intercept_time_sec or math.inf,
                        -item.confidence,
                        item.robot_id,
                    ),
                )
                weight = min(
                    self.tuning.lead_max_weight,
                    max(0.15, selected.confidence),
                )
                return (
                    Pose2D(
                        current.x
                        + (selected.intercept_point.x - current.x) * weight,
                        current.y
                        + (selected.intercept_point.y - current.y) * weight,
                        0.0,
                    ),
                    selected,
                )
        if trajectory is not None and trajectory.is_strategy_usable(
            now,
            self.tuning.lead_horizon_sec,
            min_confidence=max(0.20, self.tuning.lead_min_confidence * 0.75),
            max_uncertainty_m=min(0.60, self.tuning.lead_max_uncertainty_m),
        ):
            predicted = trajectory.position_ahead(
                now, self.tuning.lead_horizon_sec
            )
            relative = (
                max(0.0, now - trajectory.observed_at_sec)
                + self.tuning.lead_horizon_sec
            )
            if predicted is not None:
                confidence = trajectory.confidence_at(relative)
                uncertainty = trajectory.uncertainty_at(relative) or 0.60
                weight = self.tuning.lead_max_weight * max(
                    0.0,
                    min(1.0, confidence * (1.0 - uncertainty / 0.60)),
                )
                return (
                    Pose2D(
                        current.x + (predicted.x - current.x) * weight,
                        current.y + (predicted.y - current.y) * weight,
                        0.0,
                    ),
                    None,
                )
        if (
            prediction is None
            or prediction.motion_state != "rolling"
            or prediction.confidence < self.tuning.lead_min_confidence
            or prediction.uncertainty_radius_m > self.tuning.lead_max_uncertainty_m
        ):
            return current, None
        predicted = prediction.position_at(self.tuning.lead_horizon_sec)
        confidence_scale = (
            prediction.confidence - self.tuning.lead_min_confidence
        ) / max(1e-6, 1.0 - self.tuning.lead_min_confidence)
        uncertainty_scale = 1.0 - (
            prediction.uncertainty_radius_m / self.tuning.lead_max_uncertainty_m
        )
        weight = self.tuning.lead_max_weight * max(
            0.0, min(1.0, confidence_scale * uncertainty_scale)
        )
        return (
            Pose2D(
                current.x + (predicted.x - current.x) * weight,
                current.y + (predicted.y - current.y) * weight,
                0.0,
            ),
            None,
        )

    def _update_pressure(
        self,
        context: PlayContext,
        target: Pose2D,
        prediction: BallMotionPrediction | None,
        now: float,
    ) -> OpponentPressureReport | None:
        if not self.tuning.enable_pressure:
            return None
        if now - self._pressure_updated_at < self.tuning.pressure_cache_sec:
            return self._pressure
        self._pressure = self._pressure_estimator.estimate(
            now_sec=now,
            opponents=context.opponents,
            target=target,
            ball_prediction=prediction,
            game_state=context.known_game,
            opponent_team_id=self.kit.config.opponent_team_id(),
        )
        self._pressure_updated_at = now
        return self._pressure

    @staticmethod
    def _pressure_level(report: OpponentPressureReport | None) -> str:
        if report is None or report.has_unknown_pressure:
            return "urgent"
        seconds = report.pressure_time_sec
        if seconds is None:
            return "manageable"
        if seconds <= 0.55:
            return "immediate"
        if seconds <= 1.20:
            return "urgent"
        return "manageable"

    def _update_watchdog(
        self,
        context: PlayContext,
        handler_id: int | None,
        prediction: BallMotionPrediction | None,
        now: float,
    ) -> AttackWatchdogStatus | None:
        if not self.tuning.enable_watchdog:
            if handler_id is not None:
                self._attack_watchdog.reset(handler_id)
            return None
        if handler_id is None:
            return None
        pose = context.teammates[handler_id].pose
        if pose is None:
            return None
        ball = context.known_ball
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        distance = math.hypot(dx, dy)
        angle_error = abs(_wrap_angle(math.atan2(dy, dx) - pose.theta))
        active = distance <= self.tuning.watchdog_activation_distance_m
        if not active:
            phase = AttackPhase.IDLE
        elif distance > self.tuning.watchdog_align_distance_m:
            phase = AttackPhase.APPROACH
        elif angle_error > self.tuning.watchdog_align_angle_rad:
            phase = AttackPhase.ALIGN
        else:
            phase = AttackPhase.EXECUTE
        return self._attack_watchdog.update(
            handler_id,
            AttackAttemptObservation(
                t_sec=now,
                phase=phase,
                ball_distance_m=distance,
                angle_error_rad=angle_error,
                ball_speed_mps=0.0 if prediction is None else prediction.speed_mps,
                active=active,
            ),
        )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _same_action(left, right) -> bool:
    return (
        left.kind is right.kind
        and left.receiver_id == right.receiver_id
        and math.hypot(
            left.target.x - right.target.x,
            left.target.y - right.target.y,
        )
        <= 0.20
    )


class ChampionChaserRole(ChaserRole):
    """Default chaser motion with an optional Champion kick-target source."""

    def __init__(self, playbook: ChampionPlaybook):
        self._playbook = playbook

    def target(self, kit, player_id: int, context: PlayContext) -> Pose2D:
        return self._playbook.handler_chase_target(player_id, context)

    def kick_target(self, kit, player_id: int, context: PlayContext) -> Pose2D:
        return self._playbook.handler_kick_target(player_id, context)

    def build_subtree(self, kit, player_id: int):
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda context: self.target(kit, player_id, context),
                kick_target_fn=lambda context: self.kick_target(
                    kit, player_id, context
                ),
                reason_fn=lambda: self._approach_reason(kit, player_id),
                kick_reason_fn=lambda target: self._kick_reason(
                    kit, player_id, target
                ),
                kick_power_fn=lambda context: self._playbook.handler_kick_power(
                    player_id, context
                ),
            ),
        )


class ChampionSupporterRole(SupporterRole):
    """Two-candidate zonal Outlet that stays cheap enough for every tick."""

    def __init__(self, playbook: ChampionPlaybook):
        self._playbook = playbook

    def target(self, kit, player_id: int, context: PlayContext) -> Pose2D:
        return self._playbook.outlet_target(player_id, context)


class ChampionGoalkeeperRole(GoalkeeperRole):
    """Keeper may approach or kick only while the team grants ownership."""

    def __init__(self, playbook: ChampionPlaybook):
        self._playbook = playbook

    def wants_to_kick(self, kit, context: PlayContext) -> bool:
        if not self._playbook.tuning.enable_team_coordination:
            return super().wants_to_kick(kit, context)
        return self._playbook.goalkeeper_owns_ball()


class ChampionMarkerRole(RoleStrategy):
    """Goal-side marker for the selected dangerous off-ball opponent."""

    name = ROLE_MARKER

    def __init__(self, playbook: ChampionPlaybook):
        self._playbook = playbook

    def build_subtree(self, kit, player_id: int):
        return MoveToTarget(
            kit,
            player_id,
            lambda context: self._playbook.marker_position_target(
                player_id, context
            ),
            reason_fn=lambda: "marker block dangerous opponent",
            hold_vyaw=0.12,
        )


class ChampionSecondBallRole(RoleStrategy):
    """Field player protecting the clearance lane during keeper ownership."""

    name = ROLE_SECOND_BALL

    def __init__(self, playbook: ChampionPlaybook):
        self._playbook = playbook

    def build_subtree(self, kit, player_id: int):
        return MoveToTarget(
            kit,
            player_id,
            lambda context: self._playbook.second_ball_position_target(
                player_id, context
            ),
            reason_fn=lambda: "keeper takeover second-ball cover",
            hold_vyaw=0.12,
        )


__all__ = [
    "ChampionPerformance",
    "ChampionPlaybook",
    "ChampionSnapshot",
    "ChampionTuning",
    "ROLE_MARKER",
    "ROLE_SECOND_BALL",
]
