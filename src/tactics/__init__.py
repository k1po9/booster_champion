"""Pure tactic-model layer: geometry and control tools independent of BT and ROS.

Pose2D geometry helpers and team-view field frame tools
obstacle collection for opponents, teammates, and goal structure
tactic targets for support, passing, shooting, sideline recovery, and more
motion controller with avoidance, walking control, and kick commands
kick enter/exit hysteresis model

The PLAY-stage "which player chases" decision is not in this layer; it belongs
to the playbook layer in :mod:`src.play.playbook`.
"""

from .ball_prediction import (
    BallMotionPrediction,
    BallObservation,
    BallResistanceModel,
    BallStopPrediction,
    DEFAULT_BALL_RESISTANCE_MODEL,
    SlidingWindowBallPredictor,
    predict_ball_motion,
    predict_ball_stop,
)
from .ball_trajectory import (
    BallObservation as BallTrajectoryObservation,
    BallPathKnot,
    BallTrajectoryPrediction,
    EventBallTrajectoryPredictor,
    predict_launched_ball_path,
)
from .action_selection import (
    ActionCandidate,
    ActionKind,
    ActionSelection,
    ActionSelectionTuning,
    BoundedActionSelector,
)
from .attack_watchdog import (
    AttackAttemptObservation,
    AttackPhase,
    AttackWatchdog,
    AttackWatchdogConfig,
    AttackWatchdogStatus,
)
from .opponent_pressure import (
    OpponentMotion,
    OpponentMotionTracker,
    OpponentPressureConfig,
    OpponentPressureEstimate,
    OpponentPressureEstimator,
    OpponentPressureReport,
)
from .match_strategy import (
    KickoffPhase,
    KickoffStatus,
    KickoffTransaction,
    MatchRisk,
    OpponentShape,
    OpponentShapeTracker,
    calculate_match_risk,
    own_restart_target,
)
from .robot_eta import (
    DynamicRobotArrivalEstimator,
    InterceptEstimate,
    RobotArrivalEstimate,
    RobotArrivalQuery,
    RobotMotionEstimate,
    RobotMotionTracker,
    estimate_earliest_intercept,
)
from .kick_hysteresis import KickHysteresis
from .geometry import TeamFieldFrame
from .motion import MotionController
from .navigation import Obstacle, ObstacleCollector
from .ready_stance import ReadyStance
from .targeting import Targeting

__all__ = [
    "ActionCandidate",
    "ActionKind",
    "ActionSelection",
    "ActionSelectionTuning",
    "AttackAttemptObservation",
    "AttackPhase",
    "AttackWatchdog",
    "AttackWatchdogConfig",
    "AttackWatchdogStatus",
    "BallMotionPrediction",
    "BallObservation",
    "BallResistanceModel",
    "BallStopPrediction",
    "BallPathKnot",
    "BallTrajectoryObservation",
    "BallTrajectoryPrediction",
    "BoundedActionSelector",
    "DEFAULT_BALL_RESISTANCE_MODEL",
    "DynamicRobotArrivalEstimator",
    "EventBallTrajectoryPredictor",
    "InterceptEstimate",
    "KickHysteresis",
    "KickoffPhase",
    "KickoffStatus",
    "KickoffTransaction",
    "MatchRisk",
    "MotionController",
    "Obstacle",
    "ObstacleCollector",
    "OpponentMotion",
    "OpponentMotionTracker",
    "OpponentPressureConfig",
    "OpponentPressureEstimate",
    "OpponentPressureEstimator",
    "OpponentPressureReport",
    "OpponentShape",
    "OpponentShapeTracker",
    "ReadyStance",
    "RobotArrivalEstimate",
    "RobotArrivalQuery",
    "RobotMotionEstimate",
    "RobotMotionTracker",
    "SlidingWindowBallPredictor",
    "Targeting",
    "TeamFieldFrame",
    "calculate_match_risk",
    "estimate_earliest_intercept",
    "own_restart_target",
    "predict_ball_motion",
    "predict_ball_stop",
    "predict_launched_ball_path",
]
