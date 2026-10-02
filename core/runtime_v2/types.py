"""Public contracts for the verified capability runtime.

The runtime deliberately keeps semantic hypotheses separate from committed
physical state.  Every committed value carries a three-valued verdict and its
evidence provenance so that missing perception is never silently converted to
``False``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TruthValue(_StringEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class ObservationHealth(_StringEnum):
    VALID = "VALID"
    OCCLUDED = "OCCLUDED"
    AMBIGUOUS = "AMBIGUOUS"
    STALE = "STALE"
    SENSOR_FAULT = "SENSOR_FAULT"


class SpatialRelation(_StringEnum):
    FRONT = "FRONT"
    BACK = "BACK"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    INSIDE_ENVELOPE = "INSIDE_ENVELOPE"
    UNKNOWN = "UNKNOWN"


class PlacementRelation(_StringEnum):
    """Physical relation between a held payload and a destination opening.

    These labels intentionally describe evidence, not a task-specific phase.
    They are shared by geometry providers, verifiers, and the action compiler.
    """

    ABOVE_UNALIGNED = "ABOVE_UNALIGNED"
    ABOVE_ALIGNED = "ABOVE_ALIGNED"
    DESCENDING_CLEAR = "DESCENDING_CLEAR"
    RIM_CONTACT = "RIM_CONTACT"
    SEATED_HELD = "SEATED_HELD"
    RELEASED_STABLE = "RELEASED_STABLE"
    LOST = "LOST"
    UNKNOWN = "UNKNOWN"


class SpatialHealth(_StringEnum):
    VALID = "VALID"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    STALE = "STALE"
    SENSOR_FAULT = "SENSOR_FAULT"


class OptionName(_StringEnum):
    LOCATE_TARGET = "LOCATE_TARGET"
    MOVE_TO_HOVER = "MOVE_TO_HOVER"
    ALIGN_PREGRASP = "ALIGN_PREGRASP"
    DESCEND_TO_GRASP = "DESCEND_TO_GRASP"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_HOLD = "VERIFY_HOLD"
    LIFT_CLEAR = "LIFT_CLEAR"
    LOCATE_DESTINATION = "LOCATE_DESTINATION"
    TRANSFER = "TRANSFER"
    ALIGN_OPENING = "ALIGN_OPENING"
    DESCEND_TO_SEAT = "DESCEND_TO_SEAT"
    VERIFY_SEATED = "VERIFY_SEATED"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    RETREAT = "RETREAT"
    VERIFY_TASK = "VERIFY_TASK"
    RELOCALIZE = "RELOCALIZE"


class OptionStatus(_StringEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEED_DECISION = "NEED_DECISION"


class Verdict(_StringEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class FailureCode(_StringEnum):
    SENSOR_FAULT = "SENSOR_FAULT"
    TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
    TARGET_OCCLUDED = "TARGET_OCCLUDED"
    EMPTY_GRASP = "EMPTY_GRASP"
    LOST_HOLD = "LOST_HOLD"
    RIM_CONTACT = "RIM_CONTACT"
    NO_PROGRESS = "NO_PROGRESS"
    OSCILLATION = "OSCILLATION"
    OPTION_BUDGET_EXCEEDED = "OPTION_BUDGET_EXCEEDED"
    RECOVERY_BUDGET_EXCEEDED = "RECOVERY_BUDGET_EXCEEDED"
    DEPTH_AMBIGUOUS = "DEPTH_AMBIGUOUS"
    PERCEPTION_INFRASTRUCTURE_FAULT = "PERCEPTION_INFRASTRUCTURE_FAULT"


T = TypeVar("T")


@dataclass(frozen=True)
class EvidenceValue(Generic[T]):
    value: Optional[T]
    truth: TruthValue
    source: str
    confidence: float
    frame_id: Optional[int]


@dataclass(frozen=True)
class SpatialToolResult:
    """A geometry provider's observation; never a state commitment."""

    source: str
    instance_id: Optional[str]
    frame_id: Optional[int]
    target_points_camera: tuple[tuple[float, float, float], ...] = ()
    target_points_pixels: tuple[tuple[float, float], ...] = ()
    target_points_world: tuple[tuple[float, float, float], ...] = ()
    target_to_gripper_xyz: Optional[tuple[float, float, float]] = None
    # Diagonal covariance (m^2), when a provider supplies one.
    covariance: Optional[tuple[float, ...]] = None
    # Prefer this explicit standard-deviation field (m); older profiles keep
    # their historical interpretation of `covariance` for compatibility.
    uncertainty_std_m: Optional[tuple[float, float, float]] = None
    confidence: float = 0.0
    health: SpatialHealth = SpatialHealth.UNKNOWN
    stale: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialBelief:
    """Fused relative geometry with explicit uncertainty and conflicts."""

    fused_relative_xyz: Optional[tuple[float, float, float]] = None
    relations: tuple[SpatialRelation, ...] = (SpatialRelation.UNKNOWN,)
    uncertainty: Optional[tuple[float, float, float]] = None
    agreeing_sources: tuple[str, ...] = ()
    conflicting_sources: tuple[str, ...] = ()
    health: SpatialHealth = SpatialHealth.UNKNOWN
    frame_id: Optional[int] = None
    instance_id: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualMemoryEntry:
    instance_id: Optional[str]
    grasp_epoch: int
    frame_id: int
    episode_id: Optional[str] = None
    before_frame_id: Optional[int] = None
    before_agentview_ref: Optional[str] = None
    before_wrist_ref: Optional[str] = None
    agentview_ref: Optional[str] = None
    wrist_ref: Optional[str] = None
    camera_pose: Optional[tuple[float, ...]] = None
    eef_pose: Optional[tuple[float, ...]] = None
    requested_action: Optional[str] = None
    authorized_action: Optional[str] = None
    executed_action: Optional[str] = None
    motion_delta: Optional[tuple[float, float, float]] = None
    depth_summary: dict[str, Any] = field(default_factory=dict)
    route_phase: Optional[str] = None
    placement_summary: dict[str, Any] = field(default_factory=dict)
    route_epoch: int = 0
    predicted_effect: Optional[tuple[float, ...]] = None
    observed_effect: dict[str, Any] = field(default_factory=dict)
    effect_status: Optional[str] = None
    decision_summary: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GripperEnvelope:
    """Embodiment descriptor, expressed in the gripper frame."""

    name: str = "unknown"
    finger_clearance_m: float = 0.0
    fingertip_depth_m: float = 0.0
    closing_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    swept_volume_extents_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SemanticPregraspAction(_StringEnum):
    SELECT_GRASP = "SELECT_GRASP"
    VISUAL_ALIGN = "VISUAL_ALIGN"
    CORRECT_DEPTH = "CORRECT_DEPTH"
    CORRECT_LATERAL = "CORRECT_LATERAL"
    CORRECT_HEIGHT = "CORRECT_HEIGHT"
    PROBE_DEPTH = "PROBE_DEPTH"
    GRASP = "GRASP"
    UNKNOWN = "UNKNOWN"


class SemanticPlacementAction(_StringEnum):
    """Semantic placement choice exposed to a small agent.

    The runtime compiles this choice into a signed atomic motion.  The agent
    never needs to learn simulator-specific MV_LEFT/MV_RIGHT conventions.
    """

    SELECT_PLACE = "SELECT_PLACE"
    CORRECT_LATERAL = "CORRECT_LATERAL"
    CORRECT_DEPTH = "CORRECT_DEPTH"
    DESCEND = "DESCEND"
    PROBE = "PROBE"
    RECOVER_CLEAR = "RECOVER_CLEAR"
    READY_TO_VERIFY = "READY_TO_VERIFY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PlacementCandidate:
    """A provider-produced placement hypothesis in one metric frame."""

    candidate_id: str
    object_pose_world: Optional[tuple[float, ...]] = None
    eef_pose_world: Optional[tuple[float, ...]] = None
    source: str = "unknown"
    reachable: Optional[bool] = None
    orientation_supported: bool = True
    collision_clearance_m: Optional[float] = None
    containment_margin_m: Optional[float] = None
    uncertainty_m: Optional[tuple[float, ...]] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementBelief:
    """Unified spatial placement evidence.

    Coordinates are world-frame metric coordinates.  Image-space quantities
    may be retained in ``diagnostics`` for inspection but are never the formal
    placement residual.
    """

    relation: PlacementRelation = PlacementRelation.UNKNOWN
    frame_id: Optional[int] = None
    instance_id: Optional[str] = None
    grasp_epoch: Optional[int] = None
    opening_plane_origin_world: Optional[tuple[float, ...]] = None
    opening_plane_normal_world: Optional[tuple[float, ...]] = None
    opening_free_space_polygon_world: tuple[tuple[float, float], ...] = ()
    object_footprint_world: tuple[tuple[float, float], ...] = ()
    object_to_gripper_xyz: Optional[tuple[float, float, float]] = None
    selected_candidate: Optional[PlacementCandidate] = None
    eef_residual_world: Optional[tuple[float, float, float]] = None
    rim_clearance_m: Optional[float] = None
    containment_margin_m: Optional[float] = None
    uncertainty_m: Optional[tuple[float, float, float]] = None
    evidence_sources: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    fresh: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskIntent:
    target_descriptor: str
    destination_descriptor: str
    requested_options: tuple[OptionName, ...] = ()


@dataclass
class EntityTrack:
    instance_id: str
    semantic_label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    source: str
    last_confirmed_frame: int
    camera: str = "unknown"
    appearance: tuple[float, ...] = ()
    previous_bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    association_confidence: float = 1.0
    missed_frames: int = 0


@dataclass
class FailureEvent:
    code: FailureCode
    option: OptionName
    frame_id: Optional[int]
    reason: str
    action_history: tuple[str, ...] = ()


@dataclass
class RecoveryContext:
    failure: FailureEvent
    target_instance_id: Optional[str]
    resume_option: OptionName
    attempt: int = 1
    started_frame: Optional[int] = None


@dataclass
class PhysicalBeliefState:
    frame_id: Optional[int] = None
    observation_health: ObservationHealth = ObservationHealth.STALE
    target: Optional[EntityTrack] = None
    destination: Optional[EntityTrack] = None
    eef_xyz: EvidenceValue[tuple[float, float, float]] = field(
        default_factory=lambda: EvidenceValue(
            None, TruthValue.UNKNOWN, "unobserved", 0.0, None
        )
    )
    held: EvidenceValue[bool] = field(
        default_factory=lambda: EvidenceValue(
            None, TruthValue.UNKNOWN, "unobserved", 0.0, None
        )
    )
    seated: EvidenceValue[bool] = field(
        default_factory=lambda: EvidenceValue(
            None, TruthValue.UNKNOWN, "unobserved", 0.0, None
        )
    )
    alignment_residual: EvidenceValue[tuple[float, float]] = field(
        default_factory=lambda: EvidenceValue(
            None, TruthValue.UNKNOWN, "unobserved", 0.0, None
        )
    )
    world_xy_residual_m: EvidenceValue[tuple[float, float]] = field(
        default_factory=lambda: EvidenceValue(
            None, TruthValue.UNKNOWN, "unobserved", 0.0, None
        )
    )
    spatial: Optional[SpatialBelief] = None
    current_option: OptionName = OptionName.LOCATE_TARGET
    grasp_epoch: int = 0
    route_epoch: int = 0
    recovery: Optional[RecoveryContext] = None


@dataclass(frozen=True)
class OptionSpec:
    name: OptionName
    budget: int
    allowed_actions: tuple[str, ...]
    success_condition: str
    recovery_option: OptionName = OptionName.RELOCALIZE


@dataclass(frozen=True)
class TransitionVerdict:
    verdict: Verdict
    reason: str
    evidence: dict[str, Any]
    frame_id: Optional[int]
    fresh: bool


@dataclass(frozen=True)
class OptionResult:
    option: OptionName
    status: OptionStatus
    action_token: Optional[str]
    reason: str
    transition: TransitionVerdict
    failure: Optional[FailureEvent] = None


@dataclass(frozen=True)
class CriticalDecisionRequest:
    kind: str
    candidate_ids: tuple[str, ...]
    allowed_answers: tuple[str, ...]
    belief: dict[str, Any]
    reason: str
    candidates: tuple[dict[str, Any], ...] = ()
    camera: str = "agentview"
    spatial_belief: Optional[dict[str, Any]] = None
    visual_alignment: Optional[dict[str, Any]] = None
    visual_memory_refs: tuple[str, ...] = ()
    visual_memory_bundle: tuple[dict[str, Any], ...] = ()
    reflection_trigger: Optional[str] = None
    reflection_mode: str = "off"
    allowed_options: tuple[str, ...] = ()
    evidence_frame_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class EvidenceCitation:
    frame_id: int
    camera: str
    observation: str


@dataclass(frozen=True)
class AgentDecision:
    """Semantic decision payload; its selected option remains runtime-gated."""

    selected: str
    state_hypothesis: str
    evidence_for: tuple[EvidenceCitation, ...] = ()
    evidence_against: tuple[EvidenceCitation, ...] = ()
    missing_observation: str = ""
    expected_effect: str = ""
    failure_condition: str = ""
    summary: str = ""


def jsonable(value: Any) -> Any:
    """Convert runtime dataclasses/enums into the existing JSON log contract."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
