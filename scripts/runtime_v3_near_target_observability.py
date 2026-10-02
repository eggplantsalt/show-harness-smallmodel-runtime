#!/usr/bin/env python3
"""M3.4 diagnostic study of image alignment, wrist visibility and 3D proximity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.object_relative import (
    DIRECTION_ORDER, MultiScaleAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver, TargetSegmentation,
    associate_target_candidate, compare_alignment_improvements,
    make_target_identity_anchor, segmentation_from_response,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.scene_initialization import run_scene_ready_holds
from core.runtime_v3.selector import DeterministicSelector, Selection
from core.runtime_v3.state import BeliefState, StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from scripts.runtime_v3_multiscale_alignment import (
    CONTROL_TICK_MM, INIT_STATES, PRE_SETTLE_TICKS, SCALES_M, SUITE, TASK_ID,
    TARGET_PHRASE, TICK_BUDGETS, candidate_lattice_records, error_from_state,
    same_fixed_reference,
)

MAX_ALIGNMENT_STEPS = 12
WORKSPACE_Z_M = (0.02, 0.60)
TARGET_BODY = "salad_dressing_1_main"
CONTROL_TICK_M = CONTROL_TICK_MM / 1000.0
ORIENTATION_CHOICES = ("identity", "vertical_flip", "horizontal_flip", "rotate_180")
SEMANTIC_OPTION_ID = "ALIGN_TO_TARGET_BOUNDED"
WORKSPACE_BOUNDARY = "WORKSPACE_BOUNDARY"


@dataclass(frozen=True)
class NearTargetObservation:
    """One deployable visual/proprioceptive sample plus isolated oracle diagnostics."""

    step: int
    agentview_error_px: float | None
    agentview_normalized_error: float | None
    agentview_mask_area_ratio: float | None
    agentview_bbox_width_ratio: float | None
    agentview_bbox_height_ratio: float | None
    wrist_visible: bool
    wrist_candidate_count: int
    wrist_association_status: str
    wrist_candidate_id: str | None
    wrist_centroid_px: tuple[float, float] | None
    wrist_center_error_normalized: float | None
    wrist_mask_area_ratio: float | None
    wrist_bbox_width_ratio: float | None
    wrist_bbox_height_ratio: float | None
    wrist_bbox_area_ratio: float | None
    eef_position_xyz_m: tuple[float, float, float] | None
    gripper_state: str | None
    gripper_width_m: float | None
    oracle_target_position_xyz_m: tuple[float, float, float] | None
    oracle_eef_target_distance_m: float | None
    oracle_contact: bool | None
    selected_direction: str | None = None
    selected_scale_mm: float | None = None


class DiagnosticSelector(DeterministicSelector):
    def select(self, state, options):
        if any(option.option_id == SEMANTIC_OPTION_ID for option in options):
            return Selection(SEMANTIC_OPTION_ID, parsed_selection=SEMANTIC_OPTION_ID)
        return Selection("ABORT", status="ABORT", raw_output="no positive verified alignment option")


class AuditArbiter(Arbiter):
    """Retain one existing Runtime approval per semantic bounded action."""

    def __init__(self, before_authorize):
        super().__init__()
        self.before_authorize = before_authorize
        self.authorization_calls = 0
        self.alignment_authorization_calls = 0
        self.approval_count = 0

    def authorize(self, state, options, selection):
        self.authorization_calls += 1
        if selection.option_id == SEMANTIC_OPTION_ID:
            self.alignment_authorization_calls += 1
            self.before_authorize(state, options, selection)
        decision = super().authorize(state, options, selection)
        if (decision.kind == DecisionKind.APPROVED and decision.action is not None
                and decision.action.option_id == SEMANTIC_OPTION_ID):
            self.approval_count += 1
        return decision


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(record), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_directory(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def canonicalize_wrist(raw_rgb: np.ndarray, orientation: str) -> np.ndarray:
    """Apply only the selected pixel transform; never resize the source render."""
    raw = np.asarray(raw_rgb)
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise ValueError("wrist source must be genuine uint8 HxWx3 RGB")
    return CanonicalImageAdapter(orientation).transform_image(raw)


def extract_wrist_metrics(segmentation: TargetSegmentation,
                          image_shape: Sequence[int]) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    diagonal = math.hypot(width, height)
    bbox = segmentation.bbox_xyxy
    centroid = segmentation.centroid_px
    bbox_width = max(0, int(bbox[2]) - int(bbox[0])) if bbox is not None else None
    bbox_height = max(0, int(bbox[3]) - int(bbox[1])) if bbox is not None else None
    center_error = None
    if segmentation.visible and centroid is not None and diagonal > 0:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        center_error = float(np.linalg.norm(np.asarray(centroid) - np.asarray(center)) / diagonal)
    return {
        "target_visible": bool(segmentation.visible),
        "candidate_count": len(segmentation.candidates),
        "association_status": segmentation.identity_status,
        "candidate_id": segmentation.selected_candidate_id,
        "centroid_px": list(centroid) if centroid is not None else None,
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "mask_area_px": int(segmentation.area_px) if segmentation.area_px is not None else None,
        "mask_area_ratio": (float(segmentation.area_px) / (width * height)
                            if segmentation.area_px is not None and width * height else None),
        "bbox_width_ratio": float(bbox_width / width) if bbox_width is not None and width else None,
        "bbox_height_ratio": float(bbox_height / height) if bbox_height is not None and height else None,
        "bbox_area_ratio": (float(bbox_width * bbox_height / (width * height))
                            if bbox_width is not None and bbox_height is not None and width * height else None),
        "center_error_normalized": center_error,
        "image_center_px": [(width - 1) / 2.0, (height - 1) / 2.0],
    }


def oracle_distance(eef_xyz_m: Sequence[float] | None,
                    target_xyz_m: Sequence[float] | None) -> float | None:
    """Diagnostic Euclidean distance between EEF proprioception and target body origin."""
    if eef_xyz_m is None or target_xyz_m is None:
        return None
    try:
        eef = np.asarray(eef_xyz_m, dtype=float).reshape(3)
        target = np.asarray(target_xyz_m, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(eef)) or not np.all(np.isfinite(target)):
        return None
    return float(np.linalg.norm(eef - target))


def workspace_stop_reason(state: BeliefState) -> str | None:
    if not bool(state.relevant_geometry.get("workspace_valid")):
        return WORKSPACE_BOUNDARY
    raw = (state.end_effector_state or {}).get("position_xyz")
    try:
        z = float(raw[2])
    except (TypeError, ValueError, IndexError):
        return WORKSPACE_BOUNDARY
    if z <= WORKSPACE_Z_M[0] or z >= WORKSPACE_Z_M[1]:
        return WORKSPACE_BOUNDARY
    return None


def _rankdata(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation_pair(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson": None, "spearman": None}
    left, right = (np.asarray(values, dtype=float) for values in zip(*pairs))
    pearson = (float(np.corrcoef(left, right)[0, 1])
               if np.std(left) > 0 and np.std(right) > 0 else None)
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    spearman = (float(np.corrcoef(left_rank, right_rank)[0, 1])
                if np.std(left_rank) > 0 and np.std(right_rank) > 0 else None)
    return {"n": len(pairs), "pearson": pearson, "spearman": spearman}


def _name(model: Any, kind: str, index: int) -> str | None:
    lookup = getattr(model, f"{kind}_id2name", None)
    if callable(lookup):
        try:
            return str(lookup(int(index)))
        except Exception:
            return None
    try:
        return str(getattr(model, kind)(int(index)).name)
    except Exception:
        return None


def _descendants(model: Any, root_id: int) -> set[int]:
    parents = np.asarray(model.body_parentid, dtype=int)
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parents):
            if int(parent_id) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _is_robot_body(model: Any, body_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        name = (_name(model, "body", current) or "").casefold()
        if name.startswith(("robot", "gripper", "panda", "franka")):
            return True
        current = int(model.body_parentid[current])
    return False


def target_pose_diagnostic(environment: Any) -> dict[str, Any]:
    """Return target body origin strictly as experiment ground truth."""
    try:
        model, data = environment.env.sim.model, environment.env.sim.data
        body_id = int(model.body_name2id(TARGET_BODY))
        xyz = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("nonfinite target body pose")
        return {"available": True, "body": TARGET_BODY, "body_id": body_id,
                "world_position_xyz_m": xyz.tolist(), "source": "simulator_diagnostic_only",
                "quantity": "body-frame origin; not necessarily grasp point"}
    except Exception as exc:
        return {"available": False, "body": TARGET_BODY,
                "error": f"{type(exc).__name__}: {exc}", "source": "simulator_diagnostic_only"}


def robot_target_contact_diagnostic(environment: Any) -> dict[str, Any]:
    """Inspect MuJoCo contacts for a robot body against the target or descendants."""
    try:
        model, data = environment.env.sim.model, environment.env.sim.data
        target_id = int(model.body_name2id(TARGET_BODY))
        target_ids = _descendants(model, target_id)
        contacts = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            body_a = int(model.geom_bodyid[int(contact.geom1)])
            body_b = int(model.geom_bodyid[int(contact.geom2)])
            touching_target = ((body_a in target_ids and _is_robot_body(model, body_b))
                               or (body_b in target_ids and _is_robot_body(model, body_a)))
            if touching_target:
                contacts.append({"body_names": [_name(model, "body", body_a),
                                                 _name(model, "body", body_b)],
                                 "distance_m": float(contact.dist)})
        return {"available": True, "robot_target_contact": bool(contacts),
                "contacts": contacts, "diagnostic_only": True}
    except Exception as exc:
        return {"available": False, "robot_target_contact": False,
                "error": f"{type(exc).__name__}: {exc}", "diagnostic_only": True}


def _setup_episode(*, init_state: int, config: Mapping[str, Any], sam3: Any,
                   camera_resolution: int, workspace: tuple[float, float],
                   diagnostic_camera_depths: bool = False,
                   metric_depth_provider: Any | None = None):
    from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
    from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
    from core.sim.libero_task import make_libero_task
    from interpreters.libero_atomic_controller import LiberoAtomicController

    if diagnostic_camera_depths:
        # Privileged depth is enabled only by this experiment-script path.
        handle = make_libero_task(
            suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
            camera_height=camera_resolution, camera_width=camera_resolution,
            diagnostic_camera_depths=True, horizon=180, settle_steps=0,
        )
        environment = LiberoEnvironmentAdapter(handle)
    else:
        environment = LiberoEnvironmentAdapter.create(
            suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
            camera_height=camera_resolution, camera_width=camera_resolution, horizon=180,
        )
    if TARGET_PHRASE.casefold() not in environment.task_description.casefold():
        environment.close()
        raise RuntimeError("target phrase is absent from the task instruction")
    controller = LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=CONTROL_TICK_M,
        sim_steps_per_decision=1,
        position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    base_observer = LiberoObservationAdapter(
        max_eef_z_m=workspace[1], min_eef_z_m=workspace[0], safe_lift_step_m=CONTROL_TICK_M,
    )
    ready_holds = []
    reset = True
    for index in range(PRE_SETTLE_TICKS):
        result = run_v3_tick(
            environment, base_observer, controller, task_id=f"{SUITE}:{TASK_ID}",
            token=None, direction_unit=None, commanded_step_m=CONTROL_TICK_M,
            reset=reset, workspace_z_bounds_m=workspace,
        )
        reset = False
        if result.get("actions") != 1 or not result.get("backend_execution"):
            environment.close()
            raise RuntimeError(f"RobotReady hold {index + 1} failed: {result}")
        ready_holds.append(result)
    observer = ObjectRelativePerceptionObserver(
        base_observer, sam3, target_phrase=TARGET_PHRASE,
        move_vectors=config["move_vectors"],
        canonical_image_adapter=CanonicalImageAdapter("vertical_flip"),
        scene_ready_required=True, alignment_scales_m=SCALES_M,
        scale_contracts={scale: {"verified": True, "max_ticks": TICK_BUDGETS[scale]}
                         for scale in SCALES_M},
        metric_depth_provider=metric_depth_provider,
    )
    scene = run_scene_ready_holds(
        environment, observer, controller, task_id=f"{SUITE}:{TASK_ID}",
        commanded_step_m=CONTROL_TICK_M, workspace_z_bounds_m=workspace,
        max_hold_ticks=40,
    )
    if not scene.get("ready") or not observer.scene_ready:
        environment.close()
        raise RuntimeError(f"SceneReady failed: {scene}")
    return environment, controller, base_observer, observer, ready_holds, scene


def _selected_wrist_segmentation(response: Mapping[str, Any], shape: tuple[int, int],
                                 previous_anchor: Any) -> tuple[TargetSegmentation, Any]:
    current = segmentation_from_response(response, shape)
    if previous_anchor is None:
        current = replace(current, identity_status="ANCHORED") if current.visible else current
    else:
        association = associate_target_candidate(previous_anchor, current.candidates)
        candidate = association.candidate
        if candidate is None:
            current = TargetSegmentation(
                visible=False, mask=None, centroid_px=None, bbox_xyxy=None, area_px=None,
                quality_score=None, response=current.response, candidates=current.candidates,
                selected_candidate_id=None, identity_status="TARGET_IDENTITY_LOST",
                association_metrics={"view": "wrist", "candidates": list(association.candidate_metrics)},
            )
        else:
            current = TargetSegmentation(
                visible=True, mask=candidate.mask, centroid_px=candidate.centroid_px,
                bbox_xyxy=candidate.bbox_xyxy, area_px=candidate.area_px,
                quality_score=candidate.score, response=current.response,
                candidates=current.candidates, selected_candidate_id=candidate.candidate_id,
                identity_status="SAME_TARGET",
                association_metrics={"view": "wrist", "candidates": list(association.candidate_metrics)},
            )
    anchor = make_target_identity_anchor(current, target_phrase=TARGET_PHRASE,
                                         frame_id=None) if current.visible else previous_anchor
    return current, anchor


def _mask_overlay(image: np.ndarray, segmentation: TargetSegmentation,
                  *, center_error: float | None = None) -> np.ndarray:
    canvas = Image.fromarray(np.ascontiguousarray(image), mode="RGB").convert("RGBA")
    if segmentation.visible and segmentation.mask is not None:
        mask = np.asarray(segmentation.mask, dtype=bool)
        tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
        tint[mask] = (255, 40, 35, 92)
        canvas = Image.alpha_composite(canvas, Image.fromarray(tint, mode="RGBA"))
    draw = ImageDraw.Draw(canvas)
    height, width = np.asarray(image).shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    draw.ellipse((center[0]-5, center[1]-5, center[0]+5, center[1]+5), outline=(0, 255, 255, 255), width=2)
    if segmentation.bbox_xyxy is not None:
        draw.rectangle(segmentation.bbox_xyxy, outline=(255, 240, 0, 255), width=3)
    if segmentation.centroid_px is not None:
        x, y = segmentation.centroid_px
        draw.ellipse((x-5, y-5, x+5, y+5), outline=(30, 255, 70, 255), width=3)
        draw.line((center[0], center[1], x, y), fill=(30, 255, 70, 255), width=2)
    label = "visible" if segmentation.visible else "not visible"
    if center_error is not None:
        label += f" center={center_error:.4f} diag"
    draw.rectangle((4, 4, min(width-5, 250), 24), fill=(0, 0, 0, 180))
    draw.text((8, 7), f"{label}; cyan=image center", fill=(255, 255, 255, 255))
    return np.asarray(canvas.convert("RGB"))


def _save_view_artifacts(*, frame: Mapping[str, Any], raw_wrist: np.ndarray | None,
                         wrist_canonical: np.ndarray | None,
                         wrist_segmentation: TargetSegmentation | None,
                         wrist_metrics: Mapping[str, Any], directory: Path,
                         prefix: str) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | None] = {}
    agent_raw = np.ascontiguousarray(frame["raw_image"], dtype=np.uint8)
    agent_canonical = np.ascontiguousarray(frame["image"], dtype=np.uint8)
    agent_seg = frame["segmentation"]
    raw_path = directory / f"{prefix}_agentview_raw.png"
    Image.fromarray(agent_raw, mode="RGB").save(raw_path)
    paths["agentview_raw"] = str(raw_path)
    canonical_path = directory / f"{prefix}_agentview_canonical.png"
    Image.fromarray(agent_canonical, mode="RGB").save(canonical_path)
    paths["agentview_canonical"] = str(canonical_path)
    agent_mask = (np.asarray(agent_seg.mask, dtype=bool) if agent_seg.mask is not None
                  else np.zeros(agent_canonical.shape[:2], dtype=bool))
    mask_path = directory / f"{prefix}_agentview_canonical_mask.png"
    Image.fromarray(agent_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    paths["agentview_canonical_mask"] = str(mask_path)
    overlay_path = directory / f"{prefix}_agentview_canonical_overlay.png"
    Image.fromarray(_mask_overlay(agent_canonical, agent_seg), mode="RGB").save(overlay_path)
    paths["agentview_canonical_overlay"] = str(overlay_path)
    if raw_wrist is None or wrist_canonical is None or wrist_segmentation is None:
        paths.update({"wrist_raw": None, "wrist_canonical": None,
                      "wrist_mask": None, "wrist_overlay": None})
        return paths
    raw_path = directory / f"{prefix}_wrist_raw.png"
    canonical_path = directory / f"{prefix}_wrist_canonical.png"
    Image.fromarray(np.ascontiguousarray(raw_wrist), mode="RGB").save(raw_path)
    Image.fromarray(np.ascontiguousarray(wrist_canonical), mode="RGB").save(canonical_path)
    paths.update({"wrist_raw": str(raw_path), "wrist_canonical": str(canonical_path)})
    wrist_mask = (np.asarray(wrist_segmentation.mask, dtype=bool)
                  if wrist_segmentation.mask is not None
                  else np.zeros(wrist_canonical.shape[:2], dtype=bool))
    mask_path = directory / f"{prefix}_wrist_mask.png"
    Image.fromarray(wrist_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    paths["wrist_mask"] = str(mask_path)
    overlay_path = directory / f"{prefix}_wrist_overlay.png"
    Image.fromarray(_mask_overlay(wrist_canonical, wrist_segmentation,
                                  center_error=wrist_metrics.get("center_error_normalized")),
                    mode="RGB").save(overlay_path)
    paths["wrist_overlay"] = str(overlay_path)
    return paths


def _robot_state(state: BeliefState) -> dict[str, Any]:
    eef = state.end_effector_state or {}
    position = eef.get("position_xyz")
    try:
        position = tuple(float(value) for value in position)
    except (TypeError, ValueError):
        position = None
    if position is not None and len(position) != 3:
        position = None
    return {"eef_position_xyz_m": position,
            "eef_quaternion": eef.get("quaternion"),
            "gripper_state": state.gripper_state,
            "gripper_width_m": state.gripper_width_m}


def _observation_record(*, step: int, state: BeliefState,
                        frame: Mapping[str, Any], raw_wrist: np.ndarray | None,
                        sam3: Any, wrist_orientation: str, previous_wrist_anchor: Any,
                        environment: Any, selected_direction: str | None = None,
                        selected_scale_mm: float | None = None):
    wrist_canonical = (canonicalize_wrist(raw_wrist, wrist_orientation)
                       if raw_wrist is not None else None)
    response = (sam3.segment(wrist_canonical, TARGET_PHRASE, confidence_threshold=0.05)
                if wrist_canonical is not None else {"success": False, "error": "wrist_missing"})
    segmentation, next_anchor = (
        _selected_wrist_segmentation(response, wrist_canonical.shape[:2], previous_wrist_anchor)
        if wrist_canonical is not None else
        (TargetSegmentation(False, None, None, None, None, None, response), previous_wrist_anchor)
    )
    wrist_metrics = extract_wrist_metrics(segmentation, wrist_canonical.shape[:2]) if wrist_canonical is not None else {
        "target_visible": False, "candidate_count": 0, "association_status": "NO_WRIST_IMAGE",
        "candidate_id": None, "centroid_px": None, "bbox_xyxy": None,
        "mask_area_px": None, "mask_area_ratio": None, "bbox_width_ratio": None,
        "bbox_height_ratio": None, "bbox_area_ratio": None, "center_error_normalized": None,
        "image_center_px": None,
    }
    agent_seg = frame["segmentation"]
    agent_shape = frame["image"].shape[:2]
    agent_metrics = extract_wrist_metrics(agent_seg, agent_shape)
    relative = state.object_relative_state
    error = error_from_state(state)
    diagonal = math.hypot(int(agent_shape[1]), int(agent_shape[0]))
    # EEF values come from the canonical Runtime observation; simulator pose is diagnostic only.
    robot = _robot_state(state)
    target = target_pose_diagnostic(environment)
    target_xyz = target.get("world_position_xyz_m") if target.get("available") else None
    distance = oracle_distance(robot["eef_position_xyz_m"], target_xyz)
    contact = robot_target_contact_diagnostic(environment)
    record = NearTargetObservation(
        step=int(step),
        agentview_error_px=error,
        agentview_normalized_error=(error / diagonal if error is not None and diagonal else None),
        agentview_mask_area_ratio=agent_metrics["mask_area_ratio"],
        agentview_bbox_width_ratio=agent_metrics["bbox_width_ratio"],
        agentview_bbox_height_ratio=agent_metrics["bbox_height_ratio"],
        wrist_visible=bool(wrist_metrics["target_visible"]),
        wrist_candidate_count=int(wrist_metrics["candidate_count"]),
        wrist_association_status=str(wrist_metrics["association_status"]),
        wrist_candidate_id=wrist_metrics["candidate_id"],
        wrist_centroid_px=(tuple(wrist_metrics["centroid_px"])
                           if wrist_metrics["centroid_px"] is not None else None),
        wrist_center_error_normalized=wrist_metrics["center_error_normalized"],
        wrist_mask_area_ratio=wrist_metrics["mask_area_ratio"],
        wrist_bbox_width_ratio=wrist_metrics["bbox_width_ratio"],
        wrist_bbox_height_ratio=wrist_metrics["bbox_height_ratio"],
        wrist_bbox_area_ratio=wrist_metrics["bbox_area_ratio"],
        eef_position_xyz_m=(tuple(robot["eef_position_xyz_m"])
                            if robot["eef_position_xyz_m"] is not None else None),
        gripper_state=robot["gripper_state"], gripper_width_m=robot["gripper_width_m"],
        oracle_target_position_xyz_m=(tuple(target_xyz) if target_xyz is not None else None),
        oracle_eef_target_distance_m=distance,
        oracle_contact=(bool(contact["robot_target_contact"]) if contact.get("available") else None),
        selected_direction=selected_direction, selected_scale_mm=selected_scale_mm,
    )
    record_data = jsonable(record)
    record_data["target_body_diagnostic"] = target
    record_data["contact_diagnostic"] = contact
    record_data["wrist_raw_resolution"] = ([int(raw_wrist.shape[1]), int(raw_wrist.shape[0])]
                                             if raw_wrist is not None else None)
    record_data["wrist_canonical_resolution"] = ([int(wrist_canonical.shape[1]), int(wrist_canonical.shape[0])]
                                                   if wrist_canonical is not None else None)
    record_data["wrist_sam_input_resolution"] = record_data["wrist_canonical_resolution"]
    record_data["wrist_segmentation_response"] = response
    record_data["wrist_metrics"] = wrist_metrics
    record_data["agentview_metrics"] = agent_metrics
    record_data["agentview_raw_resolution"] = [int(frame["raw_image"].shape[1]), int(frame["raw_image"].shape[0])]
    record_data["eef_quaternion"] = robot["eef_quaternion"]
    record_data["target_identity_agentview"] = (relative.target_identity_status if relative else None)
    record_data["target_reference_px"] = (relative.target_reference_point_px if relative else None)
    record_data["target_reference_valid"] = bool(relative and relative.target_reference_valid)
    return record, record_data, segmentation, wrist_canonical, next_anchor


def save_orientation_audit(raw_agentview: np.ndarray, raw_wrist: np.ndarray,
                            output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_canonical = CanonicalImageAdapter("vertical_flip").transform_image(raw_agentview)
    images = {orientation: CanonicalImageAdapter(orientation).transform_image(raw_wrist)
              for orientation in ORIENTATION_CHOICES}
    paths = {}
    tile_w, tile_h = 280, 330
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 2), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    tiles = [("agentview canonical: vertical_flip", "agentview_vertical_flip", agent_canonical)] + [
        (f"wrist candidate: {name}", f"wrist_{name}", image)
        for name, image in images.items()
    ]
    for index, (label, filename, array) in enumerate(tiles):
        path = output_dir / f"orientation_{filename}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths[label] = str(path)
        thumb = Image.fromarray(array, mode="RGB")
        thumb.thumbnail((tile_w - 10, tile_h - 36))
        x0, y0 = (index % 3) * tile_w, (index // 3) * tile_h
        sheet.paste(thumb, (x0 + (tile_w-thumb.width)//2, y0 + 28))
        draw.text((x0 + 6, y0 + 6), label, fill=(255, 255, 255))
    sheet_path = output_dir / "wrist_orientation_diagnostic.png"
    sheet.save(sheet_path)
    raw_path = output_dir / "wrist_raw_rgb.png"
    Image.fromarray(raw_wrist, mode="RGB").save(raw_path)
    return {
        "agentview_raw_resolution": [int(raw_agentview.shape[1]), int(raw_agentview.shape[0])],
        "agentview_canonical_transform": "vertical_flip",
        "agentview_canonical_resolution": [int(agent_canonical.shape[1]), int(agent_canonical.shape[0])],
        "wrist_raw_resolution": [int(raw_wrist.shape[1]), int(raw_wrist.shape[0])],
        "wrist_candidate_transforms": list(ORIENTATION_CHOICES),
        "wrist_transform_selected_after_visual_audit": None,
        "candidate_image_paths": paths, "wrist_raw_path": str(raw_path),
        "diagnostic_montage": str(sheet_path),
        "sam_input_is_source_resolution_without_resize": True,
    }


def run_orientation_audit(*, init_state: int, config: Mapping[str, Any], sam3: Any,
                          camera_resolution: int, output_dir: Path,
                          selected_orientation: str | None = None) -> dict[str, Any]:
    environment = None
    try:
        environment, _controller, base, _observer, holds, scene = _setup_episode(
            init_state=init_state, config=config, sam3=sam3,
            camera_resolution=camera_resolution, workspace=WORKSPACE_Z_M,
        )
        raw = base.last_raw
        if raw is None or raw.wrist_rgb is None:
            raise RuntimeError("LIBERO observation did not contain a wrist camera render")
        result = save_orientation_audit(raw.agentview_rgb, raw.wrist_rgb, output_dir)
        # Compare SAM response geometry under each orientation for audit only.
        response_audit = {}
        for orientation in ORIENTATION_CHOICES:
            candidate_image = canonicalize_wrist(raw.wrist_rgb, orientation)
            response = sam3.segment(candidate_image, TARGET_PHRASE, confidence_threshold=0.05)
            seg = segmentation_from_response(response, candidate_image.shape[:2])
            response_audit[orientation] = {
                "visible": bool(seg.visible), "candidate_count": len(seg.candidates),
                "centroid_px": list(seg.centroid_px) if seg.centroid_px else None,
                "bbox_xyxy": list(seg.bbox_xyxy) if seg.bbox_xyxy else None,
                "mask_area_px": seg.area_px,
                "mask_area_ratio": (float(seg.area_px)/(candidate_image.shape[0]*candidate_image.shape[1])
                                    if seg.area_px is not None else None),
            }
        result.update({"init_state_index": init_state, "robot_ready_hold_ticks": len(holds),
                       "scene_ready": scene, "sam_orientation_comparison": response_audit,
                       "actions_executed": 0, "oracle_used_for_runtime": False,
                       "wrist_transform_selected_after_visual_audit": selected_orientation,
                       "orientation_selection_basis": (
                           "The direct LIBERO agentview and wrist renders share the same raw OpenGL "
                           "framebuffer row convention; vertical_flip maps both into the established "
                           "top-left canonical pixel convention. The wrist camera keeps its own "
                           "explicit transform setting."
                           if selected_orientation == "vertical_flip" else None
                       )})
        _write_json(output_dir / "orientation_audit.json", result)
        return result
    finally:
        if environment is not None:
            environment.close()


def _save_contact_sheet(images: Sequence[tuple[str, Path]], output_path: Path) -> str | None:
    if not images:
        return None
    thumb_w, thumb_h, columns = 300, 330, 2
    sheet = Image.new("RGB", (thumb_w*columns, thumb_h*math.ceil(len(images)/columns)), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(images):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w-8, thumb_h-30))
            x, y = (index % columns)*thumb_w, (index // columns)*thumb_h
            sheet.paste(image, (x+(thumb_w-image.width)//2, y+24))
            draw.text((x+6, y+5), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _stop_reason(state: BeliefState) -> str | None:
    boundary = workspace_stop_reason(state)
    if boundary:
        return boundary
    relative = state.object_relative_state
    if relative is None or not relative.target_reference_valid:
        return "REFERENCE_INVALID"
    if not relative.target_visible:
        return ("TARGET_IDENTITY_LOST" if relative.target_identity_status == "TARGET_IDENTITY_LOST"
                else "TARGET_VISIBILITY_LOST")
    if relative.target_identity_status not in {"ANCHORED", "SAME_TARGET"}:
        return "TARGET_IDENTITY_LOST"
    lattice = state.relevant_geometry.get("candidate_lattice", [])
    if not any(bool(item.get("valid")) and float(item.get("predicted_improvement_px", 0.0)) > 0
               for item in lattice if isinstance(item, Mapping)):
        return "NO_POSITIVE_OPTION"
    return None


def _post_action_stop(step: int, improvement: float | None, state: BeliefState,
                      execution: Mapping[str, Any] | None) -> str | None:
    reason = _stop_reason(state)
    if reason:
        return reason
    if execution and execution.get("termination") == "BOUNDARY_STOP":
        return WORKSPACE_BOUNDARY
    if improvement is None or improvement <= 0:
        return "EFFECT_NOT_IMPROVED"
    return "MAX_ALIGNMENT_STEPS" if step >= MAX_ALIGNMENT_STEPS else None


def _series_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in (
        "step", "agentview_error_px", "agentview_normalized_error",
        "agentview_mask_area_ratio", "agentview_bbox_width_ratio", "agentview_bbox_height_ratio",
        "oracle_eef_target_distance_m", "wrist_visible", "wrist_candidate_count",
        "wrist_association_status", "wrist_candidate_id", "wrist_centroid_px",
        "wrist_mask_area_px", "wrist_mask_area_ratio", "wrist_bbox_width_ratio",
        "wrist_bbox_height_ratio", "wrist_bbox_area_ratio",
        "oracle_target_position_xyz_m",
        "wrist_center_error_normalized", "selected_direction", "selected_scale_mm",
        "eef_position_xyz_m", "eef_quaternion", "gripper_state", "gripper_width_m",
        "oracle_contact",
        "runtime_estimated_metric_distance_m",
        "depth_metric_reference_world_m",
        "depth_metric_candidate_reference_world_m",
        "depth_metric_candidate_valid_depth_ratio",
        "depth_metric_reference_valid_depth_ratio",
        "wrist_depth_reference_coverage_status",
    )}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["step"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), ensure_ascii=False)
                             if isinstance(value, (list, tuple, dict)) else value
                             for key, value in row.items()})


def _run_episode(*, init_state: int, run_dir: Path, config: Mapping[str, Any], sam3: Any,
                 camera_resolution: int, wrist_orientation: str,
                 orientation_audit_path: str | None,
                 diagnostic_camera_depths: bool = False,
                 metric_depth_provider: Any | None = None,
                 diagnostic_callback: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    episode_dir = run_dir / f"init_state_{init_state}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = episode_dir / "observations"
    artifacts_dir.mkdir()
    environment = None
    try:
        environment, controller, base, observer, ready_holds, scene = _setup_episode(
            init_state=init_state, config=config, sam3=sam3,
            camera_resolution=camera_resolution, workspace=WORKSPACE_Z_M,
            diagnostic_camera_depths=diagnostic_camera_depths,
            metric_depth_provider=metric_depth_provider,
        )
        # Oracle and wrist signals remain local to this experiment script; they are
        # never attached to Runtime observations, state, options or decisions.
        episode_contact = {
            "detected": False, "semantic_step": None,
            "control_tick_in_episode": None, "control_tick_in_action": None,
            "record": None, "tick_count": 0,
        }
        ticks_this_action = {"count": 0}
        wrist_anchor = {"value": None}
        previous_wrist_visible = {"value": False}
        last_wrist_image = {"path": None}
        first_visible_panels: list[tuple[str, Path]] = []
        records: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = []
        active_step = {"index": 0}
        preaction = {"state": None, "option": None, "record": None,
                     "record_data": None, "frame": None, "wrist_seg": None,
                     "wrist_image": None, "metrics": None}

        def capture(state: BeliefState, frame: Mapping[str, Any], *, step: int,
                    selected_direction: str | None = None,
                    selected_scale_mm: float | None = None,
                    label: str):
            raw = base.last_raw
            raw_wrist = raw.wrist_rgb if raw is not None else None
            record, data, wrist_seg, wrist_image, next_anchor = _observation_record(
                step=step, state=state, frame=frame, raw_wrist=raw_wrist,
                sam3=sam3, wrist_orientation=wrist_orientation,
                previous_wrist_anchor=wrist_anchor["value"], environment=environment,
                selected_direction=selected_direction, selected_scale_mm=selected_scale_mm,
            )
            if diagnostic_callback is not None:
                # The experiment callback receives read-only evidence and may only
                # add report fields. Runtime state, options, and approvals never
                # consume this return value.
                try:
                    extra = diagnostic_callback(
                        frame=frame, raw=raw, state=state, environment=environment,
                        step=step, capture_label=label, record_data=data,
                    )
                    if isinstance(extra, Mapping):
                        data.update(dict(extra))
                except Exception as exc:
                    data["experiment_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
            wrist_anchor["value"] = next_anchor
            metrics = data["wrist_metrics"]
            paths = _save_view_artifacts(
                frame=frame, raw_wrist=raw_wrist, wrist_canonical=wrist_image,
                wrist_segmentation=wrist_seg, wrist_metrics=metrics,
                directory=artifacts_dir / f"step_{step:02d}", prefix=label,
            )
            data["visual_artifacts"] = paths
            records.append(data)
            if metrics["target_visible"] and not previous_wrist_visible["value"]:
                if last_wrist_image["path"]:
                    first_visible_panels.append(("wrist: preceding invisible", Path(last_wrist_image["path"])))
                if paths.get("agentview_canonical_overlay"):
                    first_visible_panels.append(("agentview: visibility onset", Path(paths["agentview_canonical_overlay"])))
                if paths.get("wrist_overlay"):
                    first_visible_panels.append(("wrist: first associated target", Path(paths["wrist_overlay"])))
                _save_contact_sheet(first_visible_panels,
                                    episode_dir / f"wrist_first_visible_step_{step:02d}.png")
            previous_wrist_visible["value"] = bool(metrics["target_visible"])
            last_wrist_image["path"] = paths.get("wrist_overlay")
            return record, data, wrist_seg, wrist_image

        def before_authorize(state, options, selection):
            option = next((item for item in options if item.option_id == selection.option_id), None)
            if option is None or option.primitive.micro_motion_spec is None:
                raise RuntimeError("no Runtime-owned bounded alignment realization")
            relative = state.object_relative_state
            if (relative is None or not relative.target_visible or not relative.target_reference_valid
                    or not observer.scene_ready):
                raise RuntimeError("target reference or SceneReady evidence invalid before authorization")
            reason = workspace_stop_reason(state)
            if reason:
                raise RuntimeError(reason)
            index = active_step["index"]
            frame = observer.perception_history[-1]
            spec = option.primitive.micro_motion_spec
            obs_record, data, wrist_seg, wrist_image = capture(
                state, frame, step=index - 1,
                selected_direction=spec.direction,
                selected_scale_mm=spec.requested_displacement_m * 1000.0,
                label="before",
            )
            preaction.update({"state": state, "option": option, "record": obs_record,
                              "record_data": data, "frame": frame,
                              "wrist_seg": wrist_seg, "wrist_image": wrist_image,
                              "metrics": data["wrist_metrics"]})
            if not trajectory:
                initial_row = _series_row(data)
                initial_row["step"] = 0
                initial_row["selected_direction"] = None
                initial_row["selected_scale_mm"] = None
                trajectory.append(initial_row)
            ready = {
                "status": "PRE_ACTION_READY", "alignment_step": index,
                "frame_id": state.frame_id, "semantic_option_id": SEMANTIC_OPTION_ID,
                "selected_direction": spec.direction,
                "selected_scale_mm": spec.requested_displacement_m * 1000.0,
                "selected_max_ticks": spec.max_ticks,
                "candidate_lattice": candidate_lattice_records(
                    state.relevant_geometry.get("candidate_lattice", []),
                    state.relevant_geometry.get("chosen_lattice_candidate")),
                "target_reference_px": relative.target_reference_point_px,
                "oracle_used_for_runtime": False,
                "wrist_signal_used_for_runtime": False,
                "visual_artifacts": data["visual_artifacts"],
            }
            ready_path = artifacts_dir / f"step_{index:02d}/PRE_ACTION_READY.json"
            _write_json(ready_path, ready)
            if json.loads(ready_path.read_text(encoding="utf-8")).get("status") != "PRE_ACTION_READY":
                raise RuntimeError("PRE_ACTION_READY read-back failed")

        arbiter = AuditArbiter(before_authorize)
        executor = Executor(LiberoPrimitiveBackend(environment, controller, arbiter), arbiter)

        def observe_tick(current_environment):
            obs = base.observe(current_environment)
            episode_contact["tick_count"] += 1
            ticks_this_action["count"] += 1
            return obs

        observer.observe_for_execution_tick = observe_tick
        initial_error = final_error = None
        reference_initial = None
        termination = "MAX_ALIGNMENT_STEPS"
        for index in range(1, MAX_ALIGNMENT_STEPS + 1):
            active_step["index"] = index
            ticks_this_action["count"] = 0
            auth0, align_auth0, approvals0 = (arbiter.authorization_calls,
                                              arbiter.alignment_authorization_calls,
                                              arbiter.approval_count)
            events: list[dict[str, Any]] = []
            runner = RuntimeV3Runner(
                observer=observer, state_builder=StateBuilder(),
                option_generator=MultiScaleAlignmentOptionGenerator(),
                selector=DiagnosticSelector(), arbiter=arbiter, executor=executor,
                effect_observer=EffectObserver(), logger=events.append,
            )
            result = runner.run_episode(environment, task_id=f"{SUITE}:{TASK_ID}",
                                        max_steps=1, reset=False)
            if not events:
                state = result.get("state") or runner.state
                if state is not None and preaction.get("record") is None:
                    frame = observer.perception_history[-1] if observer.perception_history else None
                    if frame is not None:
                        _obs, data, _seg, _wrist = capture(state, frame, step=index-1, label="stop")
                        if not trajectory:
                            initial = _series_row(data)
                            initial["step"] = 0
                            trajectory.append(initial)
                termination = (_stop_reason(state) if state is not None else None) or result.get("status", "EXECUTION_FAILED")
                steps.append({"alignment_step": index, "executed": False,
                              "termination_reason": termination,
                              "candidate_lattice": candidate_lattice_records(
                                  state.relevant_geometry.get("candidate_lattice", []) if state else []),
                              "authorization_calls_for_step": arbiter.authorization_calls-auth0,
                              "alignment_authorization_calls_for_step": arbiter.alignment_authorization_calls-align_auth0,
                              "approvals_for_step": arbiter.approval_count-approvals0})
                break

            event = events[0]
            before, after = event["state_before"], event["state_after"]
            action = event["approved_action"]
            spec = action.primitive.micro_motion_spec
            frame_after = observer.perception_history[-1]
            after_record, after_data, _wrist_seg_after, _wrist_image_after = capture(
                after, frame_after, step=index,
                selected_direction=spec.direction,
                selected_scale_mm=spec.requested_displacement_m*1000.0,
                label="after",
            )
            ref_before = (before.object_relative_state.target_reference_point_px
                          if before.object_relative_state else None)
            ref_after = (after.object_relative_state.target_reference_point_px
                         if after.object_relative_state else None)
            same_ref = same_fixed_reference(ref_before, ref_after)
            error_before, error_after = error_from_state(before), error_from_state(after)
            if initial_error is None:
                initial_error = error_before
                reference_initial = ref_before
            final_error = error_after if error_after is not None else final_error
            image_improvement = (error_before-error_after if error_before is not None
                                 and error_after is not None and same_ref else None)
            before_distance = preaction["record_data"].get("oracle_eef_target_distance_m")
            after_distance = after_data.get("oracle_eef_target_distance_m")
            distance_improvement = (float(before_distance)-float(after_distance)
                                    if before_distance is not None and after_distance is not None else None)
            execution = getattr(event.get("execution"), "result", None)
            contact_after = after_data.get("contact_diagnostic", {}).get("robot_target_contact", False)
            if contact_after:
                if not episode_contact["detected"]:
                    episode_contact.update({
                        "detected": True, "semantic_step": index,
                        "control_tick_in_episode": episode_contact["tick_count"],
                        "control_tick_in_action": ticks_this_action["count"],
                        "record": after_data.get("contact_diagnostic"),
                    })
            record = {
                "alignment_step": index, "executed": True,
                "semantic_option_id": action.option_id,
                "direction": spec.direction, "scale_m": spec.requested_displacement_m,
                "scale_mm": spec.requested_displacement_m*1000.0,
                "max_ticks": spec.max_ticks,
                "image_error_before_px": error_before, "image_error_after_px": error_after,
                "image_error_improvement_px": image_improvement,
                "oracle_distance_before_m": before_distance,
                "oracle_distance_after_m": after_distance,
                "oracle_distance_improvement_m": distance_improvement,
                "oracle_target_position_before_xyz_m": preaction["record_data"].get(
                    "oracle_target_position_xyz_m"),
                "oracle_target_position_after_xyz_m": after_data.get(
                    "oracle_target_position_xyz_m"),
                "wrist_before": preaction["record_data"].get("wrist_metrics"),
                "wrist_after": after_data.get("wrist_metrics"),
                "candidate_lattice": candidate_lattice_records(
                    before.relevant_geometry.get("candidate_lattice", []),
                    before.relevant_geometry.get("chosen_lattice_candidate")),
                "eef_pose_before": _robot_state(before), "eef_pose_after": _robot_state(after),
                "frozen_reference_unchanged": same_ref,
                "reference_matches_episode_initial": same_fixed_reference(reference_initial, ref_before),
                "target_identity_after": (after.object_relative_state.target_identity_status
                                          if after.object_relative_state else None),
                "execution": execution,
                "prediction": compare_alignment_improvements(
                    predicted_improvement_px=action.expected_effect.get("predicted_improvement_px"),
                    actual_improvement_px=image_improvement),
                "authorization_calls_for_step": arbiter.authorization_calls-auth0,
                "alignment_authorization_calls_for_step": arbiter.alignment_authorization_calls-align_auth0,
                "arbiter_approved_actions_for_step": arbiter.approval_count-approvals0,
                "oracle_used_for_runtime": False, "wrist_signal_used_for_runtime": False,
            }
            steps.append(record)
            step_path = artifacts_dir / f"step_{index:02d}/step.json"
            _write_json(step_path, record)
            row = _series_row(after_data)
            row["step"] = index
            row["selected_direction"] = spec.direction
            row["selected_scale_mm"] = spec.requested_displacement_m*1000.0
            trajectory.append(row)
            stop = _post_action_stop(index, image_improvement, after, execution)
            if stop:
                termination = stop
                record["termination_reason"] = stop
                _write_json(step_path, record)
                break
            preaction.update({"record": None, "record_data": None, "state": None,
                              "option": None, "frame": None})

        executed = [row for row in steps if row.get("executed")]
        images = [record for record in records]
        first_visible = next((int(item["step"]) for item in images if item.get("wrist_visible")), None)
        directions = [row["direction"] for row in executed if row.get("direction")]
        scales = [float(row["scale_mm"]) for row in executed if row.get("scale_mm") is not None]
        image_improvements = [float(row["image_error_improvement_px"]) for row in executed
                              if row.get("image_error_improvement_px") is not None]
        distance_improvements = [float(row["oracle_distance_improvement_m"]) for row in executed
                                 if row.get("oracle_distance_improvement_m") is not None]
        total_ticks = sum(int((row.get("execution") or {}).get("ticks_executed", 0)) for row in executed)
        episode = {
            "init_state_index": init_state, "status": "COMPLETED",
            "suite": SUITE, "task_id": TASK_ID, "seed": 0,
            "task_instruction": environment.task_description,
            "robot_ready_hold_ticks": len(ready_holds), "scene_ready": scene,
            "max_alignment_steps": MAX_ALIGNMENT_STEPS,
            "executed_semantic_steps": len(executed), "control_ticks": total_ticks,
            "termination_reason": termination,
            "initial_image_error_px": next((row.get("image_error_before_px") for row in executed),
                                            trajectory[0].get("agentview_error_px") if trajectory else None),
            "final_image_error_px": (final_error if final_error is not None else
                                      trajectory[-1].get("agentview_error_px") if trajectory else None),
            "initial_oracle_distance_m": next((row.get("oracle_distance_before_m") for row in executed),
                                               trajectory[0].get("oracle_eef_target_distance_m") if trajectory else None),
            "final_oracle_distance_m": (trajectory[-1].get("oracle_eef_target_distance_m")
                                        if trajectory else None),
            "trajectory": trajectory,
            "observations": images,
            "steps": steps,
            "direction_sequence": directions, "scale_sequence_mm": scales,
            "wrist_target_visible_observations": sum(bool(row.get("wrist_visible")) for row in images),
            "wrist_observations": len(images), "first_wrist_visible_step": first_visible,
            "contact_detected": bool(episode_contact["detected"]),
            "first_contact_semantic_step": episode_contact["semantic_step"],
            "first_contact_control_tick_in_episode": episode_contact["control_tick_in_episode"],
            "first_contact_control_tick_in_action": episode_contact["control_tick_in_action"],
            "contact_diagnostic": episode_contact["record"],
            "image_positive_steps": sum(value > 0 for value in image_improvements),
            "oracle_distance_positive_steps": sum(value > 0 for value in distance_improvements),
            "both_positive_steps": sum(
                float(row.get("image_error_improvement_px", -1)) > 0
                and float(row.get("oracle_distance_improvement_m", -1)) > 0
                for row in executed if row.get("image_error_improvement_px") is not None
                and row.get("oracle_distance_improvement_m") is not None),
            "arbiter_authorization_calls": arbiter.authorization_calls,
            "arbiter_approved_actions": arbiter.approval_count,
            "one_approval_per_executed_action": arbiter.approval_count == len(executed),
            "qwen_actions": 0, "oracle_used_for_runtime": False,
            "wrist_signal_used_for_runtime": False,
            "orientation_audit_reference": orientation_audit_path,
            "wrist_orientation_transform": wrist_orientation,
            "legacy_modified": False,
        }
        _write_json(episode_dir / "trajectory.json", trajectory)
        _write_csv(episode_dir / "trajectory.csv", trajectory)
        _write_json(episode_dir / "episode.json", episode)
        return episode
    finally:
        if environment is not None:
            environment.close()


def _pearson_spearman(episodes: Sequence[Mapping[str, Any]], x_name: str,
                      y_name: str, *, wrist: bool = False) -> dict[str, Any]:
    x, y = [], []
    for episode in episodes:
        for row in episode.get("trajectory", []):
            if wrist and not row.get("wrist_visible"):
                continue
            left, right = row.get(x_name), row.get(y_name)
            if left is not None and right is not None:
                x.append(float(left))
                y.append(float(right))
    return correlation_pair(x, y)


def _correlation_report(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distance = "oracle_eef_target_distance_m"
    return {
        "oracle_distance_vs_agentview_alignment_error": _pearson_spearman(
            episodes, "agentview_error_px", distance),
        "oracle_distance_vs_wrist_mask_area_ratio": _pearson_spearman(
            episodes, "wrist_mask_area_ratio", distance, wrist=True),
        "oracle_distance_vs_wrist_bbox_area_ratio": _pearson_spearman(
            episodes, "wrist_bbox_area_ratio", distance, wrist=True),
        "oracle_distance_vs_wrist_center_error": _pearson_spearman(
            episodes, "wrist_center_error_normalized", distance, wrist=True),
    }


def _diagnostic_quantile_bins(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for episode in episodes for row in episode.get("trajectory", [])
            if row.get("oracle_eef_target_distance_m") is not None]
    if not rows:
        return {"method": "diagnostic_quantile_bins", "cutpoints_m": None, "bins": {}}
    distances = np.asarray([float(row["oracle_eef_target_distance_m"]) for row in rows])
    low, high = np.quantile(distances, [1/3, 2/3]).tolist()
    groups: dict[str, list[Mapping[str, Any]]] = {"near": [], "medium": [], "far": []}
    for row in rows:
        distance = float(row["oracle_eef_target_distance_m"])
        group = "near" if distance <= low else "medium" if distance <= high else "far"
        groups[group].append(row)
    keys = ("agentview_error_px", "wrist_mask_area_ratio", "wrist_bbox_area_ratio",
            "wrist_center_error_normalized")
    return {"method": "diagnostic_quantile_bins_only_no_runtime_threshold",
            "cutpoints_m": {"near_max": low, "medium_max": high},
            "bins": {name: {"n": len(values), **{
                key: (float(np.mean([float(row[key]) for row in values if row.get(key) is not None]))
                      if any(row.get(key) is not None for row in values) else None)
                for key in keys}} for name, values in groups.items()}}


def _plot_series(episodes: Sequence[Mapping[str, Any]], field: str,
                 output: Path, title: str, y_label: str) -> None:
    width, height, left, top, right, bottom = 900, 500, 72, 42, 25, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (left, top, width-right, height-bottom)
    draw.line((box[0], box[1], box[0], box[3]), fill=(30, 30, 30), width=2)
    draw.line((box[0], box[3], box[2], box[3]), fill=(30, 30, 30), width=2)
    draw.text((left, 15), title, fill=(0, 0, 0))
    data = [[(int(row["step"]), float(row[field])) for row in episode.get("trajectory", [])
             if row.get(field) is not None and math.isfinite(float(row[field]))]
            for episode in episodes]
    values = [v for curve in data for _step, v in curve]
    if not values:
        draw.text((left+12, top+20), "No valid diagnostic observations", fill=(0, 0, 0))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return
    ymin, ymax = min(values), max(values)
    if math.isclose(ymin, ymax):
        pad = max(abs(ymin)*0.05, 0.01)
        ymin, ymax = ymin-pad, ymax+pad
    xmax = max((step for curve in data for step, _value in curve), default=1) or 1
    palette = ((30, 100, 210), (220, 70, 45), (30, 155, 90),
               (170, 70, 180), (235, 145, 20), (30, 160, 170))
    for index, curve in enumerate(data):
        points = [(box[0]+int((box[2]-box[0])*step/xmax),
                   box[3]-int((box[3]-box[1])*(value-ymin)/(ymax-ymin)))
                  for step, value in curve]
        if len(points) > 1:
            draw.line(points, fill=palette[index % len(palette)], width=3)
        for point in points:
            draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3),
                         fill=palette[index % len(palette)])
    draw.text((left, height-32), "semantic step", fill=(0, 0, 0))
    draw.text((5, top), y_label, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _distance_area_scatter(episodes: Sequence[Mapping[str, Any]], output: Path) -> None:
    rows = [row for episode in episodes for row in episode.get("trajectory", [])
            if row.get("oracle_eef_target_distance_m") is not None
            and row.get("wrist_mask_area_ratio") is not None]
    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 16), "Oracle EEF-target body-origin distance vs wrist mask area", fill="black")
    box = (70, 50, width-30, height-60)
    draw.line((box[0], box[1], box[0], box[3]), fill="black", width=2)
    draw.line((box[0], box[3], box[2], box[3]), fill="black", width=2)
    draw.text((box[0], height-34), "oracle EEF-target distance (m)", fill="black")
    draw.text((8, box[1]+4), "wrist mask area ratio", fill="black")
    if rows:
        xs = [float(row["oracle_eef_target_distance_m"]) for row in rows]
        ys = [float(row["wrist_mask_area_ratio"]) for row in rows]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmax <= xmin: xmax = xmin + 1e-9
        if ymax <= ymin: ymax = ymin + 1e-9
        for x, y in zip(xs, ys):
            px = box[0] + int((box[2]-box[0])*(x-xmin)/(xmax-xmin))
            py = box[3] - int((box[3]-box[1])*(y-ymin)/(ymax-ymin))
            draw.ellipse((px-4, py-4, px+4, py+4), fill=(190, 45, 40))
    else:
        draw.text((box[0]+10, box[1]+10), "No observations with visible wrist target", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def summarize(episodes: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    executed = [step for episode in episodes for step in episode.get("steps", [])
                if step.get("executed")]
    image_measured = [row for row in executed if row.get("image_error_improvement_px") is not None]
    distance_measured = [row for row in executed if row.get("oracle_distance_improvement_m") is not None]
    both_measured = [row for row in executed if row.get("image_error_improvement_px") is not None
                     and row.get("oracle_distance_improvement_m") is not None]
    image_positive = [row for row in image_measured if float(row["image_error_improvement_px"]) > 0]
    distance_positive = [row for row in distance_measured
                         if row.get("oracle_distance_improvement_m") is not None
                         and float(row["oracle_distance_improvement_m"]) > 0]
    both_positive = [row for row in both_measured
                     if float(row["image_error_improvement_px"]) > 0
                     and float(row["oracle_distance_improvement_m"]) > 0]
    all_observations = [row for episode in episodes for row in episode.get("observations", [])]
    visible = [row for row in all_observations if row.get("wrist_visible")]
    scales = {"3mm": 0, "6mm": 0, "9mm": 0}
    directions = {direction: 0 for direction in DIRECTION_ORDER}
    per_step_scale: dict[str, dict[str, int]] = {}
    scale_sequences: list[list[float]] = []
    direction_sequences: list[list[str]] = []
    for episode in episodes:
        scale_sequences.append([float(step["scale_mm"]) for step in episode.get("steps", [])
                                if step.get("executed") and step.get("scale_mm") is not None])
        direction_sequences.append([str(step["direction"]) for step in episode.get("steps", [])
                                    if step.get("executed") and step.get("direction")])
        for step in episode.get("steps", []):
            if not step.get("executed") or step.get("scale_mm") is None:
                continue
            scale_label = f"{int(round(float(step['scale_mm'])))}mm"
            if scale_label in scales:
                scales[scale_label] += 1
            direction = str(step.get("direction"))
            if direction in directions:
                directions[direction] += 1
            bucket = per_step_scale.setdefault(str(step["alignment_step"]),
                                               {"3mm": 0, "6mm": 0, "9mm": 0})
            if scale_label in bucket:
                bucket[scale_label] += 1
    _plot_series(episodes, "agentview_error_px", output_dir / "image_error_vs_step.png",
                 "Agentview frozen-reference image error", "pixel error")
    _plot_series(episodes, "oracle_eef_target_distance_m", output_dir / "physical_distance_vs_step.png",
                 "Oracle diagnostic EEF-target body-origin distance", "meters")
    _plot_series(episodes, "wrist_mask_area_ratio", output_dir / "wrist_area_vs_step.png",
                 "Wrist target SAM mask area ratio", "area ratio")
    _distance_area_scatter(episodes, output_dir / "distance_vs_wrist_area.png")
    correlations = _correlation_report(episodes)
    return {
        "episodes": len(episodes),
        "executed_actions": len(executed),
        "image_error_measured_steps": len(image_measured),
        "oracle_distance_measured_steps": len(distance_measured),
        "image_error_positive_steps": len(image_positive),
        "oracle_distance_positive_steps": len(distance_positive),
        "both_positive_steps": len(both_positive),
        "image_error_positive_rate_per_executed": len(image_positive)/len(executed) if executed else None,
        "oracle_distance_positive_rate_per_executed": len(distance_positive)/len(executed) if executed else None,
        "both_positive_rate_per_executed": len(both_positive)/len(executed) if executed else None,
        "wrist_visible_observations": len(visible),
        "total_observations": len(all_observations),
        "wrist_visibility_rate": len(visible)/len(all_observations) if all_observations else None,
        "first_visible_step_by_episode": {str(row["init_state_index"]): row.get("first_wrist_visible_step")
                                           for row in episodes},
        "correlations": correlations,
        "diagnostic_quantile_bins": _diagnostic_quantile_bins(episodes),
        "selected_scale_counts": scales, "selected_direction_counts": directions,
        "scale_by_semantic_step": per_step_scale,
        "scale_change_observed": any(any(a != b for a, b in zip(row, row[1:]))
                                      for row in scale_sequences),
        "direction_change_observed": any(any(a != b for a, b in zip(row, row[1:]))
                                          for row in direction_sequences),
        "coarse_to_fine_observed": any(any(right < left for left, right in zip(row, row[1:]))
                                       for row in scale_sequences),
        "episodes_with_contact": sum(bool(row.get("contact_detected")) for row in episodes),
        "first_contact_semantic_step_by_episode": {
            str(row["init_state_index"]): row.get("first_contact_semantic_step") for row in episodes},
        "first_contact_control_tick_in_episode_by_episode": {
            str(row["init_state_index"]): row.get("first_contact_control_tick_in_episode")
            for row in episodes},
        "total_arbiter_approvals": sum(int(row.get("arbiter_approved_actions", 0)) for row in episodes),
        "one_approval_per_executed_action": all(row.get("one_approval_per_executed_action") for row in episodes),
        "qwen_actions": 0, "oracle_used_for_runtime": False,
        "deployable_signal_candidates": [
            "agentview alignment error", "wrist target mask/bbox area ratio",
            "wrist center error", "wrist target visibility",
        ],
        "plot_paths": {
            "physical_distance_vs_step": str(output_dir / "physical_distance_vs_step.png"),
            "image_error_vs_step": str(output_dir / "image_error_vs_step.png"),
            "wrist_area_vs_step": str(output_dir / "wrist_area_vs_step.png"),
            "distance_vs_wrist_area": str(output_dir / "distance_vs_wrist_area.png"),
        },
    }


def _configure_local_proxy(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    names = ("NO_PROXY", "no_proxy")
    entries = [item.strip() for name in names for item in os.environ.get(name, "").split(",") if item.strip()]
    value = ",".join(dict.fromkeys((*entries, "127.0.0.1", "localhost", "::1")))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_near_target_observability"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--wrist-orientation", choices=ORIENTATION_CHOICES, default=None)
    parser.add_argument("--orientation-audit-only", action="store_true")
    parser.add_argument("--orientation-init-state", type=int, default=0)
    parser.add_argument("--orientation-audit-reference", default=None)
    return parser


def main() -> int:
    from core.capabilities.sam3_client import Sam3Client
    from core.config import load_yaml

    args = _parser().parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("M3.4 requires the established direct 512x512 source render")
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_local_proxy(args.sam3_url)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    run_dir = _run_directory(args.output_dir)
    if args.orientation_audit_only:
        try:
            audit = run_orientation_audit(
                init_state=args.orientation_init_state, config=config, sam3=sam3,
                camera_resolution=args.camera_resolution,
                output_dir=run_dir / "orientation_audit",
                selected_orientation=args.wrist_orientation,
            )
            print(f"Wrist orientation audit saved: {run_dir / 'orientation_audit/orientation_audit.json'}")
            return 0
        finally:
            sam3.close()

    if args.wrist_orientation is None:
        raise SystemExit("choose --wrist-orientation after reviewing the no-action orientation audit")

    episodes = []
    blockers = []
    try:
        for init_state in INIT_STATES:
            try:
                episodes.append(_run_episode(
                    init_state=init_state, run_dir=run_dir, config=config, sam3=sam3,
                    camera_resolution=args.camera_resolution,
                    wrist_orientation=args.wrist_orientation,
                    orientation_audit_path=args.orientation_audit_reference,
                ))
            except Exception as exc:
                blockers.append(f"init_state_{init_state}: {type(exc).__name__}: {exc}")
                failure = {"init_state_index": init_state, "status": "FAILED",
                           "error": f"{type(exc).__name__}: {exc}", "executed_semantic_steps": 0,
                           "oracle_used_for_runtime": False, "qwen_actions": 0}
                episodes.append(failure)
                _write_json(run_dir / f"init_state_{init_state}/failure.json", failure)
    finally:
        sam3.close()
    metrics = summarize(episodes, run_dir)
    raw_resolutions = [row.get("observations", []) for row in episodes]
    first_observation = next((rows[0] for rows in raw_resolutions if rows), {})
    stage = {
        "status": "COMPLETED" if len(episodes) == len(INIT_STATES)
                  and not blockers and all(row.get("status") == "COMPLETED" for row in episodes)
                  else "PARTIAL",
        "phase": "M3.4 Near-target Observability Audit", "branch": "runtime-v3",
        "starting_commit": "fa602dba8e8b58792fb9c2d7d19a735601957a00",
        "suite": SUITE, "task_id": TASK_ID, "seed": 0,
        "init_states": list(INIT_STATES), "max_semantic_alignment_steps": MAX_ALIGNMENT_STEPS,
        "camera_resolution_requested": [512, 512],
        "agentview_canonical_transform": "vertical_flip",
        "wrist_orientation_transform": args.wrist_orientation,
        "wrist_orientation_selection_basis": (
            "Direct agentview and wrist renders share the LIBERO raw OpenGL framebuffer row convention; "
            "vertical_flip maps the wrist source into the established top-left canonical pixel convention."
            if args.wrist_orientation == "vertical_flip" else
            "Independent wrist transform selected after reviewing the orientation audit montage."
        ),
        "wrist_orientation_audit_reference": args.orientation_audit_reference,
        "wrist_raw_resolution": first_observation.get("wrist_raw_resolution"),
        "wrist_canonical_resolution": first_observation.get("wrist_canonical_resolution"),
        "wrist_sam_input_resolution": first_observation.get("wrist_sam_input_resolution"),
        "wrist_source_is_genuine_render_not_upscaled": True,
        "motion_contracts_mm": {
            str(int(scale*1000)): {"max_ticks": TICK_BUDGETS[scale]}
            for scale in SCALES_M
        },
        "episodes": episodes, "metrics": metrics, "blockers": blockers,
        "qwen_actions": 0, "oracle_used_for_runtime": False,
        "oracle_quantity_caveat": "EEF to salad dressing body origin; not a grasp-point distance",
        "legacy_modified": False,
    }
    _write_json(run_dir / "summary.json", stage)
    print(f"M3.4 {stage['status']}: {run_dir / 'summary.json'}")
    return 0 if stage["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
