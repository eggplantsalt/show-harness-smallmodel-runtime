#!/usr/bin/env python3
"""Evaluate six-step closed-loop object-relative alignment on fixed LIBERO states."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v3.arbiter import Arbiter, DecisionKind
from core.runtime_v3.effects import EffectObserver
from core.runtime_v3.executor import Executor, LiberoPrimitiveBackend
from core.runtime_v3.object_relative import (
    CONTROL_TICK_STEP_M,
    DIRECTION_ORDER,
    MAX_ALIGNMENT_TICKS,
    ObjectRelativeAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver,
    REQUESTED_ALIGNMENT_M,
    compare_alignment_improvements,
    frozen_reference_error,
)
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.scene_initialization import run_scene_ready_holds
from core.runtime_v3.selector import DeterministicSelector, Selection
from core.runtime_v3.state import BeliefState, StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick

SUITE = "LIBERO_OBJECT"
TASK_ID = 2
TARGET_PHRASE = "salad dressing"
INIT_STATES = (0, 1, 2, 3, 4, 5)
MAX_ALIGNMENT_STEPS = 6
PRE_SETTLE_TICKS = 4
REQUESTED_MM = REQUESTED_ALIGNMENT_M * 1000.0
MAX_TICKS = MAX_ALIGNMENT_TICKS
CONTROL_TICK_MM = CONTROL_TICK_STEP_M * 1000.0
TARGET_MOTION_DIAGNOSTIC_EPSILON_M = 1e-6
OPPOSITE_DIRECTIONS = {
    "FWD": "BACK", "BACK": "FWD", "LEFT": "RIGHT",
    "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP",
}


class CountingArbiter(Arbiter):
    """Count approvals and write/read back the action artifact before each one."""

    def __init__(self, *, before_authorize: Callable[..., None]) -> None:
        super().__init__()
        self.before_authorize = before_authorize
        self.authorization_calls = 0
        self.alignment_authorization_calls = 0
        self.approval_count = 0

    def authorize(self, state, options, selection):
        self.authorization_calls += 1
        if selection.option_id == "ALIGN_TO_TARGET_SMALL":
            self.alignment_authorization_calls += 1
            self.before_authorize(state, options, selection)
        decision = super().authorize(state, options, selection)
        if (decision.kind == DecisionKind.APPROVED and decision.action is not None
                and decision.action.option_id == "ALIGN_TO_TARGET_SMALL"):
            self.approval_count += 1
        return decision


class AlignmentSelector(DeterministicSelector):
    """Select the only Runtime option; stop through the Arbiter when none exists."""

    def select(self, state, options):
        if any(option.option_id == "ALIGN_TO_TARGET_SMALL" for option in options):
            return Selection("ALIGN_TO_TARGET_SMALL", parsed_selection="ALIGN_TO_TARGET_SMALL")
        return Selection("ABORT", status="ABORT", raw_output="no positive alignment option")


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


def write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(record), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def new_run_dir(base: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(base).expanduser() / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def oracle_target_world_position(environment: Any) -> dict[str, Any]:
    """Read simulator pose for diagnostics; callers never pass it into Runtime."""
    target_body = "salad_dressing_1_main"
    try:
        sim = environment.env.sim
        model, data = sim.model, sim.data
        if callable(getattr(model, "body_name2id", None)):
            body_id = int(model.body_name2id(target_body))
        elif callable(getattr(model, "body", None)):
            body_id = int(model.body(target_body).id)
        else:
            raise AttributeError("MuJoCo model has no body-name lookup")
        position = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("body position is not finite")
        return {"available": True, "body": target_body, "body_id": body_id,
                "world_position_m": position.tolist(),
                "source": "simulator_body_pose_diagnostic_only"}
    except Exception as exc:  # diagnostic failures must not change control
        return {"available": False, "body": target_body,
                "error": f"{type(exc).__name__}: {exc}",
                "source": "simulator_body_pose_diagnostic_only"}


def classify_pre_action_stop(state: BeliefState) -> str | None:
    """Map current Runtime evidence to the phase's bounded stop reasons."""
    relative = state.object_relative_state
    geometry = state.relevant_geometry
    if relative is None:
        return "REFERENCE_INVALID"
    if not relative.target_visible:
        if (relative.target_identity_status == "TARGET_IDENTITY_LOST"
                and int(geometry.get("sam3_candidate_count", 0) or 0) > 0):
            return "TARGET_IDENTITY_LOST"
        return "TARGET_VISIBILITY_LOST"
    if relative.target_identity_status not in {"ANCHORED", "SAME_TARGET"}:
        return "TARGET_IDENTITY_LOST"
    if (not relative.target_reference_valid or relative.target_reference_point_px is None
            or relative.target_reference_invalidation_reason is not None):
        return "REFERENCE_INVALID"
    if (relative.scene_ready_gate_enabled and not relative.scene_ready):
        return "REFERENCE_INVALID"
    candidates = candidate_records(geometry.get("candidate_directions", []))
    if not any(_positive_improvement(item) for item in candidates):
        return "NO_POSITIVE_OPTION"
    return None


def _positive_improvement(candidate: Mapping[str, Any]) -> bool:
    if not candidate.get("valid"):
        return False
    try:
        value = float(candidate.get("predicted_improvement_px"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def candidate_records(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain all six hypotheses with enough geometry to audit the winner."""
    result = []
    for candidate in candidates:
        result.append({
            "direction": candidate.get("direction"),
            "valid": bool(candidate.get("valid", False)),
            "reason": candidate.get("reason"),
            "predicted_eef_projection_px": candidate.get("hypothetical_projection_px"),
            "predicted_error_px": candidate.get("predicted_error_after_px"),
            "predicted_improvement_px": candidate.get("predicted_improvement_px"),
        })
    return result


def error_from_state(state: BeliefState) -> float | None:
    relative = state.object_relative_state
    if relative is None:
        return None
    return frozen_reference_error(
        relative.target_reference_point_px, relative.eef_projection_px,
        reference_valid=relative.target_reference_valid,
    )


def count_direction_reversals(sequence: Sequence[str]) -> int:
    return sum(OPPOSITE_DIRECTIONS.get(left) == right
               for left, right in zip(sequence, sequence[1:]))


def same_fixed_reference(before: Sequence[float] | None,
                         after: Sequence[float] | None) -> bool:
    if before is None or after is None:
        return False
    try:
        left = np.asarray(before, dtype=float).reshape(2)
        right = np.asarray(after, dtype=float).reshape(2)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))
                and np.allclose(left, right, rtol=0.0, atol=1e-9))


def post_action_termination(
    *, alignment_step: int, actual_improvement_px: float | None, state_after: BeliefState,
) -> str | None:
    reason = classify_pre_action_stop(state_after)
    if reason in {"TARGET_IDENTITY_LOST", "TARGET_VISIBILITY_LOST", "REFERENCE_INVALID"}:
        return reason
    if actual_improvement_px is None or actual_improvement_px <= 0.0:
        return "EFFECT_NOT_IMPROVED"
    if reason == "NO_POSITIVE_OPTION":
        return reason
    if int(alignment_step) >= MAX_ALIGNMENT_STEPS:
        return "MAX_ALIGNMENT_STEPS"
    return None


def normalized_trajectory(errors: Sequence[float | None]) -> list[float | None]:
    if not errors or errors[0] is None or not math.isfinite(float(errors[0])) or errors[0] == 0:
        return [None for _ in errors]
    initial = float(errors[0])
    return [float(value) / initial if value is not None and math.isfinite(float(value)) else None
            for value in errors]


def make_step_visual(
    frame: Mapping[str, Any], output_dir: Path, *, prefix: str,
    selected_direction: str | None = None,
) -> dict[str, str]:
    """Save canonical source RGB and an overlay of the frozen reference and six projections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_array = np.ascontiguousarray(frame["image"], dtype=np.uint8)
    source = Image.fromarray(image_array, mode="RGB")
    rgb_path = output_dir / f"{prefix}_canonical_rgb.png"
    source.save(rgb_path)
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    relative = frame.get("object_relative_state")
    segmentation = frame.get("segmentation")
    mask_path = None
    mask = getattr(segmentation, "mask", None)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == (source.height, source.width):
            mask_path = output_dir / f"{prefix}_sam_mask.png"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            tint = np.zeros((source.height, source.width, 4), dtype=np.uint8)
            tint[mask] = (255, 35, 35, 92)
            overlay = Image.alpha_composite(overlay.convert("RGBA"),
                                            Image.fromarray(tint, mode="RGBA")).convert("RGB")
            draw = ImageDraw.Draw(overlay)

    def point(raw: Any) -> tuple[float, float] | None:
        try:
            coords = np.asarray(raw, dtype=float).reshape(2)
        except (TypeError, ValueError):
            return None
        return (float(coords[0]), float(coords[1])) if np.all(np.isfinite(coords)) else None

    def marker(raw: Any, color: tuple[int, int, int], radius: int = 6) -> None:
        p = point(raw)
        if p is not None:
            x, y = p
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)

    if relative is not None:
        marker(relative.target_reference_point_px, (255, 215, 0), 9)
        marker(relative.eef_projection_px, (0, 220, 255), 7)
        marker(relative.target_centroid_px, (255, 55, 55), 5)
    colors = ((40, 210, 90), (60, 155, 255), (240, 150, 30),
              (180, 90, 255), (30, 220, 220), (255, 90, 160))
    resolution = frame.get("resolution", {})
    lattice = resolution.get("candidate_lattice", [])
    candidates = lattice if lattice else resolution.get("candidate_directions", [])
    chosen = resolution.get("chosen_lattice_candidate")
    for index, candidate in enumerate(candidates):
        p = point(candidate.get("predicted_projection_px")
                   or candidate.get("hypothetical_projection_px"))
        if p is None:
            continue
        direction = str(candidate.get("direction", "?"))
        selected = direction == selected_direction
        if chosen is not None:
            selected = (direction == chosen.get("direction")
                        and candidate.get("displacement_m") == chosen.get("displacement_m"))
        color = (255, 255, 255) if selected else colors[index % len(colors)]
        radius = 8 if selected else 4
        x, y = p
        draw.rectangle((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        label = direction
        if candidate.get("displacement_mm") is not None:
            label += f" {float(candidate['displacement_mm']):g}mm"
        draw.text((x + 7, y + 3), label, fill=color)
    draw.rectangle((5, 5, min(source.width - 6, 268), 65), fill=(0, 0, 0))
    draw.text((10, 10), "yellow: fixed target reference", fill=(255, 215, 0))
    draw.text((10, 27), "cyan: EEF, red: current SAM centroid", fill=(245, 245, 245))
    draw.text((10, 44), f"white square: selected {selected_direction or 'none'}", fill=(255, 255, 255))
    overlay_path = output_dir / f"{prefix}_alignment_overlay.png"
    overlay.save(overlay_path)
    result = {"canonical_rgb": str(rgb_path), "alignment_overlay": str(overlay_path)}
    if mask_path is not None:
        result["sam_mask"] = str(mask_path)
    return result


def save_contact_sheet(images: Sequence[tuple[str, Path]], output_path: Path) -> str | None:
    if not images:
        return None
    thumb_w, thumb_h = 256, 276
    columns = 4
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(images):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w - 8, thumb_h - 28))
            x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
            y = (index // columns) * thumb_h + 22
            sheet.paste(image, (x, y))
        draw.text(((index % columns) * thumb_w + 6,
                   (index // columns) * thumb_h + 5), label, fill=(245, 245, 245))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def save_error_trajectory(episodes: Sequence[Mapping[str, Any]], output_path: Path) -> dict[str, Any]:
    normalized = {str(row["init_state_index"]): normalized_trajectory(row.get("error_trajectory_px", []))
                  for row in episodes}
    horizon = max((len(values) for values in normalized.values()), default=0)
    aggregate = []
    for index in range(horizon):
        values = [row[index] for row in normalized.values()
                  if index < len(row) and row[index] is not None]
        aggregate.append(float(np.mean(values)) if values else None)
    width, height = 900, 500
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot = (margin, margin, width - 25, height - margin)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=(40, 40, 40), width=2)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=(40, 40, 40), width=2)
    draw.text((margin, 18), "Normalized frozen-reference error by alignment step", fill=(0, 0, 0))
    draw.text((8, margin), "1.0", fill=(0, 0, 0))
    draw.text((8, plot[3] - 7), "0", fill=(0, 0, 0))
    max_x = max(horizon - 1, 1)

    def coords(values: Sequence[float | None]) -> list[tuple[int, int]]:
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = plot[0] + int((plot[2] - plot[0]) * index / max_x)
            y = plot[3] - int((plot[3] - plot[1]) * min(max(value, 0.0), 1.5) / 1.5)
            points.append((x, y))
        return points

    palette = ((30, 110, 220), (230, 70, 50), (30, 160, 90), (170, 80, 200),
               (230, 145, 20), (40, 170, 180))
    for index, (label, values) in enumerate(normalized.items()):
        points = coords(values)
        if len(points) > 1:
            draw.line(points, fill=palette[index % len(palette)], width=2)
    aggregate_points = coords(aggregate)
    if len(aggregate_points) > 1:
        draw.line(aggregate_points, fill=(0, 0, 0), width=4)
    draw.text((plot[0], height - 30), "step 0", fill=(0, 0, 0))
    draw.text((plot[2] - 48, height - 30), f"step {max_x}", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {"path": str(output_path), "normalized_by_episode": normalized,
            "mean_normalized_error_by_step": aggregate}


def _state_reference(state: BeliefState) -> tuple[float, float] | None:
    relative = state.object_relative_state
    if relative is None or relative.target_reference_point_px is None:
        return None
    return tuple(float(value) for value in relative.target_reference_point_px)


def _oracle_motion(before: Mapping[str, Any] | None,
                   after: Mapping[str, Any] | None) -> dict[str, Any]:
    displacement = None
    norm = None
    if before and before.get("available") and after and after.get("available"):
        delta = (np.asarray(after["world_position_m"], dtype=float)
                 - np.asarray(before["world_position_m"], dtype=float))
        displacement, norm = delta.tolist(), float(np.linalg.norm(delta))
    return {
        "before_world_position_m": before.get("world_position_m") if before else None,
        "after_world_position_m": after.get("world_position_m") if after else None,
        "displacement_vector_m": displacement,
        "displacement_norm_m": norm,
        "target_moved_diagnostic": bool(
            norm is not None and norm > TARGET_MOTION_DIAGNOSTIC_EPSILON_M),
        "motion_threshold_m": TARGET_MOTION_DIAGNOSTIC_EPSILON_M,
        "used_by_runtime": False,
    }


def _run_episode(
    *, init_state_index: int, run_dir: Path, config: Mapping[str, Any], sam3: Any,
    workspace: tuple[float, float], camera_resolution: int,
) -> dict[str, Any]:
    # Import simulator adapters only for the experiment path; unit tests can
    # exercise the loop contracts without importing LIBERO or launching MuJoCo.
    from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
    from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
    from core.runtime_v3.canonical_image import CanonicalImageAdapter
    from interpreters.libero_atomic_controller import LiberoAtomicController

    episode_dir = run_dir / f"init_state_{init_state_index}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = episode_dir / "steps"
    artifacts_dir.mkdir()
    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state_index,
        seed=0, camera_height=camera_resolution, camera_width=camera_resolution,
        # 4 RobotReady HOLDs + up to 40 SceneReady HOLDs + 6 * 5 motion
        # ticks fit inside this environment cap, even at each contract maximum.
        horizon=80,
    )
    try:
        if TARGET_PHRASE.casefold() not in environment.task_description.casefold():
            raise RuntimeError(f"task instruction does not include {TARGET_PHRASE!r}")
        controller = LiberoAtomicController(
            move_vectors=config["move_vectors"], step_m=CONTROL_TICK_MM / 1000.0,
            sim_steps_per_decision=1,
            position_scale_m=float(config.get("position_scale_m", 0.05)),
        )
        base_observer = LiberoObservationAdapter(
            max_eef_z_m=workspace[1], min_eef_z_m=workspace[0],
            safe_lift_step_m=CONTROL_TICK_MM / 1000.0,
        )
        pre_settle = []
        reset = True
        for hold_index in range(PRE_SETTLE_TICKS):
            outcome = run_v3_tick(
                environment, base_observer, controller, task_id=f"{SUITE}:{TASK_ID}",
                token=None, direction_unit=None,
                commanded_step_m=CONTROL_TICK_MM / 1000.0,
                reset=reset, workspace_z_bounds_m=workspace,
            )
            reset = False
            if outcome.get("actions") != 1 or not outcome.get("backend_execution"):
                raise RuntimeError(f"RobotReady HOLD {hold_index + 1} failed: {outcome}")
            pre_settle.append(outcome)

        observer = ObjectRelativePerceptionObserver(
            base_observer, sam3, target_phrase=TARGET_PHRASE,
            move_vectors=config["move_vectors"],
            canonical_image_adapter=CanonicalImageAdapter(), scene_ready_required=True,
        )
        scene_ready = run_scene_ready_holds(
            environment, observer, controller, task_id=f"{SUITE}:{TASK_ID}",
            commanded_step_m=CONTROL_TICK_MM / 1000.0,
            workspace_z_bounds_m=workspace, max_hold_ticks=40,
        )
        if not scene_ready.get("ready") or not observer.scene_ready:
            return {
                "init_state_index": init_state_index, "status": "INITIALIZATION_FAILED",
                "termination_reason": "SCENE_READY_INIT_FAILED",
                "pre_settle_hold_ticks": len(pre_settle), "scene_ready": scene_ready,
                "alignment_steps": [], "bounded_actions": 0,
            }
        scene_ready_trigger_tick = next(
            (int(sample["environment_step"]) for sample in scene_ready.get("samples", [])
             if sample.get("scene_ready")), None,
        )
        if scene_ready_trigger_tick is None:
            return {
                "init_state_index": init_state_index, "status": "INITIALIZATION_FAILED",
                "termination_reason": "SCENE_READY_INIT_FAILED",
                "pre_settle_hold_ticks": len(pre_settle), "scene_ready": scene_ready,
                "alignment_steps": [], "bounded_actions": 0,
            }

        steps: list[dict[str, Any]] = []
        image_panels: list[tuple[str, Path]] = []
        oracle_tick_samples: list[dict[str, Any]] = []
        oracle_episode_start = oracle_target_world_position(environment)
        active_step = {"index": 0}

        def write_pre_action_ready(state, options, selection) -> None:
            step_index = int(active_step["index"])
            frame = observer.perception_history[-1]
            chosen = next((option for option in options
                           if option.option_id == selection.option_id), None)
            if chosen is None or chosen.primitive.micro_motion_spec is None:
                raise RuntimeError("Runtime selected no bounded object-relative micro-motion")
            relative = state.object_relative_state
            if (relative is None or not relative.target_reference_valid
                    or relative.target_reference_point_px is None):
                raise RuntimeError("frozen target reference is invalid before Arbiter approval")
            if not observer.scene_ready or (relative.scene_ready_gate_enabled and not relative.scene_ready):
                raise RuntimeError("SceneReady initialization evidence is no longer valid")
            direction = chosen.primitive.micro_motion_spec.direction
            frame_dir = artifacts_dir / f"step_{step_index:02d}"
            visual = make_step_visual(frame, frame_dir, prefix="before",
                                      selected_direction=direction)
            ready = {
                "status": "PRE_ACTION_READY", "alignment_step": step_index,
                "frame_id": state.frame_id, "option_id": selection.option_id,
                "direction": direction,
                "target_reference_px": relative.target_reference_point_px,
                "target_reference_valid": relative.target_reference_valid,
                "candidate_directions": candidate_records(
                    state.relevant_geometry.get("candidate_directions", [])),
                "selected_predicted_projection_px": next((item.get("predicted_eef_projection_px")
                    for item in candidate_records(state.relevant_geometry.get("candidate_directions", []))
                    if item.get("direction") == direction), None),
                "scene_ready_initialization_passed": observer.scene_ready,
                "scene_ready_trigger_environment_tick": scene_ready_trigger_tick,
                "visual_artifacts": visual,
                "oracle_used_by_runtime": False,
            }
            ready_path = frame_dir / "PRE_ACTION_READY.json"
            write_json(ready_path, ready)
            if json.loads(ready_path.read_text(encoding="utf-8")).get("status") != "PRE_ACTION_READY":
                raise RuntimeError("PRE_ACTION_READY read-back failed")
            if step_index == 1:
                image_panels.append((f"step 0: initial, before {direction}",
                                     Path(visual["alignment_overlay"])))

        arbiter = CountingArbiter(before_authorize=write_pre_action_ready)
        backend = LiberoPrimitiveBackend(environment, controller, arbiter)
        executor = Executor(backend, arbiter)

        def observe_execution_tick(current_environment):
            observation = base_observer.observe(current_environment)
            # Diagnostic only: the return value is never included in observation
            # evidence or supplied to the Arbiter, option generator or Executor.
            oracle_tick_samples.append(oracle_target_world_position(current_environment))
            return observation

        observer.observe_for_execution_tick = observe_execution_tick
        termination_reason = "MAX_ALIGNMENT_STEPS"
        initial_error = None
        final_error = None
        reference_anchor_initial = None

        for step_index in range(1, MAX_ALIGNMENT_STEPS + 1):
            active_step["index"] = step_index
            authorizations_before = arbiter.authorization_calls
            align_authorizations_before = arbiter.alignment_authorization_calls
            approvals_before = arbiter.approval_count
            events: list[dict[str, Any]] = []
            runner = RuntimeV3Runner(
                observer=observer, state_builder=StateBuilder(),
                option_generator=ObjectRelativeAlignmentOptionGenerator(),
                selector=AlignmentSelector(), arbiter=arbiter, executor=executor,
                effect_observer=EffectObserver(), logger=events.append,
            )
            result = runner.run_episode(
                environment, task_id=f"{SUITE}:{TASK_ID}", max_steps=1, reset=False,
            )
            if not events:
                state = result.get("state") or runner.state
                if result.get("status") == "ARBITER_REJECTED":
                    termination_reason = ((classify_pre_action_stop(state) if state is not None else None)
                                          or "ARBITER_REJECTED")
                else:
                    termination_reason = "EXECUTION_FAILED"
                current_error = error_from_state(state) if state is not None else None
                if initial_error is None:
                    initial_error = current_error
                final_error = current_error if current_error is not None else final_error
                steps.append({
                    "alignment_step": step_index, "executed": False,
                    "termination_reason": termination_reason,
                    "runner_status": result.get("status"),
                    "error_before_px": current_error,
                    "target_identity_status": (state.object_relative_state.target_identity_status
                                                if state and state.object_relative_state else None),
                    "target_visible": (state.object_relative_state.target_visible
                                       if state and state.object_relative_state else False),
                    "target_reference_valid": (state.object_relative_state.target_reference_valid
                                               if state and state.object_relative_state else False),
                    "candidate_directions": (candidate_records(state.relevant_geometry.get(
                        "candidate_directions", [])) if state else []),
                    "authorization_calls_for_step": arbiter.authorization_calls - authorizations_before,
                    "alignment_authorization_calls_for_step": (
                        arbiter.alignment_authorization_calls - align_authorizations_before),
                    "approvals_for_step": arbiter.approval_count - approvals_before,
                })
                write_json(artifacts_dir / f"step_{step_index:02d}" / "step.json", steps[-1])
                frame = observer.perception_history[-1] if observer.perception_history else None
                if frame is not None:
                    visual = make_step_visual(frame, artifacts_dir / f"step_{step_index:02d}",
                                              prefix="stop")
                    if not image_panels:
                        image_panels.append((f"step 0: stop {termination_reason}",
                                             Path(visual["alignment_overlay"])))
                break

            event = events[0]
            state_before = event["state_before"]
            state_after = event["state_after"]
            before_frame, after_frame = observer.perception_history[-2:]
            action = event["approved_action"]
            direction = action.primitive.micro_motion_spec.direction
            reference = _state_reference(state_before)
            if reference_anchor_initial is None:
                reference_anchor_initial = reference
            error_before = error_from_state(state_before)
            error_after = error_from_state(state_after)
            reference_after = _state_reference(state_after)
            if error_before is not None and initial_error is None:
                initial_error = error_before
            final_error = error_after if error_after is not None else final_error
            same_reference = same_fixed_reference(reference, reference_after)
            actual_improvement = (error_before - error_after
                                  if error_before is not None and error_after is not None
                                  and same_reference else None)
            predicted_improvement = action.expected_effect.get("predicted_improvement_px")
            prediction = compare_alignment_improvements(
                predicted_improvement_px=predicted_improvement,
                actual_improvement_px=actual_improvement,
            )
            candidates = candidate_records(state_before.relevant_geometry.get(
                "candidate_directions", []))
            candidate = next((item for item in candidates if item.get("direction") == direction), {})
            step_dir = artifacts_dir / f"step_{step_index:02d}"
            before_visual = make_step_visual(before_frame, step_dir, prefix="before",
                                             selected_direction=direction)
            after_visual = make_step_visual(after_frame, step_dir, prefix="after")
            image_panels.append((f"step {step_index}: after {direction}",
                                 Path(after_visual["alignment_overlay"])))
            execution_result = getattr(event.get("execution"), "result", None)
            after_relative = state_after.object_relative_state
            identity_status = (after_relative.target_identity_status if after_relative else None)
            target_visible = bool(after_relative and after_relative.target_visible)
            ref_valid_after = bool(after_relative and after_relative.target_reference_valid)
            step = {
                "alignment_step": step_index, "executed": True,
                "option_id": action.option_id, "direction": direction,
                "frame_id_before": state_before.frame_id,
                "frame_id_after": state_after.frame_id,
                "error_before_px": error_before,
                "predicted_error_after_px": action.expected_effect.get("predicted_image_error_after_px"),
                "predicted_improvement_px": predicted_improvement,
                "actual_improvement_px": actual_improvement,
                "prediction_residual_actual_minus_predicted_px": prediction[
                    "prediction_residual_actual_minus_predicted_px"],
                "error_after_px": error_after,
                "candidate_directions": candidates,
                "chosen_candidate": candidate,
                "target_reference_before_px": reference,
                "target_reference_after_px": reference_after,
                "target_reference_unchanged": bool(same_reference),
                "reference_matches_episode_initial": same_fixed_reference(
                    reference_anchor_initial, reference),
                "target_identity_status_after": identity_status,
                "target_visible_after": target_visible,
                "target_reference_valid_after": ref_valid_after,
                "authorization_calls_for_step": arbiter.authorization_calls - authorizations_before,
                "alignment_authorization_calls_for_step": (
                    arbiter.alignment_authorization_calls - align_authorizations_before),
                "arbiter_approved_alignment_actions_for_step": arbiter.approval_count - approvals_before,
                "executor_calls_for_action": 1,
                "runner_status": result.get("status"),
                "execution": execution_result,
                "visual_artifacts": {"before": before_visual, "after": after_visual},
                "oracle_target_pose_after_diagnostic": oracle_target_world_position(environment),
                "oracle_used_by_runtime": False,
            }
            steps.append(step)
            write_json(step_dir / "step.json", step)

            stop = post_action_termination(
                alignment_step=step_index, actual_improvement_px=actual_improvement,
                state_after=state_after,
            )
            if stop is not None:
                termination_reason = stop
                break

        executed_steps = [step for step in steps if step.get("executed")]
        trajectory: list[float | None] = []
        if executed_steps:
            trajectory.append(executed_steps[0].get("error_before_px"))
            trajectory.extend(step.get("error_after_px") for step in executed_steps)
        elif steps:
            trajectory.append(steps[0].get("error_before_px"))
        directions = [str(step["direction"]) for step in executed_steps]
        positive_count = sum(float(step["actual_improvement_px"]) > 0.0
                             for step in executed_steps
                             if step.get("actual_improvement_px") is not None)
        cumulative = (initial_error - final_error
                      if initial_error is not None and final_error is not None else None)
        oracle_after = oracle_target_world_position(environment)
        oracle_diagnostic = _oracle_motion(oracle_episode_start, oracle_after)
        oracle_diagnostic["label"] = (
            "TARGET_MOVED_DIAGNOSTIC" if oracle_diagnostic["target_moved_diagnostic"]
            else "NO_TARGET_MOTION_DIAGNOSTIC"
        )
        contact_sheet = save_contact_sheet(
            image_panels, episode_dir / "contact_sheet.png",
        )
        episode = {
            "init_state_index": init_state_index, "status": "COMPLETED",
            "suite": SUITE, "task_id": TASK_ID, "seed": 0,
            "task_instruction": environment.task_description,
            "pre_settle_hold_ticks": len(pre_settle), "pre_settle_cycles": pre_settle,
            "scene_ready_initialization": scene_ready,
            "scene_ready_trigger_environment_tick": scene_ready_trigger_tick,
            "max_alignment_steps": MAX_ALIGNMENT_STEPS,
            "motion_contract": {"requested_displacement_mm": REQUESTED_MM,
                                 "max_ticks": MAX_TICKS,
                                 "control_tick_mm": CONTROL_TICK_MM},
            "initial_error_px": initial_error, "final_error_px": final_error,
            "cumulative_improvement_px": cumulative,
            "alignment_steps": steps, "executed_alignment_steps": len(executed_steps),
            "bounded_actions": len(executed_steps),
            "termination_reason": termination_reason,
            "all_steps_positive": bool(executed_steps and positive_count == len(executed_steps)),
            "step_verification_rate": (positive_count / len(executed_steps)
                                       if executed_steps else None),
            "direction_sequence": directions,
            "direction_transitions": sum(left != right for left, right in zip(directions, directions[1:])),
            "direction_reversal_count": count_direction_reversals(directions),
            "error_trajectory_px": trajectory,
            "normalized_error_trajectory": normalized_trajectory(trajectory),
            "target_reference_anchor_initial_px": reference_anchor_initial,
            "target_reference_fixed_for_episode": bool(executed_steps) and all(
                step.get("target_reference_unchanged", False)
                and step.get("reference_matches_episode_initial", False)
                for step in executed_steps),
            "arbiter_authorization_calls": arbiter.authorization_calls,
            "arbiter_alignment_authorization_calls": arbiter.alignment_authorization_calls,
            "arbiter_approved_alignment_actions": arbiter.approval_count,
            "one_approval_per_executed_step": arbiter.approval_count == len(executed_steps),
            "oracle_target_motion_diagnostic": oracle_diagnostic,
            "oracle_tick_samples": oracle_tick_samples,
            "sam_identity_retention": bool(executed_steps) and all(
                step.get("target_identity_status_after") == "SAME_TARGET"
                for step in executed_steps),
            "contact_sheet": contact_sheet,
            "legacy_modified": False, "qwen_actions": 0,
        }
        write_json(episode_dir / "episode.json", episode)
        return episode
    finally:
        environment.close()


def summarize(episodes: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    steps = [step for episode in episodes for step in episode.get("alignment_steps", [])
             if step.get("executed")]
    improvements = [float(step["actual_improvement_px"]) for step in steps
                    if step.get("actual_improvement_px") is not None]
    residuals = [float(step["prediction_residual_actual_minus_predicted_px"])
                 for step in steps
                 if step.get("prediction_residual_actual_minus_predicted_px") is not None]
    chosen_counts: dict[str, int] = {direction: 0 for direction in DIRECTION_ORDER}
    candidate_values: dict[str, list[float]] = {}
    candidate_values_by_step: dict[int, dict[str, list[float]]] = {}
    candidate_observations: dict[str, int] = {direction: 0 for direction in DIRECTION_ORDER}
    candidate_invalid: dict[str, int] = {direction: 0 for direction in DIRECTION_ORDER}
    for episode in episodes:
        for step in episode.get("alignment_steps", []):
            if step.get("executed") and step.get("direction"):
                direction = str(step["direction"])
                chosen_counts[direction] = chosen_counts.get(direction, 0) + 1
            for candidate in step.get("candidate_directions", []):
                direction = str(candidate["direction"])
                candidate_observations[direction] = candidate_observations.get(direction, 0) + 1
                if not candidate.get("valid"):
                    candidate_invalid[direction] = candidate_invalid.get(direction, 0) + 1
                value = candidate.get("predicted_improvement_px")
                if candidate.get("valid") and value is not None:
                    candidate_values.setdefault(direction, []).append(float(value))
                    candidate_values_by_step.setdefault(
                        int(step.get("alignment_step", 0)),
                        {name: [] for name in DIRECTION_ORDER},
                    )[direction].append(float(value))
    candidate_distribution = {
        direction: {
            "observations": candidate_observations.get(direction, 0),
            "valid_count": len(candidate_values.get(direction, [])),
            "invalid_count": candidate_invalid.get(direction, 0),
            "mean_px": (float(np.mean(candidate_values[direction]))
                        if candidate_values.get(direction) else None),
            "min_px": (float(np.min(candidate_values[direction]))
                       if candidate_values.get(direction) else None),
            "max_px": (float(np.max(candidate_values[direction]))
                       if candidate_values.get(direction) else None),
        }
        for direction in DIRECTION_ORDER
    }
    positive_steps = sum(value > 0.0 for value in improvements)
    monotonic_episodes = [episode for episode in episodes
                          if int(episode.get("executed_alignment_steps", 0)) > 0]
    monotonic_count = sum(bool(episode.get("all_steps_positive"))
                          for episode in monotonic_episodes)
    stable_episodes = [episode for episode in episodes
                       if not episode.get("oracle_target_motion_diagnostic", {}).get(
                           "target_moved_diagnostic")]
    stable_steps = [step for episode in stable_episodes
                    for step in episode.get("alignment_steps", []) if step.get("executed")]
    stable_actual = [float(step["actual_improvement_px"]) for step in stable_steps
                     if step.get("actual_improvement_px") is not None]
    stable_positive = sum(value > 0.0 for value in stable_actual)
    stable_evaluable_episodes = [episode for episode in stable_episodes
                                 if int(episode.get("executed_alignment_steps", 0)) > 0]
    stable_monotonic_count = sum(bool(episode.get("all_steps_positive"))
                                 for episode in stable_evaluable_episodes)
    cumulative_values = [float(episode["cumulative_improvement_px"]) for episode in episodes
                         if episode.get("cumulative_improvement_px") is not None]
    trajectories = save_error_trajectory(
        episodes, output_dir / "aggregate_normalized_error_trajectory.png",
    )
    return {
        "episodes": len(episodes),
        "actual_bounded_actions": len(steps),
        "max_allowed_bounded_actions": len(INIT_STATES) * MAX_ALIGNMENT_STEPS,
        "step_verification_rate": positive_steps / len(steps) if steps else None,
        "positive_effect_steps": positive_steps,
        "mean_improvement_per_step_px": float(np.mean(improvements)) if improvements else None,
        "mean_prediction_residual_px": float(np.mean(residuals)) if residuals else None,
        "episode_monotonic_rate": (monotonic_count / len(episodes) if episodes else None),
        "evaluable_episode_monotonic_rate": (monotonic_count / len(monotonic_episodes)
                                             if monotonic_episodes else None),
        "monotonic_episodes": monotonic_count,
        "evaluable_episodes": len(monotonic_episodes),
        "mean_cumulative_improvement_px": (float(np.mean(cumulative_values))
                                            if cumulative_values else None),
        "error_increase_exists": any(value < 0.0 for value in improvements),
        "direction_counts": chosen_counts,
        "direction_transitions": sum(int(episode.get("direction_transitions", 0))
                                      for episode in episodes),
        "direction_reversal_count": sum(int(episode.get("direction_reversal_count", 0))
                                         for episode in episodes),
        "candidate_predicted_improvement_distribution_px": candidate_distribution,
        "candidate_predicted_improvement_distribution_by_step_px": {
            str(step_index): {
                direction: {"count": len(values),
                            "mean_px": float(np.mean(values)) if values else None}
                for direction, values in directions.items()
            }
            for step_index, directions in sorted(candidate_values_by_step.items())
        },
        "mean_normalized_error_by_step": trajectories["mean_normalized_error_by_step"],
        "normalized_error_trajectory_plot": trajectories["path"],
        "arbiter_approvals": sum(int(episode.get("arbiter_approved_alignment_actions", 0))
                                  for episode in episodes),
        "one_approval_per_executed_step": all(
            episode.get("one_approval_per_executed_step") for episode in episodes),
        "qwen_actions": 0,
        "oracle_motion_diagnostic_episodes": sum(
            bool(episode.get("oracle_target_motion_diagnostic", {}).get("target_moved_diagnostic"))
            for episode in episodes),
        "target_stable_episodes": len(stable_episodes),
        "target_stable_executed_steps": len(stable_steps),
        "target_stable_step_verification_rate": (
            stable_positive / len(stable_steps) if stable_steps else None),
        "target_stable_episode_monotonic_rate": (
            stable_monotonic_count / len(stable_evaluable_episodes)
            if stable_evaluable_episodes else None),
        "oracle_used_by_runtime": False,
    }


def _configure_local_sam3_proxy_bypass(url: str) -> None:
    if urlparse(str(url)).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    names = ("NO_PROXY", "no_proxy")
    entries = [item.strip() for name in names
               for item in os.environ.get(name, "").split(",") if item.strip()]
    value = ",".join(dict.fromkeys((*entries, "127.0.0.1", "localhost", "::1")))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_multistep_alignment"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    return parser


def main() -> int:
    from core.capabilities.sam3_client import Sam3Client
    from core.config import load_yaml

    args = _parser().parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("M3.2 is frozen to the established direct 512x512 RGB path")
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_local_sam3_proxy_bypass(args.sam3_url)
    run_dir = new_run_dir(args.output_dir)
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    episodes: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        for init_state_index in INIT_STATES:
            try:
                episode = _run_episode(
                    init_state_index=init_state_index, run_dir=run_dir,
                    config=config, sam3=sam3, workspace=workspace,
                    camera_resolution=args.camera_resolution,
                )
                episodes.append(episode)
                if episode.get("termination_reason") == "SCENE_READY_INIT_FAILED":
                    blockers.append(f"init_state_{init_state_index}: SceneReady initialization failed")
            except Exception as exc:
                blockers.append(f"init_state_{init_state_index}: {type(exc).__name__}: {exc}")
                failure_dir = run_dir / f"init_state_{init_state_index}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                write_json(failure_dir / "failure.json", {
                    "init_state_index": init_state_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "bounded_actions": 0,
                })
    finally:
        sam3.close()
    metrics = summarize(episodes, run_dir)
    record = {
        "status": "COMPLETED" if len(episodes) == len(INIT_STATES) and not blockers else "PARTIAL",
        "phase": "M3.2 Multi-step Object-relative Closed-loop Alignment",
        "branch": "runtime-v3", "starting_commit": "8a3cf1554b4dbe62c5247c978131fee95a35510d",
        "suite": SUITE, "task_id": TASK_ID, "seed": 0,
        "init_states": list(INIT_STATES), "target_phrase": TARGET_PHRASE,
        "camera_resolution": [512, 512], "canonical_transform": "vertical_flip",
        "max_alignment_steps_per_episode": MAX_ALIGNMENT_STEPS,
        "motion_contract": {"requested_displacement_mm": REQUESTED_MM,
                             "max_ticks": MAX_TICKS,
                             "control_tick_mm": CONTROL_TICK_MM},
        "episodes": episodes, "metrics": metrics, "blockers": blockers,
        "oracle_used_by_runtime": False, "qwen_actions": 0,
        "legacy_modified": False,
    }
    write_json(run_dir / "summary.json", record)
    print(json.dumps(jsonable(record), indent=2, ensure_ascii=False))
    return 0 if record["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
