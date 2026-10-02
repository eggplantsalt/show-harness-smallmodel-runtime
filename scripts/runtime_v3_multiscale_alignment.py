#!/usr/bin/env python3
"""M3.3: calibrate and evaluate Runtime-owned verified alignment scales."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
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
    CONTROL_TICK_STEP_M, DIRECTION_ORDER, MultiScaleAlignmentOptionGenerator,
    ObjectRelativePerceptionObserver, compare_alignment_improvements,
    frozen_reference_error,
)
from core.runtime_v3.options import BoundedMicroMotionSpec, PrimitiveCommand, RuntimeOption
from core.runtime_v3.runner import RuntimeV3Runner
from core.runtime_v3.scene_initialization import run_scene_ready_holds
from core.runtime_v3.selector import DeterministicSelector, Selection
from core.runtime_v3.state import BeliefState, StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from scripts.runtime_v3_multistep_alignment import (
    CONTROL_TICK_MM, INIT_STATES, MAX_ALIGNMENT_STEPS, PRE_SETTLE_TICKS,
    SUITE, TASK_ID, TARGET_PHRASE,
    _configure_local_sam3_proxy_bypass, _oracle_motion, _state_reference,
    count_direction_reversals, error_from_state,
    make_step_visual, new_run_dir, normalized_trajectory, oracle_target_world_position,
    save_contact_sheet, same_fixed_reference, write_json,
)

SCALES_M = (0.003, 0.006, 0.009)
TICK_BUDGETS = {0.003: 5, 0.006: 7, 0.009: 10}
STAGE_A_STATES = (0, 1, 2)
BASELINE_SUMMARY = ROOT / "rollouts/runtime_v3_multistep_alignment/run_20261002T123520Z_3463ba55/summary.json"
SEMANTIC_OPTION_ID = "ALIGN_TO_TARGET_BOUNDED"


class MultiscaleSelector(DeterministicSelector):
    def select(self, state, options):
        option = next((option for option in options if option.option_id in {
            SEMANTIC_OPTION_ID, "CALIBRATE_BOUNDED_MOTION"}), None)
        if option is not None:
            return Selection(option.option_id, parsed_selection=option.option_id)
        return Selection("ABORT", status="ABORT", raw_output="no positive verified alignment option")


class CalibrationOptionGenerator:
    """Expose the fixed experimental cell as one Arbiter-sealed motion option."""

    def __init__(self, direction: str, scale_m: float, max_ticks: int,
                 move_vectors: Mapping[str, Sequence[float]]) -> None:
        self.direction, self.scale_m, self.max_ticks = direction, scale_m, max_ticks
        self.unit = tuple(float(value) for value in move_vectors[f"MV_{direction}"])

    def generate(self, state: BeliefState) -> list[RuntimeOption]:
        relative = state.object_relative_state
        if (relative is None or not relative.target_visible or not relative.target_reference_valid
                or not state.relevant_geometry.get("workspace_valid")):
            return []
        spec = BoundedMicroMotionSpec(
            direction=self.direction, direction_unit=self.unit,
            requested_displacement_m=self.scale_m, max_ticks=self.max_ticks,
            control_tick_step_m=CONTROL_TICK_STEP_M,
        )
        return [RuntimeOption(
            option_id="CALIBRATE_BOUNDED_MOTION", option_type="physical_contract_calibration",
            description="One independently initialized scale and direction calibration trial.",
            preconditions={"observation_fresh": True, "relevant_geometry.workspace_valid": True},
            expected_effect={"calibration_direction": self.direction,
                             "calibration_scale_m": self.scale_m},
            primitive=PrimitiveCommand(kind="micro_motion", max_steps=1,
                                       max_duration_s=5.0, micro_motion_spec=spec),
            confidence=1.0, evidence=state.evidence_refs, evidence_frame_id=state.frame_id,
        )]


class ExperimentArbiter(Arbiter):
    def __init__(self, before_authorize=None):
        super().__init__()
        self.before_authorize = before_authorize
        self.authorization_calls = 0
        self.alignment_authorization_calls = 0
        self.approval_count = 0

    def authorize(self, state, options, selection):
        self.authorization_calls += 1
        if selection.option_id in {SEMANTIC_OPTION_ID, "CALIBRATE_BOUNDED_MOTION"}:
            self.alignment_authorization_calls += 1
            if self.before_authorize:
                self.before_authorize(state, options, selection)
        decision = super().authorize(state, options, selection)
        if (decision.kind == DecisionKind.APPROVED and decision.action is not None
                and decision.action.option_id in {SEMANTIC_OPTION_ID, "CALIBRATE_BOUNDED_MOTION"}):
            self.approval_count += 1
        return decision


def _setup_trial(*, init_state: int, config: Mapping[str, Any], sam3: Any,
                 workspace: tuple[float, float], camera_resolution: int,
                 scales: Sequence[float] | None = None,
                 contracts: Mapping[float, Mapping[str, Any]] | None = None):
    from core.runtime_v3.adapters.libero_env import LiberoEnvironmentAdapter
    from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
    from core.runtime_v3.canonical_image import CanonicalImageAdapter
    from interpreters.libero_atomic_controller import LiberoAtomicController

    environment = LiberoEnvironmentAdapter.create(
        suite_name=SUITE, task_id=TASK_ID, init_state_index=init_state, seed=0,
        camera_height=camera_resolution, camera_width=camera_resolution, horizon=100,
    )
    if TARGET_PHRASE.casefold() not in environment.task_description.casefold():
        environment.close()
        raise RuntimeError(f"task instruction does not include {TARGET_PHRASE!r}")
    controller = LiberoAtomicController(
        move_vectors=config["move_vectors"], step_m=CONTROL_TICK_MM / 1000.0,
        sim_steps_per_decision=1, position_scale_m=float(config.get("position_scale_m", 0.05)),
    )
    base_observer = LiberoObservationAdapter(
        max_eef_z_m=workspace[1], min_eef_z_m=workspace[0],
        safe_lift_step_m=CONTROL_TICK_MM / 1000.0,
    )
    holds = []
    reset = True
    for hold_index in range(PRE_SETTLE_TICKS):
        outcome = run_v3_tick(
            environment, base_observer, controller, task_id=f"{SUITE}:{TASK_ID}",
            token=None, direction_unit=None, commanded_step_m=CONTROL_TICK_MM / 1000.0,
            reset=reset, workspace_z_bounds_m=workspace,
        )
        reset = False
        if outcome.get("actions") != 1 or not outcome.get("backend_execution"):
            environment.close()
            raise RuntimeError(f"RobotReady HOLD {hold_index + 1} failed: {outcome}")
        holds.append(outcome)
    observer = ObjectRelativePerceptionObserver(
        base_observer, sam3, target_phrase=TARGET_PHRASE,
        move_vectors=config["move_vectors"], canonical_image_adapter=CanonicalImageAdapter(),
        scene_ready_required=True, alignment_scales_m=scales, scale_contracts=contracts,
    )
    scene_ready = run_scene_ready_holds(
        environment, observer, controller, task_id=f"{SUITE}:{TASK_ID}",
        commanded_step_m=CONTROL_TICK_MM / 1000.0,
        workspace_z_bounds_m=workspace, max_hold_ticks=40,
    )
    if not scene_ready.get("ready") or not observer.scene_ready:
        environment.close()
        raise RuntimeError(f"SceneReady failed: {scene_ready}")
    trigger = next((int(row["environment_step"]) for row in scene_ready.get("samples", [])
                    if row.get("scene_ready")), None)
    if trigger is None:
        environment.close()
        raise RuntimeError("SceneReady evidence omitted its trigger tick")
    return environment, controller, base_observer, observer, holds, scene_ready, trigger


def _initial_state(observer, environment, task_id: str):
    builder = StateBuilder()
    state = builder.initialize(task_id)
    observation = observer.observe(environment)
    return builder, builder.update(state, observation)


def _boundary_invalid(state: BeliefState, direction: str, scale_m: float,
                      move_vectors: Mapping[str, Sequence[float]]) -> bool:
    position = state.end_effector_state.get("position_xyz") if state.end_effector_state else None
    bounds = state.relevant_geometry.get("workspace_z_bounds_m")
    if position is None or bounds is None:
        return True
    unit = move_vectors[f"MV_{direction}"]
    endpoint_z = float(position[2]) + float(unit[2]) * scale_m
    return endpoint_z < float(bounds[0]) - 1e-12 or endpoint_z > float(bounds[1]) + 1e-12


def run_calibration_trial(*, init_state: int, direction: str, scale_m: float,
                          run_dir: Path, config: Mapping[str, Any], sam3: Any,
                          workspace: tuple[float, float], camera_resolution: int) -> dict[str, Any]:
    trial_dir = run_dir / f"stage_a/init_state_{init_state}/{direction}_{int(scale_m * 1000)}mm"
    trial_dir.mkdir(parents=True, exist_ok=False)
    max_ticks = TICK_BUDGETS[scale_m]
    environment = None
    record: dict[str, Any] = {
        "init_state_index": init_state, "direction": direction,
        "requested_displacement_m": scale_m, "requested_mm": scale_m * 1000,
        "max_ticks": max_ticks, "control_tick_mm": CONTROL_TICK_MM,
        "status": "NOT_RUN", "termination": None, "oracle_used_by_runtime": False,
    }
    try:
        environment, controller, base, observer, holds, scene, scene_trigger = _setup_trial(
            init_state=init_state, config=config, sam3=sam3, workspace=workspace,
            camera_resolution=camera_resolution,
        )
        _, start_state = _initial_state(observer, environment, f"{SUITE}:{TASK_ID}")
        record.update({"robot_ready_hold_ticks": len(holds), "scene_ready": scene,
                       "scene_ready_trigger_environment_tick": scene_trigger,
                       "start_frame_id": start_state.frame_id,
                       "start_eef_xyz_m": (start_state.end_effector_state or {}).get("position_xyz")})
        if _boundary_invalid(start_state, direction, scale_m, config["move_vectors"]):
            record.update({"status": "BOUNDARY_SKIP", "termination": "BOUNDARY_SKIP",
                           "approval_calls": 0, "ticks_executed": 0})
        else:
            events = []
            arbiter = ExperimentArbiter()
            executor = Executor(LiberoPrimitiveBackend(environment, controller, arbiter), arbiter)
            runner = RuntimeV3Runner(
                observer=observer, state_builder=StateBuilder(),
                option_generator=CalibrationOptionGenerator(
                    direction, scale_m, max_ticks, config["move_vectors"]),
                selector=MultiscaleSelector(), arbiter=arbiter, executor=executor,
                effect_observer=EffectObserver(), logger=events.append,
            )
            result = runner.run_episode(environment, task_id=f"{SUITE}:{TASK_ID}",
                                        max_steps=1, reset=False)
            execution = getattr(events[0].get("execution"), "result", None) if events else None
            record.update({"status": "EXECUTED" if execution else "FAILED",
                           "runner_status": result.get("status"),
                           "approval_calls": arbiter.authorization_calls,
                           "approved_actions": arbiter.approval_count,
                           "execution": execution,
                           "termination": execution.get("termination") if execution else result.get("status"),
                           "ticks_executed": execution.get("ticks_executed", 0) if execution else 0})
            if execution:
                record.update({
                    "actual_target_axis_projection_mm": execution["actual_projection_mm"],
                    "absolute_displacement_error_mm": execution["absolute_displacement_error_mm"],
                    "overshoot_mm": max(0.0, execution["actual_projection_mm"] - scale_m * 1000),
                    "off_axis_magnitude_mm": execution["off_axis_magnitude_mm"],
                    "direction_cosine": execution["direction_cosine"],
                })
    except Exception as exc:
        record.update({"status": "TRIAL_FAILED", "termination": "TRIAL_FAILED",
                       "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if environment is not None:
            environment.close()
    write_json(trial_dir / "trial.json", record)
    return record


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None
              and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def summarize_calibration(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scale: dict[str, Any] = {}
    by_scale_direction: dict[str, Any] = {}
    for scale in SCALES_M:
        rows = [row for row in trials if math.isclose(float(row["requested_displacement_m"]), scale)]
        executed = [row for row in rows if row.get("status") == "EXECUTED"]
        reached = [row for row in executed if row.get("termination") == "TARGET_REACHED"]
        by_scale[f"{int(scale * 1000)}mm"] = {
            "n": len(rows), "executed_n": len(executed),
            "boundary_skips": sum(row.get("status") == "BOUNDARY_SKIP" for row in rows),
            "target_reached_n": len(reached),
            "target_reached_rate": len(reached) / len(executed) if executed else None,
            "mean_absolute_error_mm": _mean(executed, "absolute_displacement_error_mm"),
            "mean_overshoot_mm": _mean(executed, "overshoot_mm"),
            "max_overshoot_mm": max((float(row["overshoot_mm"]) for row in executed
                                      if row.get("overshoot_mm") is not None), default=None),
            "mean_off_axis_mm": _mean(executed, "off_axis_magnitude_mm"),
            "mean_direction_cosine": _mean(executed, "direction_cosine"),
            "mean_ticks": _mean(executed, "ticks_executed"),
        }
        for direction in DIRECTION_ORDER:
            cell = [row for row in rows if row["direction"] == direction]
            cell_exec = [row for row in cell if row.get("status") == "EXECUTED"]
            cell_reached = [row for row in cell_exec if row.get("termination") == "TARGET_REACHED"]
            by_scale_direction[f"{int(scale * 1000)}mm/{direction}"] = {
                "n": len(cell), "executed_n": len(cell_exec),
                "target_reached_rate": len(cell_reached) / len(cell_exec) if cell_exec else None,
                "mean_absolute_error_mm": _mean(cell_exec, "absolute_displacement_error_mm"),
                "mean_overshoot_mm": _mean(cell_exec, "overshoot_mm"),
                "mean_off_axis_mm": _mean(cell_exec, "off_axis_magnitude_mm"),
                "mean_direction_cosine": _mean(cell_exec, "direction_cosine"),
                "mean_ticks": _mean(cell_exec, "ticks_executed"),
            }
    base = by_scale["3mm"]
    decisions = {}
    for scale in SCALES_M:
        label = f"{int(scale * 1000)}mm"
        row = by_scale[label]
        reached_ok = row["target_reached_rate"] == 1.0 and row["executed_n"] > 0
        max_overshoot_limit = max(2.0, float(base["max_overshoot_mm"] or 0.0) + 2.0)
        off_axis_limit = float(base["mean_off_axis_mm"] or 0.0) + 1.0
        cosine_floor = float(base["mean_direction_cosine"] or 0.0) - 0.05
        checks = {
            "all_executed_trials_reached_requested_projection": reached_ok,
            "max_overshoot_within_baseline_plus_2mm": (
                row["max_overshoot_mm"] is not None
                and row["max_overshoot_mm"] <= max_overshoot_limit),
            "mean_off_axis_within_baseline_plus_1mm": (
                row["mean_off_axis_mm"] is not None
                and row["mean_off_axis_mm"] <= off_axis_limit),
            "mean_direction_cosine_within_0_05_of_baseline": (
                row["mean_direction_cosine"] is not None
                and row["mean_direction_cosine"] >= cosine_floor),
        }
        decisions[label] = {
            "status": "VERIFIED" if all(checks.values()) else "NOT VERIFIED",
            "checks": checks,
            "thresholds": {"max_overshoot_mm": max_overshoot_limit,
                           "mean_off_axis_mm": off_axis_limit,
                           "mean_direction_cosine": cosine_floor},
            "max_ticks": TICK_BUDGETS[scale],
        }
    return {"by_scale": by_scale, "by_scale_direction": by_scale_direction,
            "scale_decisions": decisions,
            "trial_count": len(trials), "expected_trial_count": 54,
            "all_trials_recorded": len(trials) == 54}


def candidate_lattice_records(candidates: Sequence[Mapping[str, Any]],
                              selected: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    selected_key = ((str(selected.get("direction")), float(selected.get("displacement_m")))
                    if selected else None)
    records = []
    for item in candidates:
        try:
            key = (str(item.get("direction")), float(item.get("displacement_m")))
        except (TypeError, ValueError):
            key = None
        records.append({
            "direction": item.get("direction"), "scale_m": item.get("displacement_m"),
            "scale_mm": item.get("displacement_mm"),
            "predicted_pixel": item.get("predicted_projection_px"),
            "predicted_error_px": item.get("predicted_error_px"),
            "predicted_improvement_px": item.get("predicted_improvement_px"),
            "workspace_valid": bool(item.get("workspace_valid")),
            "contract_valid": bool(item.get("scale_contract_valid")),
            "valid": bool(item.get("valid")), "reason": item.get("reason"),
            "max_ticks": item.get("max_ticks"), "selected": key == selected_key,
        })
    return records


def _state_stop_reason(state: BeliefState) -> str | None:
    relative, geometry = state.object_relative_state, state.relevant_geometry
    if relative is None or not relative.target_reference_valid:
        return "REFERENCE_INVALID"
    if not relative.target_visible:
        if relative.target_identity_status == "TARGET_IDENTITY_LOST":
            return "TARGET_IDENTITY_LOST"
        return "TARGET_VISIBILITY_LOST"
    if relative.target_identity_status not in {"ANCHORED", "SAME_TARGET"}:
        return "TARGET_IDENTITY_LOST"
    lattice = geometry.get("candidate_lattice", [])
    if not any(bool(item.get("valid")) and float(item.get("predicted_improvement_px", 0)) > 0
               for item in lattice if isinstance(item, Mapping)):
        return "NO_POSITIVE_OPTION"
    return None


def _post_action_stop(step_index: int, actual_improvement: float | None,
                      state: BeliefState) -> str | None:
    reason = _state_stop_reason(state)
    if reason in {"REFERENCE_INVALID", "TARGET_IDENTITY_LOST", "TARGET_VISIBILITY_LOST"}:
        return reason
    if actual_improvement is None or actual_improvement <= 0:
        return "EFFECT_NOT_IMPROVED"
    if reason == "NO_POSITIVE_OPTION":
        return reason
    return "MAX_ALIGNMENT_STEPS" if step_index >= MAX_ALIGNMENT_STEPS else None


def run_stage_b_episode(*, init_state: int, run_dir: Path, config: Mapping[str, Any],
                        sam3: Any, workspace: tuple[float, float], camera_resolution: int,
                        verified_scales: Sequence[float], contracts: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    episode_dir = run_dir / f"stage_b/init_state_{init_state}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir()
    environment = None
    try:
        environment, controller, base, observer, holds, scene, scene_trigger = _setup_trial(
            init_state=init_state, config=config, sam3=sam3, workspace=workspace,
            camera_resolution=camera_resolution, scales=verified_scales, contracts=contracts,
        )
        oracle_start = oracle_target_world_position(environment)
        image_panels: list[tuple[str, Path]] = []
        steps: list[dict[str, Any]] = []
        active = {"step": 0}

        def before_authorize(state, options, selection):
            step_index = active["step"]
            frame = observer.perception_history[-1]
            option = next((row for row in options if row.option_id == selection.option_id), None)
            if option is None or option.primitive.micro_motion_spec is None:
                raise RuntimeError("Runtime did not generate a bounded ALIGN realization")
            relative = state.object_relative_state
            if (relative is None or not relative.target_reference_valid
                    or not relative.target_visible or not observer.scene_ready):
                raise RuntimeError("pre-action target reference/SceneReady contract is invalid")
            spec = option.primitive.micro_motion_spec
            visual = make_step_visual(frame, steps_dir / f"step_{step_index:02d}",
                                      prefix="before", selected_direction=spec.direction)
            lattice = candidate_lattice_records(
                state.relevant_geometry.get("candidate_lattice", []),
                state.relevant_geometry.get("chosen_lattice_candidate"),
            )
            record = {
                "status": "PRE_ACTION_READY", "semantic_option_id": SEMANTIC_OPTION_ID,
                "alignment_step": step_index, "frame_id": state.frame_id,
                "selected_direction": spec.direction,
                "selected_scale_m": spec.requested_displacement_m,
                "selected_max_ticks": spec.max_ticks,
                "candidate_lattice": lattice,
                "target_reference_px": relative.target_reference_point_px,
                "scene_ready": observer.scene_ready,
                "visual_artifacts": visual, "oracle_used_by_runtime": False,
            }
            ready_path = steps_dir / f"step_{step_index:02d}/PRE_ACTION_READY.json"
            write_json(ready_path, record)
            if json.loads(ready_path.read_text(encoding="utf-8")).get("status") != "PRE_ACTION_READY":
                raise RuntimeError("PRE_ACTION_READY read-back failed")
            if step_index == 1:
                image_panels.append((f"step 0: initial, before {spec.direction} {spec.requested_displacement_m*1000:g}mm",
                                     Path(visual["alignment_overlay"])))

        arbiter = ExperimentArbiter(before_authorize)
        executor = Executor(LiberoPrimitiveBackend(environment, controller, arbiter), arbiter)
        initial_error = final_error = None
        reference_initial = None
        termination = "MAX_ALIGNMENT_STEPS"
        for step_index in range(1, MAX_ALIGNMENT_STEPS + 1):
            active["step"] = step_index
            auth_before = arbiter.authorization_calls
            align_auth_before = arbiter.alignment_authorization_calls
            approval_before = arbiter.approval_count
            events = []
            runner = RuntimeV3Runner(
                observer=observer, state_builder=StateBuilder(),
                option_generator=MultiScaleAlignmentOptionGenerator(),
                selector=MultiscaleSelector(), arbiter=arbiter, executor=executor,
                effect_observer=EffectObserver(), logger=events.append,
            )
            result = runner.run_episode(environment, task_id=f"{SUITE}:{TASK_ID}",
                                        max_steps=1, reset=False)
            if not events:
                state = result.get("state") or runner.state
                termination = (_state_stop_reason(state) if state is not None else None) or result.get("status", "EXECUTION_FAILED")
                frame = observer.perception_history[-1] if observer.perception_history else None
                lattice = candidate_lattice_records(
                    state.relevant_geometry.get("candidate_lattice", []) if state else [])
                row = {"alignment_step": step_index, "executed": False,
                       "termination_reason": termination,
                       "candidate_lattice": lattice,
                       "authorization_calls_for_step": arbiter.authorization_calls - auth_before,
                       "approvals_for_step": arbiter.approval_count - approval_before}
                steps.append(row)
                write_json(steps_dir / f"step_{step_index:02d}/step.json", row)
                if frame is not None and not image_panels:
                    visual = make_step_visual(frame, steps_dir / f"step_{step_index:02d}", prefix="stop")
                    image_panels.append((f"step 0: stop {termination}", Path(visual["alignment_overlay"])))
                break

            event = events[0]
            state_before, state_after = event["state_before"], event["state_after"]
            action = event["approved_action"]
            spec = action.primitive.micro_motion_spec
            before_frame, after_frame = observer.perception_history[-2:]
            reference = _state_reference(state_before)
            if reference_initial is None:
                reference_initial = reference
            error_before, error_after = error_from_state(state_before), error_from_state(state_after)
            final_error = error_after if error_after is not None else final_error
            if initial_error is None:
                initial_error = error_before
            same_reference = same_fixed_reference(reference, _state_reference(state_after))
            actual_improvement = (error_before - error_after
                                  if error_before is not None and error_after is not None
                                  and same_reference else None)
            prediction = compare_alignment_improvements(
                predicted_improvement_px=action.expected_effect.get("predicted_improvement_px"),
                actual_improvement_px=actual_improvement,
            )
            selected = state_before.relevant_geometry.get("chosen_lattice_candidate")
            lattice = candidate_lattice_records(
                state_before.relevant_geometry.get("candidate_lattice", []), selected)
            frame_dir = steps_dir / f"step_{step_index:02d}"
            before_visual = make_step_visual(before_frame, frame_dir, prefix="before",
                                             selected_direction=spec.direction)
            after_visual = make_step_visual(after_frame, frame_dir, prefix="after")
            image_panels.append((f"step {step_index}: after {spec.direction} {spec.requested_displacement_m*1000:g}mm",
                                 Path(after_visual["alignment_overlay"])))
            execution = getattr(event.get("execution"), "result", None)
            relative_after = state_after.object_relative_state
            row = {
                "alignment_step": step_index, "executed": True,
                "semantic_option_id": action.option_id,
                "direction": spec.direction, "scale_m": spec.requested_displacement_m,
                "scale_mm": spec.requested_displacement_m * 1000.0,
                "max_ticks": spec.max_ticks,
                "frame_id_before": state_before.frame_id, "frame_id_after": state_after.frame_id,
                "error_before_px": error_before, "predicted_error_after_px": action.expected_effect.get("predicted_image_error_after_px"),
                "predicted_improvement_px": action.expected_effect.get("predicted_improvement_px"),
                "actual_improvement_px": actual_improvement, "error_after_px": error_after,
                "prediction_residual_actual_minus_predicted_px": prediction["prediction_residual_actual_minus_predicted_px"],
                "candidate_lattice": lattice,
                "target_reference_before_px": reference,
                "target_reference_after_px": _state_reference(state_after),
                "target_reference_unchanged": same_reference,
                "reference_matches_episode_initial": same_fixed_reference(reference_initial, reference),
                "target_identity_status_after": relative_after.target_identity_status if relative_after else None,
                "target_visible_after": bool(relative_after and relative_after.target_visible),
                "target_reference_valid_after": bool(relative_after and relative_after.target_reference_valid),
                "authorization_calls_for_step": arbiter.authorization_calls - auth_before,
                "alignment_authorization_calls_for_step": arbiter.alignment_authorization_calls - align_auth_before,
                "arbiter_approved_alignment_actions_for_step": arbiter.approval_count - approval_before,
                "executor_calls_for_action": 1,
                "execution": execution,
                "visual_artifacts": {"before": before_visual, "after": after_visual},
                "oracle_target_pose_after_diagnostic": oracle_target_world_position(environment),
                "oracle_used_by_runtime": False,
            }
            steps.append(row)
            write_json(frame_dir / "step.json", row)
            termination_after = _post_action_stop(step_index, actual_improvement, state_after)
            if termination_after:
                termination = termination_after
                break

        executed = [step for step in steps if step.get("executed")]
        trajectory = ([executed[0].get("error_before_px"),
                       *(step.get("error_after_px") for step in executed)] if executed else [])
        directions = [str(step["direction"]) for step in executed]
        scales = [float(step["scale_mm"]) for step in executed]
        improvements = [float(step["actual_improvement_px"]) for step in executed
                       if step.get("actual_improvement_px") is not None]
        tick_count = sum(int((step.get("execution") or {}).get("ticks_executed", 0)) for step in executed)
        physical_mm = sum(float(np.linalg.norm((step.get("execution") or {}).get(
            "total_displacement_xyz_mm", [0.0, 0.0, 0.0]))) for step in executed)
        oracle_end = oracle_target_world_position(environment)
        oracle_diag = _oracle_motion(oracle_start, oracle_end)
        episode = {
            "init_state_index": init_state, "status": "COMPLETED", "suite": SUITE,
            "task_id": TASK_ID, "seed": 0, "task_instruction": environment.task_description,
            "robot_ready_hold_ticks": len(holds), "scene_ready_initialization": scene,
            "scene_ready_trigger_environment_tick": scene_trigger,
            "verified_scales_mm": [scale * 1000 for scale in verified_scales],
            "max_alignment_steps": MAX_ALIGNMENT_STEPS,
            "initial_error_px": initial_error, "final_error_px": final_error,
            "cumulative_improvement_px": (initial_error - final_error
                                            if initial_error is not None and final_error is not None else None),
            "alignment_steps": steps, "executed_alignment_steps": len(executed),
            "control_ticks": tick_count, "total_physical_displacement_mm": physical_mm,
            "termination_reason": termination,
            "all_steps_positive": bool(executed and len(improvements) == len(executed)
                                        and all(value > 0 for value in improvements)),
            "step_verification_rate": (sum(value > 0 for value in improvements) / len(executed)
                                       if executed else None),
            "direction_sequence": directions, "scale_sequence_mm": scales,
            "direction_transitions": sum(a != b for a, b in zip(directions, directions[1:])),
            "direction_reversal_count": count_direction_reversals(directions),
            "error_trajectory_px": trajectory,
            "normalized_error_trajectory": normalized_trajectory(trajectory),
            "mean_improvement_per_step_px": float(np.mean(improvements)) if improvements else None,
            "mean_improvement_per_control_tick_px": (sum(improvements) / tick_count if tick_count else None),
            "mean_prediction_residual_px": float(np.mean([
                step["prediction_residual_actual_minus_predicted_px"] for step in executed
                if step.get("prediction_residual_actual_minus_predicted_px") is not None])) if executed else None,
            "target_reference_fixed_for_episode": bool(executed) and all(
                row.get("target_reference_unchanged") and row.get("reference_matches_episode_initial")
                for row in executed),
            "arbiter_authorization_calls": arbiter.authorization_calls,
            "arbiter_alignment_authorization_calls": arbiter.alignment_authorization_calls,
            "arbiter_approved_alignment_actions": arbiter.approval_count,
            "one_approval_per_executed_step": arbiter.approval_count == len(executed),
            "oracle_target_motion_diagnostic": oracle_diag,
            "oracle_used_by_runtime": False, "qwen_actions": 0,
            "contact_sheet": save_contact_sheet(image_panels, episode_dir / "contact_sheet.png"),
            "legacy_modified": False,
        }
        write_json(episode_dir / "episode.json", episode)
        return episode
    finally:
        if environment is not None:
            environment.close()


def _mean_normalized(episodes: Sequence[Mapping[str, Any]]) -> list[float | None]:
    series = [episode.get("normalized_error_trajectory", []) for episode in episodes]
    length = max((len(row) for row in series), default=0)
    return [float(np.mean(values)) if (values := [float(row[index]) for row in series
                if index < len(row) and row[index] is not None]) else None for index in range(length)]


def save_comparison_plot(baseline: Sequence[Mapping[str, Any]],
                         multiscale: Sequence[Mapping[str, Any]], path: Path) -> None:
    width, height, margin = 900, 500, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (margin, margin, width - 25, height - margin)
    draw.line((box[0], box[1], box[0], box[3]), fill="black", width=2)
    draw.line((box[0], box[3], box[2], box[3]), fill="black", width=2)
    draw.text((margin, 18), "Mean normalized frozen-reference error", fill="black")
    base, multi = _mean_normalized(baseline), _mean_normalized(multiscale)
    max_x = max(len(base), len(multi), 2) - 1
    def points(values):
        return [(box[0] + int((box[2] - box[0]) * i / max_x),
                 box[3] - int((box[3] - box[1]) * max(0.0, min(1.5, v)) / 1.5))
                for i, v in enumerate(values) if v is not None]
    for values, color in ((base, (30, 100, 210)), (multi, (220, 60, 50))):
        pts = points(values)
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
    draw.text((box[0] + 8, box[1] + 8), "blue: fixed 3mm; red: multi-scale", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_scale_histogram(episodes: Sequence[Mapping[str, Any]], path: Path) -> dict[str, int]:
    counts = {"3mm": 0, "6mm": 0, "9mm": 0}
    for episode in episodes:
        for value in episode.get("scale_sequence_mm", []):
            label = f"{int(round(float(value)))}mm"
            if label in counts:
                counts[label] += 1
    width, height = 600, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Selected physical scale count", fill="black")
    maximum = max(counts.values(), default=1) or 1
    for index, (label, value) in enumerate(counts.items()):
        y = 80 + index * 82
        draw.text((30, y + 12), label, fill="black")
        draw.rectangle((110, y, 110 + int(400 * value / maximum), y + 45), fill=(45, 120, 190))
        draw.text((520, y + 12), str(value), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return counts


def summarize_stage_b(episodes: Sequence[Mapping[str, Any]], run_dir: Path,
                      baseline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [step for episode in episodes for step in episode.get("alignment_steps", [])
             if step.get("executed")]
    improvements = [float(step["actual_improvement_px"]) for step in steps
                    if step.get("actual_improvement_px") is not None]
    residuals = [float(step["prediction_residual_actual_minus_predicted_px"]) for step in steps
                 if step.get("prediction_residual_actual_minus_predicted_px") is not None]
    ticks = sum(int(episode.get("control_ticks", 0)) for episode in episodes)
    baseline_steps = [step for episode in baseline for step in episode.get("alignment_steps", [])
                      if step.get("executed")]
    baseline_improvements = [float(step["actual_improvement_px"]) for step in baseline_steps
                             if step.get("actual_improvement_px") is not None]
    baseline_ticks = sum(int((step.get("execution") or {}).get("ticks_executed", 0))
                         for step in baseline_steps)
    scales = save_scale_histogram(episodes, run_dir / "scale_selection_histogram.png")
    save_comparison_plot(baseline, episodes, run_dir / "baseline_vs_multiscale_normalized_error.png")
    transitions = sum(int(row.get("direction_transitions", 0)) for row in episodes)
    reversals = sum(int(row.get("direction_reversal_count", 0)) for row in episodes)
    directions = {direction: sum(step.get("direction") == direction for step in steps)
                  for direction in DIRECTION_ORDER}
    def avg(rows, key):
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    candidate_count = sum(len(step.get("candidate_lattice", [])) for step in steps)
    scale_counts_by_step: dict[str, dict[str, int]] = {}
    for episode in episodes:
        for step in episode.get("alignment_steps", []):
            if not step.get("executed"):
                continue
            step_label = str(step["alignment_step"])
            bucket = scale_counts_by_step.setdefault(step_label, {"3mm": 0, "6mm": 0, "9mm": 0})
            label = f"{int(round(float(step['scale_mm'])))}mm"
            if label in bucket:
                bucket[label] += 1
    return {
        "episodes": len(episodes), "executed_semantic_steps": len(steps),
        "control_ticks": ticks,
        "positive_effect_steps": sum(value > 0 for value in improvements),
        "step_verification_rate": (sum(value > 0 for value in improvements) / len(steps) if steps else None),
        "episode_monotonic_rate": (sum(bool(row.get("all_steps_positive")) for row in episodes) / len(episodes)
                                    if episodes else None),
        "mean_improvement_per_step_px": float(np.mean(improvements)) if improvements else None,
        "mean_prediction_residual_px": float(np.mean(residuals)) if residuals else None,
        "mean_improvement_per_control_tick_px": sum(improvements) / ticks if ticks else None,
        "mean_cumulative_improvement_px": avg(episodes, "cumulative_improvement_px"),
        "mean_initial_error_px": avg(episodes, "initial_error_px"),
        "mean_final_error_px": avg(episodes, "final_error_px"),
        "mean_semantic_steps_per_episode": avg(episodes, "executed_alignment_steps"),
        "mean_control_ticks_per_episode": avg(episodes, "control_ticks"),
        "mean_physical_displacement_mm_per_episode": avg(episodes, "total_physical_displacement_mm"),
        "scale_selection_count": scales, "scale_selection_by_semantic_step": scale_counts_by_step,
        "direction_counts": directions,
        "direction_transitions": transitions, "direction_reversals": reversals,
        "candidate_lattice_entries_logged": candidate_count,
        "all_lattice_entries_logged": all(len(step.get("candidate_lattice", [])) ==
            6 * len(episode.get("verified_scales_mm", []))
            for episode in episodes for step in episode.get("alignment_steps", [])
            if step.get("executed")) if episodes else True,
        "arbiter_approvals": sum(int(row.get("arbiter_approved_alignment_actions", 0)) for row in episodes),
        "one_approval_per_executed_step": all(row.get("one_approval_per_executed_step") for row in episodes),
        "qwen_actions": 0, "oracle_used_by_runtime": False,
        "target_moved_diagnostic_episodes": sum(bool(row.get("oracle_target_motion_diagnostic", {}).get(
            "target_moved_diagnostic")) for row in episodes),
        "baseline": {
            "mean_initial_error_px": avg(baseline, "initial_error_px"),
            "mean_final_error_px": avg(baseline, "final_error_px"),
            "mean_cumulative_improvement_px": avg(baseline, "cumulative_improvement_px"),
            "mean_improvement_per_step_px": (float(np.mean(baseline_improvements))
                                              if baseline_improvements else None),
            "mean_improvement_per_control_tick_px": (sum(baseline_improvements) / baseline_ticks
                                                       if baseline_ticks else None),
            "episode_monotonic_rate": avg(baseline, "all_steps_positive"),
            "step_verification_rate": (sum(value > 0 for value in baseline_improvements)
                                       / len(baseline_steps) if baseline_steps else None),
            "control_ticks": baseline_ticks,
        },
        "normalized_error_curve": {"baseline": _mean_normalized(baseline),
                                    "multi_scale": _mean_normalized(episodes)},
    }


def _configure_proxy(url: str) -> None:
    if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        return
    names = ("NO_PROXY", "no_proxy")
    entries = [entry.strip() for name in names for entry in os.environ.get(name, "").split(",") if entry.strip()]
    value = ",".join(dict.fromkeys((*entries, "127.0.0.1", "localhost", "::1")))
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def main() -> int:
    from core.capabilities.sam3_client import Sam3Client
    from core.config import load_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/runtime_v3_multiscale_alignment"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    args = parser.parse_args()
    if args.camera_resolution != 512:
        raise SystemExit("M3.3 is frozen to the established direct 512x512 RGB path")
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    _configure_proxy(args.sam3_url)
    run_dir = new_run_dir(args.output_dir)
    workspace = (0.02, 0.60)
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    trials = []
    episodes = []
    blockers = []
    try:
        for init_state in STAGE_A_STATES:
            for scale in SCALES_M:
                for direction in DIRECTION_ORDER:
                    try:
                        trials.append(run_calibration_trial(
                            init_state=init_state, direction=direction, scale_m=scale,
                            run_dir=run_dir, config=config, sam3=sam3,
                            workspace=workspace, camera_resolution=args.camera_resolution,
                        ))
                    except Exception as exc:
                        blockers.append(f"stage_a {init_state}/{direction}/{scale}: {type(exc).__name__}: {exc}")
                        trials.append({"init_state_index": init_state, "direction": direction,
                                       "requested_displacement_m": scale, "status": "TRIAL_FAILED",
                                       "termination": "TRIAL_FAILED", "error": str(exc)})
        calibration = summarize_calibration(trials)
        write_json(run_dir / "stage_a/calibration_summary.json", calibration)
        if len(trials) != 54 or not calibration["all_trials_recorded"]:
            blockers.append("Stage A did not produce all 54 independently initialized trial records")
        verified = [scale for scale in SCALES_M
                    if calibration["scale_decisions"][f"{int(scale*1000)}mm"]["status"] == "VERIFIED"]
        if 0.003 not in verified:
            blockers.append("Stage A did not verify 3mm; Stage B gate remains closed")
        else:
            contracts = {scale: {"verified": True, "max_ticks": TICK_BUDGETS[scale]}
                         for scale in verified}
            for init_state in INIT_STATES:
                try:
                    episodes.append(run_stage_b_episode(
                        init_state=init_state, run_dir=run_dir, config=config, sam3=sam3,
                        workspace=workspace, camera_resolution=args.camera_resolution,
                        verified_scales=verified, contracts=contracts,
                    ))
                except Exception as exc:
                    blockers.append(f"stage_b init_state_{init_state}: {type(exc).__name__}: {exc}")
                    failure = {"init_state_index": init_state, "status": "FAILED",
                               "error": f"{type(exc).__name__}: {exc}", "executed_alignment_steps": 0}
                    episodes.append(failure)
                    write_json(run_dir / f"stage_b/init_state_{init_state}/failure.json", failure)
    finally:
        sam3.close()

    try:
        baseline_record = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
        baseline_episodes = baseline_record["episodes"]
    except Exception as exc:
        baseline_episodes = []
        blockers.append(f"M3.2 baseline unavailable: {type(exc).__name__}: {exc}")
    stage_b_metrics = summarize_stage_b(episodes, run_dir, baseline_episodes) if baseline_episodes else {}
    baseline_final = stage_b_metrics.get("baseline", {}).get("mean_final_error_px")
    multi_final = stage_b_metrics.get("mean_final_error_px")
    baseline_step = stage_b_metrics.get("baseline", {}).get("mean_improvement_per_step_px")
    multi_step = stage_b_metrics.get("mean_improvement_per_step_px")
    faster = bool(baseline_final is not None and multi_final is not None
                  and multi_final < baseline_final and baseline_step is not None
                  and multi_step is not None and multi_step > baseline_step)
    stage_a_ok = len(trials) == 54 and 0.003 in verified
    stage_b_ok = len(episodes) == len(INIT_STATES) and all(row.get("status") == "COMPLETED" for row in episodes)
    record = {
        "status": "COMPLETED" if stage_a_ok and stage_b_ok else "PARTIAL",
        "phase": "M3.3 Multi-scale Verified Object-relative Options",
        "branch": "runtime-v3", "starting_commit": "b0c794e385544b407ef72ed90483ed03f125dcf2",
        "suite": SUITE, "task_id": TASK_ID, "seed": 0,
        "stage_a": {"states": list(STAGE_A_STATES), "directions": list(DIRECTION_ORDER),
                    "scales_mm": [scale * 1000 for scale in SCALES_M],
                    "tick_budgets": {f"{int(scale*1000)}mm": TICK_BUDGETS[scale] for scale in SCALES_M},
                    "control_tick_mm": CONTROL_TICK_MM, "trials": trials,
                    "metrics": calibration, "verified_scales_mm": [scale * 1000 for scale in verified]},
        "stage_b": {"states": list(INIT_STATES), "max_semantic_steps_per_episode": MAX_ALIGNMENT_STEPS,
                    "verified_scales_mm": [scale * 1000 for scale in verified],
                    "episodes": episodes, "metrics": stage_b_metrics},
        "efficiency_judgment": {"multi_scale_reduces_error_faster_under_same_step_budget": faster,
                                "criterion": "lower mean final error and higher mean actual improvement per semantic step than M3.2"},
        "blockers": blockers, "qwen_actions": 0, "oracle_used_by_runtime": False,
        "legacy_modified": False,
    }
    write_json(run_dir / "summary.json", record)
    print(f"M3.3 {record['status']}: {run_dir / 'summary.json'}")
    return 0 if record["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
