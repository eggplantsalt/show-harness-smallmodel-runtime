#!/usr/bin/env python3
"""M3.5: audit MuJoCo depth as experiment-only observability evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_v3.canonical_image import CanonicalImageAdapter
from core.runtime_v3.depth import MoGeMetricDepthProvider
from core.runtime_v3.state import StateBuilder
from core.runtime_v3.temporal_calibration import run_v3_tick
from scripts import runtime_v3_near_target_observability as m34
from scripts.runtime_v3_metric_entity_depth_diagnostic import (
    DiagnosticSimulatorDepthProvider,
    MetricEntityReference,
    classify_metric_reference_visibility,
    freeze_metric_reference,
    invalid_metric_entity_reference,
    metric_entity_reference_from_rgbd,
    metric_proximity_distance_m,
    mujoco_depth_clip_planes_m,
    normalized_depth_to_metric,
)


HOLD_OBSERVATIONS = 5
STATIC_FRAME_JITTER_LIMIT_M = 0.003  # Existing smallest calibrated motion scale.
CSV_FIELDS = (
    "init_state_index", "sample_index", "sample_role", "frame_id",
    "candidate_valid", "candidate_invalid_reason", "depth_source",
    "candidate_reference_world_x_m", "candidate_reference_world_y_m",
    "candidate_reference_world_z_m", "frozen_reference_world_x_m",
    "frozen_reference_world_y_m", "frozen_reference_world_z_m",
    "valid_depth_count", "mask_pixel_count", "valid_depth_ratio",
    "depth_median_m", "depth_spread_m", "simulator_gt_reference_world_x_m",
    "simulator_gt_reference_world_y_m", "simulator_gt_reference_world_z_m",
    "estimate_vs_simulator_gt_reference_error_m", "depth_mae_m",
    "depth_median_absolute_error_m", "depth_valid_pixel_count",
    "frame_to_frame_displacement_m", "oracle_offset_x_m", "oracle_offset_y_m",
    "oracle_offset_z_m", "oracle_offset_norm_m", "oracle_body_origin_world_x_m",
    "oracle_body_origin_world_y_m", "oracle_body_origin_world_z_m",
    "raw_depth_min", "raw_depth_max", "estimated_depth_min_m", "estimated_depth_max_m",
    "simulator_gt_depth_min_m", "simulator_gt_depth_max_m",
    "oracle_used_by_runtime", "qwen_actions",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value):
        value = asdict(value)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in CSV_FIELDS} for row in rows)


def _reference_record(reference: Any | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "entity_key": reference.entity_key,
        "camera": reference.camera,
        "coordinate_frame": getattr(reference, "coordinate_frame",
                                     getattr(reference, "frame", "world")),
        "reference_world_m": (list(reference.reference_world_m)
                              if reference.reference_world_m is not None else None),
        "valid_depth_count": reference.valid_depth_count,
        "mask_pixel_count": reference.mask_pixel_count,
        "valid_depth_ratio": reference.valid_depth_ratio,
        "depth_median_m": reference.depth_median_m,
        "depth_spread_m": reference.depth_spread_m,
        "depth_source": getattr(reference, "depth_source", "simulator_gt"),
        "source_frame_id": reference.source_frame_id,
        "valid": reference.valid,
        "invalid_reason": reference.invalid_reason,
    }


def _depth_visualization(depth_m: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8), {
            "metric_depth_min_m": None, "metric_depth_max_m": None,
        }
    low, high = (float(value) for value in np.percentile(depth[valid], (2.0, 98.0)))
    span = max(high - low, 1e-6)
    gray = np.clip((depth - low) / span, 0.0, 1.0)
    gray[~valid] = 0.0
    image = np.repeat((gray * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    return image, {
        "metric_depth_min_m": float(np.min(depth[valid])),
        "metric_depth_max_m": float(np.max(depth[valid])),
        "visualization_range_min_m": low,
        "visualization_range_max_m": high,
    }


def _depth_model_environment_audit(depth_provider: MoGeMetricDepthProvider) -> dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        gpu = None
        gpu_memory_gib = None
        if cuda_available:
            device = torch.device(depth_provider.device)
            properties = torch.cuda.get_device_properties(device)
            gpu = str(properties.name)
            gpu_memory_gib = round(float(properties.total_memory) / (1024 ** 3), 2)
    except Exception as exc:
        torch_version = f"unavailable: {type(exc).__name__}: {exc}"
        cuda_available = False
        gpu = None
        gpu_memory_gib = None
    try:
        from importlib.metadata import version

        transformers = version("transformers")
    except Exception:
        transformers = "not installed; not required by MoGe v2 loader"
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "gpu_memory_gib": gpu_memory_gib,
        "transformers": transformers,
        "checkpoint_size_gb": 1.31,
        "checkpoint_size_source": "official Hugging Face checkpoint file listing",
    }


_SIMULATOR_DEPTH_PROVIDER = DiagnosticSimulatorDepthProvider()


def _simulator_depth_candidate(frame: Mapping[str, Any], raw: Mapping[str, Any], environment: Any):
    """Build an experiment-only candidate from this frame's raw simulator depth."""
    segmentation = frame.get("segmentation")
    mask = getattr(segmentation, "mask", None)
    depth_raw = _SIMULATOR_DEPTH_PROVIDER.estimate_normalized_buffer(raw, "agentview")
    calibration = frame.get("calibration")
    model = getattr(getattr(environment, "env", environment), "sim", None)
    model = getattr(model, "model", None)
    try:
        if model is None:
            raise AttributeError("MuJoCo model is unavailable")
        near, far = mujoco_depth_clip_planes_m(model)
        return metric_entity_reference_from_rgbd(
            entity_key=m34.TARGET_PHRASE,
            camera="agentview",
            source_frame_id=frame.get("frame_id", "unknown"),
            target_mask_canonical=mask,
            depth_buffer_raw=depth_raw,
            near_m=near,
            far_m=far,
            calibration=calibration,
            image_adapter=CanonicalImageAdapter("vertical_flip"),
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return (
            invalid_metric_entity_reference(
                entity_key=m34.TARGET_PHRASE,
                camera="agentview",
                source_frame_id=frame.get("frame_id", "unknown"),
                invalid_reason="depth_diagnostic_calibration_unavailable",
                mask_pixel_count=int(np.asarray(mask, dtype=bool).sum()) if mask is not None else 0,
            ),
            None,
        )


def _save_static_sample(
    *,
    output_dir: Path,
    init_state: int,
    sample_index: int,
    sample_role: str,
    frame: Mapping[str, Any],
    base_observer: Any,
    simulator_raw: Mapping[str, Any],
    candidate: MetricEntityReference,
    estimated_depth: np.ndarray | None,
    simulator_gt_candidate: MetricEntityReference,
    simulator_gt_depth: np.ndarray | None,
    frozen_reference: MetricEntityReference | None,
    environment: Any,
    previous_candidate: MetricEntityReference | None,
) -> dict[str, Any]:
    sample_dir = output_dir / f"init_state_{init_state}" / f"frame_{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    raw = base_observer.last_raw
    if raw is None:
        raise RuntimeError("base observer has no fresh source observation")
    canonical_rgb = np.asarray(frame["image"], dtype=np.uint8)
    segmentation = frame["segmentation"]
    mask = (np.asarray(segmentation.mask, dtype=bool) if segmentation.mask is not None
            else np.zeros(canonical_rgb.shape[:2], dtype=bool))
    if estimated_depth is None or simulator_gt_depth is None:
        raise RuntimeError("deployable or simulator diagnostic depth was absent")
    estimated_depth = np.asarray(estimated_depth, dtype=np.float32)
    simulator_gt_depth = np.asarray(simulator_gt_depth, dtype=np.float32)
    if (estimated_depth.shape != mask.shape or simulator_gt_depth.shape != mask.shape
            or canonical_rgb.shape[:2] != mask.shape):
        raise RuntimeError("canonical RGB, SAM mask, estimated depth, and simulator GT are misaligned")

    Image.fromarray(canonical_rgb, mode="RGB").save(sample_dir / "canonical_rgb.png")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(sample_dir / "sam_mask.png")
    estimated_vis, estimated_stats = _depth_visualization(estimated_depth)
    simulator_gt_vis, simulator_gt_stats = _depth_visualization(simulator_gt_depth)
    Image.fromarray(estimated_vis, mode="RGB").save(sample_dir / "estimated_depth_visualization.png")
    Image.fromarray(simulator_gt_vis, mode="RGB").save(
        sample_dir / "simulator_gt_depth_visualization.png")
    overlay = estimated_vis.copy()
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32)
        + 0.45 * np.asarray([255.0, 48.0, 24.0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(sample_dir / "mask_depth_overlay.png")
    gt_overlay = simulator_gt_vis.copy()
    gt_overlay[mask] = (
        0.55 * gt_overlay[mask].astype(np.float32)
        + 0.45 * np.asarray([255.0, 48.0, 24.0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(gt_overlay, mode="RGB").save(sample_dir / "mask_simulator_gt_depth_overlay.png")
    rgb_overlay = canonical_rgb.copy()
    rgb_overlay[mask] = (
        0.55 * rgb_overlay[mask].astype(np.float32)
        + 0.45 * np.asarray([255.0, 48.0, 24.0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(rgb_overlay, mode="RGB").save(sample_dir / "rgb_mask_overlay.png")

    raw_depth_value = _SIMULATOR_DEPTH_PROVIDER.estimate_normalized_buffer(
        simulator_raw, "agentview")
    if raw_depth_value is None:
        raise RuntimeError("experiment diagnostic did not receive agentview simulator depth")
    raw_depth = np.asarray(raw_depth_value, dtype=np.float32).squeeze()
    raw_valid = np.isfinite(raw_depth)
    comparison = (mask & np.isfinite(estimated_depth) & (estimated_depth > 0.0)
                  & np.isfinite(simulator_gt_depth) & (simulator_gt_depth > 0.0))
    abs_error = np.abs(estimated_depth[comparison] - simulator_gt_depth[comparison])
    depth_mae = float(np.mean(abs_error)) if abs_error.size else None
    depth_median_abs_error = float(np.median(abs_error)) if abs_error.size else None
    oracle = m34.target_pose_diagnostic(environment)
    oracle_xyz = (np.asarray(oracle["world_position_xyz_m"], dtype=float)
                  if oracle.get("available") else None)
    candidate_xyz = (np.asarray(candidate.reference_world_m, dtype=float)
                     if candidate is not None and candidate.valid
                     and candidate.reference_world_m is not None else None)
    gt_reference_xyz = (np.asarray(simulator_gt_candidate.reference_world_m, dtype=float)
                        if simulator_gt_candidate.valid
                        and simulator_gt_candidate.reference_world_m is not None else None)
    point_error = (float(np.linalg.norm(candidate_xyz - gt_reference_xyz))
                   if candidate_xyz is not None and gt_reference_xyz is not None else None)
    offset = candidate_xyz - oracle_xyz if candidate_xyz is not None and oracle_xyz is not None else None
    displacement = None
    if (candidate_xyz is not None and previous_candidate is not None
            and previous_candidate.valid and previous_candidate.reference_world_m is not None):
        displacement = float(np.linalg.norm(
            candidate_xyz - np.asarray(previous_candidate.reference_world_m, dtype=float)
        ))
    payload = {
        "init_state_index": init_state,
        "sample_index": sample_index,
        "sample_role": sample_role,
        "frame_id": frame.get("frame_id"),
        "rgb_resolution_hw": list(canonical_rgb.shape[:2]),
        "estimated_depth_resolution_hw": list(estimated_depth.shape),
        "simulator_gt_depth_resolution_hw": list(simulator_gt_depth.shape),
        "sam_mask_resolution_hw": list(mask.shape),
        "canonical_orientation": "vertical_flip",
        "estimated_depth_units": "meters",
        "simulator_gt_depth_units": "meters",
        "raw_depth_min": float(np.min(raw_depth[raw_valid])) if np.any(raw_valid) else None,
        "raw_depth_max": float(np.max(raw_depth[raw_valid])) if np.any(raw_valid) else None,
        "estimated_depth_stats": estimated_stats,
        "simulator_gt_depth_stats": simulator_gt_stats,
        "depth_error_on_mask": {
            "valid_pixel_count": int(abs_error.size),
            "mae_m": depth_mae,
            "median_absolute_error_m": depth_median_abs_error,
            "estimate_vs_simulator_gt_reference_error_m": point_error,
        },
        "estimated_metric_entity_reference": _reference_record(candidate),
        "simulator_gt_depth_reference_diagnostic": _reference_record(simulator_gt_candidate),
        "frozen_reference": _reference_record(frozen_reference),
        "oracle_reference_offset_diagnostic": {
            "available": offset is not None,
            "offset_vector_m": offset.tolist() if offset is not None else None,
            "offset_norm_m": float(np.linalg.norm(offset)) if offset is not None else None,
            "oracle_body_origin_world_m": oracle_xyz.tolist() if oracle_xyz is not None else None,
            "oracle_used_by_runtime": False,
        },
        "visual_artifacts": {
            "canonical_rgb": str(sample_dir / "canonical_rgb.png"),
            "estimated_depth_visualization": str(sample_dir / "estimated_depth_visualization.png"),
            "simulator_gt_depth_visualization": str(sample_dir / "simulator_gt_depth_visualization.png"),
            "sam_mask": str(sample_dir / "sam_mask.png"),
            "mask_depth_overlay": str(sample_dir / "mask_depth_overlay.png"),
            "mask_simulator_gt_depth_overlay": str(sample_dir / "mask_simulator_gt_depth_overlay.png"),
            "rgb_mask_overlay": str(sample_dir / "rgb_mask_overlay.png"),
        },
        "qwen_actions": 0,
    }
    _write_json(sample_dir / "metric_reference.json", payload)
    return {
        "init_state_index": init_state,
        "sample_index": sample_index,
        "sample_role": sample_role,
        "frame_id": frame.get("frame_id"),
        "candidate_valid": bool(candidate and candidate.valid),
        "candidate_invalid_reason": candidate.invalid_reason if candidate else "candidate_missing",
        "depth_source": candidate.depth_source,
        "candidate_reference_world_x_m": candidate_xyz[0] if candidate_xyz is not None else None,
        "candidate_reference_world_y_m": candidate_xyz[1] if candidate_xyz is not None else None,
        "candidate_reference_world_z_m": candidate_xyz[2] if candidate_xyz is not None else None,
        "frozen_reference_world_x_m": (frozen_reference.reference_world_m[0]
                                         if frozen_reference and frozen_reference.valid else None),
        "frozen_reference_world_y_m": (frozen_reference.reference_world_m[1]
                                         if frozen_reference and frozen_reference.valid else None),
        "frozen_reference_world_z_m": (frozen_reference.reference_world_m[2]
                                         if frozen_reference and frozen_reference.valid else None),
        "valid_depth_count": candidate.valid_depth_count if candidate else 0,
        "mask_pixel_count": candidate.mask_pixel_count if candidate else int(mask.sum()),
        "valid_depth_ratio": candidate.valid_depth_ratio if candidate else 0.0,
        "depth_median_m": candidate.depth_median_m if candidate else None,
        "depth_spread_m": candidate.depth_spread_m if candidate else None,
        "simulator_gt_reference_world_x_m": (
            gt_reference_xyz[0] if gt_reference_xyz is not None else None),
        "simulator_gt_reference_world_y_m": (
            gt_reference_xyz[1] if gt_reference_xyz is not None else None),
        "simulator_gt_reference_world_z_m": (
            gt_reference_xyz[2] if gt_reference_xyz is not None else None),
        "estimate_vs_simulator_gt_reference_error_m": point_error,
        "depth_mae_m": depth_mae,
        "depth_median_absolute_error_m": depth_median_abs_error,
        "depth_valid_pixel_count": int(abs_error.size),
        "frame_to_frame_displacement_m": displacement,
        "oracle_offset_x_m": offset[0] if offset is not None else None,
        "oracle_offset_y_m": offset[1] if offset is not None else None,
        "oracle_offset_z_m": offset[2] if offset is not None else None,
        "oracle_offset_norm_m": float(np.linalg.norm(offset)) if offset is not None else None,
        "oracle_body_origin_world_x_m": oracle_xyz[0] if oracle_xyz is not None else None,
        "oracle_body_origin_world_y_m": oracle_xyz[1] if oracle_xyz is not None else None,
        "oracle_body_origin_world_z_m": oracle_xyz[2] if oracle_xyz is not None else None,
        "raw_depth_min": payload["raw_depth_min"],
        "raw_depth_max": payload["raw_depth_max"],
        "estimated_depth_min_m": estimated_stats.get("metric_depth_min_m"),
        "estimated_depth_max_m": estimated_stats.get("metric_depth_max_m"),
        "simulator_gt_depth_min_m": simulator_gt_stats.get("metric_depth_min_m"),
        "simulator_gt_depth_max_m": simulator_gt_stats.get("metric_depth_max_m"),
        "simulator_gt_is_experiment_diagnostic_only": True,
        "qwen_actions": 0,
        "_directory": str(sample_dir),
    }


def _collect_static_trial(
    *, init_state: int, run_dir: Path, config: Mapping[str, Any], sam3: Any,
    camera_resolution: int, depth_provider: Any,
) -> dict[str, Any]:
    from core.runtime_v3.adapters.libero_observation import LiberoObservationAdapter
    from core.runtime_v3.state import BeliefState

    environment = None
    samples: list[dict[str, Any]] = []
    try:
        environment, controller, base, observer, ready_holds, scene = m34._setup_episode(
            init_state=init_state,
            config=config,
            sam3=sam3,
            camera_resolution=camera_resolution,
            workspace=m34.WORKSPACE_Z_M,
            diagnostic_camera_depths=True,
            metric_depth_provider=depth_provider,
        )
        if not isinstance(base, LiberoObservationAdapter):
            raise RuntimeError("M3.5 requires the LIBERO diagnostic observation adapter")
        if not observer.perception_history:
            raise RuntimeError("SceneReady completed without a retained perception frame")
        frame = observer.perception_history[-1]
        candidate = frame.get("metric_entity_reference_candidate")
        estimated_depth = frame.get("metric_depth_estimate_m")
        if candidate is None:
            raise RuntimeError("deployable metric depth provider did not produce a reference")
        simulator_raw = environment.get_observation()
        simulator_gt_candidate, simulator_gt_depth = _simulator_depth_candidate(
            frame, simulator_raw, environment)
        frozen = observer.metric_reference_anchor
        previous = None
        samples.append(_save_static_sample(
            output_dir=run_dir, init_state=init_state, sample_index=0,
            sample_role="scene_ready_reference", frame=frame, base_observer=base,
            simulator_raw=simulator_raw,
            candidate=candidate, estimated_depth=estimated_depth,
            simulator_gt_candidate=simulator_gt_candidate,
            simulator_gt_depth=simulator_gt_depth,
            frozen_reference=frozen, environment=environment, previous_candidate=previous,
        ))
        previous = candidate
        builder = StateBuilder()
        state = builder.initialize(f"{m34.SUITE}:{m34.TASK_ID}")
        for hold_index in range(HOLD_OBSERVATIONS):
            before_step = int(environment.step_count)
            result = run_v3_tick(
                environment, base, controller,
                task_id=f"{m34.SUITE}:{m34.TASK_ID}", token=None,
                direction_unit=None, commanded_step_m=m34.CONTROL_TICK_M,
                reset=False, workspace_z_bounds_m=m34.WORKSPACE_Z_M,
            )
            if (result.get("actions") != 1 or result.get("approved_action") != "CALIBRATION_HOLD"
                    or not result.get("backend_execution")
                    or int(environment.step_count) != before_step + 1):
                raise RuntimeError(f"static validation must execute one bounded HOLD: {result}")
            source_observation = base.last_observation
            if source_observation is None:
                raise RuntimeError("Runtime HOLD did not yield a fresh source observation")
            observation = observer.observe_from_base_observation(environment, source_observation)
            state = builder.update(state, observation)
            frame = observer.perception_history[-1]
            current = frame.get("metric_entity_reference_candidate")
            if current is None:
                raise RuntimeError(f"deployable reference missing at hold {hold_index + 1}")
            estimated_depth = frame.get("metric_depth_estimate_m")
            simulator_raw = environment.get_observation()
            simulator_gt_candidate, simulator_gt_depth = _simulator_depth_candidate(
                frame, simulator_raw, environment)
            samples.append(_save_static_sample(
                output_dir=run_dir, init_state=init_state, sample_index=hold_index + 1,
                sample_role=f"hold_{hold_index + 1}", frame=frame,
                base_observer=base, simulator_raw=simulator_raw,
                candidate=current, estimated_depth=estimated_depth,
                simulator_gt_candidate=simulator_gt_candidate,
                simulator_gt_depth=simulator_gt_depth,
                frozen_reference=frozen, environment=environment,
                previous_candidate=previous,
            ))
            previous = current
        valid = [row for row in samples if row["candidate_valid"]]
        displacements = [float(row["frame_to_frame_displacement_m"]) for row in samples
                         if row["frame_to_frame_displacement_m"] is not None]
        max_jitter = max(displacements) if displacements else None
        return {
            "init_state_index": init_state,
            "status": "COMPLETED",
            "robot_ready_hold_ticks": len(ready_holds),
            "scene_ready": scene,
            "hold_observations": HOLD_OBSERVATIONS,
            "frames_including_scene_ready_reference": len(samples),
            "valid_reference_frames": len(valid),
            "valid_reference_rate": len(valid) / len(samples) if samples else 0.0,
            "mean_valid_depth_ratio": (float(np.mean([row["valid_depth_ratio"] for row in valid]))
                                       if valid else None),
            "max_frame_to_frame_jitter_m": max_jitter,
            "static_stability_limit_m": STATIC_FRAME_JITTER_LIMIT_M,
            "stable": bool(len(valid) == len(samples) and max_jitter is not None
                           and max_jitter <= STATIC_FRAME_JITTER_LIMIT_M),
            "frozen_metric_reference": _reference_record(frozen),
            "depth_observation_audit": {
                "agentview_rgb_shape": list(base.last_raw.agentview_rgb.shape),
                "agentview_depth_shape": list(np.asarray(simulator_raw["agentview_depth"]).shape),
                "agentview_depth_dtype": str(np.asarray(simulator_raw["agentview_depth"]).dtype),
                "agentview_depth_raw_range": [
                    float(np.nanmin(simulator_raw["agentview_depth"])),
                    float(np.nanmax(simulator_raw["agentview_depth"])),
                ],
                "wrist_rgb_shape": (list(base.last_raw.wrist_rgb.shape)
                                    if base.last_raw.wrist_rgb is not None else None),
                "wrist_depth_shape": list(np.asarray(simulator_raw["robot0_eye_in_hand_depth"]).shape),
                "wrist_depth_dtype": str(np.asarray(simulator_raw["robot0_eye_in_hand_depth"]).dtype),
                "wrist_depth_raw_range": [
                    float(np.nanmin(simulator_raw["robot0_eye_in_hand_depth"])),
                    float(np.nanmax(simulator_raw["robot0_eye_in_hand_depth"])),
                ],
                "depth_source": "LIBERO robosuite camera_depths=True source render; diagnostic only",
                "depth_is_upsampled": False,
                "formal_runtime_depth_dependency": False,
            },
            "samples": samples,
            "oracle_used_by_runtime": False,
            "qwen_actions": 0,
        }
    finally:
        if environment is not None:
            environment.close()


def _stage_b_depth_callback():
    """Return an isolated telemetry callback and its local frozen reference."""
    frozen: dict[str, MetricEntityReference | None] = {"value": None}

    def callback(*, frame: Mapping[str, Any], raw: Any, state: Any,
                 environment: Any, step: int, capture_label: str,
                 record_data: Mapping[str, Any]) -> Mapping[str, Any]:
        candidate = frame.get("metric_entity_reference_candidate")
        if candidate is None:
            candidate = invalid_metric_entity_reference(
                entity_key=m34.TARGET_PHRASE, camera="agentview",
                source_frame_id=frame.get("frame_id", "unknown"),
                invalid_reason="deployable_metric_reference_missing",
            )
        simulator_raw = environment.get_observation()
        simulator_gt_candidate, simulator_gt_depth = _simulator_depth_candidate(
            frame, simulator_raw, environment)
        frozen["value"] = freeze_metric_reference(frozen["value"], candidate)
        reference = frozen["value"]
        eef = record_data.get("eef_position_xyz_m")
        distance = metric_proximity_distance_m(eef, reference)
        estimated_depth = frame.get("metric_depth_estimate_m")
        segmentation = frame.get("segmentation")
        mask = getattr(segmentation, "mask", None)
        depth_mae = median_depth_error = None
        depth_valid_count = 0
        if estimated_depth is not None and simulator_gt_depth is not None and mask is not None:
            estimated = np.asarray(estimated_depth, dtype=np.float32)
            simulator_gt = np.asarray(simulator_gt_depth, dtype=np.float32)
            selected = (np.asarray(mask, dtype=bool) & np.isfinite(estimated) & (estimated > 0.0)
                        & np.isfinite(simulator_gt) & (simulator_gt > 0.0))
            errors = np.abs(estimated[selected] - simulator_gt[selected])
            depth_valid_count = int(errors.size)
            if errors.size:
                depth_mae = float(np.mean(errors))
                median_depth_error = float(np.median(errors))
        estimate_vs_gt_reference_error = None
        if (candidate.valid and simulator_gt_candidate.valid
                and candidate.reference_world_m is not None
                and simulator_gt_candidate.reference_world_m is not None):
            estimate_vs_gt_reference_error = float(np.linalg.norm(
                np.asarray(candidate.reference_world_m, dtype=float)
                - np.asarray(simulator_gt_candidate.reference_world_m, dtype=float)
            ))
        wrist_coverage = {"status": "UNRESOLVED", "projected_pixel_px": None,
                          "reason": "depth_reference_unavailable"}
        if reference is not None and raw is not None and raw.wrist_rgb is not None:
            try:
                from core.capabilities.camera_geometry import make_mujoco_calibrations

                raw_env = getattr(environment, "env", environment)
                wrist_shape = np.asarray(raw.wrist_rgb).shape[:2]
                wrist_calibration = make_mujoco_calibrations(
                    raw_env,
                    {"wrist": "robot0_eye_in_hand"},
                    image_shapes={"wrist": wrist_shape},
                )["wrist"]
                wrist_coverage = classify_metric_reference_visibility(
                    reference,
                    wrist_calibration,
                    image_adapter=CanonicalImageAdapter("vertical_flip"),
                    sam_detected=bool(record_data.get("wrist_visible", False)),
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                wrist_coverage = {"status": "UNRESOLVED", "projected_pixel_px": None,
                                  "reason": "wrist_camera_calibration_unavailable"}
        return {
            "depth_provider": "monocular_metric",
            "depth_metric_capture_role": capture_label,
            "deployable_metric_reference": _reference_record(candidate),
            "deployable_metric_frozen_reference": _reference_record(reference),
            "deployable_metric_candidate_valid": bool(candidate.valid),
            "deployable_metric_candidate_invalid_reason": candidate.invalid_reason,
            "depth_metric_candidate_reference_world_m": (
                list(candidate.reference_world_m) if candidate.reference_world_m is not None else None
            ),
            "depth_metric_reference_world_m": (
                list(reference.reference_world_m)
                if reference is not None and reference.reference_world_m is not None else None
            ),
            "depth_metric_candidate_valid_depth_ratio": candidate.valid_depth_ratio,
            "depth_metric_reference_valid_depth_ratio": (
                reference.valid_depth_ratio if reference is not None else None
            ),
            "runtime_estimated_metric_distance_m": distance,
            "simulator_gt_depth_reference": _reference_record(simulator_gt_candidate),
            "estimate_vs_simulator_gt_reference_error_m": estimate_vs_gt_reference_error,
            "depth_mae_vs_simulator_gt_m": depth_mae,
            "depth_median_absolute_error_vs_simulator_gt_m": median_depth_error,
            "depth_valid_pixel_count_vs_simulator_gt": depth_valid_count,
            "wrist_depth_reference_coverage_status": str(wrist_coverage.get("status", "UNRESOLVED")),
            "wrist_depth_reference_projection": wrist_coverage,
            "simulator_gt_may_affect_runtime_decision": False,
            "diagnostic_step": int(step),
        }

    return callback, frozen


def _correlations(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Keep pair filtering aligned across each comparison.
    metric_oracle = [(float(row["runtime_estimated_metric_distance_m"]),
                      float(row["oracle_eef_target_distance_m"]))
                     for episode in episodes for row in episode.get("trajectory", [])
                     if row.get("runtime_estimated_metric_distance_m") is not None
                     and row.get("oracle_eef_target_distance_m") is not None]
    image_oracle = [(float(row["agentview_error_px"]),
                     float(row["oracle_eef_target_distance_m"]))
                    for episode in episodes for row in episode.get("trajectory", [])
                     if row.get("agentview_error_px") is not None
                     and row.get("oracle_eef_target_distance_m") is not None]
    metric_image = [(float(row["runtime_estimated_metric_distance_m"]),
                     float(row["agentview_error_px"]))
                    for episode in episodes for row in episode.get("trajectory", [])
                    if row.get("runtime_estimated_metric_distance_m") is not None
                    and row.get("agentview_error_px") is not None]
    return {
        "estimated_metric_distance_vs_oracle_body_distance": m34.correlation_pair(
            [pair[0] for pair in metric_oracle], [pair[1] for pair in metric_oracle]),
        "agentview_2d_error_vs_oracle_body_distance": m34.correlation_pair(
            [pair[0] for pair in image_oracle], [pair[1] for pair in image_oracle]),
        "estimated_metric_distance_vs_agentview_2d_error": m34.correlation_pair(
            [pair[0] for pair in metric_image], [pair[1] for pair in metric_image]),
    }


def _annotate_metric_steps(episode: dict[str, Any]) -> dict[str, Any]:
    observations = episode.get("observations", [])
    for step in episode.get("steps", []):
        if not step.get("executed"):
            continue
        index = int(step.get("alignment_step", -1))
        direction = step.get("direction")
        scale = step.get("scale_mm")

        def find(role: str, frame_step: int):
            return next((row for row in observations
                         if row.get("depth_metric_capture_role") == role
                         and int(row.get("diagnostic_step", -999)) == frame_step
                         and row.get("selected_direction") == direction
                         and (scale is None or row.get("selected_scale_mm") is None
                              or abs(float(row["selected_scale_mm"]) - float(scale)) < 1e-6)), None)

        before = find("before", index - 1)
        after = find("after", index)
        before_distance = (before or {}).get("runtime_estimated_metric_distance_m")
        after_distance = (after or {}).get("runtime_estimated_metric_distance_m")
        step["runtime_estimated_metric_distance_before_m"] = before_distance
        step["runtime_estimated_metric_distance_after_m"] = after_distance
        step["runtime_estimated_metric_distance_improvement_m"] = (
            float(before_distance) - float(after_distance)
            if before_distance is not None and after_distance is not None else None
        )
        step["simulator_gt_diagnostic_only"] = True
    return episode


def _stage_b_summary(episodes: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    steps = [step for episode in episodes for step in episode.get("steps", [])
             if step.get("executed")]
    estimated_measured = [step for step in steps
                          if step.get("runtime_estimated_metric_distance_improvement_m") is not None]
    oracle_measured = [step for step in steps
                       if step.get("oracle_distance_improvement_m") is not None]
    paired = [step for step in steps
              if step.get("runtime_estimated_metric_distance_improvement_m") is not None
              and step.get("oracle_distance_improvement_m") is not None]
    estimated_positive = [step for step in estimated_measured
                          if float(step["runtime_estimated_metric_distance_improvement_m"]) > 0.0]
    oracle_positive = [step for step in oracle_measured
                       if float(step["oracle_distance_improvement_m"]) > 0.0]
    both_positive = [step for step in paired
                     if float(step["runtime_estimated_metric_distance_improvement_m"]) > 0.0
                     and float(step["oracle_distance_improvement_m"]) > 0.0]
    trajectory_episodes = [episode for episode in episodes if episode.get("trajectory")]
    m34._plot_series(
        trajectory_episodes, "runtime_estimated_metric_distance_m",
        output_dir / "metric_distance_vs_step.png",
        "Frozen RGB-only estimated EEF-to-entity distance", "meters",
    )
    _plot_xy(
        [(float(row["runtime_estimated_metric_distance_m"]),
          float(row["oracle_eef_target_distance_m"]))
         for episode in episodes for row in episode.get("trajectory", [])
         if row.get("runtime_estimated_metric_distance_m") is not None
         and row.get("oracle_eef_target_distance_m") is not None],
        output_dir / "runtime_vs_oracle_distance.png",
        "RGB-only estimated metric distance vs simulator oracle distance",
        "estimated metric distance (m)", "oracle diagnostic distance (m)",
    )
    _plot_xy(
        [(float(row["agentview_error_px"]),
          float(row["runtime_estimated_metric_distance_m"]))
         for episode in episodes for row in episode.get("trajectory", [])
         if row.get("agentview_error_px") is not None
         and row.get("runtime_estimated_metric_distance_m") is not None],
        output_dir / "2d_vs_metric_distance.png",
        "Agentview 2D alignment error vs RGB-only estimated metric distance",
        "agentview image error (px)", "estimated metric distance (m)",
    )
    statuses: dict[str, int] = {
        "OUTSIDE_FRUSTUM": 0,
        "INSIDE_FRUSTUM_SAM_MISS": 0,
        "VISIBLE_DETECTED": 0,
        "UNRESOLVED": 0,
    }
    for episode in episodes:
        for row in episode.get("observations", []):
            status = str(row.get("wrist_depth_reference_coverage_status", "UNRESOLVED"))
            statuses[status] = statuses.get(status, 0) + 1
    correlations = _correlations(episodes)
    _write_json(output_dir / "metric_grounding_correlations.json", correlations)
    _plot_wrist_status(episodes, output_dir / "wrist_frustum_status.png")
    return {
        "episodes": len(episodes),
        "executed_steps": len(steps),
        "estimated_metric_distance_measured_steps": len(estimated_measured),
        "estimated_metric_distance_positive_steps": len(estimated_positive),
        "oracle_distance_positive_steps": len(oracle_positive),
        "both_positive_steps": len(both_positive),
        "wrist_frustum_classification_counts": statuses,
        "correlations": correlations,
        "qwen_actions": 0,
        "simulator_gt_depth_used_for_runtime_decisions": False,
        "plots": {
            "metric_distance_vs_step": str(output_dir / "metric_distance_vs_step.png"),
            "runtime_vs_oracle_distance": str(output_dir / "runtime_vs_oracle_distance.png"),
            "2d_vs_metric_distance": str(output_dir / "2d_vs_metric_distance.png"),
            "wrist_frustum_status": str(output_dir / "wrist_frustum_status.png"),
        },
    }


def _plot_wrist_status(episodes: Sequence[Mapping[str, Any]], output: Path) -> None:
    from PIL import ImageDraw

    rows = [(int(episode["init_state_index"]), row)
            for episode in episodes for row in episode.get("observations", [])]
    width, row_height = 1000, 24
    height = max(90, 60 + len(rows) * row_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), "Wrist projection coverage by observation", fill="black")
    colors = {
        "OUTSIDE_FRUSTUM": (70, 110, 220),
        "INSIDE_FRUSTUM_SAM_MISS": (220, 145, 35),
        "VISIBLE_DETECTED": (45, 165, 85),
        "UNRESOLVED": (130, 130, 130),
    }
    for index, (init_state, row) in enumerate(rows):
        y = 42 + index * row_height
        status = str(row.get("wrist_depth_reference_coverage_status", "UNRESOLVED"))
        color = colors.get(status, colors["UNRESOLVED"])
        draw.text((12, y + 3), f"state {init_state} step {row.get('step')}", fill="black")
        draw.rectangle((220, y, 970, y + 19), fill=(238, 238, 238))
        draw.rectangle((220, y, 970, y + 19), fill=color)
        draw.text((230, y + 3), status, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _plot_xy(points: Sequence[tuple[float, float]], output: Path, title: str,
             x_label: str, y_label: str) -> None:
    from PIL import ImageDraw

    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 65, width - 35, height - 85)
    draw.text((20, 18), title, fill="black")
    draw.rectangle(box, outline=(50, 50, 50), width=2)
    draw.text((box[0], height - 55), x_label, fill="black")
    draw.text((8, box[1] + 4), y_label, fill="black")
    finite = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if finite:
        xs, ys = zip(*finite)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xspan, yspan = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
        for x, y in finite:
            px = box[0] + int((x - xmin) / xspan * (box[2] - box[0]))
            py = box[3] - int((y - ymin) / yspan * (box[3] - box[1]))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(35, 125, 205))
        draw.text((box[0] + 8, box[1] + 8), f"n={len(finite)}", fill="black")
    else:
        draw.text((box[0] + 15, box[1] + 15), "No paired observations", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    from core.capabilities.sam3_client import Sam3Client
    from core.config import load_yaml

    if args.camera_resolution != 512:
        raise SystemExit("M3.5 requires direct 512x512 RGB and depth source renders")
    config = load_yaml(args.config)
    if config.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(config["libero_dir"])
    m34._configure_local_proxy(args.sam3_url)
    depth_provider = MoGeMetricDepthProvider(
        checkpoint=args.moge_checkpoint,
        revision=args.moge_revision,
        repo_dir=args.moge_repo_dir,
        device=args.moge_device,
        local_files_only=args.moge_local_files_only,
    )
    depth_provider.load()
    sam3 = Sam3Client(url=args.sam3_url, python=args.sam3_python,
                      timeout_s=args.sam3_timeout_s, max_attempts=1)
    run_dir = m34._run_directory(args.output_dir)
    stage_a: list[dict[str, Any]] = []
    stage_b: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        for init_state in m34.INIT_STATES:
            try:
                stage_a.append(_collect_static_trial(
                    init_state=init_state, run_dir=run_dir, config=config,
                    sam3=sam3, camera_resolution=args.camera_resolution,
                    depth_provider=depth_provider,
                ))
            except Exception as exc:
                blockers.append(f"stage_a_init_state_{init_state}: {type(exc).__name__}: {exc}")
                stage_a.append({"init_state_index": init_state, "status": "FAILED",
                                "error": f"{type(exc).__name__}: {exc}",
                                "oracle_used_by_runtime": False, "qwen_actions": 0})
        _write_csv(run_dir / "static_metric_reference.csv", [
            sample for trial in stage_a for sample in trial.get("samples", [])
        ])
        all_static_rows = [sample for trial in stage_a for sample in trial.get("samples", [])]
        offsets = [float(row["oracle_offset_norm_m"]) for row in all_static_rows
                   if row.get("oracle_offset_norm_m") is not None]
        depth_mae_values = [float(row["depth_mae_m"]) for row in all_static_rows
                            if row.get("depth_mae_m") is not None]
        gt_reference_errors = [float(row["estimate_vs_simulator_gt_reference_error_m"])
                               for row in all_static_rows
                               if row.get("estimate_vs_simulator_gt_reference_error_m") is not None]
        stable = (len(stage_a) == len(m34.INIT_STATES) and not blockers
                  and all(trial.get("stable") for trial in stage_a))
        all_stage_a_trials_complete = (
            len(stage_a) == len(m34.INIT_STATES)
            and all(trial.get("status") == "COMPLETED" for trial in stage_a)
        )
        stage_a_summary = {
            "status": ("STABLE" if stable else "UNSTABLE"
                       if all_stage_a_trials_complete and not blockers else "INCOMPLETE"),
            "init_states": list(m34.INIT_STATES),
            "all_trials_complete": all_stage_a_trials_complete,
            "unstable_init_states": [trial["init_state_index"] for trial in stage_a
                                     if trial.get("status") == "COMPLETED"
                                     and not trial.get("stable")],
            "frames_per_state_including_anchor": HOLD_OBSERVATIONS + 1,
            "hold_observations_per_state": HOLD_OBSERVATIONS,
            "valid_reference_rate": (
                sum(bool(row.get("candidate_valid")) for row in all_static_rows)
                / len(all_static_rows) if all_static_rows else 0.0
            ),
            "reference_depth_source": "monocular_metric",
            "mean_valid_depth_ratio": (
                float(np.mean([float(row["valid_depth_ratio"]) for row in all_static_rows
                               if row.get("candidate_valid")]))
                if any(row.get("candidate_valid") for row in all_static_rows) else None
            ),
            "frame_to_frame_jitter_m": {
                "mean": float(np.mean([row["frame_to_frame_displacement_m"] for row in all_static_rows
                                        if row.get("frame_to_frame_displacement_m") is not None]))
                if any(row.get("frame_to_frame_displacement_m") is not None for row in all_static_rows)
                else None,
                "max": max([row["frame_to_frame_displacement_m"] for row in all_static_rows
                            if row.get("frame_to_frame_displacement_m") is not None], default=None),
                "stability_limit_m": STATIC_FRAME_JITTER_LIMIT_M,
                "limit_basis": "smallest existing calibrated motion scale (3 mm)",
            },
            "oracle_reference_offset_norm_m": {
                "mean": float(np.mean(offsets)) if offsets else None,
                "std": float(np.std(offsets)) if offsets else None,
                "sample_count": len(offsets),
                "oracle_used_by_runtime": False,
            },
            "estimated_depth_vs_simulator_gt_depth_mae_m": {
                "mean": float(np.mean(depth_mae_values)) if depth_mae_values else None,
                "median": float(np.median(depth_mae_values)) if depth_mae_values else None,
                "sample_count": len(depth_mae_values),
                "diagnostic_only": True,
            },
            "estimated_reference_vs_simulator_gt_depth_reference_error_m": {
                "mean": float(np.mean(gt_reference_errors)) if gt_reference_errors else None,
                "median": float(np.median(gt_reference_errors)) if gt_reference_errors else None,
                "sample_count": len(gt_reference_errors),
                "diagnostic_only": True,
            },
            "trials": stage_a,
            "oracle_used_by_runtime": False,
            "qwen_actions": 0,
        }
        _write_json(run_dir / "stage_a_summary.json", stage_a_summary)
        if stable:
            for init_state in m34.INIT_STATES:
                try:
                    stage_b.append(m34._run_episode(
                        init_state=init_state, run_dir=run_dir / "stage_b",
                        config=config, sam3=sam3, camera_resolution=args.camera_resolution,
                        wrist_orientation="vertical_flip",
                        orientation_audit_path="M3.4 established direct-render vertical_flip",
                        diagnostic_camera_depths=True,
                        metric_depth_provider=depth_provider,
                        diagnostic_callback=_stage_b_depth_callback()[0],
                    ))
                except Exception as exc:
                    blockers.append(f"stage_b_init_state_{init_state}: {type(exc).__name__}: {exc}")
                    episode_dir = run_dir / "stage_b" / f"init_state_{init_state}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    failure = {"init_state_index": init_state, "status": "FAILED",
                               "error": f"{type(exc).__name__}: {exc}",
                               "executed_semantic_steps": 0, "oracle_used_for_runtime": False,
                               "qwen_actions": 0}
                    _write_json(episode_dir / "failure.json", failure)
                    stage_b.append(failure)
    finally:
        sam3.close()

    stage_b_summary = _stage_b_summary(stage_b, run_dir) if stage_b else {
        "status": "NOT_RUN_STAGE_A_GATE_FAILED", "episodes": 0, "executed_steps": 0,
        "qwen_actions": 0, "oracle_used_by_runtime": False,
    }
    if stage_b:
        _plot_wrist_status(stage_b, run_dir / "wrist_frustum_status.png")
    all_observations = [row for episode in stage_b for row in episode.get("observations", [])]
    stage_b_summary["wrist_frustum_classification_counts"] = stage_b_summary.get(
        "wrist_frustum_classification_counts", {}
    )
    final = {
        "phase": "M3.5 Deployable Metric Entity Grounding",
        "branch": "runtime-v3",
        "starting_commit": "0973f4c107124415c790a214a98ec480c1076935",
        "suite": m34.SUITE,
        "task_id": m34.TASK_ID,
        "seed": 0,
        "init_states": list(m34.INIT_STATES),
        "camera_resolution_requested": [512, 512],
        "depth_estimator_uses_rgb_only": True,
        "formal_runtime_additional_inputs": ["proprioception", "camera calibration"],
        "depth_provider": {
            "type": "monocular_metric",
            "model": depth_provider.checkpoint,
            "revision": depth_provider.revision,
            "input": "canonical RGB only",
            "output": "metric depth in meters at input resolution",
            "metric_output": True,
            "simulator_calibration_required": False,
            "model_code_repo": str(depth_provider.repo_dir) if depth_provider.repo_dir else None,
        },
        "depth_model_environment_audit": {
            **_depth_model_environment_audit(depth_provider),
        },
        "diagnostic_simulator_depth_render_enabled": True,
        "simulator_depth_role": "diagnostic-only upper-bound reference",
        "simulator_depth_conversion": "robosuite get_real_depth_map formula: near / (1 - d * (1 - near/far))",
        "simulator_depth_units": "meters",
        "coordinate_frame": "world",
        "stage_a": stage_a_summary,
        "stage_b": stage_b_summary,
        "stage_b_episodes": stage_b,
        "privileged_information_audit": {
            "formal_runtime_reads_simulator_target_pose": False,
            "formal_runtime_reads_simulator_depth": False,
            "formal_runtime_reads_simulator_contact": False,
            "formal_runtime_reads_task_success_gt": False,
            "simulator_values_are_diagnostic_only": True,
            "metric_reference_used_by_option_ranking_or_3d_control": False,
        },
        "wrist_coverage_status": ("EVALUATED" if stage_b else "NOT_EVALUATED_STAGE_A_GATE_FAILED"),
        "wrist_observations": len(all_observations) if stage_b else None,
        "wrist_visible_detected": (
            sum(row.get("wrist_depth_reference_coverage_status") == "VISIBLE_DETECTED"
                for row in all_observations) if stage_b else None
        ),
        "blockers": blockers,
        "qwen_actions": 0,
        "oracle_used_by_runtime": False,
        "grasp_release_place_actions": 0,
        "legacy_modified": False,
        "deployable_metric_entity_grounding_reliable": bool(stable),
        "next_milestone": (
            "consider 3D metric alignment only after a stable deployable reference"
            if stable else "improve or replace deployable metric sensing; do not proceed to 3D control"
        ),
        "status": ("COMPLETED" if stable and len(stage_b) == len(m34.INIT_STATES)
                   and not blockers else "STAGE_A_ONLY" if not stable and not blockers else "PARTIAL"),
        "artifacts": {
            "static_metric_reference_csv": str(run_dir / "static_metric_reference.csv"),
            "stage_a_summary": str(run_dir / "stage_a_summary.json"),
            "metric_grounding_correlations": str(run_dir / "metric_grounding_correlations.json")
            if stage_b else None,
            "metric_distance_vs_step": str(run_dir / "metric_distance_vs_step.png")
            if stage_b else None,
            "runtime_vs_oracle_distance": str(run_dir / "runtime_vs_oracle_distance.png")
            if stage_b else None,
            "2d_vs_metric_distance": str(run_dir / "2d_vs_metric_distance.png")
            if stage_b else None,
            "wrist_frustum_status": str(run_dir / "wrist_frustum_status.png")
            if stage_b else None,
        },
        "run_directory": str(run_dir),
    }
    _write_json(run_dir / "summary.json", final)
    print(json.dumps({"status": final["status"], "run_directory": str(run_dir),
                      "stage_a_stable": stable,
                      "stage_b_episodes": len(stage_b), "blockers": blockers}, indent=2))
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/robot_libero_clean_qwen3vl.yaml"))
    parser.add_argument("--output-dir", default=str(
        ROOT / "rollouts/runtime_v3_metric_entity_grounding_rgb_only"))
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--sam3-python", default="/root/autodl-tmp/openeta-services/sam3/.venv/bin/python")
    parser.add_argument("--sam3-timeout-s", type=float, default=120.0)
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--moge-repo-dir", default="/root/autodl-tmp/openeta-services/MoGe")
    parser.add_argument("--moge-checkpoint", default=MoGeMetricDepthProvider.default_checkpoint)
    parser.add_argument("--moge-revision", default=MoGeMetricDepthProvider.default_revision)
    parser.add_argument("--moge-device", default="cuda:0")
    parser.add_argument("--moge-local-files-only", action="store_true")
    return parser


if __name__ == "__main__":
    run_experiment(_parser().parse_args())
