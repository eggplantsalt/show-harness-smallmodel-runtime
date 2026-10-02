"""Deployable image-only metric depth providers for Runtime V3.

Simulator depth is intentionally absent from this module. A provider receives
only RGB and returns a metric estimate produced by a deployable sensor/model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class MetricDepthEstimate:
    depth_m: np.ndarray
    source: str
    model_name: str | None = None
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("metric depth output must be an HxW raster")
        if self.source not in {"monocular_metric", "rgbd_sensor"}:
            raise ValueError("formal depth source must be monocular_metric or rgbd_sensor")
        valid_mask = self.valid_mask
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != depth.shape:
                raise ValueError("depth validity mask must match the depth raster")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "valid_mask", valid_mask)


@runtime_checkable
class DepthProvider(Protocol):
    """A deployable depth estimator whose complete sensor input is RGB."""

    def estimate(self, rgb: np.ndarray) -> MetricDepthEstimate: ...


class MoGeMetricDepthProvider:
    """Lazy MoGe-2 ViT-L metric monocular depth provider.

    The estimator accepts a single RGB image and returns same-resolution depth
    in meters. It is independent of task IDs, masks, robot state, and simulator
    data. ``inference`` is a narrow injection point for deterministic tests.
    """

    source = "monocular_metric"
    default_checkpoint = "Ruicheng/moge-2-vitl"
    default_revision = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"

    def __init__(
        self,
        *,
        checkpoint: str = default_checkpoint,
        revision: str = default_revision,
        repo_dir: str | Path | None = None,
        device: str = "cuda:0",
        use_fp16: bool = True,
        local_files_only: bool = False,
        inference: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.revision = str(revision)
        self.repo_dir = Path(repo_dir).expanduser() if repo_dir is not None else None
        self.device = str(device)
        self.use_fp16 = bool(use_fp16)
        self.local_files_only = bool(local_files_only)
        self._inference = inference
        self._model: Any = None
        self._torch: Any = None
        self.load_error: str | None = None

    @property
    def model_name(self) -> str:
        return f"{self.checkpoint}@{self.revision}"

    def load(self) -> None:
        if self._inference is not None or self._model is not None or self.load_error:
            return
        try:
            if self.repo_dir is not None:
                repo = str(self.repo_dir.resolve())
                if repo not in sys.path:
                    sys.path.insert(0, repo)
            import torch
            from moge.model.v2 import MoGeModel

            model = MoGeModel.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            self._model = model.to(torch.device(self.device)).eval()
            self._torch = torch
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"could not load MoGe metric model: {self.load_error}") from exc

    def estimate(self, rgb: np.ndarray) -> MetricDepthEstimate:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError("RGB input must have shape HxWx3")
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("RGB input must be numeric")
        image = np.ascontiguousarray(image)
        if self._inference is not None:
            prediction = self._inference(image)
        else:
            self.load()
            rgb01 = image.astype(np.float32)
            if np.issubdtype(image.dtype, np.integer):
                rgb01 /= float(np.iinfo(image.dtype).max)
            elif float(np.nanmax(rgb01)) > 1.0:
                rgb01 /= 255.0
            if (not np.isfinite(rgb01).all() or float(rgb01.min()) < 0.0
                    or float(rgb01.max()) > 1.0):
                raise ValueError("RGB values must be finite and in [0, 1] or [0, 255]")
            tensor = self._torch.from_numpy(rgb01).to(self.device).permute(2, 0, 1)
            with self._torch.inference_mode():
                prediction = self._model.infer(tensor, use_fp16=self.use_fp16)
        if not isinstance(prediction, Mapping) or "depth" not in prediction:
            raise ValueError("MoGe inference must return a metric 'depth' map")
        depth = prediction["depth"]
        if hasattr(depth, "detach"):
            depth = depth.detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.shape != image.shape[:2]:
            raise ValueError(
                f"depth output shape {depth.shape} does not match RGB source {image.shape[:2]}"
            )
        valid_mask = prediction.get("mask")
        if hasattr(valid_mask, "detach"):
            valid_mask = valid_mask.detach().cpu().numpy()
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
                valid_mask = valid_mask[0]
        valid = np.isfinite(depth) & (depth > 0.0)
        if valid_mask is not None:
            valid &= valid_mask
        depth = depth.copy()
        depth[~valid] = np.nan
        return MetricDepthEstimate(
            depth_m=depth,
            source=self.source,
            model_name=self.model_name,
            valid_mask=valid,
        )


def metric_entity_reference_from_estimate(
    *,
    entity_key: str,
    camera: str,
    source_frame_id: Any,
    target_mask: np.ndarray | None,
    estimate: MetricDepthEstimate,
    calibration: Any,
) -> "MetricEntityReference":
    """Build a robust world-frame visible-surface point from estimated depth."""
    from core.capabilities.camera_geometry import opencv_camera_points_to_world
    from .metric_entity import MetricEntityReference

    if estimate.source not in {"monocular_metric", "rgbd_sensor"}:
        raise ValueError("simulator diagnostic depth is forbidden in formal reference construction")
    if target_mask is None:
        return _invalid_reference(entity_key, camera, source_frame_id, "target_mask_missing", 0)
    mask = np.asarray(target_mask, dtype=bool)
    depth = estimate.depth_m
    if mask.ndim != 2 or mask.shape != depth.shape:
        return _invalid_reference(entity_key, camera, source_frame_id, "mask_depth_shape_mismatch",
                                  int(mask.sum()) if mask.ndim == 2 else 0)
    area = int(mask.sum())
    if area == 0:
        return _invalid_reference(entity_key, camera, source_frame_id, "target_mask_empty", 0)
    if calibration is None or depth.shape != (int(calibration.height), int(calibration.width)):
        return _invalid_reference(entity_key, camera, source_frame_id,
                                  "camera_calibration_resolution_mismatch", area)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if estimate.valid_mask is not None:
        valid &= estimate.valid_mask
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return _invalid_reference(entity_key, camera, source_frame_id, "no_valid_mask_depth_pixels", area)

    z_all = depth[ys, xs].astype(np.float64)
    median = float(np.median(z_all))
    mad = float(np.median(np.abs(z_all - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(3.5 * robust_sigma, 1e-4)
    keep = np.abs(z_all - median) <= threshold
    ys, xs, z = ys[keep], xs[keep], z_all[keep]
    if z.size == 0:
        return _invalid_reference(entity_key, camera, source_frame_id, "all_mask_depth_rejected", area)

    height, width = depth.shape
    focal = (height / 2.0) / math.tan(math.radians(float(calibration.fovy_deg)) / 2.0)
    x_camera = (xs.astype(np.float64) - width / 2.0) * z / focal
    y_camera = (ys.astype(np.float64) - height / 2.0) * z / focal
    points_world = opencv_camera_points_to_world(
        np.column_stack((x_camera, y_camera, z)),
        camera_to_world=calibration.camera_to_world,
        position_world=calibration.position_world,
        rotation_degrees=getattr(calibration, "rotation_degrees", 0),
        flip=getattr(calibration, "flip", "none"),
    )
    reference = np.median(points_world, axis=0)
    z_median = float(np.median(z))
    z_spread = 1.4826 * float(np.median(np.abs(z - z_median)))
    count = int(z.size)
    return MetricEntityReference(
        entity_key=entity_key,
        camera=camera,
        coordinate_frame="world",
        reference_world_m=tuple(float(value) for value in reference),
        valid_depth_count=count,
        mask_pixel_count=area,
        valid_depth_ratio=float(count / area),
        depth_median_m=z_median,
        depth_spread_m=z_spread,
        depth_source=estimate.source,
        source_frame_id=str(source_frame_id),
        valid=True,
    )


def _invalid_reference(entity_key: str, camera: str, frame_id: Any,
                       reason: str, mask_pixels: int) -> "MetricEntityReference":
    from .metric_entity import MetricEntityReference

    return MetricEntityReference(
        entity_key=entity_key, camera=camera, coordinate_frame="world",
        reference_world_m=None, valid_depth_count=0,
        mask_pixel_count=max(0, int(mask_pixels)), valid_depth_ratio=0.0,
        depth_median_m=None, depth_spread_m=None, depth_source="monocular_metric",
        source_frame_id=str(frame_id), valid=False, invalid_reason=reason,
    )
