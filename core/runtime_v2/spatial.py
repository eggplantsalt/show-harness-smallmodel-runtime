"""Model-agnostic spatial capability bus.

The online runtime only consumes this module's typed, uncertainty-bearing
results.  Optional providers (MoGe, SAM3 and GraspGenX) can be attached by the
runner without making those packages hard dependencies of the LIBERO tests.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .types import SpatialBelief, SpatialHealth, SpatialRelation, SpatialToolResult


def _finite_xyz(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def classify_relation(relative_xyz: Sequence[float], uncertainty: Sequence[float] | None = None,
                      *, envelope_radius_m: float = 0.012) -> tuple[SpatialRelation, ...]:
    xyz = _finite_xyz(relative_xyz)
    if xyz is None:
        return (SpatialRelation.UNKNOWN,)
    unc = _finite_xyz(uncertainty) if uncertainty is not None else np.zeros(3)
    # A relation is only emitted if its sign survives its uncertainty interval.
    labels: list[SpatialRelation] = []
    inside = bool(np.all(np.abs(xyz) <= (float(envelope_radius_m) + np.maximum(unc, 0.0))))
    if inside:
        labels.append(SpatialRelation.INSIDE_ENVELOPE)
    if xyz[0] > float(unc[0]):
        labels.append(SpatialRelation.FRONT)
    elif xyz[0] < -float(unc[0]):
        labels.append(SpatialRelation.BACK)
    if abs(float(xyz[1])) > float(unc[1]):
        labels.append(SpatialRelation.LEFT if xyz[1] > 0 else SpatialRelation.RIGHT)
    if abs(float(xyz[2])) > float(unc[2]):
        labels.append(SpatialRelation.ABOVE if xyz[2] > 0 else SpatialRelation.BELOW)
    return tuple(labels or [SpatialRelation.UNKNOWN])


def fuse_spatial_results(
    results: Iterable[SpatialToolResult],
    *,
    frame_id: Optional[int],
    instance_id: Optional[str],
    max_age_frames: int = 2,
    conflict_sigma: float = 2.5,
    require_uncertainty: bool = False,
) -> SpatialBelief:
    """Fuse independent providers conservatively.

    A stale, failed, or wrong-instance provider cannot drive a move.  Sources
    that disagree beyond their declared uncertainty produce AMBIGUOUS rather
    than an averaged direction.
    """
    usable: list[tuple[SpatialToolResult, np.ndarray, np.ndarray]] = []
    for result in results:
        if result.instance_id not in {None, instance_id} or result.health != SpatialHealth.VALID:
            continue
        if result.stale or (frame_id is not None and result.frame_id is not None and frame_id - result.frame_id > max_age_frames):
            continue
        xyz = _finite_xyz(result.target_to_gripper_xyz)
        if xyz is None:
            continue
        if result.uncertainty_std_m is not None:
            unc = _finite_xyz(result.uncertainty_std_m)
        elif result.covariance is not None and len(result.covariance) >= 3:
            raw_uncertainty = _finite_xyz(tuple(result.covariance[:3]))
            # The V2.2 contract treats this legacy-named field as diagonal
            # covariance (m^2). Earlier profiles retain their existing units.
            unc = np.sqrt(np.maximum(raw_uncertainty, 0.0)) if require_uncertainty and raw_uncertainty is not None else raw_uncertainty
        else:
            unc = None if require_uncertainty else np.ones(3) * 0.02
        if unc is None or not np.all(np.isfinite(unc)):
            continue
        unc = np.maximum(np.abs(unc), 1e-4)
        usable.append((result, xyz, unc))
    if not usable:
        return SpatialBelief(frame_id=frame_id, instance_id=instance_id, diagnostics={"usable_sources": 0})

    values = np.stack([item[1] for item in usable])
    scales = np.stack([item[2] for item in usable])
    weighted = 1.0 / np.square(scales)
    fused = np.sum(values * weighted, axis=0) / np.maximum(np.sum(weighted, axis=0), 1e-9)
    spread = np.sqrt(np.sum(weighted * np.square(values - fused), axis=0) / np.maximum(np.sum(weighted, axis=0), 1e-9))
    conflict = np.any(spread > conflict_sigma * np.maximum(np.mean(scales, axis=0), 1e-4))
    agreeing = tuple(item[0].source for item in usable if not conflict)
    conflicting = tuple(item[0].source for item in usable if conflict)
    health = SpatialHealth.AMBIGUOUS if conflict else SpatialHealth.VALID
    fused_uncertainty = np.sqrt(
        1.0 / np.maximum(np.sum(weighted, axis=0), 1e-9) + np.square(spread)
    )
    relations = (SpatialRelation.UNKNOWN,) if conflict else classify_relation(fused, fused_uncertainty)
    return SpatialBelief(
        fused_relative_xyz=tuple(float(v) for v in fused),
        relations=relations,
        uncertainty=tuple(float(v) for v in fused_uncertainty),
        agreeing_sources=agreeing,
        conflicting_sources=conflicting,
        health=health,
        frame_id=frame_id,
        instance_id=instance_id,
        diagnostics={"providers": [asdict(item[0]) for item in usable]},
    )


def triangulate_metric_points(
    pixels_a: Sequence[Sequence[float]],
    pixels_b: Sequence[Sequence[float]],
    projection_a: np.ndarray,
    projection_b: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """RANSAC-free minimal triangulation for a temporally tracked mask sample.

    The caller supplies calibrated camera projection matrices from normal robot
    poses.  We intentionally return a spread estimate so the verifier can reject
    low-baseline or inconsistent tracks instead of inventing depth.
    """
    checked = triangulate_metric_points_checked(
        pixels_a, pixels_b, projection_a, projection_b, min_correspondences=1
    )
    if checked is None:
        return None
    return checked["center_world"], checked["surface_spread_m"]


def triangulate_metric_points_checked(
    pixels_a: Sequence[Sequence[float]],
    pixels_b: Sequence[Sequence[float]],
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    *,
    min_correspondences: int = 4,
    max_reprojection_error_px: float = 3.0,
) -> dict[str, Any] | None:
    """Triangulate real image correspondences with cheirality and reprojection gates.

    ``surface_spread_m`` describes the tracked object's visible 3-D extent. It
    is deliberately separate from ``median_reprojection_error_px`` so object
    shape is never reported as measurement uncertainty.
    """
    try:
        pa = np.asarray(projection_a, dtype=float).reshape(3, 4)
        pb = np.asarray(projection_b, dtype=float).reshape(3, 4)
        a = np.asarray(pixels_a, dtype=float)
        b = np.asarray(pixels_b, dtype=float)
        if (
            a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape
            or len(a) < max(1, int(min_correspondences))
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
        ):
            return None
    except (TypeError, ValueError):
        return None
    points: list[np.ndarray] = []
    reprojection_errors: list[float] = []
    for (ua, va), (ub, vb) in zip(a, b):
        matrix = np.stack([ua * pa[2] - pa[0], va * pa[2] - pa[1], ub * pb[2] - pb[0], vb * pb[2] - pb[1]])
        _, _, vt = np.linalg.svd(matrix)
        homogeneous = vt[-1]
        if abs(float(homogeneous[3])) < 1e-9:
            continue
        xyz = homogeneous[:3] / homogeneous[3]
        if not np.all(np.isfinite(xyz)):
            continue
        xyz_h = np.append(xyz, 1.0)
        projected_a = pa @ xyz_h
        projected_b = pb @ xyz_h
        if projected_a[2] <= 1e-9 or projected_b[2] <= 1e-9:
            continue
        pixel_a = projected_a[:2] / projected_a[2]
        pixel_b = projected_b[:2] / projected_b[2]
        error = max(float(np.linalg.norm(pixel_a - [ua, va])), float(np.linalg.norm(pixel_b - [ub, vb])))
        if not np.isfinite(error) or error > float(max_reprojection_error_px):
            continue
        points.append(xyz)
        reprojection_errors.append(error)
    if len(points) < max(1, int(min_correspondences)):
        return None
    cloud = np.stack(points)
    center = np.median(cloud, axis=0)
    spread = float(np.median(np.linalg.norm(cloud - center, axis=1)))
    return {
        "center_world": center,
        "surface_spread_m": spread,
        "median_reprojection_error_px": float(np.median(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
        "valid_correspondences": int(len(points)),
    }


class SpatialToolBus:
    """Small registry used by the runner; providers may be unavailable."""

    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}
        self.last_results: list[SpatialToolResult] = []

    def register(self, name: str, provider: Any) -> None:
        self.providers[str(name)] = provider

    def infer(self, **kwargs: Any) -> SpatialBelief:
        results: list[SpatialToolResult] = []
        for name, provider in self.providers.items():
            try:
                result = provider.infer(**kwargs)
            except Exception as exc:  # provider failures are typed, not semantic
                result = SpatialToolResult(source=name, instance_id=kwargs.get("instance_id"), frame_id=kwargs.get("frame_id"), health=SpatialHealth.SENSOR_FAULT, diagnostics={"error": f"{type(exc).__name__}: {exc}"})
            if isinstance(result, SpatialToolResult):
                results.append(result)
        self.last_results = results
        return fuse_spatial_results(results, frame_id=kwargs.get("frame_id"), instance_id=kwargs.get("instance_id"))
