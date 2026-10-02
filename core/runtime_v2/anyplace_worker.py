"""Minimal AnyPlace subprocess protocol endpoint.

The worker intentionally refuses to fabricate geometry when the external model
runner is not configured.  A deployment can replace this endpoint with the
official AnyPlace inference entry point while preserving the parent/child NPZ
and response JSON contract.
"""
from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import numpy as np


_OFFICIAL_BACKENDS = {}


def _official_candidates(request: dict, parent: np.ndarray, child: np.ndarray) -> dict:
    """Run the installed official backend when explicitly configured.

    The subprocess receives world-frame parent/child points.  AnyPlace returns
    relative object placement transforms in that same metric frame.  We keep
    the full rotation in diagnostics and do not invent an EEF pose because the
    current executor is translation-only.
    """
    config = request.get("official_backend")
    if not isinstance(config, dict) or not config.get("anyplace_root") or not config.get("config_path"):
        return {
            "candidates": [],
            "diagnostics": {
                "reason": "official_anyplace_runner_not_configured",
                "parent_points": int(len(parent)),
                "child_points": int(len(child)),
            },
        }
    root = str(config["anyplace_root"])
    config_path = str(config["config_path"])
    module_path = Path("/root/autodl-tmp/OpenETA/tools/anyplace_core.py")
    if not module_path.exists():
        return {"candidates": [], "diagnostics": {"reason": "official_adapter_missing"}}
    cache_key = (root, config_path)
    backend = _OFFICIAL_BACKENDS.get(cache_key)
    if backend is None:
        spec = importlib.util.spec_from_file_location("show_harness_anyplace_core", module_path)
        if spec is None or spec.loader is None:
            return {"candidates": [], "diagnostics": {"reason": "official_adapter_load_failed"}}
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        backend = module.AnyPlaceBackend(anyplace_root=root, config_path=config_path)
        _OFFICIAL_BACKENDS[cache_key] = backend
    loaded = backend._get_loaded_backend()
    raw = backend._predict_with_loaded_backend(
        backend=loaded,
        object_pcd=child.astype(np.float32, copy=False),
        placement_region_pcd=parent.astype(np.float32, copy=False),
    )
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    transforms = np.asarray(raw, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        return {"candidates": [], "diagnostics": {"reason": "official_invalid_pose_batch"}}
    candidates = []
    for index, transform in enumerate(transforms[: int(request.get("top_k", 5))]):
        if not np.isfinite(transform).all():
            continue
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        candidates.append({
            "candidate_id": f"anyplace-{index}",
            "object_pose_world": [float(value) for value in transform[:3, :].reshape(-1)],
            "rotation": [[float(value) for value in row] for row in rotation],
            "orientation_supported": False,
            "reachable": False,
            "diagnostics": {
                "source": "official_anyplace",
                "translation_world": [float(value) for value in translation],
                "pose_convention": "p_placed = R @ p_current + t",
            },
        })
    return {
        "candidates": candidates,
        "diagnostics": {
            "reason": "official_anyplace_shadow",
            "parent_points": int(len(parent)),
            "child_points": int(len(child)),
        },
    }


def main(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    point_clouds = np.load(request["point_clouds"])
    parent = point_clouds["parent"]
    child = point_clouds["child"]
    try:
        payload = _official_candidates(request, parent, child)
    except Exception as exc:  # pragma: no cover - exercised in isolated env
        payload = {
            "candidates": [],
            "diagnostics": {
                "reason": "official_anyplace_inference_failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[-400:],
                "parent_points": int(len(parent)),
                "child_points": int(len(child)),
            },
        }
    output = request.get("output")
    if output:
        Path(output).write_text(json.dumps(payload), encoding="utf-8")
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
