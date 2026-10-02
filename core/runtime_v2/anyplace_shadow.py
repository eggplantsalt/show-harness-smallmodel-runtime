"""Isolated AnyPlace shadow adapter.

AnyPlace is intentionally not imported by the main VCR process.  Its released
implementation assumes a different Torch environment and contains hard-coded
CUDA placement.  This adapter exchanges only JSON/NPZ over a short-lived
GPU1 subprocess and exposes typed candidates without granting action authority.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .types import PlacementCandidate


class AnyPlaceShadowProvider:
    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "shadow",
        python: str = "/root/autodl-tmp/openeta-services/anyplace-env/bin/python",
        worker: Optional[str] = None,
        command: Optional[Sequence[str]] = None,
        device: str = "cuda:1",
        timeout_s: float = 20.0,
        top_k: int = 5,
        supports_orientation: bool = False,
        anyplace_root: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or "shadow").lower()
        if self.mode != "shadow":
            raise ValueError("AnyPlace is shadow-only until the V2.2 activation gates pass")
        self.python = str(python)
        self.worker = str(worker or Path(__file__).with_name("anyplace_worker.py"))
        self.command = tuple(str(item) for item in command) if command else None
        self.device = str(device)
        self.timeout_s = max(0.1, float(timeout_s))
        self.top_k = max(1, int(top_k))
        self.supports_orientation = bool(supports_orientation)
        self.anyplace_root = str(anyplace_root) if anyplace_root else None
        self.config_path = str(config_path) if config_path else None
        self.last_result: dict[str, Any] = {"health": "UNKNOWN", "reason": "not_called"}

    @classmethod
    def from_config(cls, section: Any) -> Optional["AnyPlaceShadowProvider"]:
        if not isinstance(section, dict) or not bool(section.get("enabled", False)):
            return None
        return cls(
            enabled=True,
            mode=str(section.get("mode", "shadow")),
            python=str(section.get("python", "/root/autodl-tmp/openeta-services/anyplace-env/bin/python")),
            worker=section.get("worker"),
            command=section.get("command"),
            device=str(section.get("device", "cuda:1")),
            timeout_s=float(section.get("timeout_s", 20.0)),
            top_k=int(section.get("top_k", 5)),
            supports_orientation=bool(section.get("supports_orientation", False)),
            anyplace_root=section.get("anyplace_root"),
            config_path=section.get("config_path"),
        )

    @staticmethod
    def _array(value: Any) -> Optional[np.ndarray]:
        try:
            result = np.asarray(value, dtype=np.float32).reshape(-1, 3)
        except (TypeError, ValueError):
            return None
        return result if len(result) and np.all(np.isfinite(result)) else None

    def _parse_candidates(self, payload: Any, *, frame_id: Optional[int], instance_id: Optional[str]) -> list[PlacementCandidate]:
        values = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return []
        result: list[PlacementCandidate] = []
        for index, item in enumerate(values[: self.top_k]):
            if not isinstance(item, dict):
                continue
            pose = item.get("eef_pose_world") or item.get("object_pose_world")
            if isinstance(pose, (list, tuple)):
                pose_tuple = tuple(float(value) for value in pose if isinstance(value, (int, float)))
            else:
                pose_tuple = None
            rotation = item.get("rotation") or item.get("rotation_matrix")
            orientation_supported = bool(item.get("orientation_supported", self.supports_orientation))
            diagnostics = dict(item.get("diagnostics", {})) if isinstance(item.get("diagnostics"), dict) else {}
            if rotation is not None:
                diagnostics["predicted_rotation"] = rotation
            if not orientation_supported:
                diagnostics["rejection"] = "UNSUPPORTED_ORIENTATION"
            object_pose = item.get("object_pose_world")
            if isinstance(object_pose, (list, tuple)):
                try:
                    object_pose_tuple = tuple(float(value) for value in object_pose)
                except (TypeError, ValueError):
                    object_pose_tuple = None
            else:
                object_pose_tuple = None
            uncertainty = item.get("uncertainty_m")
            if isinstance(uncertainty, (list, tuple)):
                try:
                    uncertainty_tuple = tuple(float(value) for value in uncertainty)
                except (TypeError, ValueError):
                    uncertainty_tuple = None
            else:
                uncertainty_tuple = None
            result.append(
                PlacementCandidate(
                    candidate_id=str(item.get("candidate_id", f"anyplace-{index}")),
                    object_pose_world=object_pose_tuple,
                    eef_pose_world=pose_tuple,
                    source="anyplace_shadow",
                    reachable=item.get("reachable") if orientation_supported else False,
                    orientation_supported=orientation_supported,
                    collision_clearance_m=(float(item["collision_clearance_m"]) if item.get("collision_clearance_m") is not None else None),
                    containment_margin_m=(float(item["containment_margin_m"]) if item.get("containment_margin_m") is not None else None),
                    uncertainty_m=uncertainty_tuple,
                    diagnostics={**diagnostics, "frame_id": frame_id, "instance_id": instance_id},
                )
            )
        return result

    def infer(
        self,
        *,
        parent_points: Any,
        child_points: Any,
        frame_id: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> dict[str, Any]:
        parent = self._array(parent_points)
        child = self._array(child_points)
        if not self.enabled:
            self.last_result = {"health": "UNKNOWN", "reason": "disabled", "candidates": []}
            return self.last_result
        if parent is None or child is None:
            self.last_result = {"health": "UNKNOWN", "reason": "invalid_point_cloud", "candidates": []}
            return self.last_result
        if self.command:
            command = list(self.command)
        else:
            command = [self.python, self.worker]
        device = self.device.split(":", 1)[1] if ":" in self.device else self.device
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = device
        try:
            with tempfile.TemporaryDirectory(prefix="show_harness_anyplace_") as directory:
                root = Path(directory)
                npz_path = root / "point_clouds.npz"
                request_path = root / "request.json"
                output_path = root / "response.json"
                np.savez_compressed(npz_path, parent=parent, child=child)
                request_path.write_text(
                    json.dumps({
                        "point_clouds": str(npz_path),
                        "output": str(output_path),
                        "top_k": self.top_k,
                        "frame_id": frame_id,
                        "instance_id": instance_id,
                        "official_backend": {
                            "anyplace_root": self.anyplace_root,
                            "config_path": self.config_path,
                        } if self.anyplace_root and self.config_path else None,
                    }),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    command + [str(request_path)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"worker_exit_{completed.returncode}: {completed.stderr[-400:]}")
                payload = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else json.loads(completed.stdout)
                candidates = self._parse_candidates(payload, frame_id=frame_id, instance_id=instance_id)
                self.last_result = {
                    "health": "VALID" if candidates else "UNKNOWN",
                    "reason": "shadow_candidates" if candidates else "worker_returned_no_candidates",
                    "candidate_count": len(candidates),
                    "candidates": [candidate for candidate in candidates],
                    "device": f"cuda:{device}",
                }
        except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError) as exc:
            self.last_result = {
                "health": "UNKNOWN",
                "reason": f"shadow_provider_unavailable:{type(exc).__name__}",
                "error": str(exc),
                "candidates": [],
                "device": f"cuda:{device}",
            }
        return self.last_result

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "isolation": "subprocess",
            "device": self.device,
            "top_k": self.top_k,
            "supports_orientation": self.supports_orientation,
            "official_backend_configured": bool(self.anyplace_root and self.config_path),
            "last_result": self.last_result,
        }
