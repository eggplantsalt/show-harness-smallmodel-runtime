"""Optional capability providers used by the V2.1 spatial bus.

Imports and checkpoint loading are lazy so V1 and unit tests do not acquire a
GPU or require MoGe/GraspGenX dependencies.
"""
from __future__ import annotations

import time
import base64
import atexit
import json
import os
import select
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .types import SpatialHealth, SpatialToolResult


def _prepared_image_fov_x_deg(
    calibration: dict[str, Any], *, image_width: int, image_height: int
) -> float:
    """Derive horizontal FOV after a known quarter-turn and resize."""
    raw_width = int(calibration["width"])
    raw_height = int(calibration["height"])
    raw_fovy = float(calibration["fovy_deg"])
    if min(raw_width, raw_height, int(image_width), int(image_height)) <= 0:
        raise ValueError("camera dimensions must be positive")
    tangent = np.tan(np.deg2rad(raw_fovy) / 2.0)
    raw_fovx = 2.0 * np.arctan(tangent * raw_width / raw_height)
    view_fovy = raw_fovx if int(calibration.get("rotation_degrees", 0)) % 180 else np.deg2rad(raw_fovy)
    view_aspect = float(image_width) / float(image_height)
    return float(np.rad2deg(2.0 * np.arctan(np.tan(view_fovy / 2.0) * view_aspect)))


class MogeDepthProvider:
    source = "moge2"

    def __init__(self, *, repo_dir: str = "/root/autodl-tmp/openeta-services/MoGe", checkpoint: str = "Ruicheng/moge-3-vitl", device: str = "cuda:1", max_latency_s: float = 0.5, model_version: str = "v3", service_python: Optional[str] = None) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint = checkpoint
        self.device = device
        self.max_latency_s = float(max_latency_s)
        self.model_version = str(model_version or "v3").lower()
        if self.model_version == "v2_fallback":
            self.model_version = "v2"
        if self.model_version not in {"v2", "v3"}:
            raise ValueError(f"unsupported MoGe model version: {self.model_version}")
        self.source = f"moge{self.model_version[-1]}"
        self.model = None
        self.service_python = str(service_python) if service_python else None
        self._process: Optional[subprocess.Popen[str]] = None
        self._service_lock = threading.Lock()
        self.load_error: Optional[str] = None

    def load(self) -> None:
        if self.model is not None or self._process is not None or self.load_error:
            return
        try:
            if self.service_python:
                worker = Path(__file__).with_name("moge_worker.py")
                command = [
                    self.service_python, "-u", str(worker),
                    "--repo-dir", str(self.repo_dir), "--checkpoint", str(self.checkpoint),
                    "--device", self.device, "--model-version", self.model_version,
                ]
                env = dict(os.environ)
                env.setdefault("PYTHONUNBUFFERED", "1")
                self._process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=None, text=True, bufsize=1, env=env,
                )
                ready = self._read_service_line(timeout_s=240.0)
                if ready.get("status") != "ready":
                    raise RuntimeError(str(ready.get("error", "MoGe worker failed to initialize")))
                atexit.register(self._stop_service)
                return
            import sys
            import torch
            if str(self.repo_dir) not in sys.path:
                sys.path.insert(0, str(self.repo_dir))
            if self.model_version == "v2":
                from moge.model.v2 import MoGeModel
            else:
                from moge.model.v3 import MoGeModel
            self.model = MoGeModel.from_pretrained(self.checkpoint).to(torch.device(self.device)).eval()
        except Exception as exc:
            self._stop_service()
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _read_service_line(self, *, timeout_s: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("MoGe worker is not running")
        ready, _, _ = select.select([process.stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"MoGe worker response exceeded {timeout_s:.0f}s")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"MoGe worker exited (code={process.poll()})")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("MoGe worker returned a non-object response")
        return value

    def _stop_service(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def close(self) -> None:
        """Stop the isolated worker when the owning runtime shuts down."""
        self._stop_service()

    @staticmethod
    def _mask_as_rle(mask: np.ndarray) -> dict[str, Any]:
        spans: list[list[int]] = []
        for y, row in enumerate(mask):
            xs = np.flatnonzero(row)
            if not len(xs):
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(xs) > 1) + 1]
            ends = np.r_[starts[1:] - 1, len(xs) - 1]
            spans.extend([[int(y), int(xs[a]), int(xs[b])] for a, b in zip(starts, ends)])
        return {"format": "row_span_rle", "shape": list(mask.shape), "rle": spans}

    def _infer_service(self, *, image: Any, mask: Any, bbox_xyxy: Any, instance_id: Optional[str], frame_id: Optional[int], camera_calibration: Any, started: float) -> SpatialToolResult:
        if mask is None:
            return SpatialToolResult(source=self.source, instance_id=instance_id, frame_id=frame_id, health=SpatialHealth.UNKNOWN, diagnostics={"reason": "grounded_instance_mask_required"})
        try:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("image must be HxWx3")
            selected = CoTrackerOnlineProvider._decode_mask(mask, array.shape[:2])
            if selected is None:
                raise ValueError("instance mask missing, empty, or shape-mismatched")
            if not selected.any():
                raise ValueError("instance mask is empty")
            request = {
                "image_shape": list(array.shape),
                "image_rgb_u8_b64": base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii"),
                "mask": self._mask_as_rle(selected),
                "instance_id": instance_id,
                "frame_id": frame_id,
            }
            calibration = camera_calibration if isinstance(camera_calibration, dict) else {}
            try:
                request["fov_x_deg"] = _prepared_image_fov_x_deg(
                    calibration, image_width=int(array.shape[1]), image_height=int(array.shape[0])
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return SpatialToolResult(source=self.source, instance_id=instance_id, frame_id=frame_id, health=SpatialHealth.UNKNOWN, diagnostics={"reason": "known_camera_fov_required"})
            with self._service_lock:
                self.load()
                if self._process is None or self._process.stdin is None:
                    raise RuntimeError(self.load_error or "MoGe worker unavailable")
                self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                self._process.stdin.flush()
                response = self._read_service_line(timeout_s=120.0)
            elapsed = time.perf_counter() - started
            if response.get("status") != "ok":
                raise RuntimeError(str(response.get("error", "MoGe inference failed")))
            points = response.get("points", [])
            return SpatialToolResult(
                source=self.source, instance_id=instance_id, frame_id=frame_id,
                target_points_camera=tuple(tuple(float(x) for x in row) for row in points),
                target_points_pixels=tuple(tuple(float(x) for x in row) for row in response.get("pixels", [])),
                covariance=None, confidence=0.5, health=SpatialHealth.VALID, stale=False,
                diagnostics={**response.get("diagnostics", {}), "inference_latency_s": elapsed, "service_isolated": True},
            )
        except Exception as exc:
            return SpatialToolResult(source=self.source, instance_id=instance_id, frame_id=frame_id, health=SpatialHealth.SENSOR_FAULT, diagnostics={"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - started})

    def infer(self, *, image: Any, mask: Any = None, bbox_xyxy: Any = None, instance_id: Optional[str] = None, frame_id: Optional[int] = None, camera_calibration: Any = None, **_: Any) -> SpatialToolResult:
        if self.service_python:
            return self._infer_service(image=image, mask=mask, bbox_xyxy=bbox_xyxy, instance_id=instance_id, frame_id=frame_id, camera_calibration=camera_calibration, started=time.perf_counter())
        self.load()
        if self.model is None:
            return SpatialToolResult(source=self.source, instance_id=instance_id, frame_id=frame_id, health=SpatialHealth.SENSOR_FAULT, diagnostics={"error": self.load_error or "not loaded"})
        started = time.perf_counter()
        try:
            import torch
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError("image must be HxWx3")
            tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
            with torch.inference_mode():
                calibration = camera_calibration if isinstance(camera_calibration, dict) else {}
                width = int(array.shape[1])
                height = int(array.shape[0])
                fov_x = None
                if calibration:
                    fov_x = _prepared_image_fov_x_deg(
                        calibration, image_width=width, image_height=height
                    )
                if fov_x is None:
                    raise ValueError("known camera FOV calibration required for metric MoGe")
                output = self.model.infer(tensor, fov_x=fov_x)
            points = output["points"].detach().float().cpu().numpy()
            valid = output.get("mask")
            valid = valid.detach().cpu().numpy().astype(bool) if valid is not None else np.ones(points.shape[:2], dtype=bool)
            if valid.ndim > 2:
                valid = np.squeeze(valid)
            selected = CoTrackerOnlineProvider._decode_mask(mask, valid.shape) if mask is not None else None
            if selected is None:
                raise ValueError("MoGe requires a grounded instance mask")
            valid &= selected
            ys, xs = np.nonzero(valid)
            samples = points[ys, xs]
            if len(samples) < 4:
                raise ValueError("no valid target point samples")
            center = np.median(samples, axis=0)
            extent = np.quantile(samples, 0.95, axis=0) - np.quantile(samples, 0.05, axis=0)
            elapsed = time.perf_counter() - started
            return SpatialToolResult(
                source=self.source,
                instance_id=instance_id,
                frame_id=frame_id,
                target_points_camera=tuple(tuple(float(x) for x in row) for row in samples[::max(1, len(samples) // 64)]),
                target_points_pixels=tuple(tuple(float(x) for x in row) for row in np.column_stack((xs, ys))[::max(1, len(samples) // 64)]),
                # Point-cloud extent is shape, not measurement uncertainty.
                # Leave covariance unspecified so the fusion layer applies its
                # explicit conservative prior instead of treating extent as sigma.
                covariance=None,
                confidence=0.5,
                health=SpatialHealth.VALID,
                stale=False,
                diagnostics={"inference_latency_s": elapsed, "sample_count": int(len(samples)), "instance_mask_applied": True, "point_extent_m": extent.tolist(), "uncertainty_source": "not_estimated_by_single_frame_moge", "intrinsics": output.get("intrinsics").detach().cpu().numpy().tolist() if output.get("intrinsics") is not None else None},
            )
        except Exception as exc:
            return SpatialToolResult(source=self.source, instance_id=instance_id, frame_id=frame_id, health=SpatialHealth.SENSOR_FAULT, diagnostics={"error": f"{type(exc).__name__}: {exc}", "latency_s": time.perf_counter() - started})


class CoTrackerOnlineProvider:
    """GPU-bound online point tracker for a single camera and object epoch.

    The provider samples queries only inside a grounded instance mask. It keeps
    a bounded stream window and returns correspondences at CoTracker's native
    update cadence; missing masks, visibility, or model weights are explicit
    unavailable results, never a bbox-based substitute.
    """

    def __init__(
        self,
        *,
        repo_dir: str = "/root/autodl-tmp/openeta-services/CoTracker",
        checkpoint: str = "/root/autodl-tmp/openeta-services/checkpoints/cotracker3/scaled_online.pth",
        device: str = "cuda:1",
        window_len: int = 16,
        max_points: int = 64,
        min_visible_points: int = 4,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint = Path(checkpoint)
        self.device = str(device)
        self.window_len = max(8, int(window_len))
        self.max_points = max(4, int(max_points))
        self.min_visible_points = max(2, int(min_visible_points))
        self.model = None
        self.load_error: Optional[str] = None
        self._key: Optional[tuple[str, Optional[str], int]] = None
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_len)
        self._frame_ids: deque[int] = deque(maxlen=self.window_len)
        self._step = self.window_len // 2
        self._initialized = False
        self._next_update = self._step * 2
        self._query_points: Optional[np.ndarray] = None
        self._query_frame_id: Optional[int] = None
        self._initial_mask: Optional[np.ndarray] = None
        self._total_frames = 0

    def load(self) -> None:
        if self.model is not None or self.load_error:
            return
        try:
            import sys
            import torch

            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"CoTracker checkpoint not found: {self.checkpoint}")
            if str(self.repo_dir) not in sys.path:
                sys.path.insert(0, str(self.repo_dir))
            from cotracker.predictor import CoTrackerOnlinePredictor

            self.model = CoTrackerOnlinePredictor(
                checkpoint=str(self.checkpoint), offline=False,
                window_len=self.window_len,
            ).to(torch.device(self.device)).eval()
            self._step = int(self.model.step)
            self._frames = deque(maxlen=self.window_len)
            self._frame_ids = deque(maxlen=self.window_len)
            self._next_update = self._step * 2
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _decode_mask(mask: Any, shape: tuple[int, int]) -> Optional[np.ndarray]:
        if isinstance(mask, np.ndarray):
            result = np.asarray(mask, dtype=bool)
            return result if result.shape == shape and result.any() else None
        if not isinstance(mask, dict) or mask.get("format") != "row_span_rle":
            return None
        if tuple(mask.get("shape", ())) != tuple(shape):
            return None
        result = np.zeros(shape, dtype=bool)
        spans = mask.get("rle")
        if not isinstance(spans, list):
            return None
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 3:
                continue
            try:
                y, x0, x1 = (int(value) for value in span)
            except (TypeError, ValueError):
                continue
            if 0 <= y < shape[0] and 0 <= x0 <= x1 < shape[1]:
                result[y, x0 : x1 + 1] = True
        return result if result.any() else None

    def _reset_stream(self, key: tuple[str, Optional[str], int]) -> None:
        self._key = key
        self._frames.clear()
        self._frame_ids.clear()
        self._initialized = False
        self._next_update = self._step * 2
        self._query_points = None
        self._query_frame_id = None
        self._initial_mask = None
        self._total_frames = 0

    def reset(self) -> None:
        """Forget online state at episode or identity boundaries."""
        self._key = None
        self._frames.clear()
        self._frame_ids.clear()
        self._initialized = False
        self._next_update = self._step * 2
        self._query_points = None
        self._query_frame_id = None
        self._initial_mask = None
        self._total_frames = 0

    def infer(
        self,
        *,
        image: Any,
        mask: Any,
        camera: str = "wrist",
        instance_id: Optional[str] = None,
        grasp_epoch: int = 0,
        frame_id: Optional[int] = None,
        **_: Any,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or frame_id is None:
            return {"health": "UNAVAILABLE", "reason": "invalid_image_or_frame"}
        key = (str(camera).lower(), instance_id, int(grasp_epoch))
        if self._key != key:
            self._reset_stream(key)
        decoded = self._decode_mask(mask, array.shape[:2])
        if not self._frames and decoded is None:
            return {"health": "UNAVAILABLE", "reason": "grounded_instance_mask_required"}
        if self._total_frames == 0:
            self._initial_mask = decoded
            self._query_frame_id = int(frame_id)
        self.load()
        if self.model is None:
            return {"health": "SENSOR_FAULT", "reason": self.load_error or "model_unavailable"}
        self._frames.append(np.ascontiguousarray(array, dtype=np.uint8))
        self._frame_ids.append(int(frame_id))
        self._total_frames += 1
        count = self._total_frames
        if not self._initialized:
            if count < self._step:
                return {"health": "WARMING", "frames_seen": count, "required_frames": self._step}
            initial_mask = self._initial_mask
            if initial_mask is None:
                return {"health": "UNAVAILABLE", "reason": "initial_instance_mask_missing"}
            ys, xs = np.nonzero(initial_mask)
            if len(xs) < self.min_visible_points:
                return {"health": "UNAVAILABLE", "reason": "mask_has_too_few_pixels"}
            sample_ids = np.linspace(0, len(xs) - 1, min(self.max_points, len(xs))).round().astype(int)
            self._query_points = np.column_stack((xs[sample_ids], ys[sample_ids])).astype(np.float32)
            self._run_model(self._stack_frames()[:, : self._step], first=True)
            self._initialized = True
            self._next_update = count + self._step
            return {"health": "INITIALIZED", "query_frame_id": self._query_frame_id, "query_count": len(self._query_points)}
        if count < self._next_update:
            return {"health": "WARMING", "frames_seen": count, "next_update": self._next_update}
        if count < self._step * 2:
            return {"health": "WARMING", "frames_seen": count, "required_frames": self._step * 2}
        try:
            import torch

            video = self._stack_frames()
            with torch.inference_mode():
                tracks, visibility = self.model(video_chunk=video, is_first_step=False)
            xy = tracks[0, -1].detach().float().cpu().numpy()
            visible = visibility[0, -1].detach().cpu().numpy().astype(bool).reshape(-1)
            query = self._query_points
            if query is None or len(xy) != len(query) or len(visible) != len(query):
                return {"health": "UNAVAILABLE", "reason": "track_shape_mismatch"}
            good = visible & np.isfinite(xy).all(axis=1)
            if int(good.sum()) < self.min_visible_points:
                return {"health": "AMBIGUOUS", "reason": "too_few_visible_tracks", "visible_count": int(good.sum())}
            self._next_update += self._step
            elapsed = time.perf_counter() - started
            return {
                "health": "VALID",
                "camera": key[0],
                "instance_id": instance_id,
                "grasp_epoch": int(grasp_epoch),
                "query_frame_id": self._query_frame_id,
                "frame_id": int(frame_id),
                "source_frame_id": int(frame_id),
                "query_points_xy": query[good].tolist(),
                "current_points_xy": xy[good].tolist(),
                "visible_count": int(good.sum()),
                "inference_latency_s": elapsed,
            }
        except Exception as exc:
            self._reset_stream(key)
            return {"health": "SENSOR_FAULT", "reason": f"{type(exc).__name__}: {exc}"}

    def _stack_frames(self):
        import torch

        frames = list(self._frames)[-self.window_len :]
        video = np.stack(frames).astype(np.float32)
        return torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)

    def _run_model(self, video: Any, *, first: bool):
        import torch

        points = np.asarray(self._query_points, dtype=np.float32)
        queries = np.concatenate(
            [np.zeros((len(points), 1), dtype=np.float32), points], axis=1
        )
        query_tensor = torch.from_numpy(queries).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            self.model(
                video_chunk=video, is_first_step=first,
                queries=query_tensor, grid_size=0,
            )
