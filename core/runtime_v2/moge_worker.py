"""Persistent isolated MoGe inference worker used by the V2 runtime.

The simulator environment can keep its CUDA/PyTorch dependencies untouched,
while this process owns MoGe and its matching inference dependencies on GPU1.
The line protocol is JSON in/JSON out; stdout is reserved for protocol data.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any


def _decode_mask(payload: Any, shape: tuple[int, int]):
    import numpy as np

    if not isinstance(payload, dict) or payload.get("format") != "row_span_rle":
        raise ValueError("grounded row-span instance mask required")
    if tuple(payload.get("shape", ())) != shape:
        raise ValueError("instance mask shape mismatch")
    spans = payload.get("rle")
    if not isinstance(spans, list):
        raise ValueError("invalid instance mask spans")
    mask = np.zeros(shape, dtype=bool)
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            raise ValueError("invalid row-span instance mask entry")
        y, x0, x1 = (int(value) for value in span)
        if not (0 <= y < shape[0] and 0 <= x0 <= x1 < shape[1]):
            raise ValueError("instance mask span is out of bounds")
        mask[y, x0 : x1 + 1] = True
    if not mask.any():
        raise ValueError("instance mask is empty")
    return mask


def _write(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-version", choices=("v2", "v3"), required=True)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch

        repo = str(Path(args.repo_dir).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        with contextlib.redirect_stdout(sys.stderr):
            if args.model_version == "v2":
                from moge.model.v2 import MoGeModel
            else:
                from moge.model.v3 import MoGeModel
            model = MoGeModel.from_pretrained(args.checkpoint).to(torch.device(args.device)).eval()
        _write({"status": "ready", "model_version": args.model_version, "device": args.device})
    except Exception as exc:
        _write({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    for line in sys.stdin:
        try:
            request = json.loads(line)
            shape = tuple(int(x) for x in request["image_shape"])
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError("image must be HxWx3")
            raw = __import__("base64").b64decode(request["image_rgb_u8_b64"], validate=True)
            if len(raw) != shape[0] * shape[1] * shape[2]:
                raise ValueError("image payload length does not match shape")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            mask = _decode_mask(request.get("mask"), shape[:2])
            started = time.perf_counter()
            tensor = torch.from_numpy(image.copy()).to(device=args.device, dtype=torch.float32).permute(2, 0, 1) / 255.0
            fov_x = float(request["fov_x_deg"])
            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                output = model.infer(tensor, fov_x=fov_x)
            points = output["points"].detach().float().cpu().numpy()
            valid = output.get("mask")
            valid = valid.detach().cpu().numpy().astype(bool) if valid is not None else np.ones(points.shape[:2], dtype=bool)
            valid = np.squeeze(valid)
            if valid.shape != points.shape[:2]:
                raise ValueError("MoGe validity map shape mismatch")
            if mask.shape != valid.shape:
                raise ValueError("instance mask does not match MoGe point map")
            valid &= mask
            ys, xs = np.nonzero(valid)
            if len(xs) < 4:
                raise ValueError("fewer than four valid target point samples")
            indices = np.linspace(0, len(xs) - 1, min(64, len(xs))).round().astype(int)
            sampled = points[ys[indices], xs[indices]]
            if not np.isfinite(sampled).all():
                raise ValueError("MoGe returned non-finite target points")
            extent = np.quantile(points[valid], 0.95, axis=0) - np.quantile(points[valid], 0.05, axis=0)
            intrinsics = output.get("intrinsics")
            intrinsics = intrinsics.detach().float().cpu().numpy().tolist() if intrinsics is not None else None
            _write({
                "status": "ok",
                "points": sampled.tolist(),
                "pixels": np.column_stack((xs[indices], ys[indices])).astype(float).tolist(),
                "diagnostics": {
                    "sample_count": int(len(xs)),
                    "instance_mask_applied": True,
                    "point_extent_m": extent.tolist(),
                    "uncertainty_source": "not_estimated_by_single_frame_moge",
                    "intrinsics": intrinsics,
                    "worker_inference_s": time.perf_counter() - started,
                },
            })
        except Exception as exc:
            _write({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
