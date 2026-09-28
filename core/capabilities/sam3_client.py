"""Small resilient client for the existing OpenETA SAM3 MCP service."""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


class Sam3Client:
    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8773/sse",
        python: str = "/root/autodl-tmp/openeta-services/sam3/.venv/bin/python",
        timeout_s: float = 45.0,
    ) -> None:
        self.url = str(url)
        self.python = str(python)
        self.timeout_s = float(timeout_s)
        self.bridge = Path(__file__).resolve().parents[2] / "scripts/capabilities/sam3_bridge.py"
        self._process: Optional[subprocess.Popen[str]] = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        self._process = subprocess.Popen(
            [self.python, str(self.bridge), "--url", self.url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=dict(os.environ),
        )

        def read_lines() -> None:
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {"success": False, "error": "invalid_sam3_bridge_json"}
                self._responses.put(payload)

        self._reader = threading.Thread(target=read_lines, name="sam3-bridge-reader", daemon=True)
        self._reader.start()

    def segment(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float = 0.5,
    ) -> dict[str, Any]:
        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        with io.BytesIO() as buffer:
            Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        request = {
            "tool": "segment",
            "arguments": {
                "image_base64": encoded,
                "prompt": str(prompt),
                "image_format": "png",
                "confidence_threshold": float(confidence_threshold),
            },
        }
        with self._lock:
            try:
                self._start()
                if self._process is None or self._process.stdin is None:
                    return {"success": False, "error": "sam3_bridge_not_started"}
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                return self._responses.get(timeout=self.timeout_s)
            except (OSError, queue.Empty, BrokenPipeError) as exc:
                self.close()
                return {"success": False, "error": f"sam3_client: {exc}"}

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
