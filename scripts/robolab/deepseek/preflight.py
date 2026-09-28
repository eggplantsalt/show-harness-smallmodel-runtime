#!/usr/bin/env python3
"""Offline checks by default; --api explicitly sends one SiliconFlow request.

This script never imports Isaac Lab or launches the simulator. Run with RoboLab's
Python interpreter. API credentials are read from the environment / a local env file.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="store_true", help="Send one billed vision request; no simulation")
    parser.add_argument("--robot-config", type=Path, default=ROOT / "configs/robot_robolab_deepseek.yaml")
    parser.add_argument("--output", type=Path, help="Optional JSON report, without credentials")
    args = parser.parse_args()
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    check("python", sys.version_info[:2] == (3, 11), sys.version.split()[0])
    robolab = Path(os.environ.get("ROBOLAB_ROOT", str(ROOT.parent / "RoboLab"))).expanduser()
    check("RoboLab", (robolab / "robolab").is_dir(), str(robolab))
    for module, distribution in (
        ("numpy", "numpy"), ("requests", "requests"), ("yaml", "PyYAML"),
        ("PIL", "Pillow"), ("imageio", "imageio"), ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("cv2", "opencv-python"), ("torch", "torch"), ("isaacsim", "isaacsim"),
        ("isaaclab", "isaaclab"), ("zerorpc", "zerorpc"),
    ):
        # find_spec does not import these top-level modules (Isaac must not be imported here).
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "version metadata unavailable"
        check(module, available, version if available else "missing from this interpreter")
    for lib in ("libGLU.so.1", "libcuda.so.1"):
        try:
            ctypes.CDLL(lib)
            check(lib, True, "loadable")
        except OSError:
            check(lib, False, "not loadable; see native dependencies in the guide")
    check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "missing")
    asset_dir = ROOT / "assets/robolab_franka"
    for relative in ("panda_short_finger.usda", "Props/panda_short_finger_left.usda",
                     "Props/panda_short_finger_right.usda"):
        # Asset references are checked below by scanning local USD reference names,
        # rather than assuming a particular generated Props filename.
        if relative == "panda_short_finger.usda":
            check("Panda asset", (asset_dir / relative).is_file(), str(asset_dir / relative))
    check("Panda Props", (asset_dir / "Props").is_dir() and any((asset_dir / "Props").glob("*.usd*")),
          str(asset_dir / "Props"))
    check("robot config", args.robot_config.is_file(), str(args.robot_config))

    if importlib.util.find_spec("yaml") is not None and args.robot_config.is_file():
        from core.config import load_secrets_env, load_yaml, resolve_vlm_config

        load_secrets_env()
        env_file = Path(os.environ.get("SILICONFLOW_ENV_FILE", str(Path.home() / ".config/show-harness/siliconflow.env"))).expanduser()
        load_secrets_env(env_file)
        cfg = load_yaml(args.robot_config)
        vlm = resolve_vlm_config(cfg)
        official = urlparse(vlm["base_url"]).scheme == "https" and urlparse(vlm["base_url"]).netloc == "api.siliconflow.cn"
        check("SiliconFlow endpoint", official, vlm["base_url"])
        check("model", bool(str(vlm.get("model", "")).strip()), str(vlm.get("model", "")))
        has_key = bool(os.environ.get("SILICONFLOW_API_KEY", "").strip())
        check("SILICONFLOW_API_KEY", has_key, "set (value hidden)" if has_key else "not set")
        # A zero external-target pose channel is intentional: images + robot proprioception only.
        check("episode time limit", cfg.get("episode_length_s") is None, "must use the task's authored time limit")
        if args.api:
            if not official or not has_key:
                check("vision API", False, "fix endpoint / credential configuration first")
            elif not all(importlib.util.find_spec(m) for m in ("requests", "PIL", "numpy")):
                check("vision API", False, "install requests, Pillow and numpy first")
            else:
                from PIL import Image, ImageDraw
                from core.vlm.vlm_client import VLMClient
                from core.record.images import image_to_data_url

                picture = Image.new("RGB", (384, 256), "white")
                draw = ImageDraw.Draw(picture)
                draw.rectangle((35, 65, 160, 195), fill="red")
                draw.ellipse((235, 70, 340, 175), fill="blue")
                import numpy as np

                client = VLMClient(
                    base_url=vlm["base_url"], model=vlm["model"], api_key=vlm["api_key"],
                    timeout_s=60, max_tokens=256, temperature=0,
                    provider="openai", api_dialect="siliconflow", max_retries=2,
                    chat_template_kwargs={"enable_thinking": False},
                )
                client.session.trust_env = False
                payload = client._finalize_payload({
                    "model": vlm["model"], "max_tokens": 256, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe the two colored shapes and which is on the left. Answer briefly in English."},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(np.asarray(picture))}},
                    ]}],
                })
                try:
                    response = client._post_chat(payload)
                    answer = str(response["choices"][0]["message"].get("content") or "")
                    check("vision API", bool(answer.strip()), answer[:500])
                    print("Expected: red square on the left, blue circle on the right. Verify the answer yourself.")
                    print("This checks API/image transport only; it is not a robot task success.")
                except Exception as exc:
                    message = str(exc).replace(vlm["api_key"], "[REDACTED]")
                    check("vision API", False, message[:700])
    elif args.api:
        check("vision API", False, "configuration / YAML dependency missing")

    report = {"simulation_started": False, "api_requested": args.api, "checks": checks,
              "passed": all(item["ok"] for item in checks)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
