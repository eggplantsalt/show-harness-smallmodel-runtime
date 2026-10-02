"""Thin, dataset-free adapter for the vendored standard LIBERO benchmark.

Only BDDL files and the checked-in ``*.pruned_init`` states are needed.  The
demonstration HDF5 files are never opened.  Imports are deferred so importing
Show-Harness on a non-LIBERO machine remains safe.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def libero_root() -> Path:
    candidates = []
    if os.environ.get("LIBERO_DIR"):
        candidates.append(Path(os.environ["LIBERO_DIR"]).expanduser())
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[3] / "OpenETA" / "vendor" / "LIBERO",
            Path.cwd() / "OpenETA" / "vendor" / "LIBERO",
        ]
    )
    for root in candidates:
        if (root / "libero" / "libero" / "benchmark").is_dir():
            return root
    raise RuntimeError(
        "LIBERO checkout not found. Set LIBERO_DIR to the directory containing "
        "libero/libero/ (the existing checkout is OpenETA/vendor/LIBERO)."
    )


def ensure_libero_path() -> Path:
    root = libero_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return root


def _load_init_states(path: Path) -> np.ndarray:
    import torch

    # PyTorch 2.6 changed torch.load's default to weights_only=True.  LIBERO's
    # trusted pruned_init files are plain numpy arrays and need the legacy loader.
    try:
        states = torch.load(str(path), weights_only=False)
    except TypeError:  # torch < 2.6
        states = torch.load(str(path))
    return np.asarray(states)


@dataclass
class LiberoTaskHandle:
    env: Any
    suite_name: str
    task_id: int
    task_name: str
    task_description: str
    bddl_file: str
    init_states: np.ndarray
    init_state_index: int


def make_libero_task(
    suite_name: str = "LIBERO_OBJECT",
    task_id: int = 0,
    *,
    init_state_index: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    diagnostic_camera_depths: bool = False,
    horizon: int = 500,
    control_freq: int = 20,
    seed: int = 0,
    settle_steps: int = 5,
) -> LiberoTaskHandle:
    root = ensure_libero_path()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite_name = str(suite_name).upper()
    benchmark = get_benchmark(suite_name)()
    task_id = int(task_id)
    task = benchmark.get_task(task_id)
    # Resolve benchmark files relative to the checkout we just selected.  The
    # upstream package reads ~/.libero/config.yaml, which can silently point at
    # an old clone after a machine is reconfigured; no global config mutation is
    # needed for this adapter.
    benchmark_root = root / "libero" / "libero"
    init_states_path = benchmark_root / "init_files" / task.problem_folder / task.init_states_file
    init_states = _load_init_states(init_states_path)
    if not 0 <= int(init_state_index) < len(init_states):
        raise ValueError(f"init_state_index must be in [0, {len(init_states)-1}]")
    bddl_file = str(benchmark_root / "bddl_files" / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        controller="OSC_POSE",
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        camera_depths=bool(diagnostic_camera_depths),
        control_freq=int(control_freq),
        horizon=int(horizon),
        ignore_done=False,
    )
    env.seed(int(seed))
    env.reset()
    obs = env.set_init_state(init_states[int(init_state_index)])
    for _ in range(max(0, int(settle_steps))):
        obs, _reward, _done, _info = env.step(np.zeros(7, dtype=np.float32))
    env._showharness_last_obs = obs
    return LiberoTaskHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_name=task.name,
        task_description=task.language,
        bddl_file=bddl_file,
        init_states=init_states,
        init_state_index=int(init_state_index),
    )


def reset_libero(
    env: Any,
    init_state: np.ndarray,
    settle_steps: int = 5,
    hold_action: Optional[np.ndarray] = None,
):
    env.reset()
    obs = env.set_init_state(init_state)
    action = np.zeros(7, dtype=np.float32) if hold_action is None else np.asarray(hold_action)
    for _ in range(max(0, int(settle_steps))):
        obs, _reward, done, info = env.step(action)
        if done:
            break
    return obs, False, False


def step_libero(env: Any, action: np.ndarray):
    obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    return obs, bool(done), False, info if isinstance(info, dict) else {"raw_info": info}


def libero_success(env: Any) -> bool:
    return bool(env.check_success())


def libero_rgb(obs: dict, camera: str) -> np.ndarray:
    key = "agentview_image" if camera == "agentview" else "robot0_eye_in_hand_image"
    if key not in obs:
        raise KeyError(f"LIBERO observation has no {key!r}; keys={sorted(obs)}")
    return np.ascontiguousarray(np.asarray(obs[key], dtype=np.uint8))


def libero_tcp(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(3)


def libero_quat(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_quat"], dtype=float).reshape(4)


def libero_gripper_width(obs: dict) -> float:
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=float).reshape(-1)
    return float(abs(q[0] - q[1])) if q.size >= 2 else 0.0
