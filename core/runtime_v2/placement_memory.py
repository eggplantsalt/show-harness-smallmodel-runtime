"""Build a bounded same-episode raw-image memory panel for a frozen VLM request."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def build_visual_memory_panel(
    run_dir: str | Path,
    bundle: list[dict[str, Any]],
    *,
    instance_id: str,
    grasp_epoch: int,
    current_frame: int,
    episode_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use at most two past frame pairs and the current pair, or fail closed."""
    if not bundle or len(bundle) > 3:
        raise ValueError("placement memory must contain one to three frames")
    root = Path(run_dir).resolve()
    frames: list[int] = []
    images: list[tuple[Image.Image, Image.Image]] = []
    refs: list[str] = []
    episode_ids = {entry.get("episode_id") for entry in bundle}
    if len(episode_ids) > 1:
        raise ValueError("placement memory mixes episode boundaries")
    observed_episode_id = next(iter(episode_ids), None)
    if episode_id is not None and observed_episode_id != episode_id:
        raise ValueError("placement memory belongs to a different episode")
    for entry in bundle:
        frame = int(entry["frame_id"])
        if frame in frames or (frames and frame <= frames[-1]) or frame > current_frame:
            raise ValueError("placement memory frames are stale or unordered")
        if entry.get("instance_id") != instance_id or int(entry.get("grasp_epoch", -1)) != grasp_epoch:
            raise ValueError("placement memory identity or grasp epoch changed")
        pair: list[Image.Image] = []
        for camera in ("agentview", "wrist"):
            ref = entry.get(f"{camera}_ref")
            expected = f"images/raw_{camera}/{frame:04d}.png"
            if ref != expected:
                raise ValueError(f"missing or non-raw {camera} memory reference")
            path = (root / ref).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"unresolvable placement memory reference: {ref}")
            with Image.open(path) as source:
                pair.append(source.convert("RGB").resize((256, 256)))
            refs.append(ref)
        images.append((pair[0], pair[1]))
        frames.append(frame)
    if frames[-1] != current_frame:
        raise ValueError("placement memory lacks the frozen live frame")
    panel = Image.new("RGB", (512, 256 * len(images)), "white")
    draw = ImageDraw.Draw(panel)
    for row, (agentview, wrist) in enumerate(images):
        top = row * 256
        panel.paste(agentview, (0, top))
        panel.paste(wrist, (256, top))
        for x, label in ((0, "AgentView"), (256, "Wrist")):
            draw.rectangle((x + 2, top + 2, x + 168, top + 18), fill="white")
            draw.text((x + 4, top + 4), f"{label} frame {frames[row]}", fill="black")
    array = np.asarray(panel)
    return array, {
        "episode_id": observed_episode_id,
        "frame_ids": frames,
        "refs": refs,
        "panel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "panel_shape": list(array.shape),
    }


# Compatibility name retained for existing placement call sites and tests.
build_placement_panel = build_visual_memory_panel
