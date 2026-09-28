#!/usr/bin/env python3
"""Run Show-Harness's full API zero-shot stack on one standard LIBERO task."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml
from core.record.episode_logger import EpisodeLogger
from core.sim.launch import build_config, make_vlm_client
from core.sim.libero_task import ensure_libero_path, make_libero_task
from core.sim.zeroshot_libero_runner import ZeroshotLiberoRunner
from core.capabilities.visual_harness import VisualHarness
from core.capabilities.verified_runtime import VerifiedEmbodiedRuntime
from core.v0_types import V0Config
from plugins.config import PluginsConfig
from plugins.recovery import RecoveryPlugin
from plugins.visual_route import VisualRoutePlugin
from scripts.run_robolab_zeroshot import build_zero_shot_stack
from core.prompting.prompt_loader import load_prompt_dir
from interpreters.libero_atomic_controller import LiberoAtomicController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "robot_libero.yaml"))
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"))
    parser.add_argument("--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--prompt-log-every", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DEBUG") == "1")
    parser.set_defaults(task_suite_name=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_secrets_env()
    cfg = load_yaml(args.robot_config)
    for arg, key in (
        ("suite_name", "suite_name"),
        ("task_id", "task_id"),
        ("init_state_index", "init_state_index"),
        ("seed", "seed"),
        ("episode_index", "episode_index"),
        ("episodes", "episodes"),
        ("max_steps", "max_steps"),
        ("loop_period_s", "loop_period_s"),
        ("log_dir", "log_dir"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            cfg[key] = value
    if cfg.get("libero_dir"):
        os.environ["LIBERO_DIR"] = str(cfg["libero_dir"])
    ensure_libero_path()
    # build_config resolves the selected provider, key and retry policy exactly as the
    # RoboLab entry point does; the LIBERO adapter never embeds credentials.
    args.task_suite_name = None
    args.task_id = None
    cfg = build_config(args, cfg)
    client = make_vlm_client(args, cfg)
    prompts = load_prompt_dir(args.prompts_dir)
    planner, controls = build_zero_shot_stack(
        client=client,
        cfg=cfg,
        prompts=prompts,
        include_robolab_context=False,
        merge_pregrasp=False,
        common_context_extra=(
            "LIBERO backend calibration: this is the standard LIBERO Panda camera frame. "
            "The AgentView image is upright after the configured 180-degree transform: "
            "screen-down is world +X (MV_FWD), screen-up is world -X (MV_BACK), "
            "screen-right is world -Y (MV_RIGHT), and screen-left is world +Y (MV_LEFT). "
            "The wrist image is already upright with the black gripper entering from the "
            "top; use the same screen-direction mapping for horizontal and depth corrections. "
            "For every LIBERO pick task, the plan must contain a distinct APPROACH stage "
            "before GRASP; do not merge approach and grasp into one stage. Complete the "
            "APPROACH stage using AgentView while high above the table, then use Wrist "
            "only for final local alignment in GRASP. "
            "When multiple same-class objects or bottles are visible, the plan must include "
            "a distinguishing visual attribute such as cap color or body color in TARGET or "
            "AFFORD (for this scene, distinguish the green-capped salad-dressing bottle from "
            "the other bottles); never leave the controller with only a generic bottle query. "
            "In a GRASP stage, align both X and Y while above the table, then descend only into "
            "the calibrated final pre-grasp height band. If the Wrist view is empty but AgentView "
            "alignment is already within tolerance, do not infer that another descent is needed: "
            "use AgentView and robot height, and issue GRASP when the target is between the open "
            "fingers. If the EEF is below the final band, issue MV_UP before GRASP. The target's "
            "screen-down offset includes camera parallax because it is lower than the EEF. "
            "When the EEF is high and left/right alignment is already good, use MV_DOWN to "
            "enter the final height band before using MV_FWD/MV_BACK for depth correction. "
            "If a target "
            "remains on one side after a move, repeat that same direction; do not alternate "
            "left/right or forward/back merely because the image is noisy."
        ),
        controller_prompt_override=(
            prompts.get("controller_libero_prompt")
            if str(cfg.get("research_profile", "")).startswith("clean_qwen3vl")
            else None
        ),
    )

    visual_harness = VisualHarness.from_config(cfg)
    verified_runtime = VerifiedEmbodiedRuntime.from_config(cfg)
    visual_route_plugin = VisualRoutePlugin.from_config(cfg, client=client)

    # One decision contains the commanded motion steps plus optional settle
    # steps; GRASP/RELEASE can hold the gripper for longer.  Keep the
    # simulator horizon above the policy budget so an internal horizon
    # cutoff is not mistaken for an agent failure.
    max_decisions = int(cfg.get("max_steps", 100))
    motion_steps = int(cfg.get("sim_steps_per_decision", 4))
    settle_steps = int(cfg.get("settle_steps_per_decision", 1))
    gripper_steps = int(cfg.get("gripper_hold_steps", 12))
    # A policy decision may be a 12-step gripper hold, followed by an immediate
    # release during recovery. Budget the simulator horizon for the worst normal
    # decision cost rather than assuming every decision is a 4-step MOVE. This is
    # still only a safety budget; the runner stops as soon as LIBERO reports done.
    recovery_margin = max(200, 8 * gripper_steps)
    per_decision_sim_budget = max(
        motion_steps + settle_steps,
        2 * gripper_steps,
    )
    horizon = max_decisions * per_decision_sim_budget + recovery_margin
    horizon = max(horizon, max_decisions + 10)

    handle = make_libero_task(
        suite_name=str(cfg["suite_name"]),
        task_id=int(cfg["task_id"]),
        init_state_index=int(cfg.get("init_state_index", 0)),
        camera_height=int(cfg.get("camera_height", 256)),
        camera_width=int(cfg.get("camera_width", 256)),
        horizon=int(horizon),
        seed=int(cfg.get("seed", 0)),
        settle_steps=0,
    )
    print(f"LIBERO: {handle.suite_name} task {handle.task_id} — {handle.task_description}")
    print(f"BDDL: {handle.bddl_file}")

    controller = LiberoAtomicController(
        move_vectors=cfg["move_vectors"],
        step_m=float(cfg["step_m"]),
        sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
        position_scale_m=float(cfg.get("position_scale_m", 0.05)),
    )
    plugins_cfg = PluginsConfig.from_config(cfg)
    recovery = RecoveryPlugin(
        enabled=plugins_cfg.enabled("recovery", default=False),
        empty_width_m=float(cfg.get("empty_width_m", 0.004)),
        # ``open_width_m`` is accepted for legacy configs but is not used to
        # infer object occupancy; the visual Agent performs that confirmation.
        open_width_m=float(cfg.get("open_width_m", 0.06)),
    )
    invocation = uuid4().hex[:12]
    root_log = Path(ROOT) / cfg["log_dir"] / f"run_{invocation}"
    episodes = max(1, int(cfg.get("episodes", 1)))
    successes = 0
    for episode in range(episodes):
        logger = EpisodeLogger(root_log, int(cfg.get("episode_index", 0)) + episode, variant=f"LIBERO-{handle.suite_name}-{handle.task_id}")
        logger.write_metadata({
            "suite_name": handle.suite_name,
            "task_id": handle.task_id,
            "task_name": handle.task_name,
            "task_description": handle.task_description,
            "init_state_index": handle.init_state_index,
            "bddl_file": handle.bddl_file,
            "seed": int(cfg.get("seed", 0)),
            "control_loop": "libero_zeroshot_full_harness",
            "vlm_backend": cfg["vlm"].get("backend"),
            "vlm_model": cfg["vlm"].get("model"),
            "model_revision": cfg.get("model_revision"),
            "object_state_to_policy": False,
            "research_profile": cfg.get("research_profile"),
            "capabilities": cfg.get("capabilities"),
            "verified_runtime": (
                verified_runtime.metadata()
                if verified_runtime is not None
                else {"enabled": False}
            ),
            "visual_route": visual_route_plugin.metadata(),
        })
        runner = ZeroshotLiberoRunner(
            env=handle.env,
            init_state=handle.init_states[handle.init_state_index],
            task_description=handle.task_description,
            task_name=handle.task_name,
            seed=int(cfg.get("seed", 0)),
            episode_index=episode,
            controller=controller,
            planner=planner,
            controls=controls,
            logger=logger,
            config=V0Config.from_dict(cfg.get("v0", {})),
            max_steps=int(cfg["max_steps"]),
            num_steps_wait=int(cfg.get("num_steps_wait", 5)),
            loop_period_s=float(cfg.get("loop_period_s", 0.0)),
            sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
            settle_steps_per_decision=int(cfg.get("settle_steps_per_decision", 1)),
            agentview_camera=str(cfg.get("agentview_camera", "agentview")),
            wrist_camera=str(cfg.get("wrist_camera", "robot0_eye_in_hand")),
            agentview_rotation_degrees=int(cfg.get("agentview_rotation_degrees", 180)),
            wrist_rotation_degrees=int(cfg.get("wrist_rotation_degrees", 180)),
            agentview_flip=str(cfg.get("agentview_flip", "none")),
            wrist_flip=str(cfg.get("wrist_flip", "none")),
            use_wrist_image=bool(cfg.get("use_wrist_image", True)),
            auto_release=None,
            recovery_plugin=recovery,
            action_chunk_plugin=controls.controller.agent.action_chunk_plugin,
            variable_step_plugin=controls.controller.agent.variable_step_plugin,
            visual_harness=visual_harness,
            visual_route_plugin=visual_route_plugin,
            verified_runtime=verified_runtime,
            legacy_stage_token_normalization=bool(
                cfg.get("legacy_stage_token_normalization", True)
            ),
            lift_clear_height_m=float(cfg.get("lift_clear_height_m", 0.17)),
            table_height_m=float(cfg.get("table_height_m", 0.8)),
            fingertip_offset_m=0.0,
            physical_fine_step_m=float(cfg.get("fine_step_m", 0.02)),
            physical_up_step_m=float(cfg.get("up_step_m", 0.04)),
            sim_command_scale=1.0,
            empty_width_m=float(cfg.get("empty_width_m", 0.004)),
            gripper_close_threshold_m=float(cfg.get("gripper_close_threshold_m", 0.008)),
            recent_moves_max=int(cfg.get("mem_text_len", 5)),
            debug=args.debug,
            prompt_log_every=int(args.prompt_log_every),
            agentview_square_size=cfg.get("agentview_square_size"),
            agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
            wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
            wrist_square_size=cfg.get("wrist_square_size"),
            gripper_hold_steps=int(cfg.get("gripper_hold_steps", 12)),
            close_env=False,
        )
        result = runner.run()
        successes += int(result.success)
        print(f"[episode {episode}] success={result.success} steps={result.steps} end_reason={result.end_reason}")
        print(f"  run dir: {result.run_dir}")
    handle.env.close()
    print(f"Success rate: {successes}/{episodes}")
    return 0 if successes else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
