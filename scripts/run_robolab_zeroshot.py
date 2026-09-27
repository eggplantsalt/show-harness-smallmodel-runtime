#!/usr/bin/env python3
"""Show-Harness zero-shot planner/controller deployment on RoboLab."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent.stage_control import Controller, StageControlSuite
from core.config import load_secrets_env, load_yaml
from core.prompting.prompt_loader import load_prompt_dir
from core.record.episode_logger import EpisodeLogger
from core.sim.launch import build_config, make_vlm_client
from core.v0_types import V0Config
from core.vlm.roles import ControllerAgent
from plugins.config import PluginsConfig
from plugins.action_chunk import ActionChunkPlugin
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.variable_step import VariableStepPlugin
from plugins.subgoal import SubgoalPlanner, SubgoalPlannerAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness RoboLab zero-shot planner/controller"
    )

    parser.add_argument(
        "--robot-config",
        default=str(
            ROOT / "configs" / "robot_robolab_deepseek.yaml"
        ),
    )
    parser.add_argument(
        "--prompts-dir",
        default=str(ROOT / "prompts"),
    )

    parser.add_argument("--task", default=None)
    parser.add_argument("--instruction-type", default=None)
    parser.add_argument("--camera-preset", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)

    parser.add_argument("--renderer", default=None)
    parser.add_argument("--rendering-type", default=None)
    parser.add_argument("--gui", action="store_true")

    parser.add_argument(
        "--vlm-backend",
        default=os.environ.get("VLM_BACKEND"),
    )
    parser.add_argument(
        "--vlm-url",
        default=(
            os.environ.get("VLM_URL")
            or os.environ.get("VLLM_BASE_URL")
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL"),
    )

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--prompt-log-every", type=int, default=1)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("DEBUG") == "1",
    )

    # Required by core.sim.launch.build_config.
    parser.set_defaults(
        task_suite_name=None,
        task_id=None,
    )

    return parser.parse_args()


def fold_overrides(
    args: argparse.Namespace,
    robot_cfg: dict,
) -> dict:
    for arg_name, cfg_key in [
        ("task", "task"),
        ("instruction_type", "instruction_type"),
        ("camera_preset", "camera_preset"),
        ("device", "device"),
        ("seed", "seed"),
        ("episode_index", "episode_index"),
        ("episodes", "episodes"),
        ("renderer", "renderer"),
        ("rendering_type", "rendering_type"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            robot_cfg[cfg_key] = value

    if args.gui:
        robot_cfg["headless"] = False

    return robot_cfg


def build_zero_shot_stack(
    *,
    client,
    cfg: dict,
    prompts: dict,
    common_context_extra: str = "",
    include_robolab_context: bool = True,
    merge_pregrasp: bool = True,
    controller_prompt_override: str | None = None,
):
    """Build the real Show-Harness zero-shot controller semantics for RoboLab."""

    plugins = PluginsConfig.from_config(cfg)

    if not plugins.enabled("subgoal", default=False):
        raise ValueError(
            "Zero-shot RoboLab requires plugins.subgoal: true"
        )

    common_context = prompts["common_context"]
    common_context = "\n\n".join(
        part for part in (
            common_context,
            (
            "RGB observation rule: only claim an object or the fingers are visible "
            "when they can actually be identified in the current image. The image "
            "center is not automatically the robot grasp point. If a target is "
            "occluded or outside the wrist image, say WRIST: NO and use the other "
            "view to recover visibility before attempting to grasp. If the target is "
            "not visibly between the fingers in the wrist image, never issue GRASP "
            "and never make a low horizontal move based on a claimed wrist position: "
            "raise above the table first, then use AgentView to approach. For an "
            "any-fruit task, choose the clearly visible, round orange if it is "
            "unobstructed; otherwise choose the nearest clearly visible fruit. Do "
            "not commit to a distant or occluded fruit. While the fingertip height "
            "is above the table clearance, AgentView is the source for horizontal "
            "approach and the wrist image is only a local confirmation. Do not "
            "descend below clearance until the chosen fruit is near the end effector "
            "in AgentView; use the wrist image only for final alignment and closing."
            if include_robolab_context
            else ""
            ),
            common_context_extra,
        ) if part
    )
    marker_cfg = cfg.get("wrist_grasp_marker") or {}
    if marker_cfg.get("enabled", False):
        common_context += (
            "\n\nThe cyan cross in the wrist view projects a fixed robot grasp point "
            "from hand/camera geometry. It is a near-contact alignment reference, "
            "not an object detector or evidence of a successful grasp. At large "
            "height, parallax means a farther object with the same world XY need "
            "not coincide with the cross. Inspect both views and finger height. "
            "The geometric marker has not been validated for this runtime camera."
        )

    planner = SubgoalPlanner(
        SubgoalPlannerAgent(
            client=client,
            common_context=common_context,
            max_tokens=int(
                cfg.get("planner_max_tokens", 4096)
            ),
        ),
        merge_pregrasp=merge_pregrasp,
    )

    vlm_cfg = cfg["vlm"]
    cot_mode = bool(vlm_cfg.get("reasoning_cot", False))

    backend_prompt = (
        prompts.get(
            f"controller_{vlm_cfg.get('backend')}_prompt"
        )
        if cot_mode
        else None
    )
    controller_prompt = (
        backend_prompt
        or controller_prompt_override
        or prompts["controller_prompt"]
    )

    variable_step_plugin = VariableStepPlugin(
        enabled=plugins.enabled(
            "variable_step",
            default=False,
        ),
        coarse_step_m=float(
            cfg.get("coarse_step_m", 0.04)
        ),
        high_above_table_m=float(
            cfg.get("high_above_table_m", 0.08)
        ),
    )

    action_chunk_plugin = ActionChunkPlugin(
        enabled=plugins.enabled(
            "action_chunk",
            default=False,
        ),
        step_num=int(
            cfg.get("action_chunk_step_num", 3)
        ),
    )

    mem_text_plugin = MemTextPlugin(
        enabled=plugins.enabled(
            "mem_text",
            default=False,
        ),
        max_recent=int(
            cfg.get("mem_text_len", 5)
        ),
    )

    proprio_plugin = ProprioceptionPlugin(
        enabled=plugins.enabled(
            "proprioception",
            default=False,
        ),
        high_above_table_m=float(
            cfg.get("high_above_table_m", 0.08)
        ),
        fine_step_m=float(
            cfg.get("fine_step_m", 0.02)
        ),
        coarse_step_m=(
            float(cfg.get("coarse_step_m", 0.04))
            if variable_step_plugin.enabled
            else None
        ),
    )

    table_height_m = float(cfg["table_height_m"])

    controller_agent = ControllerAgent(
        client=client,
        prompt_template=controller_prompt,
        common_context=common_context,
        transport_prompt_template=prompts.get("controller_libero_transport_prompt"),
        cot_mode=cot_mode,
        gripper_color=str(cfg["gripper_color"]),
        proprio_plugin=proprio_plugin,
        mem_text_plugin=mem_text_plugin,
        variable_step_plugin=variable_step_plugin,
        action_chunk_plugin=action_chunk_plugin,
        table_height_m=table_height_m,
    )

    controls = StageControlSuite(
        controller=Controller(controller_agent)
    )

    return planner, controls


def main() -> int:
    args = parse_args()

    load_secrets_env()

    robot_cfg = fold_overrides(
        args,
        load_yaml(args.robot_config),
    )
    cfg = build_config(args, robot_cfg)

    # Build the hosted VLM before paying Isaac Sim's cold-start cost.
    client = make_vlm_client(args, cfg)

    vlm_cfg = cfg["vlm"]
    print(
        "VLM backend: "
        f"{vlm_cfg.get('backend')} "
        f"(model={vlm_cfg.get('model')}, "
        f"base_url={vlm_cfg.get('base_url')}, "
        f"thinking={vlm_cfg.get('enable_thinking')})"
    )

    # Only a local vLLM backend needs the /models readiness poll.
    if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
        client.health_check(
            wait_s=float(
                vlm_cfg.get("startup_wait_s", 0)
            ),
            poll_s=float(
                vlm_cfg.get("startup_poll_s", 5)
            ),
        )

    prompts = load_prompt_dir(args.prompts_dir)
    planner, controls = build_zero_shot_stack(
        client=client,
        cfg=cfg,
        prompts=prompts,
    )

    # IMPORTANT: no RoboLab / IsaacLab import before the app is launched.
    from core.sim.robolab_task import launch_isaac

    simulation_app = launch_isaac(
        headless=bool(cfg.get("headless", True)),
        device=str(cfg.get("device", "cuda:0")),
    )

    try:
        return run(args, cfg, client, planner, controls)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


def run(
    args,
    cfg,
    client,
    planner,
    controls,
) -> int:
    # Imports deliberately happen only after Isaac Sim starts.
    from interpreters.robolab_atomic_controller import (
        RobolabAtomicController,
    )
    from core.sim.robolab_task import make_robolab_task
    from core.sim.zeroshot_robolab_runner import (
        ZeroshotRobolabRunner,
    )

    episodes = max(1, int(cfg.get("episodes", 1)))
    seed = int(cfg["seed"]) + int(
        cfg.get("episode_index", 0)
    )
    first_episode_index = int(cfg.get("episode_index", 0))
    # The standard EpisodeLogger timestamp has second precision. A separate invocation
    # directory avoids collisions even if a failed launch is restarted immediately.
    invocation_id = uuid4().hex[:12]
    invocation_log_dir = Path(ROOT) / cfg["log_dir"] / f"run_{invocation_id}"

    logger = EpisodeLogger(
        invocation_log_dir,
        first_episode_index,
        variant=f"RL-ZS-{cfg['task']}",
    )

    handle = make_robolab_task(
        task=str(cfg["task"]),
        robot=str(cfg.get("robot", "franka")),
        num_envs=1,
        device=str(cfg.get("device", "cuda:0")),
        seed=seed,
        instruction_type=str(
            cfg.get("instruction_type", "default")
        ),
        camera_preset=str(
            cfg.get("camera_preset", "WRIST_LEFT")
        ),
        renderer=str(
            cfg.get("renderer", "realtime")
        ),
        rendering_type=cfg.get("rendering_type"),
        output_dir=logger.run_dir,
        enable_subtask=bool(
            cfg.get("enable_subtask", False)
        ),
        verbose=args.debug,
    )

    print(
        f"Env: {handle.env_name} | "
        f"task: {handle.task} | "
        f"action_dim: {handle.action_dim} | "
        f"ik_scale: {handle.ik_scale}"
    )
    print(f"Instruction: {handle.task_description}")

    if handle.action_dim != 7:
        raise RuntimeError(
            "Expected RoboLab 7-dim relative-IK action space, "
            f"got {handle.action_dim}"
        )

    atomic_controller = RobolabAtomicController(
        move_vectors=cfg["move_vectors"],
        step_m=float(cfg["step_m"]),
        ik_scale=handle.ik_scale,
        sim_steps_per_decision=int(
            cfg["sim_steps_per_decision"]
        ),
        max_delta_m=float(
            cfg.get("max_delta_m", 0.05)
        ),
    )

    plugins_cfg = PluginsConfig.from_config(cfg)

    recovery_plugin = RecoveryPlugin(
        enabled=plugins_cfg.enabled(
            "recovery",
            default=False,
        ),
        empty_width_m=float(
            cfg.get("empty_width_m", 0.005)
        ),
        open_width_m=float(
            cfg.get("open_width_m", 0.06)
        ),
    )

    def make_runner(ep_logger, episode_index):
        return ZeroshotRobolabRunner(
            env=handle.env,
            task_description=handle.task_description,
            task_name=handle.task,
            seed=seed,
            episode_index=episode_index,
            controller=atomic_controller,
            planner=planner,
            controls=controls,
            logger=ep_logger,
            config=V0Config.from_dict(
                cfg.get("v0", {})
            ),
            max_steps=int(cfg["max_steps"]),
            num_steps_wait=int(cfg["num_steps_wait"]),
            loop_period_s=float(
                cfg["loop_period_s"]
            ),
            sim_steps_per_decision=int(
                cfg["sim_steps_per_decision"]
            ),
            settle_steps_per_decision=int(
                cfg.get(
                    "settle_steps_per_decision",
                    0,
                )
            ),
            agentview_camera=str(
                cfg["agentview_camera"]
            ),
            wrist_camera=str(
                cfg["wrist_camera"]
            ),
            agentview_rotation_degrees=int(
                cfg["agentview_rotation_degrees"]
            ),
            wrist_rotation_degrees=int(
                cfg["wrist_rotation_degrees"]
            ),
            agentview_flip=str(
                cfg.get("agentview_flip", "none")
            ),
            wrist_flip=str(
                cfg.get("wrist_flip", "none")
            ),
            use_wrist_image=bool(
                cfg["use_wrist_image"]
            ),
            # The zero-shot path uses the official RecoveryPlugin instead
            # of the MVTOKEN-only AutoRelease reflex.
            auto_release=None,
            recovery_plugin=recovery_plugin,
            action_chunk_plugin=(
                controls.controller.agent.action_chunk_plugin
            ),
            variable_step_plugin=(
                controls.controller.agent.variable_step_plugin
            ),
            table_height_m=float(cfg["table_height_m"]),
            fingertip_offset_m=float(cfg.get("fingertip_offset_m", 0.1323)),
            wrist_grasp_marker=cfg.get("wrist_grasp_marker"),
            physical_fine_step_m=float(
                cfg.get("fine_step_m", 0.02)
            ),
            physical_up_step_m=float(
                cfg.get("up_step_m", 0.04)
            ),
            sim_command_scale=(
                float(cfg["step_m"])
                / float(cfg.get("fine_step_m", 0.02))
            ),
            empty_width_m=float(
                cfg.get("empty_width_m", 0.005)
            ),
            gripper_close_threshold_m=float(
                cfg.get("gripper_close_threshold_m", 0.07)
            ),
            recent_moves_max=int(
                cfg.get("mem_text_len", 5)
            ),
            debug=args.debug,
            prompt_log_every=int(
                args.prompt_log_every
            ),
            agentview_square_size=cfg.get(
                "agentview_square_size"
            ),
            agentview_crop_aspect=cfg.get(
                "agentview_crop_aspect"
            ),
            wrist_crop_aspect=cfg.get(
                "wrist_crop_aspect"
            ),
            wrist_square_size=cfg.get(
                "wrist_square_size"
            ),
            gripper_hold_steps=int(
                cfg.get("gripper_hold_steps", 0)
            ),
            close_env=False,
        )

    metadata = {
        "task": handle.task,
        "env_name": handle.env_name,
        "task_description": handle.task_description,
        "targets": handle.targets,
        "action_dim": handle.action_dim,
        "ik_scale": handle.ik_scale,
        "seed": seed,
        "episode_index": first_episode_index,
        "episode_layout_policy": "fixed scene repeats; changing seed does not guarantee a different authored layout",
        "invocation_id": invocation_id,
        "robot_config": cfg,
        "control_loop": "robolab_zeroshot",
        "planner": "SubgoalPlannerAgent",
        "controller": "ControllerAgent",
        "vlm_backend": cfg["vlm"].get("backend"),
        "vlm_model": cfg["vlm"].get("model"),
        "debug": args.debug,
        "observation_source": "rendered RGB and robot proprioception",
        "object_state_to_policy": False,
        "fingertip_offset_m": float(cfg.get("fingertip_offset_m", 0.1323)),
        "table_height_m": float(cfg["table_height_m"]),
        "wrist_grasp_marker": {
            "config": cfg.get("wrist_grasp_marker") or {"enabled": False},
            "source": "robot_camera_geometry",
            "runtime_validated": False,
            "no_object_state": True,
        },
        "scripted_action_fallback": False,
        "success_source": "RoboLab task termination predicate",
    }
    logger.write_metadata(metadata)

    successes = 0

    for episode in range(episodes):
        ep_logger = (
            logger
            if episode == 0
            else EpisodeLogger(
                invocation_log_dir,
                first_episode_index + episode,
                variant=f"RL-ZS-{cfg['task']}",
            )
        )

        if episode > 0:
            ep_logger.write_metadata(
                {
                    **metadata,
                    "episode_index": first_episode_index + episode,
                }
            )
            handle.env.reset_eval_state()

        result = make_runner(ep_logger, first_episode_index + episode).run()

        successes += int(result.success)

        print(
            f"[episode {episode}] "
            f"success={result.success} "
            f"steps={result.steps} "
            f"end_reason={result.end_reason}"
        )
        print(f"  run dir: {result.run_dir}")
        print(f"  video:   {result.video_path}")

    print(f"Success rate: {successes}/{episodes}")

    try:
        handle.env.close()
    except Exception:
        pass

    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
