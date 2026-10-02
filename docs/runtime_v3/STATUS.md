# Runtime V3 Status

## Completed

- Frozen the complete pre-V3 worktree as commit `02f10732bd6db0d7578f33a3407aa22f925692c1` and tag `legacy-full-harness-0928`.
- Built the Runtime V3 Single Authority path: fresh observation, canonical state, bounded option, selector, Arbiter approval, one-tick Executor, and fresh effect observation.
- Completed one deterministic LIBERO smoke action and 18 one-step calibration actions across six translation atoms.
- Measured HOLD settling and repeated atomic control response for task 2 / seed 0 / init state 0. In that baseline experiment, every physical tick used a separate one-step V3 authority cycle; the bounded execution unit was added in the current phase.
- Upgraded the physical Runtime V3 execution unit to a same-direction bounded micro-motion. One Arbiter-approved semantic option carries a 3 mm target, a maximum of five 5 mm control ticks, the starting EEF position, and the observed workspace Z bounds; Executor observes after every tick and cannot replan.
- Ran all 18 fixed cross-state trials for task 2 / seed 0 / init states 0, 1, and 2 with four HOLD pre-settle ticks and six directions per state. All 18 reached the requested projection, all stopped as `TARGET_REACHED`, and every trial used one Arbiter approval for its bounded motion. Per-tick observations and videos are saved.
- Started the local Qwen3-VL-8B-Instruct endpoint from existing weights with Hugging Face offline mode enabled. The no-action selector evaluation now has 12 live semantic cases and four invalid-output fixtures.
- Expanded `tests/runtime_v3` from the previous 29 to 38 passing tests.

## Current calibration findings

- `LiberoAtomicController.hold_action()` emits `[0, 0, 0, 0, 0, 0, -1]`: zero translation and zero axis-angle while preserving its open-gripper command. The loaded OSC_POSE config has `control_delta=true` and `control_ori=true`; zero axis-angle leaves its current orientation goal unchanged.
- Reset settling is present and repeatable. Mean HOLD displacement is about 1.085 mm on tick 1 and 1.132 mm on tick 2, then falls to 0.338 mm on tick 3 and 0.140 mm on tick 4. The 4→5 tick displacement is 0.130 mm.
- The tested standardized pre-settle is four HOLD ticks. It leaves about 2.567 mm cumulative displacement from reset, while the incremental drift has diminished. No numerical pass threshold was imposed.
- The common first-tick negative-Y drift in the earlier calibration is largely explained by reset settling: the matched HOLD baseline removes it from corrected X and Z primitive effects. Other axis coupling remains, especially Z during FWD/BACK.
- The 18 bounded executions reached a 3 mm projection in 3–4 ticks (mean 3.33–4.00 by direction), with 18/18 target attainment. Direction means ranged from 3.157 mm BACK to 4.058 mm DOWN. FWD/BACK mean off-axis magnitude was 0.795/0.919 mm; DOWN had 1.058 mm mean projection overshoot. These are diagnostics, not added pass criteria.
- Opposite-pair corrected effects point in opposing directions. Four-tick pair residuals are 0.170 mm for LEFT/RIGHT, 0.407 mm for FWD/BACK, and 0.397 mm for UP/DOWN.

## Current selector findings

- Model: `Qwen/Qwen3-VL-8B-Instruct`, local weights at `/root/autodl-tmp/huggingface/hub/Qwen/Qwen3-VL-8B-Instruct`; endpoint `http://127.0.0.1:8001/v1`; GPU 0. The server runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; no model was downloaded.
- The 12 live cases achieved schema-valid 12/12, option-valid 12/12, and semantic accuracy 9/12 (75%). Category scores: geometric 1/4, ambiguous-state reobserve 2/2, stage decision 2/2, and task-conditioned choice 4/4. Both same-state/different-instruction pairs selected different expected options.
- The four injected malformed/raw-action fixtures all failed closed. No Executor was constructed and no robot action ran.
- Thinking was not evaluated in this phase. Its prior 64-token truncated outputs are not treated as decision results.

## Artifacts

- One-action smoke: `rollouts/runtime_v3_smoke/run_20261002T043157Z_003fda80/`.
- Earlier one-step calibration: `rollouts/runtime_v3_calibration/run_20261002T050110Z_c209f2e7/`.
- HOLD and temporal response: `rollouts/runtime_v3_response/run_20261002T063447Z_ca3cc2bc/` (`hold_trials.jsonl`, `action_trials.jsonl`, `summary.json`, `response_table.csv`, and `README.txt`). One discarded incomplete first attempt is disclosed in `calibration_failures.jsonl` and excluded from metrics.
- Instruct selector evaluation: `rollouts/runtime_v3_selector_eval/instruct_20261002.json`.
- Cross-state bounded micro-motion: `rollouts/runtime_v3_micro_motion/run_20261002T081115Z_34899d71/` (`trials.jsonl`, `summary.json`, and 18 per-trial MP4s plus PNG frames).

## Not implemented

- No target perception/identity provider is wired into V3 task execution.
- The bounded micro-motion result supports the first 3 mm same-direction RuntimeOption scale on the tested task and fixed init states. It is not a general manipulation reliability claim: there is one execution per direction/state and no cross-task or repeat-variance evidence.
- One-tick primitives remain available. The new bounded multi-tick path is limited to same-direction micro-motion; no controller, movement vector, or legacy policy was changed.
- No SAM3, VisualRoute, Recovery policy, recursive reflection, RSI, experience/visual memory, object-relative execution, long-horizon planning, full pick/place, training, or benchmark run.

## Git and legacy isolation

- Active branch: `runtime-v3`, starting this phase from `3d1bfbb3c0f802e314f4429a657dfd777098e195`.
- The frozen legacy tag and its policy files were not modified.
- This task's code, experiment, and selector artifacts remain in the Runtime V3 worktree; do not merge them into the frozen legacy baseline.

## Next gate

Bounded micro-motion is **ready for the next bounded RuntimeOption stage** under
the tested 3 mm contract: all 18/18 executions reached the target across three
fixed init states, with no negative-progress, boundary, or max-tick stops. This
is narrow empirical readiness, not broad physical reliability; inspect the
FWD/BACK off-axis values and DOWN overshoot when defining the next option
envelope. The 12-case Qwen semantic selector result remains unchanged, and
geometric decisions remain Runtime-owned. Do not implement object-relative
options in this milestone.
