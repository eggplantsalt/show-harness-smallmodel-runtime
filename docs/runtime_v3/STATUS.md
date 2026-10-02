# Runtime V3 Status

## Completed

- Frozen the complete pre-V3 worktree as commit `02f10732bd6db0d7578f33a3407aa22f925692c1` and tag `legacy-full-harness-0928`.
- Built the Runtime V3 Single Authority path: fresh observation, canonical state, bounded option, selector, Arbiter approval, one-tick Executor, and fresh effect observation.
- Completed one deterministic LIBERO smoke action and 18 one-step calibration actions across six translation atoms.
- Measured HOLD settling and repeated atomic control response for task 2 / seed 0 / init state 0. Every physical tick used a separate one-step V3 authority cycle. No RuntimeOption or Executor multi-tick behavior was added.
- Started the local Qwen3-VL-8B-Instruct endpoint from existing weights with Hugging Face offline mode enabled. The no-action selector evaluation now has 12 live semantic cases and four invalid-output fixtures.
- Expanded `tests/runtime_v3` from the previous 29 to 38 passing tests.

## Current calibration findings

- `LiberoAtomicController.hold_action()` emits `[0, 0, 0, 0, 0, 0, -1]`: zero translation and zero axis-angle while preserving its open-gripper command. The loaded OSC_POSE config has `control_delta=true` and `control_ori=true`; zero axis-angle leaves its current orientation goal unchanged.
- Reset settling is present and repeatable. Mean HOLD displacement is about 1.085 mm on tick 1 and 1.132 mm on tick 2, then falls to 0.338 mm on tick 3 and 0.140 mm on tick 4. The 4→5 tick displacement is 0.130 mm.
- The tested standardized pre-settle is four HOLD ticks. It leaves about 2.567 mm cumulative displacement from reset, while the incremental drift has diminished. No numerical pass threshold was imposed.
- The common first-tick negative-Y drift in the earlier calibration is largely explained by reset settling: the matched HOLD baseline removes it from corrected X and Z primitive effects. Other axis coupling remains, especially Z during FWD/BACK.
- Corrected projections increase across one to four repeated control ticks, but the one-tick response is only about 0.46–0.56 mm against a 5 mm per-tick command. Current evidence favors measuring a bounded multi-tick micro-motion as the semantic unit; it does not authorize changing Executor.
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

## Not implemented

- No target perception/identity provider is wired into V3 task execution.
- No generic actuation or decision contract has been declared ready for verified runtime options. Results are one task/init/seed, with one action repeat per primitive/horizon.
- RuntimeOption and Executor remain single-tick. No controller, movement vector, or legacy policy was changed.
- No SAM3, VisualRoute, Recovery policy, recursive reflection, RSI, experience/visual memory, object-relative execution, long-horizon planning, full pick/place, training, or benchmark run.

## Git and legacy isolation

- Active branch: `runtime-v3`, based on `2f6aa8154b12045a3454b2913851f2bd40ab7be7` for this work.
- The frozen legacy tag and its policy files were not modified.
- This task's code, experiment, and selector artifacts remain in the Runtime V3 worktree; do not merge them into the frozen legacy baseline.

## Next gate

Actuation Contract readiness: **NO**. Bounded Qwen Decision Contract readiness: **NO**. Do not enter object-relative verified options. Review the corrected temporal response and selector errors before setting the next bounded milestone.
