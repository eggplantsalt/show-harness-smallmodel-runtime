# Runtime V3 Status

## Target reference phase (2026-10-02)

- Started from `030f20fdca1385adb93f281560db1ffa78e8cba3` on `runtime-v3`; the frozen-reference offline replay passed with improvements of +1.881, +1.934, and +2.041 px. The old dynamic-SAM metric remained negative in all three episodes.
- Added a transient `TargetReferenceAnchor`, separate from target identity. Geometry and the formal error use the initial associated visual mask centroid while the reference remains valid. Post-action SAM stays diagnostic-only.
- Ran three fresh `LIBERO_OBJECT` task 2 / seed 0 trials at 512×512, init states 0/1/2. Each had four HOLD ticks and exactly one bounded micro-motion (`DOWN`) with one Arbiter approval. All three had `PRE_ACTION_READY` before authorization.
- Frozen-reference improvement was +1.881, +1.934, and +2.041 px (3/3; mean +1.952 px). Dynamic-SAM improvement was −24.168, −24.334, and −24.620 px (0/3; mean −24.374 px); same-target identity stayed 3/3 and centroid shift was about 29.74 px each.
- The oracle-only body pose diagnostic recorded `[0.0000013, -0.0000107, -0.056593] m` displacement (norm 56.593 mm) in each action window. The camera signature stayed unchanged. Runtime received no contact/grasp/release/target-motion signal, so its reference remained valid even though the diagnostic shows target motion. Causality is unresolved because there was no matched no-action control.
- Root-cause judgment: **MIXED / UNRESOLVED**. Frozen geometry points in the predicted direction and reduces distance to the initial pixel reference, but the target moved during the same observation window; the old trials have no before/after oracle poses. This result does not justify multi-step alignment.
- Added 19 focused tests; `tests/runtime_v3` now passes 94 tests. Qwen was not used and no Qwen action was run.
- A first re-test attempt stopped before alignment authorization when before-overlay setup failed; it executed zero bounded motions. The corrected fresh run is `run_20261002T103924Z_241fbc7a`.

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
- Frozen-reference replay and unavailable historical oracle-motion fields: `rollouts/runtime_v3_object_relative_alignment/run_20261002T095805Z_f05c03ac/frozen_reference_offline_replay.json` and `oracle_motion_availability.json`.
- Target-reference re-test: `rollouts/runtime_v3_object_relative_alignment/run_20261002T103924Z_241fbc7a/` (`summary.json`, per-trial `PRE_ACTION_READY.json`, oracle target-motion diagnostics, and reference-comparison overlays).

## Not implemented

- No reliable Runtime contact detector or visual target-motion detector is available to invalidate a reference when the target moves.
- No matched no-action control explains the 56.593 mm target-body displacement measured during the bounded-motion windows.
- No multi-step alignment, GRASP, RELEASE, placement, recovery, VisualRoute, reflection, RSI, experience/visual memory, learned tracker, training, or benchmark run was added.
- Results cover one LIBERO task, seed 0, and three fixed init states; there is no cross-task validation or repeat-variance study.

## Git and legacy isolation

- Active branch: `runtime-v3`, starting this phase from `030f20fdca1385adb93f281560db1ffa78e8cba3`.
- The frozen legacy tag and its policy files were not modified.
- This task's code, experiment, and selector artifacts remain in the Runtime V3 worktree; do not merge them into the frozen legacy baseline.

## Next gate

Do **not** advance to multi-step object-relative alignment yet. The fixed pixel
reference improved in 3/3 trials, but the target body moved by 56.593 mm during
each observation window while Runtime had no invalidation signal. First
diagnose that motion with a matched no-action control and establish a
non-oracle observation signal for target motion; keep oracle poses diagnostic
only. Reconsider multi-step alignment only after a same-object, stationary
reference gate is supported by the evidence.
