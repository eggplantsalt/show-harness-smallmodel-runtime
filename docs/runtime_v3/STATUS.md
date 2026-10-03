# Runtime V3 Status

## M3.6 zero-code-change cross-object ALIGN transfer (2026-10-03)

- Baseline: branch `runtime-v3`, clean at `95c040d` before this milestone.
- Added immutable `TaskSpec` / `EntitySpec`, a manifest-based reference
  compiler, and `RuntimeEntityState` inside the existing canonical
  `BeliefState`; there is no parallel world state.
- Changed the experiment-facing perception adapter to accept an `EntitySpec`.
  The semantic phrase reaches SAM grounding; the physical multi-scale option
  reports the entity key and does not inspect an object name.
- Frozen `experiments/runtime_v3/cross_object_align_manifest.json` before
  formal rollout: task 2 anchor plus tasks 0 (alphabet soup), 6 (butter), and 7
  (milk), seed 0. A reset-only canonical 512x512 agentview preview was manually
  checked for each task and showed the named target visibly present. The preview
  did not execute control or rank tasks by performance.
- Added a cross-object Stage A runner that reuses the verified 3/6/9 mm
  contracts and the existing six-direction candidate lattice, with automatic
  same-code expansion to states 3–5 only after a 3/3 positive initial gate.
  Simulator target-pose diagnostics are disabled for these runs.
- First Stage A attempt at `253afeb` is retained at
  `rollouts/runtime_v3_cross_object_alignment/run_20261003T082839Z_bac7770d`,
  but is invalid as transfer evidence: the harness raised `NameError: uuid is
  not defined` after PRE_ACTION_READY, so no ALIGN action was executed. The
  failure was found after the frozen run completed; the code was not changed
  between task rollouts. This attempt is excluded from all transfer metrics.
- Corrected formal Stage A ran at `3ddc109791c5cca36294f61a8095195d690ff928`
  in `rollouts/runtime_v3_cross_object_alignment/run_20261003T083700Z_9865926b/`.
  Task 2 passed 3/3 episodes, task 0 expanded after its 3/3 gate and passed 6/6,
  task 7 likewise passed 6/6, and task 6 failed SceneReady in 0/3 attempts before
  any semantic ALIGN observation. The task 6 grounding, identity, and reference
  rates are unknown because they were unobserved. No expansion or task-specific
  recovery rule was applied to task 6.
- The three tasks that reached ALIGN executed 90 steps; all 90 effects improved
  the pixel alignment error and all 15 episodes were monotonic. Tasks 0 and 7
  passed unseen-object transfer; the overall cross-object result is PARTIAL
  because task 6 did not reach SceneReady. All selected scales were 9 mm, while
  directions varied across target/layouts. Oracle target pose, GT depth, GT
  contact, and task success were disabled.
- Corrected reporting was recomputed from saved episode traces. The original raw
  summary is retained as `summary_raw_metrics.json`; the corrected `summary.json`
  records its source hash and marks the post-processing.
- Added a strict one-call `QwenTaskCompiler` and an offline binding audit, plus a
  Stage B driver that passes only the audited `EntitySpec` into the frozen ALIGN
  runner. Qwen cannot provide physical fields or select direction/scale.
- Offline Qwen ran once per frozen instruction (4 calls total); all four outputs
  had valid schemas and no physical fields. The raw phrases were `the salad
  dressing`, `the alphabet soup`, `the butter`, and `the milk`. A deterministic
  saved-response audit strips one leading English article when comparing with
  the manifest phrase; raw model text is retained and the runtime receives the
  original Qwen phrase. This gives 4/4 semantic matches; literal string matches
  were 0/4. No second model call or prompt change was made.
- A first Stage B driver preflight at `f431ff8` stopped before any simulated
  episode: it used a strict article comparison and omitted the driver hash from
  its own source snapshot. It ran zero ALIGN actions and is excluded. The audit
  and signature checks were corrected before the formal Qwen rollout.
- Formal Qwen Stage B ran at commit `b04742d1732ec2db05019352ffd07b89a1c637f5`
  in `rollouts/runtime_v3_qwen_cross_object_alignment/run_20261003T090106Z_b9ac410e/`.
  Task 2 passed 3/3; task 0 failed SceneReady in 0/3; task 6 failed SceneReady
  in 0/3; task 7 passed 6/6 after its initial 3/3 gate. The two failed tasks
  have unknown grounding, identity, and reference rates because no semantic
  ALIGN observations were reached.
- Qwen Stage B executed 54 ALIGN steps across nine episodes. All 54 effects were
  positive and all nine episodes were monotonic. Task 2 improved 164.52→135.24
  px (17.81% normalized reduction; +4.879 px/step; DOWN×18, 9mm×18). Task 7
  improved 166.64→141.45 px (15.12%; +4.198 px/step; RIGHT×34, DOWN×2;
  9mm×36). The cross-object physical and semantic-coprocessor judgments are
  PARTIAL because Qwen-bound task 0 did not reach SceneReady. Failure taxonomy
  is `SCENE_NOT_READY×6`.
- Qwen and Stage A used the same physical config hash. Every ALIGN physical file
  matched the Stage A reference; all four tasks ran on one commit, with no
  per-object threshold or per-task direction/scale rule. The Qwen Stage B
  contact sheet and task 2 / task 7 episode sheets were manually reviewed; the
  target overlays matched the dressing bottle and milk carton. The frozen
  task-6 preflight image also shows the named butter package.
- `PYTHONPATH=. /root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python -m pytest -q tests/runtime_v3`:
  216 passed. Compileall and `git diff --check` also pass.

## M3.5 observation boundary and deployable metric depth (2026-10-03)

- Started from stable commit `0973f4c107124415c790a214a98ec480c1076935` on
  `runtime-v3`. Kept simulator depth out of the formal Runtime observation and
  decision path. Its reader and comparisons live only under `scripts/`; the
  formal depth provider accepts RGB and emits a pinned MoGe-2 metric estimate.
- Added the generic `MetricEntityReference`, robust mask-conditioned estimated
  depth unprojection, and the explicit formal/diagnostic observation boundary.
  Metric references remain telemetry; the existing option generator, candidate
  ranking, Arbiter, Executor, and termination policy remain unchanged. No 3D
  control or manipulation action was added.
- Model audit: `Ruicheng/moge-2-vitl`, revision
  `39c4d5e957afe587e04eec59dc2bcc3be5ecd968`; metric depth in meters at source
  resolution; RGB-only model input; 1.31 GB checkpoint. Loaded and ran 512×512
  inference in the LIBERO Python 3.10.12 / PyTorch 2.14.0+cu130 environment on
  an RTX 4080 SUPER (31.47 GiB). `transformers` is not installed; MoGe uses its
  own v2 model loader. No training or oracle scale fitting was performed.
- Stage A ran `LIBERO_OBJECT` task 2, seed 0, states 0–5. All 36/36 references
  were valid and the valid mask-depth ratio was 100%. Only 3/6 states passed the
  existing 3 mm jitter gate. Mean successive world-reference displacement was
  1.62 mm; the maximum was 4.67 mm. State maxima 0–5 were 2.48, 1.66, 2.83,
  3.30, 4.67, and 3.61 mm.
- Formal-vs-oracle comparison remained post-inference diagnostic only. Mask
  region depth MAE averaged 0.300 m (median of frame MAEs 0.308 m); estimated
  reference versus same-mask GT-depth reference error averaged 0.308 m (median
  0.315 m). Estimated-reference offset from the target body origin averaged
  0.296 m, which includes visible-surface/body-origin differences and is not a
  pure depth-model error.
- The model's median range over the SAM mask exceeded GT depth by 0.310 m on
  average. Same-mask GT-depth reference jitter peaked at only 0.056 mm while the
  monocular estimate reached 4.67 mm, indicating that estimator variation, not
  observed scene/mask motion, dominates the reference jitter in this static test.
- Stage B was gated off due to Stage A instability: zero alignment steps and no
  wrist-frustum coverage evaluation. This is a deployability blocker for the
  current RGB-only metric grounding contract on the audited images. Do not
  proceed to 3D metric alignment based on the simulator reference.
- Runtime V3 suite: 197 passed. Full experiment artifacts are under
  `rollouts/runtime_v3_metric_entity_grounding_rgb_only/run_20261002T155631Z_422a6ba3/`.
- The `legacy-full-harness-0928` baseline files were not modified. An unrelated
  working-tree edit to `AGENTS.md` appeared during the turn and is preserved;
  it is excluded from the M3.5 commit unless separately requested.

### M3.5 judgment and next gate

Deployable metric entity grounding reliability: **NO** for the current
model/task images under the 3 mm stability requirement. Diagnostic depth explains
range and geometry disagreement but cannot improve Runtime information. RGB-only
still lacks accurate, stable, independently observable metric range here; the
task also has no direct visibility into contact, hidden geometry, or occluded
target motion. Reassess a deployable depth model or real RGB-D sensor contract
before Stage B or any 3D control.

## SceneReady and single-step validation (2026-10-02)

- Started from `218d2e0e30e4bed5854fef175dd8955ff88760a3` on `runtime-v3`; the starting tree was clean and synchronized with `origin/runtime-v3`.
- Added `scripts/runtime_v3_validate_scene_ready.py` with a hold-only Stage A and a separately gated Stage B. Stage A inspects the available init-state inventory, then uses the first six states in order.
- Stage A ran `LIBERO_OBJECT` task 2 / seed 0 / states 0–5 at 512×512. The existing 3-frame, 0.02 px centroid, 0 px bbox-edge, 1 px area gate was unchanged, with a 40-HOLD limit. RobotReady completed at tick 4; SceneReady triggers were ticks 10, 12, 11, 13, 10, 11 (mean 11.17): 6/6 ready, 0/6 timeout, 0/6 false-ready.
- Per-tick xyz/z curves show the reset object motion decays into a near-stationary tail around tick 10. All three post-ready HOLD deltas per trial continue the decaying tail without a renewed excursion. Trigger assessment: 2 aligned with tick 10 and 4 late by 1–3 ticks. Keep the existing threshold for this task/seed/state range; stochastic SAM and cross-task generality remain untested.
- After reviewing Stage A raw curves, Stage B ran the same six states with one bounded `ALIGN_TO_TARGET_SMALL` per episode. Runtime selected `DOWN` in all six; there were exactly six bounded motions and six Arbiter approvals total. Qwen actions: 0.
- All six Stage B trials had valid frozen-reference metrics and improved: mean error 164.394 → 162.395 px, mean actual improvement +1.9995 px versus +1.4980 px predicted, with +0.5015 px mean residual. Same-target identity was retained 6/6.
- Oracle target displacement during the action windows was at most 4.42e-9 m and continued the Stage A HOLD-only settling tail. The projected shift rounded to 0.00 px at the existing 0.01 px precision; maximum post-SAM centroid shift was 0.008 px. No trial showed obvious post-ready target motion. Oracle data remained outside Runtime state and authority.
- The earlier 56.593 mm action/control displacement is now explained by reset settling: matched ACTION and NO_ACTION curves were identical, and this gate waits until the tail has decayed. A matched hold comparison showed excess motion 0.
- Added 15 SceneReady-focused tests. Final `PYTHONPATH=. .venv/bin/pytest tests/runtime_v3 -q`: 127 passed. Full `.venv` suite: 341 passed, 2 skipped, and one CoTracker fixture failed because that Python lacks `torch`; that exact test passed in the LIBERO Python environment (1 passed).
- Stage A and Stage B artifacts, traces, `oracle_motion_curves.png`, contact sheets, PRE_ACTION_READY records, and comparison overlays are under `rollouts/runtime_v3_scene_ready_validation/run_20261002T114329Z_08d48db9/`.

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
- No multi-step alignment, GRASP, RELEASE, placement, recovery, VisualRoute, reflection, RSI, experience/visual memory, learned tracker, training, or benchmark run was added.
- Results cover one LIBERO task, seed 0, and six fixed init states; there is no cross-task validation or repeat-variance study.

## Git and legacy isolation

- Active branch: `runtime-v3`, starting this phase from `218d2e0e30e4bed5854fef175dd8955ff88760a3`.
- The frozen legacy tag and its policy files were not modified.
- This task's code, experiment, and selector artifacts remain in the Runtime V3 worktree; do not merge them into the frozen legacy baseline.

## Historical M3.4 gate

The earlier M3.4 hold-only and single-step 2D frozen-reference checks passed in
the tested six states. That evidence supports the existing 2D objective only.
The later M3.5 deployable metric-depth gate failed; it supersedes any suggestion
that 3D metric alignment is the next milestone. Preserve the M3.4 capability, but
do not use simulator depth to clear the M3.5 blocker.
