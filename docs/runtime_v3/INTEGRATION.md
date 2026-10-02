# Runtime V3 LIBERO Integration

## Reused legacy low-level interfaces

| V3 adapter / purpose | Existing interface used |
| --- | --- |
| `LiberoEnvironmentAdapter` creates, resets, steps, checks, and closes the simulator | `core.sim.libero_task.make_libero_task`, `reset_libero`, `step_libero`, `libero_success`; task reset is configured with `settle_steps=0` |
| `LiberoObservationAdapter` reads cameras and robot state | `libero_rgb`, `libero_tcp`, `libero_quat`, and `libero_gripper_width` from `core.sim.libero_task` |
| `LiberoPrimitiveBackend` converts a sealed V3 action to a LIBERO action | Existing `interpreters.libero_atomic_controller.LiberoAtomicController`; one-tick primitives still map directly to one 7D OSC_POSE action. A bounded micro-motion maps its semantic direction (for example, `LEFT`) to the configured `MV_LEFT` token inside the backend, then submits one 5 mm control-tick command per Executor iteration. |
| `QwenSelectorAdapter` for the no-action selector check | Existing `core.vlm.vlm_client.VLMClient.complete_json`, constructed with `core.sim.launch.build_config` / `make_vlm_client` |
| Raw smoke records | Small JSONL/JSON writer in `scripts/runtime_v3_smoke.py`; raw camera frames are saved as PNG |

`core.capabilities.sam3_client.Sam3Client`, `VisualHarness`,
`core.capabilities.camera_geometry`, and `core.record.episode_logger.EpisodeLogger`
were audited but are not needed for a target-independent vertical infrastructure
smoke. `LiberoEnvironmentAdapter.check_success()` wraps the existing success
predicate, but the smoke does not call it or use task success as policy input.

## Policy isolation

The active V3 path does not import or instantiate `VisualRoutePlugin`,
`VerifiedEmbodiedRuntime`/`VerifiedCapabilityRuntime`, `RecoveryPlugin`, the
recursive reflection path, legacy text/RSI memory, the VisualHarness action
guards, or the stage-control suite. No SAM3 result is needed for this smoke.

## Deterministic real call path

```text
LiberoEnvironmentAdapter.create
→ LiberoObservationAdapter.observe
→ StateBuilder.update (BeliefState)
→ SmokeOptionGenerator.generate
→ DeterministicSelector.select
→ Arbiter.authorize
→ Executor.execute (one approved option)
→ LiberoPrimitiveBackend.execute_approved_action or execute_approved_micro_tick
→ LiberoEnvironmentAdapter.step / core.sim.libero_task.step_libero / env.step
→ LiberoObservationAdapter.observe
→ StateBuilder.update
→ EffectObserver.compare
```

The smoke option uses the existing `MV_UP` atom with `step_m=0.005` and a
single controller/environment step. Before it is offered, the observation
adapter checks that measured EEF Z is finite and that the configured
smoke-only Z envelope contains both the current and proposed Z. The default
envelope is 0.02–0.60 m and is CLI-overridable. This is an infrastructure check,
not a task policy or a general robot safety certification.

The effect record is computed from the pre-action and post-action EEF positions.
It records the requested delta, measured delta, projected progress, and whether
the minimum positive Z movement was observed. The controller return value is
only a backend receipt and is not used as evidence of physical effect.

## Atomic action semantics and calibration

`configs/robot_libero_clean_qwen3vl.yaml` maps the six `MV_*` atoms directly to
XYZ unit vectors. `LiberoAtomicController` multiplies that vector by
`step_m / sim_steps_per_decision`, divides by `position_scale_m`, clips the
normalized components, and emits `[x, y, z, 0, 0, 0, gripper]`. With the V3
calibration settings, `MV_UP` at 5 mm emits `[0, 0, 0.1, 0, 0, 0, -1]`.
LIBERO's loaded OSC_POSE config has `control_delta=true` and maps normalized XYZ
input to a ±0.05 m goal increment. That goal increment is not a guarantee that
the observed EEF reaches the increment in one `env.step`.

Both the OSC controller's current EEF position and LIBERO's `robot0_eef_pos`
observation come from the MuJoCo world-frame EEF site. OSC_POSE adds the scaled
delta to that position. The atomic controller adds no coordinate transform.
LIBERO runs at 20 Hz: one `env.step` advances 0.05 s and runs 25 simulation
substeps at a 0.002 s model timestep. In the frozen old runner, one move token
is sent for four `env.step` calls (`sim_steps_per_decision: 4`), followed by one
hold/settle step (`settle_steps_per_decision: 1`) when the episode continues.
Runtime V3 sends one action through one adapter `step` and currently has no
post-move settle step.

The 5 mm one-step calibration used LIBERO_OBJECT task 2, seed 0, init state 0,
and three independent resets per configured token. Every completed trial
followed Option → deterministic Selector → Arbiter → Executor → backend →
environment, then re-observed EEF position. All 18 trials completed. Summary:

The adapter defines a Z workspace envelope of 0.02–0.60 m, which the trial
generator checked before offering a vertical move. No trial crossed that
boundary. The current V3 observation contract does not define X/Y workspace
bounds, so the experiment records the configured movement vectors and measured
effects without claiming an X/Y boundary certificate.

| Primitive | Commanded | Mean projection | Realization | Mean off-axis | Direction cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| MV_FWD | 5 mm | 0.558 mm | 0.112 | 1.072 mm | 0.462 |
| MV_BACK | 5 mm | 0.378 mm | 0.076 | 1.097 mm | 0.326 |
| MV_LEFT | 5 mm | -0.517 mm | -0.103 | 0.205 mm | -0.930 |
| MV_RIGHT | 5 mm | 1.612 mm | 0.322 | 0.208 mm | 0.992 |
| MV_UP | 5 mm | 0.746 mm | 0.149 | 1.068 mm | 0.573 |
| MV_DOWN | 5 mm | 0.354 mm | 0.071 | 1.069 mm | 0.314 |

All three repeats per token were numerically identical from this fixed reset,
so measured standard deviations were zero (within floating-point precision).
This only describes this task/init state and one 50 ms control interval. The
common approximately -1.065 mm Y component on `MV_FWD`, `MV_BACK`, `MV_UP`,
and `MV_DOWN` is evidence of a repeatable cross-axis component, but this
experiment had no hold-only control trial and does not identify its cause.
The `MV_LEFT` net Y displacement is negative, while `MV_RIGHT` is negative in
the commanded world-Y direction; the common Y component means these net
measurements alone do not prove a left/right sign reversal. Controller dynamics,
initial settling, and axis coupling remain hypotheses to distinguish in a later
bounded calibration. The confirmed 5 mm → 0.746 mm result is that V3 issued one
50 ms step toward a 5 mm OSC goal and observed only 0.746 mm Z projection at the
end of that interval. `step_m` is a controller goal increment, not a measured
final displacement.

The prior `effect_achieved_legacy` predicate only checks positive projected
movement above 0.1 mm. It marked 15 of 18 trials achieved, including most
low-realization and high-off-axis results. Calibration therefore records that
flag separately from `contract_metrics`; it does not set a new pass threshold.
Machine-readable data are in `rollouts/runtime_v3_calibration/run_20261002T050110Z_c209f2e7/`.

## HOLD settling and temporal response

The follow-up experiment uses `LiberoAtomicController.hold_action()`, which
returns `[0, 0, 0, 0, 0, 0, -1]` at the reset controller state. The loaded
robosuite OSC_POSE config uses relative position and orientation control with
orientation control enabled; zero axis-angle leaves the current orientation
goal unchanged, and the controller's existing gripper command is preserved.
This is a no-displacement target command, not an action that names a new EEF
position.

All conditions use `LIBERO_OBJECT` task 2, seed 0, init state 0, and unchanged
configured MOVE vectors. HOLD horizon and pre-settle conditions each have
three independent resets. After the measured per-tick HOLD drift had diminished
by tick 4, action trials used four HOLD ticks as their standardized pre-settle.
Each primitive/horizon pair had one independent reset. Each repeated token was
reselected and approved in a separate Runtime V3 cycle with a fresh observation
before the next tick; Executor still executed exactly one approved tick per
cycle. The first `MV_FWD`/one-tick attempt hit an aggregation indexing error
after one simulated step; its position was not persisted and it is excluded.
The same condition was rerun after the fix. The incomplete attempt is recorded
in `calibration_failures.jsonl` and included only in the simulated-tick count.

| HOLD tick | Trials contributing | Mean dx (mm) | Mean dy (mm) | Mean dz (mm) | Mean per-tick norm (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.081 | -1.065 | 0.190 | 1.085 |
| 2 | 9 | 0.158 | -1.088 | 0.267 | 1.132 |
| 3 | 6 | 0.147 | -0.267 | 0.147 | 0.338 |
| 4 | 6 | 0.105 | -0.001 | 0.093 | 0.140 |
| 5 | 3 | 0.073 | 0.086 | 0.065 | 0.130 |

Each 1/2/4/5-tick horizon had three separate resets. The larger `n` at early
ticks pools independent runs whose horizon was at least that long.

| Pre-settle HOLD ticks k | Trials | Mean drift dx (mm) | Mean drift dy (mm) | Mean drift dz (mm) | Mean drift norm (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 3 | 0.081 | -1.065 | 0.190 | 1.085 |
| 2 | 3 | 0.239 | -2.153 | 0.458 | 2.214 |
| 4 | 3 | 0.491 | -2.421 | 0.697 | 2.567 |

The four-tick recommendation follows the reduction in incremental drift
(0.140 mm on tick 4 and 0.130 mm on tick 5), not a predefined threshold. The
accumulated reset-relative drift remains 2.567 mm after four HOLD ticks.

The next table reports each independently reset action horizon endpoint.
Values are signed target-axis projections in millimeters for ticks 1, 2, 3,
and 4.

| Primitive | Raw projection (t1 / t2 / t3 / t4 mm) | HOLD-corrected projection (t1 / t2 / t3 / t4 mm) |
| --- | --- | --- |
| MV_FWD | 0.534 / 1.393 / 2.292 / 3.223 | 0.461 / 1.267 / 2.124 / 3.017 |
| MV_BACK | 0.402 / 1.211 / 2.121 / 3.106 | 0.475 / 1.337 / 2.288 / 3.312 |
| MV_LEFT | 0.634 / 1.822 / 3.130 / 4.459 | 0.549 / 1.634 / 2.853 / 4.119 |
| MV_RIGHT | 0.441 / 1.378 / 2.457 / 3.610 | 0.527 / 1.566 / 2.734 / 3.950 |
| MV_UP | 0.610 / 1.729 / 2.927 / 4.127 | 0.545 / 1.616 / 2.773 / 3.937 |
| MV_DOWN | 0.493 / 1.567 / 2.769 / 4.009 | 0.558 / 1.680 / 2.923 / 4.200 |

Four-tick baseline-corrected endpoint vectors and opposite-pair residuals:

| Pair | Corrected effect A (mm) | Corrected effect B (mm) | Residual A+B (mm) | Residual norm (mm) |
| --- | --- | --- | --- | ---: |
| LEFT / RIGHT | [-0.015, 4.119, -0.009] | [-0.001, -3.950, 0.004] | [-0.016, 0.169, -0.005] | 0.170 |
| FWD / BACK | [3.017, -0.002, -0.959] | [-3.312, -0.001, 0.680] | [-0.295, -0.003, -0.280] | 0.407 |
| UP / DOWN | [-0.407, -0.004, 3.937] | [0.110, 0.002, -4.200] | [-0.297, -0.003, -0.263] | 0.397 |

The common reset drift explains most of the earlier shared negative-Y
component: HOLD's first two ticks move about -1.1 mm in Y, and matched
correction removes it from FWD/BACK/UP/DOWN effects. Residual cross-axis motion
remains, primarily along Z for FWD/BACK. The opposite pairs are directionally
consistent in this fixed reset, with the residuals above and no pass threshold.

The 5 mm target increment realizes only about 0.46–0.56 mm of corrected
projection after one 50 ms tick. Four repeated ticks accumulate about
3.0–4.2 mm projection. This favors studying a bounded multi-tick micro-motion
as an option semantic unit, but this experiment did not change RuntimeOption
or Executor; both remain one-tick. Full raw and corrected projections, ratios,
off-axis magnitudes, direction cosines, and per-tick trajectories are in
`rollouts/runtime_v3_response/run_20261002T063447Z_ca3cc2bc/`.

## Bounded micro-motion execution and cross-state trial

`BoundedMicroMotionOptionGenerator` emits semantic IDs such as
`MOVE_LEFT_SMALL`; the option stores a `BoundedMicroMotionSpec` with a 3 mm
projection request, at most five ticks, and a 5 mm command per control tick.
The Arbiter seals the spec with the fresh starting EEF position and observed
workspace Z bounds. The Executor repeats only the approved direction and calls
the Observer after every control tick. It stops on target projection, negative
incremental progress, the tick limit, or before the next tick would cross the Z
envelope. The backend maps semantic direction to `MV_*`; the RuntimeOption does
not expose that controller token. Each execution remains one option selection
and one Arbiter approval.

The calibration used `LIBERO_OBJECT` task 2, seed 0, init states 0, 1, and 2.
Every trial started with four HOLD ticks and tested each of FWD, BACK, LEFT,
RIGHT, UP, and DOWN once, for 18 bounded executions. Target success is exactly
projection >= 3 mm. No prompt tuning, alternative seeds, or movement-vector
changes were used.

| Direction | n | Target reached | Mean projection (mm) | Mean absolute error (mm) | Mean ticks | Mean off-axis (mm) | Mean direction cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FWD | 3 | 3/3 | 3.213 | 0.213 | 4.00 | 0.795 | 0.9703 |
| BACK | 3 | 3/3 | 3.157 | 0.157 | 4.00 | 0.919 | 0.9590 |
| LEFT | 3 | 3/3 | 3.376 | 0.376 | 3.33 | 0.214 | 0.9968 |
| RIGHT | 3 | 3/3 | 3.823 | 0.823 | 4.00 | 0.256 | 0.9972 |
| UP | 3 | 3/3 | 3.413 | 0.413 | 3.33 | 0.230 | 0.9977 |
| DOWN | 3 | 3/3 | 4.058 | 1.058 | 4.00 | 0.312 | 0.9958 |

All 18 trials terminated as `TARGET_REACHED`; there were no `MAX_TICKS_REACHED`,
`NEGATIVE_PROGRESS`, or `BOUNDARY_STOP` outcomes. By init state, each state
reached 6/6 targets. Mean projection / absolute error / off-axis / direction
cosine were: state 0, 3.534 / 0.534 / 0.522 mm / 0.9848; state 1, 3.359 /
0.359 / 0.517 mm / 0.9842; and state 2, 3.627 / 0.627 / 0.324 mm / 0.9894.
FWD and BACK have the largest cross-axis displacement (0.795 and 0.919 mm);
DOWN has the largest mean overshoot (1.058 mm). These remain diagnostic values,
not extra pass thresholds.

Machine-readable tick observations, projections, off-axis deltas, execution
durations, termination reasons, and the state-by-direction summary are in
`rollouts/runtime_v3_micro_motion/run_20261002T081115Z_34899d71/`. Every trial
also has an `agentview`/`wrist` MP4 and its source PNG frames. The data cover
three init states from one LIBERO task with one execution per direction/state;
they do not measure cross-task transfer or repeat variance.

## Qwen selector contract

The Qwen adapter stores only the existing VLM client and task instruction. It
receives a task instruction, compact state fields (including allowlisted target
geometry), and two to four bounded options. It has no environment, controller,
backend, or Executor reference and receives no images. It requests exactly one
JSON object with a single `selection` key whose value must be a current option
ID, `REOBSERVE`, or `ABORT`. Invalid JSON, unknown IDs, extra keys,
natural-language commands, and raw motion tokens fail closed as
`INVALID_SELECTION`; there is no fuzzy matching. The evaluation script never
constructs an Executor.

The local Instruct endpoint was started on port 8001 using the existing
`Qwen/Qwen3-VL-8B-Instruct` directory with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. The earlier project-owned Thinking server (PID 2863,
launched by `scripts/serve_clean_qwen3vl.sh` from the Show-Harness-Rebuild-0928
checkout) was stopped after its model ID and project process ownership were
confirmed. Instruct served from PID 13101 on GPU 0; no weights were downloaded.

The no-action evaluation ran 12 semantic cases: four geometric choices, two
ambiguous-state reobserve choices, two stage decisions, and four
task-conditioned choices in two matched pairs where state and options were
held constant and the instruction changed. Instruct returned exact-schema
output in 12/12 cases, selected a listed option in 12/12, and matched the
expected answer in 9/12 (75%). Scores were geometric 1/4, ambiguous 2/2,
stage 2/2, and task-conditioned 4/4. The four malformed/raw-action fixtures
all failed closed. Mean latency was 0.413 seconds and mean completion was 9.17
tokens. Thinking was not needed; the previous 64-token truncated Thinking
responses are not treated as model decisions. No Executor was constructed and
no robot action ran. Machine-readable cases and token/latency records are in
`rollouts/runtime_v3_selector_eval/instruct_20261002.json`.

## Current verification record

- Deterministic real smoke: `LIBERO_OBJECT`, task 2, init state 0, seed 0.
- Exactly one V3-approved action was sent. EEF Z moved from 0.260454598 m to
  0.261200609 m (observed `delta_z=+0.000746011 m`) after the bounded 5 mm
  `MV_UP` command; the minimum positive-Z effect threshold was 0.0001 m.
- Temporal response artifacts are in
  `rollouts/runtime_v3_response/run_20261002T063447Z_ca3cc2bc/`.
- Qwen selector evaluation artifacts are in
  `rollouts/runtime_v3_selector_eval/instruct_20261002.json`.
- Neither result tests pick/place, target grounding, task success, or
  manipulation reliability.
