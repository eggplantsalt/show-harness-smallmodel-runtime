# Runtime V3 LIBERO Integration

## Reused legacy low-level interfaces

| V3 adapter / purpose | Existing interface used |
| --- | --- |
| `LiberoEnvironmentAdapter` creates, resets, steps, checks, and closes the simulator | `core.sim.libero_task.make_libero_task`, `reset_libero`, `step_libero`, `libero_success`; task reset is configured with `settle_steps=0` |
| `LiberoObservationAdapter` reads cameras and robot state | `libero_rgb`, `libero_tcp`, `libero_quat`, and `libero_gripper_width` from `core.sim.libero_task` |
| `LiberoPrimitiveBackend` converts a sealed V3 action to one LIBERO action | Existing `interpreters.libero_atomic_controller.LiberoAtomicController`; `action_for_atomic("MV_UP", step_m=0.005)` produces the existing 7D OSC_POSE action, and the backend submits it through the environment adapter once |
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
→ Executor.execute
→ LiberoPrimitiveBackend.execute_approved_action
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

## Qwen selector contract

The Qwen adapter stores only the existing VLM client and task instruction. It
receives a compact task/state summary and three smoke-only bounded options. It
does not receive an environment, controller, backend, executor, image, route
trace, or legacy memory. It requests one JSON object with exactly one key,
`selection`, whose value must be an exact listed option ID, `REOBSERVE`, or
`ABORT`. Invalid JSON, unknown IDs, and raw commands such as `MV_LEFT` become
`INVALID_SELECTION`; there is no fuzzy matching. This path stops after
selection and never invokes Arbiter or Executor.

The fixed no-action evaluation used the same adapter for 12 live cases (eight
clear choices and four low-confidence reobserve cases), then injected four
invalid raw outputs (`MV_LEFT`, `move left`, a disallowed JSON selection, and an
extra schema key). The active local server at `127.0.0.1:8002` was healthy but
served `Qwen/Qwen3-VL-8B-Thinking`, not the configured Instruct model. All 12
live responses were truncated before a final selection at the adapter's
64-token limit: schema validity 0/12, option validity 0/12, accuracy 0/12. The
four injected invalid outputs all returned `INVALID_SELECTION`. No Executor
was constructed and no robot action ran. The configured Instruct model is
already present at `/root/autodl-tmp/huggingface/hub/Qwen/Qwen3-VL-8B-Instruct`,
but its configured endpoint at `127.0.0.1:8001` was not started because GPU 0
was occupied by the separate active 8B Thinking server. No model weights were
downloaded or changed.

## Current verification record

- Deterministic real smoke: `LIBERO_OBJECT`, task 2, init state 0, seed 0.
- Exactly one V3-approved action was sent. EEF Z moved from 0.260454598 m to
  0.261200609 m (observed `delta_z=+0.000746011 m`) after the bounded 5 mm
  `MV_UP` command; the minimum positive-Z effect threshold was 0.0001 m.
- Qwen selector evaluation artifacts are in
  `rollouts/runtime_v3_calibration/run_20261002T050110Z_c209f2e7/qwen_selector_evaluation.json`.
- Neither result tests pick/place, target grounding, task success, or
  manipulation reliability.
