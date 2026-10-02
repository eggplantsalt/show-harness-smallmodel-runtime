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

## Qwen selector contract

The Qwen adapter stores only the existing VLM client and task instruction. It
receives a compact task/state summary and three smoke-only bounded options. It
does not receive an environment, controller, backend, executor, image, route
trace, or legacy memory. It requests one JSON object with exactly one key,
`selection`, whose value must be an exact listed option ID, `REOBSERVE`, or
`ABORT`. Invalid JSON, unknown IDs, and raw commands such as `MV_LEFT` become
`INVALID_SELECTION`; there is no fuzzy matching. This path stops after
selection and never invokes Arbiter or Executor.

## Current verification record

- Deterministic real smoke: `LIBERO_OBJECT`, task 2, init state 0, seed 0.
- Exactly one V3-approved action was sent. EEF Z moved from 0.260454598 m to
  0.261200609 m (observed `delta_z=+0.000746011 m`) after the bounded 5 mm
  `MV_UP` command; the minimum positive-Z effect threshold was 0.0001 m.
- Qwen offline selector: blocked before a completion request because no server
  responded at `http://127.0.0.1:8001/v1/models`. No robot action was issued in
  this mode.
- Neither result tests pick/place, target grounding, task success, or
  manipulation reliability.
