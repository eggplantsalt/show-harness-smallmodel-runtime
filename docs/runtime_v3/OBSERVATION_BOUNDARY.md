# Runtime V3 Observation Boundary

## Formal Runtime inputs

| Input | Current source | Real-world source |
|---|---|---|
| RGB | Canonical agentview and wrist images | Robot-mounted RGB cameras |
| Proprioception | EEF position/orientation and gripper width in the LIBERO observation | Robot state estimator, joint encoders, forward kinematics, and gripper sensors |
| Camera intrinsics and extrinsics | LIBERO calibration metadata adapted as `CameraCalibration` | Intrinsic calibration and fixed-camera or hand-eye extrinsic calibration |
| Task instruction | Natural-language task description | Operator or task system |
| Visual target mask | SAM output from RGB | Same deployable segmentation service/model class |
| Monocular metric depth | Pinned MoGe-2 ViT-L estimate from canonical RGB only | Same RGB-only model on the robot's RGB stream |
| Formal metric entity reference | Robust mask-conditioned estimated depth plus camera calibration | Same visual mask, depth estimate, and calibrated geometry |

The formal estimator has no parameter for depth render, object pose, body ID,
contact, or task success. Camera calibration is sensor calibration, not scene
ground truth. Formal Runtime execution must remain possible with the equivalent
real-world sensing stack shown above. At this stage, depth quality on task
images is an empirical gate, not an assumed capability.

## Diagnostic only

| Diagnostic evidence | Used for |
|---|---|
| MuJoCo / LIBERO rendered depth | Post-inference depth and reference error comparisons |
| Simulator target pose and body ID | Reference bias / visible-surface offset analysis and oracle distance trends |
| MuJoCo contact records | Offline contact timing analysis |
| LIBERO task success flag | Offline evaluation only; not used for Runtime stopping |

Diagnostic data is acquired by the experiment scripts, outside
`core/runtime_v3`. `DiagnosticSimulatorDepthProvider` is implemented under
`scripts/`. The formal LIBERO observation adapter exposes only RGB and
proprioception; it does not copy depth into `RobotObservation`, `RawObservation`,
or Runtime evidence. The diagnostic experiment may inspect the cached simulator
observation after Runtime perception and decisions have been formed.

**Diagnostic information may evaluate Runtime, but may never change Runtime
behavior.** Simulator ground truth is evaluation evidence, not Runtime
information.

## Enforced code boundary

- `core/runtime_v3/depth.py` defines the deployable RGB-only depth provider and
  rejects non-deployable source labels.
- `core/runtime_v3/metric_entity.py` rejects `simulator_gt` references.
- `core/runtime_v3/adapters/libero_observation.py` does not read simulator depth
  fields or success state.
- The diagnostic depth reader and GT comparisons live in
  `scripts/runtime_v3_metric_entity_depth_diagnostic.py` and
  `scripts/runtime_v3_metric_entity_grounding.py`.
- Tests scan the Runtime core for simulator-depth hooks and check that
  diagnostic fields do not change option ranking or termination.

No M3.5 diagnostic value is used for option generation, candidate ranking,
NearTarget decisions, arbitration, execution, effect verification, prompts, or
termination. The existing 2D multiscale objective remains the only alignment
objective in this phase. No grasp, place, release, or 3D control is part of the
phase.
