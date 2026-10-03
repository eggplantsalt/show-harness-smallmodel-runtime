# Runtime V3 Cross-Object Generalization

## M3.6 question

M3.6 asks whether the established projective ALIGN contract transfers across
LIBERO_OBJECT manipulands and initial layouts while task semantics change and
the Runtime physical path stays frozen. This phase uses RGB/SAM masks, camera
calibration, EEF proprioception, the fixed visual reference, the existing
six-direction by three-scale candidate lattice, deterministic ranking, one
Arbiter approval per semantic step, bounded Executor motion, and fresh effect
verification. Metric depth is telemetry only. No GRASP or NearTarget behavior
is part of this gate.

## Minimal task/entity interface

- `TaskSpec` is an immutable instruction, entity list, focus entity key, and
  `ALIGN` goal kind.
- `EntitySpec` is an immutable key, semantic phrase, and role. It contains no
  coordinates, directions, distances, scales, thresholds, or controller fields.
- `RuntimeEntityState` is stored in the canonical `BeliefState` alongside the
  existing object-relative evidence. It binds the entity key and semantic
  phrase to current identity/reference anchors and visible/valid flags; it does
  not create a second world-state store.
- `ReferenceTaskCompiler` reads the frozen experiment binding and produces a
  `TaskSpec`. It is a test fixture/compiler for known semantic bindings, not a
  task planner.

The semantic phrase is consumed by the SAM grounding adapter. The physical
ALIGN generator receives the canonical belief state and uses the entity key
only as evidence metadata. Direction, scale, geometry, ranking, arbitration,
execution, and effect checks do not branch on task IDs or object labels.

## Responsibility scope

| Layer | Scope | M3.6 owner |
|---|---|---|
| Runtime Core | Task-independent observation/state/authority plumbing | Existing V3 runtime |
| ALIGN skill contract | Object-agnostic projective geometry, direction/scale ranking, bounded action, effect check | Runtime |
| TaskSpec / EntitySpec | Task-specific instruction and semantic entity binding | Reference compiler in Stage A; one-call Qwen compiler after the Stage A gate |
| RuntimeEntityState | Per-episode identity and visual-reference evidence | Perception observer, held in canonical BeliefState |
| Robot adapter/config | Robot- and simulator-specific sensing and action realization | LIBERO adapter/controller |

## Frozen Stage A task manifest

The pre-rollout manifest is
`experiments/runtime_v3/cross_object_align_manifest.json`. It fixes anchor task
2 and three unseen targets (tasks 0, 6, and 7), seed 0, init states 0–2, a
maximum of six semantic ALIGN steps per episode, and the conditional expansion
to states 3–5 only after all three initial episodes for an unseen task have net
improvement with valid grounding, identity, reference, and SceneReady evidence.
The task list is not changed in response to rollout outcomes.

The frozen scale contracts come from the existing task-2 calibration artifact:
3 mm / 5 ticks, 6 mm / 7 ticks, and 9 mm / 10 ticks. M3.6 reuses those values
unchanged for every selected task. No per-object threshold, direction rule, or
scale rule is permitted.

## Semantic and physical failure separation

Reports distinguish semantic binding, perception, SceneReady, identity,
reference, lack of a valid physical candidate, execution, and effect
verification. Reference binding is reported separately from any later Qwen
binding experiment so a semantic parse failure is not miscounted as physical
transfer failure.

## Corrected Stage A result

The valid formal run used commit `3ddc109791c5cca36294f61a8095195d690ff928`,
one source/config signature across all tasks, and no oracle target pose, GT
depth, GT contact, or task-success inputs. Tasks ran in frozen order 2, 0, 6,
and 7 with seed 0.

| Task | Initial episodes | Expansion | ALIGN actions | Positive effects | Judgment |
|---|---:|---:|---:|---:|---|
| 2, salad dressing | 3/3 | — | 18 | 18/18 | PASS |
| 0, alphabet soup | 3/3 gate | states 3–5; 6/6 total | 36 | 36/36 | PASS |
| 6, butter | SceneReady 0/3 | none | 0 | unobserved | FAIL before ALIGN |
| 7, milk | 3/3 gate | states 3–5; 6/6 total | 36 | 36/36 | PASS |

All 15 episodes that reached ALIGN were monotonic, with 90/90 positive
per-step effects. The unseen-object transfer passes are tasks 0 and 7. The
overall cross-object judgment is PARTIAL because butter failed SceneReady before
semantic ALIGN evidence existed; its grounding, identity, and reference rates
are unknown, rather than zero. Every executed choice used 9 mm, while selected
directions varied across tasks. This is an observation about the run, not an
object-specific rule. Metrics were post-processed from saved traces to correct
failure taxonomy and unobserved-rate reporting; both raw and corrected summaries
are preserved in the run directory.

## Experimental status

The task manifest is frozen. The first formal attempt at commit `253afeb` is
preserved at
`rollouts/runtime_v3_cross_object_alignment/run_20261003T082839Z_bac7770d`, but
is invalid as transfer evidence: approval tracing raised `NameError: uuid is
not defined` before Executor invocation. No physical actions occurred. The
failure was discovered after the frozen run ended and is excluded from the
metrics above. The corrected Stage A run is
`rollouts/runtime_v3_cross_object_alignment/run_20261003T083700Z_9865926b/`.
The one-call Qwen semantic compiler and offline/physical Stage B drivers are
implemented. Stage B is pending; Qwen may output only semantic entity binding
and cannot select a physical direction or scale.
