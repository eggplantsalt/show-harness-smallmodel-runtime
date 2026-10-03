# Runtime V3 Cross-Object Generalization

## M3.7 readiness evidence boundary and results

M3.6 exposed two different pre-ALIGN failures: physically settled butter lost
SAM evidence, and the same alphabet-soup RGB frames produced candidates for
`alphabet soup` but none for Qwen's `the alphabet soup`. The old SceneReady
gate only consumed target identity and mask geometry, so it conflated physical
scene stability and semantic entity evidence. M3.7 separates those inputs into
`SceneMotionReady` (full-frame canonical RGB only) and
`EntityObservationReady` (one associated entity's normalized visual window).
See [`READINESS_EVIDENCE.md`](READINESS_EVIDENCE.md) for the frozen rules.

The held-out manifest freezes tasks 1 (cream cheese) and 8 (chocolate pudding),
init states 0–2, before formal post-refactor runs. Stage A completed all 18
HOLD-only episodes. RobotReady and SceneMotionReady passed 18/18. Entity
observation, grounding, identity, and reference were valid in 9/18. Oracle
grading found zero false-ready episodes and nine false-not-ready episodes:
butter, cream cheese, and chocolate pudding each remained without stable SAM
evidence across all three states, even though their target pose was settled at
the diagnostic tail. Oracle measurements were not passed into Runtime.

| Task | Episodes | SceneMotionReady | EntityObservationReady | Grounding / identity / reference | Ticks to ready | Timeouts | Oracle false-ready / false-not-ready |
|---|---:|---:|---:|---:|---|---:|---:|
| 0 alphabet soup | 3 | 3/3 | 3/3 | 3/3 / 3/3 / 3/3 | 14, 13, 11 | 0 | 0 / 0 |
| 2 salad dressing | 3 | 3/3 | 3/3 | 3/3 / 3/3 / 3/3 | 10, 12, 13 | 0 | 0 / 0 |
| 6 butter | 3 | 3/3 | 0/3 | 0/3 / 0/3 / 0/3 | — | 3 | 0 / 3 |
| 7 milk | 3 | 3/3 | 3/3 | 3/3 / 3/3 / 3/3 | 11, 12, 15 | 0 | 0 / 0 |
| 1 cream cheese (held out) | 3 | 3/3 | 0/3 | 0/3 / 0/3 / 0/3 | — | 3 | 0 / 3 |
| 8 chocolate pudding (held out) | 3 | 3/3 | 0/3 | 0/3 / 0/3 / 0/3 | — | 3 | 0 / 3 |

Task 0 is not a valid target-grounding pass: visual review shows SAM consistently
associated the neighboring Milk carton instead of the requested soup can. The
raw phrase sensitivity result remains `QUERY_SENSITIVITY`: the same saved RGB
frames returned zero candidates for `the alphabet soup` and candidates for
`alphabet soup`. The generic determiner normalizer makes both inputs use the
same query, but it cannot correct the wrong selected entity. The frozen held-out
tasks are retained as failures; no replacements were selected.

Reference-binding Stage B was selected after Stage A and frozen in
`experiments/runtime_v3/m3_7/stage_b_reference_manifest.json`. Task 2 passed
3/3 episodes and task 7 passed 6/6 after the existing expansion rule. All 54
ALIGN effects were positive and all nine episodes monotonic. Task 2 improved
164.56 → 135.13 px (17.90% normalized reduction, DOWN×18, 9mm×18). Task 7
improved 166.70 → 141.53 px (15.11%, RIGHT×34 / DOWN×2, 9mm×36). One approval
per semantic step was preserved. The runtime and ALIGN source signatures were
constant across tasks. Full trace:
[`align_stage_b_summary.json`](../../experiments/runtime_v3/m3_7/align_stage_b_summary.json).

After Stage B, Qwen Stage C made one call per instruction. Its semantic-only
outputs were `the salad dressing` and `the milk`; both schemas and bindings
matched (2/2), and the uniform runtime normalizer sent `salad dressing` and
`milk` to SAM. Both tasks reached valid readiness in all episodes. The 54 ALIGN
steps had the same positive-effect, monotonicity, direction, scale, and
normalized-reduction results as reference binding. Qwen emitted no physical
direction or scale and authorized zero actions. This supports transfer across
the tested salad-dressing / milk pair and article variation, but not across the
two held-out package appearances because both failed semantic grounding.

The overall M3.7 judgment is **PARTIAL**. Target-independent scene-motion
readiness generalizes across all six tested task IDs. Same-entity observation
readiness still depends on SAM grounding availability and semantic correctness;
the frozen held-out evaluation exposes this as the remaining general contract
failure. Do not proceed to NearTarget on this evidence alone. Stage A, Stage B,
Stage C, binding, and contact-sheet artifacts are recorded under
`experiments/runtime_v3/m3_7/`.

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
verification. M3.8 uses `GROUNDING_PROPOSAL_MISS` when the pool has no target
proposal, `SEMANTIC_SELECTION_WRONG` when a different entity is selected,
`SEMANTIC_SELECTION_NO_MATCH` when the selector abstains,
`ENTITY_OBSERVATION_NOT_READY` when the selected mask cannot pass Runtime's
temporal gate, `IDENTITY_FAILURE` when association is lost,
`SCENE_MOTION_NOT_READY` for the independent scene gate, and
`PHYSICAL_ALIGNMENT_FAILURE` for failed ALIGN execution or effects. These
states are reported separately; low proposal coverage is not called
perception jitter. Reference binding is reported separately from any later
Qwen binding experiment so a semantic parse failure is not miscounted as
physical transfer failure.

## Corrected Stage A result

The valid formal run used commit `3ddc109791c5cca36294f61a8095195d690ff928`,
one source/config signature across all tasks, and no oracle target pose, GT
depth, GT contact, or task-success inputs. Tasks ran in frozen order 2, 0, 6,
and 7 with seed 0.

| Task | Episodes / SceneReady | Grounding / identity / reference | ALIGN steps; positive effects | Monotonic episodes | Initial → final error (normalized reduction) | Mean improvement / step | Control ticks / step | Directions / scales | Judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2, salad dressing | 3 / 3 | 3/3 / 3/3 / 3/3 | 18; 18/18 | 3/3 | 164.50 → 135.26 px (17.79%) | 4.874 px | 145 / 18 = 8.06 | DOWN×18 / 9mm×18 | PASS |
| 0, alphabet soup | 6 / 6 | 6/6 / 6/6 / 6/6 | 36; 36/36 | 6/6 | 208.81 → 184.31 px (11.74%) | 4.083 px | 291 / 36 = 8.08 | DOWN×32, RIGHT×4 / 9mm×36 | NUMERIC PASS; TARGET INVALIDATED |
| 6, butter | 3 / 0 | unobserved / unobserved / unobserved | 0; unobserved | 0/3 | unobserved | unobserved | 0 | — | FAIL before ALIGN |
| 7, milk | 6 / 6 | 6/6 / 6/6 / 6/6 | 36; 36/36 | 6/6 | 166.61 → 141.42 px (15.13%) | 4.198 px | 296 / 36 = 8.22 | RIGHT×34, DOWN×2 / 9mm×36 | PASS |

All 15 episodes that reached ALIGN were monotonic, with 90/90 positive
per-step effects. M3.6's task-0 measurements remain historical numerical
results, but M3.7 visual review showed its stable mask was the neighboring Milk
carton. The M3.6 alphabet-soup correct-target ALIGN claim is therefore
**INVALIDATED**; only task 7 counts as valid unseen-object transfer from that
run. The overall cross-object judgment remains PARTIAL because butter failed
SceneReady before semantic ALIGN evidence existed; its grounding, identity,
and reference rates are unknown, rather than zero. Every executed choice used 9 mm, while selected
directions varied across tasks. This is an observation about the run, not an
object-specific rule. Metrics were post-processed from saved traces to correct
failure taxonomy and unobserved-rate reporting; both raw and corrected summaries
are preserved in the run directory.

## Qwen semantic binding and Stage B result

The local model was `Qwen/Qwen3-VL-8B-Instruct`. It was called once for each
frozen instruction (four calls, temperature 0, no retry). All four outputs
passed the strict JSON schema and selected `target` / `MANIPULAND`; no physical
fields appeared. Raw phrases were `the salad dressing`, `the alphabet soup`,
`the butter`, and `the milk`. Literal phrase equality was 0/4. The saved-response
audit also reports semantic match after case folding, whitespace collapse, and
removing one leading English article, yielding 4/4. It preserves the raw
responses, makes no additional model calls, and passes the original Qwen phrase
to Runtime. The runtime never receives a Qwen direction or scale.

The formal Qwen physical run used commit
`b04742d1732ec2db05019352ffd07b89a1c637f5`, the same config hash as Stage A,
and the same physical source files as reference commit
`3ddc109791c5cca36294f61a8095195d690ff928`.

| Task | Episodes / SceneReady | Grounding / identity / reference | ALIGN steps; positive effects | Monotonic episodes | Initial → final error (normalized reduction) | Mean improvement / step | Control ticks / step | Directions / scales | Judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2, salad dressing | 3 / 3 | 3/3 / 3/3 / 3/3 | 18; 18/18 | 3/3 | 164.52 → 135.24 px (17.81%) | 4.879 px | 145 / 18 = 8.06 | DOWN×18 / 9mm×18 | PASS |
| 0, alphabet soup | 3 / 0 | unobserved / unobserved / unobserved | 0; unobserved | 0/3 | unobserved | unobserved | 0 | — | FAIL: SceneReady 0/3 |
| 6, butter | 3 / 0 | unobserved / unobserved / unobserved | 0; unobserved | 0/3 | unobserved | unobserved | 0 | — | FAIL: SceneReady 0/3 |
| 7, milk | 6 / 6 | 6/6 / 6/6 / 6/6 | 36; 36/36 | 6/6 | 166.64 → 141.45 px (15.12%) | 4.198 px | 296 / 36 = 8.22 | RIGHT×34, DOWN×2 / 9mm×36 | PASS |

Nine episodes reached semantic ALIGN: 54/54 effects improved alignment and
9/9 episodes were monotonic. Task 0 passed with reference binding in Stage A
but did not reach SceneReady with Qwen's phrase. The available trace does not
show a semantic ALIGN observation, so grounding, identity, and reference rates
remain unknown; this points to a semantic-grounding / SceneReady boundary for
follow-up, but does not establish the root cause. Task 6 failed SceneReady in
both modes. The Qwen cross-object physical judgment and the semantic-coprocessor
judgment are both PARTIAL: the anchor and one unseen object pass, while one
unseen object has a pre-ALIGN failure. Failure taxonomy is
`SCENE_NOT_READY×6`.

The cross-task contact sheet and task 2 / task 7 contact sheets were manually
reviewed. Qwen-bound overlays identified the dressing bottle and milk carton.
The task 6 preflight image was also reviewed and shows the named butter package.
Compact machine-readable reports, both code-change audits, raw/reconciled Qwen
outputs, and the excluded preflight abort are retained in
`experiments/runtime_v3/m3_6/`; full frame-level traces remain in the referenced
local rollout directories.

## Experimental status

The task manifest is frozen. The first formal attempt at commit `253afeb` is
preserved at
`rollouts/runtime_v3_cross_object_alignment/run_20261003T082839Z_bac7770d`, but
is invalid as transfer evidence: approval tracing raised `NameError: uuid is
not defined` before Executor invocation. No physical actions occurred. The
failure was discovered after the frozen run ended and is excluded from the
metrics above. The corrected Stage A run is
`rollouts/runtime_v3_cross_object_alignment/run_20261003T083700Z_9865926b/`.
The one-call Qwen compiler and offline/physical Stage B completed. A first
driver preflight at `f431ff8` aborted before simulated episodes because of
article-comparison and self-signature audit bugs; it executed zero ALIGN
actions and is preserved in `experiments/runtime_v3/m3_6/`. Both were corrected
before the formal Stage B commit. Qwen remained a semantic coprocessor: it
provided `TaskSpec` semantic binding, while Runtime retained all physical
authority.

## M3.8 generalized grounding status

The M3.8 manifest froze the remaining compatible unseen object tasks before
formal runs: BBQ sauce, ketchup, tomato sauce, and orange juice, each at init
states 0–2 with seed 0. The K=4 proposal pool plus frozen Qwen prompts reached
Runtime reference readiness on 12/12 Stage A episodes. Human RGB review found
the intended target mask in every pool, but the selector chose the correct
object in only 4/12 and a wrong object in 8/12. The four correct cases are
ketchup init state 2 and orange juice init states 0–2. Only those four qualify
for M3.8 ALIGN evidence; stable temporal association and a valid reference do
not establish correct semantic identity. Stage B completed 18/18 positive
effects across three executions (3/3 monotonic; 17.37% mean normalized error
reduction), with one eligible rerun failing closed on candidate-ID mismatch
before action. Stage C completed the orange-juice instruction in all three
states (18/18 positive; 3/3 monotonic; 18.46% mean normalized reduction); its
ketchup run failed closed before ALIGN. There is partial cross-object ALIGN
evidence for ketchup and orange juice, and partial Agent-coprocessor evidence;
generic grounding remains the blocker. The soup development path abstained
instead of binding the previous Milk mask. Detailed candidate, Runtime
reference, and ALIGN reports are in `experiments/runtime_v3/m3_8/`.
