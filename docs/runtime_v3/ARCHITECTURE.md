# Runtime V3 Architecture

## Research question

Can a frozen compact VLM, such as Qwen 8B or 9B, control a robot reliably when
the runtime supplies verified options, one canonical belief state, and fresh
effect observations without training the model parameters?

The model does not directly own raw physical control. The runtime does not
replace the model with another opaque policy: deterministic geometry and state
checks produce a small option set, while the model resolves bounded semantic
choices that those tools cannot settle.

## Why the previous composition was unstable

The legacy composition gave several modules overlapping control privileges.
VisualHarness, VerifiedRuntime, VisualRoute, RecoveryPlugin, stage guards,
reflection, and placement review could each interpret state or affect the next
action. They also kept partially duplicated stage and holding state. This led
to policy competition and composition failures even when individual modules
worked in isolation.

## V3 control path

```text
Observer -> RobotObservation -> StateBuilder -> BeliefState
    -> OptionGenerator -> RuntimeOption[]
    -> Selector -> SelectedOptionID
    -> Arbiter -> ApprovedAction | REOBSERVE | ABORT
    -> Executor -> one bounded primitive or one bounded micro-motion
    -> Observer -> new BeliefState -> EffectObserver -> EffectRecord
```

## Online initialization preconditions

Object-relative control starts only after this Runtime-owned chain completes:

```text
Observation → RobotReady → SceneMotionReady ─┐
                                             ├→ Entity Identity → Reference
Semantic Entity Grounding → EntityObservationReady ┘
            → Verified ALIGN Option Generation
```

`RobotReady` is the existing four bounded zero-translation HOLDs. After it,
`SceneMotionReady` checks target-independent canonical RGB change while
`EntityObservationReady` checks a short normalized geometry and mask window for
one associated semantic entity. Both must pass before identity/reference
initialization. Their interfaces and frozen shared thresholds are documented
in [`READINESS_EVIDENCE.md`](READINESS_EVIDENCE.md). Oracle poses are excluded
from Runtime and remain experiment diagnostics only.

The combined readiness gate is a precondition for establishing the stable
visual reference. It does not choose the robot's next action and is not a
planner, policy, or recovery behavior. A timeout reports the responsible
readiness layer and control does not proceed.

The only canonical state is `core.runtime_v3.state.BeliefState`. It carries
task/step/stage, target identity and confidence, target pose or image position,
end-effector and gripper state, holding/contact evidence, relevant geometry,
the previous action and expected/observed effects, uncertainty, and evidence
references. Fields remain optional while adapters are being added.

M3.6 adds immutable `TaskSpec` and `EntitySpec` as the semantic input boundary,
plus one `RuntimeEntityState` field in that same `BeliefState`. The task/entity
specification carries instruction, entity key, semantic phrase, role, focus,
and the current ALIGN goal kind. RuntimeEntityState binds identity and visual
reference anchors to the entity key. The phrase is used for visual grounding;
ALIGN geometry and physical authority use observed geometry and the canonical
entity state, with no object-name or task-ID control branch. See
[`GENERALIZATION.md`](GENERALIZATION.md).

M3.8 inserts a semantic grounding boundary between `EntitySpec` and the
existing Runtime identity/reference chain:

```text
TaskSpec / EntitySpec
    ↓
Proposal Provider: direct text-SAM baseline + Qwen RGB semantic region
    ↓
SAM point refinement → generic Candidate Pool (K ≤ 4)
    ↓
Semantic Agent Selector: strict {candidate_id | NO_MATCH}
    ↓
Runtime Verifier: SceneMotionReady → SAM point association → EntityObservationReady
    ↓
RuntimeEntityState in the canonical BeliefState
    ↓
unchanged physical ALIGN contract
```

The proposal path has no class-agnostic automatic-mask API in the OpenETA
service. Qwen proposes bounded image regions from the named `EntitySpec`; the
existing OpenETA SAM3 point-prompt tool refines each region into masks. The
direct normalized text-SAM request remains in the same candidate pool and is
also measured as the baseline. Pool filtering and deduplication use only
generic mask score, normalized area, bounds, and overlap. Candidate IDs are
synthetic `C0`–`C3`; no task or simulator identifier reaches the selector.

The local capability audit found that OpenETA MCP exposes semantic `segment`
and positive/negative point `segment_points` prompts (up to 64 points, with
three multimask results). It does not expose an automatic-mask generator,
class-agnostic object proposals, a point-grid auto-proposal mode, or a box
endpoint. The underlying SAM3 processor has a box geometric-prompt method,
but the current OpenETA service does not expose it. M3.8 therefore uses the
specified fallback: Qwen supplies a semantic RGB region and SAM refines its
center point. Qwen is part of visual-semantic grounding only; its region never
becomes a robot coordinate or physical control input.

The selector receives the instruction, semantic phrase, full canonical RGB,
and candidate cards that pair a full-scene mask with a magnified candidate
detail. Its one bounded output is strict JSON selecting an existing candidate
or `NO_MATCH`. It cannot access direction, scale, pose, Arbiter, Executor, or
controller interfaces. A selected mask is not immediately promoted: proposals
wait until `SceneMotionReady`, then Runtime applies temporal mask association
and `EntityObservationReady` before establishing the visual reference in the
canonical state. Semantic selection accuracy is measured separately from this
temporal/physical validity check. Formal reports also require a post-hoc visual
audit before allowing an episode into ALIGN evaluation.

Responsibility ownership is explicit: proposal generation belongs to the
Perception/Runtime service; semantic candidate identity belongs to the Agent;
temporal and physical entity validity belongs to Runtime; physical action
selection and execution remain Runtime-owned. The Agent performs no physical
decision and cannot alter the ALIGN lattice, ranking, Arbiter approval,
Executor, or effect verification. See
[`OBSERVATION_BOUNDARY.md`](OBSERVATION_BOUNDARY.md) and
[`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md).

For pre-contact object-relative alignment, semantic target identity and the
control reference are separate. `TargetIdentityAnchor` associates later SAM
candidates with the selected instance. A stage-local `TargetReferenceAnchor`
freezes the initial associated mask centroid in canonical camera pixels. The
geometry resolver uses that fixed point for predicted and observed alignment
errors; later SAM centroids remain diagnostic and cannot replace it. See
[`TARGET_REFERENCE.md`](TARGET_REFERENCE.md) for establishment, invalidation,
and re-grounding rules.

M3.5 adds a generic `MetricEntityReference` produced from a deployable RGB-only
monocular metric-depth estimate, a SAM mask, and calibrated camera geometry. It
is carried as perception evidence but is not consumed by option generation,
candidate ranking, Arbiter, Executor, or termination. MuJoCo depth and target
pose comparisons live in experiment scripts and are evaluation evidence only;
the Stage A stability gate failed, so this result does not authorize 3D
alignment. See [`OBSERVATION_BOUNDARY.md`](OBSERVATION_BOUNDARY.md) and
[`METRIC_ENTITY_GROUNDING.md`](METRIC_ENTITY_GROUNDING.md).

`OptionGenerator` accepts explicit geometry candidates and turns them into
typed options. It cannot access the environment or Executor. Each option has an
ID, semantic description, preconditions, expected effect, bounded primitive,
confidence, and evidence references. The VLM sees only compact state fields
and those options. Its schema accepts an option ID, `REOBSERVE`, or `ABORT`.
Invalid output is returned as `INVALID_SELECTION` and never guessed.

`Arbiter` is the only action authority. It verifies option membership,
preconditions, evidence freshness, and the execution bound, then produces a
sealed `ApprovedAction`. `Executor` checks that approval and is the only module
that calls the atomic controller. A bounded micro-motion is one approval whose
spec fixes a semantic direction, requested displacement, and maximum of five
control ticks. Executor repeats only that direction, observes the EEF after
each tick, and stops at the target projection, on negative progress, at the
tick limit, or before crossing the approved workspace Z bounds. It cannot
change direction or replan inside the approved motion. Runner resumes option
selection only after this bounded realization completes.

The earlier temporal-response calibration repeated tokens as separate
single-tick V3 cycles. The bounded micro-motion calibration now measures the
production execution unit: four separately approved HOLD pre-settle ticks,
followed by one Arbiter-approved `MOVE_<DIRECTION>_SMALL` option. Executor
repeats that direction with a fresh physical observation after each control
tick. The target is 3 mm and the initial hard budget is five ticks; this is the
first conservative scale, not a claim that 3 mm is optimal.

Memory currently defines only `ExperienceRecord` and `ExperienceStore`. It
does not learn, alter policy, or inject legacy RSI/textual lessons.

## Principles

- Single State: one immutable `BeliefState` value is current at a time.
- Verified Options: every candidate carries its preconditions and evidence.
- Bounded Semantic Selection: the VLM chooses among IDs or requests reobserve/abort.
- Single Arbiter: only Arbiter creates an approved action.
- Bounded Execution: one primitive or one same-direction micro-motion is
  executed before option selection resumes.
- Effect Re-observation: expected effects are compared with new robot/image evidence.
- Stable Visual Reference: identity association may continue across frames, while
  the control point remains fixed until an explicit invalidation and re-ground.
