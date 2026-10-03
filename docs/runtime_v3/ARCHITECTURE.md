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
Observation → RobotReady → SceneReady → Target Grounding → Identity Anchor
            → Reference Anchor → Verified Option Generation
```

`RobotReady` and `SceneReady` are separate milestones. RobotReady confirms the
existing four bounded zero-translation HOLDs completed. SceneReady checks that
the same associated SAM target has a consecutive visual window with stable
centroid, bounding-box edges, and mask area. The current gate uses a three
observation window, thresholds of 0.02 px / 0 px / 1 px, and a maximum of 40
HOLD ticks. Oracle poses are excluded from this gate and remain experiment
diagnostics only.

SceneReady is a Runtime precondition for establishing a stable visual
reference. It does not choose the robot's next action and is not a planner,
policy, or recovery behavior. After it passes, target grounding and the two
stage-local anchors provide evidence for ordinary verified option generation.
If the bounded window expires, initialization reports `SCENE_READY_TIMEOUT`
and control does not proceed.

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
