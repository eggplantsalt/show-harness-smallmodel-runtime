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
    -> Executor -> one bounded primitive
    -> Observer -> new BeliefState -> EffectObserver -> EffectRecord
```

The only canonical state is `core.runtime_v3.state.BeliefState`. It carries
task/step/stage, target identity and confidence, target pose or image position,
end-effector and gripper state, holding/contact evidence, relevant geometry,
the previous action and expected/observed effects, uncertainty, and evidence
references. Fields remain optional while adapters are being added.

`OptionGenerator` accepts explicit geometry candidates and turns them into
typed options. It cannot access the environment or Executor. Each option has an
ID, semantic description, preconditions, expected effect, bounded primitive,
confidence, and evidence references. The VLM sees only compact state fields
and those options. Its schema accepts an option ID, `REOBSERVE`, or `ABORT`.
Invalid output is returned as `INVALID_SELECTION` and never guessed.

`Arbiter` is the only action authority. It verifies option membership,
preconditions, evidence freshness, and the one-primitive bound, then produces a
sealed `ApprovedAction`. `Executor` checks that approval and is the only module
that calls the atomic controller. Each execution is followed by a new
observation before another selection.

The temporal calibration procedure can repeat the same bounded token across
multiple control ticks to measure its response. Each tick still runs a fresh
observe → option → selector → Arbiter → Executor cycle, and each Executor call
remains one tick. The experiment does not change the production option or
execution bound.

Memory currently defines only `ExperienceRecord` and `ExperienceStore`. It
does not learn, alter policy, or inject legacy RSI/textual lessons.

## Principles

- Single State: one immutable `BeliefState` value is current at a time.
- Verified Options: every candidate carries its preconditions and evidence.
- Bounded Semantic Selection: the VLM chooses among IDs or requests reobserve/abort.
- Single Arbiter: only Arbiter creates an approved action.
- Bounded Execution: one primitive is executed before observation resumes.
- Effect Re-observation: expected effects are compared with new robot/image evidence.
