# Runtime V3 Decision Responsibilities

## Runtime-owned decisions

Runtime V3 owns decisions that depend on measured geometry and physical limits:

- deterministic geometry and geometric validity checks;
- mapping a valid motion choice to its configured physical direction;
- physical bounds and workspace validity;
- realizing one approved option within its fixed execution budget;
- observing and verifying the resulting physical effect;
- stopping when the target projection is reached, progress reverses, the tick
  budget is exhausted, or the next tick would cross a workspace boundary.
- applying the same ALIGN candidate generation/ranking and effect contract to
  any valid `RuntimeEntityState`, independent of its semantic phrase.
- separating target-independent RGB scene motion from target-specific entity
  observation stability before establishing identity/reference evidence.

For a bounded micro-motion, the Arbiter approves one semantic option and seals
its direction, displacement request, tick limit, starting EEF position, and
workspace Z bounds in `ApprovedAction`. The Executor can repeat that same
direction and observe after each tick. It cannot choose a new direction or
replan while realizing that approval.

## Qwen-owned decisions

Qwen may decide among the current physically valid semantic options. Its
responsibilities include:

- resolving semantic ambiguity in the task and observation;
- choosing a stage-level or task-conditioned option;
- requesting an additional observation with `REOBSERVE`;
- choosing among multiple options that Runtime has already established as
  physically valid.
- optionally binding an instruction to an `EntitySpec` after the reference-
  binding cross-object gate passes; this output stays semantic and cannot carry
  a physical direction, displacement, scale, or controller action.

Qwen receives option IDs and compact state. It does not receive controller
tokens or an execution interface. Qwen is a bounded semantic decision maker.

## Readiness evidence ownership

`SceneMotionReady` owns only full-frame canonical RGB temporal differences.
`EntityObservationReady` owns only normalized stability evidence from an
associated SAM entity and its semantic grounding query. The query normalizer
is a linguistic boundary adapter; it preserves the raw TaskSpec phrase and
defines no object-specific aliases. Neither gate reads simulator pose, depth,
contact, task identity, or task success.

## Selector evaluation interpretation

The Instruct evaluation returned valid schemas and listed options in all 12
live cases, with 9/12 expected choices overall. Its geometric category was
1/4, while ambiguous-state reobservation was 2/2, stage choice was 2/2, and
task-conditioned choice was 4/4. The geometric result is assigned to Runtime's
deterministic geometry responsibility; it is not a reason to tune the prompt or
give Qwen physical motion authority. Semantic selection remains useful when
Runtime offers more than one valid bounded option.
