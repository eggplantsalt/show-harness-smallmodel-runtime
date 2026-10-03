# Runtime V3 Readiness Evidence

M3.7 separates two facts that the former `SceneReadyEvidence` combined:
observable image motion and stable evidence for one semantically grounded
entity. Readiness is evaluated after the existing four RobotReady HOLDs.

## RobotReady

The LIBERO adapter completes four bounded zero-translation HOLDs through the
existing Runtime tick path. This is an initialization precondition, not an
object-stability measurement.

## SceneMotionReady

`SceneMotionReady` consumes only canonical RGB frames. It computes the mean
absolute channel difference over the full frame and divides by 255. There is no
ROI. Three consecutive frame pairs must score at or below `0.00005`. The
threshold is shared across tasks and frozen from the development captures:
the largest stable-tail per-task p95 was `0.0000351`, while settling motion
was substantially larger. The same RGB sequence therefore has the same scene
motion evidence for every phrase, identity, task, or object.

The interface does not accept semantic text, masks, identities, simulator pose,
depth, contact, task ID, or task success. A size or shape change in a SAM mask
cannot make this gate fail.

## Semantic grounding and EntityObservationReady

The raw semantic phrase stays in `EntitySpec`, `TaskSpec`, and episode logs.
At the SAM interface, the generic `GroundingQueryNormalizer` collapses
whitespace, case-folds, and removes at most one leading English determiner
(`the`, `a`, or `an`). It defines no entity aliases.

Existing candidate association must report `ANCHORED` or `SAME_TARGET`. A
three-observation window then checks adjacent mask IoU, centroid displacement
relative to the previous bbox diagonal, bbox-edge movement relative to that
diagonal, and relative area change. Current generic limits are respectively
`0.50`, `0.02`, `0.10`, and `0.20`. Invalid or lost identity clears the window.
The thresholds permit non-identical masks; they are not physical motion
measurements.

The observer selects a representative mask by pairwise-IoU medoid and uses the
coordinate-wise median centroid from that same-entity window. This initializes
the identity/reference evidence without mixing observations from another
entity key or normalized grounding query. The raw semantic phrase remains the
anchor's identity label.

## Combined gate and failure evidence

The initialization gate passes when scene motion is ready, entity evidence is
ready, and the current associated grounding and identity remain valid. A fixed
visual reference is then created from the robust same-entity representative.
Reports retain the two readiness records separately and distinguish
`SCENE_MOTION_NOT_READY`, `ENTITY_OBSERVATION_NOT_READY`,
`SEMANTIC_GROUNDING_FAILURE`, `IDENTITY_FAILURE`, and reference failures.

The 40-HOLD limit is shared across tasks. Oracle object poses, simulator depth,
ground-truth contact, and task success are absent from the observer and hold
interfaces. Stage A may record those values in a separate diagnostic report
after each returned observation; they grade false-ready and false-not-ready
outcomes only and do not feed back into Runtime.
