# Runtime V3 Target Reference Stability

## Two different anchors

`TargetIdentityAnchor` answers **which object is being followed**. It stores
the initial associated SAM mask and is used to compare later candidates by
mask overlap and local continuity.

`TargetReferenceAnchor` answers **where the stable control reference is for
this stage**. It copies the centroid of the initial associated target mask,
its source frame and box, the mask area, the camera name, and a signature of
the calibrated camera and canonical image coordinates.

Semantic instance evidence is not the same as a stable control reference. SAM
can keep associating the same semantic instance while changes in the predicted
mask shape move its centroid. Runtime must not equate “same semantic instance”
with “stable servo reference”.

## Establishment and lifetime

The first valid target identity association creates one reference point from
that frame's mask centroid. The reference belongs to the current pre-contact
alignment stage, current camera, and current visual coordinate convention. It
is not an object pose and is never reconstructed from simulator state.

While the anchor is valid, the geometry resolver and alignment error use its
fixed `reference_point_px`. Later SAM centroids remain available for identity,
visibility, motion evidence and visualization diagnostics. They cannot change
the reference point. A SAM miss or identity-association failure suppresses a
new alignment option; it does not silently create another reference.

The reference is invalidated by a camera-signature change, explicit possible
contact evidence, explicit target-motion evidence, grasp/release events, or an
explicit re-ground request. The current Runtime V3 LIBERO observer has no
reliable contact detector, so contact invalidation is wired to existing
evidence fields (`possible_contact`, `contact_detected`, or `contact_state`)
and remains a TODO for a later sensor-backed phase. A mask-centroid shift by
itself is not treated as target-motion evidence because it is the instability
this reference is meant to avoid.

Re-grounding is explicit: invalidate the previous reference, confirm the target
identity at the new stage, then call
`ObjectRelativePerceptionObserver.request_target_reference_reground()`. This
resets the image-space identity association and establishes a new visual
reference from the next valid observation. No action can be generated while
the reference is invalid.

## Control and verification metrics

For a valid anchor, the formal alignment error is

```text
alignment_error_px = || reference_point_px - projected_eef_px ||_2
```

The expected improvement comes from projecting the candidate EEF position
after the requested 3 mm direction. The observed improvement uses the same
fixed reference before and after execution. The prediction residual is
`actual_improvement_px - predicted_improvement_px`. Both predicted and observed
EEF projections, the reference point, and the error terms are stored per trial.

The dynamic SAM metric is retained as a diagnostic only. It uses the initial
SAM centroid before execution and the post-action associated SAM centroid
after execution. Its difference from the frozen-reference result quantifies
how re-segmentation affects evaluation; it does not authorize or steer motion.

## Oracle and authority boundary

The trial script may read the target body position to a separate
`oracle_target_motion_diagnostic` record. That value is not added to
`RobotObservation`, `BeliefState`, geometry candidates, options, selection,
Arbiter, Executor, or Qwen. Runtime geometry reads the visual reference and
camera-projected robot proprioception only.

Before Arbiter authorization, the trial runner checks output and artifact
directories, saves the pre-action visual evidence, writes and reads back a
`PRE_ACTION_READY` record, and captures any oracle pose only into that
diagnostic record. A write failure raises before the Arbiter can seal an action.

## Scope and limits

This is a stage-local visual pixel reference for a fixed camera and a target
assumed stationary before contact. It is not a learned tracker, object pose,
world-space target estimate, grasp reference, or persistent memory. The current
evidence does not establish a general contact detector or cross-task control
reliability.

## 2026-10-02 evaluation result

The three bounded LIBERO re-tests kept the camera signature unchanged and
retained semantic target identity in all trials. The fixed-reference error
improved in all three trials, while the dynamic SAM error worsened by about
24.2–24.6 px. However, the diagnostic-only target body position changed by
`[0.0000013, -0.0000107, -0.056593] m` in each action window (norm about
56.593 mm). Runtime saw no possible-contact, grasp, release, or target-motion
signal, so the visual reference remained marked valid. This means the
stationary-target assumption was not verified after motion. The recorded
fixed-reference improvement only proves motion toward the initial pixel point;
it does not prove improved alignment to the moved target. The measurements do
not establish whether the bounded motion caused the target displacement because
there is no matched no-action control trial.

Do not use this result to authorize multi-step alignment. First explain the
target movement and establish an observation-only way to distinguish actual
target motion from segmentation-shape changes without using oracle state in
Runtime.
