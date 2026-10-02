# Runtime V3 object-relative verified alignment

This milestone adds one bounded visual alignment option to Runtime V3. It is a
geometry check with a single 3 mm requested micro-motion, followed by a fresh
SAM3 segmentation and camera projection. It does not add grasping, placement,
recovery, or a multi-step alignment loop.

## Evidence path

The Runtime V3 LIBERO adapter renders `agentview` and `robot0_eye_in_hand` at
512×512. `LiberoObservationAdapter` preserves each simulator RGB array as
captured. `Sam3Client` encodes that same RGB array as PNG; its server reports
the decoded source dimensions. The Qwen visual smoke uses `VLMClient` image
encoding and records the exact image dimensions from the finalized request
audit. Neither path resizes or upsamples the input.

SAM3 receives the task-metadata target phrase `salad dressing`. Runtime V3
decodes the highest ranked returned mask only when its dimensions match the
source frame. It derives centroid, half-open bounding box, and pixel area from
the mask. SAM3's returned `score` is retained as a predicted mask-quality
signal; it is not treated as a calibrated probability. A missing or malformed
mask leaves the target invisible.
The Runtime V3 trial uses the explicit SAM3 API threshold `0.05` because the
returned target score for this small object is below its default `0.5`; the raw
score remains visible in each trial record and is still only the model's mask
quality signal, not a probability.

The LIBERO `agentview` raw frame uses the same 180° camera orientation already
configured for the existing policy view. Runtime applies that orientation in
the camera projection calibration so projected EEF coordinates share the raw
frame's pixel axes; RGB sent to SAM3 and Qwen remains unchanged.

`ObjectRelativeState` is part of the canonical Runtime V3 `BeliefState`. It
holds target visibility and mask measurements, the target centroid, the EEF
projection, their image-space error, the raw camera name, timestamp, and
source dimensions. There is no second belief-state representation.

## Runtime geometry owns physical direction

Runtime reads the current EEF position from proprioception and camera
intrinsics/extrinsics from the active MuJoCo `agentview` camera. It projects
the EEF with the existing `camera_geometry.project_point` function. For each
configured physical direction (`MV_FWD`, `MV_BACK`, `MV_LEFT`, `MV_RIGHT`,
`MV_UP`, `MV_DOWN`), it projects a hypothetical EEF point 3 mm along that
physical vector and computes:

```text
error_before = distance(target_centroid_px, projected_eef_px)
error_after_i = distance(target_centroid_px, projected_hypothetical_eef_px_i)
predicted_improvement_i = error_before - error_after_i
```

The Runtime geometry resolver chooses the valid candidate with the greatest
positive predicted improvement. It records all six candidates, including
invalid projections and workspace rejections. Pixel axes are never mapped
directly to robot axes. Missing target evidence, invalid camera/EEF projection,
invalid workspace bounds, or no positive candidate produces no alignment
option, so the selector returns `REOBSERVE`.

## Option and authority

The generated option exposed above the physical realization is
`ALIGN_TO_TARGET_SMALL`. Runtime seals the selected physical vector in a
`BoundedMicroMotionSpec` before Arbiter authorization. Its requested
displacement is 3 mm, each calibrated control tick is 5 mm, and the maximum is
five ticks. The existing Executor only runs the sealed direction and stops
when its displacement contract is reached, a boundary is encountered, or the
tick limit is reached. SAM3 runs on the initial observation and after the
bounded movement; intermediate ticks collect only fresh proprioception and
images for the bounded Executor's motion accounting.

Arbiter approves the object-relative option once. Runtime records the actual
physical projection and off-axis displacement returned by Executor. It then
resegments the target, reprojects the EEF, and reports whether the measured
target-to-EEF pixel error decreased. The per-trial artifact directory keeps
the before/after RGB images, binary masks, and overlays with target centroid
and projected EEF markers.

## Responsibility boundary

**Runtime owns geometry. Qwen owns bounded semantics.** The real alignment
trials use no Qwen call. The separate Qwen visual smoke receives one raw
high-resolution image, the compact `ObjectRelativeState`, and semantic choices
such as `OPTION_A = ALIGN_TO_TARGET`, `OPTION_B = REOBSERVE_TARGET`, and
`OPTION_C = ABORT`. The prompt and option descriptions contain no robot
direction vocabulary. The smoke has no controller or backend reference and
executes zero robot actions.

Qwen is not asked to choose LEFT/RIGHT or any other physical direction. If it
is used in a future stage, it may choose among already valid semantic options;
Runtime must still construct each option's geometry and physical realization.

## Trial command and output

With the existing OpenETA SAM3 service and the configured local Qwen endpoint
available, run:

```bash
OPENETA_SAM3_CHECKPOINT_PATH=/root/autodl-tmp/openeta-services/models/sam3/sam3.pt \
  /root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python \
  scripts/runtime_v3_object_relative_alignment.py
```

The script uses `LIBERO_OBJECT` task 2, seed 0, init states 0–2, four V3 HOLD
pre-settle ticks per trial, and at most one authorized bounded micro-motion per
trial. It writes per-trial `trial.json`, before/after `RGB`, `mask`, and
`overlay` PNGs, one `qwen_visual_no_action.json`, and an aggregate `summary.json`
under `rollouts/runtime_v3_object_relative_alignment/`.
