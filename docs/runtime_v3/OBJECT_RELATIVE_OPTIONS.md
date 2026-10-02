# Runtime V3 canonical images and same-target alignment

This phase adds one canonical image convention and one-frame target identity
continuity to the bounded Runtime V3 alignment experiment. It does not add
closed-loop alignment, recovery, grasping, placement, or a memory component.

## Image convention

The vendored robosuite sets `IMAGE_CONVENTION = "opengl"`; its mapping is `1`,
and `RobotEnv.camera_rgb` returns `img[::convention]`. The captured LIBERO RGB
array therefore keeps the OpenGL bottom-up row order. `libero_rgb` only converts
to `uint8` and makes the array contiguous. The V3 observation adapter preserves
that array. A same-frame contact sheet confirmed that the red carton's text is
upside down in raw and becomes upright after only a vertical flip. A horizontal
flip leaves it upside down; a 180-degree rotation also mirrors the scene
horizontally and is not the required correction.

`CanonicalImageAdapter` is the only V3 spatial image transform. It vertically
flips the raw array once, before storage in `RobotObservation` or any call to
SAM3. SAM3 PNG serialization and Qwen's `image_to_data_url` preserve the supplied
array's orientation and dimensions. Runtime V3 RGB artifacts and overlays use
the same canonical pixels. SAM3 masks returned for that canonical PNG are
already in canonical coordinates.

`project_point` computes floating-point pixel coordinates in a top-left,
row-down camera frame. `CanonicalImageAdapter.transform_projected_point`
converts that projection through the raw OpenGL row convention, then applies
the same canonical mapping as the image. V3 camera calibration uses zero
rotation and no geometry-only flip. The previous `rotation_degrees=180` value
was applied to projected pixels only; it never transformed RGB. The older
general LIBERO config still has a 180-degree policy-view value, but this V3
experiment does not call that image-preparation path.

## Resolution audit

Each entry below uses a direct simulator render, four standardized V3 HOLD
settle ticks, and one SAM3 request on the canonical image. No source image was
resized.

| Renderer and SAM input | Target detections | Candidate counts (states 0/1/2) | Mean top score | Top-mask area fraction | Normalized center spread |
|---|---:|---:|---:|---:|---:|
| 512×512 | 3/3 | 3 / 4 / 5 | 0.1497 | 0.00967 | <0.03 px at 512 scale |
| 768×768 | 3/3 | 2 / 4 / 2 | 0.1107 | 0.00966 | <0.05 px at 512 scale |

Visual review of all top masks confirms the salad-dressing bottle. Mask area
fractions and normalized centers are effectively the same across resolutions.
768 returned fewer candidates overall, while 512 returned higher top scores;
there was no mask-stability gain at 768. Runtime V3 therefore remains at
512×512. The Qwen OCR diagnostic used a separate direct 768×768 render.

## SAM3 candidates and identity anchor

The existing SAM3 MCP response includes a ranked `detections` list, with score,
box, area, backend index, and a mask for each candidate. `Sam3Client.segment`
preserves that response. The former V3 parser discarded all but its first
decodable detection. The revised V3 parser retains all candidate metadata and
masks. The first valid, top-ranked `salad dressing` detection establishes one
transient anchor per trial: target phrase, initial binary mask, centroid,
half-open box, area, candidate ID, and frame ID.

After motion, candidates are associated with that anchor using two gates:
mask IoU at least `0.25`, and centroid displacement no greater than the larger
of 12 px or half the initial box diagonal. The log also records box IoU, area
ratio, centroid displacement, candidate ID, rank, and SAM score. SAM score is
not used as the identity criterion. If no candidate passes both gates, the
state is `TARGET_IDENTITY_LOST`; it has no target centroid or after-error and
cannot produce an alignment option or an improvement metric. This anchor lives
only for the current trial and is not persistent visual memory.

## Bounded trial results

The recorded run uses `LIBERO_OBJECT` task 2, seed 0, init states 0–2, four
HOLD ticks per reset, and one Arbiter-approved bounded alignment per trial.
Qwen did not participate in target selection or action decisions.

| Init state | Before centroid | After candidates | Associated candidate | Mask IoU | Centroid shift | Direction | Error before → after (px) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | (193.14, 254.47) | 2 | 0 | 0.377 | 29.74 px | DOWN | 140.19 → 164.36 |
| 1 | (193.12, 254.48) | 2 | 0 | 0.377 | 29.74 px | DOWN | 143.10 → 167.43 |
| 2 | (193.12, 254.47) | 3 | 0 | 0.377 | 29.76 px | DOWN | 130.73 → 155.35 |

Identity was retained in 3/3 trials. Mean predicted improvement was 1.44 px;
mean measured improvement was -24.37 px, so 0/3 same-identity trials improved
the measured error. The projected EEF motion and observed EEF pixel motion had
the same direction in all three trials. The target masks' centroids shifted by
about 30 px despite the visually stationary bottle; this is recorded as SAM3
mask-shape instability and limits confidence in the error-change measurement.
The result does not pass the gate for a multi-step alignment milestone.

## Qwen and authority

On the same 768×768 scene and identical one-word question, Qwen3-VL answered
`MILK` for the raw orientation and `Milk` for the canonical orientation. Both
responses identify the word; this is orientation sanity evidence, not action
selection. Robot actions from Qwen: 0. Ground-truth/oracle diagnostics were not
used. Runtime geometry selected the physical direction, the deterministic
selector exposed the one semantic option, Arbiter approved once, and Executor
ran the bounded motion.

## Artifacts and commands

The orientation and resolution audit artifacts are under
`rollouts/runtime_v3_object_relative_alignment/perception_audit_*`. The
recorded trials, all-candidate masks, anchor visualizations, projections, and
before/after overlays are under
`rollouts/runtime_v3_object_relative_alignment/run_20261002T095805Z_f05c03ac`.
The audit script is `scripts/runtime_v3_perception_audit.py`; the bounded
experiment is `scripts/runtime_v3_object_relative_alignment.py`.

An earlier trial attempt at
`rollouts/runtime_v3_object_relative_alignment/run_20261002T095705Z_bb1ef022`
completed one bounded motion for each reset init state, then hit a result-logging
`UnboundLocalError` before writing measurements. Those three executions are
excluded from the reported metrics; the corrected, recorded experiment above
ran three fresh reset episodes and one motion per episode.
