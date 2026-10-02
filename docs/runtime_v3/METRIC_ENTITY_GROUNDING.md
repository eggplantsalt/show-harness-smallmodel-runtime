# Runtime V3 Metric Entity Grounding

This phase adds an RGB-only deployable depth estimate and a generic visible-surface
reference. It does not add 3D control. The existing alignment objective, option
ranking, arbiter, executor, and stop policy remain on the established 2D and
proprioceptive path.

## Formal perception path

```text
canonical RGB
  → deployable monocular metric depth
  → SAM target mask
  → robust mask-conditioned depth samples
  → calibrated pinhole projection
  → world-frame MetricEntityReference
```

`MoGeMetricDepthProvider.estimate(rgb)` receives one RGB array and returns a
same-resolution metric depth estimate. It has no task ID, object pose, target
body ID, contact, or simulator input. `metric_entity_reference_from_estimate`
filters invalid and outlier mask pixels, back-projects the remaining samples
using camera intrinsics and extrinsics, and takes the coordinate-wise median as
a visible-surface reference. This is not an object center or grasp point.

`MetricEntityReference` is object-agnostic and accepts `monocular_metric` or a
future real `rgbd_sensor` source. It rejects `simulator_gt`. The observer stores
the estimate alongside the existing object-relative belief state. The 2D
alignment objective does not consume it; M3.5 only reports the estimated
EEF-to-reference distance as telemetry.

## Deployable model audit

- Model: MoGe-2 ViT-L (`Ruicheng/moge-2-vitl`)
- Pinned Hugging Face revision: `39c4d5e957afe587e04eec59dc2bcc3be5ecd968`
- Output: metric depth in meters, at input image resolution
- Input: RGB only
- Training: none in this phase
- LIBERO environment: Python 3.10.12, PyTorch 2.14.0+cu130, CUDA available,
  NVIDIA GeForce RTX 4080 SUPER (31.47 GiB); `transformers` is not installed and
  is not needed by the upstream MoGe v2 API; `huggingface_hub` 2.0.0
- Model code: upstream MoGe checkout at commit
  `74fbce054ebed49800de42d0ad0e83495065719a`; checkpoint fetched
  from its [official Hugging Face model page](https://huggingface.co/Ruicheng/moge-2-vitl/tree/39c4d5e957afe587e04eec59dc2bcc3be5ecd968)
- License: the official model page lists MIT; check the upstream repository's
  noted third-party DINOv2 license before redistribution

The configured LIBERO Venv loaded the checkpoint on CUDA and completed a
512×512 inference. The checkpoint file is approximately 1.31 GB. The upstream
[upstream README](https://github.com/microsoft/MoGe/blob/main/README.md)
reports same-resolution metric depth output and lists the ViT-L MoGe-2
checkpoint as metric scale; its published speed figure is hardware-specific and
is not used as this machine's latency claim. A uniform test image produced
implausible large depths, so that smoke verified the interface only. No
simulator depth scale correction is applied.

MoGe-3 is also listed upstream, but the LIBERO environment lacks `flex_gemm`;
the v2 ViT-L path imported, loaded, and inferred successfully without it. The
local smoke measured 7.47 s to load and about 0.95 s for a first 512×512
inference. These are local smoke timings, not a throughput claim.

## Simulator depth audit boundary

MuJoCo depth is enabled only by the M3.5 experiment script. The
`DiagnosticSimulatorDepthProvider` lives under `scripts/` and reads the cached
LIBERO raw observation after the formal Runtime observation has been acquired.
The experiment converts its normalized z-buffer with the robosuite near/far
formula, then compares it with the deployable estimate and saves the comparison.
Simulator depth, target body pose, and contact records never enter the Runtime
observation, belief state, options, selector, arbiter, executor, or termination
policy.

Simulator ground truth is evaluation evidence, not Runtime information.

## Stage gates and interpretation

Stage A observes task 2, seed 0, initial states 0–5. After SceneReady it records
the reference and five additional hold observations. It compares estimated and
simulator-depth mask-region errors, reference offset from a GT-depth-derived
reference, offset to the simulator body origin, valid depth ratio, and
frame-to-frame reference displacement. The stability gate is a maximum
frame-to-frame displacement of 3 mm and a valid reference on all six frames.
Those diagnostic comparisons do not adjust Runtime parameters or candidates.

Stage B runs only if every Stage A state passes the gate. It uses the existing
2D multiscale objective, for at most 12 semantic steps. It records estimated
metric distance from RGB depth plus EEF proprioception and independently records
the simulator EEF-to-target distance. Correlations and agreement of distance
trends are reports only. Wrist frustum status is a diagnostic projection of the
estimated reference and camera calibration; no wrist search behavior is added.

If the Stage A gate fails, the report is a deployable metric perception
contract failure. A simulator-depth-only success cannot justify the next
milestone.

## M3.5 Stage A result (2026-10-02)

Run artifacts: `rollouts/runtime_v3_metric_entity_grounding_rgb_only/run_20261002T155631Z_422a6ba3/`.

- Coverage: task 2, seed 0, init states 0–5; 6 frames per state; 36/36 estimated
  references valid, with 100% mean valid mask-depth ratio.
- Stability gate: 3/6 states passed. Mean per-frame world-reference displacement
  was 1.62 mm; maximum was 4.67 mm (state 4). State maxima in order were 2.48,
  1.66, 2.83, 3.30, 4.67, and 3.61 mm. Median mask depth frame-to-frame change
  averaged 1.58 mm and reached 4.69 mm. Using the same mask and calibration,
  the GT-depth-derived reference had a maximum displacement of only 0.056 mm;
  this points to variation in the monocular depth estimate, rather than visible
  scene/mask movement, as the main source of the observed reference jitter.
- Depth comparison on SAM masks: mean frame-level MAE 0.300 m (median 0.308 m);
  mean frame-level median absolute error 0.309 m (median 0.317 m).
- The median predicted mask range exceeded GT depth by 0.310 m on average
  (median 0.317 m), with no GT scale correction applied.
- Estimated visible-surface reference versus the same-mask GT-depth reference:
  mean Euclidean error 0.308 m (median 0.315 m). The mean world-frame signed
  delta was approximately `[-0.250, -0.029, -0.178] m`.
- Offset from the simulator target body origin averaged 0.296 m. This is not a
  pure model error: a visible-surface point is not expected to equal the hidden
  body-frame origin.
- Stage B was correctly skipped because the six-state stability gate failed.
  It executed zero semantic alignment steps; wrist-frustum coverage was not
  evaluated in this run.
- Runtime actions used no diagnostic depth, target pose, body ID, contact, or
  success state. Qwen actions and grasp/place/release actions were zero.

This result demonstrates what simulator depth can diagnose: the estimated
mask-region range is systematically displaced from the rendered range, and
the estimated camera-to-world reference has both a large bias and frame jitter.
RGB-only perception currently lacks sufficiently accurate and repeatable
metric range on these task images. It also does not expose hidden object shape,
contact, or target motion when the target becomes occluded. No Stage B trend or
wrist-visibility conclusion can be drawn from this gated run.
