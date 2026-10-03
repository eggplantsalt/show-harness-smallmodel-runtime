# M3.7 Formal Evaluation Artifacts

The forensic development audit and frozen held-out manifest are stored beside
these reports. Formal post-refactor readiness ran at Runtime commit
`3aae19d29d5840641dc94b0c17432127f3904119`.

## Reports

- `readiness_stage_a_evaluation.json`: compact Stage A counts, frozen protocol,
  oracle grading, and interpretation.
- `readiness_stage_a_summary.json`: all 18 per-episode traces, both readiness
  evidence records, and separate post-hoc oracle diagnostics.
- `stage_b_reference_manifest.json`: the valid-reference task set frozen after
  Stage A and before ALIGN revalidation.
- `align_stage_b_summary.json`: unchanged reference-binding ALIGN contract and
  all per-task/per-episode metrics.
- `qwen_binding.json`: one-call-per-instruction Qwen schema and binding audit.
- `qwen_stage_c_summary.json`: physical transfer using only audited semantic
  TaskSpecs from that Qwen run.

The Stage B and Stage C traces record no target-pose, simulator-depth,
ground-truth-contact, or task-success input to Runtime. Stage A records target
pose only in its separate oracle diagnostic object for false-ready grading.
ALIGN execution used the same source signature across all tasks, and the legacy
tree was not modified.

## Visual review

`visuals/` contains the forensic same-frame phrase comparison, butter and
control-task SAM traces, plus readable Stage A sheets for alphabet soup,
butter, the two frozen held-outs, and the normalized Qwen phrase parity case.
These contact sheets were opened and visually reviewed. Task 0's stable mask
is the neighboring Milk carton, which is why it is excluded from the frozen
Stage B reference manifest despite passing the numerical gates.
