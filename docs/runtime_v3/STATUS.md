# Runtime V3 Status

## Completed in this migration

- Frozen the complete pre-V3 worktree as commit `02f10732bd6db0d7578f33a3407aa22f925692c1` and tag `legacy-full-harness-0928`.
- Created local branch `runtime-v3` from that baseline.
- Added LIBERO environment and observation adapters, a bounded Qwen selector adapter, a smoke-only safe-lift option generator, and one-action CLI runner.
- Completed one real deterministic LIBERO closed-loop action and logged raw before/after observations and the measured EEF effect.
- Added integration notes with the legacy-interface map and actual control path.
- Added one-step primitive calibration and no-action Qwen selector evaluation tools.
- Calibrated all six configured translation atoms for three independently reset trials each (18 approved real actions total).
- Added adapter, authority, and calibration tests; the V3 test suite passes 29 tests.

## Not implemented

- No valid Qwen semantic selection was obtained. The configured Qwen3-VL-8B-Instruct weights exist locally, but `127.0.0.1:8001` is down. The currently active local endpoint on `127.0.0.1:8002` serves Qwen3-VL-8B-Thinking; all 12 evaluation requests were truncated at the adapter's 64-token limit before a final JSON selection.
- No target perception/identity provider is wired into the V3 task path.
- The single-step translation effects are not reliable enough to call their semantic contracts verified; no new effect threshold or multi-step executor has been chosen.
- No experience learning or legacy RSI migration.
- No full task policy, recovery heuristics, long episodes, training, or benchmark runs.

## Git state

- Current branch: `runtime-v3`.
- Current upstream: `origin/runtime-v3`.
- Legacy tag: `legacy-full-harness-0928`.
- Legacy tag remains unchanged and points to the frozen Full-Harness-0928 commit.
- Deterministic smoke: one action. Artifacts are under `rollouts/runtime_v3_smoke/run_20261002T043157Z_003fda80/`.
- Actuation calibration: `rollouts/runtime_v3_calibration/run_20261002T050110Z_c209f2e7/`; 18/18 actions completed, three resets per token, no workspace skips.
- Qwen no-action evaluation: 12 live model cases and four invalid-output fixtures; live schema validity, option validity, and semantic accuracy were all 0/12 because the active Thinking endpoint returned no final answer before truncation. All four injected raw-action fixtures failed closed. No robot action ran.
- Local tests: `tests/runtime_v3` — 29 passed in both the project venv and the LIBERO venv. The code has not modified any file covered by the frozen legacy tag.

## Next experiment

The single-step LIBERO infrastructure path is validated, but primitive semantics and live Qwen selection are not. Keep the next research milestone at calibration/selector-contract follow-up; do not enter object-relative verified options or start a full manipulation episode yet.
