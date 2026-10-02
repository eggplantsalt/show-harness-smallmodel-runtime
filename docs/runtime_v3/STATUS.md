# Runtime V3 Status

## Completed in this migration

- Frozen the complete pre-V3 worktree as commit `02f10732bd6db0d7578f33a3407aa22f925692c1` and tag `legacy-full-harness-0928`.
- Created local branch `runtime-v3` from that baseline.
- Added LIBERO environment and observation adapters, a bounded Qwen selector adapter, a smoke-only safe-lift option generator, and one-action CLI runner.
- Completed one real deterministic LIBERO closed-loop action and logged raw before/after observations and the measured EEF effect.
- Added integration notes with the legacy-interface map and actual control path.
- Added adapter/authority tests; the V3 test suite passes 18 tests.

## Not implemented

- Qwen offline selector adapter is implemented, but no selection completion was obtained because the configured vLLM endpoint was not running.
- No target perception/identity provider is wired into the V3 task path.
- No experience learning or legacy RSI migration.
- No full task policy, recovery heuristics, long episodes, training, or benchmark runs.

## Git state

- Current branch: `runtime-v3`.
- Current upstream: `origin/runtime-v3`.
- Legacy tag: `legacy-full-harness-0928`.
- Legacy tag remains unchanged and points to the frozen Full-Harness-0928 commit.
- Deterministic smoke: `REAL_SMOKE_PASSED`, one action. Artifacts are under `rollouts/runtime_v3_smoke/run_20261002T043157Z_003fda80/`.
- Qwen offline smoke: `QWEN_ADAPTER_BLOCKED`; local endpoint `127.0.0.1:8001` refused the health check. No action ran.
- Local tests: `tests/runtime_v3` — 18 passed in the project venv and 18 passed in the LIBERO venv. `compileall` and `git diff --check` passed.

## Next experiment

The single-step LIBERO infrastructure path is validated. The next research milestone is object-relative verified options and bounded Qwen selection, beginning offline with a live compact-model endpoint; do not start a full manipulation episode in this phase.
