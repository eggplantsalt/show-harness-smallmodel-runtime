# Runtime V3 Status

## Completed in this migration

- Frozen the complete pre-V3 worktree as commit `02f10732bd6db0d7578f33a3407aa22f925692c1` and tag `legacy-full-harness-0928`.
- Created local branch `runtime-v3` from that baseline.
- Added canonical state, observation, option, selector, Arbiter, executor, effect, memory-interface, and short runner modules under `core/runtime_v3/`.
- Added architecture and migration notes.
- Added authority and import-boundary tests plus a mock one-step runner smoke test; the V3 test module passes 9 tests.

## Not implemented

- No V3 perception provider or Qwen adapter is wired to the LIBERO runtime yet.
- No experience learning or legacy RSI migration.
- No full task policy, recovery heuristics, long episodes, training, or benchmark runs.
- The private GitHub repository could not be created because `gh` is not installed. The old `origin` is retained as `upstream-legacy`; no new `origin` exists yet.

## Git state

- Current branch: `runtime-v3`.
- Legacy tag: `legacy-full-harness-0928`.
- New remote: blocked (`GITHUB_REMOTE_BLOCKED`).
- Local smoke/test status: `./.venv/bin/python -m pytest -q tests/runtime_v3/test_runtime_v3.py` — 9 passed. Static compile and whitespace checks are recorded in the migration report.

## Next experiment

Start with one mock one-step control cycle. Then wire one existing LIBERO observation source and atomic controller through V3 adapters and run a single bounded dry-run decision with a frozen compact-model checkpoint. Do not start a multi-episode benchmark until the evidence and action receipt path is visible in logs.
