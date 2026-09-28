# Capability-aware harness progress — 2026-09-26

## Current implementation slice

- The clean Agent is the official `Qwen/Qwen3-VL-8B-Instruct` with no Show-Harness
  adapter. Existing 2B/9B and simulation adapters remain historical references.
- MolmoPoint is not part of the first profile because its GPU footprint is high and the
  current failure evidence does not establish a control benefit.
- `core/capabilities/` contains a host-owned visual harness with an optional SAM3 MCP
  bridge, CPU template tracking, and bounded episodic memory.
- Evidence is injected into the controller prompt and logged per step; it does not pick
  or execute Move Tokens.
- LIBERO camera geometry now projects only current EEF proprioception through MuJoCo
  camera calibration. It reports target-vs-EEF pixel error and calibrated token
  candidates, while retaining Agent action authority. C1 explicitly disables geometry;
  C2/C3 enable it.

## First experiment

- Config: `configs/robot_libero_clean_qwen3vl.yaml`
- Task: `LIBERO_OBJECT` task 2, salad dressing → basket
- Model revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Model server: `scripts/serve_clean_qwen3vl.sh` on port 8001
- Capability profile: SAM3 + CPU tracking + episodic memory
- Ablation configs: `_direct` (C0), `_c1` (SAM3), `_c2` (SAM3 + CPU tracking), `_c3`
  (SAM3 + tracking + memory), `_c4_guard` (active commit guard)
- The guard is disabled in C0–C3. Its shadow/active decision is logged separately from
  the model token, so a blocked physical commit cannot be mistaken for a model action.
- In active mode, a GRASP also fails closed when fresh geometry does not place the target
  within the configured final-alignment window; visibility alone is not treated as proof
  that the object is between the fingers.
- Legacy action chunk, variable step, and LIBERO token normalization are disabled.

## Clean local smoke evidence

- C0 direct, state 0, 5 decisions: 0/1 at the short diagnostic budget; 6 clean Qwen
  calls (planner + controller), 6,329 total tokens, about 21.3 seconds model latency.
- C1, state 0, 5 decisions: SAM3 grounded the bottle on fresh frames (three tool calls,
  no failures), but the weak agent still repeated `MV_DOWN`.
- C2, state 0, 5 decisions: CPU tracking supplied frames 1--3 without new SAM3 calls;
  the agent still repeated the same vertical action.
- C3 before geometry, state 0, 5 decisions: episodic memory correctly measured a
  0-pixel stall after repeated `MV_DOWN`, but the model ignored the warning.
- C3 with geometry, state 0, 5 decisions: the model switched to calibrated `MV_RIGHT`
  for a target whose projected EEF error was roughly `[+65,+46]` pixels. A 30-decision
  pilot reduced horizontal error to about 24 pixels; the longer pilot reached about
  5 pixels horizontally before its old 50-step subgoal budget truncated.
- The old LIBERO horizon formula also omitted per-decision settle steps. It has been
  corrected in `scripts/run_libero_zeroshot.py`; this prevents simulator horizon from
  being mistaken for an agent failure in longer pilots.

## Runtime constraints

- Use the Hugging Face mirror specified by `AGENTS.md` for model downloads.
- Preserve `rollouts/` and the existing dirty worktree.
- Do not pass simulator object state or `check_success()` into harness evidence.
- Record model revision, GPU usage, SAM3 calls, failures, and per-step evidence before
  interpreting success-rate changes.

## Hosted Kimi 2.6 pilot

- Added `configs/robot_libero_siliconflow_kimi26_c3.yaml` and its 500-step pilot
  layer `configs/robot_libero_siliconflow_kimi26_c3_long.yaml`. They inherit the
  same LIBERO C3 harness and action contract; only the agent backend changes to
  SiliconFlow's `Pro/moonshotai/Kimi-K2.6` over the OpenAI-compatible vision API.
- A one-image API smoke test succeeded (one call, 3.65 s, 147 prompt and 60
  completion tokens). The API key is resolved from `SILICONFLOW_API_KEY` and is
  not written to YAML or rollout metadata.
- The first Kimi run was invalid as a model result: the separate SAM3 service had
  exited, producing `Broken pipe` on all grounding calls. That short run is kept
  under `rollouts/libero_siliconflow_kimi26_c3_long/run_0e411d41daf2` as an
  infrastructure-failure record.
- After releasing the local Qwen vLLM GPU allocation, SAM3 was restarted in a
  persistent foreground service and independently returned detections. The
  recovered Kimi screening run is under
  `rollouts/libero_siliconflow_kimi26_c3_long/run_0bf128f871cf`; it reached 126
  logged decisions before deliberate interruption, remained in APPROACH, and
  recorded 87 SAM3 calls / 74 abstentions. During the visible/evidence loss,
  Kimi continued to infer target location from RGB instead of treating
  `visible=false` as movement authority. This is evidence for the next C4
  active-evidence-gate experiment, not a success-rate conclusion.

## SAM3 abstain root-cause follow-up

- The first Kimi run's 14/14 `Broken pipe` results were infrastructure failures.
- In the recovered run, the 74 abstentions were mostly backend-empty results caused by
  `confidence_threshold=0.5`: the same target frame returns the correct bbox with
  score `0.42--0.46` when the threshold is `0.4`.
- Added structured abstain telemetry in `core/capabilities/visual_harness.py` and recorded
  the evidence in `docs/sam3_abstain_root_cause_2026-09-26.md`.
- The threshold-0.4 follow-up is under
  `rollouts/libero_siliconflow_kimi26_c3_threshold04/run_17d923a67356`. It kept the
  correct target bbox through the initial approach and reduced horizontal error to near
  zero, but still failed because Kimi never ended APPROACH after geometry became aligned;
  continued motion then caused gripper/arm occlusion and a second, genuine abstain mode.

## Qwen GRASP 预抓取复核

- 用户要求切回 Qwen 后，新增配置
  `configs/robot_libero_clean_qwen3vl_c3_threshold04_approachguard.yaml`，使用本地
  `Qwen/Qwen3-VL-8B-Instruct`，保留 SAM3 threshold `0.4`、active APPROACH guard 和
  Wrist→AgentView 受控回退。
- 在 `run_dfc65557abfc` 中，frame 136 已由 host guard 发出 `DONE` 并进入 GRASP；
  frame 137--147 连续执行 `MV_DOWN`，说明模型已经进入预抓取阶段，而不是没有理解
  GRASP 子目标。
- 复核后修正判断：Wrist 空检不代表物体不可抓。step 147 的 AgentView 已显示绿色瓶子
  位于夹爪两指之间，指尖高度约 `0.15164 m`；Wrist 只是因安装位置和近距离遮挡看不到
  瓶身。原有“Wrist 必须确认”规则错误地把不可观测当成不可抓，已改为 AgentView 可独立
  支持 GRASP，Wrist 仅作辅助。
- 另一轮空抓暴露目标身份风险：planner 的模糊 `salad dressing / body of the bottle`
  让 SAM3 在两个瓶子候选中选错目标。已要求 planner 输出颜色/瓶盖等区分属性，并让
  SAM3 对近分数候选记录 `ambiguous_top_detections` 而不是按返回顺序盲选。
- 修复了 `zeroshot_robolab_runner.py` 的分支优先级：fresh active APPROACH guard
  现在优先于 recovery/chunk，避免在恰好对齐的帧被恢复动作吞掉阶段转换。
- 修复了多图请求的视角标签：`Image A: agentview RGB` 与 `Image B: wrist RGB`
  现在会随请求文本发送，避免模型自行猜测两张图的角色。

## Live-frame audit: AgentView versus Wrist

- The complete Qwen rollout `run_999692f75a40` was allowed to run without a manual
  interrupt. Frames 0087/0095 show the green-capped bottle visibly seated between the
  fingers in AgentView, while the Wrist camera, mounted above/on the gripper, sees mostly
  the occluding gripper and only the lower/top edge of the bottle. Therefore a Wrist SAM3
  abstain is an observation-availability event, not evidence that the grasp is impossible.
- The controller now treats AgentView as sufficient for GRASP when it shows the target
  between the fingers at the calibrated height; Wrist remains supplementary and can be
  occluded at contact.
- The same rollout measured a stable closed width of `0.0371 m` after GRASP. LIBERO's
  empty/open width is about `0.08 m`, but the config had `open_width_m=0.03 m`, so recovery
  incorrectly classified the real hold as an unsettled open close and inserted STOP every
  few steps. The LIBERO profile is now calibrated to `open_width_m=0.06 m`, yielding
  `0.0015 -> empty`, `0.0371 -> holding`, `0.08 -> open`.
- The VLM gripper state is now based on the binary close command rather than comparing
  object-holding width to the empty-close threshold. A verified non-empty GRASP also
  completes the GRASP subgoal directly, so the stage cannot wait indefinitely for a
  redundant model-generated DONE.

## Next physical bottleneck: monotonic LIFT

- In the same live rollout, the verified GRASP advanced to LIFT, but the generic visual
  policy oscillated between `MV_UP` and `MV_DOWN`: the held bottle remains visually below
  the hand, and the policy misread that image relation as a reason to descend. The next
  LIBERO runner revision makes LIFT monotonic: issue `MV_UP` until fingertip height reaches
  the calibrated `0.17 m` clearance band, then complete LIFT. This uses robot geometry only,
  not simulator object state.
