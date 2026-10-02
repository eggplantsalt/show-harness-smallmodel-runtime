# 当前实验记录：V2.1 恢复与 Qwen Thinking 演进

更新时间：2026-09-30 UTC

## 当前基线与结论

- 按用户最新决定，本轮活动基线是恢复的 V2.1：`configs/robot_libero_clean_qwen3vl_runtime_v2.yaml` 的抓取空间/语义 harness，加独立 `Qwen3-VL-8B-Thinking` 的实验覆盖配置 `configs/robot_libero_clean_qwen3vl_thinking_runtime_v21.yaml`。V2.2 放置重构保留为历史参考，不作为本轮线上控制基线。
- 旧实验文档曾报告 V2.1 到达 `VERIFY_HOLD`，用户也记得 GPU1 新几何模型后抓取与搬运改善；当前磁盘缺少足以复核该次运行的精确原图、动作日志和 checkpoint revision。因此这是重要的待复现实验线索，不计为已复现成功率，也不能断言仅由 MoGe 导致。
- 新的 V2.2 三次抓取阶段尝试没有进入放置，但它们不能回答恢复的 V2.1+Thinking 是否能抓取。恢复版 V2.1+Thinking 的在线测试现已到达真实 memory-assisted Qwen 决策：输入图板成功发出，但首轮反思未引用当前帧，被运行时安全拒绝；尚未闭爪，因此抓取仍未验证。

### 隔离 V2.1 副本进展（2026-09-30，最新）

- `/root/autodl-tmp/Show-Harness-V21` 是独立的 V2.1 复现轨，使用 task 2/init 0/seed 0、专用 8002 的 `Qwen/Qwen3-VL-8B-Thinking`，V2.2 placement authority 关闭；本节的 rollout 与主工作树的 512 双次反思序列分别记录。
- 两次早期尝试都在 action step 0 前被 planner final 截断。根因现已实证：该 Thinking 权重的本地 chat template 无条件生成 `<think>`，所以 `enable_thinking:false` 无法关掉推理；planner 之前也未传 `thinking_token_budget`。VLMClient 当前对该 checkpoint 强制声明 Thinking、按请求绑定 reasoning 上限并为 JSON final 留出 token；planner 和原子决策显式使用最多 1024 reasoning tokens。新截断异常会带 completion token 与 request hash，便于立即识别预算是否真生效。
- 修复后单轮 baseline `run_47b7ee44b206` 成功完成模型 plan 请求（55.24s，1674 completion tokens，五个 subgoal），随后总计执行 81 个动作，`success=false` / `runtime_v2_option_budget_exceeded`。没有闭爪、持握或放置，不是完整成功。
- 查看 40.5 秒失败视频与原始 AgentView/Wrist 帧，并对照 runtime action receipt 后，最早的循环是第 40 步世界 XY 首次进 8 mm tolerance 开始下降；下移后误差约 9.8 mm，略超同一阈值，runtime 又授权横向校正，随后因 XY/下降小幅交替耗尽 80 步 hover budget。这是 tolerance 与下降阶段细小横向漂移相互触发，不是模型未产出 plan，也没有证据需要更改相机投影。
- 隔离副本已增加有 fresh support-plane + EEF 证据约束的下降阶段 XY hysteresis：在原始 2×世界对齐 tolerance 内继续下降，超出回归到横向校正，失去证据则 STOP。专项 `pytest` 结果 **96 passed**，`py_compile` 通过；下一步仅同初始状态单条复跑并逐帧检查 approach 与抓取节点。`env.check_success()` 成功 episode 仍为 0。

## 2026-09-30：原生 512 输入与 V2.1 架构复现核对

- 同一 init state 的原生 512 run：`run_a9dedad2cace`，配置为 `configs/robot_libero_clean_qwen3vl_thinking_runtime_v21_512.yaml`，记录在 `rollouts/v21_thinking_evolution_0930_native512/run_a9dedad2cace/LIBERO-LIBERO_OBJECT-2/0930/task_0/20-43-44/`。该 profile 将模拟器渲染和公共预处理都提升到 512×512；不是把 256 图简单放大。
- 256 run 中 Qwen 曾把右侧橙色瓶状干扰物选成目标；离线 BDDL/相机投影诊断表明该候选并非任务目标。512 run 的全局 AgentView 中，SAM3 首帧将中心绿色瓶体作为第一候选（bbox `[297,226,335,313]`，confidence `0.9336`），且同帧另一个瓶状候选仍被保留。这支持“原生分辨率帮助本例区分目标”，但目前只有一个任务/初始状态样本，不能据此宣称泛化已验证。
- Qwen 抓取决策的实际 HTTP 图像载荷为 512×1024；保存的 Qwen panel 和 SHA-256 在 event `request_audit` 中可查。用同一 Thinking checkpoint 的本地 `AutoProcessor` 重放该面板，`image_grid_thw=[1,64,32]`，对应原始 1024×512 张量网格；processor 配置允许的最大像素远高于本请求，因此没有把请求缩回 256。关键帧 AgentView 与 Wrist 的原图、SAM3 输入均为 512×512。局部 ROI 是从 512 原图裁取后放大 2.5 倍，只增加目标在面板中的占比，不产生额外像素信息。
- 512 没有恢复抓取。视频显示机械臂始终未到目标附近，夹爪保持 OPEN；没有 GRASP/闭爪、`VERIFY_HELD`、transport 或 `env.check_success()`。step 4 的 Qwen 使用同一 episode 的 `[0,3,4]` 帧 memory，reasoning 字段实际存在，最终返回 `UNKNOWN`；随后最多两次受限 `PROBE_DEPTH` 都被 runtime 编译成 `MV_BACK`，之后 Qwen 再次 UNKNOWN，run 在没有新有效进展时停止。
- 这次最早阻塞是“可见的全局偏差没有成为可选取证动作”：同帧 AgentView 的已关联目标仍在夹爪投影右侧约 63 px、上方约 100 px；Wrist 主视图的 SAM3 却产生了另一个置信度仅 `0.4141` 的候选框。AgentView 次视图仍将正确目标 `target-0001` 关联到当前帧，但空间融合是 UNKNOWN，V2.1 pregrasp 提供给 Qwen 的选项因此只有 `PROBE_DEPTH/UNKNOWN`。Qwen文本注意到未夹住目标，但接口没有 `VISUAL_ALIGN` 选项；两次深度 probe 不能纠正显著的全局视图偏差。这属于跨视角 grounding 与 ReAct 选项缺口，不是高分辨率无效，也不是可以归因于“Qwen不会推理”。
- MoGe v2 在 512 帧上有 64/64 点通过像素重投影（约 `0.71 px`），但 Wrist 目标 mask 对应的 median world Z 为 `-0.13558 m`，低于配置支撑平面 `0.015 m`；AgentView 的 secondary 几何 median Z 为 `-0.05164 m`，同样被拒。故此回放将空间健康状态保持 UNKNOWN 是正确的。像素重投影只证明投影方向一致，不能证明深度尺度/世界位置正确。checkpoint SHA-256 为 `3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5`，配置使用的 MoGe v2 权重与模型类一致。
- 当前服务进程实际以 `--reasoning-parser deepseek_r1` 运行；Qwen 响应确实分出了 reasoning 与最终 JSON，但计划中指定的 `qwen3` parser 尚未在在线 run 使用。下轮服务配置应先对照真实 vLLM 响应验证 parser，不把字段“存在”误作 parser 配置一致。
- Mermaid 中的配置/runner、SAM3、MoGe/视差、V2.1 runtime、Thinking 决策、原子控制、当前 episode logger 等模块在源码中均有对应，512 run 也真实经过其中多个节点。隔离副本早期两次 planner 截断已被定位并修复；其第一条有动作 run 目前停在 approach XY/下降切换。故结论更新为“模型 plan 请求已通、V2.1 approach hysteresis 正在修复、抓取/搬运/放置仍未验收”，仍不能说已复现 Mermaid 任务闭环。

### 下一步的局部计划

1. 保留 512 原生渲染作为实验配置，不直接改其他任务默认值；同一 init state 的后续 run 保持 task/model/控制器一致，只审计身份、两视角 grounding、图像载荷和第一项有界纠正。
2. 在 V2.1 semantic pregrasp 中增加受限 `VISUAL_ALIGN` 取证选项：仅当当前 AgentView 实例身份、帧号和 EEF 标定投影均新鲜有效时开放；runtime 从最大且超过容差的像素残差轴编译一个校准方向的原子运动，动作后必须取得新帧再决定。Qwen 选择“需要视觉对齐/需要新证据”，不能输出任意坐标或绕过 runtime。缺少同帧投影时继续 UNKNOWN/STOP。
3. 不把无跨相机对应的 Wrist SAM 候选冒充同一实例；保留 AgentView 的可靠实例 track，并将 Wrist 不确定身份作为未定位观测。继续校准 MoGe 的 metric world Z 与 support-plane consistency；不做手动坐标符号翻转。
4. 先用回放和单测验证 `VISUAL_ALIGN` 的陈旧/歧义/缺标定拒绝、每次仅一个动作、失败方向不重复；再在同 init state 做单 episode。只在真实闭爪且 `VERIFY_HELD` 通过后统计抓取恢复，之后才验证搬运和最终任务。

## 2026-09-30 第六次 rollout：Qwen 重选实例造成接近方向漂移

- `run_66c443edd81c`（同一 `LIBERO_OBJECT/task 2/init 0/seed 0`）以 `success=false` 结束，82 步停在 `ALIGN_PREGRASP`，未闭爪，因此既不能算一次成功抓取，也不能证明目标在物理上抓不住。
- 最早错误发生在前两帧的实例确认：两只相近瓶形候选被 Qwen 分别选中。第 0 帧选右侧候选；第 1 帧虽然同相机 tracker 已把连续检测关联到原实例、association confidence 约 0.886，但 runtime 的 temporal tie gate 只接受 0.90 以上，于是重新询问 Qwen；Qwen 随后切换到相邻候选。之后 approach 方向随目标切换而变化，并进入小范围重复调整。该证据说明 Thinking 本身不能弥补身份锁接口在已确认实例后再次开放。
- 修复改为：同一实例刚经 Qwen 确认后，尊重 tracker 的短时 motion-gated `decision_lock_active`；它与单纯高置信度无关，可避免因阈值附近的小幅分数波动重新做语义选物。锁只覆盖短时同相机、同实例的有效关联；失配、过期或跨相机仍走原有拒绝路径。
- 新增回归复现“association 有效但低于 0.90”场景，确认 runtime 不再重复 SELECT_INSTANCE；身份专项 3 项通过，V2.1 Thinking/profile、空间、合同、memory、runtime、resolver、runner 集成集合 **84 passed**。`py_compile` 与 `git diff --check` 通过。
- 下一步以同一初始状态重跑一集，逐帧核对目标身份是否稳定、approach 是否到达 GRASP、遮挡节点 Qwen 是否真实收到带 step 56/57 的同 episode 双视角 memory panel，以及 Thinking 的 scene reflection/最终 decision hash。没有真实闭爪和 hold verification 前不声称抓取恢复；没有 `env.check_success()` 不声称任务完成。

## 2026-09-30 第七次 rollout：memory 已送达，双阶段证据契约不一致

- 同一任务的新 run `run_320ffbcde0fb`（`LIBERO_OBJECT/task 2/init 0/seed 0`）共 72 步，最终 `success=false`、`runtime_v2_option_budget_exceeded`。没有闭爪或 `VERIFY_HELD`；本次证明的是身份锁与记忆请求链，不是抓取成功。
- 新 identity lock 在线有效：step 0 Qwen 选择 `candidate-0`，生成 `target-0001`；step 1 的同实例、同相机 tracker association confidence 为 0.88598，低于旧 0.90 阈值，但 runtime 采用有效短时 `decision_lock_active`，此后到 GRASP 入口始终保持该实例。世界残差单调从远距趋近，step 55 `DONE`，没有再次向 Qwen 重选相邻瓶形目标。
- step 56 的 PREGRASP 双次反思实际收到同一 episode raw memory panel，帧号 `[54,55,56]`，输入图像 hash `7691bc0f…`，panel 为 512×768。条目还包含帧 54/55 的真实动作回执、约 5 mm EEF 下移和 `MOTION_OBSERVED`。Qwen Thinking 首轮延迟 26.94 秒，1212 prompt tokens、816 completion tokens，确实返回结构化 scene description。
- 首轮只引用 frames 54/55，没有引用冻结当前 frame 56。role 的 JSON schema 只要求引用一个允许帧，事后 runtime 验证却要求覆盖当前帧，造成“schema 合法、运行时拒绝”；第二次判断因此没有发送，结果为 `UNKNOWN`，runtime 正确保持 STOP。后续同一 occlusion 没有新视角/新证据，最多重复 STOP 至第 71 步并耗尽 15 步 option budget。这是接口与恢复策略的可复现缺陷，不是证据表明 Qwen 没有 Thinking。
- 已统一契约：双次反思首轮 schema 新增必填 `current_observation`，其 `frame_id` 只能是冻结当前帧；明确要求当前画面无法确认时也引用当前帧并写 UNKNOWN。输入输出验证仍检查历史 evidence 的帧号、相机和内容；缺当前引用、错帧或坏类型安全返回 UNKNOWN，不会崩溃。相关回归现为 **15 passed**，下一步需跑完整专项并在线确认两次调用同图板 hash。
- 视频及原始帧检查：AgentView frame 55/56 的目标瓶仍在桌面、机器人已对齐；Wrist frame 56 被夹爪结构遮住目标。Qwen 说物体仍在桌面、未被夹住与观测相符，但它漏掉当前 frame 56 引用；没有据此强行闭爪是正确的安全结果。

## 本轮已实现 / 已验证

- 检查合并配置确认模型为 `Qwen/Qwen3-VL-8B-Thinking`、V2.2 placement 开关关闭、V2.1 语义预抓取及 MoGe v2 在 `cuda:1`，对象参考高度为 `null`；MoGe checkpoint 文件存在。注册表现在使用 provider 归一后的 `moge2` 源名。
- 修复 `infer_spatial()` 中 camera calibration 在 spatial provider 调用后才赋值的问题；MoGe 相机投影往返检查不再只限于 V2.2。新增有效投影正例和错误投影拒绝例。
- V2.1 Thinking 的抓取关键决策现在要求结构化最终 JSON：当前状态假设、支持/反对证据、缺失观察、预期效果、失败条件和简短摘要。图像证据必须来自当前同 episode 帧并经 roles/runtime 两侧复核；合法判断写回当前帧 memory，明确标为未核验模型假设。
- `SPATIAL UNKNOWN` 不再强制选择 `PROBE_DEPTH`；Qwen 可结合视觉和当前 episode 动作效果选择一次有界 probe，或对证据不足 abstain。
- LIBERO planner 移除了绿瓶盖沙拉酱瓶特例、硬编码的 approach→grasp 顺序、固定视角分工和“按原方向重复”的旧策略。相机标定保留为 embodiment 信息；动作仍受 runtime 的可行 option 与效果检查约束。
- 本轮最近专项测试：**43 passed**（Thinking 结构化决策/双次图板、V2.1 memory、MoGe 几何、V2.1 profile 合并及上下文回归）；相关文件 `py_compile` 和 `git diff --check` 通过。Qwen vLLM endpoint health check 成功；GPU0 使用约 23.4/32.8 GB，GPU1 约 5.4/32.8 GB。

## 2026-09-30 V2.1 第三次 rollout 与 memory 修复

- `run_e809899d7cee`（`LIBERO_OBJECT/task 2/init 0/seed 0`）共 72 个决策，`success=false`。它先完成长程 approach 并进入 GRASP，但全程没有发出 GRASP/闭爪，也没有 `VERIFY_HELD`。因此目前能确认的是 approach 工作过、抓取入口未完成；不能说这次“抓住后掉了”，也不能说已经稳定抓取。
- 第 57–65 步，Wrist 的 SAM3 目标连续不可见，AgentView 仍可看到场景与接近中的夹爪。runtime 将该情况标为 `OCCLUDED`，但 `DESCEND_TO_GRASP` 路径仍用上次几何继续下探，EEF 从约 0.135 m 到达配置安全下限约 0.105 m；其后两次 relocalize 仍不可见，恢复预算结束。失败最早发生在第 57 步的决策接口：未要求 Qwen 解释双视角/时序证据，也未让它决定是否清障观察，模型没有机会在这个节点发挥 Thinking。
- 从原始帧 56–65 的双视角联系表可以看到，AgentView 中夹爪进入目标区域，Wrist 中夹爪主体遮住了待抓对象；这更符合“入口视角遮挡与旧下降分支”的证据，而不是已证实的空抓。原图目录虽保存了 PNG，但该 run 的 72 个 VCR event 中 `visual_memory_refs` 全为空；runner 在 V2.1 下于 `observe_frame()` 后才落 raw 图，memory 因而只留下动作摘要，没有可解析的历史图像。该 run 没有在线验证过 visual memory。
- 已修复 runner 时序：有 VCR memory 的 profile 在 runtime 记忆写入前持久化 raw AgentView/Wrist，再把 episode-local raw path 传给 runtime；无图像时不伪造引用。新增回归把两帧实际保存后写进 memory，并通过相同 episode、实例、grasp epoch 和帧序校验重新生成可读图板。
- 已修正 V2.1 遮挡决策：同一实例/grasp epoch 第一次出现“Wrist 遮挡、同相机目标身份仍新鲜、EEF 在配置安全带”的事件时冻结观测并调用 Thinking，最多双次、同一图板；可选项只有一次有界 `MV_UP` 或 `UNKNOWN`。同事件再次遮挡时不重问，不得用陈旧视觉残差继续移动。若身份过期、视图不完整或高度条件不符，则 fail-closed。
- 单测与接口回归合计 **82 passed**；`py_compile`、`git diff --check` 通过。本机 `/v1/models` 返回 `Qwen/Qwen3-VL-8B-Thinking`，SAM3 SSE preflight 可连通。此前模型确实用于候选解析，但第 57 步没有进入 semantic pregrasp role；不能把前三次作为 Thinking memory/reflection 的在线比较。
- 第四次运行 `run_b2650a17be4e` 验证了 raw 引用写入：step 57 请求带帧 `[1,57]` 及四张 raw view 引用。但 Qwen 未实际收到图板；构图器按要求拒绝，因为 V2.1 runtime.reset() 仍生成随机 episode UUID，而 builder 要求 `episode_id == logger.run_dir.resolve()`。该 episode 在遮挡后连续 STOP，直到 `DESCEND_TO_GRASP` 预算终止；没有执行进一步移动/闭爪。因人工中断退出清理且没有 summary.json，标记为 memory namespace 接口失败，不计任务成功或有效抓取尝试。
- run 的日志进一步暴露 memory 检索顺序问题：候选 bundle 在存在较早阶段关键帧时没有保留最新动作前帧，step 57 取到 `[1,57]`，没有 step 56。已将三帧窗口改为“最新事件关键帧 + 当前动作前帧 + 当前冻结帧”，两帧窗口优先使用动作前帧；仍按帧号递增组图。回归用短期队列溢出情形验证 frame 8 不会被 frame 1 挤掉。
- 已修复 episode reset：runner 对任何带 `visual_memory` 的 VCR runtime 都传入当前唯一 logger 路径作为 episode id；legacy runtime 仍调用无参 reset。新增回归断言 memory UUID 必须和图像引用所在的 episode logger 目录完全一致。最新专项集合仍为 **82 passed**。
- 下一步是相同 init 单 episode 重跑，检查 step 57 的 memory bundle 明确包含动作前 step 56 和当前 step 57，第三帧如有则为最近事件关键帧；同时核对同 episode refs、Qwen 请求 image hash、两阶段同板证据，以及 `UNKNOWN` 不移动、`MV_UP` 最多一次、新观察后才允许后续接近/抓取。验证真实 GRASP、持握和搬运之前，不开始多次成功率统计。

### 第五次 V2.1 + Thinking rollout：runner→role 接口异常

- `run_c7558b7f1c22` 再次完成 56 步 approach，step 57 到达遮挡反思节点。错误发生在实际发请求之前：V2.1 有 memory panel 时，runner 仍读取了只在 V2.2 ROI 路径中赋值的 `crop_meta`，触发 `UnboundLocalError`。summary 明确 `success=false`、`end_reason=error`；step 57 raw 双视角已经保存，异常前没有执行闭爪、抬升或下降。
- 这证明 episode namespace 修复已让 panel builder 越过同 episode 校验；但 Qwen 尚未收到这次图板，双次 Thinking 也尚未在线验证。`steps.jsonl` 到 frame 56 截止，保留 summary 的接口错误记录。
- 已将 `crop_meta` 在共享 resolver 路径初始化为空，并新增 runner 级回归：V2.1 memory bundle 可生成 512×512 panel、传给 pregrasp role、保留 double 模式和帧 audit，而不依赖 V2.2 ROI。相关专项目前 **83 passed**；`py_compile` 和 `git diff --check` 通过。
- 下一次继续同 init 单 episode，必须实际检查 Qwen 的两阶段请求审计：bundle 要有 frame 56 和 57，panel image hashes 相同，episode id 与 logger 路径一致；第一阶段 scene description 引用可用帧，第二阶段通过结构化 schema 后 runtime 才能执行 `MV_UP`/`UNKNOWN`。若 UNKNOWN 则继续安全停止，不强行抓取。

## 仍未验证

- 已在 GPU1 用本地 MoGe v2 checkpoint、历史 rollout 保存的 SAM3 mask 和 wrist calibration 完成一帧离线推理。首次调用发现 direct provider 从 mask 点构造 `xs/ys` 未定义，导致 `SENSOR_FAULT`；修复后模型返回 65 个点，65/65 可重投影，中值像素残差约 0.71 px，runtime 输出 `VALID`。相对坐标约 `[0.201, -0.007, -0.302] m` 只是该冻结帧上的几何估计，重投影只验证坐标变换一致性，不证明 metric accuracy；缺少同帧 active-parallax 对照，不能直接拿它授权真实抓取。
- 同一旧 episode 的 Thinking 双次反思离线回放已完成，产物为 `rollouts/offline_v21_thinking_replay_0930.json`。它只用于隔离接口诊断，不是在线 visual memory；记录明确标注 `offline_only=true`、`online_memory_source=false`。在同一冻结图板上，第一次调用生成带帧号的结构化场景描述，第二次结合该描述和空间 belief 输出完整 `UNKNOWN` 决策，没有动作授权。两次使用同一图像 payload/hash；分别耗时 29.458s / 48.752s、输出 876 / 1405 completion tokens。结果证明 Thinking、reasoning/final 分离、schema 与双次同板接口能运行，不证明双次比一次更准确，也不证明在线任务有效。
- GPU1 CoTracker3 使用 profile 中的真实 checkpoint 处理连续 Wrist 帧 56→71，返回 `VALID`，推理耗时约 0.712s；MoGe v2 隔离 worker 使用实际 mask 和标定处理历史 Wrist 帧，64/64 点通过重投影，中值残差约 0.71 px。此前 direct provider 暴露的 `xs/ys` 未定义问题已修复。两项只证明模型路径、输入和坐标接口可运行，不证明跟踪改善了接触判断或 MoGe 达到物理精度。
- 首次在线 run `run_d75d45f284b9`：step 0–19 均为 `STOP`，未执行机械臂动作。SAM3 在两只相近瓶形物体之间反复报 `ambiguous_top_detections`（最高分约 0.957/0.949）；Qwen 虽然返回候选，但其索引顺序随检测分数波动，runtime 每帧都重新调用 resolver。代码根因是 `base_health=AMBIGUOUS` 时忽略了 tracker 同帧的有效关联，只有 detector 原始输出是 VALID 才采用 tracker 健康状态；Qwen 选定的同 episode instance lock 没有解除 detector tie。旧 resolver 单次仅 160 max tokens / 96 thinking tokens，也不足以支持认真比较。run 在 20 个控制步后停止，日志、双视角原图和视频均保留；`success=false`，不计为抓取或任务尝试结果。
- V2.1 针对身份缺口增加受限的通用修复：每帧候选按空间位置编号而非 score 顺序；full-view 候选框与放大 crop 用同色和 ID 对应；Thinking identity decision 提供 1024 reasoning tokens；只有同相机、原 instance ID、当前帧更新且关联置信度 ≥0.90 的 tracker 结果才可解除 detector tie。跟踪自身有歧义、跨相机或过期仍停住。
- 第二次 run `run_25907141366a` 验证身份闭环后开始真实接近，但 `step 2–39` 走 `MV_RIGHT` 后因 `ALIGN_PREGRASP` 的 40 步预算结束；不是方向反转或无效动作：世界 XY 残差从约 `[-0.029,-0.258] m` 降到 `[-0.020,-0.076] m`，每个决策约有 5 mm 有效进展，controller 持续报告 `IMPROVING`。相机帧中机械臂确实横向靠近目标，但仍有约 7 cm metric 残差。根因是 option 在 temporal target track 恢复前按空的 SAM bbox 选成短 `ALIGN_PREGRASP`；有效跟踪和 support-plane 投影随后算出较大 metric residual，却没有更新 option/budget。
- V2.1 Thinking profile 新增 `metric_approach_uses_hover_budget`：有新鲜 target track 的 APPROACH 依据相机标定 support-plane 的 metric 残差选择已有 `MOVE_TO_HOVER`（80-step bounded budget）或近距离 `ALIGN_PREGRASP`（40-step budget），阈值沿用配置的通用 world alignment tolerance，不改步长也不按任务坐标判断。回归含远距离转 hover budget 的放行例和靠近后转精调预算的正例。第三次同 init run 正在按此修复验证。
- 最近单测：identity tie 正负例、空间稳定 candidate id、有色候选板及 metric approach option 的 5 项专项全部通过；更广 profile/spatial/memory 集合此前为 78 passed（命令重复传入 resolver 测试文件，精确去重总数待复跑）。Qwen endpoint 返回 Thinking checkpoint 且 HTTP 200。真实抓取、持握/搬运和完整成功仍未确认。

---

## V2.2 历史实验记录（非当前活动基线）

更新时间：2026-09-30 UTC（grasp 决策选项修复通过离线门槛；准备第三次单 episode 验证）

## 2026-09-30 后续增量：Thinking、抓取证据与观察图板

本节为当前最新状态，覆盖本文较早的“第三次单 episode 待运行”等记录。没有新增成功 rollout。

### 结论先行

- 最近可恢复的三次在线尝试都停在抓取/接近阶段，未进入放置；完整 episode 成功仍为 **0**。第三次 run 的 runtime 在 step 57 把 Qwen 的 `GRASP` 记成 `held=FALSE`。这能确认抓取流程被 runtime 否决，不能仅凭现存日志判断物理上是否曾短暂夹住物体。
- 这不是“Instruct 完全不会推理”的证据。已确认旧调用多为短 token/分类协议，可能限制分析深度；现在单独加入 Thinking checkpoint，并通过最终答案与 reasoning 分离接口使用它。
- 现阶段有代码和离线回归证据，没有 Thinking 服务的真实输出、真实新抓取验证或完整成功任务。不得把 114 个测试通过写作物理任务成功。

### 按观察到的最早问题做的改动

1. `run_1cfaa52e0a63` step 57：Qwen 在 UNKNOWN 几何和约 16.49 px 垂直残差下选择 GRASP，画面中的腕部目标太小且指间关系不清；旧 runtime 又只按固定机械空载夹爪宽度写 `held=FALSE`。V2.2 不再让绝对夹爪宽度单独否决持握；请求当前 detector bbox 的比例 ROI（仅帧/实例/相机都新鲜时），之后要求一次有界抬升和真实 EEF/目标相对运动及稳定帧核验，模型 YES 本身仍不算持握。
2. 两个更早 run `run_79a617a272f4`、`run_129e12d313bf` 都在 probe/遮挡阶段停止。这些帧不会跨 episode 进入 memory；它们只作为离线诊断，不会伪装成本 episode 历史。
3. Thinking 的 critical-role 调用显式启用 `enable_thinking`、温度 1.0 与最多 4096 token。VLMClient 只读取服务最终 `content`；空最终答案、reasoning-only 及 `finish_reason=length` 等截断响应拒绝动作。Instruct profile 仍保留作后续同 harness 对照。
4. 修正 MuJoCo 相机像素 y 轴坐标契约和旋转相机下的 rim 世界竖直方向采样；检查单来源 geometry 时保留正的不确定性。相关测试覆盖旋转/翻转投影往返、来源不确定性与 safety gate。
5. CoTracker3 的窗口先前只在 GRASP 调用，许多抓取窗口没有凑到首个 16 帧输出；现将同一 held-mask 的 AgentView 流延续到当前 episode 的 TRANSPORT，并把点位移作为观测效果摘要写入 memory，不授予动作权。离线 16 帧接口 smoke 输出 51 个可见轨迹，但片段几乎静止，所以不构成跟踪收益证据。
6. V2.2 的通用 RoboLab 上下文还残留了“any-fruit 选择橙色”的旧类别提示。现在仅对 V2.2 去除此规则，保留相机可见性和接近阶段的通用说明；非 V2.2 profile 的上下文不变。
7. LIBERO V2.2 planner context 中的“绿色瓶盖沙拉酱瓶”和“目标偏一侧就重复同方向”也已移除；V2.2 使用相机标定和测得动作效果指导取证/重选项。其他 profile 原样保留。
8. 增加 `--grasp-verification-only` 评测开关；只有当前 episode 的 V2.2 runtime 已写入真实 `verified_grasp` 才结束抓取专项。这个早停结果不计环境任务成功，普通配置默认关闭。

### 本轮验证与待完成项

- `tests/runtime_v2`、VisualRoute、grasp fallback、camera geometry、VLM request audit、logger resilience、V2.2 planner/对象类别上下文和抓取专项早停回归：**120 passed in 3.74s**；`py_compile`、`git diff --check` 通过。
- Thinking checkpoint revision `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b` 正由 HF mirror 下载至 `/root/autodl-tmp/huggingface/hub/Qwen/Qwen3-VL-8B-Thinking`。vLLM 0.24 的 dry-run 参数包含 16K context、单序列、`qwen3` reasoning parser；权重下载完成前不报告模型已部署，dry-run 也不等于能处理完整多图输出。
- 下一步按 plan05 放行顺序：检查模型完整 shard 和配置 → 单次最大图像服务测试，确认 `reasoning_content` 与 `content` 分离、无截断、显存/延迟 → 持握门槛正负例 → 同 init 三次抓取专项（含另一物体）→ 视频/原帧人工核查 → 再做一个完整 episode。任一层不通过就停在该层，不开始 5 次和 27 次成功率统计。

> 本文记录真实代码、测试和 rollout 证据。单元测试通过不等于 LIBERO 完整任务成功；当前没有任何一次 V2.2 episode 通过 `env.check_success()`。

> **Visual memory 范围澄清（按用户最新定义）：**仅指当前 episode 中当前帧之前的观测与动作效果；绝不从以前的 episode 读取图像、判断或实例记忆。每个 episode 用新的 `episode_id` 重置 memory/tracker；请求组图时再次拒绝 episode id 不一致的条目。旧 episode 只可作为隔离的离线评测材料，不能进入在线 memory 或模型输入。

## 当前结论

V2.2 已经从“运输终点算错/三套控制器争权”推进到一个更窄的放置瓶颈：同一 init state 已经能够抓取、保持目标身份、沿 rim-plane 路线运输，并实际进入 `ABOVE_ALIGNED → MV_DOWN → fresh belief`。但在最终 seating 阶段，系统仍把接触边界解释成 `RIM_CONTACT`，执行一次 `MV_UP` 后再次自动下降，未形成 `SEATED_HELD → VERIFY_SEATED → RELEASE`，episode 最终因 transport 子目标 step cap 失败。

当前尚无成功 episode。已修通关键节点的原始双视角记忆、收紧 contact/release 门槛，并完成三组冻结帧的 Qwen 离线回放；最近两次正确环境中的在线尝试都停在抓取阶段，没有进入 placement/release；这些结果不等于在线闭环成功。

### 2026-09-30 单 episode 诊断与修复

- 使用正确的 LIBERO/RoboLab 仿真环境启动 `run_79a617a272f4`；另一次错误 venv 尝试在仿真创建前失败，没有环境动作，不计作 rollout。
- 当前 run 在 step 56 进入抓取观察；step 57–60 Qwen 连续请求语义选项 `PROBE_DEPTH`。runtime 虽已在 step 59 从允许动作里移除了实测反向的原语 `MV_BACK`，语义编译器仍把 `PROBE_DEPTH` 映射回 `MV_BACK`，所以原有 primitive watchdog 被绕过。step 67 因抓取子目标预算结束，整次 `success=false`，没有进入放置/释放阶段。
- 检查了 34 秒 `rollout_failure.mp4` 联系表及关键双视角帧；视频确认机械臂已到物体附近，失败集中在腕部观察含糊后反复沿同一错误方向探测。这里的关键点是 runtime 约束未作用于语义选项，而不能据此归因为模型“没有智能”。
- 第二次 run `run_129e12d313bf` 同样在 step 67 因 `DESCEND_TO_GRASP` 的 15 步预算结束（68 步，`success=false`）。step 57 的反向探测确实改善了视觉残差，但 step 59 仍回到初始方向；根因是 controller 的 wrong-direction 连续计数阈值为 2，单次反证没有跨过一次有效的反向观测持续保留。
- 第二次 run 的 `allowed_answers` 在 step 57–60 只有 `PROBE_DEPTH` 与 `UNKNOWN`：空间融合为 UNKNOWN，prompt 还明确让模型在 UNKNOWN 时 probe。因而 Qwen 没有机会以可见指间关系选择 `GRASP`。这是本轮更早的 harness 决策接口缺口。
- 已修正：一次有显著残差恶化的探测会锁定到当前已观测的反向；继续观测到反向改善时不再切回初始方向，反向本身不改善则 STOP 并要求刷新证据/重规划。V2.2 在新鲜 Wrist 身份、同帧机器人位姿、配置的 EEF 接近安全带均成立时，即使空间融合 UNKNOWN 也向 Qwen 开放语义 `GRASP`；执行后仍由独立 hold verifier 确认。V2.2 prompt 不再把 UNKNOWN 自动等同于必须 probe，其他 profile 未启用此回退。
- 当前单 episode memory 输入也已核实：PREGRASP 请求的持久 `visual_memory_refs` 为空，但 runner 实际把**当前帧和同一 episode 的紧邻前帧**拼成 `BEFORE/NOW` 双视角图板发给 Qwen；没有跨 episode 图像。放置阶段的 memory bundle 则由带当前 `episode_id` 的多个关键节点构建，引用不匹配会 fail-closed。
- 离线门槛最新结果：专项集合 `97 passed`，`py_compile` 与 `git diff --check` 通过。第三次单 episode 尚未重跑，结果不计入成功统计。

### 本轮新增的离线结果

- Runtime 的 `VisualMemoryEntry` 现在引用 episode logger 预先保存的 raw AgentView/Wrist，并记录真实 action receipt、EEF 位移、route phase 与 placement summary。最多两个历史关键节点加当前帧组成有序图板；实例、grasp epoch、帧序或原图引用不符时请求返回 `UNKNOWN`。
- V2.2 提供 `placement_reflection_mode: off/single/double`。`double` 先只看原始时序图和真实动作效果生成结构化场景描述，再以**同一冻结图板**、描述和几何证据判断关系；默认线上 profile 目前设为 `single`。实际图板 hash、帧号、引用、两次响应与耗时写入 critical response；`qwen_input` 不再误存 provider overlay。
- 冻结帧回放见 `placement_reflection_replay_0930.json`：`run_6e9a...` step 170、`run_de319...` step 138、`run_32852...` step 178。近似旧版带预填 `relation` 的输入在接触疑点帧至少一次复现了错误 `RIM_CONTACT`；去掉预填关系后，`off/single/double` 在本轮均输出 `ABOVE_UNALIGNED`。双次调用没有改善三组关系答案，且额外增加一次调用与约 659–879 token 的第一阶段消耗，故当前不启用。重放的旧版判断有一次波动，不能用三帧推断成功率或证明记忆本身有效。
- `contact_candidate` 现仅为复核疑点，不再被 VisualRoute 直接编译成 `RIM_CONTACT`；未确认下降 stall 时，即使 Qwen 回答 contact，runtime 也不会因此抬升。V2.2 的 Qwen `SEATED_HELD` 需通过新鲜 held、grasp epoch、包含裕量、rim 深度与实际下降支撑门槛，才能授权 release。其他 profile 的接口行为保持原状。
- 专项测试实际运行：`77 passed`（`tests/runtime_v2`、VisualRoute、grasp fallback、episode logger resilience）；另有 `py_compile` 与 `git diff --check` 通过。

## 已实现并通过离线测试的 V2.2 部分

- `core/runtime_v2/` 已建立统一 `PlacementRelation`、`PlacementCandidate`、`PlacementBelief`、`SemanticPlacementAction` 和 Placement Spatial Harness。
- V2.2 下 VCR runtime 是唯一动作权；VisualRoute 只提供 opening/rim-plane 路线、mask、footprint、free-space polygon、候选和可视化证据。legacy reviewer、legacy verifier recovery action 和 VisualRoute Qwen intent 不再直接覆盖动作。
- opening mask 不再固定投影到 table/support plane；路线使用估计 rim plane，并用 grasp epoch 锁定的 `object_to_gripper_xyz` 修正 EEF placement target。
- held object 的 SAM3 mask 会跨 tracker-only 帧保留并随实例运动平移；没有足够点云时使用 mask lower contour 构造保守 footprint，不再用单个 active-parallax 点伪造 containment。
- receptacle opening 支持与外轮廓共运动：新 opening mask 必须与当前 outer silhouette 几何一致；检测被遮挡时可由 outer bbox 观测位移平移上一份 mask；不一致时 route 进入 stale/UNKNOWN，禁止复用旧 signed residual。
- clipped 但非退化的 opening mask 可以继续作为 rim 几何证据；这样不会因为 receptacle 离开图像边界而无条件丢失有效 contour。
- stale/UNKNOWN placement evidence 最多允许一次已验证持有状态下的高净空可逆 probe，随后仍 fail-closed；不会在不变的过期 evidence 上持续运动。
- raw RGB、provider overlay、Qwen input panel、UI composite 和事件流分别保存，便于核对视觉输入是否被 overlay 污染。
- AnyPlace 保持 GPU1 独立 subprocess、shadow-only；姿态信息保留，当前不支持的 orientation 不会被误标记为可执行候选。

本轮专项测试结果：

```text
PYTHONPATH=. /root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python \
  -m pytest -q tests/runtime_v2 tests/capabilities/test_visual_route.py \
  tests/capabilities/test_grasp_view_fallback.py
69 passed
```

另有 `py_compile` 和 `git diff --check` 通过。这里的 69 项是本轮实际执行的专项集合，不把它写成完整 pytest 结果。

## 最新 rollout 记录

### 1. `run_6e9a0808aeb8`：opening target 漂移，接触升降循环

路径：

`rollouts/libero_clean_qwen3vl_verified_capability_runtime_v22/run_6e9a0808aeb8/LIBERO-LIBERO_OBJECT-2/0930/task_0/00-52-57/`

- 运输能够到达 `PRE_DESCENT`，但 opening bbox 长时间为 `[30,104,72,125]`，明显向篮子外侧延伸；route 目标随错误 opening 几何稳定漂移。
- step 170 的 `rim_clearance` 约 `-0.00696m`，Qwen 返回 `RIM_CONTACT`，随后 runtime 反复执行 `MV_UP/MV_DOWN`，没有 release。
- 关键问题不是 Qwen 单独“不会看图”，而是它拿到的几何上下文已经把错误 opening 当成可信目标；视觉 verifier 只能在这个错误目标上做语义分类。

### 2. `run_de31988be086`：opening 共运动有效，但 clipped mask 被 route validation 错误拒绝

路径：

`rollouts/libero_clean_qwen3vl_verified_capability_runtime_v22/run_de31988be086/LIBERO-LIBERO_OBJECT-2/0930/task_0/01-13-01/`

- opening 与 outer bbox 的共运动修正开始生效，路线目标会随容器在 AgentView 中的平移更新。
- step 139 新 opening mask 的 bbox 触碰图像边界。VisualHarness 已认为它可能是合法的 clipped mask，但 VisualRoute 仍要求 opening bbox 完全离开边界，导致 refresh 被拒绝。
- 从 step 139 到 186，旧 route 被标记 stale，runtime 按 fail-closed 规则持续 `STOP`，transport 子目标在 120 步后结束。
- 这次失败说明“安全停机”本身还需要一个可恢复的 evidence refresh；否则 provider 短时遮挡会变成无意义的原地耗预算。

### 3. `run_32852afad823`：最新，已经越过 stale/STOP 瓶颈，但 seating 仍失败

路径：

`rollouts/libero_clean_qwen3vl_verified_capability_runtime_v22/run_32852afad823/LIBERO-LIBERO_OBJECT-2/0930/task_0/01-24-22/`

结果：

```text
success=False
steps=187
end_reason=subgoal_step_cap_exceeded
```

可验证现象：

- step 66 执行 `GRASP`，抓取后 holding arbiter 和 hold verifier 正常；没有出现未验证 release。
- step 166 进入 `PRE_DESCENT`，随后连续出现 `ABOVE_ALIGNED → MV_DOWN`；这是 plan03 要求的关键路径，已在真实 rollout 观察到。
- step 178：`rim_clearance=-0.00779m`、`containment_margin=+0.04073m`，Qwen 返回 `RIM_CONTACT`，runtime 执行 `MV_UP`。
- step 179 以后 route 又把关系判为 `ABOVE_ALIGNED` 并重新 `MV_DOWN`；step 181、184 再次触发同样接触/恢复模式，直到 step 186，仍未到 `SEATED_HELD`。
- step 178 的 held bbox 与 outer receptacle bbox 只有约 1 像素的边界重叠，且该下降动作还没有通过动作效果模型确认 stall；因此当前 `contact_candidate` 很可能被极小的二维 overlap 提前触发。这个 contact gate 是下一步需要修正的代码问题，尚未实现。

## Visual Memory 历史问题与本轮修复

当前实现的记忆检索严格限制在同一 episode：历史关键帧的 episode id 必须等于当前 logger run id，runtime reset 会清空记忆和 tracker stream。这里的“历史”是本轮正在运行的 episode 内先前帧，不是历史 rollout；三组旧冻结帧回放只用于离线诊断，原始图像缺失时不会伪称完成图像回放，也不会把那些内容送进在线 agent。

以下是**修复前 rollout** 的调用链，不能作为本轮代码效果：

1. runtime 在每帧动作前后把 `VisualMemoryEntry` 保存到 `instance_id × grasp_epoch` 的 bounded memory 中，内容主要是 frame、requested/authorized/executed action、motion/depth summary 和 tags。
2. `PREGRASP_DECISION` 的 critical request 会携带最近少量 `visual_memory_refs`；runner 将它传给 `resolve_pregrasp`，roles.py 只把最近两条压缩成结构化 JSON，并同时发送当前/前一时刻的 temporal image panel。因此抓取前 Qwen 确实能看到一小段历史，但不是完整历史图像序列。
3. V2.2 transport 的普通动作主要由 runtime geometry 直接编译，Qwen 不参与每一步 signed motion。
4. `VERIFY_SEATED` 当前只把当前双视角和当前 `placement_evidence` 传给 `verify_place`；critical request 没有携带 `visual_memory_refs`，placement verifier 也没有历史场景摘要或前后帧反思字段。
5. 更进一步的代码和日志核查发现：这些旧条目的 image ref 实际未填，最新 run 的 pregrasp 和 placement `visual_memory_refs` 都是空数组。pregrasp 所见的前后帧来自临时图板，不是被引用的持久记忆。旧 prompt 还直接给出了 `relation=RIM_CONTACT`，会锚定语义判断。本轮已修复载荷并去掉 V2.2 verifier 的预填关系，真实线上收益仍待新 rollout 验证。

这不是模型能力已经被证明不足，而是当前 memory-to-agent 接口尚未覆盖放置关键节点。

## 当前能力判定

| 能力 | 状态 | 证据 |
|---|---|---|
| 目标身份与 grasp epoch | 有效，待多次统计 | 最新 run 保持 held object track，未见未验证实例切换 |
| 抓取 | 单次有效 | step 66 `GRASP` 后 `held=True` |
| opening/rim-plane 几何 | 已明显改善，仍需跨任务验证 | 共运动、mask、rim-plane 和 grasp offset 已进入 route |
| 单一动作权 | 已实现 | 最新 run 的 requested/authorized/executed 链一致 |
| `ABOVE_ALIGNED → MV_DOWN → fresh belief` | **已在真实 rollout 观察到** | 最新 run step 166–177 |
| stale/UNKNOWN fail-closed | 已实现，但恢复仍需验证 | de319 在 stale 后停止；最新 run 已绕过 clipped-mask 停机 |
| `RIM_CONTACT` | **未解决** | 小 overlap 提前触发，up/down 重复 |
| visual memory 进入 placement Qwen | **代码与离线回放已实现，线上待验** | raw 图板和结构化 timeline 已接通；双次模式离线未优于单次 |
| `SEATED_HELD → RELEASE → VERIFY_TASK` | 未通过 | 最新 run 无 release |
| 完整 episode | **0 次成功** | 最新 3 次 V2.2 rollout 均 `success=False` |

## 下一步在线验证门槛

本轮离线接口与安全回归已完成。下一阶段仍需要在准备好 provider/Qwen 服务后做单 episode，不能把冻结帧分类计为任务成功：

1. 先核对新请求日志：关键帧引用和 Qwen input panel hash 非空、两次调用（若启用）图板相同、action receipt 与观测位移一致。
2. 单 episode 必须越过旧 step 178 型疑点，不因像素级 overlap 抬升；只在真实支撑与独立门槛成立后出现 `SEATED_HELD → VERIFY_SEATED → OPEN_GRIPPER`。最终成功仍只看 `env.check_success()`。
3. 若单次闭环通过，再按原门槛做同一 init state 5 次统计，之后扩展其他对象和 init state；`double` 仅在后续配对证据显示比 `single` 更好时启用。

## 尚未宣称完成

- 没有成功 episode，不能计入 3/5 门槛。
- AnyPlace 仍 shadow-only，尚未参与候选排序或动作。
- 两阶段反思接口已实现但没有在线验证，当前 profile 默认单次带记忆。
- 尚未完成 seating/contact 的通用闭环。
- 没有使用对象名称、坐标、尺寸或模拟器 pose 作为正式策略规则；离线诊断只用于解释 rollout 现象。

## 2026-10-01：隔离的 0928 Show-Harness 重建分支

- 为回到更贴近 Show-Harness 的检查点，另建 `/root/autodl-tmp/Show-Harness-Rebuild-0928`，从 0928 检查点 `f235aea58076b30a083dcbf1056221db481a5c53` 恢复；主工作树和 V2.1/V2.2 历史未被覆盖。候选保留原 Harness runner、SAM3/跟踪、episode visual memory、相机几何、VisualRoute、Qwen 客户端、verified runtime、原子控制器和 logger。
- 候选分支已完成定向离线修改：关键节点 Thinking 双轮结构化反思、同图同记忆 hash 审计、runtime 有限动作选项、失败后待验证的抽象 RSI 经验，以及 raw AgentView/Wrist、帧 ID、action receipt 的 runner 接口。24 项定向测试通过；候选分支仍未进行新的完整 rollout。
- 服务 smoke 确认 Thinking 模型 ID、完整最终 JSON 和两轮图像一致；一次双轮调用约 83.6 秒，冻结失败帧上仍有上下文不匹配，说明在线策略有效性未证实。SAM3 在同帧返回两个相近候选，策略保持弃权。
- 当前仅一张 RTX 4080 SUPER 32 GB 可见，Thinking+SAM3 同驻约 28.8 GB。MoGe 尚未并入在线策略；待真实首错证明度量深度缺失后再安排单 GPU 调度试验。这是阶段性硬件/证据约束，不是取消空间工具目标。
- 候选 rollout `run_d7cf4e98cbaa` 在 step 54 主动中断：task 2 / init 0 / seed 0、单 episode、100 个决策预算，停在 GRASP、夹爪 OPEN。`summary.json` 记录 `success=false`、`end_reason=interrupted`，但 `check_success=null`，因此这是未完成诊断轨迹，不能声称环境评分失败或成功。26.5 秒视频与 54 个 step/raw dual-view 已保留。step 49 video frame 显示 AgentView 横向已基本对准，而 model 仍根据旧 Wrist offset 判定两视角都需右移。离线确认空 memory 变化幻觉、跨视角 evidence 无法进入 runtime、进度标量错误、阶段状态/效果/action 不一致未拒绝。本轮下一步先修最早错误分支，再同初态单条复跑。最终成功只认完整 episode 的 `env.check_success() == true`。


## 2026-10-01：0928 候选输入分辨率与目标身份修复进行中

- 用户指出之前目标身份选错（不是红色物体）。旧候选诊断记录证明当时 Qwen 与 SAM3 输入均为 256×256；SAM3 的两个相似 bottle 候选分差很小并触发弃权，Qwen 首轮却自行补出“red cap”特征。当前修复不使用颜色硬编码。
- 隔离候选配置已将 LIBERO 原生渲染、AgentView/Wrist 方图预处理与模型输入统一提高至 512×512，并按两倍分辨率缩放关键像素门槛。SAM3 metadata 记录 request 图像 shape/hash；Qwen request audit 已记录每张实际输入图尺寸/hash。
- GRASP runtime 使用当前主/副视角的新鲜几何生成可执行动作集合。歧义时只提供有限视角探测动作与 STOP，不允许下降或闭爪；正常决策 schema、递归反思 schema 和 runner 最终执行口都强制这一动作集合。新鲜 AgentView 副视角可替代 Wrist 遮挡证据，过期视角不授权抓取。
- 另修复递归反思 fail-closed 返回值错误：图像 hash 不一致或结构化回答冲突时，runner 现在真正收到 STOP。定向验证 **28 passed**，`py_compile`、`git diff --check` 通过。
- 下一步为相同 task 2 / init 0 / seed 0 单条 rollout，预算 150 个决策。先核验日志中的两路输入尺寸都是 512，再按视频/关键帧复核目标身份、抓取与首个偏差；未完成任务只记为未成功/未评分，不把模型自述作成功。


## 2026-10-01：高分辨率诊断与目标身份跟踪调整

- 候选高分辨率 run `run_1892174f3f36` 在 step 3（总计 5 个原子动作）因明确的身份/结构化证据问题中断；夹爪 OPEN，summary 的 `check_success=null`，不计成功或完整失败。保存的视频、step JSON、SAM 元数据、raw 双视角和 Thinking 图像哈希均可复查；实际输入均为 512×512。
- 真实 SAM3 对同一 512 AgentView 使用更自然的 `salad dressing bottle` query 时，把用户指出的非红/橙候选之外的前景绿帽候选排在橙色瓶之前（0.7695 vs 0.6484）。分数只作候选排序依据，不写入固定类别/颜色逻辑。
- 首帧的相机 fallback 未启用，导致 SAM 后续歧义时缺少同 episode 主 tracker identity。现启用 Wrist→AgentView fallback，并在初次无歧义 AgentView 检测后保留 bbox/frame/confidence 锚点；歧义刷新时使用原 tracker，不按最高分重新切换。Qwen 收到同 episode identity anchor 及“外观冲突则 UNKNOWN”约束。
- runtime 已在未决身份时只开放视角探测/STOP，并在动作 schema 与执行前授权处双重限制；这次 trace 中未出现闭爪、下降或错误 release。一次双轮自相矛盾被校验器转为真实 STOP，暴露了 fail-closed 行为生效。
- 当前定向验证 **30 passed**，代码 compile 与 diff whitespace 检查通过；修复后新 episode 尚待启动。

## 2026-10-01: SAM3 本地权重服务诊断

- 单条启动尝试 run_a0fa49d4506f 使用 task 2 / init 0 / seed 0 和 512×512 观测。Qwen3-VL-8B-Thinking 完整返回规划 JSON（1472 completion tokens，50.6 秒），没有 token 截断。
- 该尝试因诊断时终止而记为 end_reason=interrupted、check_success=null；summary 记有 1 step，但 steps.jsonl 没有原子动作回执，所以不将它计作有效控制结果。
- 根因是重启 SAM3 时未设置 OPENETA_SAM3_CHECKPOINT_PATH，服务尝试从 Hugging Face 下载配置，而本次进程无法联网；SAM3 请求失败。已终止该尝试并用本地 checkpoint /root/autodl-tmp/openeta-services/models/sam3/sam3.pt 重启服务。
- 对前一条 rollout 保存的真实 AgentView 512×512 原帧重新做了 SAM3 查询 salad dressing bottle：输出 2 个候选，前景瓶 bbox [297,226,335,313]、score 0.7695；右侧橙色瓶 bbox [364,182,400,260]、score 0.6484。分差 0.1211 高于当前 0.05 歧义阈值。该结果来自图像与任务语义，不添加颜色或坐标硬编码。
- 下一条实验仍是同一初态的单条 episode。启动 SAM3 时必须显式传本地 checkpoint 环境变量；启动前确认 8002 的模型 ID、8773 的 MCP/SAM smoke、两路图像尺寸及显存。检查运行的视频、每帧原图、SAM 候选、Qwen 两轮输入哈希和 runtime 的请求/授权/执行回执。

## 2026-10-01: 512 分辨率下的身份分叉与修复

- 单条诊断 run_061d133d4a4a 使用 task 2 / init 0 / seed 0，模拟器渲染、SAM3 和 Qwen 图像均为 512×512。完整保留双视角原帧、25 个 step、视频、提示和 debug payload；夹爪始终 OPEN，summary 为 end_reason=interrupted、check_success=null，不算任务结果。
- 已从原始 AgentView 和 Wrist 帧确认首错。AgentView 的目标语义查询返回两个候选：前景瓶 bbox [297,226,335,313]、score 0.7695；右侧橙色瓶 bbox [364,182,400,260]、score 0.6484，分差超过歧义阈值。Wrist 的两次保留完整目标语义的查询弃权，随后旧逻辑生成 affordance-only 泛查询 near-center-mass-excluding-capped bottle，返回三个附近圆柱体；主 tracker 接受了最高分候选，身份错误导致 runtime 连续授权横向移动。AgentView close guard 因对齐残差 [63.32,140.19] 超过阈值而阻止闭爪，动作方向没有受同样身份 guard 约束。
- 从最早分叉修复 core/capabilities/visual_harness.py：SAM3 的每个回退查询都保留完整任务目标。affordance 只可补充明确出现的容器类别词，不再单独查询属性词或目标最后一个泛化词。Wrist 没有语义候选时，既有同帧 AgentView fallback 现在能接管并锚定目标实例。
- 新增查询身份保留及 Wrist 泛类误检→AgentView fallback 测试。相关测试 33 passed；模块 py_compile 和 git diff --check 通过。
- 512×512 已能把关键目标候选分开；本轮错误发生在跨视角查询降级，不需要继续盲目放大分辨率。保持视觉语义、候选来源和跨帧跟踪联合判断，不写颜色、物体名称或坐标专用规则。
- 下一条只跑同一初态一条 episode，首步核查 Wrist 的所有 query_attempts 是否保留 salad dressing 目标语义；若 Wrist 弃权，必须看到 AgentView 候选成为主身份锚点后才允许对齐。逐步检查 action receipt、真实图像运动、夹爪状态与最终 check_success。

## 2026-10-01: 原始帧回放确认跨视角身份回退

用已保存的 run_061d133d4a4a frame 0 原始双视角、当前 SAM3 服务和已修复的 VisualHarness 做离线回放（不推进模拟器）：Wrist 的两个语义查询均为 0 detections；随后 AgentView fallback 的完整目标查询返回 2 个候选，并将 bbox [297,226,335,313]、score 0.7695 记录为 fresh_unambiguous_sam3 identity anchor。另一候选 bbox [364,182,400,260]、score 0.6484。Wrist 与 AgentView request hash 分别为 33bedf390b3b 和 5a7baf54c360，输入均 512×512。回放确认新的查询规则触发 AgentView 主视角与目标身份 anchor。接下来进入同初态单条真实复跑。

## 2026-10-01: Wrist 正语义误检与 AgentView 主身份

- 修复查询降级后单条 run_afc99c8bab61 在 24 个记录动作后于 gripper OPEN 时中断，check_success=null。frame 0 的 AgentView 正确建立前景瓶 identity anchor；到 frame 8，Wrist 对完整目标短语产生一个高分检测，旧的 probe 分支将 camera primary 切回 Wrist，错误候选又覆盖全局身份。
- 已从 run 的双视角原帧确认：同一帧 Wrist 候选与 AgentView 锚定候选不保证是同一个物体。保留目标语义仍不能让 Wrist 在多物体近景中可靠承担全局身份。
- 现在配置启用 AgentView identity 后，GRASP/RELEASE 均固定 AgentView 为主身份与几何视角；Wrist SAM3 结果只作为独立 secondary evidence，不可覆盖主 track。原始 Wrist 图片仍进入 Thinking 请求。
- 使用真实保存帧 0 和 8，在当前 SAM3 服务上重新回放：AgentView frame 0 将 [297,226,335,313]、score 0.7305 锚定为 frame-0 identity；frame 8 AgentView 更新到 [296,238,334,323]，identity anchor 保持 frame 0。frame 8 Wrist 同时返回另一框 [430,215,515,268]、score 0.4609，但作为 secondary evidence 隔离，没有切换主目标。
- 当前查询身份测试集仍为 33 passed；py_compile 与 git diff --check 通过。下一条同初态复跑重点验证连续 AgentView 主身份、横向/深度对齐分量和 runtime 授权，逐帧确认不再跟随 Wrist 候选。

## 2026-10-01: run_4d357a5c0a64 与收敛计划的基线

- 在 0928 隔离副本保存了调整前代码快照：`/root/autodl-tmp/Show-Harness-0928-preconverge-20261001.tar.gz`；未覆盖原 Show-Harness 或 V2.1 工作树。
- 单条 `LIBERO_OBJECT/task 2/init 0/seed 0` 运行 `run_4d357a5c0a64` 在 27 个 simulator 决策步后因诊断中断，`check_success=null`，夹爪始终 OPEN，未进入持握/搬运/落座/放手。26 条 `steps.jsonl` 动作回执、512×512 双视角原帧、视频、提示与 debug payload 均已保存；这不是完整 episode 成败评分。
- AgentView 主身份贯穿本条轨迹：前景目标 bbox 从 frame 0 的 `[297,225,335,313]` 关联到后续帧，Wrist 对相似物体的检测没有再覆盖全局身份。水平残差约从 63 px 降到 15 px；frame 19 后开始前进/下降。相邻观测中 MV_RIGHT 的世界位移中位数约 5 mm、MV_DOWN 约 4.5 mm，而日志动作名义步长为 20 mm；需要先核对控制器实际动作效果与预算。
- frame 15 的双次 Thinking 完整返回，三张 512×512 输入在两轮 hash 一致；frame 22 因 `finish_reason=length` 触发 fail-closed STOP，随后仍执行了 frame 23–25 的 MV_DOWN。当前反思以“重复同方向”触发，虽然近期水平误差持续改善，属于误触发。
- 部分 768×768 与 `recursive_reflection.max_tokens=8192` 已写入未跟踪配置，但尚未进行真实服务验证；角色调用仍传 `max_tokens=None`，实际总生成上限仍是客户端 4096。768 的像素阈值也尚未归一化，禁止把这份配置当作已验证 profile。当前仅可见一张 RTX 4080 SUPER 32 GB，检查时 8002 与 8773 服务未运行。
- 用户确认继续收敛当前 0928 隔离副本。下一阶段先打通证据→授权→实际动作→观测的单一闭环、真实动作效果、关键事件反思和服务压力测试，再让同初态一条 episode 在持续取得进展时运行至完整终局。

## 2026-10-01: 动作尺度标定与执行链收敛

- 继续 0928 隔离副本，先读 AGENTS.md 并保留既有快照/视频。以相同 LIBERO_OBJECT/task 2/init 0/seed 0 初态做隔离单轴 MV_RIGHT 标定，4 个 motion step + 1 个 settle step：旧每步命令 0.005 m 的 EEF 世界 Y 净位移为 -0.004302 m；每步命令 0.020 m 的净位移为 -0.018897 m。两组均未触发 success；标定不当作 episode。LIBERO OSC_POSE 每个 env.step 从当前 EEF 位置重设相对目标，因此原适配器将原子请求除以 4，使 20 mm 请求只移动约 4 mm。现在 LIBERO 控制器重复完整请求量，按动作后 EEF 回执核实，而不是沿用 RoboLab 的除法约定。
- 原生 768×768 AgentView/Wrist、SAM3、Qwen 的阈值比例已接线，512 参考阈值乘 1.5。递归反思每轮真实传入 max_tokens=8192；同方向但实测持续收敛不再触发双轮。抓取 guard 只提供证据/否决；runner 的最终 option gate 记录 requested、candidate、authorized、executed，guard 不主动发出闭爪。针对关键行为的 23 个定向测试通过。
- 8002 返回真实 Qwen/Qwen3-VL-8B-Thinking、16K 上下文；8773 用本地 checkpoint，SAM3 的 768 输入查询返回两个候选，score 0.765625/0.65625。双服务常驻显存约 29.1 GiB / 32.8 GiB。三图 768、两轮 8192 服务压力测试正在进行，未通过前不将该分辨率视为已验证。

## 2026-10-01: 768 原生单条诊断 run_7ad374fb96ab

- 单条 task 2/init 0/seed 0 运行，保留 768×768 AgentView/Wrist 原帧、steps.jsonl、debug payload 与视频 rollout_failure.mp4。43 个决策后因为明确的 RELEASE→STOP 无进展循环人工诊断中断；summary 为 end_reason=interrupted、check_success=null，不能计作完整 episode 失败评分或成功。实际到达 GRASP 闭爪一次，未确认持握，未进入 TRANSPORT/落座/放手。
- 控制尺度修正有效：MV_RIGHT 请求 20 mm，首步实测世界 Y -18.9 mm；相同方向 5 次把水平残差 95→7 px。MV_DOWN 每步约 20 mm，MV_FWD 每步约 20 mm，AgentView 纵向残差 226→31 px；连续取得进展时没有触发重复动作双轮。身份锚点持续指向绿色盖瓶，右侧橙色相似瓶没有覆盖目标。
- 最早决策分叉 frame 22：目标仍在收敛且 AgentView dy 30.77 px，但 SAM3 最近检测停在 frame 20，freshness=1 导致 GRASP 选项不可用；低位 runtime 仍无条件给 MV_UP，Qwen 上抬约 44.5 mm，下一帧又下移。frame 24 的 SAM3 刷新及两步下降后 frame 25 闭爪，语义提议/授权/执行均为 GRASP。
- frame 25 闭爪前 AgentView dy 31.64 px、Wrist 局部 dy 83.13 px（其对齐参考 48 px）；旧 guard 只凭 AgentView 对齐、固定高度上限和检测新鲜度放行。原帧显示物体主体仍在指尖下方。frame 26 夹爪闭合，frame 27 有界上抬约 40 mm 后，AgentView 原帧明确显示瓶子仍立在桌面，未随末端上抬；这是空抓而非持握。
- 闭爪后两轮视觉 verifier 的第一轮几乎只列动作回执，并把同一回执同时列为支持和反证；第二轮虽答 YES，矛盾检查将最终判为 UNKNOWN。上抬后未对前后帧再次执行持握动作效果核验。随后旧 recovery plugin 把约 0.002 m 夹爪宽度直接译成 RELEASE，最终 option gate 因无语义放手授权将其挡成 STOP，frame 27–42 形成重复无进展。该宽度只有辅助意义，不能单独驱动语义放手。
- 下轮先修三个通用缺口：接近闭爪窗口时刷新 SAM3 并把新鲜 Wrist 局部冲突作为闭爪否决证据；只有发生真实接触/反向误差时才把上抬当作对齐选项；闭爪后执行一次有界上抬并复核目标与末端的相对运动，宽度只报告辅助信号。通过冻结帧及另一物体反例后再跑同初态一条完整 episode。

## 2026-10-01: 新身份分叉与持握核验

隔离副本已接上闭爪后的固定相机 mask/EEF 位移核验。旧冻结帧 25→27 的 EEF 上移约 33 px、目标底边位移 0 px，证明目标未随手运动；两轮 Thinking 单凭图像仍错答 YES，因此 runtime 必须用真实动作效果否决。新 fresh Wrist 低位冲突否决闭爪，固定夹爪宽度只记辅助信号。39 项定向测试通过。

单条 `run_dd9260919c6e` 在 task 2/init 0/seed 0 保存 768 原帧、7 条动作和视频；SAM3 从首帧起对两个瓶子近分弃权，主目标身份未建立，Agent 连续提出 MV_RIGHT。因身份无进展诊断中断，`end_reason=interrupted`、`check_success=null`，不算完整评分。单轮 Thinking 离线候选选择还虚构不可辨瓶身文字并选错候选。当前最早错误是歧义候选身份裁决，路线转向结构化视觉事实及有界主动取证；不把模拟器真值或当前物体颜色写入正式规则。

用户指出瓶身文字镜像。候选身份诊断改用同 reset 的 1536 原生图裁出两张局部图，仅对文字图水平翻转，原控制 768 图与相机几何不翻转。编号并排对照图消除了三图输入时的 Image B/C 归属颠倒：冻结帧两轮短 JSON 均把候选 0 读成 `CREAMY Ranch Dressing`、候选 1 读成 `Tomato Ketchup`，最终选 0；两轮 `finish_reason=stop`、图像 hash 相同，输出 106/199 token。身份 OCR 子调用仍使用本地 Thinking 权重，但限定 2048 token，避免此前 8192 token 耗尽仍无最终答案。只有 SAM3 两项近分、无身份锚点且两轮一致时才接收此索引；每 episode 最多尝试两次。41 项相关测试通过。下一条真实 episode 尚待运行，成功数仍 0。

## 2026-10-01: run_1958ca9b5729 与 plan02

本条保存了 768 双视角原帧、动作事件、debug payload 和失败视频；60 步后诊断中断，`check_success=null`，成功 episode 仍为 0。第 24–26 步 EEF 高约 0.124→0.122 m，verified runtime 连续授权 20 mm `MV_FWD`；绿瓶开始倾斜并在第 27 帧横倒，但旧屏幕残差指标仍报 `IMPROVING`。第 27–30 帧使用遮挡记忆旧框，第 31 帧 SAM3 语义刷新错选橙瓶，现有实例关联未覆盖 GRASP。第 36–58 帧 MV_LEFT 不在 runtime 选项内，持续 STOP。冻结第 28 帧文字提示 `bottle`/`plastic bottle` 无检测；SAM3 `segment_points` 返回包含倒地绿瓶的三张 mask，最高分只覆盖部分主体，不能直接选最高分。下一步按隔离副本 `docs/revolution/plan02.md` 修近接触和变姿态重识别。

## 2026-10-01: plan02 接线及冻结复现，真实复跑进行中

0928 隔离副本已在 GRASP 阶段维护同一实例，旧框不再计算当前进展。冻结旧轨迹第 22/24 帧显示：近接触时 Wrist 分别未定位目标/目标偏离指间轴约 200 px，新 option gate 仅给左右、上抬、后退、STOP，撤掉前进、下降与闭爪；近接触每步上限 5 mm。倒地第 28 帧用同 episode 身份框、当前变化点与 SAM3 点提示找回完整瓶身，bbox `[447,458,507,576]`；两轮图像 hash 一致，第一轮历史锚点可见，第二轮选同一实例，最终仅提议 STOP，不计作抓取。Qwen Thinking 模板强制开启思考，旧第二轮 8192 token 耗尽无 JSON；现在第一轮保留推理，第二轮同权重的预填最终答案通道在 139 token 内返回完整 JSON。35 项定向测试通过，8002 模型 ID/16K 和 8773 本地 SAM3 已复核。单条真实 episode 正在运行，尚无新评分。

首条真实接线诊断 `run_d86e510a71d2` 在初帧由 SAM3 取得两个候选，却因 runner 错把语义裁决发给控制器外壳而报 `semantic_resolver_unavailable`；两步 MV_BACK 未建立目标身份，3 steps 后诊断中断，`check_success=null`。身份、变姿态、抓取/放置/任务视觉核验入口已统一接到实际 `controls.controller.agent`，36 项相关测试通过。下一条仍用同初态单 episode 核对首帧身份和近接触行为。

## 2026-10-01: run_7bf8267e26bb 与 plan03 起点

0928 隔离副本单条 task 2/init 0/seed 0 运行在 30 步后诊断中断，`check_success=null`，未闭爪，成功 episode 仍为 0。AgentView 全局身份保持稳定，近接触门阻止旧版连续前进碰倒目标；横向残差缩小后，Wrist 原图第 27 帧可见瓶盖在两指之间，但 SAM3 完整目标短语仍弃权，局部关系 `UNKNOWN`，没有进入抓取。冻结图上可见部件短语和点提示能给出候选 mask，但其本身不能证明完整瓶身或安全闭爪。原帧、动作事件和视频保存在隔离副本 `rollouts/libero_showharness_0928_thinking_recursive_768/run_7bf8267e26bb/LIBERO-LIBERO_OBJECT-2/1001/task_0/21-49-29/`；视频 SHA256 `007babd34ea644c4abb07fcf143ab92a3d8aa6a035af83709bff170ad44bf664`，AgentView/Wrist 第 27 帧 SHA256 分别为 `090537e2bc7caa0356c33c83a1ded302ed8e19a604ed981290002f5086e77e22` / `d72b8a7050df251bed0b6b1108441a92f198d39d0d4b4a8890bff0be3625148c`。下一轮按 plan03 保存检查点，并围绕对象信念、主动取证和动作效果契约改革。
