# V2.2：让小模型具备可靠的观察、记忆和纠错闭环

## 2026-09-30 最新补充：V2.1 分辨率与 Mermaid 架构复现审计

按用户后续决定，本轮活动基线是恢复的 V2.1 + Qwen3-VL-8B-Thinking；以下记录覆盖早期 V2.2 状态，V2.2 内容作为历史方案保留。

- 256×256 原始渲染的目标辨认确实存在风险：当前同 init run 中 Qwen 曾将相邻橙色瓶状干扰物判成 salad dressing；离线相机/BDDL 诊断支持该候选错误。原生 512×512 后，SAM3 在 AgentView 中把中心绿色瓶体选为首候选（`0.9336`），而非把 256 帧双线性放大。因此“提高渲染分辨率”在本例有可观察收益。
- Thinking 抓取决策实际请求图像为 512×1024，processor 的 patch grid 为 `[1,64,32]`；输入没有被处理器缩回 256。VLM panel 还有基于新鲜 bbox 的 2.5× ROI，ROI 不创造源图细节。SAM3、MoGe、trackers 的单帧源图也是 native 512。
- 高分辨率没有带来持握：run `run_a9dedad2cace` 共 12 控制步后因没有有效进展停止；没有闭爪、`VERIFY_HELD`、搬运或成功评分。当前帧 AgentView 仍看见正确目标，但 Wrist 的 SAM3 proposal 指向另一候选，MoGe metric world-Z 被物理支撑平面检查拒绝。V2.1 的 Qwen pregrasp allowed set 只有 `PROBE_DEPTH/UNKNOWN`，没有能对全局图像残差执行一次有界校准纠正的 option；两次 `PROBE_DEPTH→MV_BACK` 后 agent abstain。问题已经从单纯目标分辨率扩展为跨视角身份、几何可信度和决策 option 覆盖不全。
- 直接旧副本 `Show-Harness-V21` 的 Thinking 尝试两次都在 planner final 截断处失败，没有环境动作。旧 planner 使用 no-think chat-template kwargs，旧 Instruct endpoint `8002` 也未运行。当前主仓的 V2.1 evolution profile 可以实际进入 Thinking/Memory 决策，但它不是旧副本的逐字节复现。Mermaid 所列主要模块在源码中存在；不能把静态架构连线等同于端到端抓取/搬运验收。
- 本次计划相对之前仅增加一个证据驱动的局部步骤：给 V2.1 semantic pregrasp 补一个可选 `VISUAL_ALIGN`。runtime 只在当前 AgentView 身份与 EEF 相机投影同帧有效时开放，按 calibrated pixel residual 编译一个 bounded 原子动作，然后必须看新帧；不让 Qwen生成坐标，不用对象名/位置/尺寸规则。若证据不足仍 STOP。该变化源于 run 事件中两个现象（正确全局目标偏离夹爪投影约 `[+63,-100] px`；Qwen allowed set 不含视觉对齐），不是预设状态机扩张。
- 下一步仍先跑回放/单测，再做一个同 init state episode。关闭把无跨相机关联的 Wrist proposal 当成目标身份的路径；MoGe 的 metric-Z 问题按官方模型输出坐标、输入 FOV 与相机标定逐项核验，不用试符号修补。抓取必须真实闭爪并由 `VERIFY_HELD` 通过；之后才做 transport、release 与泛化。

## 2026-09-30 最新执行记录：V2.1 memory/遮挡入口修复

本节按用户后续决定将活动基线覆盖为 V2.1 + Qwen3-VL-8B-Thinking；下方 V2.2 计划保留为历史设计参考，不能当作当前运行配置。

- 第三次 V2.1+Thinking run `run_e809899d7cee` 完成 approach 并进入 GRASP，但没闭爪、没通过持握核验，最终 `success=false`。第 57 步起 Wrist 遮挡，runtime 仍以旧几何走 `DESCEND_TO_GRASP` 到配置安全下限后停止。故当前不应回答“稳定抓住了”或“物理空抓已确定”；已证明的问题是 occlusion 时没有语义决策节点。
- 回看逐帧日志发现，第三次 run 的全部 `visual_memory_refs` 为空。raw PNG 虽事后保存，V2.1 在 `observe_frame()` 之后才保存图像，所以 memory 实际只有动作摘要。前三次 online Thinking rollout 不能证明历史图像 memory 或双次反思有效。
- 已改为在任何 VCR runtime 写 memory 前保存 raw AgentView/Wrist 并传 episode-local 引用；新增图像文件可读、同 episode、同实例、同 grasp epoch、严格帧序面板回归。缺少视图时不造引用。
- 新的 GRASP 遮挡入口反思针对一次可观测事件：同一 instance/grasp epoch 第一次 Wrist occlusion、最近 AgentView identity 新鲜、EEF 处于配置安全带时，冻结双视角与最多两个此前关键帧，调用 Thinking；`double` 两次共享同一 panel，选项仅为一次有界 `MV_UP` 或 `UNKNOWN`。runtime 仍授权动作；随后新观测缺失时不继续用旧残差移动，也不在同事件反复询问。它不依赖精确 stage-transition 当帧标志，不读取仿真真值或任务对象坐标。
- 回归结果：相关集合 **82 passed**；`py_compile`、`git diff --check` 通过；Qwen `/v1/models` 与 SAM3 SSE preflight 可达。离线通过仅代表接口可用。
- 第四次受控 run `run_b2650a17be4e` 验证了 raw 图像引用写入，但 Qwen 没收到图板：V2.1 reset 尚未将 runtime episode id 绑定到 logger 目录，构图器据此拒绝跨 namespace 引用。它在遮挡后持续 STOP，没闭爪；没有 `summary.json`，不记成功。
- 已修复：VCR runtime reset 一律带当前 logger 目录 episode id；memory bundle 优先保留紧邻当前帧的动作前观测，三帧时再加入最新的不同事件关键帧。由此 step 57 应包含动作前 frame 56 和当前 frame 57，历史第三帧按当前 episode 实际事件标记确定。
- 第五次同 init run `run_c7558b7f1c22` 到达 step 57，但共享 runner→role 路径读取未初始化的 `crop_meta`，在 Qwen 请求前退出。已修 `crop_meta=None` 并加 V2.1 runner 集成回归。
- 回归结果目前 **83 passed**，通过 panel builder、runner 传递和 role 结构化反思三层测试；真实 Qwen 在线 double 请求仍未验证。
- 第六次同 init run `run_66c443edd81c` 停在 `ALIGN_PREGRASP`，82 步未闭爪。逐帧核查发现 Qwen 相邻帧切换了两个瓶形候选：原实例的 motion-gated tracker association 有效但 confidence 约 0.886，低于既有 0.90 tie gate，runtime 因此重新调用 resolver，造成 identity 与 approach 方向漂移。已让刚确认实例后的短时 tracker `decision_lock_active` 也可维持同实例；失配、过期和 camera handoff 仍不放行。新增低于 0.90 的连续关联回归，当前 V2.1 Thinking 相关集合 **84 passed**。该 rollout 没有物理闭爪，不计成功。
- 最近修复不是提高 Qwen reasoning token 或更改物体几何规则，而是补齐“语义选定实例 → 同帧连续追踪 → runtime 目标锁”的状态传递。下一次同 init rollout 必须先确认身份不漂移，再审计遮挡反思确实收到当前 episode memory 双视角图板，然后观察真实抓取/持握；否则优先按最早错误证据修复。
- 第七次同 init run `run_320ffbcde0fb` 已证明以上身份修复有效：Qwen frame 0 选 candidate-0，frame 1 tracker association 0.88598 时仍保持 target-0001，稳定接近并到达 GRASP。step 56 double reflection 实际向 Qwen 发送当前 episode 的 frames `[54,55,56]`、raw panel 512×768；Qwen 首轮延迟 26.94s，输出 816 tokens 的结构化描述，但只引用 54/55。role schema 允许这种答案，runtime 要求 citation 包含冻结帧 56，因而首轮被拒绝、第二轮没有调用，runtime 选择 UNKNOWN/STOP。之后因无新观察耗尽 `DESCEND_TO_GRASP` 15 步预算；没有闭爪或 hold verification。
- 已统一首轮契约：`current_observation` 是必填结构字段，`frame_id` 的 schema enum 精确锁到冻结当前帧；结构化验证仍检查合法 camera/evidence refs。缺失、错帧、非整数 citation 均 fail-closed。定向 tests **15 passed**；完成完整回归后再跑同一 init，验收 Qwen 两次请求同图 hash、首轮当前帧引用、第二轮结构化输出和动作审计，再判断是否进入真正抓取。
- 下一关：在原 init state 做一次受控单 episode，审计 episode id、raw refs、双次 Qwen 调用实际收到的 panel hash/帧号、第一阶段描述与第二阶段最终选项；核验第一次遮挡时最多一个 `MV_UP`，`UNKNOWN` 不移动，之后必须看新观测。再核验是否真实闭爪与 VERIFIED_HOLD。只有持握通过后才分析 transport/放置，只有 `env.check_success()` 才算任务成功。

## 2026-09-30 用户决策补充：以恢复的 V2.1 为演进基线

用户决定先恢复曾具备抓取/搬运能力的 V2.1 harness，再换入独立的 Qwen3-VL-8B-Thinking 权重做受控演进。因而本文中“只改 V2.2”和“V2.2 是当前实验基线”的范围已被这项后续决定覆盖；本文的泛化、安全、当前 episode memory 和证据审计要求继续有效。

- V2.1 基线配置为 `configs/robot_libero_clean_qwen3vl_runtime_v2.yaml`；Thinking 实验配置为 `configs/robot_libero_clean_qwen3vl_thinking_runtime_v21.yaml`。保留 V2.1 的语义抓取选项、GPU1 MoGe v2 + 主动视差、transport 与 runtime 验证路径，不把 V2.2 放置重构混入本轮。
- GPU1 对历史帧的真实 MoGe v2 推理曾因 direct provider 引用未定义 `xs/ys` 返回 `SENSOR_FAULT`；已修复并以同帧 SAM3 mask、wrist 标定重跑，65/65 点通过 camera reprojection（中值约 0.71 px）。该检查证明路径能运行且坐标变换自洽，不把单帧 MoGe 估计等同于 metric 精度或已验证抓取。
- Qwen 在关键语义节点使用 Thinking 和结构化最终决策：状态假设、当前帧证据、反证、缺失观察、预期效果、失败条件及受限 option。runtime 校验当前 episode 帧引用后才可授权；模型假设以未核验状态写入同 episode memory。
- 当前 episode 的最多两个历史节点加冻结当前双视角，仅在进入 grasp 或观察到动作效果异常时最多触发一次双次反思。两次使用同一图板；重复模型判断不是独立物理证据。普通空间 UNKNOWN 不再被 prompt 强制映射为 `PROBE_DEPTH`。
- 已移除 LIBERO planner 中指定绿瓶盖沙拉酱瓶、固定 approach/grasp 顺序和按旧画面重复同方向的场景策略；保留相机标定和 runtime 提供的安全选项。
- 离线接口门槛已通过：V2.1 + Thinking 合并配置、Thinking 最终 JSON/帧引用、MoGe 正反几何回归、真实权重 CoTracker 连续帧跟踪，以及同一冻结图板双次调用均已验证。双次离线调用返回证据不足时的 `UNKNOWN`，成本约 78.21s/2281 completion tokens；这说明接口可用，不证明比一次判断准确。
- V2.1 + Thinking 已完成三次同 init 单 episode；第三次日志在 `rollouts/v21_thinking_evolution_0930/run_e809899d7cee/`。前三次都没有完整任务成功，第三次未闭爪；上面的最新执行记录说明了接下来要先验证的 memory/遮挡决策修复。历史 V2.1 成功叙述在精确 rollout 原图/元数据缺失时仍只作为待复现线索。

### 首次线上验证暴露的身份解析缺口

- 首次 run `run_d75d45f284b9` 在前 20 个控制步全部执行 `STOP`，无机械臂动作。SAM3 每帧有两只高分瓶形候选并报 `ambiguous_top_detections`；Qwen 在每帧重复选择，但 runtime 下一帧仍以 detector 的 `AMBIGUOUS` 覆盖已锁定 instance。旧 Qwen identity role 的上限仅 160 输出/96 thinking tokens；候选以检测 score 排序，分数微变会交换 `candidate-0/1`。
- 这不是抓取几何或 Thinking 是否能分析接触的问题，而是 SELECT_INSTANCE → track association → runtime observation health 的接口断裂。该 run 保留原图、视频和动作日志后停止，不能算物理抓取失败。
- 仅在 V2.1 Thinking profile 增加 `temporal_identity_lock_on_detector_ties`：候选 ID 每帧按空间顺序生成，full-view 框与 crop 同色/同 ID；Thinking resolver 最高 2048 输出 tokens，其中按 profile 给 1024 thinking tokens。只有同相机同实例、last_confirmed_frame 正好等于当前帧、tracker 明确 `instance_associated` 且关联置信度至少 0.90 时，才以时间连续性解除 detector tie；过期、camera handoff、tracker 内部 ambiguity 继续 fail-closed。
- 定向回归 42 passed，覆盖目标延续、过期/含糊拒绝、候选 ID 排序与标注图板。此修复已在第二次及之后的运行中使用；后续 episode 结果见本文顶部的最新执行记录。

### 第二次线上验证暴露的 approach option/budget 缺口

- `run_25907141366a` 证明身份修复有效，并开始执行接近动作。残差从 `[-0.029,-0.258]m` 改善到 `[-0.020,-0.076]m`，每步 EEF 实测约移动 5mm，transition 连续标为 `IMPROVING`。step 40 仍因 `ALIGN_PREGRASP` 40 步 option budget 停止，未到 GRASP。
- 最早错误不是方向映射：`_option_for_stage()` 在目标 track 恢复前看到 SAM bbox=null，因此选短 `ALIGN_PREGRASP`；其后同帧有效 temporal track 和 support-plane backprojection 算出约 0.258m 的远距离残差，控制器继续细步改善，却不再重新评估 option。由此耗尽 40 步时还剩约 71mm。
- V2.1 Thinking profile 新增 `metric_approach_uses_hover_budget`。在 fresh target track 和有效标定几何都成立后，APPROACH 使用当前 world residual 选择现有 `MOVE_TO_HOVER` 或 `ALIGN_PREGRASP`：大于配置世界对齐 tolerance 的两倍进入 80 步 bounded hover option；否则走 40 步 local alignment。它不增加动作权限、不引入对象尺寸/位置规则、不改变物理步长；跟踪或几何不可信时仍不移动。
- 远距离/近距离两个 option 回归均通过。第三次相同 init 的运行完成 approach 并进入 GRASP，证据和失败原因已转记在本文顶部；成功标准仍是真实 `VERIFY_HELD`、运输、release chain 和 `env.check_success()`，不能用从远处接近成功替代抓取成功。

后续章节仍保留 V2.2 的诊断与实现历史，不能把它们当作本轮 V2.1 Thinking 的线上验证结果。

> **范围澄清（按用户后续明确要求）：**本文中的 visual memory 仅指当前 episode 内当前帧之前的观测和动作效果。运行时不得把以往 episode 的图像、判断或实例信息作为 memory 发给 agent；episode reset 后清空记忆，并用 episode id 校验引用。旧 episode 只能用于隔离的离线评测。

## 0. 2026-09-30 实施补充与当前关口

本节记录按本计划开始接入 Thinking 后得到的更新证据，覆盖下文中较早的 rollout 状态；研究目标和验收标准不变。

- Thinking 使用独立的 `Qwen/Qwen3-VL-8B-Thinking` checkpoint、独立 V2.2 配置与 vLLM `qwen3` reasoning parser。Instruct 不是“不能推理”；旧的原子短答案接口没有给它足够预算表达复杂判断。本轮在关键语义事件请求 Thinking 模板和最多 4096 token，并且只解析最终 `content`，reasoning-only、空最终答案和截断输出不得授权动作。
- 最近磁盘中可恢复的三次抓取阶段 run 为 `run_79a617a272f4`、`run_129e12d313bf`、`run_1cfaa52e0a63`，它们都没有到达放置阶段。step 57 的第三次 run 中，Qwen 选择 `GRASP`，但几何为 UNKNOWN 且垂直对齐残差为 16.49 px；runtime 随后记为 `held=FALSE`。所保存的 Wrist 图像中物体与指爪关系仍含糊，不能据日志断言它物理上抓稳。该 run 的旧固定空载夹爪宽度判据与 VLM 判断冲突；代码现已取消 V2.2 的单独宽度否决，改为一次受 runtime 授权的有界抬升及多帧相对运动核验。
- 针对 step 57 图像中目标过小的问题，V2.2 pregrasp 图板现在可追加按当前检测框比例生成的原始像素 ROI；只在实例、相机和 bbox 帧号匹配当前冻结观测时加入，完整双视角仍保留。裁剪只放大已有像素，不添加场景知识。该改动通过 stale bbox、未知相机、内容比例和 prompt 一致性回归，仍待 Thinking 实际输入与真实抓取验证。
- 几何契约回归发现并修复 MuJoCo 图像 y 方向及相机旋转下的 rim 上缘选取问题；单来源几何不确定性不能塌缩为零。CoTracker3 + SAM3 在旧 run 的 16 张连续原始 AgentView 上完成隔离 smoke test，51 条轨迹可见、median delta 约 `[0.08,-0.01] px`、耗时 0.674 秒。此片段基本静止且遮挡，不证明动作效果跟踪已经改善；在线接线也还未由新 rollout 验证。
- 还发现 controller 的旧通用上下文含有“any-fruit 选择橙色”的对象类别建议，会进入非水果的 V2.2 任务。V2.2 现在只保留基于可见性和相机职责的通用规则；该对象特例仍留在其他旧 profile，避免扩大本轮改动范围。
- LIBERO 入口另有“绿色瓶盖沙拉酱瓶”指定和“同方向重复移动”的旧规划提示，会锚定对象类别并覆盖实际动作效果。V2.2 现改用相机标定、可行语义选项和每步效果核验的通用 planner context；非 V2.2 profile 保留既有文本。
- 为按计划独立统计抓取，新增 `--grasp-verification-only` 评测开关：只在 V2.2 runtime 已完成真实 `verified_grasp` 后结束本次试验；该结束不记作 `env.check_success()`，默认 profile 不启用。
- 本轮实际专项集合：`120 passed`；相关模块 `py_compile` 与 `git diff --check` 通过。vLLM 0.24 的启动参数 dry-run 接受 `--reasoning-parser qwen3` 并显式绑定 GPU0，但 Thinking 模型权重下载尚未完成，尚无服务响应或真实 Thinking rollout，普通 rollout 保持暂停。
- 放行顺序仍为：模型下载完成 → 最大图像输入下单次服务完整性/显存/延迟检查 → 持握正负例和有界抬升门槛 → 同一 init state 三次抓取专项 → 再恢复单次完整 episode。每次真实 rollout 检查视频和关键原始双视角帧。成功只能按 `env.check_success()` 记数。

## 1. 判断与目标

**目前有充分理由继续改进 harness，但还不能把失败归因于 Qwen“没有智能”。** 代码中已经发现几处会削弱观察、误导判断或阻断完成任务的问题，应先解决这些确定的问题。

| 已发现的问题 | 对任务的影响 |
|---|---|
| active parallax 将前后 bbox 中心和四角直接配对三角化 | 遮挡、框形变化后，它们未必对应同一个物理点，几何结果可能错误 |
| placement uncertainty 使用固定 tolerance，来源列表包含同一几何结果的派生项 | “证据新鲜、相互一致”不等于几何可信 |
| V2.2 关闭旧 intent 生成，但动作效果异常仍依赖它的预测 | 无效动作可能无法触发预期的反思 |
| 下降 stall 被直接解释为 `RIM_CONTACT`，并推动恢复阶段 | 底部支撑、碰边、控制受限可能进入相同的抬升分支 |
| 记忆主要保存帧和动作摘要，反思描述没有形成持续更新的判断记录 | Qwen 缺少“此前为什么失败、哪个假设已被否定”的信息 |
| Qwen 主要输出关系分类，缺少主动取证和选择恢复方案的接口 | 看不清时容易停住，判断错误后容易重复原方案 |

这些问题分别可在 [runner](/root/autodl-tmp/Show-Harness/core/sim/zeroshot_robolab_runner.py:1123)、[VisualRoute](/root/autodl-tmp/Show-Harness/plugins/visual_route/plugin.py:929) 和 [runtime](/root/autodl-tmp/Show-Harness/core/runtime_v2/runtime.py:1402) 中核对。

小模型可能较弱的能力，先作为设计假设：**小目标视觉辨认、跨帧对应、动作因果理解、冲突证据处理、失败后的策略切换。** 针对这些能力提供明确的证据和有限选择，避免要求模型从拥挤图板中自行重建整个物理过程。

按你的要求，**不安排大小模型对照，不调用付费模型**。本轮验收依据是本地系统的真实成功率、泛化和成本；“追赶闭源大模型”保留为长期目标，不提前宣称已经证明。

## 2. 重建 visual memory 的“写、读、用”

### 写：保存动作造成的变化

新增明确的动作转移记录：

`动作前观测 → requested / authorized / executed receipt → 动作后观测 → 效果核验`

- 每条记录绑定 episode、对象实例、grasp epoch、前后帧号、相机标定、实际位移和工具来源。
- 区分事实、模型假设和核验结论。Qwen 的描述不能直接升级为事实。
- 保存“预期变化、实际变化、是否符合、尚缺什么证据、哪个方案已失败”。
- 修正离线回放中 outgoing action 与前一段位移配对的问题。
- 关键事件按事件发生次数去重，避免持续 contact risk 把有限记忆全部挤成相邻重复帧。

### 读：围绕当前问题选证据

每次最多两个历史节点加当前节点，但选择规则改为：

1. 当前疑点对应动作的执行前帧；
2. 最近一次可靠状态，或与当前判断相矛盾的关键帧；
3. 当前冻结观测。

同时提供原始双视角、必要的局部裁剪和对应关系。裁剪用于突出目标，不能当作新增细节；目标被遮挡时应请求新观察。

请求前验证引用、时序、实例、相机和上下文长度。记录**实际发送的图像、文本、处理参数及 hash**，由发送接口生成审计记录，避免日志图板与真实载荷不一致。

### 用：让记忆改变下一次决策

增加有期限的结构化判断记录：

- 当前假设及支持、反对证据；
- 已失败的 option 和失败条件；
- 下一次需要观察的量；
- 假设成立、被否定或过期的状态。

实例切换、掉落、重新抓取或几何重建后，相关判断失效。跨 episode 只复用经过开发集验证的通用经验，不携带当前物体坐标或实例判断。

默认继续使用**一次带记忆判断**。双次反思保留为实验开关：同一冻结观测，第一次描述证据，第二次输出既有关系；重复一次模型判断不算独立物理证据。

## 3. GPU1 的模块选择与接入方式

当前 GPU1 约有 32 GB 空闲，但接入顺序由实际瓶颈决定。

| 模块 | 明确职责 | 本轮安排 |
|---|---|---|
| 现有 SAM3 | 对象、容器和可见区域的分割与重新定位 | 保留，向下游传递真实 mask 和来源；传播得到的 mask 单独标记 |
| **现有 MoGe-2 + 标定几何** | 提供对象局部点云、容器可见表面及几何不确定性 | 优先修复并接入 placement；明确使用匹配的 v2 类和 checkpoint |
| **CoTracker3 online** | 提供跨帧点对应、可见性、相对运动与遮挡线索 | 新增 GPU1 服务，先用于动作效果和几何验证 |
| 现有 AnyPlace | 提议物体放置位姿 | 继续 shadow；本轮不让不可执行候选进入控制 |

MoGe 官方支持度量点图，且要求模型类与 checkpoint 版本匹配；这不意味着它在当前场景已经具备毫米级精度，仍需标定与误差检查。[MoGe 官方实现](https://github.com/microsoft/MoGe)

CoTracker3 提供在线点跟踪和可见性输出，适合补上当前 bbox 配对缺失的对应关系。[CoTracker3 官方实现](https://github.com/facebookresearch/co-tracker)

具体接入要求：

- 在分割区域内选择稀疏点，分别跟踪 held object、容器和静态参考区域；同一相机内维护轨迹，不直接混合 Wrist 与 AgentView 像素坐标。
- 三角化必须通过点对应、视差、正深度和重投影检查。**持握物体在运动时，不得按静态场景直接三角化**。
- MoGe 使用对象 mask 提取点云；mask 缺失或不匹配时返回不可用，不能静默使用整幅场景代替对象。
- 分开记录推理耗时和观测年龄；冻结观测不会仅因推理超过 0.5 秒就变成过期帧。
- 不再用对象点云的空间大小直接冒充估计误差。几何不确定性来自重投影误差、跨帧残差和来源间差异。
- GPU0 保留 Qwen；GPU1 的几何与跟踪服务显式绑定设备、缓存模型、限制历史窗口。先测峰值显存和延迟，再决定驻留组合，避免每次调用重新加载模型。

AnyPlace 当前候选全部标为姿态不受支持、不可执行。它需要先解决输入点云质量和执行能力匹配，开启候选排序本身不能解决当前闭环。

## 4. 改进 ReAct，并修正接触与放手协议

采用低频语义决策、连续效果核验的流程：

```text
更新观测与记忆
    ↓
Qwen 判断当前状态、缺失证据及下一项 option
    ↓
runtime 检查前置条件并执行一个有界动作
    ↓
新观测核对预期效果
    ├─ 符合：继续 option
    └─ 不符：写入失败证据，重新取证或切换 option
```

这借鉴 ReAct 的观察—行动交替，以及 Inner Monologue 将环境反馈送回规划器的机制；具体机器人效果仍由本项目验证。[ReAct](https://react-lm.github.io/)、[Inner Monologue](https://innermonologue.github.io/)

### 给 Qwen 有效的决策接口

保留 `PlacementRelation` 和 placement verifier，新增 V2.2 专用的语义决策接口：

- 选择 runtime 提供的可行 option；
- 请求 `REFRESH_GEOMETRY`、`REOBSERVE`、`CHANGE_VIEW` 或 `REPLAN_PLACEMENT`；
- 指定要验证的证据项及预期变化；
- 返回无法判断的原因，而不只返回一个 `UNKNOWN`。

Qwen 不能输出任意坐标或绕过动作授权。runtime 为每个 option 提供前置条件、效果检查和预算；被拒绝时，将具体原因反馈给 agent。

动作效果预测直接由当前 option 建立，不再依赖已关闭的旧 intent。连续两次有效执行但没有预期进展，或出现四步往返时，必须进入重新取证或重规划；同一证据版本下不得重复已经失败的方案。

默认每个决策事件一次 Qwen 调用。双次反思只用于指定关键事件；同一冻结证据上的调用总数封顶，禁止反复询问直到得到期望答案。

### 区分接触现象与接触位置

- 二维 overlap 只产生疑点。
- stall 先记为“执行响应异常或可能接触”，不能直接判为碰边。
- 先核对实际指令是否执行、是否被安全限制，再结合包含关系、物体与容器位置、跨帧变化判断 rim contact、support 或 UNKNOWN。
- provider 报告 stall 后，先进入核验节点，不能提前把 route 切成必然抬升的恢复路径。

### 同时验证安全性和可完成性

用有来源、有时间范围的支撑证据替换 `_placement_contact_verified` 单一布尔值。

`SEATED_HELD → VERIFY_SEATED → OPEN_GRIPPER` 要求新鲜身份、可信包含关系、下降后的支撑证据和稳定性核验。夹爪保持不动、EEF stall 或 Qwen 的单次肯定均不能独立满足门槛。

证据不足时，先刷新感知；只有路径、持握与净空前置条件成立，才允许一次有新观测的有界探测。探测后仍不确定则结束该恢复分支，不循环耗尽预算。

**测试必须包含正确落座能够放行的正例。** 如果现有传感条件无法区分支撑与碰边，就明确记录为可观测性缺口，增加有效观察，不能用放松阈值掩盖。

## 5. 实施顺序、测试与验收

### 阶段一：修复接口与证据可信度

先修动作时序、请求载荷审计、效果预测、stall 分类、MoGe 版本选择和 V2.2 隔离。

补充回归：

- 空引用、过期帧、实例切换、错序、上下文超限；
- 安全层拦截动作与真正 stall 的区别；
- step 178 型 overlap 且未 stall，不得直接抬升或放手；
- 底部支撑不会直接被转成碰边恢复；
- 错误落座被拦截、正确落座能够进入 release；
- 所有新行为仅在 V2.2 生效。

目前历史 rollout 原图已缺失，最近一次运行因日志目录消失异常退出。先核对可恢复记录；不能恢复的实验标记“原始证据缺失”，不能继续当作已完成回放。普通 rollout 保持暂停，直到接口与安全门槛测试通过。

### 阶段二：完成记忆、点跟踪和语义恢复接口

按“动作证据 → 记忆检索 → 几何与跟踪 → 语义恢复”的顺序接入，每次保留独立开关。

离线比较仅使用本地 Qwen：

- 当前帧；
- 一次带记忆；
- 双次反思。

保持其余输入一致；无记忆组同时移除历史图像和历史文本。标注区分“实际状态”和“输入中能否看出来”，避免把不可见接触强行标成模型应知事实。

同时记录错误接触、错误 seated、有效 UNKNOWN、纠错成功、延迟和 token。双次无收益则保持关闭。

### 阶段三：打通一个真实 episode

离线门槛通过后恢复单 episode。每次失败分析**最早出现错误证据或错误决策的位置**，而不只分析最终 timeout。

验收必须真实出现：

`SEATED_HELD → VERIFY_SEATED → OPEN_GRIPPER → RETREAT`

并且 `env.check_success()` 成功。基础设施异常单独统计。

### 阶段四：稳定性与泛化

沿用已有路线：

- 同一 init state 五次，至少三次成功；
- 三类对象、三个初始状态、三个种子，共 27 次；总计至少 15/27，且每类至少 5/9；
- 通过后进行 50 个固定初始状态评测。

只做本地系统消融，衡量记忆、点跟踪和恢复机制各自的收益。成本包含失败尝试、GPU 时间、工具延迟和 Qwen token，报告每次成功的总成本。

只有在输入证据已经充分、同类语义错误仍稳定复现时，才进入局部纠错 LoRA；训练数据来自本地轨迹、人工复核和隔离诊断，不依赖闭源教师。

**默认边界：只改 V2.2；正式策略不读取模拟器对象真值；不引入对象名称、固定坐标或尺寸规则；同步维护 `progress.md` 与 `route.md`。优先实现可验证的完整闭环，再逐项证明新增模块的贡献。**
