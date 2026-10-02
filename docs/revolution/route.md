# Show-Harness 当前路线：V2.1 + Qwen3-VL-8B-Thinking

更新时间：2026-09-30 UTC

## 当前实验分支

用户已决定恢复曾改善抓取和搬运的 V2.1 harness，再切换到 Thinking 做受控演进。V2.2 放置路线及失败分析保留在下方供查阅，不是本轮活动 baseline。

### 最新分辨率与复现结论（2026-09-30）

- 当前对照 profile 是原生 512×512 的 `configs/robot_libero_clean_qwen3vl_thinking_runtime_v21_512.yaml`。512 输入改善了本例的全局目标 grounding：SAM3 首帧定位到中心绿色目标瓶，256 运行曾将右侧相邻橙色瓶状物交给 Qwen 作为目标候选。Qwen 实际收到了 512×1024 图板；本地 processor 重放的 patch grid 与输入原始尺寸一致，没有缩回 256。
- 512 run `run_a9dedad2cace` 仍未闭爪。正确 AgentView target track 与 Wrist 的 SAM3 proposal 不一致，MoGe 的世界高度又未通过支撑平面检查；Qwen 在拿到同 episode 图像 memory 后选择 UNKNOWN / 两次深度 probe，系统没有可选的“基于可靠全局视图做一次校准视觉对齐”动作。因此目前可判断“分辨率确实是一个问题，但不是唯一瓶颈”。
- 隔离副本 `/root/autodl-tmp/Show-Harness-V21` 的最初两次测试曾停在 planner；后续已核实并修复 Thinking final 截断，第一条有动作 baseline 现在停在 V2.1 的 approach XY/下降切换。8002 当前运行的是精确 ID `Qwen/Qwen3-VL-8B-Thinking`，不是 Instruct。抓取、搬运和完整成功仍未复现。

```text
保留 V2.1 + Thinking 的原生 512 实验配置
  → 修复 V2.1 pregrasp 决策缺少视觉对齐选项，并补齐同帧 AgentView EEF 投影
  → 回归 stale / ambiguous / cross-camera mismatch 均拒绝授权
  → 单 init state 验证一次有界视觉纠正 → 新帧复核 → 真实闭爪 → VERIFY_HELD
  → 持握通过后验证 transport，之后再恢复完整放置任务
  → 做 3-run 抓取复现，再测另一个物体和初始状态
```

本轮不把 512 提升成全局默认分辨率，也不在 MoGe metric-Z 问题未解释前恢复普通 rollout 批次。每次 online run 仍须看视频与关键原图；目标可见但身份/投影不够可信时，动作数为零。

### 隔离 V2.1 Thinking 分支：旧 profile 的单轮 baseline

这里记录 `/root/autodl-tmp/Show-Harness-V21`，与上方主工作树的原生 512 memory-assisted 运行分开，不复用对方的 V2.2 修改或 episode memory。

```text
Qwen Thinking 真实 max-multiview service check [通过，reasoning/final 分离，完整 JSON]
  → VLMClient 为每个 Qwen Thinking 请求加显式 reasoning cap [通过，模板不支持用 false 关闭 think]
  → 单轮同状态 baseline run_47b7ee44b206 [已运行，计划有效，approach 失败]
  → 固定 approach 已对齐后小幅 XY 漂移触发横向/下降循环 [已定位，视频+原帧+action receipt]
  → fresh-evidence descent XY hysteresis [离线回归 96 passed，已实现]
  → 同状态单条复跑，核对是否到达 Qwen 抓取前决策 / GRASP / VERIFY_HELD [下一步]
  → 只在通过单轮基线后启用事件触发的双次反思 [待实现/验证]
  → 持握后逐段检查 TRANSPORT 与 placement；最终只认 env.check_success() == true
```

`run_47b7ee44b206` 路径为 `Show-Harness-V21/rollouts/libero_clean_qwen3vl_verified_capability_runtime_v21_thinking_single/run_47b7ee44b206/LIBERO-LIBERO_OBJECT-2/0930/task_0/22-34-49/`。结果 `success=false`、81 actions、`runtime_v2_option_budget_exceeded`；模型 planner 55.24s 后返回五个 subgoals，之后 80-step hover option 在 world XY threshold 8 mm 与下降动作的小幅横向残差间来回。第 40 步开始下降，下降后观察到约 9.8 mm residual，超过 8 mm 原 threshold。现在只在当前 fresh XY/EEF evidence 位于双倍 threshold 内继续有界下降；证据过期/丢失仍停止。当前完整 `env.check_success()` 成功数仍为 0。

```text
核对恢复版 V2.1 的配置与 MoGe v2 路径
  → Thinking 结构化决策、帧引用和同 episode 双次反思离线检查 [通过]
  → MoGe v2 使用真实 SAM mask/相机标定做 GPU1 推理与重投影验收 [接口通过]
  → 首次 V2.1 + Thinking episode 暴露身份选择未闭环 [已诊断]
  → 稳定候选编号、同帧追踪延续与充分 Thinking 预算 [回归通过]
  → 第二次运行暴露远距离 APPROACH 被短 fine-align budget 截断 [已诊断]
  → 按跟踪后的 metric residual 选择 bounded hover/align option [回归通过]
  → 同一 init state 第三次运行，检查图像、动作回执和 Qwen 输入 [已诊断：未闭爪]
  → 修复 V2.1 raw-frame memory 写入时序和遮挡入口决策 [接口回归通过]
  → 第四、五次 run 暴露 episode namespace 与 runner→role 接口错误 [已修复]
  → 第六次 run 暴露已确认目标被 Qwen 重选、approach 漂移 [已复现]
  → 短时 tracker identity lock 连续性修复 [84 项专项回归通过]
  → 第七次 run 验证身份锁稳定、当前 episode memory 实际送达 [接口已观察]
  → 统一 double reflection 的 schema/runtime 当前帧证据契约 [回归通过，线上待验证]
  → 同一 init state 单 episode，验证双次调用 hash、受限取证选项与真实持握 [下一步]
  → 首次可靠持握后确认 transport，再定位放置的最早失败点
  → 同 init state 多次复现；之后再做跨物体/初始状态泛化
```

### 当前证据状态

- 旧文档报告 V2.1 曾到达 `VERIFY_HOLD`，但该运行的原图/动作日志与模型 revision 目前不完整；“GPU1 几何模块改善抓取”的记忆作为待复现实验假设。
- 当前 Thinking profile 已合并到 V2.1，V2.2 placement 开关关闭；`/v1/models` 返回 `Qwen/Qwen3-VL-8B-Thinking`。Thinking 输出要经过 JSON schema、当前帧引用和 runtime action gate。
- 43 项针对性离线测试通过。GPU1 MoGe v2 worker 使用历史 SAM3 mask 和相机标定完成推理，64/64 点通过重投影，中值约 0.71 px；直接 provider 也经修复后通过 65/65。重投影只验证变换自洽，不证明度量精度。CoTracker3 使用真实 checkpoint 在 Wrist 连续帧 56→71 返回 `VALID`，延迟约 0.712s；尚未证明点轨迹能改善动作效果或抓取。
- Thinking 双次反思离线回放见 `rollouts/offline_v21_thinking_replay_0930.json`：同一冻结图板、同一图像 hash，两次输出完整；第一阶段描述时序证据，第二阶段在证据不足时返回带引用的 `UNKNOWN`。代价约 78.21s、2281 completion tokens；这是可用性/成本样本，不构成单次与双次准确率比较。回放图来自旧 episode，明确仅为离线诊断，绝不作为在线 memory。
- 首次线上 run `run_d75d45f284b9` 在 step 0–19 反复 `STOP`，机器人没有移动。SAM3 把两个瓶形候选判为歧义，Qwen 虽每帧都提交候选，runtime 却没有让同帧高置信追踪延续清除分数平局；候选编号还随检测排序改变。该 run 在无环境动作后停止，原帧、events、steps 和视频已留存，标为身份解析接口失败，不当作抓取失败或成功。
- V2.1 已按此证据补上空间稳定 candidate ID、候选框/crop 对照、1024 token Thinking 预算，以及同实例/同相机/当前帧高置信 track 才能解除检测器歧义的 gate。新回归检查了有效追踪放行、旧/含糊追踪继续拒绝、候选排序稳定和有色图板；当前定向集合 42 passed。
- 第二次 run `run_25907141366a` 到达并执行接近，但 step 40 达到 `ALIGN_PREGRASP` 40 步预算时仍有约 71 mm 最大 world-XY 残差；日志显示同方向动作持续改善、EEF 实际移动约 5 mm/决策，没有振荡或 stall。因为原 detector bbox 为空，option 在获取 tracker/support-plane 结果前被选为 fine align；后算出的远距离 metric residual 没切换到 hover 选项。
- V2.1 Thinking profile 现在按新鲜跟踪与标定 support-plane 的距离，在已有 `MOVE_TO_HOVER` 80 步预算和 `ALIGN_PREGRASP` 40 步预算间选择；采用现有世界对齐 tolerance 的两倍作 coarse/local 分界，不改当前 2 cm 命令步长或针对任务坐标。单测覆盖远离目标转 hover、近目标保持精调。
- 第三次 run `run_e809899d7cee` 共 72 步、`success=false`：它完成 approach 并进入 GRASP 子目标，但没有闭爪，也没有 `VERIFY_HELD`。第 57–65 步 Wrist 目标被夹爪遮挡，runtime 仍允许 `DESCEND_TO_GRASP` 按旧几何继续下探，到达约 0.105 m 安全下限后进入 relocalize 并耗尽恢复预算。原始双视角帧 56–65 保留；不能将这次描述为物理空抓，因为没有执行闭爪。
- 对第三次 run 的 memory 审计发现：raw 双视角 PNG 事后存在，但全部 72 个 VCR event 的 `visual_memory_refs` 为空。runner 只在 V2.2 开启时于 `observe_frame()` 前设置 refs；V2.1 则于调用后才保存原图，故前几次 Thinking rollout 未测试到持久历史帧 memory。现在对所有带 VCR memory 的 profile 在 runtime 观察前保存 raw views，并把解析路径传入；测试验证两帧保存后的实际文件可重新组成同 episode 512×512 图板。
- V2.1 新增一次性 GRASP 遮挡入口反思：不再依赖脆弱的 stage-change 当帧标志；在同一实例/grasp epoch 首次遇到 Wrist 遮挡、AgentView 身份新鲜且 EEF 高度合规时，用冻结双视角及最多两个此前关键节点调用 Thinking。double 的两次共享 panel，只能选一次 bounded `MV_UP` 或 `UNKNOWN`；后续遮挡 fail-closed，不反复问也不沿旧几何下探。该分支及 memory capture 由测试覆盖，但尚未在线复现。
- 第四次 run `run_b2650a17be4e` 的同帧/同 episode 检查又暴露两个接口断点：V2.1 reset 没有绑定 logger 的 episode id，构图器因此拒绝请求；bundle 也曾被旧 stage keyframe 挤掉紧邻动作前帧。这个 run 在 step 57 后安全地连续 STOP，到 `DESCEND_TO_GRASP` 子目标预算结束；没有闭爪，未产生任务成功 summary，也没有发生遮挡后的环境移动。
- 已将所有带 `visual_memory` 的 VCR runtime reset 到唯一 logger episode id；memory 检索现在优先取最近动作前帧，再取不同的近期关键帧与当前帧。二帧上限时优先动作前帧/当前帧。因果时序和画板帧序都可审计。
- 第五次 run `run_c7558b7f1c22` 已抵达 step 57 遮挡反思；reset namespace 与动作前帧检索已越过 builder 校验，但 V2.1 runner 读取了未初始化的 V2.2 `crop_meta`，在 Qwen 请求发出前崩溃。保存 `success=false/end_reason=error` summary；frame 57 raw views 已写，未执行闭爪、抬升或下降。
- 已初始化共享 resolver 使用的 `crop_meta=None`，并添加 V2.1 runner→role 集成回归；相关专项现在 **83 passed**。该测试证明 panel 会传到 role，但真实 Qwen 双次调用还没发生。
- 第六次 run `run_66c443edd81c` 说明此前限制不止在遮挡入口：开局 Qwen 在相邻帧选择不同瓶形候选，尽管 tracker 有效关联原目标（confidence 约 0.886），0.90 的 tie gate 仍要求重新选择，导致目标切换和 approach 残差循环。第七次 `run_320ffbcde0fb` 已显示短时 identity lock 可稳定保持原实例，连续接近至 GRASP 入口。
- 第七次 run 的 step 56 实际请求 panel 为同一 episode frames `[54,55,56]`，首轮 Thinking 带 request audit 和图像 hash，说明 visual memory 的读链终于在线走通。但 Qwen 首轮只引用 54/55；schema 未要求当前帧，而 runtime 又要求引用 56，于是二次调用被跳过并安全 STOP。已把必填当前帧 citation 加进首轮 schema；相关定向测试 15 项通过。
- 下一次 run 核验两阶段请求共用 panel hash、首轮结构化描述明确引用当前帧、第二轮输出被完整记录，以及 runtime 是否按 Qwen 的受限选项执行一次安全取证。若结果 UNKNOWN，需保留新 observation/reobserve 分支，不能以重复静止帧空耗 option budget。真实闭爪和 hold verification 未通过前不声称抓取恢复。
- Qwen 的 visual memory 只来自同一 episode 的历史帧和实际动作效果。旧 rollout 仅用于离线诊断，绝不作为在线 memory。

### 放行条件

1. MoGe 使用当前实例 mask 和 camera calibration，输出经投影往返校验的几何；坏几何保持 UNKNOWN。正式决策仍记录单目几何的不确定性，并与可用的视差/动作效果证据核对。
2. Thinking 完整返回最终 schema；reasoning-only、过期帧证据、截断和图片 hash 不一致都不得产生动作。
3. 双次反思只在进入 grasp 和动作效果异常等事件触发，期间环境不动作，两次使用同一冻结双视角图板；同一 episode reset 后记忆清空。
4. 单 episode 后检查真实 hold verifier 与动作回执、逐帧视频/原图。只有 `env.check_success()` 计完整任务成功；一次抓取结果不外推为泛化能力。当前前三次 V2.1+Thinking run 都未验证持握；第四次首先验记忆/反思实际入参，不因测试通过预先宣称抓取改善。

---

## V2.2 历史路线（非当前活动基线）

更新时间：2026-09-30 UTC（V2.2 视觉 grasp 选项及方向效果记忆通过离线门槛；准备再次单 episode）

## 最新状态覆盖（2026-09-30，Thinking 接入阶段）

本节覆盖下文较早的“准备再次单 episode”结论。当前不运行普通 rollout，直到 Thinking 服务、最大图像请求、持握正负例及一次有界诊断抬升门槛通过。

- 最近三次可恢复的 V2.2 run `run_79a617a272f4`、`run_129e12d313bf`、`run_1cfaa52e0a63` 都没有进入 placement。前两次在 probe/遮挡阶段 STOP；第三次 step 57 Qwen 选 GRASP 后 runtime 写 `held=FALSE`。该标签受旧固定空载宽度规则影响，不足以证明当时物理状态；保留为未解决抓取案例，不统计为成功。
- `run_1cfaa52e0a63` step 57 当前可查到的决策上下文为空间 UNKNOWN、约 16.49 px 垂直对齐残差，Wrist 物体像素较小且指间夹持关系不充分清楚。现有错误链有两处：小模型在证据弱时给了确定 GRASP；runtime 又用受物体形状影响的固定绝对宽度单独判空抓。V2.2 新实现既提供新鲜 detector ROI，也要求真实动作效果和稳定性确认。裁剪不等于新信息，Thinking 收益尚未实测。
- 几何方面修正了 MuJoCo 原始图像纵轴变换与旋转相机 rim 世界竖直方向；CoTracker 在 TRANSPORT 延续在线流的代码已接入。离线 CoTracker 16 帧 smoke 只有接近零的图像位移，因此只能证明模型/API/同帧 SAM3 mask 可运行，不能证明抓取共运动证据有效。
- 已从 V2.2 controller/planner context 移除橙色水果及绿色瓶盖沙拉酱瓶等类别提示，也移除按单帧偏移重复同方向的旧动作建议。V2.2 保留相机标定及每步效果核验；其他 profile 行为未改，兼容性有测试覆盖。
- 新增 `--grasp-verification-only` 专项入口，在 V2.2 runtime 确认持握后保存记录并停止；不会把截断 episode 标记为任务成功，普通任务 profile 默认关闭。
- 当前离线验收为 **120 项测试通过**；服务脚本 dry-run 验证了 `Qwen3-VL-8B-Thinking` 绑定 GPU0、16K context 与 `qwen3` parser 启动命令。checkpoint 下载和完整响应测试未结束，未启动真实 rollout。Qwen final answer 必须是完整 schema JSON，reasoning-only 或截断统一 abstain。

### 当前局部路线

```text
完成 Thinking 权重
  → 最大多图请求服务验证（分离 reasoning/final、截断拒绝、显存/延迟）
  → 持握 false-positive / false-negative 门槛
  → `--grasp-verification-only` 同 init state 抓取 3 次 + 另一个物体 1 次，逐次检查视频和原始双视角
  → 通过后恢复 1 次完整 episode
  → 完成 env.check_success() 后才进入 5-run 与泛化统计
```

完整任务仍需实际观察 `VERIFY_HELD → TRANSPORT → SEATED_HELD → VERIFY_SEATED → OPEN_GRIPPER → RETREAT` 并由 `env.check_success()` 确认。现在没有这样的成功记录。

## 用户澄清：Visual Memory 的范围

Visual memory 只包含**当前 episode 到当前时刻为止**的观测与动作效果。episode reset 时清空，新的 `episode_id` 隔离历史条目；在线请求禁止读取其他 episode 的图像、模型判断或实例记忆。旧 rollout 如被用于配对诊断，只属于离线评测输入，不属于在线 memory。

## 最近单 episode 的最早错误及修复

- 正确仿真环境下的 `run_79a617a272f4` 和 `run_129e12d313bf` 都在抓取阶段失败（各 68 步，`success=false`，没有到放置阶段）；两条 34 秒 rollout 视频均已抽帧检查。
- 第一次错误是 `PROBE_DEPTH` 固定映射到已走反的 `MV_BACK`，原语动作 watchdog 没有约束语义 option。第二次补丁反向执行 `MV_FWD` 后虽然残差改善，但因只出现一次反证又回到 `MV_BACK`。现保留单次清晰 wrong-way 证据：继续沿反向动作只要改善就不切回；反向也无进展则停止并请求重新观察/规划。
- 第二次 run 还证明 `SpatialBelief.UNKNOWN` 时 `allowed_answers` 只含 `PROBE_DEPTH/UNKNOWN`，prompt 强制 probe，Qwen 不能选择可见的指间 `GRASP`。V2.2 现仅在最新 Wrist 身份、同帧机器人位姿和配置的 EEF 安全接近带成立时开放视觉 `GRASP`；runtime 随后仍独立验证 held。其他配置未启用该路径。
- PREGRASP 实际请求使用当前帧 + 同一 episode 的紧邻前帧，拼成标注有 `BEFORE/NOW` 的 AgentView/Wrist 图板。该处持久 memory refs 为空；放置关键节点才用 episode-id 校验后的较长 memory bundle。没有任何跨 episode 图像或判断进入在线请求。
- 专项离线测试现为 `97 passed`，`py_compile` 和 `git diff --check` 通过。第三次单 episode 待运行；只有 `env.check_success()` 实际成功后才进入同一 init state 的 5 次阶段。

## V2.2 当前执行边界

V2.2 使用独立配置 `configs/robot_libero_clean_qwen3vl_runtime_v22.yaml`，不改变 V1/V2.1 profile。放置阶段的唯一动作权是 VCR runtime；VisualRoute 只提供 rim-plane 路线、mask/footprint/free-space 几何、候选和可视化证据。AnyPlace 只在 GPU1 subprocess shadow 运行，不能排序候选、授权或执行动作。

事务式 option 图为：

```text
TRANSFER → ALIGN_OPENING → DESCEND_TO_SEAT
          → VERIFY_SEATED → OPEN_GRIPPER → RETREAT → VERIFY_TASK
```

当前实际路线仍可表现为：

```text
CLEARANCE → TRANSFER → PRE_DESCENT → DESCENT
          → seating/contact evidence → VERIFY_SEATED
          → (SEATED_HELD: release) / (RIM_CONTACT: clear and fresh re-observe)
```

## 已验证的架构性质

- opening target 使用估计 rim plane，而不是 table/support plane；`destination_xy_world` 是使用锁定 grasp offset 修正后的 EEF target，opening 中心单独保存在 `opening_xy_world`。
- SAM3 opening mask、outer receptacle evidence、held mask 和 point-cloud/active-parallax evidence 进入同一 `PlacementBelief`；mask 只在几何一致时更新，遮挡时可依据外轮廓共运动保持。
- 关系协议只允许 runtime 编译原子动作：`ABOVE_ALIGNED/DESCENDING_CLEAR → MV_DOWN`，`ABOVE_UNALIGNED → signed route correction`，`RIM_CONTACT → clearance`，`UNKNOWN/conflict → STOP` 或一次受控高净空 probe。
- hysteresis 以 footprint containment margin、uncertainty 和 action resolution 生成；同一 candidate/grasp epoch 下不因单帧毫米级噪声随意回退 phase。
- V2.2 不调用 VisualRoute Qwen intent、legacy `review_place_alignment` 或 legacy route gate。placement verifier 只报告关系；它不能直接生成移动 token或覆盖 VCR。
- raw AgentView/Wrist、provider overlay、Qwen input panel 和 UI composite 分开保存，rollout 后必须同时检查原始帧和 overlay 帧。

## 最新真实验证结论

同一 `LIBERO_OBJECT/task 2/init 0` 已连续做了 V2.2 单 episode 调整验证，均未成功：

| run | 结果 | 主要现象 |
|---|---|---|
| `run_6e9a0808aeb8` | `success=False`，step 187 cap | opening mask 漂移到外轮廓之外；错误目标上出现 `RIM_CONTACT → MV_UP/MV_DOWN` 循环 |
| `run_de31988be086` | `success=False`，step 187 cap | opening 共运动后，边界 clipped mask 被 VisualRoute 拒绝；step 139 后 stale/UNKNOWN 持续 STOP |
| `run_32852afad823` | `success=False`，step 187 cap | 已越过 stale/STOP；step 166 进入 `PRE_DESCENT`，观察到 `ABOVE_ALIGNED → MV_DOWN`；step 178/181/184 接触判定仍重复，未 release |

最新 run 的关键证据：step 178 `rim_clearance=-0.00779m`、`containment_margin=+0.04073m`；held/outer bbox 只有约 1 像素边界重叠，下降尚未由 action-effect 模型确认 stall，但 contact candidate 已触发，Qwen 返回 `RIM_CONTACT`。这说明下一步需要收紧 contact evidence gate，而不是调 Qwen 动作 prompt 或增加动作步数。

## Visual Memory 当前真实边界

旧 rollout 的 `visual_memory_refs` 实际为空；pregrasp 只有临时上一帧/当前帧图板，placement 完全没有历史图像。本轮代码已修正该断点：

- V2.2 在请求前持久化 raw AgentView/Wrist，按 instance × grasp epoch 保留关键帧引用及真实 action receipt、EEF 位移、phase、placement summary；最多两个历史节点加当前节点。
- runner 对每个引用检查实例、grasp epoch、帧序和原图路径；失败返回 `UNKNOWN`。Qwen 实际收到的有序图板、hash 与请求元数据进入日志，`qwen_input` 不再指向 provider overlay。
- V2.2 transport 的普通 signed motion由 runtime直接编译，不是每步都让 Qwen看memory。
- `VERIFY_SEATED` 现在支持 `off/single/double`。`double` 的第一阶段只见原始时序图与实际动作效果，第二阶段使用同一图板、场景描述与新鲜几何证据；两阶段之间无环境动作。默认 profile 暂为 `single`，因为三组冻结帧中 `double` 没有优于 `single`。

离线回放还显示，旧 prompt 预填的 `RIM_CONTACT` 会影响 Qwen：`contact_candidate=true` 而无 stall 的帧在近似旧版输入下至少一次复现了 contact 答案；去掉预填关系后同帧输出“仍在上方”。这只是输入诊断，尚不能证明 visual memory 提升在线成功率。

## 已完成的离线改动与下一关口

以下接口、门槛与离线回放已完成，接下来需要一个真实单 episode 验证闭环：

1. 低频 placement memory bundle 在进入 seating、首次 contact risk/confirmed contact、clear recovery 或动作效果异常时采样，按事件和 route epoch 去重。
2. 两次反思已作为可配置实验实现；离线三组相同帧比较了近似旧版、去预填关系的单帧、单次带记忆、双次带记忆。双次额外调用未带来更好的关系答案，暂保留而不在线默认启用。
3. Qwen仍不拥有 `MV_LEFT/MV_RIGHT/MV_FWD/MV_BACK/MV_UP/MV_DOWN` 的最终权；runtime根据三维残差、动作效果和 evidence gate编译实际动作，release 仍必须经过 seating chain。
4. `RIM_CONTACT` 现在只由实际下降 stall 升级；单个像素级 bbox overlap 仅触发关键节点复核。Qwen 的 seated 答案也无法越过独立 release gate。

## 0928 Show-Harness 重建候选（独立隔离）

当前另在 `/root/autodl-tmp/Show-Harness-Rebuild-0928` 从 0928 检查点恢复一条接近原 Show-Harness 的候选主线，排查近期 runtime 改造偏离 Harness 的风险。候选不会覆盖本主工作树或本文件已有的 V2.2 实验记录。架构继续遵循：双视角/本体观测 → SAM3 与跨帧身份跟踪、episode visual memory、相机几何及 VisualRoute → Thinking 语义判断 → Harness verified runtime/guard 授权 → 单原子动作 → 新观测与 logger。RSI 只从失败轨迹提出待 replay 验证的抽象经验；候选经验在验证前不影响策略。

候选双轮 Thinking 只在关键阶段进入、闭爪后持握检查、route 首次下降、无进展/证据冲突及放手前触发。两轮共享完全相同的原图和 memory，第一轮提取带帧号的结构化事实，第二轮从 runtime 有限选项中选择；reasoning 文本不传给 runtime，也不能构成新物理证据。事件锁存抑制同一无进展事件逐帧重复调用。

当前候选尚未接入 MoGe 在线推理，保留 SAM3、跨帧 memory、CameraGeometry/active motion 反馈等空间与视觉证据源。单 GPU（RTX 4080 SUPER 32 GB）同时运行 Qwen Thinking 和 SAM3 约占 28.8 GB；是否补 MoGe 由首条真实 episode 的空间证据缺口决定，若必要则设计关键节点顺序调度或卸载。偏离图中 MoGe 子模块是暂时的硬件与证据驱动安排，不改变“空间工具供证据、runtime 授权”的架构。

下一关口：确认 8002 Thinking 与 8773 SAM3 服务后，启动单条 task 2/init 0/seed 0 rollout；保存原始双视角、输入哈希、模型最终 JSON/耗时、runtime 请求与授权、真实动作效果、视频和 `env.check_success()`。先找最早错误证据，不以中间阶段或模型自述记作成功。

## 实验门槛与约束

- 先同一 init state 做单 episode，必须看到 `ABOVE_ALIGNED → MV_DOWN → fresh belief`，随后完成 `SEATED_HELD → VERIFY_SEATED → OPEN_GRIPPER`；最终成功只能来自 `env.check_success()`。
- 同一 init state 之后做 5 次，完整成功至少 3/5；不得有未验证 release，不得有超过 4 步的左右或升降振荡。
- 再扩展到 salad dressing、alphabet soup、milk 的配对 init states；开发对象之外不能新增名称、坐标、高度或尺寸规则。
- provider 过期、实例不符、MoGe/parallax 冲突时动作数必须为 0；高净空 probe 也必须有已验证 held 和事务 phase 前置条件。
- 正式 profile 不读取模拟器对象 pose、渲染深度或对象 ID；oracle 仅用于离线诊断标签和评测。
- V1 保持原样；AnyPlace 在 top-k 可行率和 false-ready 门槛满足前保持 shadow-only。


## 0928 候选：目标身份与 512×512 输入验证

当前错误线索指向双视角有效像素不足：旧服务请求审计实际为 256×256；SAM3 对多个相似瓶子弃权，Qwen 却从模糊画面自行推断“红帽”。候选已统一把模拟器相机渲染、预处理和 SAM3/Qwen 图像输入提高到 512×512。颜色、对象名和位置不成为新的硬编码目标判据。

逐条 rollout 检查：

1. 从保存的 AgentView/Wrist 原图、SAM3 `request_image.shape/hash` 和 Qwen image manifest 核对实际输入为 512×512。
2. 对 SAM3 弃权或跨视角冲突，只允许 bounded viewpoint probing/STOP；不允许下降或闭爪。身份与目标关系变为新鲜、可见、无冲突证据后，runtime 才开放当前几何所支持的纠偏与对齐抓取选项。
3. 记录 Qwen 选择的对象线索、SAM 候选、跨帧身份和实际抓取对象的画面证据，按首个错误分叉修复；不使用任务专用颜色、坐标或尺寸规则。
4. 每次一条 episode。完整验收仍须经历真实抓取、持握验证、搬运、落座、放手，并得到 `env.check_success() == true`；随后才进行同初态重复和跨物体检查。


## 高分辨率身份连续性复跑

最新单条高分辨率诊断只走 5 个动作，夹爪仍 OPEN；模型两轮三图哈希一致、实际图像尺寸 512×512，`check_success=null`。首帧 SAM query 对相似瓶子的辨识结果随措辞变化，随后同类别重检测变成模糊候选；这证明升分辨率改善了可见像素，但还需 episode tracker 固定目标身份。候选已改用 planner target 的自然组合 query、Wrist→AgentView fallback、同 episode 首次无歧义 bbox 锚点和 tracker continuity。目标选择遵循用户“不是红色那个”的纠正，颜色本身不作判据。

后续单条复跑要证明：

1. SAM 新鲜目标候选与保存的 AgentView frame hash/shape 匹配，tracker 歧义刷新仍围绕初始语义目标实例。
2. Qwen 双轮只引用当前 episode 可见证据，语言目标、可读标签和工具锚点一致；矛盾/过期即 UNKNOWN。
3. 目标未解析时无抓取/下降授权；解析后每个原子动作受 runtime 有限集合和安全 guard 检查。
4. 观察完成抓取、持握、搬运、落座、放手和最终 `env.check_success() == true` 后，才进入复跑与另一物体验证。

## 2026-10-01: SAM3 服务恢复后的单条复跑

实验仍固定在 f235aea 隔离副本和 LIBERO_OBJECT task 2 / init 0 / seed 0。上一条 run_a0fa49d4506f 只作服务配置诊断：SAM3 启动缺少本地 checkpoint 环境变量，远程下载失败；summary 为 interrupted、check_success=null，不作为有效策略结果。

SAM3 必须按本地模型启动：
OPENETA_SAM3_CHECKPOINT_PATH=/root/autodl-tmp/openeta-services/models/sam3/sam3.pt
对已保存的 512×512 起始帧 smoke 已确认 MCP 服务成功并产生两个目标候选。此后才允许启动单条真实 rollout。目标身份继续由 task 语义、原始双视角、SAM3 候选及当前 episode tracker 共同判断；模型回答和颜色不能单独授权抓取。下次分析最早错误分叉，并核对完整动作回执；不得把中间阶段或中断计为成功。

## 2026-10-01: 目标身份查询的修复路线

run_061d133d4a4a 的首错已定位到 SAM3 查询回退：512×512 AgentView 对任务目标和右侧相似候选给出可区分的检测；Wrist 对完整任务语义弃权，但旧查询回退丢掉 salad dressing 目标短语，只用位置/affordance 属性加 bottle，命中邻近容器。主 tracker 因而绑定错误候选。grasp guard 阻止了闭爪，runtime 的横向对齐没有被 identity uncertainty 一起限制。此 run 在 gripper OPEN 时于 25 steps 中断，check_success=null。

已将所有 SAM3 fallback 改为保留完整任务目标；当 Wrist 不能提供语义候选时，由同帧 AgentView 提供全局身份和实例跟踪锚点。未引入目标颜色规则。33 项定向测试通过，py_compile 与 diff whitespace 检查通过。

下一条仍用同一 512×512 配置和 task 2 / init 0 / seed 0，仅跑一条。先验证 fallback 日志、bbox 与原帧对应同一语义候选，再看 runtime 所授权动作是否使目标/末端关系按预测变化。若目标身份、图像和回执不一致，停止闭爪并从那一帧修复；完整成功只按 env.check_success() 计。

## 2026-10-01: 修复后离线帧核验完成

已在当前 SAM3 服务上回放 run_061d133d4a4a 的 frame 0；Wrist 保留目标短语查询弃权后，AgentView 返回两个候选并将分差足够的首选实例锚定到同 episode。输入尺寸、请求 hash、bbox 和候选分数均有记录。现在可以启动同一 task/init/seed 的单条复跑；首帧仍需复核，后续只按实际观测和授权/执行回执判断。

## 2026-10-01: 多物体下的跨视角身份规则

run_afc99c8bab61 证明，即使 SAM3 的所有 Wrist fallback query 都保留完整 task phrase，Wrist 仍可能将高分异物当作目标。AgentView 在 frame 0 已正确建立目标轨迹，但旧 probe 在 frame 8 因 Wrist 有检测而把它设为主视角。该 run 于 24 个记录动作后中断，夹爪 OPEN、check_success=null。

GRASP 与 RELEASE 现在在配置启用时始终以 AgentView 维护全局身份和空间关系。Wrist mask/框/分数继续作为局部 secondary evidence 记录，不能替代 AgentView 主 bbox。已用 run_afc 的真实双视角 frame 0/8、SAM3 服务和当前 tracker 回放确认主身份保持在同一实例，Wrist 的另一候选留在 secondary。33 项定向测试通过。

下一条同初态单 episode 复跑从 AgentView 身份稳定性开始验收；检查 runtime 使用 AgentView 的新鲜几何证据，Wrist 的独立候选不进入全局 target offset，close guard 仍在身份和对齐满足条件前拒绝闭爪。

## 2026-10-01: 当前收敛路线

1. 继续 0928 隔离副本；每轮先读 `AGENTS.md`，保留上述调整前快照和所有诊断视频。最新 run_4d357a5c0a64 为中断、`check_success=null`，成功数仍为 0。
2. 审计并收敛 Qwen、verified runtime、抓取 guard、恢复、VisualRoute 对动作的多重改写；运行时统一产生可追踪的请求、授权、执行和动作后证据。动作有真实进展时不因重复方向而触发双次反思或提前中断。
3. 原生 768 输入先做单卡三图服务测试；像素阈值按分辨率统一换算，若 16K 上下文/显存不能承受，先压缩文本，再降到原生 640。AgentView 保持全局身份，Wrist 只给独立局部证据；MoGe 健康验证前只作为 shadow。
4. 将两轮结构化反思的 8192 总生成上限真实传到模型请求；按真实闭爪、持握、route 下降、实测无进展/冲突、放手疑点触发。失败时记录是哪轮、finish_reason、usage、请求 hash，保持 UNKNOWN 并有界取证。
5. 每次只跑一条完整 episode；根据首个证据/决策分叉修通用问题，并从失败记录提出 replay-gated 的抽象 harness 经验。首个成功仅以 `env.check_success() == true` 计，随后同初态重复和另一物体检查。

## 2026-10-01: 动作尺度与服务关口

隔离单轴标定表明 LIBERO OSC_POSE 的重复相对目标不能用 RoboLab 的“总位移除以控制步数”接口：0.005 m 命令四次只走 0.004302 m，0.020 m 命令四次走 0.018897 m。已在 LIBERO 专属原子适配器修正，并保留动作后实测回执；这是使目标身份、接近决策和阶段推进连成闭环的首要通用修复。当前完成 23 项相关离线测试，8002 Thinking 与 8773 SAM3 模型身份/本地权重确认；768 三图递归服务验证待完成，随后启动一条同初态完整 episode。

## 2026-10-01: 首个真实闭爪后的恢复方向

run_7ad374fb96ab 在原生 768 输入下成功把同一目标水平残差 95→7 px、纵向残差 226→31 px，并真实执行一次语义 GRASP；但上抬后目标留在桌上，未进入 TRANSPORT。根因优先级：接近闭爪时 SAM3 新鲜度与局部 Wrist 几何未进入最终闭爪门；闭爪后的双轮事实提取用回执替代视觉；固定宽度 recovery 反复提出 RELEASE，被最后授权点挡成 STOP。下一次代码修复聚焦这些证据与授权接口，保留当前已验证的目标身份和控制尺度。run 在 43 步诊断中断，check_success=null，成功数仍 0。

## 2026-10-01: 歧义身份与持握动作效果

最新 run_dd9260919c6e 的首错前移到 SAM3 近分候选：未建立 AgentView 主实例时，连续方向移动没有增加身份信息。针对该条件，保留 768 校准的双视角控制输入，额外使用 1536 原生渲染的当前帧裁出两张候选细节图；Thinking 对同一冻结三图先列可见包装事实，再选 SAM3 已给出的候选索引。结果若缺失、截断、图像 hash 不一致或未选在候选集合内，仍为 UNKNOWN；此过程不输出动作。静态 1536 隔离渲染已完成，冻结身份回放和接线测试继续进行。

闭爪后的有界上抬增加固定 AgentView mask 底边与已执行 EEF 投影位移的对比。这个动作效果只可反驳或支持 Thinking 的持握判断，不能单独把模型回答当作任务成功。下一条真实运行先核对身份选择与原帧，随后检查抓取前双视角、实际闭爪、抬手共运动和阶段推进；仍只按完整 episode 的 check_success 验收。

镜像文字诊断进一步收敛了身份接口：1536 原生帧只供局部候选裁图，两个候选水平翻转并排编号后形成同一张模型输入图；两次短 JSON 请求重复看这张完全相同的图。控制所用 768 AgentView/Wrist、SAM3 框和标定始终保持原方向。冻结帧的两轮候选标签/索引与原图吻合，下一条真实运行先核对在线候选与输入对照图的实例、帧号和 hash。

## 2026-10-01: plan02

最新运行把首错收敛到近接触连续前进导致目标姿态变化，随后旧框误报进展、GRASP 阶段实例跳到另一瓶以及重复 STOP。保留 plan01 的 Show-Harness 主线；先扩展实例关联与进展有效性，再接近接触双视角审查、5 mm 单步探测，以及同 episode visual memory + SAM3 点提示的变姿态重识别。每次真实运行仅一条，仍以完整 `env.check_success()==true` 为首次验收。

当前隔离副本已完成近接触 Wrist 局部关系门与变姿态视觉记忆重识别的冻结复现。单条 task2/init0/seed0 真实复跑进行中；结束后先核对原帧、模型输入、请求/授权/执行与动作效果，定位最早错误分叉，再修通用原因。闭爪、持握、搬运、落座、放手只按真实执行和观测记录；没有完整 `check_success()==true` 前成功数为 0。

`run_d86e510a71d2` 首步先发现语义角色接线错误：SAM3 候选存在，runner 取错控制器层级，使身份裁决与多个视觉核验方法不可用。已统一指向 `controls.controller.agent`，入口契约测试通过。下一条首先核对两轮身份选择是否进入真实 capability evidence，再继续检查近接触与闭爪；此诊断条 `check_success=null`，成功数仍 0。

## 2026-10-01: plan03 收敛路线

最新 `run_7bf8267e26bb` 的最早未解决分叉是局部可见部件无法与全局目标关联，导致近接触时反复 `UNKNOWN`。保存 0928 代码检查点后，仍保留 Show-Harness 感知→belief→语义关键决策→runtime 授权→原子动作→重新观察链；将视觉记忆改成对象与事件索引，引入受验证的部件候选和安全主动取证，在线用实际回执校准短视野动作效果。冻结帧和错误物体反例通过后一次只跑一条完整 episode；只有 `env.check_success()==true` 计成功。
