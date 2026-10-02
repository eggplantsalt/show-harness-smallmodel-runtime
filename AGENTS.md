本文件是本项目长期有效的最高优先级工程与科研约束。开始任何任务前先阅读本文件，再阅读项目当前状态文档和当前 milestone 文档。

项目当前目标不是“把某一个 LIBERO case 跑通”，而是建立一个：

**Frozen Small Agent / VLM + Generalizable Verified Runtime**

系统。

核心研究目标是：让小模型只负责其擅长的语义决策，把确定性的、可测量的、可验证的物理决策交给 Runtime；最终系统必须能够跨物体、跨布局、跨任务迁移，而不是针对单一场景写死。

---

# 1. 总体工作原则

不要乱发散。

先完成当前 milestone，再进入下一阶段。

不要看到一个 failure 就立即添加一个新模块、新 heuristic、新状态机或新的模型。

出现问题时先判断 failure 属于哪一层：

- perception
- semantic grounding
- entity identity
- geometry
- scene initialization
- physical execution
- effect verification
- semantic decision
- skill contract

然后修对应层的通用 contract。

禁止：

“这个 case 不工作 → 针对这个 case 写特殊规则”。

优先建立可复用、可解释、可验证的机制。

---

# 2. 当前项目的核心原则

系统责任必须明确分开。

## Agent / Qwen 负责

- 理解自然语言任务
- semantic task reasoning
- entity / role binding
- ordinal / relational reasoning
- 例如判断：
  - 哪个是“第二个抽屉”
  - 哪个物体是 manipuland
  - 哪个物体是 destination
- 在多个 physically valid semantic options 中进行选择，例如：
  - ALIGN
  - REOBSERVE
  - GRASP
  - PLACE
  - ARTICULATE
  - ABORT

## Runtime 负责

- perception state organization
- camera geometry
- object-relative geometry
- workspace validity
- physical direction
- displacement scale
- controller realization
- bounded execution
- physical preconditions
- effect verification
- scene readiness
- robot readiness
- deterministic physical reasoning

Qwen / Agent 不应直接负责：

- LEFT / RIGHT
- FWD / BACK
- UP / DOWN
- 3 mm / 6 mm / 9 mm
- controller tick 数
- raw motor command
- workspace boundary
- physical effect checking

核心原则：

**能被可靠的传统几何、传感器、proprioception、controller feedback 或 Runtime verification 确定的问题，不要再交给小模型猜。**

---

# 3. 最高优先级硬约束：禁止使用仿真真值完成任务

这一条是项目最重要的约束。

一句话：

**Simulator ground truth may grade the system, but may never answer the task for it.**

正式 Runtime 决策路径禁止使用任何真实部署时拿不到的 simulator privileged information。

包括但不限于：

- `sim.data.xpos[target]`
- object body pose
- target object world pose
- body ID 用于目标定位
- simulator object center
- object joint GT
- drawer joint GT
- simulator contact GT
- task success flag
- hidden scene graph
- simulator instance segmentation GT
- simulator semantic segmentation GT
- simulator object dimensions
- simulator ground-truth depth
- 任何只能从 MuJoCo / LIBERO 内部直接读取的任务真值

这些信息：

**只允许 diagnostic / evaluation 使用。**

可以用于：

- 判断实验真实发生了什么
- failure attribution
- 画曲线
- 计算误差
- benchmark evaluation
- 与正式估计结果做离线对比

绝对不能影响：

- Runtime decision state
- option generation
- candidate ranking
- NearTarget 判定
- grasp 判定
- placement 判定
- Arbiter
- Executor
- effect verification
- Runtime termination
- Qwen prompt
- recovery decision

严禁：

先读取 oracle，再偷偷转换成一个“普通变量”送进 Runtime。

这仍然属于 oracle 泄漏。

---

# 4. 正式 Runtime 允许使用的信息

任何正式 Runtime 输入，都必须能回答：

“换成真实机器人后，这个信息从哪里来？”

允许：

- RGB camera
- wrist RGB camera
- robot proprioception
- EEF pose
- joint state
- gripper state
- camera intrinsics
- camera extrinsics
- 真实可标定参数
- deployable perception model output
- SAM / segmentation output
- frozen VLM output
- deployable monocular depth estimator output
- 如果未来明确使用真实 RGB-D camera，则允许真实 sensor depth
- 如果未来真实机器人有 force / torque sensor，则允许对应真实传感器输入

不允许因为 simulator 有更方便的信息就直接读取。

---

# 5. Depth / 3D 信息规则

Depth 可以成为正式方法的一部分，但：

**MuJoCo / LIBERO ground-truth depth 不能成为正式 Runtime 的输入。**

正式 Runtime 如果使用 depth，应通过可部署接口获得，例如：

- monocular metric depth model
- real RGB-D camera
- multi-view geometry
- triangulation
- calibrated physical geometry

建议保持统一 `DepthProvider` / geometry provider 接口。

Simulator depth 只能作为：

`DiagnosticSimulatorDepthProvider`

用于 upper bound 和误差分析。

禁止：

- 用 simulator depth 控制机器人
- 用 simulator depth 校准 monocular depth 的 scale 后再参与同一实验
- 用 object GT pose 修正 estimated depth
- oracle correction

如果正式 deployable depth 不可靠，应报告 blocker，而不是偷偷退回 GT depth。

---

# 6. 通用性 / 泛化是硬门槛

本项目不是：

“salad dressing controller”。

任何新增逻辑必须问：

**如果 target 从 salad dressing 换成 apple / bowl / bottle / drawer handle，这段代码是否仍然成立？**

禁止：

```python
if task_id == 2:
    ...

if object_name == "salad dressing":
    ...

if "ketchup" in instruction:
    move_right(...)
```

禁止：

- 固定 object xyz
- 固定 pixel coordinate
- 固定 direction sequence
- per-object motion script
- per-init-state hack
- per-task physical policy
- 针对当前物体写死 grasp width
- 针对当前任务写死 placement point

Runtime Core 必须尽量：

- object-agnostic
- scene-agnostic
- task-agnostic

允许 task / object 作为：

semantic data

进入 Entity / Task representation。

但不能作为：

physical control branch。

---

# 7. 通用 Runtime 表示目标

长期目标不是让 Runtime 理解“salad dressing”这个字符串。

应该逐步形成：

## RuntimeEntity

包含类似：

- identity
- semantic label
- role
- visual evidence
- metric / relational geometry
- affordances
- relations
- state

例如：

apple：

- role = MANIPULAND
- affordance = GRASPABLE

basket：

- role = DESTINATION
- affordance = CONTAINER

plate：

- affordance = SUPPORT_SURFACE

drawer：

- affordance = ARTICULATED

drawer handle：

- affordance = GRASPABLE_HANDLE

Runtime 应处理：

Entity / relation / affordance

而不是具体 object name。

---

# 8. Task 表示原则

长期不要让 Runtime core 直接依赖 LIBERO task ID。

自然语言任务应逐渐转换成：

- Entities
- Roles
- Goal Relations

例如：

“put apple in basket”

应表示为：

- A = MANIPULAND
- B = DESTINATION
- Goal = INSIDE(A, B)

“put bowl on plate”

应表示为：

- A = MANIPULAND
- B = SUPPORT
- Goal = ON(A, B)

“open the second drawer”

应表示为：

- C = ARTICULATED_PART
- handle = associated handle
- Goal = OPEN(C)

不要一次构建庞大 symbolic planner。

优先建立最小、清晰、可验证的 TaskSpec / GoalSpec。

---

# 9. Verified Option 原则

Runtime Option 不能只是一个字符串。

理想结构：

**Semantic Intent + Preconditions + Physical Realization + Expected Effect + Verification**

例如：

`ALIGN_TO_TARGET`

应该包含：

- target identity valid
- target geometry valid
- workspace valid
- physical realization
- bounded movement
- expected effect
- post-action verification

Qwen 最终看到的是：

- ALIGN
- REOBSERVE
- GRASP
- PLACE
- ARTICULATE
- ABORT

Qwen 不应该看到：

- DOWN 9mm
- RIGHT 6mm
- controller tick

这些属于 Runtime physical realization。

---

# 10. Single Authority 硬规则

继续维持：

**Arbiter = 唯一 action authorizer**

**Executor = 唯一 robot execution owner**

禁止：

- Runner 直接执行物理 `env.step`
- Perception 执行动作
- Geometry 执行动作
- OptionGenerator 执行动作
- Qwen 直接执行 controller
- Recovery 绕过 Arbiter
- 一个 Arbiter approval 执行多个 semantic actions

原则：

一个 semantic physical option
=
一个新的 Arbiter approval。

Executor 在一个 approval 内：

- 不允许 replan
- 不允许换 direction
- 不允许换 scale
- 不允许自己 recovery

需要改变动作：

返回 Runtime decision loop。

---

# 11. 不要重新引入旧 Show-Harness 的多 Authority 结构

Legacy 代码保留用于参考。

禁止重新引入：

- VisualRoute takeover
- VerifiedRuntime takeover
- legacy Recovery direct control
- recursive reflection control
- RSI direct control
- 多个模块分别 veto / rewrite / execute action

如果复用 legacy：

只复用：

- pure helper
- low-level capability
- environment adapter
- camera geometry
- perception client

不得重新导入旧 policy 架构。

---

# 12. Legacy Isolation

冻结：

`legacy-full-harness-0928`

以及 legacy runner / policy。

除非用户明确要求：

不要修改 legacy。

当前开发主线：

`runtime-v3`

如果当前实际 branch 与这里不同：

先确认再继续。

---

# 13. 不允许为了成功偷偷调整实验

禁止：

- seed cherry-picking
- init-state cherry-picking
- task cherry-picking
- baseline 故意调弱
- 失败后不断改 threshold 直到成功但不报告
- 删除失败 rollout
- 只汇报最好的一次
- 用 oracle 结果修改同一次实验

每次正式实验必须记录：

- task
- seed
- init state
- parameters
- code commit
- success / failure
- termination reason

如果第一次失败：

保留结果。

---

# 14. Threshold / 常数原则

不要把某个当前场景观察到的固定数字直接升级成通用规则。

例如：

- pixel threshold
- gripper width
- object size
- bbox size
- mask area
- depth
- contact distance

如果必须使用常量：

1. 集中配置；
2. 说明来源；
3. 说明适用范围；
4. 说明失效边界；
5. 尽量使用 normalized / relative quantity；
6. 后续必须跨 state / object 验证。

禁止：

“这个 object 是 0.03，所以所有 object 都按 0.03”。

---

# 15. 抓取成功 / Holding 判断

不要通过一个固定 gripper width 就宣布：

GRASP SUCCESS。

正式 grasp / holding verification 应综合：

- gripper proprioception
- relative gripper motion
- visual evidence
- target follows EEF
- small lift evidence
- temporal stability
- deployable signals

Simulator contact / attachment GT：

只能 diagnostic。

Agent / VLM 可以参与高层 semantic judgement，但不能替代可验证 physical evidence。

---

# 16. 场景和机器人初始化必须分开

RobotReady 和 SceneReady 是不同概念。

RobotReady：

机器人 reset transient 结束。

SceneReady：

场景目标 / 环境物体稳定。

不要假设：

机器人稳定 = 场景稳定。

不要通过 fixed sleep 替代可验证条件，除非有充分实验依据。

---

# 17. 图像规则

当前正式视觉输入必须来自：

真实 source render / sensor resolution。

禁止：

224/256 图片 resize 到 768 然后声称是高分辨率。

如果提高 resolution：

必须从 renderer / sensor 源头提高。

图像方向必须统一经过 Canonical Image pipeline。

注意 OpenGL / MuJoCo 常见：

vertical flip / origin convention。

SAM、Qwen、overlay、geometry projection 必须使用同一个 pixel convention。

不能：

模型看到正图，但 geometry 使用倒图坐标。

---

# 18. 视频 / Rollout 调试

每次 rollout 如果有视频 / frame artifact：

调试时应实际查看关键帧。

不要只看终端数字。

至少检查：

- before
- selected action
- after
- mask
- overlay
- target identity
- EEF projection
- unexpected contact
- scene movement

视觉证据和日志必须对应。

---

# 19. 跨任务验证门槛

不要长期只在一个 task 上优化。

新 physical contract：

建议顺序：

1. 3 states 快速验证
2. 6 states
3. cross-object
4. cross-scene
5. cross-task

当 ALIGN + GRASP 基本工作以后：

必须尽快进行第一次：

**zero-control-code-change cross-object transfer**

硬要求：

Runtime Core：
0 code change

ALIGN contract：
0 code change

GRASP contract：
0 code change

只允许改变：

- instruction
- semantic entity binding
- observations

如果必须新增：

`special_case_for_object_x`

则记录：

GENERALIZATION FAILURE。

---

# 20. Skill-specific 不等于 Task-specific

允许存在 reusable physical contracts：

- ALIGN
- GRASP
- PLACE_IN
- PLACE_ON
- ARTICULATE
- PRESS
- REOBSERVE

例如 drawer task 可以需要：

ARTICULATE contract。

这不等于 task-specific。

禁止：

`OPEN_SECOND_DRAWER_TASK_17`

这种 task-specific skill。

Runtime Core 应保持不变。

---

# 21. 当前研究主线

当前主线不是“让 Qwen 学会更多物理动作”。

而是：

**Responsibility Factorization**

也就是：

Runtime 吸收确定性的 physical decision；

Frozen compact Qwen 保留 bounded semantic decision。

任何新设计都应问：

这个决策到底应该：

Runtime owned

还是：

Qwen owned？

不要因为“模型能做”就默认应该让模型做。

---

# 22. Qwen 接入规则

正式 Qwen semantic routing 只在 Runtime 已经生成 physically valid semantic options 后发生。

Qwen 输入应尽量是：

- task instruction
- compact Runtime state
- relevant visual evidence
- available semantic options

输出：

- existing option ID
- REOBSERVE
- ABORT

Invalid output：

fail closed。

禁止：

Qwen 输出 raw motor commands。

禁止依赖：

CoT
reflection
retry-until-valid

来掩盖 selector 不稳定。

---

# 23. 不要过早加入 Memory / RSI / Self-improvement

当前 physical Runtime / generalization 没有稳定前：

不要加：

- memory
- reflection
- RSI
- recursive self-improvement
- experience prompt accumulation

只有跨任务 Runtime 基本成立以后才考虑。

未来 experience 应优先影响：

- option construction
- candidate validity
- verification confidence
- reusable failure knowledge

而不是简单给 Qwen 堆历史文字。

---

# 24. 调试顺序

出现 failure 时优先按以下顺序定位：

1. Observation 是否正确
2. image orientation / resolution 是否正确
3. target grounding 是否正确
4. identity 是否正确
5. scene 是否稳定
6. metric geometry 是否正确
7. candidate prediction 是否正确
8. physical motion 是否实现
9. effect verification 是否正确
10. semantic decision 是否正确

不要直接：

“失败 → 调 prompt”。

---

# 25. 新模块准入原则

只有当现有模块无法表达一个真正独立的职责时，才新增模块。

新增前先回答：

- 这个问题属于哪一层？
- 当前 module 为什么无法解决？
- 新模块是否具有跨任务意义？
- 是否只是给当前 failure 打补丁？

禁止无限扩张：

manager
planner
recovery
reviewer
critic
reflector
judge

等角色。

保持 Runtime 简洁。

---

# 26. 状态机原则

不是绝对禁止状态机。

但是：

不要因为 agent 不稳定就用大量 task-specific FSM 把完整任务写死。

允许：

少量通用 physical stage / contract lifecycle。

例如：

READY
ALIGNING
NEAR
HOLDING

如果它们具有跨任务物理意义。

禁止：

TASK2_STEP1
TASK2_STEP2
TASK2_STEP3

这种脚本化状态机。

---

# 27. 环境 / 依赖安装规则

安装依赖优先按照：

官方 README
官方 requirements
官方 installation guide

不要自作聪明随意删减依赖。

pip：

禁止走 git clone 学术代理。

如果 pip 慢：

优先尝试清华源等稳定镜像。

如果依赖编译失败：

可以：

- 查官方 issue
- 查 wheel
- clone 对应 dependency
- 本地安装 wheel
- 根据错误自行诊断

但不要破坏当前已工作的主环境。

尽量使用独立 venv / conda env。

---

# 28. Git clone 网络规则

只有：

`git clone`

需要时才使用学术加速。

AutoDL 场景可使用：

```bash
source /etc/network_turbo
```

clone 完成后必须立即：

```bash
unset http_proxy
unset https_proxy
```

如果当前服务器项目提供了专用代理脚本：

优先按照项目已有脚本使用。

例如存在：

`proxy_on.sh`
`proxy_off.sh`

则：

clone 前开，
clone 后立即关。

禁止长期保持 proxy 开启。

pip 时必须确保代理关闭。

---

# 29. Hugging Face 下载

Hugging Face 下载优先：

- 检查本地 cache
- 检查已有模型目录
- 使用项目当前 Hugging Face mirror

不要重复下载几十 GB 权重。

下载前：

先确认模型是否已经存在。

---

# 30. 长时间任务规则

涉及：

- model loading
- vLLM startup
- rollout
- large inference
- long preprocessing
- dependency compilation
- large download

不要每几秒频繁查询终端输出浪费 token。

根据任务实际时长：

约 1 分钟
3 分钟
5 分钟

再检查。

不要因为暂时没有输出就误判卡死。

但是如果：

GPU / CPU / process 明显异常

再主动诊断。

---

# 31. 不要滥用查询命令

不要重复运行：

`git status`
`git log`
`pwd`
`ls`

浪费上下文。

Git 状态通常只需要：

- 开始 major milestone
- commit 前
- push 后
- 出现 repository 异常时

查询应该有明确目的。

---

# 32. 计划允许根据证据更新

开始任务前：

先给自己形成清晰 Plan。

执行过程中如果实验事实证明 Plan 已经过时：

允许调整。

但是：

不要随意发散。

调整前明确：

- 原 Plan 哪个假设被推翻
- 新证据是什么
- 为什么新方案更合理
- 是否影响整体研究路线

优先局部修正。

---

# 33. 文档维护

所有项目长期文档位于：

`/root/autodl-tmp/Show-Harness/docs`

Runtime V3 文档主要位于：

`docs/runtime_v3/`

必须持续维护至少：

- `STATUS.md`
- `ARCHITECTURE.md`
- `RESPONSIBILITY.md`
- `OBSERVATION_BOUNDARY.md`
- `GENERALIZATION.md`

以及当前 milestone 对应文档。

至少维护两个层级：

## 全局路线

记录：

- 项目最终目标
- architecture
- milestone roadmap
- current research hypothesis

## 当前进度

记录：

- 最新 commit
- 当前已经验证的能力
- 最新实验
- blocker
- 当前下一步

这样换 session 后 Agent 可以先读文档恢复上下文。

---

# 34. 每个 milestone 的标准流程

尽量遵循：

1. 阅读 AGENTS.md
2. 阅读最新 STATUS / milestone 文档
3. 阅读真实相关代码
4. 明确当前 baseline
5. 写最小实现
6. unit test
7. integration test
8. 小规模真实 rollout
9. 分析 artifact
10. 扩大必要实验
11. 更新文档
12. commit
13. push

不要：

先大规模跑几十个 rollout 再看代码对不对。

---

# 35. 实验规模纪律

遵循：

**1–3 天快速验证信号。**

不要一开始：

50 tasks × 10 seeds。

先：

最小实验
→ 找到 signal
→ 确认机制
→ 再扩展。

但是不能：

永远只在一个 case 上。

---

# 36. 真实世界迁移检查

每新增一个正式 Runtime input / module，都必须回答：

“真机上这个东西从哪里来？”

例如：

RGB：
真实相机

EEF pose：
机器人 proprioception

camera intrinsics：
camera calibration

camera extrinsics：
hand-eye / fixed-camera calibration

depth：
monocular depth model 或真实 RGB-D sensor

如果无法给出真实来源：

它不能进入正式 Runtime。

---

# 37. Diagnostic 和 Formal Runtime 必须隔离

推荐形成明确概念：

## Formal Runtime

决定机器人行为。

只使用 deployable information。

## Diagnostic Oracle

只在实验后评价 Runtime。

允许使用 simulator internal truth。

两者代码依赖必须尽量隔离。

Diagnostic output：

不能反向影响正式 Runtime。

---

# 38. Failure attribution

尽量统一 failure taxonomy：

- PERCEPTION_FAILURE
- IDENTITY_FAILURE
- SCENE_NOT_READY
- METRIC_GEOMETRY_FAILURE
- NO_VALID_OPTION
- ARBITER_REJECTED
- PHYSICAL_EXECUTION_FAILURE
- EFFECT_VERIFICATION_FAILURE
- SEMANTIC_DECISION_FAILURE
- GRASP_FAILURE
- HOLDING_FAILURE
- PLACEMENT_FAILURE

不要只记录：

`failed = True`。

---

# 39. 代码风格

优先：

- small modules
- clear dataclasses
- explicit ownership
- typed data
- simple interfaces
- traceable logs

避免：

- 巨大单文件
- hidden state
- implicit side effects
- 多层 callback
- 巨型 plugin registry
- 不必要 framework

工程目标：

可读
可审计
可复现。

---

# 40. 测试中的重要不变量

持续确保：

1. Formal Runtime 无法读取 simulator target pose
2. Formal Runtime 无法读取 simulator GT depth
3. Formal Runtime 无法读取 simulator contact GT
4. Formal Runtime 无法用 task success GT 决策
5. Qwen 无 controller/backend 引用
6. Runner 无 direct robot execution bypass
7. one semantic physical step = one Arbiter approval
8. approval token 不可重复消费
9. Executor 不可 replan
10. Executor 不可改变 approved direction / scale
11. Oracle diagnostics 不可改变 candidate ranking
12. task-specific control branch 不进入 Runtime core

---

# 41. 不要掩耳盗铃

项目目标不是获得一个漂亮 demo 视频。

如果系统：

- 依赖 oracle
- 依赖 task-specific rule
- 只在一个 object 工作
- 只在一个 init state 工作
- 需要人工告诉它 target coordinate
- 需要人工写 movement sequence

即使成功率 100%，也不算项目成功。

相反：

一个真实暴露泛化 failure 的负结果，比一个 oracle-assisted 100% demo 更有价值。

任何时候：

**优先保持实验真实，而不是保持结果漂亮。**

---

# 42. 当前最终评价标准

最终系统应尽量满足：

- Frozen compact Qwen
- 无 embodiment-specific model training，或极少训练且明确说明
- Runtime Core 跨物体复用
- Runtime Core 跨布局复用
- Runtime Core 跨多个 manipulation tasks 复用
- 不使用 simulator privileged information
- Agent 只做 semantic decisions
- Runtime 负责 deterministic physical decisions
- physical action 可验证
- failures 可归因
- 新 object / scene 不需要新 physical control code

对于同一 skill family：

目标是：

**zero Runtime-core code change transfer。**

例如：

salad dressing → basket

换成：

apple → bowl

应该主要改变：

instruction
entity grounding
goal relation

而不是：

control implementation。

---

# 43. 最重要的一句话

整个项目始终遵守：

**不要把模型训练成一个更小的大模型，而是重新设计 Agent 与 Runtime 的责任边界，让小模型只解决真正需要语义智能的问题。**

以及：

**不要让 simulator ground truth 帮系统答题。**

以及：

**不要为了当前任务成功，牺牲跨任务泛化。**


