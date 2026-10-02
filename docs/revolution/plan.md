# Show-Harness V2：Verified Capability Runtime 改革与研究方案

> **版本关系（2026-09-29）**：本文保留 V2 的总体研究契约；当前实现以 [V2.1 Adaptive Capability Harness](plan02.md) 为执行增量。plan02 已获授权在 GPU1 增加可替换的空间能力 provider（MoGe / active parallax / GraspGenX shadow），因此本文后文“线上不增加模型”的旧约束不再适用于 V2.1；V1 仍保持独立可复现。

## 一、结论与目标

当前瓶颈不是 visual prompt 写得不够好，而是系统把小模型最弱的能力——连续空间控制、物理状态判断、失败诊断和长程恢复——全部留给 Qwen8B 临场完成。visual prompt 只改善了信息呈现，没有改变控制闭环。

现有失败暴露出七个系统性问题：

1. **目标没有持久身份**：SAM3 每帧独立选最高分候选，重检测时可以从落下的目标切换到背景中的另一个瓶子。
2. **没有规范化物理状态**：`_reacquire_required` 只是布尔门控，无法表达 `UNKNOWN`、传感器故障、目标歧义、掉落位置和恢复上下文。
3. **Qwen8B承担了伺服控制**：它必须从单帧判断左右、前后、高低；错误动作没有被控制器学习和纠正。
4. **动作不验证效果**：系统执行动作后，没有比较“预期残差变化”和“实际残差变化”，因此错误方向可以反复执行。
5. **恢复等于回退 stage**：物体掉落后重新进入 APPROACH，却仍使用导致失败的感知和决策机制。
6. **过程成功与任务成功混淆**：抓住、稳定持有、进入篮筐、释放后留在篮筐没有独立 verifier。
7. **基础设施故障泄漏为语义判断**：SAM3 Broken Pipe 被解释成 `TARGET_LOST`，Qwen仍继续运动。

目标是在不增加线上模型的前提下，以 **Qwen8B + SAM3 + CPU状态估计/控制器** 构建 `Verified Capability Runtime v2（VCR-v2）`：

- Qwen负责语义目标、初始 option 计划和少数关键歧义决策。
- Runtime负责 option 内的原子动作、状态提交、验证、超时和恢复。
- SAM3负责候选分割，不再负责无状态地决定目标身份。
- 第一工程门槛：当前失败初始状态运行5次，至少3次成功，且至少成功恢复1次注入掉落。
- V1完整保留，V2并行开发；特权状态只允许进入隔离的诊断 profile。

## 二、V2架构与接口

### 2.1 控制链

```text
原始双视角图像 + 机器人本体状态
        ↓
传感器健康检查 → SAM3候选生成 → 跨帧实体关联
        ↓
Canonical Physical Belief（带 TRUE/FALSE/UNKNOWN、时间和证据）
        ↓
Qwen TaskIntent / 关键决策
        ↓
通用 pick-place option graph
        ↓
残差控制器选择一个原子动作
        ↓
执行一次 → 获取新观测 → 验证动作效果 → 提交状态或进入恢复
```

visual prompt 降级为两种用途：

- 向 Qwen展示经过验证的状态、候选编号和关键决策信息。
- 作为日志和人工排障可视化。

它不再是控制闭环的核心，也不得用陈旧或歧义状态覆盖原始图像。

### 2.2 新增公共类型

在独立的 `core/runtime_v2/` 中定义：

| 类型 | 核心字段与语义 |
|---|---|
| `TaskIntent` | `target_descriptor`、`destination_descriptor`、Qwen建议的option序列 |
| `ObservationHealth` | `VALID / OCCLUDED / AMBIGUOUS / STALE / SENSOR_FAULT` |
| `EntityTrack` | 稳定`instance_id`、mask/bbox、外观摘要、运动历史、最近确认帧、关联置信度 |
| `EvidenceValue[T]` | `value`、`TRUE/FALSE/UNKNOWN`、来源、置信度、时间戳 |
| `PhysicalBeliefState` | 目标/容器track、EEF、持有状态、空间关系、当前option、grasp/route epoch |
| `OptionSpec` | 前置条件、不变量、残差、允许动作、成功条件、预算、恢复边 |
| `OptionResult` | `RUNNING / SUCCEEDED / FAILED / NEED_DECISION`及证据 |
| `TransitionVerdict` | `PASS / FAIL / UNKNOWN`、证据、数据新鲜度 |
| `FailureEvent` | 类型化失败码、失败option、首个异常帧、动作历史 |
| `RecoveryContext` | 原目标身份、恢复入口、尝试次数、恢复后应返回的option |
| `CriticalDecisionRequest` | 候选集合、原始双视角、已验证状态、允许选择，不接受自由动作文本 |

状态只能通过 verifier 事务式提交。Qwen输出首先是“假设”，不能直接把 `HELD`、`SEATED` 或 `SUCCESS` 写入状态。

### 2.3 Option graph

首版只覆盖通用单物体 pick-and-place：

```text
LOCATE_TARGET
→ MOVE_TO_HOVER
→ ALIGN_PREGRASP
→ DESCEND_TO_GRASP
→ CLOSE_GRIPPER
→ VERIFY_HOLD
→ LIFT_CLEAR
→ LOCATE_DESTINATION
→ TRANSFER
→ ALIGN_OPENING
→ DESCEND_TO_SEAT
→ VERIFY_SEATED
→ OPEN_GRIPPER
→ RETREAT
→ VERIFY_TASK
```

Qwen选择语义目标和高层option；编译器把它转成以上通用图。Runtime验证前置条件并在option内部选择原子动作。

首版不扩展到抽屉、开门等其他技能；先将 LIBERO_OBJECT 的完整闭环跑通，再增加新的 option graph。

## 三、关键机制

### 3.1 目标身份与感知健康

- SAM3返回全部候选，禁止直接用最高分覆盖当前目标。
- 通过运动可达范围、时序IoU、mask面积/长宽比、HSV外观直方图、与EEF的相对关系进行级联关联。
- 候选无法唯一匹配时将状态置为 `AMBIGUOUS`，保留原 `instance_id`，不得切换到新物体。
- Qwen只在候选歧义时看到原始双视角和编号crop，并从候选ID中选择。
- Wrist失败时允许AgentView fallback，但所有尝试和最终来源必须入日志。
- SAM请求最多重试2次；连续3个观测周期失败即终止为 `perception_infrastructure_fault`。故障期间执行STOP，不生成语义动作。
- 每次正式rollout前运行Qwen和SAM3 preflight；服务不健康时拒绝启动环境动作。

### 3.2 Runtime原子动作控制

每个option定义自己的残差，而不是让Qwen从整张图自由判断方向：

- 水平对齐：目标/容器与EEF或持有物中心的二维视觉残差。
- 高度控制：EEF高度、安全工作空间、Wrist接近证据、接触/停滞证据。
- 搬运：路线航点残差和持有不变量。
- 放置：持有物与篮筐开口中心、边缘距离和下降进度。

动作效果模型以“相机 × embodiment × option”为单位维护每个原子动作对残差的预期变化：

1. 初始先验来自独立空场标定和已有控制器坐标定义。
2. 每执行一个动作就获取新观测。
3. 更新该动作的实际残差变化。
4. 两次连续反方向效果触发方向重估。
5. 三次连续无有效进展触发 `NO_PROGRESS`。
6. 最近8步出现重复状态—动作环时触发 `OSCILLATION`，禁止继续左右互换。

前后左右只控制水平运动，`UP/DOWN`只用于高度；禁止再把图像纵向误差直接解释为机器人上下运动。

默认硬预算（V2 基线；V2.1 route 配置可按本体安全重规划开销覆盖）：

- 定位：10个有效观测。
- HOVER：80步；TRANSFER：V2 基线80步，当前 V2.1 配置为120步以容纳通用路线重规划和 verifier recovery，不绑定对象或坐标。
- 精对齐：40步。
- 抓取、验证和释放：各15步。
- 单次恢复：60步。
- 最多2次同类恢复。
- 单episode总预算300步。

预算同时受距离估计动态缩短；到达预算后必须产生明确失败事件，不能继续尝试到200步以上。

### 3.3 独立验证器

`VERIFY_HOLD` 必须综合：

- 闭爪命令已执行。
- 连续多帧中目标与EEF共同运动且相对位姿稳定。
- 目标不再保持原支撑关系。
- Wrist/AgentView视觉证据。
- gripper width只能提供辅助否决，不能单独证明抓住或丢失。
- 几何证据冲突时才调用Qwen做受限的 `HELD / NOT_HELD / UNKNOWN` 判断。

`VERIFY_SEATED` 必须在释放前确认：

- 物体投影位于篮筐开口安全区域内，而非仅与外框重叠。
- 下降过程有效，且没有持续顶住边缘。
- 持有关系仍成立。
- 几何与视觉证据允许物体被支撑。

`VERIFY_TASK` 使用视觉状态作为策略判断；`env.check_success()`仅用于最终评分，永不反馈给策略。

### 3.4 类型化恢复

- **掉落恢复**：停止运动 → 保持原目标ID → 作废旧grasp/route epoch → 重新定位同一实例 → 从HOVER重新抓取 → 验证新grasp → 重建路线 → 返回LIFT/TRANSFER。
- **空抓恢复**：张开夹爪 → 回到最近安全高度 → 重定位目标 → 调整预抓姿态；不直接重复闭爪。
- **边缘碰撞**：保持持有 → 上提清障 → 重定位开口 → 重新对齐和下降。
- **目标歧义**：未持物时完全停止；持物时只允许安全上提和保持，随后请求候选消歧。
- **无进展/振荡**：作废当前动作效果估计与局部track，不作废任务目标；重新观察一次后换备选方向。
- **传感器故障**：STOP并尝试恢复服务，不能进入普通重定位流程。

每条恢复边都保存 `resume_option`，成功后回到正确阶段，而不是统一回滚至APPROACH。

## 四、实施阶段

### Phase 0：冻结基线和修复契约

- 冻结现有runner和配置为V1基线；V2使用新的runner入口和配置，关闭V2时行为不得变化。
- 先解决当前3个能力测试漂移：
  - 保留approach height前置条件，更新旧fixture和契约说明。
  - fallback测试验证完整查询顺序与AgentView兜底，不再假设固定2次调用。
  - 机器日志继续使用稳定字段`progress_delta_px`，人类prompt统一显示`motion_since_previous`。
- 建立V2事件schema和版本号。
- 按AGENTS.md建立全局路线文档、当前实验记录；每次rollout保存原始双视角、调试overlay、完整事件流和视频。

完成门槛：现有测试全绿，V1 smoke test可复现，Qwen/SAM3 preflight能在动作前阻止坏服务。

### Phase 1：离线回放与状态层

- 实现传感器健康、entity track、三值belief和事务式verifier。
- 回放 `run_d427...` 的保存帧：
  - step 161不得把高分错误瓶子写成目标。
  - 无唯一匹配时必须进入 `AMBIGUOUS`。
  - 不允许出现后续左右振荡。
  - step 136掉落后必须形成包含原目标ID和resume point的恢复上下文。
- 用损坏/缺失SAM响应重放基础设施故障，确认不会输出运动动作。

完成门槛：失败episode可在不启动仿真的情况下稳定重现并阻止已知首个错误分叉。

### Phase 2：Option控制和掉落恢复

- 实现完整pick-place option graph、残差控制器、动作效果更新和超时机制。
- 保留现有atomic controller、环境封装和logger，替换单体runner中的控制与恢复职责。
- 建立隔离的 `diagnostic_oracle` profile，用仿真状态执行标准化掉落注入；该模块不得被正式profile导入。
- 掉落注入发生在 `VERIFY_HOLD` 通过后的TRANSFER阶段，对物体施加小幅侧向/向下扰动；策略只能通过正常视觉和本体观测发现。

工程门槛采用同一任务、同一历史失败初始状态的5次运行：

- 3次正常，2次带掉落注入。
- 总成功至少3/5。
- 至少1/2掉落实验在60步恢复预算内完成任务。
- `SENSOR_FAULT`期间运动动作数为0。
- 不发生未验证目标身份切换。
- 不发生超过8步的左右振荡环。
- 所有成功都具有完整 `VERIFY_HOLD → VERIFY_SEATED → VERIFY_TASK` 证据链。

### Phase 3：跨物体泛化

选择几何差异明显的三个任务：

- salad dressing：开发对象。
- alphabet soup：圆罐类held-out。
- milk：纸盒类held-out。

每个对象选3个初始状态，每个状态运行3个随机种子，共27次；V1和V2严格配对。

进入全量评估的门槛：

- V2总成功率至少15/27。
- 每个对象至少5/9成功。
- 相对V1提升至少20个百分点。
- held-out对象不允许新增对象名称、尺寸、坐标或专属阈值。
- 人工审查所有视频及失败首帧，更新失败分类而非直接调任务专属常数。

### Phase 4：研究评估与能力归因

正式主实验：

- LIBERO_OBJECT全部10个任务，每任务5个固定初始状态，共50个配对episode。
- 主比较：`Qwen8B-V1`、`Qwen8B-VCR-v2`。
- 分层30-episode子集运行消融：
  - 仅identity/belief。
  - 加option controller。
  - 加transition verifier。
  - 加typed recovery，即完整V2。
- 10个额外掉落注入episode评估恢复成功率和恢复步数。
- 使用同一组状态运行已配置的frontier agent参考基线，但它只用于衡量差距，不进入最终系统。

能力诊断通过单因素oracle intervention完成：

- C1目标身份/grounding。
- C3局部动作选择。
- C4持有和物理状态。
- C5高层option规划。
- C6恢复策略。
- C7后置条件判断。

每次只替换一个能力，报告：

```text
ACE_i = Success(Qwen8B + Oracle_i) - Success(Qwen8B)
GapClosure = (Success(Qwen8B-V2) - Success(Qwen8B-V1))
             / (Success(Frontier-V1) - Success(Qwen8B-V1))
```

特权真值必须写入独立`oracle`日志字段，并在正式结果配置中通过启动断言强制关闭。

### Phase 5：条件触发的小模型训练

V2稳定前不训练模型，也不做完整大模型轨迹模仿。

完成能力干预后，仅当以下任一条件成立时启动Qwen8B LoRA：

- 某项Qwen关键决策错误占失败的20%以上。
- 对应oracle intervention带来至少15个百分点成功率提升。

训练数据只包含V2运行中“关键决策点—局部纠正—结果”，例如候选消歧、抓取状态冲突、恢复分支和放置确认；不训练原子运动长轨迹。训练后仍由相同VCR-v2 verifier约束。

## 五、日志、指标和测试

每个动作事件记录：

- `observation_health`
- `belief_before / belief_after`
- `entity_tracks`
- `evidence`
- `active_option`
- `residual_before / residual_after`
- `requested_action / executed_action`
- `predicted_effect / observed_effect`
- `transition_verdict`
- `failure_event / recovery_context`
- `qwen_decision`
- 隔离的`oracle_metadata`

核心指标：

- 任务成功率和配对成功差。
- 首个错误分叉位置。
- 抓取、持有、放置verifier的假阳性/假阴性。
- 目标身份错误切换率。
- 振荡率与无进展步数。
- 掉落检测延迟、恢复成功率和恢复步数。
- 传感器故障时的非法动作数。
- Qwen调用次数、token、延迟、SAM调用与故障率。
- 各option成功率和预算耗尽率。
- V1到V2的gap closure与各能力oracle收益。

必需测试：

- 三值状态和事务式提交单元测试。
- 多候选交叉、遮挡、重现和高分干扰物的identity测试。
- 动作实际效果与预期方向相反时的在线纠正测试。
- 无进展、振荡、传感器故障和恢复预算测试。
- 抓空、运输掉落、篮筐边缘碰撞、释放后滚出的集成测试。
- `run_d427`记录帧回放回归测试。
- V2关闭时V1行为和日志兼容测试。
- 正式profile不导入oracle模块的静态测试。
- 至少一个held-out物体和初始状态的非过拟合测试。

## 六、研究定位与约束

研究贡献应表述为：

> 将小VLM缺失的连续控制、物理状态维护、转移验证和恢复能力，编译为带证据的外部runtime；只在不可算法化的关键语义决策点调用小VLM。

不把贡献表述成“增加视觉工具”“增加行为树”或“让模型预演动作”，因为这些方向分别已被 [Robo-Harness K1](https://arxiv.org/abs/2609.29389)、[VLM与Reactive Planner/Behavior Tree](https://arxiv.org/abs/2503.15202) 和 [World Action Agent](https://arxiv.org/abs/2609.29964) 覆盖。与原始 [Show-Harness](https://arxiv.org/abs/2609.10522) 相比，V2的关键变化是从“VLM直接决定细粒度物理动作”转向“VLM选择语义option，verified runtime闭环执行”；训练策略遵循局部on-policy correction，避免已有研究指出的完整强模型轨迹模仿退化问题 [Co-Evolving Harnesses and Models](https://arxiv.org/abs/2609.09134)。

默认约束：

- 最终线上系统只使用Qwen8B、SAM3、图像和正常机器人本体状态。
- 不使用物体真值位置、任务成功真值或仿真对象ID作为策略输入。
- 不写当前物体、任务、尺寸或坐标的专属规则。
- 阈值均进入embodiment/runtime配置，并在开发集冻结后用于held-out任务。
- 每次rollout必须保存视频并检查关键帧。
- 第一阶段优先跑通，不在闭环稳定前扩展到其他LIBERO suite或追求论文规模结果。
