# Show-Harness V2.1：Adaptive Capability Harness

## 一、核心判断

方向正确：不要继续要求 Qwen8B 从静态 RGB 中同时完成三维重建、坐标变换、抓取姿态计算和动作闭环。Harness 应把这些可结构化能力外置，Qwen只承担语义和关键歧义决策。

当前空抓已有明确证据：

- 最终 EEF 深度约为 `x=0.095m`，诊断真值中的目标约为 `x=0.05m`，夹爪在目标前方约 4.5cm。
- support-plane 回投也错误估计到约 `x=0.095m`，因此控制器停在了自己构造的错误目标点。
- step 57–66 中 Qwen八次判断 `MV_DOWN`；到达高度下界后，后端又把部分下降改成上升，形成上下反弹。
- 当前 prompt 错误地给 AgentView 和 Wrist 使用同一屏幕方向映射，忽略了固定相机和眼在手相机不同的运动视差。
- 当前 pregrasp 请求只有当前帧，没有动作前后对比，也没有真正的深度证据。

所以不能只“加一个深度模型”。完整解决需要：

```text
SAM3身份与mask
    ↓
Visual Memory：动作前后帧、相机位姿、真实执行动作
    ↓
MoGe深度先验 + 主动视差三角化
    ↓
带不确定性的Spatial Belief
    ↓
Gripper swept-volume / GraspGenX抓取候选
    ↓
Qwen查看原图、时序对比、3D证据并选择语义决策
    ↓
Runtime编译成一个原子动作 → 验证效果 → 更新belief
```

新模型是证据生产者，不能直接写入 `HELD/SUCCESS` 或绕过 verifier。

## 二、Harness机制与接口改革

### 1. 先修复控制契约

- 删除 V2 中“AgentView 与 Wrist 共用 screen-down→MV_FWD”这一假设。相机到动作的关系按“相机×embodiment×option”在线学习。
- Runtime显式记录 `qwen_requested_action → authorized_action → actually_executed_action`；动作效果模型只能学习实际执行动作，禁止后端静默改写后仍按原动作更新。
- 到达机器人下方安全带时，从允许动作中移除 `MV_DOWN`，向Qwen明确说明“当前剩余误差是平面深度，不是高度”。
- 不再用 `wrist_grasp_anchor_px`、对象参考高度或固定像素中心作为正式抓取目标。

### 2. 增加真正的 Visual Memory

新增 `VisualMemoryEntry`：

- `instance_id / grasp_epoch / frame_id`
- 原始AgentView与Wrist
- SAM mask与track
- 相机位姿、EEF位姿
- 请求动作与实际动作
- 动作前后目标运动
- 深度/点云摘要及不确定性
- `BEST_ALIGNMENT / BEFORE_ACTION / AFTER_ACTION / FIRST_FAILURE`标签

规则：

- 每个目标和grasp epoch最多保存8个关键帧。
- Qwen每次只接收当前帧、上次动作前帧、动作后帧和最近最佳帧，避免把整段轨迹塞入上下文。
- 掉落或重新抓取时创建新epoch；旧记忆保留用于诊断，但不得作为当前对齐状态。
- 原始图、工具overlay和Qwen输入panel分别保存，避免标注污染SAM或状态估计。

### 3. 新增 Spatial Tool Bus

新增公共结果类型：

```text
SpatialToolResult
- source
- instance_id
- frame_id
- target_points_camera/world
- target_to_gripper_xyz
- covariance
- confidence
- health
- stale
- diagnostics

SpatialBelief
- fused_relative_xyz
- FRONT / BACK / LEFT / RIGHT / ABOVE / BELOW / INSIDE_ENVELOPE / UNKNOWN
- uncertainty
- agreeing_sources
- conflicting_sources
```

正式深度来源：

1. **MoGe-3 ViT-L**作为首选RGB几何工具。它可从单图输出metric point map、depth、normal和intrinsics，并允许输入已知FOV；370M版本足以放在新增GPU上。[MoGe官方仓库](https://github.com/microsoft/moge)
2. **主动多视角三角化**作为独立验证源：在安全高度利用连续Wrist帧、已知手眼位姿、SAM mask内特征和RANSAC恢复metric 3D，不依赖对象高度。
3. 若两者冲突超过联合不确定性，状态为 `AMBIGUOUS`，Runtime执行一次可逆微小侧向probe后重新估计，而不是让Qwen猜方向。
4. LIBERO渲染深度和对象真值只进入 `diagnostic_oracle`，绝不进入正式策略。

MoGe不能直接决定动作；必须先经过mask过滤、时序一致性和主动视差校验。若MoGe-3在诊断集不达标，再按固定顺序评测 UniDepthV2 和 Depth Anything V2，选择规则依次为：最低false-ready、最高深度方向正确率、最低延迟。[Depth Anything V2官方metric模型](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md)

### 4. 用gripper geometry替代对象硬标定

定义 `GripperEnvelope`：

- 来自URDF和swept-volume descriptor。
- 描述指间空间、指尖深度、闭合方向和安全进入方向。
- 目标点云转换到gripper frame后，判断目标位于指间、夹爪前方、后方或侧方。
- 所有阈值相对于gripper envelope和估计不确定性，不按瓶子、罐头、纸盒分别设置。

本地已有 GraspGenX、Franka descriptor和checkpoint。它能根据目标点云和gripper swept volume生成跨夹爪6-DoF候选，因此先以shadow模式接入；通过离线门槛后，只激活与当前LIBERO顶抓执行器姿态兼容的候选。[GraspGenX官方仓库](https://github.com/NVlabs/GraspGenX)

第一阶段不立刻扩展完整6-DoF控制器：先使用候选的平移目标和gripper-envelope验证解决当前深度空抓。旋转执行留到稳定抓取以后。

### 5. 重定义Qwen与Harness的合作接口

取消让Qwen从静态帧直接输出 `MV_FWD/MV_BACK` 的 pregrasp 接口，改成：

```text
SELECT_GRASP:<candidate_id>
CORRECT_DEPTH
CORRECT_LATERAL
CORRECT_HEIGHT
PROBE_DEPTH
GRASP
UNKNOWN
```

Qwen输入包括：

- 原始双视角。
- 动作前后时序panel。
- 目标mask和深度overlay。
- gripper-frame俯视/侧视小图。
- 候选抓取编号。
- 工具一致性、置信度和“不确定”状态。
- 上一步实际执行动作及其真实效果。

职责划分：

- Qwen判断目标身份、可抓部位、优先修正哪个轴、选择候选、是否视觉上可以闭爪。
- Runtime根据 `SpatialBelief` 把 `CORRECT_DEPTH` 编译成正确符号的原子动作。
- `GRASP`只有在Qwen确认、目标进入gripper envelope、深度证据有效且身份新鲜时才执行。
- 任一证据冲突则主动probe或STOP，不用单一模型分数强行抓取。

这保留了Qwen的“看和判断”，同时不再要求8B模型自己解决难以从单帧辨识的三维坐标与动作符号。

## 三、实施顺序与GPU安排

### Phase A：复现并堵住当前错误分叉

- 将 `run_f9794...` 固化为回归案例。
- 修复实际动作回执、Wrist时序记忆、安全高度动作集和相机动作语义。
- 补齐关键步骤的原始双视角保存；当前只有合成视频，不满足后续深度评测要求。
- 回归门槛：失败帧必须判为“夹爪在目标前方/深度未知”，不得再解释成需要持续下降。

### Phase B：离线空间工具选择

使用隔离的深度oracle profile，在 salad dressing、alphabet soup、milk 各3个init state采集：

- RGB、SAM mask、相机位姿。
- 渲染深度和对象位置仅作为标签。
- 高位接近、低位pregrasp、遮挡和偏置抓取帧。

门槛：

- 当真实深度残差超过15mm时，方向符号准确率开发对象≥95%、held-out≥90%。
- “已进入抓取包络”的假阳性率低于5%。
- 不确定性覆盖率≥90%。
- 单次关键帧调用延迟低于500ms。
- 不允许通过对象名称或专属阈值达到门槛。

### Phase C：在线pregrasp闭环

- MoGe与SAM3常驻GPU1。
- 主动视差默认使用自然运动历史；只有来源冲突时才执行微小可逆probe。
- GraspGenX先shadow记录候选；候选方向和oracle一致率达90%、false-ready低于5%后才进入active。
- GPU0独占Qwen8B；GPU1运行SAM3、MoGe和GraspGenX，启动时确保至少保留4GB显存余量。
- 任一工具连续三次故障时STOP，不退化为Qwen静态RGB猜深度。

阶段门槛：

- 相同失败init state做5次抓取测试，至少4次通过 `VERIFY_HOLD`。
- 空夹最多1/5。
- 不出现超过4步的UP/DOWN或FWD/BACK振荡。
- 每次闭爪都有“身份→空间belief→Qwen选择→gripper envelope→闭爪→hold verifier”完整证据链。

### Phase D：恢复完整任务和泛化

- 回到原计划的5次完整episode门槛：成功至少3/5。
- 再加入2次运输掉落，至少成功恢复1次。
- 在alphabet soup和milk上不得新增对象名称、尺寸、抓取高度或坐标规则。
- 抓取稳定后再接入本地AnyPlace作为放置候选工具；此前不同时扩张抓取和放置变量。

## 四、测试、日志与研究消融

必需测试：

- 固定相机与Wrist相机的动作效果不能共享错误映射。
- 请求动作被安全层改写后，Runtime只能学习实际动作。
- 深度结果过期、实例ID不符或服务故障时不能驱动运动。
- MoGe与主动视差冲突时进入probe/UNKNOWN。
- 替换gripper descriptor后，抓取包络自动变化，不修改对象代码。
- 正式profile不能导入渲染深度、对象pose或`env.check_success()`。
- V2关闭时V1行为保持不变。

新增日志指标：

- `visual_memory_refs`
- `depth_sources / depth_uncertainty`
- `target_to_gripper_xyz`
- `gripper_envelope_relation`
- `grasp_candidates`
- `qwen_semantic_choice`
- `requested / authorized / executed_action`
- `depth_direction_accuracy`
- `false_grasp_ready`
- `empty_close_rate`
- `probe_count`
- 各工具延迟、显存和故障率

实验消融：

1. Qwen8B-V1。
2. 当前RGB-only VCR。
3. VCR + Visual Memory。
4. VCR + Visual Memory + active triangulation。
5. VCR + MoGe。
6. VCR + MoGe + GraspGenX，即完整Spatial Capability Harness。

论文贡献仍然表述为“自适应地外置并验证小VLM缺失的物理能力”，而不是“又加了深度/抓取工具”。深度模型、GraspGenX和以后可能的AnyPlace只是可替换的capability providers；真正的方法是 visual memory、带不确定性的状态融合、critical capability allocation、事务式验证以及Qwen与工具之间的受限协作。

实施时同步更新 [总体计划](/root/autodl-tmp/Show-Harness/docs/revolution/plan.md)、新增全局路线和当前实验记录，并按 [AGENTS.md](/root/autodl-tmp/Show-Harness/AGENTS.md) 保留每次rollout的视频、原始关键帧和首个错误分叉。

## 五、默认约束

- 用户本轮授权覆盖原计划“线上不增加模型”的限制：允许在第二张GPU增加专业工具模型。
- 正式策略仍不得读取模拟器对象ID、对象真值位置、渲染深度或任务成功真值。
- 不使用对象专属坐标、高度、尺寸、夹爪宽度或像素锚点。
- Qwen始终是语义目标、候选选择、冲突判断和恢复分支的上层Agent；专业工具不能独立宣布成功。
- 当前第一优先级是消灭深度空抓并恢复稳定 `VERIFY_HOLD`，在此之前不继续扩展放置或其他LIBERO suite。

## 六、执行状态索引（2026-09-29 暂停点）

本计划的实施细节和实验结论不在此处反复改写，统一记录于：

- [V2.1 当前实验记录](progress.md)：最近 rollout、已验证能力、未通过的 placement 闭环和恢复顺序。
- [V2.1 总体路线](route.md)：阶段边界、option graph 当前状态和下一次实验检查项。

当前实现已完成 Visual Memory、空间工具总线、MoGe fallback、active parallax、gripper envelope、语义 pregrasp、动作回执、运输 `PRE_DESCENT` 边界和独立 placement verifier。最新已完成 run 已能在 step 66 通过抓取/持有验证，但尚未完成 `VERIFY_SEATED` 或完整 `VERIFY_TASK`；placement feedback 的最新代码补丁尚未经过新的 rollout。用户要求暂停，因此此处不把未验证结果写成门槛达标。
