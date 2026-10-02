# V2.2 通用空间放置闭环规划

## 结论与根因

下一步不应继续调 prompt 或增加步数。应新增“Placement Spatial Harness”，并将放置阶段的动作权统一收回 VCR runtime。AnyPlace 暂时只接 shadow，不立即控制机械臂。

当前失败由四层问题叠加：

| 问题 | 现有证据 | 后果 |
|---|---|---|
| 开口投影平面错误 | route 把开口投影到 `z=0.015m` 桌面，但同一 run 估计 rim 约 `0.073m`；斜视相机下造成约 10cm 世界 X 偏差 | EEF 到达错误终点，提前进入 `PRE_DESCENT` |
| 混用不同高度的二维中心 | step 153 的 metric residual 已为 0，但瓶子—开口纵向误差仍约 2.2 个开口高度 | runtime 与视觉证据互相矛盾 |
| 三套控制器争夺动作权 | route intent、placement verifier、legacy alignment reviewer 都能影响下一动作 | 形成 `DONE→MV_UP→LEFT/RIGHT→STOP` 循环 |
| phase 无滞回 | 约 1cm 动作步长在 1cm goal tolerance 附近跨界 | `TRANSFER/DESCENT` 和蓝绿线反复跳变 |

另一个直接接口错误是：当前 placement verifier 在“物体仍在篮子上方”时不能返回 `MV_DOWN`，只能选择 `MV_UP/水平动作/HOLD`，因此第一次否决后几乎必然抬高并进入循环。

## 核心架构修改

### 1. 建立统一的 Placement Belief

在 `core/runtime_v2/` 增加以下公共契约：

- `PlacementRelation`：`ABOVE_UNALIGNED / ABOVE_ALIGNED / DESCENDING_CLEAR / RIM_CONTACT / SEATED_HELD / RELEASED_STABLE / LOST / UNKNOWN`。
- `PlacementCandidate`：候选物体位姿、对应 EEF 位姿、来源、可达性、碰撞/包含裕量和不确定性。
- `PlacementBelief`：开口平面与多边形、物体点云、物体—夹爪刚性变换、选中候选、EEF 三维残差、rim clearance、containment margin、证据来源及冲突。
- `SemanticPlacementAction`：`SELECT_PLACE / CORRECT_LATERAL / CORRECT_DEPTH / DESCEND / PROBE / RECOVER_CLEAR / READY_TO_VERIFY / UNKNOWN`。

Qwen 只选择候选和语义动作；runtime 根据三维残差决定实际正负方向。Qwen 不再直接决定 `MV_LEFT/MV_RIGHT/MV_FWD/MV_BACK`。

### 2. 修正空间表示

- 从 SAM3 响应中保留目标、外部容器和开口的 mask，不再只取 bbox。
- 将抓取前或刚完成 `VERIFY_HOLD` 的目标点云变换到 gripper frame，锁定当前 grasp epoch 的 `T_gripper_object`。
- 使用开口 mask 轮廓和容器边界带拟合 rim plane；禁止把开口中心固定投影到桌面平面。
- 在同一世界坐标系计算物体 footprint、开口 free-space polygon 和候选 EEF 目标，彻底取消“不同高度 bbox 中心直接相减”作为正式放置残差。
- 复用 GPU1 上的 MoGe-2 metric point map 和 active parallax；来源冲突时只允许高净空 probe 或 STOP。MoGe-2本身可输出 metric 3D point map，适合作为证据源而非动作决策器。[MoGe-2](https://arxiv.org/abs/2507.02546)
- 所有安全裕量由开口尺寸、物体 footprint、估计协方差和机器人动作分辨率共同生成，不使用篮子、瓶子或固定像素专属常数。

### 3. 收敛为单一动作权

- V2.2 下 `VisualRoutePlugin` 只生成运输路线、几何候选和可视化，不再拥有 placement Qwen intent 或动作 gate。
- VCR runtime 唯一决定 requested/authorized action；runner 只执行并回传 receipt。
- 禁用 V2.2 下 legacy `review_place_alignment` 的动作覆盖和 verifier recovery action。verifier 只报告物理关系，不生成移动 token。
- 通用 option 生命周期固定为：

```text
TRANSFER → ALIGN_OPENING → DESCEND_TO_SEAT
→ VERIFY_SEATED → OPEN_GRIPPER → RETREAT → VERIFY_TASK
```

这不是场景状态机，而是跨对象通用、由证据前置条件驱动的事务式技能图。

- `READY_TO_VERIFY` 只有在 `SEATED_HELD` 或足够强的 seating 候选证据下才进入 Qwen 的允许答案；`ABOVE_ALIGNED` 必须编译为一次 `DESCEND`。
- `RIM_CONTACT` 才允许 `RECOVER_CLEAR`；“仍在上方”不能再触发 `MV_UP`。

### 4. 稳定闭环和可视化

- 对 alignment 使用 Schmitt hysteresis：进入对齐要求 footprint containment margin 大于 `2σ + 半个动作量`；只有连续两帧越过退出边界才退回 ALIGN。
- 一旦候选和 grasp epoch 未变化，`PRE_DESCENT/DESCENT` 不因单帧毫米级噪声回退到 TRANSFER。
- 接近目标后缩小原子步长；若一个动作使三维残差变差，动作效果模型标记该方向并重估，而不是立即反向循环。
- Qwen 输入使用原始双视角加稳定的候选 footprint/不确定性轮廓；停止显示会随 phase 翻色的多段路线。
- 分别保存 raw RGB、provider overlay、Qwen input panel 和 UI composite，避免当前 rollout 中 overlay 污染原始回放。

## 工具模块策略

- 第一优先级是上述 Placement Spatial Harness，不再添加另一个自由动作模型。
- 本地 AnyPlace 使用 parent/child point cloud 预测通用 placement pose，方向与研究目标匹配。[AnyPlace](https://arxiv.org/abs/2502.04531)
- AnyPlace 通过独立 GPU1 subprocess 接入，避免其 Torch 1.13 环境和硬编码 `.cuda()` 污染主进程；第一阶段仅 shadow 输出 top-k `PlacementCandidate`。
- 不得丢弃 AnyPlace 候选旋转后只取平移。当前控制器无法执行的姿态标记为 `UNSUPPORTED_ORIENTATION`。
- 只有在 geometry-only 闭环达到 3/5 完整成功，且 AnyPlace 在 held-out 点云上的 top-k 可行率达到 80%、没有增加 false-ready 后，才允许它参与候选排序。M2T2 等同类模型本阶段不再重复引入。

## 实施与验证顺序

1. 将 `run_32e0d28e5b7e` 的 step 145–186 固化为回归：重现错误平面、三方动作冲突和 phase 抖动。
2. 先修 rim-plane 投影、物体—EEF offset、verifier 关系协议和单一动作权；此时不接 AnyPlace。
3. 接入 SAM3 mask → MoGe/parallax → Placement Belief，完成离线关键帧评测。
4. 跑一个单 episode，必须观察到 `ABOVE_ALIGNED → MV_DOWN → fresh belief`，且不再重复 `READY_TO_RELEASE`。
5. 同一 init state 跑 5 次：完整成功至少 3/5、无未验证 release、无超过 4 步的左右或升降振荡。
6. 在 salad dressing、alphabet soup、milk 各 3 个 init state 做配对验证；开发对象之外不得新增名称、坐标、高度或尺寸规则。
7. 每次实现和 rollout 后同步更新 `docs/revolution/progress.md` 与 `route.md`。

## 测试和验收

- 实际 LIBERO 相机回归：开口在 rim plane 与 support plane 的投影必须可区分，route 终点使用前者。
- 高度不变性测试：同一世界 XY 的物体随 Z 改变产生像素视差时，三维水平残差仍接近零。
- 抓取偏心测试：不同 `T_gripper_object` 自动改变 EEF placement target。
- verifier 测试：`ABOVE_ALIGNED→DESCEND`、`ABOVE_UNALIGNED→轴修正`、`RIM_CONTACT→RECOVER_CLEAR`、`UNKNOWN→PROBE/STOP`。
- 权限测试：每帧只能有一个 requested/authorized/executed 链，legacy reviewer 不得覆盖 VCR。
- 抖动测试：目标边界加入一动作量以内噪声时 option 和 overlay 不翻转。
- provider 过期、实例不符、MoGe/parallax 冲突时动作数为 0。
- pre-release 必须有 identity→placement belief→containment/rim evidence→`VERIFY_SEATED` 链；最终成功仍只由 `env.check_success()` 计分，不反馈策略。

## 默认假设

- V1 保持原样，所有修改进入独立 V2.2 profile。
- simulator depth、对象 pose 和对象 ID 只允许在隔离的离线 diagnostic oracle 中作为评测标签。
- 当前先解决平移式 top-down 放置；完整 6-DoF AnyPlace 激活留到旋转执行与验证接口完成以后。
- GPU0 保留 Qwen8B，GPU1 运行 SAM3、MoGe 和 shadow AnyPlace。
