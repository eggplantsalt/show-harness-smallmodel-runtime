# SAM3 abstain 根因排查 — 2026-09-26

## 结论

恢复后的 Kimi C3 运行中，SAM3 的主要问题不是“目标不可见”，而是
**SAM3 对目标给出了低分真阳性，但 `confidence_threshold: 0.5` 在服务端先把它过滤掉**。
Show-Harness 随后把“服务错误、空检测、低分检测”统一记录成 `sam3_abstain`，所以之前的
日志无法直接区分这些机制。

另有一类独立的基础设施故障：第一次 Kimi 运行的所有 SAM3 调用都是 `Broken pipe`，这是
SAM3 SSE 服务/bridge 已断开的错误，不能与后面恢复运行中的空检测混为一谈。

## 证据

| 证据 | 结果 | 含义 |
| --- | --- | --- |
| `run_0e411d41daf2` | 14/14 次 `Broken pipe` | 基础设施失败，SAM3 没有完成请求 |
| `run_0bf128f871cf` | 126 个决策，其中 13 次 `sam3`、39 次 CPU tracker、74 次 `sam3_abstain` | 恢复后主要是“成功返回但空 detection” |
| 同一 `0053.png`、原始提示词、阈值 0.5 | 0 detections | 触发原有 abstain |
| 同一 `0053.png`、原始提示词、阈值 0.4 | 1 detection，`score=0.4199`，bbox=`[148,119,167,161]` | 目标确实被 SAM3 找到，只是分数低于 0.5 |
| 失败段的 `0053/0060/0070/0080/0090/0100/0110/0120/0125.png` | 阈值 0.5 均为空；阈值 0.4 均返回同一个目标框，分数约 0.42--0.46 | 不是目标突然离屏，也不是图像编码损坏 |

原始 planner 组合出的 SAM3 查询是：

```text
salad dressing bottle, body of green-capped salad dressing bottle
```

这是一个较长的组合短语。对同一帧使用较宽泛的 `salad dressing, body of the bottle`
会产生两个候选，其中最高分候选是另一只相似瓶子；因此不能简单地用宽泛 fallback 并盲取
第一个框。当前更安全的实验变量是只把阈值从 0.5 改成 0.4，保留原语义查询。

## 失败如何传到后续控制

```text
target + affordance
        ↓
SAM3 target score ≈ 0.42--0.46
        ↓  threshold=0.5 在 SAM3 backend 过滤
空 detections，但 response.success=true
        ↓
VisualHarness: source=sam3_abstain, visible=false
        ↓
CPU tracker 不再有 seed；episodic memory 失效；geometry 不再有 target bbox
        ↓
prompt 只得到 “no current verified target evidence”
        ↓
C3 没有 active evidence gate，Agent 仍可依据 RGB 猜动作
        ↓
后半段动作脱离可验证的 target-center feedback，未提交 GRASP
```

在 Kimi 记录中，SAM3 恢复且有 bbox 时，水平误差从约 `+31.7 px` 降到约 `+1.1 px`；
垂直误差仍约 `+60 px`。从 frame 48 开始进入低分 abstain 后，geometry 的
`target_minus_eef_px` 变为未知，Agent 又回到基于画面印象的 `MV_RIGHT/MV_FWD` 交替和
持续移动。这说明 abstain 是后续失去闭环反馈的重要触发点，但它不是全部控制失败：即使
目标被检测到，C3 仍然依赖 VLM 自己决定何时完成垂直对齐和 GRASP。

## 已采取的排查改动

- `core/capabilities/visual_harness.py` 现在记录 `abstain_reason`、实际 SAM3 query、阈值、
  backend metadata、detection count 和 raw score，可区分 `request_failed`、
  `no_detections_after_backend_threshold`、`score_below_harness_threshold` 等原因。
- 新增 `configs/robot_libero_siliconflow_kimi26_c3_threshold04.yaml`：只将 SAM3 阈值改为
  `0.4`，其余 Kimi C3 long 配置不变，用于下一轮因果验证。

下一轮先看三个问题：

1. abstain 是否从持续出现降到接近 0，并保持正确目标框；
2. geometry 是否持续提供 target-center error，Agent 是否完成 APPROACH/GRASP；
3. 如果感知闭环恢复但仍失败，再单独归因给 VLM 的垂直方向/GRASP commit，而不是继续把
   失败归咎于 SAM3。

## 阈值 0.4 对照结果

运行目录：
`rollouts/libero_siliconflow_kimi26_c3_threshold04/run_17d923a67356/`

- 180 个决策全部停留在 `APPROACH`，`success=false`，没有发出 `GRASP` 或 `DONE`。
- 证据来源：27 次 fresh `sam3`、81 次 CPU tracking、72 次 abstain。
- 前 100 帧的正确 bbox 持续存在，水平误差从约 `+31.7 px` 降到约 `0--3 px`，垂直误差
  从约 `+70 px` 降到约 `+7--16 px`。这证明降低阈值确实修复了第一层感知断环。
- frame 100 的 `alignment_ready=true`，但 Agent 仍继续发 `MV_FWD`，随后在 frame 107
  附近让目标被夹爪/机械臂遮挡，进入第二类 abstain。
- 第二类不是简单的 0.4 阈值问题：frame 107 对原始组合 query 没有稳定高分框，但只用
  `green-capped bottle` 可得到约 `0.57` 的正确框；frame 120 宽 query 仍能给出约
  `0.44` 的框。到 frame 140 以后，多个 query 都没有 detection，图像中目标已被当前
  视角的机械臂严重遮挡。

因此当前失败的分层归因是：

1. **已确认并可修复：** `0.5` 阈值过滤掉 0.42--0.46 的真实目标检测。
2. **仍存在的感知边界：** 目标接近/进入夹爪遮挡区后，组合语义 query 的分数进一步下降；
   完全遮挡时任何 text-SAM3 query 都只能 abstain。
3. **当前主控制瓶颈：** APPROACH 没有在“目标可见且几何已对齐”时可靠地结束。Agent
   不发 `DONE`，继续移动，主动把一个原本可观测的目标推入遮挡区；这使后续 abstain 成为
   结果，而不是最初的根因。下一步应单独验证 APPROACH completion/GRASP 进入条件，不能
   再把全部失败归因于 SAM3。

## Qwen 预抓取复核

Qwen 的本地 rollout：
`rollouts/libero_clean_qwen3vl_c3_threshold04_approachguard/run_dfc65557abfc/`

这次记录明确区分了“准备抓取”和“提交 GRASP”：

- frame 136：active APPROACH guard 发出 `DONE`，进入 `GRASP`；
- frame 137--147：连续 `MV_DOWN`，模型的 reasoning 是把目标带入 Wrist 视角；没有发出
  `GRASP`；
- Wrist frame 0099 对组合 query `salad dressing, body of the bottle` 为空，但
  `green bottle` 找到 `[44,45,106,106]`、score `0.84375`；
- Wrist frame 0136 组合 query 仍为空，`green bottle` 只能得到局部框
  `[115,41,149,55]`、score `0.4765625`；frame 0147 多个 Wrist query 均为空。

因此 Qwen 并非不知道抓取；更准确地说，Wrist 的空检不能推出“物体不可抓”。step 147
的 AgentView 已显示绿色瓶子位于夹爪两指之间，指尖高度约 `0.15164 m`，而 Wrist 由于
安装在夹爪上方且被近距离结构遮挡，仍然看不到瓶身。原有“Wrist 必须确认”的 prompt
判据把传感器不可观测错误地当成物理不可抓，导致模型继续 `MV_DOWN`。当前应由 AgentView
承担全局相对位置和最终夹爪中线判据，Wrist 只作辅助证据；同时仍需限制连续下降，避免
真正把目标推入遮挡区。

另一次 Qwen 对照还暴露了目标身份问题：planner 只生成 `salad dressing / body of
the bottle` 时，SAM3 在两个瓶子候选之间选中了右侧错误瓶子，随后 AgentView gate 虽然
正确提交了 `GRASP`，但 recovery 发现空抓。因此必须让 planner 在同类物体之间提供
颜色/瓶盖等视觉区分，并禁止 SAM3 对近分数候选按返回顺序盲选。
