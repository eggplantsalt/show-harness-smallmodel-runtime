# Show-Harness / Qwen8B LIBERO 实验进度与换 Session 交接

更新时间：2026-09-27（UTC）  
当前目标：在不偏离 Show-Harness 的“感知—规划—原子动作—验证—恢复”范式的前提下，只调用本地 Qwen3-VL-8B，让一个 LIBERO episode 真正完成，并从失败轨迹中定位机制性问题。

## 一、当前结论先说清楚

截至本文记录时，还没有拿到新的 `success=True` episode，不能把当前实现宣称为成功。最新实验仍在运行，已经完成过多次 `APPROACH → GRASP → LIFT → MOVE → PLACE`，也触发过放置验证和恢复；随后一次持握丢失，控制器回退到 `APPROACH` 重新获取。但在当前 live run 的后段，控制器连续停留在 `APPROACH`，Qwen 反复输出 `MV_DOWN`/`MV_FWD`/少量 `MV_LEFT`，尚未再次提交 `GRASP`。

这说明问题已经从早期的“根本不知道何时 GRASP”推进到两个更具体的问题：

1. **GRASP 判断已经可以被 AgentView 触发，但失手之后重新获取仍未闭环。**
2. **PLACE 的视觉验证和恢复链已经真正运行起来，但还没有证明它能稳定把歪放、接触停滞和丢持握恢复到成功。**

后续 session 应以 `success=True` 为验收条件，不能用“模型说 DONE”、夹爪宽度变化或肉眼看起来抓住了替代 LIBERO 的 `env.check_success()`。

## 二、当前环境与模型

- 工作目录：`/root/autodl-tmp/Show-Harness`
- VLM：本地 `Qwen/Qwen3-VL-8B-Instruct`，vLLM 端口 `8001`。
- SAM3：`openeta-services` 的 MCP/SSE 服务，端口 `8773`。
- 本轮已经按要求从 Kimi 切回 Qwen；不要因为旧文档中出现 Kimi 或 Qwen 35B 的历史记录而切换后端。

服务启动示例：

```bash
# Qwen
bash scripts/serve_clean_qwen3vl.sh

# SAM3
OPENETA_SAM3_CHECKPOINT_PATH=/root/autodl-tmp/openeta-services/models/sam3/sam3.pt \
  /root/autodl-tmp/openeta-services/sam3/.venv/bin/python \
  /root/autodl-tmp/OpenETA/tools/sam3_mcp_server.py \
  --transport sse --host 0.0.0.0 --port 8773
```

当前主实验命令：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
bash scripts/run_libero_zeroshot.sh \
  --robot-config configs/robot_libero_clean_qwen3vl_c3_threshold04_approachguard.yaml \
  --task-id 2 --init-state-index 0 \
  --max-steps 700 --prompt-log-every 1
```

## 三、已经证实的机制性问题

### 1. Wrist 不是全局抓取真值，AgentView 必须进入主判断

Wrist 相机安装在夹爪上方/夹爪附近。夹爪接近物体时，夹爪本体会遮挡瓶身；因此 Wrist 看不到瓶子，可能只是“不可观测”，不能推断“没有物体”或“不能抓”。此前 live frame 的关键证据是：

- AgentView 明确看到绿色瓶子位于两指之间；
- Wrist 主要看到夹爪和被遮挡的物体边缘；
- Wrist 上 SAM3 abstain 的同时，AgentView 仍足以支持 GRASP。

当前原则：

- **AgentView 是全局目标、夹爪、容器相对关系的主要视角。**
- **Wrist 是局部辅助视角。** 它可以补充接触/近距离信息，但不能作为抓取成功的必要条件。
- 在 GRASP 决策中，Qwen 一次性查看带有明确标签的 AgentView 与 Wrist；若 AgentView 已显示目标位于两指之间，应允许提交 `GRASP`，即使 Wrist SAM3 abstain。

这不是放松感知标准，而是修正相机几何造成的观测偏差。

### 2. SAM3 abstain 有三类原因，不能混为一种失败

目前记录到的 SAM3 abstain 来源包括：

1. **基础设施失败**：服务不可用、checkpoint/env 缺失、Broken pipe、ClosedResourceError 等。此时应记录 telemetry 并走视觉降级/重试，不应把服务异常解释为目标不存在。
2. **阈值过滤过严**：原阈值 `0.5` 会过滤真实候选分数约 `0.42–0.46` 的检测；当前阈值降为 `0.4`，同时保留候选分数和候选框信息。
3. **真实遮挡/视角不可见**：例如 Wrist 被夹爪遮挡。这是“该视角无法确认”，不是“抓取不可能”。

SAM3 还增加了近分数候选的歧义记录，避免多个相似物体时按返回顺序盲选。planner 需要输出颜色、瓶盖、形状等可区分属性，而不能只给模糊的 `salad dressing / body of the bottle`。

### 3. 不应把某个夹爪宽度当成跨物体的持握真值

此前确实发现过一个具体 bug：某轮真实夹住瓶子时宽度约 `0.0371 m`，但配置的 `open_width_m=0.03 m`，于是该状态被错误当成“仍然 open”；同时旧的 VLM 上下文只用很小的宽度阈值判断 `OPEN`，会诱导模型反复 `GRASP`/`STOP`。

这个数值是**诊断证据**，不是通用策略。当前不能恢复以下做法：

- 为每个 episode 或每种物体维护一个固定持握宽度字典；
- 用固定的“持握宽度区间”判断所有物体是否抓住；
- 让夹爪宽度替代 Agent 的视觉判断。

当前语义是：

- 夹爪宽度只用于低层发现明显的空夹/闭合异常，`empty_width_m≈0.004` 是空闭合的低层信号；
- VLM 的 `gripper_state` 主要反映当前开合命令，不把一个物体的持握宽度硬编码成“OPEN”；
- `GRASP` 后由上层 Agent 结合 AgentView/Wrist 做一次正向视觉确认；确认成功后完成 GRASP 子目标；
- 视觉确认失败才进入恢复，不因为某个物体的几何宽度不符合固定数值而否定抓取。

## 四、当前 PLACE 设计与通用性判断

用户提出的“放歪以后应 lift，再重新横向对齐，再 down”已经落实为模型驱动的闭环，而不是写死某个物体或场景的方向规则。

### PLACE 入口的一次性对齐复核

进入 PLACE 时调用 Qwen 的 `review_place_alignment`，同时看 AgentView 和 Wrist，输出结构化结果：

```json
{
  "aligned": "YES|NO|UNKNOWN",
  "recommended_action": "MV_DOWN|MV_UP|MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK|HOLD",
  "reasoning": "..."
}
```

当前行为：

- `NO`：执行模型推荐的一个恢复动作；不能把所有水平修正强行改成 `MV_DOWN`。
- 推荐 `MV_UP`：先抬高，清除容器边缘/夹爪遮挡，然后清空对齐状态，下一帧重新做对齐复核，再决定横向修正或下降。
- `YES`：允许继续放置。
- `UNKNOWN`：保持保守，不应因为单一 Wrist 图不可见就直接 release。

### DONE 或下降物理停滞时做放置验证

`verify_place` 在模型提交 PLACE `DONE` 时调用；若执行 `MV_DOWN` 前后 TCP 的 z 几乎没有变化，也会把它当作“接触/下降停滞事件”触发一次验证。这个 z 差值只是低层动作事件检测，不是“物体高度等于某个固定值”。验证结果为：

```json
{
  "placed": "YES|NO|UNKNOWN",
  "recovery_action": "MV_UP|MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK|HOLD",
  "reasoning": "..."
}
```

- `YES`：允许 PLACE 完成并进入 release/retreat。
- `NO`：不 release，执行 Agent 推荐的恢复动作；复位验证/对齐状态。
- `MV_UP` 后必须重新看新帧，而不是继续沿用旧的“已对齐”结论。

这比固定 `place_height <= 0.14` 或 `<= 0.16` 更通用。代码中若仍有 `0.16`，需要区分语义：当前 `_physical_step_for` 中的高度带只用于粗/细动作步长或运动控制，不再作为“放置成功”的真值。不要再把它解释成跨场景通用的物体/容器高度阈值。

### 为什么放歪以前没有被发现

旧路径容易把“夹爪已下降”当成“已经放好”，而 AgentView 中明显的偏心、贴边、倾斜或未坐入容器没有被单独复核。现在把“对齐复核”和“放置成功验证”分成两个上层判断，正是为了让 Qwen 利用 AgentView 的全局信息发现放歪。

## 五、恢复机制的当前状态

### 丢持握后的回退点

若 GRASP 后确认曾经持握，但后续视觉/低层信号显示丢失，恢复不再回到已经过时的 `GRASP`。当前回退到最近的前置 `APPROACH`，让 Agent 重新观察当前物体位置、重新对齐、重新决定何时 GRASP。这样避免“旧 GRASP token + 新场景位置”造成死循环。

### 最近一次 live run 暴露的新问题

最新 run：

```text
rollouts/libero_clean_qwen3vl_c3_threshold04_approachguard/run_cd8aa076e0e0/
  LIBERO-LIBERO_OBJECT-2/0927/task_0/13-07-53/steps.jsonl
```

已观察到：

1. 前段完成过 GRASP、LIFT、MOVE、PLACE；
2. PLACE 阶段出现过模型驱动的 `NO → MV_UP → 重新 alignment review` 链条；
3. 后续发生丢持握，控制器正确回退到 APPROACH；
4. 但从约 step 479 到至少 step 588，持续处于 `APPROACH`，模型主要输出 `MV_DOWN`，间或输出 `MV_FWD`/`MV_LEFT`，没有重新进入 GRASP。

这说明“回退到 APPROACH”的方向是对的，但恢复闭环仍未完成。下一 session 必须检查：

- 丢持握后的 AgentView 视频帧中，目标是否仍在原位置、是否已经被放歪/撞移；
- target anchor 是否仍指向旧位置，是否需要由新一轮 SAM3/AgentView 重新定位；
- APPROACH guard 是否把目标误判为已经 XY 对齐，导致只允许下降而没有 GRASP commit；
- `MV_DOWN` 在当前 z 值下是否已经是无效/碰撞动作；
- Qwen 是否在图像中已看到两指之间目标，但 prompt/状态机没有把该视觉判断转成 `GRASP`；
- 每次原子动作后的截图，而不是只看最后一帧，是否显示了“接近—对齐—进入夹爪”的真实过程。

### 本 session 新增：丢持握后的下降闸（2026-09-27）

最新 `run_e23090b586b7` 复现了更窄的失败：丢持握后进入 APPROACH，Qwen 在目标当前空间关系尚未重新确认时连续输出 `MV_DOWN`，从 z=0.214m 一直下降到 0.150m；这不是“lift 没有执行”，而是恢复阶段把 AgentView 的屏幕下方误当成可直接下降的深度指令。

因此在 `core/sim/zeroshot_robolab_runner.py` 增加恢复期空间闸：当 `_reacquire_required` 为真且当前 visual harness 证据尚未给出 `alignment_ready` 时，拦截 `APPROACH` 的 `MV_DOWN`，保留 Qwen 对当前帧的水平/深度修正决策；直到新鲜视觉关系对齐后才允许下降，`GRASP` 后清除闸。该改动不使用物体类别、固定坐标或绝对持握宽度，属于“观测—原子动作—再观测”的主线收紧。

这一段目前是待排查问题，不能简单增加 `max_steps` 解决；如果状态机条件不满足，更多步数只会重复同一动作。

## 六、历史失败与修复对应关系

### 放置失败后的恢复语义（更新）

这里的“恢复”不是把状态跳变回之前的快照，也不是重新执行旧的 GRASP。若瓶子仍被夹爪抓住、但落在篮口外侧或卡在篮沿，恢复应当遵循真实的连续动作：

`保持夹持 → MV_UP 脱离篮沿 → 新帧视觉复核 → X/Y 横向对准 → MV_DOWN → 新的放置验证 → 通过后才 RELEASE`

因此，任何 PLACE 验证 `NO` 都先强制执行一次 `MV_UP`，即使 Qwen 的 `recovery_action` 直接建议了横向动作；横向建议要等物体离开篮沿后才使用。恢复期间不 `RELEASE`，也不回滚到旧的 `GRASP`/`MOVE` 位姿。只有明确检测到丢持握，才进入另一条“重新定位—重新抓取”的恢复路径。

### 2026-09-27 当前排查结论：PLACE 需要逐原子动作复核

最新 rollout 显示，Qwen 能识别“物体在篮口右侧”，但同一个 PLACE 阶段随后会连续执行多次横向/下降动作，旧的 alignment 结论没有在每个原子动作后重新确认。这会把一次方向判断放大成持续偏移，并在接触篮沿后进入 `MV_UP → MV_DOWN` 循环。

因此 PLACE 现在采用逐步视觉伺服：每执行一个原子动作，下一步都重新检查 AgentView/Wrist；不缓存上一帧的对齐结论，也不把某个场景的方向或坐标写入规则。该改动属于 Show-Harness 原有“观测—原子动作—验证”范式内的闭环收紧，不改变任务、物体或篮子的先验。

本轮 rollout 进一步发现：Qwen 可能在一个图像方向上持续输出同一横向动作，而机器人几何回投已经显示该轴接近对齐、误差转移到另一轴。为避免单一语言方向把物体推过篮口，LIBERO 的通用视觉几何伺服现在也覆盖 PLACE：使用当前视觉目标回投与 EEF 世界坐标，在误差最大的轴上选择横向原子动作；两轴都进入当前配置的对齐容差后才允许下降。该逻辑不读取模拟器物体坐标，也不依赖瓶子/篮子类别。

### 2026-09-27 丢落物体恢复的逐帧结论

另一条 live run（`run_58f0218e4a0d`）的关键阶段是：`MOVE 143–243`、`PLACE 244–290`。截图显示瓶子在 288–289 仍靠近夹爪/篮口，执行 290 后在 291 已经横躺到篮子左前方桌面；这不是“已经放好但 verifier 漏检”，而是持握在连续横向修正中先丢失。

PLACE 阶段的实际原子动作序列为：

```text
MV_LEFT×3 → MV_DOWN×17 → MV_UP×1 → MV_LEFT×14
→ MV_DOWN×1 → MV_LEFT×10 → MV_DOWN×1 → lost_grasp
```

这里暴露了两个不同问题：

1. 篮子检测框长期贴在 AgentView 左边缘（例如 `x0=0`），但 host 仍把裁剪框中心回投到世界坐标；连续 `MV_LEFT` 时 EEF 的世界 X 也因篮沿接触发生漂移，说明此时不应继续贴着篮沿横移。
2. 丢落后进入 APPROACH 时，完整查询 `salad dressing bottle, green cap` 对横躺瓶子返回 0 检测。Qwen 仅凭图像和缺失的 host 框，多次把“瓶子在右侧/较低”误判成继续 `MV_DOWN` 或错误的横向动作，恢复没有方向闭环。

针对第二点，`VisualHarness` 已增加不依赖当前物体类别表的 SAM3 查询压缩 fallback：先用 planner 原始短语；无检测时从同一 target/affordance 生成属性+头名词短语（如 `green-capped bottle`）；仍无检测才尝试头名词，并继续执行候选歧义检查。实测同一失败帧上：完整短语 0 检测，`green-capped bottle` 命中正确框且 score 约 `0.414–0.449`，泛化查询 `bottle` 返回两个候选并保持 abstain。这使丢落后恢复可以重新得到当前物体 bbox，再由几何证据和 Qwen 决定横向/深度动作。

这次中断还暴露了实验基础设施风险：运行器理论上在 `finally` 中关闭 logger，但本次中断后 `run_58f0218e4a0d` 目录未保留在磁盘。因此后续长跑必须确认中断/结束后 `steps.jsonl`、PNG、视频和 `summary.json` 仍存在；失败证据不能只依赖终端输出。

| 现象 | 机制原因 | 当前处理 |
|---|---|---|
| Wrist abstain 后不抓 | 把局部不可见当成抓取不可能 | AgentView 可独立支持 GRASP，Wrist 为辅助 |
| 抓住瓶子却被判 OPEN | `open_width_m=0.03` 与真实宽度语义冲突 | 不再用固定持握宽度作高层真值；低层只识别明显空夹 |
| LIFT 在 UP/DOWN 间振荡 | 把“物体仍低于手”误读成继续下降 | LIFT 改为单调上升到 clearance band |
| MOVE 后 z 漂移 | 横向动作没有保持安全高度 | MOVE 阶段增加高度保持 |
| PLACE 横向修正后总是 DOWN | 控制器把水平 token 强制映射为下降 | 使用 Agent 推荐动作；每次修正后重新看图 |
| 放歪后直接 release 或卡死 | 没有独立 alignment/placed verifier | 增加 alignment review 与 placed verifier |
| verifier NO 后回到旧 GRASP | recovery index 过时 | 丢持握回退到最近 APPROACH |
| 多次恢复导致 LIBERO horizon crash | 内部 horizon 只按 motion 步数估计 | horizon 预算覆盖 gripper/recovery 动作 |
| 相似瓶子选错 | planner 目标描述太模糊、SAM3 盲选 | 增加颜色/瓶盖/形状属性和候选歧义记录 |

历史 run 仍保留在 `rollouts/`，不要删除失败轨迹。重点历史目录包括：

- `run_70fb97d729f8`：宽度阈值把真实持握误判为 open；
- `run_999692f75a40`：AgentView/Wrist 视角冲突和 LIFT 振荡；
- `run_b81a7e4108f9`：MOVE 方向/目标对齐问题；
- `run_dc9c0329add6`：MOVE 后高度漂移；
- `run_6c604f5b0bc7`：放置对齐恢复重复、丢持握、旧 recovery 回 GRASP；
- `run_04d965292647`：有 alignment review，但尚未加入下降停滞 verifier；
- `run_cd8aa076e0e0`：最新加入停滞 verifier 和 APPROACH 回退，仍未完成成功 episode。

## 七、换 Session 后的第一组操作

### 1. 先读取最新 rollout，而不是立刻改规则

```bash
python - <<'PY'
import glob, json, os

paths = glob.glob(
    'rollouts/libero_clean_qwen3vl_c3_threshold04_approachguard/**/steps.jsonl',
    recursive=True,
)
p = max(paths, key=os.path.getmtime)
rows = [json.loads(line) for line in open(p)]
print('latest:', p, 'rows:', len(rows))
for z in rows[-20:]:
    print(
        z.get('i'), z.get('stage'), z.get('act'),
        z.get('eef'), z.get('w'),
        z.get('place_alignment'), z.get('place_verification'),
        z.get('recover'),
    )
PY
```

同时检查对应的 `frames/` 或 debug payload，按时间序列看 AgentView/Wrist 截图。尤其不能只读 JSON 里的动作 token 来推断“已经对齐”。

### 2. 编译和 diff 检查

```bash
python -m py_compile \
  core/vlm/roles.py \
  core/agent/stage_control.py \
  core/sim/zeroshot_robolab_runner.py \
  core/sim/zeroshot_libero_runner.py \
  scripts/run_libero_zeroshot.py \
  plugins/recovery/plugin.py

git diff --check
```

### 3. 继续跑一个完整 episode

不要在看到中间一两次 `GRASP` 或一张“看起来放好”的图片时提前 interrupt。rollout 会持续生成 live frame；应至少观察到完整的：

```text
APPROACH → GRASP → LIFT → MOVE → PLACE → RELEASE → RETREAT
```

并以最终的 `success=True` 作为成功标准。如果中途失败，保留整个视频/截图序列，再根据“哪一帧开始偏离”定位，而不是只针对最后一个 token 打补丁。

## 八、明确不要重新引入的规则

- 不要恢复 Wrist-only 的 GRASP 必要条件。
- 不要重新引入按物体/episode 查表的夹爪持握宽度。
- 不要用固定 `0.14/0.16` 高度直接判定放置成功。
- 不要把任何 `MV_LEFT/MV_RIGHT/MV_FWD/MV_BACK` 无条件改成 `MV_DOWN`。
- 不要在没有新图像复核的情况下重复使用旧 alignment 结论。
- 不要把放置失败恢复写成固定方向；应让 Qwen 根据两个视角给出 `MV_UP`、横向修正或 HOLD。
- 不要把丢持握恢复到旧的 GRASP 快照；先回到 APPROACH 重新定位。
- 不要用模拟器内部物体坐标/任务 oracle 给模型直接补答案；可以使用机器人自身低层运动状态和截图做验证。

## 九、仍待完成的验收

1. 同一 task、同一 init state 得到至少一个 `success=True`。
2. 对另一个 `init_state_index` 重复，确认不是单帧/单轨迹偶然成功。
3. 检查丢持握后的重新定位是否能提交新的 GRASP。
4. 检查放歪场景是否真的走 `lift → fresh review → horizontal correction → down`，且不 release 错物。
5. 再扩展 task 0/2 或更多 object，验证没有因某个瓶子的高度、宽度或颜色写死规则。

本文件是换 session 的工作交接记录；历史详细实验仍见：

- `docs/capability_harness_progress_2026-09-26.md`
- `docs/sam3_abstain_root_cause_2026-09-26.md`
- `docs/libero_zero_shot_progress_2026-09-26.zh-CN.md`

## 十、2026-09-27：按“Agent 自主反思”重构放置失败恢复

最新一次被打断的 run 为 `run_2c4487ae6f81`。逐帧和 `steps.json` 显示：瓶子仍在夹爪中，MOVE 结束时篮子框已经贴到 AgentView 左边缘（`bbox=[0,99,54,155]`），EEF 投影与检测框中心仍有约 `[-18,27]px` 误差；Qwen 却在 `step=266` 输出 `DONE`。进入 PLACE 后，Qwen 的 alignment reviewer 已经能识别“太低/接近篮沿”，随后识别出“瓶子在篮口右侧”，但原 runner 仍有两处会把恢复流程写死：

1. MOVE 的 `DONE` 没有被新鲜视觉几何证据约束，导致未对齐就进入 PLACE。
2. verifier `NO` 后 runner 曾无条件先执行 `MV_UP`，这会替 Agent 决定“应该 lift、横移还是重新观察”。

本轮修改遵循“不做有限状态机”的要求：

- Qwen controller prompt 增加了每步的静默三项反思：物体是否仍在夹爪、物体 footprint 与开口是否真正重合/倾斜/碰沿、上一步是否改善关系；放置失败后由 Qwen 自己区分“仍持握”和“已掉落”，决定 lift、横向修正或重新 approach/grasp。
- harness 只提供可审计的视觉提示：当前开口与 EEF 的投影误差、bbox 是否被画面边缘裁剪；不替 Qwen 选择动作。
- MOVE 的 `DONE` 只在 host 明确有未对齐证据时被拒绝，拒绝后仍由下一次 Qwen 观察决定动作；未知证据不拦截，不使用物体类别/坐标写死规则。
- 删除 verifier `NO` 后 runner 的固定 lift-first 强制动作；现在只执行 Qwen verifier 从新图像给出的建议，下一步重新审查。
- 增加 whole-task visual verifier：计划结束但 LIBERO 尚未成功时，Qwen 从新鲜 AgentView/Wrist 判断是否真的放入、放歪、倾倒或丢失。若不完整，runner 将当前图像和失败原因交回 planner，由 Qwen 基于当前物体 silhouette 自主生成新的 approach/grasp/lift/place 计划；这是有上限的 Agent-driven replan，不是写死的 recovery 状态机。
- Qwen controller 开启短 CoT self-review（384 token 上限）；planner/verifier 的严格 JSON 仍关闭 thinking，避免结构化输出被思考文本冲掉。该 profile 允许最多 2 次 whole-task replan。

下一轮验收必须看完整视频和最终 `success=True`，重点核对：未对齐时 MOVE 是否被 Qwen 继续修正；放歪/倾倒后 final verifier 是否要求重新规划；新的 planner 是否能从当前 AgentView 重新定位瓶子并再次 GRASP，而不是回到旧抓取位姿。

随后短跑 `run_3b81e120e3a1` 又发现：日志中的 Qwen 原始 `decision=MV_RIGHT` 被实际执行层记录成 `token=MV_FWD`。根因是 `legacy_stage_token_normalization: false` 只关闭了最后一层旧规则，却仍无条件执行 `_normalize_visual_xy_token`。现已改为 clean profile 下保留 Agent 的方向 token，仅保留机器人高度安全保护、host 证据和完成约束。下一轮还观察到 EEF 已在 `0.105–0.115m` 最终抓取带内但 Qwen 仍在 `MV_DOWN/MV_UP` 间振荡；harness 现会明确提示“高度已在带内，屏幕下方是深度对齐候选，不是继续下降理由”。

`run_6039b3ad4da3` 的 GRASP 帧进一步说明“抓不住”并非单一原因：GRASP 后第 97 帧宽度约 `0.019m`，AgentView 可见瓶子在夹爪下方，之后 Qwen 在 LIFT 阶段连续输出 `MV_DOWN`，宽度从 `0.016→0.009→0.007m`，随后变为 `0.076m`，更符合向下压/失持后张开，而非 GRASP 原子动作本身必然失败。于是新增了 LIFT 阶段的 Agent 语义提示和下降安全拦截。

为增强 visual-centric 能力，`VisualHarness` 现在在 GRASP 首次 grounding 时同时对 AgentView 和 Wrist 做目标 grounding，并把两套关系分别记录：AgentView 的全局目标/EEF 关系、Wrist 的局部手指入口/目标关系；不会把一套视角覆盖另一套。若 Wrist 有目标但 AgentView 也在同一帧给出足够新鲜且对齐的全局证据，grasp guard 可以使用 AgentView 作为全局 close 证据，Wrist 仍作为局部补充。该机制不读取物体 pose，也不使用持握宽度表。

## 十一、2026-09-27：Visual Route Harness 架构实现

本轮不再增加单动作高度/方向补丁，新增独立 `plugins/visual_route/`。它使用已有 AgentView、SAM3/tracker、相机标定和 EEF proprioception，在 GRASP 后构造共享的 `CLEARANCE → TRANSFER → PRE_DESCENT → DESCENT` 路线；CPU 计算三段 waypoint，并将细蓝/绿路线、节点、红色 outer receptacle 框、橙色 opening 框和黄色 carried-object 框绘制到传给 Qwen 的 AgentView 副本，不叠加文字。目标容器的真实 rim/opening 仍保持可见；原始帧仍先交给 SAM3/tracker。

Qwen 只在路线生成、净空切换、进入下降和接触/停滞恢复等关键事件做 JSON review，不生成坐标，也没有部署第二个模型。overlay 回退到截图对应的轻量视觉提示：不画任何文字或图例，只画细蓝/绿轨迹、节点、红色 outer receptacle 框、橙色 opening 框和黄色 carried-object 框。路线坐标由 CPU 几何使用当前相机标定和视觉 bbox 投影；Qwen 负责关键帧 rethink 和动作选择，不负责手绘像素坐标。route gate 只记录 Qwen 的 requested/executed token、路线 phase 和证据摘要，不替换动作；低净空、rim 风险、对齐偏差都作为 Qwen 的反思输入，由 Qwen 自己决定继续上升、横移、下降或恢复。原有机械空夹 recovery 仍独立保留。所有 route、review、requested/executed token、gate reason 和 phase 都写入 `steps.jsonl`。配置关闭时 runner 不调用 route hook，保持严格 ablation。
路线对齐判据随后改为优先使用当前 opening bbox 的归一化尺寸和 held-object bbox 的相对关系，而不是跨场景复用固定像素误差；缺少 opening bbox 时不再把 outer receptacle bbox 冒充 opening 生成路线，而是让 route 低置信/无效并触发新鲜视觉确认。新增合成测试覆盖不同 opening 尺寸和缺 opening 的拒绝路径。

当前已通过本地 CPU 几何、renderer、disabled identity、CLEARANCE/TRANSFER/RECOVER_CLEAR gate 的直接测试；真实 LIBERO success rollout 仍是未完成验收项。
