# Show-Harness × LIBERO Zero-Shot 实验进度

> 最新的 Qwen8B 实验状态、PLACE 对齐/放置验证、SAM3 与 AgentView/Wrist 结论，以及换 session 的执行清单，见 [show_harness_qwen8b_progress_2026-09-27.zh-CN.md](show_harness_qwen8b_progress_2026-09-27.zh-CN.md)。本文保留早期实验历史。

更新时间：2026-09-26  
当前目标：使用 Show-Harness 的完整 planner → controller → plugins → LIBERO 仿真闭环，在简单 LIBERO 任务上完成至少一个成功 episode。当前只评估 API zero-shot 路径；finetune/MVTOKEN 路径不在本实验范围内。

## 先确认代码归属

被评估和修改的项目是：

```text
/root/autodl-tmp/Show-Harness
```

LIBERO 的代码和资源只作为已经存在的仿真环境供应方：

```text
/root/autodl-tmp/OpenETA/vendor/LIBERO
/root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python
```

Show-Harness 负责入口、配置、VLM API、planner、controller、插件、轨迹日志和成功率统计。OpenETA 没有承担 Show-Harness 的推理逻辑，也没有把实验迁移到 OpenETA；它只提供 vendored LIBERO、BDDL、`*.pruned_init` 和已安装的 robosuite/MuJoCo Python 环境。

## 当前运行入口

推荐从 Show-Harness 根目录执行：

```bash
cd /root/autodl-tmp/Show-Harness
unset http_proxy && unset https_proxy
unset HTTP_PROXY && unset HTTPS_PROXY
export DEBUG=1

bash scripts/run_libero_zeroshot.sh \
  --vlm-backend siliconflow_qwen35_35b \
  --suite-name LIBERO_OBJECT \
  --task-id 1 \
  --init-state-index 0 \
  --episodes 1 \
  --prompt-log-every 1 \
  --debug
```

当前默认配置文件是 [`configs/robot_libero.yaml`](../configs/robot_libero.yaml)，默认 suite/task 是 `LIBERO_OBJECT/task 0`；实验命令用参数覆盖到了 task 1。每次运行的结果在：

```text
rollouts/libero_zeroshot/run_<invocation>/LIBERO-LIBERO_OBJECT-<task>/0925/task_0/<time>/
```

重点文件：`summary.json`、`steps.jsonl`、`steps.json`、`images/agentview/`、`images/wrist/`、`controller_prompts/`、`debug_payloads/` 和 `rollout_failure.mp4`。

## 当前 Show-Harness LIBERO 适配

已加入或修改的主要文件：

| 文件 | 作用 |
| --- | --- |
| [`core/sim/libero_task.py`](../core/sim/libero_task.py) | 定位现有 LIBERO checkout，读取 BDDL/初始状态，创建 OffScreenRenderEnv，适配 RGB、TCP、夹爪宽度和 `check_success()`。不下载 demonstration 数据集。 |
| [`core/sim/zeroshot_libero_runner.py`](../core/sim/zeroshot_libero_runner.py) | 复用 Show-Harness 的 staged zero-shot runner，只替换 reset、step、RGB、proprioception 和成功判定。 |
| [`interpreters/libero_atomic_controller.py`](../interpreters/libero_atomic_controller.py) | 将 `MV_* / GRASP / RELEASE` 映射到 LIBERO robosuite OSC_POSE 的 7D action。 |
| [`scripts/run_libero_zeroshot.py`](../scripts/run_libero_zeroshot.py) | LIBERO 入口，调用 Show-Harness 的完整 planner、controller、plugins 和 EpisodeLogger。 |
| [`scripts/run_libero_zeroshot.sh`](../scripts/run_libero_zeroshot.sh) | 设置 `LIBERO_DIR`、`PYTHONPATH`、EGL 和正确的 LIBERO venv。 |
| [`scripts/run_robolab_zeroshot.py`](../scripts/run_robolab_zeroshot.py) | 共用 zero-shot stack；已增加 backend-specific common context、是否合并 pre-grasp stage 的参数。RoboLab 默认行为保持不变。 |
| [`plugins/subgoal/plugin.py`](../plugins/subgoal/plugin.py) | `SubgoalPlanner` 新增 `merge_pregrasp` 开关；LIBERO 关闭接近/抓取合并，RoboLab 默认仍开启。 |
| [`core/sim/zeroshot_robolab_runner.py`](../core/sim/zeroshot_robolab_runner.py) | 新增 backend 可覆盖的 `_normalize_stage_token()` hook；默认不改 VLM token。 |
| [`configs/robot_libero.yaml`](../configs/robot_libero.yaml) | LIBERO 相机、OSC_POSE、动作轴、插件和 API 后端配置。 |

LIBERO 入口仍然使用 Show-Harness 的这些模块：`SubgoalPlannerAgent`、`ControllerAgent`、`RecoveryPlugin`、`VariableStepPlugin`、`ActionChunkPlugin`、`MemTextPlugin`、EpisodeLogger、JSON/debug prompt 记录和官方环境成功谓词。

## 当前配置要点

- API 默认后端：`siliconflow_qwen35_35b`。
- 模型：`Qwen/Qwen3.5-35B-A3B`，通过 SiliconFlow OpenAI-compatible API 调用；这是当前要求的多模态模型。
- Kimi 后端 `siliconflow_kimi_k27_code` 仍保留，便于对照，但不是当前默认。
- `gemini_flash` 配置曾加入过，但当前 `GEMINI_API_KEY` 返回 `HTTP 400 INVALID_ARGUMENT: Please pass a valid API key`，不要把 Gemini 当作可用后备。
- SiliconFlow 曾出现 `HTTP 402 account balance is insufficient`；之后 Qwen 请求已经恢复，最近几次规划请求成功。若再次出现 402，先记录并停止无意义重试，不要修改任务逻辑来掩盖 API 额度问题。
- LIBERO AgentView 使用 `rotation_degrees: 180`；原始 wrist view 已经是 fingers 从图像顶部进入的训练方向，因此 `wrist_rotation_degrees: 0`。
- 当前相机校准动作轴：

  ```yaml
  MV_FWD:   [ 1, 0, 0]
  MV_BACK:  [-1, 0, 0]
  MV_LEFT:  [ 0, 1, 0]
  MV_RIGHT: [ 0,-1, 0]
  MV_UP:    [ 0, 0, 1]
  MV_DOWN:  [ 0, 0,-1]
  ```

  这是根据 LIBERO AgentView 的 upright 变换和实际 Panda world frame 推导的：画面下方对应 `+X/MV_FWD`，画面右方对应 `-Y/MV_RIGHT`。
- `subgoal: true`，LIBERO 关闭 pre-grasp merge，要求 planner 产生独立 `APPROACH → GRASP` 阶段。
- `proprioception: false`。原因是通用 RoboLab proprioception prompt 中的“高度超过 8 cm 就先 `MV_DOWN`”会让 LIBERO 模型从初始 25 cm 高度连续撞向桌面，完全不做水平对齐。
- `recovery`、`variable_step`、`action_chunk`、`mem_text` 仍开启。
- `ZeroshotLiberoRunner` 有一个保守的 stage hook：APPROACH 阶段若模型在高度超过 10 cm 时输出 `MV_DOWN`，先按 LIBERO 的 global-view 规则执行 `MV_FWD`；GRASP/PLACE 阶段不改 vertical token。这个 hook 只在 LIBERO 子类生效。

## 已完成的环境验证

以下均已通过：

1. LIBERO `OffScreenRenderEnv` 可以创建并加载 `LIBERO_OBJECT` task 0/task 1 的 BDDL 与 pruned init state。
2. 256×256 AgentView 和 wrist RGB 可以渲染、保存、送入 VLM。
3. 7D OSC_POSE action 能推动 TCP，GRASP/RELEASE 能改变夹爪宽度。
4. `env.check_success()` 能作为唯一成功判据。
5. `python3 -m compileall -q Show-Harness`、入口 `--help`、LIBERO env smoke 和 dummy runner smoke 已通过。
6. 日志和视频会在失败时正常写入，API 返回的 planner/controller 原始 JSON、reasoning 和 prompt 都可追溯。

## 真实 API episode 记录

目前还没有成功 episode；所有失败都是可复现的控制/视觉对齐问题，不是环境启动问题。

| run | 设置/现象 | 结论 |
| --- | --- | --- |
| `run_a79090de4a4a/.../21-53-47` | task 0，wrist 错误旋转 180°；模型在 GRASP 中上下振荡，50 步没有 `GRASP`。 | 找到 wrist 方向错误。 |
| `run_549cba1d13fc/.../21-59-01` | wrist 修正为 0°；task 0 仍没有接近并抓取。 | 方向修正后仍有水平轴/任务难度问题。 |
| `run_df6f35fdb3e9/.../22-03-23` | task 1，旧 FWD/BACK 映射；轨迹证明模型的 `MV_FWD` 把 TCP 从目标方向推开。 | 确认需要 LIBERO 专用轴校准。 |
| `run_9c2da639a6ab/.../22-07-41` | FWD/BACK 临时翻转；TCP 沿 `-X` 更远离目标。 | 证明临时翻转是错的，已恢复 `FWD=+X`。 |
| `run_f40e3412e02a/.../22-13-04` | 轴向恢复正确；planner 直接给 GRASP，模型一直输出移动，不提交 GRASP。 | 发现 pre-grasp merge 和 GRASP 阶段混用是主要问题。 |
| `run_47a29d6bf22f/.../22-31-33` | 关闭 merge 前的后续尝试；模型仍受高度提示影响，反复下降。 | 发现 proprioception height-first prompt 与 LIBERO 冲突。 |
| `run_bd734b606973/.../22-37-12` | 独立 APPROACH，但仍连续 `MV_DOWN`，50 步撞到/接近桌面。 | 关闭 LIBERO 的 proprioception prompt。 |
| `run_fb3cf3fc450c/.../22-45-53` | 恢复原 controller prompt，Qwen 请求成功；模型能输出 FWD/LEFT/BACK，但在 APPROACH 中来回，未进入 GRASP。 | 当前最新基线；剩余问题是 Qwen 对 LIBERO 俯视图的方向判定不稳定。 |

上表的 run 目录均在 `/root/autodl-tmp/Show-Harness/rollouts/libero_zeroshot/` 下。不要删除失败 rollout，它们是下一轮分析视觉方向和 token 选择的证据。

## 下一 session 应从这里继续

### 1. 先确认最新代码和 Qwen API

```bash
cd /root/autodl-tmp/Show-Harness
unset http_proxy && unset https_proxy
unset HTTP_PROXY && unset HTTPS_PROXY
python3 -m py_compile \
  scripts/run_libero_zeroshot.py \
  core/sim/zeroshot_robolab_runner.py \
  core/sim/zeroshot_libero_runner.py \
  plugins/subgoal/plugin.py
```

### 2. 先尝试更近的简单目标

当前 task 1 的 cream cheese 并不适合当作“最简单”任务。LIBERO_OBJECT 的 task 文件按 benchmark 顺序为：

```text
0 alphabet soup
1 bbq sauce
2 butter
3 chocolate pudding
4 cream cheese
5 ketchup
6 milk
7 orange juice
8 salad dressing
9 tomato sauce
```

注意：CLI 的 task id 使用 LIBERO benchmark order，不要仅按 BDDL 文件名字的直觉判断。建议先写一个只读 probe，打印每个初始状态中 task 目标物体与 eef 的 XY 距离，再选最近的 task；不要把目标物体状态直接注入 VLM prompt。

### 3. 跑单集并检查轨迹

```bash
bash scripts/run_libero_zeroshot.sh \
  --vlm-backend siliconflow_qwen35_35b \
  --suite-name LIBERO_OBJECT \
  --task-id 2 \
  --init-state-index 0 \
  --episodes 1 \
  --prompt-log-every 1 \
  --debug
```

然后读取最新 run 的 `summary.json` 和 `steps.jsonl`，重点检查：

- 是否先出现 `APPROACH`，而不是 planner 直接进入 GRASP；
- `MV_FWD` 后 TCP 的 X 是否朝目标方向变化；
- `MV_RIGHT` 后 TCP 的 Y 是否朝目标方向变化；
- 模型在目标已经位于两个指尖之间时是否真正输出 `GRASP`；
- GRASP 后夹爪宽度是否变小且环境成功谓词仍为 false/true；
- LIFT/MOVE/PLACE 是否使用 AgentView 而不是把 wrist 局部图当成全局地图。

### 4. 继续修复的优先级

1. **先修方向判定，不要先加步数。** 最近失败中模型反复输出 `MV_BACK` 或左右交替；增加 `max_subgoal_steps` 只会让错误轨迹更长。
2. 检查 `controller_prompts/` 中完整 prompt 与 `debug_payloads/` 的原始 reasoning，确认模型是否看到了正确的两张图和动作 token。
3. 若 Qwen 在多个最近目标上仍把画面下方/左方映射成错误 token，优先做一个 LIBERO 专用、只改方向说明的 prompt ablation，并保留原 prompt 作为对照；不要修改 Show-Harness 的 RoboLab 默认 prompt。
4. 若方向正确但始终不提交 `GRASP`，再增加一个只依赖视觉完成条件和当前夹爪宽度的 GRASP commit guard；不能读取 LIBERO 物体坐标或任务内部状态来直接作弊。
5. 一旦得到一个成功 episode，先用同一 task 的另一个 `init_state_index` 复现，再扩展到 task 0/2，记录 `success/episodes`，不要直接宣称 suite 成功率。

## 不要重复走的弯路

- 不要把 Show-Harness 的 LIBERO 实验代码放到 OpenETA；OpenETA 只应作为环境供应方。
- 不要把 `wrist_rotation_degrees` 改回 180°；LIBERO 原始 wrist 图已经是 fingers 在顶部。
- 不要恢复通用 `proprioception` 的 8 cm height-first 提示；它已被真实 rollout 证明会导致连续下降。
- 不要把 task planner 的 pre-grasp merge 重新打开；LIBERO 需要独立 APPROACH 和 GRASP 以避免高处用 wrist 微调。
- 不要把 Gemini 当作可用 API：当前 key 无效。Qwen 35B 是当前实验模型；Kimi 只作为保留的对照后端。
- 失败 episode 的 `success=false` 必须保留；LIBERO 成功只能来自 `env.check_success()`，不能用模型的 `DONE` 或“看起来抓住了”替代。
