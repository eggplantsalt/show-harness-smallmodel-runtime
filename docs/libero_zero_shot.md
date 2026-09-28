# 在现有 LIBERO 环境中运行 Show-Harness Zero-Shot

当前实验进度、真实 API rollout 记录和下一 session 接续步骤见：
[`libero_zero_shot_progress_2026-09-26.zh-CN.md`](libero_zero_shot_progress_2026-09-26.zh-CN.md)。

这条路径复用 Show-Harness 的完整 API 控制链：

`SubgoalPlannerAgent → ControllerAgent → plugins → LIBERO OSC_POSE → 成功判定`

它使用当前配置的 SiliconFlow 模型（默认是 `moonshotai/Kimi-K2.7-Code`），不走
finetune/MVTOKEN policy，也不读取 demonstrations 数据集。LIBERO 的 BDDL、资产和
`*.pruned_init` 初始状态已经在仓库中，因此第一个目标选用 `LIBERO_OBJECT` 的第 0
个任务：`pick up the alphabet soup and place it in the basket`。

## 1. 先确认现有目录

默认目录应当是：

```text
/root/autodl-tmp/Show-Harness
/root/autodl-tmp/OpenETA/vendor/LIBERO
/root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python
```

`Show-Harness/.venv-libero` 当前只是空的 Python 虚拟环境，不能作为 LIBERO 解释器。
因此不需要把 pip 指向它，也不需要重新 clone 或下载数据集。脚本会把 Show-Harness
checkout 和 LIBERO checkout 放入 `PYTHONPATH`，这是当前代码最少、最可复现的接法。

如果 LIBERO 位于别处，只需在运行前设置：

```bash
export LIBERO_DIR=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero-venv/bin/python
```

## 2. 检查 secrets

API key 只从 `configs/secrets.env` 的 `SILICONFLOW_API_KEY` 读取，配置文件没有保存
key。先确认该变量存在，值不要打印到终端：

```bash
cd /root/autodl-tmp/Show-Harness
grep -q '^SILICONFLOW_API_KEY=' configs/secrets.env
```

## 3. 运行一个 episode

推荐使用包装脚本，它会设置 MuJoCo/EGL、`LIBERO_DIR` 和 `PYTHONPATH`：

```bash
cd /root/autodl-tmp/Show-Harness
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
bash scripts/run_libero_zeroshot.sh
```

这条命令默认使用：

```text
suite: LIBERO_OBJECT
task_id: 0
init_state_index: 0
episodes: 1
max_steps: 100
camera: agentview + robot0_eye_in_hand
controller: OSC_POSE (7D)
```

需要换任务时，先保持同一个 suite，修改 `--task-id`：

```bash
bash scripts/run_libero_zeroshot.sh --suite-name LIBERO_SPATIAL --task-id 0
```

初始状态在一个任务内通常有 50 个，可以选择另一个状态：

```bash
bash scripts/run_libero_zeroshot.sh --task-id 0 --init-state-index 3
```

模型和后端可以临时覆盖：

```bash
bash scripts/run_libero_zeroshot.sh \
  --vlm-backend siliconflow_kimi_k27_code \
  --model moonshotai/Kimi-K2.7-Code
```

## 4. 看结果

每次调用都会创建一个独立目录，终端会打印 `run dir`。目录中最有用的文件是：

```text
summary.json             # success、steps、end_reason
subgoals.json            # planner 原始计划和子目标
planner_diagnostics.json # planner 重试/解析信息
steps.json               # 完整逐步记录（包含完整 reasoning）
steps.jsonl              # 截断 reasoning 的紧凑记录
images/agentview/*.png
images/wrist/*.png
rollout_success.mp4 或 rollout_failure.mp4
```

`success` 只由 LIBERO 的 `env.check_success()` 决定；模型输出 `DONE` 或达到步数上限
不会被记为成功。这样可以区分“模型结束了”与“任务真的完成了”。

## 5. 这次适配具体做了什么

`core/sim/libero_task.py` 负责定位 LIBERO、读取 BDDL 和 pruned init 状态、创建
`OffScreenRenderEnv`、转换旧式 4-tuple `step` 接口，并提供 RGB、EEF、夹爪宽度和成功
信号。它对 PyTorch 2.6 显式传入 `weights_only=False`，解决 LIBERO 老版本初始状态
文件与新 PyTorch 默认值不兼容的问题；这些文件来自当前 checkout，不是外部下载。

`interpreters/libero_atomic_controller.py` 把 `MV_* / GRASP / RELEASE` 映射到
LIBERO 的 7D OSC_POSE action。OSC_POSE 的位置尺度是 0.05 m，配置将一个约 2 cm 的
动作拆成 4 个控制步，并保持旋转增量为零。

`core/sim/zeroshot_libero_runner.py` 只替换仿真 I/O；planner、controller、
proprioception、recovery、variable-step、action-chunk、memory 和日志均沿用原有
Show-Harness 实现。`core/sim/zeroshot_robolab_runner.py` 增加了后端 hooks，所以
RoboLab 路径仍使用原来的 hooks，LIBERO 不复制一套 planner 逻辑。

## 6. 如果确实需要安装包

当前已验证的 LIBERO venv 已包含 `robosuite==1.4.1`、`mujoco==3.3.0`、PyTorch、PIL
和 gym，正常运行不需要 pip。若以后缺少某个包，先进入同一个解释器的环境并关闭
代理，再安装；不应把包装到空的 `.venv-libero`：

```bash
source /etc/network_turbo
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
/root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python -m pip install <package>
```

只有要从 HuggingFace 下载模型时才配置镜像；这条 LIBERO 评测路径本身不下载模型或
数据集。若网络必须走外部代理，再按机器已有的 `proxy_on.sh` 使用。

## 7. 先做不调用 API 的环境探针

要只检查 LIBERO 构造和相机，可以运行：

```bash
export LIBERO_DIR=/root/autodl-tmp/OpenETA/vendor/LIBERO
export PYTHONPATH=/root/autodl-tmp/Show-Harness:$LIBERO_DIR
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
/root/autodl-tmp/OpenETA/sim/venvs/libero/bin/python - <<'PY'
import numpy as np
from core.sim.libero_task import make_libero_task, step_libero
h = make_libero_task("LIBERO_OBJECT", 0, camera_height=64, camera_width=64)
obs = h.env._showharness_last_obs
print(obs["agentview_image"].shape, obs["robot0_eye_in_hand_image"].shape)
step_libero(h.env, np.zeros(7, dtype=np.float32))
h.env.close()
PY
```

看到两路 `(64, 64, 3)` 后，说明环境、资产、初始状态和 EGL 渲染链已经就绪；API
调用失败时应看对应 run 目录里的 `planner_diagnostics.json` 和 `debug_payloads/`，
不需要重新安装 LIBERO。
