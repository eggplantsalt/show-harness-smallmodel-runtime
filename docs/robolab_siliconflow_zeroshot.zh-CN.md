# Show-Harness 零样本 RoboLab：硅基流动 DeepSeek V4 Flash

本目录只处理 API 零样本路径：规划器先把自然语言任务拆成子目标，控制器在每一步读取 RoboLab 的前视图、腕视图和机器人自身状态，输出 Show-Harness 原子动作，再由 RoboLab 的官方任务成功谓词判定结果。它不加载、不比较也不修改微调模型。

当前默认任务是 `GrabAFruitTask`：从桌上抓起任意一个水果并离开桌面即可成功。这是一个单阶段抓取任务，适合作为“完整 API 闭环是否可用”的第一个目标。它仍是视觉抓取，不保证首次成功。

## 目录与职责

| 路径 | 用途 |
| --- | --- |
| `configs/robot_robolab_deepseek.yaml` | 唯一的零样本实验配置，选择 SiliconFlow 和 DeepSeek V4 Flash。 |
| `scripts/run_robolab_zeroshot.py` | 真正的 Show-Harness planner → controller → 原子动作闭环入口。 |
| `core/sim/zeroshot_robolab_runner.py` | RoboLab 执行器；只以 RoboLab 官方成功谓词判定成功。 |
| `scripts/robolab/deepseek/setup_env.sh` | 安装或补齐运行环境。 |
| `scripts/robolab/deepseek/preflight.py` | 不启动仿真的依赖检查；加 `--api` 才发送一个图像 API 请求。 |
| `scripts/robolab/deepseek/run_single.sh` | 运行一个 episode。 |
| `scripts/robolab/deepseek/eval_batch.sh` | 每个任务/seed 单独启动一个进程，保留所有失败和缺失结果。 |

## 1. 准备代码和系统环境

从当前工作目录开始：

```bash
cd /root/autodl-tmp
export ROBOLAB_ROOT=/root/autodl-tmp/RoboLab
cd Show-Harness
```

如果 RoboLab 尚未存在，按用户环境要求克隆：先加载 AutoDL 学术加速，克隆完成后再继续。不要在 `pip` 或 `uv pip` 前保留代理环境变量。

```bash
source /etc/network_turbo
git clone <你的 RoboLab 仓库地址> /root/autodl-tmp/RoboLab
cd /root/autodl-tmp/Show-Harness
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
```

RoboLab 需要 Python 3.11、NVIDIA GPU、Isaac Sim/Isaac Lab 和 `ffmpeg`。默认选择 RoboLab 的 `isaac50` 锁定环境；若你的现有 RoboLab 明确使用 `isaac51`，设置 `ISAAC_STACK=isaac51`。

```bash
bash scripts/robolab/deepseek/setup_env.sh
```

已经有可用 RoboLab `.venv` 时，不要重装 Isaac，只安装 Harness 客户端依赖：

```bash
bash scripts/robolab/deepseek/setup_env.sh --extras-only
```

这个安装脚本会在安装前取消 `http_proxy`、`https_proxy` 和大写代理变量，并优先设置 `HF_ENDPOINT=https://hf-mirror.com`。它不会启动仿真或调用任何模型。

## 2. 配置硅基流动密钥

模型配置已经写在 `configs/robot_robolab_deepseek.yaml`：

```yaml
base_url: https://api.siliconflow.cn/v1
model: deepseek-ai/DeepSeek-V4-Flash
api_key_env: SILICONFLOW_API_KEY
enable_thinking: false
```

创建只属于本机的密钥文件。文件不应进入仓库、命令历史或实验日志：

```bash
mkdir -p ~/.config/show-harness
cp configs/secrets.deepseek.env.example ~/.config/show-harness/siliconflow.env
chmod 600 ~/.config/show-harness/siliconflow.env
# 使用编辑器把 SILICONFLOW_API_KEY='' 中的空字符串替换成你的硅基流动 key
```

运行入口会自动加载此文件。也可以使用另一位置：

```bash
export SILICONFLOW_ENV_FILE=/安全路径/siliconflow.env
```

不要把 key 写进 YAML、`--model` 参数或 `metadata.json`。EpisodeLogger 会递归脱敏已知密钥字段。

## 3. 在不启动仿真的情况下检查

先阅读 NVIDIA Isaac Sim 的许可条款，然后明确接受：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source scripts/robolab/deepseek/common_env.sh
"$ROBO_PYTHON" scripts/robolab/deepseek/preflight.py
```

这一步检查 Python 3.11、RoboLab、Isaac 依赖、`libGLU.so.1`、GPU 驱动库、短手指资产、配置和密钥是否存在；它不启动 Isaac。

如要先验证 API 是否能接收一张合成 PNG，再运行：

```bash
"$ROBO_PYTHON" scripts/robolab/deepseek/preflight.py --api
```

这会产生一次付费 API 调用，但仍不启动仿真。输出应描述“左边红色方形、右边蓝色圆形”。该检查只验证联网、认证和图像传输，不能代表机器人成功。

## 4. 先成功一个 episode

启动单个固定任务 episode：

```bash
bash scripts/robolab/deepseek/run_single.sh
```

启动完成后，查看终端末尾的 `Success rate: 1/1` 和新生成目录中的 `summary.json`。只有该 JSON 的 `success: true` 才表示成功；模型自行输出 `DONE`、子目标耗尽、或进程返回 0 都不算成功。

若结果失败，先读该运行目录中的：

1. `summary.json`：`end_reason` 与官方成功布尔值。
2. `subgoals.json`、`planner_diagnostics.json`：规划器是否返回了可执行子目标。若出现 fallback，运行器会拒绝执行，而不会凭脚本伪造计划。
3. `steps.jsonl` 与 `images/agentview/`、`images/wrist/`：每一步模型看到的图像、动作和高度。
4. `debug_payloads/`：每一步发给 API 的提示与响应，适合核对 JSON 格式或图像传输。

不要通过拉长 RoboLab task 的 `episode_length_s` 来把超时变成成功；配置未覆盖任务作者定义的标准时限。也不要把物体位姿、目标对象名称或抓取成功状态塞进控制提示。控制器可用的附加信息只包括机器人自身的指尖高度、夹爪宽度和可选的机器人几何腕部十字标记。

## 5. 若单次失败，怎样做可解释调试

先只保存并检查相机和动作轴，不调用 API：

```bash
source scripts/robolab/deepseek/common_env.sh
"$ROBO_PYTHON" -u scripts/run_robolab_mvtoken.py \
  --robot-config configs/robot_robolab_deepseek.yaml \
  --task GrabAFruitTask --dump-views --probe-axes --no-rollout \
  --log-dir rollouts/robolab_diagnostics
```

检查 `views/wrist_sent.png`：手指应位于画面顶部；`agentview_sent.png` 应能看见候选水果。再看 `calibration.json`：六个 `MV_*` 的测得位移必须与相应坐标方向匹配，且约为每个决策 2 cm。方向错、位移明显不对、或相机看不到目标时，先修相机/坐标配置，不能通过提示词掩盖。

`wrist_grasp_marker.enabled` 默认关闭。开启它只会在腕图画一个由固定手部/相机几何计算的青色十字；它不是检测器，且当前标记注明 `runtime_validated: false`。只有对比诊断图确认位置后才可设为 `true`。

## 6. 批量评测与成功率

在单 episode 真正成功之后再开始批量评测：

```bash
bash scripts/robolab/deepseek/eval_batch.sh --episodes 10 --task GrabAFruitTask \
  --output results/robolab_siliconflow
```

脚本会为每个 episode 启动一个新的 Isaac 进程并在启动前写 `manifest.json`。它不会从终端文本猜成功，也不会跳过缺失 `summary.json`：崩溃、API 失败和缺失 summary 都保留在预期分母内。结果在该批次的 `summary.csv`、`summary.json` 和 `manifest.json`。

RoboLab 许多任务使用作者写死的 USD 初始物体布局。不同 seed 是独立的过程和随机流，但不自动等于不同物体布局。报告时应写“固定布局的独立过程重复”，除非你另外启用并验证了任务布局随机化。

## 常见错误

| 现象 | 首先检查 | 处理 |
| --- | --- | --- |
| `SILICONFLOW_API_KEY is unset` | `~/.config/show-harness/siliconflow.env` | 确认变量名、文件权限和 `SILICONFLOW_ENV_FILE`。 |
| 401/403 | 密钥、账户权限 | 在硅基流动控制台重新生成或确认该模型权限；不把密钥贴进日志。 |
| 429/503/504 | `x-siliconcloud-trace-id` 与运行日志 | 运行器会退避重试；仍失败时保留失败 episode，稍后新开批次。 |
| Isaac 在启动时崩溃 | `preflight.py` 的 `libGLU.so.1`、Python 与 Isaac 版本 | 先补齐 `libglu1-mesa`，且只安装一个 `isaac50`/`isaac51` 栈。 |
| 所有动作不动 | `calibration.json` | 检查时间线、GPU、相对 IK 注册和 `move_vectors`。 |
| 一直空抓 | 两张诊断图与指尖高度 | 检查腕相机方向、抓取点投影和粗/细步长；不要用任务真值强制 `GRASP`。 |
| API 有文本回复但图像理解差 | `preflight.py --api` 与每步 `debug_payloads` | 确认 V4 Flash 的当前账户视觉能力；若平台拒绝视觉内容，该模型/端点不能驱动本视觉闭环。 |

## 当前边界

这套代码恢复并整理了零样本 Harness 的完整 planner/controller/plugin 路径，并改为硅基流动 V4 Flash。它尚未用你的硅基流动密钥在当前检查点实际跑过 RoboLab；因此文档中的“成功一个 episode”是下一步要执行的验证目标，不是已经宣称的实验结果。
