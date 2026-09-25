# ScienceWorld 正式运行指令

所有命令在 **WSL Ubuntu** 执行。输出必须放在 Linux home，不放 `/mnt/d`。
复用本机已配置好的 `/home/yangchengyu/asg_scienceworld_venv`，不需要重装环境。
以下命令只供用户启动；交付验证没有启动正式 Train120/Test90。

## 1. Learned Ours：三个 seed 并行

每个 seed 独立从空 Bank 开始：Train120 → Train-only Compiler → 只读 Dev10 → 同一 Frozen Test90。
seed42 的 Test 顺序重复三轮；seed43、44 各一轮。没有跨 seed 的 Train/Test barrier。

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
set -a; source .env; set +a
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
SW_RUN="$HOME/sw_learned_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SW_RUN"
nohup /home/yangchengyu/asg_scienceworld_venv/bin/python \
  -m experiments.scienceworld_campaign \
  --root "$SW_RUN" --seeds 42 43 44 --workers 3 \
  > "$SW_RUN/supervisor.log" 2>&1 &
echo "PID=$!  ROOT=$SW_RUN"
```

各 seed 的 `train.log`、`dev.log`、`test_repeat1.log` 等位于 `$SW_RUN/seed42/` 等目录。
完整结果位于各阶段目录；Bank 为 `seed42/compiled/data_v3/` 等。
Train 原库 `seed42/train/data_v3/` 保留不改写。Dev 不能产生新资产或修改偏好。

```bash
tail -F "$SW_RUN"/seed*/train.log
```

只暂停 seed42（当前题完成后停）：

```bash
touch "$SW_RUN/seed42/STOP_AFTER_TASK"
```

确认原 supervisor 已退出/该 seed 已停止后，恢复同一路径：

```bash
rm "$SW_RUN/seed42/STOP_AFTER_TASK"
/home/yangchengyu/asg_scienceworld_venv/bin/python \
  -m experiments.scienceworld_campaign --root "$SW_RUN" --seeds 42 --workers 1 --resume
```

`--resume` 不会重跑已完成阶段。不静默吞掉中断请求的未知用量；强杀中断需要按日志保留证据处理，不能当作 task-boundary 暂停。
不同终端重新设置 `SW_RUN` 为第一次输出的实际目录，并重新执行环境加载三行。
不要在运行中修改源码、模型配置或 `.env` 中的 provider 配置。

## 2. 手工 Frozen Bank：独立 Test，无 Train

已经存储的独立 reference Bank：

```text
/home/yangchengyu/sw_authored_final_20260925_r3/frozen/data_v3
```

Windows 资源管理器：

```text
\\wsl.localhost\Ubuntu\home\yangchengyu\sw_authored_final_20260925_r3
```

这个 Bank 与正式 learned Train 完全隔离，结果标记 `authored_reference`。
共登记 108 项（A01–A30 的 A/I/T 三层、G01–G18）；通过真实资格验证的 76 项为 Active：
25 Atomic + 25 Implementation + 25 Tool + G03。其余 32 项保留 Draft，不作为可用资产。
A13/A24/A25/A27/A29 与其配套 I/T，以及其余 17 个 G **没有获得虚构的执行信用**。
G 是 partial capability，不能冒充官方 score-100 完整合同或 P0。
详细记录见该目录 `audit/release_checks.json`、`real_jvm_replays.json`、`zero_llm_workflows.json`。

下面只运行这个已存储 Bank 的只读 Test：三个 seed 并行，42 顺序三轮，43/44 各一轮。

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
set -a; source .env; set +a
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
SW_REFERENCE="$HOME/sw_reference_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SW_REFERENCE"
nohup /home/yangchengyu/asg_scienceworld_venv/bin/python \
  -m experiments.scienceworld_reference_test \
  --bank /home/yangchengyu/sw_authored_final_20260925_r3/frozen/data_v3 \
  --root "$SW_REFERENCE" --seeds 42 43 44 --workers 3 \
  > "$SW_REFERENCE/supervisor.log" 2>&1 &
echo "PID=$!  ROOT=$SW_REFERENCE"
```

监控 `tail -F "$SW_REFERENCE"/seed*/test_repeat1.log`。
暂停/恢复同上使用每个 seed 的 `STOP_AFTER_TASK`，原命令复用**同一 root**并加 `--resume`。
此 launcher 不运行 Train，不会把手工 Bank 注入 learned Ours，也不会重建/改写 Bank。

## 3. Baseline 五条线路

继续使用本机 baseline 代码目录。B0/B1 无学习阶段；B3/B4/B5 保留各自原算法的训练/选择/冻结流程。
所有方法 seed42 Test 三轮、seed43/44 各一轮。最多并发三个 method/seed lane，不是同时开十五个 JVM。

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
set -a; source .env; set +a
export PYTHONPATH="$PWD/src:$PWD:$PWD/.external/skillopt:$PWD/.external/gepa/src" PYTHONUNBUFFERED=1
SW_BASELINES="$HOME/sw_baselines_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SW_BASELINES"
nohup /home/yangchengyu/asg_scienceworld_venv/bin/python \
  -m experiments.baselines.scienceworld.campaign \
  --root "$SW_BASELINES" --seeds 42 43 44 --workers 3 \
  > "$SW_BASELINES/supervisor.log" 2>&1 &
echo "PID=$!  ROOT=$SW_BASELINES"
```

每条线路日志为 `$SW_BASELINES/b3_skillopt_seed42.log` 等；完整结果在对应 method/seed 目录。
续跑用同一路径加 `--resume`，不能换 root。只选部分线路可加 `--methods b3_skillopt b5_gepa`。

建议三个实验组分别启动，不要一次同时打开三组的九个 lane；三组之间互不依赖。
如机器资源或 API 并发受限，把 `--workers 3` 改为 `--workers 1`，实验定义不变。
