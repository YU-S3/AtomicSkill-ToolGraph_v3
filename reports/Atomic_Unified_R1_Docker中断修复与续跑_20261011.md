# Atomic Unified R1：Docker 中断修复与续跑

核对时间：2026-10-11 12:38（北京时间）。四个实验均使用独立 Bank、DeepSeek v4 Flash、seed42；实际受测源码固定为 `80762b4c84fc71d56a42d9783f1fdd3e70ce43ab`。本次恢复没有修改 Ours 方法实现、配置、模型提示、晋升标准或预算，未启动新模型请求。

## 实际进度

计数来自各阶段 `run.sqlite3` 中 `status=completed` 的记录。失败日志中的重复 error episode 不算完成题。

| Benchmark | Train | Val | Test | 状态 |
|---|---:|---:|---:|---|
| SearchQA | 300/300 | 24/24 | 1400/1400 | 完成；Test 1139/1400，81.36% |
| LiveMath | 60/60 | 17/17 | 25/100 | 原 Test 第 26 题中断，已准备续跑；已完成 Test 正确 3/25，此数不是全 Test 准确率 |
| OfficeQA | 120/120 | 24/24 | 102/102 | 完成；Test 3/102，2.94% |
| SpreadsheetBench | 77/200 | 0/20 | 0/180 | 原 Train 第 78 题中断，已准备续跑；已完成 Train 正确 51/77 |

OfficeQA 完整结束，但准确率低。低分是现有评分结果，本次没有基于结果调整方法、复跑已完成题或改变评分。

以下为总 Governor 实际已计量用量，包含上一次启动时结转的旧前缀费用，也包含中断题已经发出的请求，不仅是已完成题费用：

| Benchmark | Tokens | HTTP attempts | 原冻结 token / HTTP 上限 |
|---|---:|---:|---:|
| SearchQA | 8,106,584 | 2,189 | 196,704,000 / 83,580 |
| LiveMath | 3,004,640 | 167 | 23,232,000 / 9,165 |
| OfficeQA | 10,221,986 | 1,069 | 36,096,000 / 284,070 |
| SpreadsheetBench | 5,801,296 | 616 | 59,200,000 / 558,000 |

合计 27,134,506 tokens、4,041 次 HTTP。四个 Governor 均为全部 attempt 已计量，`unknown_billing=false`；本次恢复前后账本哈希不变。阶段 summary 的 unknown 计数口径包含未发出 HTTP 的逻辑请求，本报告费用使用实际 HTTP Governor 账本，不将缺失字段猜填为用量。

## 已确认故障及处理

两个未完成实验均在 Docker 创建容器时停止：`Docker control failed` → `OCI runtime create failed` → `no space left on device`。最小只读容器也能复现同样错误。

故障时 Docker Desktop WSL 命名空间有 **99,965 个挂载**，内核 `fs.mount-max` 为 **100,000**。重复的 Ubuntu bind mount 路径大量累积。Ubuntu 文件系统剩余约 746 GB，inode 使用约 1%；该报错对应挂载资源耗尽，而非实验输出把磁盘写满。

停止故障中的 Docker Desktop 后重新启动其 WSL 后端，清除了累积挂载。启动时另遇到两个残留的 Windows socket：`docker-secrets-engine/engine.sock` 和 `Docker/run/dockerInference`。已确认其目录各仅有一个零字节 socket，在 Docker 进程停下时将两个目录改名备份，再创建空的 IPC 目录启动 Desktop。备份为：

- `C:\Users\YCY\AppData\Local\docker-secrets-engine.stale-20261011`
- `C:\Users\YCY\AppData\Local\Docker\run.stale-20261011`

没有执行 factory reset、prune、删除镜像、清空实验目录或调整隔离限制。Docker 已恢复，真实 Worker 验证后挂载数为 **137/100,000**。

新启动脚本增加启动前挂载余量检查，达到上限的 80% 时拒绝启动；也会创建一个锁定镜像的只读容器检查 Docker 可执行性。这次已消除当前挂载耗尽，尚未声称修复 Docker Desktop 长期累积 bind mount 的实现。后续大批次运行应检查挂载数，必要时在实验停止后重启 Desktop。

## 保留现场的恢复方式

不能直接把中断 Program 视为执行完成，也不能清除 unknown 后重放原生事件。LiveMath 旧 checkpoint 有 `program_started=true`，Spreadsheet 旧 `execute_python` 原生事件为 unknown；原 runner 正确拒绝盲目继续同一个 attempt。

两个中断题都在 `task_started`，没有评分，也没有该题的 Learner/Builder 更新。已通过只读检查确认其 adapter 支持从原公开输入重建：LiveMath 为 single answer，Spreadsheet 为独立 workspace copy。

| Cell | 中断题 | 保留的 attempt | 下次执行 |
|---|---|---:|---:|
| LiveMath / Test | `livemath:202512:3` | 1 | 2 |
| SpreadsheetBench / Train | `spreadsheet:43589` | 1 | 2 |

已备份原 SQLite、completion、run manifest、budget、请求审计和全部该题 checkpoint。唯一运行状态变动是将这两个无结果的 SQL task 行由 `running` 标记为 `interrupted_infrastructure`；attempt 计数仍为 1，结果仍为空。原 `--resume` 在下次启动时自动建立 attempt 2，恢复公开初始输入，跳过全部已完成题。

Bank、Frozen、已完成任务结果、旧 checkpoint、旧请求文件及 budget 的哈希均经过恢复前后比对，保持一致。没有增加 repair 次数或最多 3 次 task attempt 限制。新 attempt 仍使用相同 physical task 的 parent budget scope，已花费的本题 tokens 不会清零；普通模型失败仍按原规则记分。

每个 cell 的恢复证据：

```text
/home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/
  livemath/seed42/infrastructure_recovery/28f4618674b80f05_attempt1/
  spreadsheetbench/seed42/infrastructure_recovery/9207dbdd2724a87d_attempt1/
```

各目录包含 `prepared.json`、`receipt.json`、原始 SQLite/请求/账本及 `original_checkpoint/`。旧 `completion.json` 中的失败状态保留为历史事实，用户执行续跑后才会更新正式运行状态。本次核对结束时没有正在执行的实验进程。

## 验证

- 无 LLM 恢复回归 2 项通过：使用真实 runner、EmpiricalSystem、checkpoint、SQLite 和 Governor，仅拦截 HTTP；验证普通 resume 拒绝原中断执行、已完成前缀不重跑、新 attempt 执行、旧审计和 Bank 保留、计费与 parent scope 连续，以及非此类故障不得使用本次恢复。
- 真实 Docker Worker 检查 5 项通过：正常 RPC、预算共享、沙箱隔离、三种文件发布声明情况。
- `bash -n` 与新启动脚本 `--print` 通过，校验固定源码、旧结果、旧 checkpoint、预算、恢复 receipt、无活动实验、Docker 挂载余量及实际容器启动。
- 所有验证合计新增付费 HTTP 为 **0**；未执行正式实验 `--execute`。

本机工程材料：`D:\T3S_exp\deliveries\outputs\atomic_unified_r1_recovery_20261011\`，包含恢复工具、拦截 HTTP 测试、JUnit、现场快照、恢复 receipt 及 `final_status.json`。

## 新指令

在 Ubuntu WSL 中执行，**仅并行续跑两个未完成实验**，其余两个完整结果保留：

```bash
nohup bash /home/yangchengyu/start_atomic_unified_r1_selector_seed42_resume_20261011.sh --execute \
  > /home/yangchengyu/atomic_unified_r1_selector_seed42_resume_20261011.launch.log 2>&1 < /dev/null &
```

此脚本对应仓库 `scripts/resume_atomic_seed42_docker_20261011.sh`，已安装到上述 WSL 路径；它调用原干净的 `80762b4` 源码，而非随本报告更新的 main checkout。两条 CLI 都使用原输出目录和 `--resume`，没有额外前缀数量限制、stop-after-val 或新 Bank。

查看新启动检查：

```bash
cat /home/yangchengyu/atomic_unified_r1_selector_seed42_resume_20261011.launch.log
```

查看两项后续逐题进度（日志追加，旧报错保留）：

```bash
tail -n 10 -F \
  /home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/logs/livemath.seed42.log \
  /home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/logs/spreadsheetbench.seed42.log
```

不要重新执行旧的四项 fresh launcher，也不要改到最新源码目录给旧 run 宣称原样 resume。
