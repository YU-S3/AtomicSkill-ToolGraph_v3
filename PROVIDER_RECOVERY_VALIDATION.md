# Provider response recovery — 2026-09-15

## 原因与范围

seed42 的 `train_36` 在 RuntimePreparation 收到 HTTP 200，但 `response.json()` 的结果不是 object；原适配器记录 `provider_invalid_response` 后立即退出。该次请求用量 unavailable，Trace 与 attempt capture 完整。中断前已完成 35/120，30 成功、5 失败。API 端返回内容的具体原因不能从现有脱敏记录推断。

这属于外部 provider 基础设施故障；本次修复客户端重试、审计和恢复边界，不改任务语义、实体选择、R0/static/R1、Planner/Runtime 策略、模型、reasoning、token/action budget 或任务清单。

## 行为

- 无效 JSON、非 object JSON、缺失/无效 usage 的不可用响应，在原 `max_retries=4` 内退避重试。与现有 timeout、临时网络错误、429/5xx 共用计数，最多原请求加 4 次重试，重试不改变请求内容。
- 鉴权错误、配置/请求错误不盲目重试；已经有计量信息的协议错误仍交现有 session 记录用量，模型生成的错误工具调用保留原协议修复路径，不在 HTTP 层重新采样。
- 每次尝试保留独立本地 request id、上游 request id、状态码、错误类型、起止时间、retry_count、请求 fingerprint。记录响应长度/hash；非对象响应额外记录 JSON 类型，不写原始响应、密钥或私有 reasoning。
- 对已完整记录、没有被执行为 Agent 动作的基础设施失败，未知用量不阻断后续成功的已计量模型决策。成功但缺 usage、无 Trace、无法归属的请求及缺失 Agent-turn usage 继续拒绝。
- `resource_usage_complete` 不改成 true。报表新增未知用量请求数、token 下界标记；完整 cost 为 null，未知 tokens 为 null，不伪装为零。重试成功只补充自身用量，无法补回上一次失败的真实计费。
- 学习是否允许与费用是否完整分开：有效 metered 决策仍按原成功/验证条件学习；失败 HTTP 尝试不产生成功或部署 credit。

本补丁改变了旧的“任何未知费用都阻断整场实验”的规则：现在允许记录完整的 infrastructure uncertainty，但必须公开缺失费用。运行准确率可以继续统计，含未知请求的 token/费用不能声称是精确总量。

## 旧 run 的显式修复续跑

原始 run、checkpoint 和 Trace 的版本字段不覆盖。`--resume --provider-recovery` 创建单独不可变 `provider_recovery.json`，记录旧代码 hash、新执行 hash、原 manifest hash、原配置、精确修复声明和恢复任务。

`experiments/provider_recovery_release.json` 固定允许的原版本和新源码完整 inventory；任意额外代码变动、配置变动、checkpoint 损坏、未捕获 attempt、Trace 哈希变化都会拒绝。不是通用忽略 code mismatch 的开关。

后续 Trace 带 `execution_provenance`；冻结 provenance 包含同一 recovery receipt，Frozen134 继续验证精确新代码及原 train 身份。前 35 个任务不会改成由新代码运行。此次结果属于有明确基础设施补丁分界的续跑，非全程同一代码 hash 的 fresh run。

## 验证

- 全量 pytest：**1306 passed in 40.80s**。覆盖空/非对象/缺 usage 响应、混合失败共用重试上限、鉴权与 metered 协议错误边界、未知费用透明报告、缺决策用量拒绝、checkpoint/源码漂移拒绝、恢复凭据幂等。
- 真实 Runtime/Builder helper，注入第一次 HTTP 200 `null`：重试后生成工具，3 步工具试运行、R1、父流程均通过，92.59 秒，43,661 个已报告 tokens。`live_null_retry/summary.json` 的 `passed=true`。
- 上述 node harness 的 Trace 是精简诊断格式；额外用正式 Trace 审计器读取它时退出，原日志保留。没有据此改正式格式校验。单独执行了真实 provider fault smoke，注入 null 后取得正常响应，核验两份独立请求记录及未知用量下界报告，`live_api_audit/summary.json` 为 passed=true。
- 最终修复版本真实 provider capability probe：passed=true。
- 对停止实验的完整副本实际恢复 checkpoint，验证 ManifestStore 与知识摘要、全部 attempt/Trace、最终用量 gate：通过。保留 35 completed，剩余 85，第一个 `alfworld_train_36_pick_heat_then_place_in_recep`；未知请求数 1、resource_usage_complete=false，旧 manifest 字节未改。
- `git diff --check` 通过。没有启动正式 train/test。

执行源码 hash：`b0f1657ef7e7947ac8faaa02bfa9be3573abfa02f1df21a822e7228dd4c8749f`。

验收材料：本地 `runs/provider_recovery_validation/`；WSL 原件 `/home/yangchengyu/asg_provider_recovery_validation/`。旧实验完整数据与日志备份在该 WSL 目录的 `original_run_backup/`。

## 启动（WSL）

停止的原实验目录代码已更新后，执行：

```bash
(
set -euo pipefail
cd /home/yangchengyu/asg_r922_seed42_TqvxxJ
set -a
source /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
: "${MODEL_API_KEY:?请填写原仓库的.env}"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
"$ASG_PY" -m experiments.run_v3_train --config configs/alfworld_train_full_120_r92_seed42.yaml --resume --provider-recovery 2>&1 | tee -a train_resume.log
"$ASG_PY" -m experiments.run_v3_frozen_eval --config configs/alfworld_frozen_eval_134_r92_seed42.yaml 2>&1 | tee test.log
)
```

train 成功且冻结审计通过才执行 test。resume 仍按现有正式任务加载逻辑扫描游戏，不从该任务中间的一条动作继续，而是从任务前 bank 重新执行 `train_36`。保留前 5 次真实任务失败；它们不因 resume 而重试。
