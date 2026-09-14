# R9.2.2 修复与验收记录

验收日期：2026-09-15（Asia/Shanghai）。基线：`c36a5f1c3025f272c015e374d07881a975c03917`，仅修改 main。

## 结论

R9.2.2 六项定向修复及规定验收已完成，可以启动 fresh seed42 Full-120，再由现有正式冻结审计放行 Frozen-134。本次没有启动正式 train/test；定向验收通过不代表正式实验成功率保证。

## 实现对应

| 范围 | 实际修改与边界 |
| --- | --- |
| A：实体角色引用 | ALFWorld `_expected_args` 不再按谓词角色名隐式查绑定。普通值保持 literal；`$role` 和既有 `BindingExpression` 显式解析路径保留。 |
| B：语义输入与具体输出 | Runtime draft/native schema、Builder 输入、R0、AtomicValidator 接通 `output_semantic_constraints`。R0 校验声明角色、类型、resolution、derivation；R1 使用 Harness 兼容性。具体候选只过滤当前权威事实，不覆盖输入、不生成事实，同名语义到具体的升级仍拒绝。 |
| C：可见验证来源 | 公共谓词 schema 展示既有 `validation_source`；Runtime/Builder 说明 literal/reference、语义类别/实体、最小 final Effects 与事实来源。公共 projection 不成为验证真值。 |
| D：动态 catalog 遍历 | 仅 action_catalog FOR_EACH 支持可选 boolean `refresh_each_iteration`；默认 snapshot 不变。每次重新查询当前 catalog、按投影值去重、保留外层 lexical scope 与迭代/动作预算。必需动作不可用仍拒绝。 |
| E：增量回放证据 | 使用现有 EvidenceLedger 追加 `REPLAY_VALIDATED/REPLAY_REJECTED`。证书匹配 executable signature、case id、完整 case hash、authority version；匹配前仍核验源 Trace。缓存复用不算物理回放，不产生部署 credit。先保存不可变 Trace，再提交其证书事件。 |
| F：可执行身份与 Direct | 已准入的相同 executable 新增 evidence 不再递增 ToolRef，也不产生等价 ImplementationRef 或自指派生关系。维护通过显式 evidence view 获取完整历史；不可变 Tool payload 保留。原 discovery Trace 再处理保持 exactly-once discovery credit。 |

报表新增 observation/fresh execution/certificate reuse/failure 计数，并报告唯一 executable 数、证据数和每个 executable 的 Implementation 数；bank 数量取最后一次快照，不跨任务求和。maintenance 纳入回放统计。

模型、reasoning、100K/node 与 300K/task token cap、动作预算、benchmark won、R0/static/R1 验证 authority、正式任务清单及 Direct 选择策略均未修改。未加入生产代码的 benchmark 对象/任务特判。

## 自动化验证

最终在原仓库与既有 WSL 环境执行：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
PYTHONPATH=src:. /home/yangchengyu/asg_alfworld_venv/bin/python -m pytest -q --tb=short
```

结果：**1291 passed in 39.53s**，无失败、无跳过。新增覆盖：literal/显式引用、多实例输出正负例/过期事实、约束类型与同名 resolution 边界、动态候选出现/消失及 snapshot 回归、非法 live flag、10-case 线性回放、同 case 幂等、改变程序/authority 不继承证书、Frozen digest 不变且不写证书、真实执行路径 Direct、重复 discovery credit、多份 native 提案的验收统计。`git diff --check` 通过。

## 实际定向验收

原始材料统一备份到本仓库忽略目录 `runs/r922_validation/`。WSL 原件位于 `/home/yangchengyu/asg_r922_acceptance/`。每个目录保留请求、响应、Trace、catalog 快照及用量；不是正式实验结果。

| 最终验收 | 结果 | 执行用时 | 外部 API tokens |
| --- | --- | ---: | ---: |
| `live_controlled_v2`：真实 Runtime/Builder + 动态多实例受控 Harness | 通过；3 步工具动作，R1 通过，父流程完成；2 个 apple 实例 | 78.27 秒 | 38,278 |
| `live_alfworld_v2`：真实 Runtime/Builder + ALFWorld train env_index 23 | 通过；2 个 CD 实例，具体输出 `cd_2` / `dresser_1`，22 步试运行，R1 通过，父流程后续 TAKE 成功 | 195.21 秒 | 65,082 |
| `replay_v3`：旧库同 executable 的 10 条独立来源 case，复制到隔离空库 | 通过；10 次 fresh replay / reset / initialize，第 5 条重复提交新增 0 次；1 Tool、1 Implementation、10 证书；随后真实 Direct autonomous，候选数 1 | 221.18 秒 | 0 |
| `provider_final`：正式配置 DeepSeek 能力探测 | 通过；5 次 HTTP 200，结构提交、reasoning replay、Extractor token limit 均通过 | 见原始探测记录 | 2,754 |

表内用时为 runner 记录的实际执行阶段，不包含启动前代码指纹扫描。定向入口只强制选择被验收的 native 路线，草稿、程序和输出值由真实模型生成；不向生产代码加入固定 fixture 行为。

### 原始失败与汇总修正

- `live_alfworld_v1` 真实失败已保留：模型把位置输出错误约束到输入 literal `location`，R1 正确拒绝。补充公共说明，强调约束比较输入实际值而非参数类型，关系派生输出无需伪造类别输入。没有放宽 validator。该次 196.74 秒、58,921 tokens。
- `live_alfworld_v2` 首份 native 提案包含不存在的谓词，被 R0 拒绝；模型在原会话预算内修正第二份提案并完成完整链。旧汇总函数假设只有一份 scripted draft，因此原 `summary.json` 的 `case_passed=false` 是汇总误判。新 R9.2.2 汇总逐份保留失败和成功记录，成本按整个会话计一次，未改旧 R9.2.1 汇总器。
- 上述原 `summary.json` **未覆盖**。由原 `trace.json`、`internal_000_usage.json` 重算出的 `live_alfworld_v2/reconciled_summary.json` 为验收结论：`passed=true`、draft_count=2、r0_rejection_count=1、actual_trial_count=1、r1_pass_count=1。没有为了修正统计额外重复付费调用。
- `replay_v1` 的 fixture 初始错误也保留：把一个 case 与不对应的 current_task 一起提交，来源 authority 正确拒绝。修正隔离 runner 的来源传递后，`replay_v2`/`replay_v3` 通过。
- 真实运行之后增加的原 discovery 再处理幂等修正和多草稿汇总修正，已由定向测试及最终全量 pytest 覆盖；最终 provider probe 使用这些修正后的源码。

这些结果证明规定路线可执行与证据边界正确，不消除真实模型产生无效提案的可能性；无效提案仍按原规则拒绝。

## 来源与指纹

旧诊断前缀未续跑、未覆盖：

`/home/yangchengyu/asg_r921_seed42_8atBao/runs/alfworld_train_full_120_r92_seed42`

其中 40 份真正 Trace = 34 份正式任务 Trace + 6 份 maintenance；正式任务 31 次 strict success。claim/owner sidecar 不计为 Trace。逐 Trace SHA256 及来源信息保存在 `runs/r922_validation/diagnostic_prefix_identity.json`。

- 旧 `data_v3/state.sqlite3` SHA256：`d23326d20f2f5516913a6c480181cbd0de05ffb700cf8cfb846f313b19627753`。
- 旧 task manifest SHA256：`c309873f9d3a4cd78485479fe2ed32dd485f3436e7d0151a4b27b4d79f4ed61b`。
- 最终 probe 的源码副本 code hash：`d904298c7b217e68209b3353981b81b112d19812a532eabeeb13aa5af9d0b0f7`。
- 正式 train 配置 hash：`308610c19ef1c18b6bbab6583aa5e85f373e517d8c68e535b1e2b25d7bfa503f`。
- Builder instruction 长度 12613，SHA256：`2b9d008b0563c4eea0c3bef0d70aed8d608fafe82fbace2d0b68d0c28529871b`。

probe 的 code hash 对应验收用 WSL 源码副本；新正式 checkout 会按自身提交文件重新计算并探测，不复制旧 probe 或旧 bank。

## fresh seed42 启动

在 WSL 运行下列命令。沿用现有环境和原仓库 `.env`，从已提交的本地 main 创建全新 WSL 工作目录，避免 NTFS checkpoint/mv 权限问题。Full-120 完成且现有冻结审计通过才进入 Frozen-134；异常会停止。旧诊断前缀不使用 `--resume`。

```bash
(
set -euo pipefail
RUN=$(mktemp -d /home/yangchengyu/asg_r922_seed42_XXXXXX)
git clone --quiet --no-hardlinks --single-branch --branch main /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3 "$RUN"
cd "$RUN"
set -a
source /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
: "${MODEL_API_KEY:?请填写原仓库的.env}"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
echo "实验目录：$RUN"
"$ASG_PY" -m experiments.run_v3_train --config configs/alfworld_train_full_120_r92_seed42.yaml 2>&1 | tee train.log
"$ASG_PY" -m experiments.run_v3_frozen_eval --config configs/alfworld_frozen_eval_134_r92_seed42.yaml 2>&1 | tee test.log
)
```

配置文件沿用 `r92` 名称，版本以新提交和 run manifest 的 code fingerprint 为准。日志在新目录的 `train.log`/`test.log`，Trace 与报告分别位于对应 `runs/alfworld_*_r92_seed42/traces/` 和 `reports/`。
