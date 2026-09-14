# R9.2 审查修复与验证记录

审查基线：`1fd8ab1b9b071005d14b01d2e555345187d16a09`。
代码修复：`6599c9c5cdb01faf52c40bc4270b968af22e5cf4`。
对照文件：`AtomicSkillGraph_R9.2_1fd8ab1_代码审查与实验放行意见.md`。

## A：Tool IR 循环作用域

确认原解释器允许内层 FOR_EACH 覆盖外层同名 local，且离开内层时不恢复。
每个循环现在保存原绑定并用 finally 恢复；原来不存在的变量在离开时移除。
正常退出、STOP_WHEN、RETURN_PROGRAM、FAIL_TOOL、BENCHMARK_TERMINAL 和异常
共用此边界。循环次数、动作执行、控制步预算、terminal 信号和证据路径保持原语义。

新增测试检查同名/不同名嵌套循环、Tool input 与 local 命名空间、全部退出路径，
以及动作预算、控制步预算、执行历史和循环计数不被清理逻辑重置。
旧 `test_gate7_stop_when_breaks_loop_then_return_runs` 曾在循环外读取局部变量，
依赖了该缺陷；其 RETURN 改为读取真实执行产生的 `agent.at_location` 证据，
继续验证 STOP_WHEN 后运行 RETURN、仅执行一次动作及返回正确地点。

## B：未发布 producer 输出的只读关系评估

`RuntimePlanContextBuilder.build` 增加显式 `producer_output_candidates` 输入。
正式发布值优先；没有正式值时，仅评估当前 revision、具体值且在当前公开事实或
动作目录中可见的候选。来源记录为 `effect_resolution`、`agent_proposal` 或
`input_identity`，并携带 revision、resolution、candidate 状态和
`is_binding_authority=false`。

NodeExecutor 从契约明确允许的 input_identity 和当前 grounded 输入构造候选，
并在当前效果解析后、Repeat 检查及输出发布前接入候选评估。实际 witness 的证据
级别与契约声明的最低 resolution 分开处理；公开可见性仍独立检查。
效果候选评估记录在 Trace 的 `metadata.producer_output_candidate_assessments`。
没有候选、仅语义类、旧 revision 或只有私有事实时保持 unknown。

此通道不发布输出、不提交输入或 Repeat 绑定、不创建语义 anchor 或新证据，
不参与成功判定。下游仍仅来自正式 DataFlow 边。策略上下文的无损压缩保留新字段。
新增测试覆盖提交前 supported/contradicted、上述拒绝边界、Agent 候选不改写
identity、真实效果解析接线、正式输出优先及多个 consumer 边保留。

## Gate 1：全仓回归

在现有 WSL ALFWorld 虚拟环境运行：

```bash
PYTHONPATH=src:. /home/yangchengyu/asg_alfworld_venv/bin/python -m pytest -q
```

最终结果：**1188 passed in 29.54s**，包含本轮新增的 31 项用例。
`git diff --check` 通过。
没有更改模型、reasoning、预算、R0/static/R1、Replay/Repeat 或 Frozen 准入规则。

## Gate 2：真实模型小样本

使用 `configs/alfworld_train_full_120_r92_seed42.yaml` 的现有真实 7-task smoke，
先做 preflight 和 provider probe，再使用 smoke 自建的独立 empty bank。
日志：`runs/r92_review_scope_candidates_smoke_20260914.log`。
preflight 通过：120 个任务、六类各 20 个，清单哈希
`5d3b3d1ef4ce891c2bfaf531f942a9f943caff3c3ac64b8843cee5aae27a1640`。
Provider probe 通过，4 次 HTTP 请求均为 200，用量完整。
本次代码指纹：`e7760b80e872d1747eb46cfb3a31c2ccbe90888ae03df64406751248d6dbb300`。

真实 smoke 运行目录：
`runs/alfworld_train_full_120_r92_seed42/run_20260914T042720.318075Z_587816`。
下文中的报告和 Trace 路径均相对此目录。
7-task 清单哈希为
`02f1090520275b2ec1414ecbafb3d23426c945bd3787f52a77629e0a63cd0f85`，
与前一轮固定 smoke 相同，没有替换测试题。
2026-09-14 北京时间 12:27:20 创建运行目录，12:53:03 报告写出，约 25 分 43 秒；
从第一题开始至最后一题结束为 1474.495 秒（24 分 34.5 秒）。不含前置 preflight/probe。

结果：**6/7 成功（85.71%），smoke passed=false，Gate 2 未通过。**
前五道 pick-and-place 和第一道 heat 均成功，最后的
`alfworld_train_25_pick_heat_then_place_in_recep` 失败。
四层资产存在（Atomic 4、Composite 4、Implementation 12、Tool 12），
`cold_dynamic_success=true`、`actual_started_direct=true`。
`validated_dataflow=false`；最后任务在首个 heat 节点失败，未完成下游消费。
不能将本轮失败解释为仅有此前允许忽略的 dataflow smoke 设置差异。

全部 7 题 `infrastructure_failure=false`、`resource_usage_complete=true`。
未知动作、benchmark/contract 不一致、token mismatch、未归属 token 均为 0。
报告目录还有独立 maintenance Trace，它不计入七题成功率。

### 自工具化与成本漏斗

| 指标 | 本轮 |
|---|---:|
| offered / interface projected | 43 / 43 |
| proposal | 4 |
| R0 pass / reject | 2 / 2 |
| Builder called / no_tool | 2 / 2 |
| static pass / reject | 0 / 0 |
| trial started / completed / terminal | 0 / 0 / 0 |
| R1 pass / reject | 0 / 0 |
| parent resumed / completed after trial | 0 / 0 |
| trial 内部动作 / LLM bypass 动作 | 0 / 0 |
| Runtime Builder tokens | 18,568 |
| Parent Runtime tokens | 460,886 |
| 全部七题 tokens（含各 Agent bucket） | 1,051,524 |

未配置可用的美元计价，报告 cost USD 为 n/a，不能虚构美元费用。
本轮没有实际 trial，因此既未证明完整自主自工具化链路，也不能声称已经带来成本改善。
注册 Tool 的 Direct 执行以及 E2 的工具生成不作为 Runtime 自主 trial 的替代证据。

### 最后一题失败证据

Trace：`traces/trace_45a226c97a0443f9bf866e58b7f05cd8.json`。
失败码为 `runtime_node_token_budget_exhausted`，任务总 token 为 223,850；
Preparation 与 Seeded 均记录了 session token 超限，最终没有完成第一个 heat Atomic。

实际执行 14 个动作，均被环境接受：搜索冰箱、台面、微波炉和橱柜，随后在餐桌
找到并拿起 `egg_1`，最后前往 `stoveburner_1`。没有执行 HEAT 动作。
该任务的一次 draft 因同一输出存在多个 Effect witness authority 被 R0 拒绝；
修改后的搜索 draft 通过 R0，但 Builder 认为候选地点枚举和发现效果证据不足，
返回 `no_tool`。前面的 train_11 也有一次搜索 draft 得到相同类型的 Builder 拒绝。

这些记录表明自主提案仍未转化为实际可执行试用；这是尚未解决的真实模型表现/验收
问题，不能因为 A/B 回归通过而宣称放行。没有发生 Runtime trial，因而本次耗尽不能
归因于 trial 内嵌套循环变量覆盖；也没有证据支持通过放宽 R0、增大预算或强迫提案
解决本轮失败。A/B 修复不扩展为未经文档要求的新策略或权限变更。

原始汇总：`reports/real_alfworld_smoke.md`，同时包含 CSV 与 JSONL。
全部任务、Planner、Runtime、Extractor、Builder 与维护证据保留在本次运行目录。

## 正式实验边界

本轮不启动正式 Full-120 / Frozen-134，也不覆盖旧实验产物。
按照审查文档第 6 节，Gate 1 已通过，Gate 2 尚未通过，**不放行 Gate 3 正式实验**。
只有真实模型的自主 draft → Builder → 多动作试用 → R1 或合法 terminal
收束及父节点独立完成得到证据后，才能依审查意见判断是否放行新单 seed。
