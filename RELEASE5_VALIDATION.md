# Release5 验证与正式实验放行

结论：文档范围内的修复与开发验收通过，可以从新根启动五轮 Frozen Test134。正式实验尚未启动；旧 Release4 不再续跑。

实现提交：`8638e1d79ca42e3eccda5897831705901f76c16d`。
实际封存及验证代码指纹：`2551bf81b015f28c067902651317fbc8c889ecd802069644d3ac1d1d4e28ed34`。
本报告后续提交仅增加 Markdown/交付索引，不改变实验代码指纹。

## 实际执行结果

| 验证 | 结果 |
|---|---|
| 全部 pytest（最终整体验证脚本） | 1829 passed，127.60 秒 |
| 新原生合同定向测试 | 18 passed，包括真实 preflight/ToolRunner、一次 repair、HTTP/Session/usage 对账 |
| 新图与停止机制测试 | 4 passed，含真实任务循环、SQLite 和 AttemptTraceLedger capture |
| 历史原始拒绝 | 35/35 保留原拒绝分类：30 schema、4 多调用、1 无动作 |
| 三库逐 ref 对比 | 通过；42 仅增加 1.0.1，43/44 正文、文件 hash、状态不变 |
| 原三库无模型公开发现/数据流专项 | 3/3，通过，bank digest 不变 |
| seed42 新冷却图真实环境专项 | 未开/已开两例均通过；0 模型调用 |
| 固定六族 × 三 seed current 自然开发集 | 18/18 成功 |
| 最终 HTTP/Session 原生 schema 对账 | 82/82 请求一致；无遗漏用量/Support调用关联 |
| diff、shell 语法检查 | 通过 |
| 五份正式配置生成及 released_stream 无 seed 放行预检 | 通过；各正式 output 均尚不存在 |

全量回归保留原有效反例；仅更新一条 Draft 提示断言，因为 Draft 不应叠加 Node 搜索/完成提示。没有删除原有效测试换取通过。
NC09 的重复 JSON key、非法数字和重复 call ID 等解析反例沿用已有全量协议测试；新增测试补齐共享诊断、调用分类与执行接线。

## 自然开发结果与成本

固定任务为 train_0、34、4、36、6、2；每 seed 各一次，没有更换任务或择优重跑。

| Seed | 成功 | 物理模型请求 | Prompt + completion tokens |
|---|---:|---:|---:|
| 42 | 6/6 | 26 | 217,802 |
| 43 | 6/6 | 27 | 226,682 |
| 44 | 6/6 | 29 | 255,995 |
| 总计 | 18/18 | 82 | 700,479 |

Prompt 554,280，completion 146,199；其中 reasoning 135,707，已包含在 completion 内，不再相加。
每题均值 38,915.5，中位数 37,640；最大为 seed44 heat：86,594。
自然 episode 从首题开始至末题结束墙钟约 **10分21秒**；各题运行时长相加约28分07秒（三路并行，不能当墙钟）。这不包括先前封存、pytest、共享能力探针和无模型专项。

18题均以 `stored_composite` 路线开始。原生生成调用：Implementation 38、Support 31、environment_action 12、atomic_validation 1。
自然开发集协议拒绝0、repair请求0、Support preflight拒绝0、未知用量请求0、未关联Support调用0。
因此自然18题不能证明模型一定会遵守修复反馈；一次repair接线由固定错误 provider 的生产 Session→preflight→ToolRunner 测试及最终 HTTP 对账证明。

共享 provider capability probe 为单独4请求、用量完整，保存在 `dev/provider_probe/`；未混入上述82请求/700,479 tokens。
原 Release4 在途未知费用单列于旧留存报告，未归零或拼接到新版结果。

## 冷却图的真实证据

新图采用原五个 Atomic，且原正文与状态全部保留。真实受控 train_6 中：

- 处理站点与最终目的地不同；由测试驱动根据公开 catalog 显式提交，Runtime 无硬编码选择规则。
- 未开站点：OPEN 为 `direct_autonomous_success`；已开站点：`already_satisfied`，不执行冗余 OPEN。
- 两例 COOL 均为 `direct_autonomous_success`，实际消费 NAV.location→COOL.station。
- DELIVER 实际消费 COOL.object；仍由显式调用提供尚未准备好的目的地，未将它冒充自动选定。
- NAV 下游提示同时包含 OPEN/container 与 COOL/station，真实 trace 和数据流 hash 已保存。

## 材料位置

WSL 最终根：`/home/yangchengyu/asg_oldfirst_20260923_release5_final/`

Windows 资源管理器：
`\\wsl.localhost\Ubuntu\home\yangchengyu\asg_oldfirst_20260923_release5_final`

- 实施说明：仓库 `RELEASE5_IMPLEMENTATION.md`。
- `native_contract_audit.json`、`native_protocol_regression.json`、`seed42_graph_revision_audit.json`。
- `seed42/`、`seed43/`、`seed44/` 内各 `reseal_comparison.json`、`frozen/release_manifest.json`。
- `dev/coverage_report.json`、`dev/coverage_report.md`、`dev/coverage_acceptance.json`、`dev/controlled_acceptance.json`。
- 原始完整自然 episode：`dev/seed{42,43,44}/current/<task_id>/`，保留 traces、provider_payload_audit、all_usage、attempt_history、run_state 和 manifests。
- 原三库专项：`dev/seed*/automation/`；新冷却专项：`dev/seed42/graph_revision/`。
- 最终完整测试日志：`dev_validation.log`；另一次同版本全量回归：`pytest.log`（1829 passed，123.01秒）。
- 正式矩阵：`evaluation_plan.json`、`configs/seed42_rep01.json`～`rep03.json`、seed43/44各rep01。
- 旧停止留存：`/home/yangchengyu/asg_release4_stop_20260923/`。

中间根 `asg_oldfirst_20260923_release5` 在真实开发调用前停止，仅保留早一版审计代码的封存及未完成pytest日志；不得拿它启动或resume。最终根重建后完整跑了一次18题。

## 正式启动（WSL）

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
bash scripts/run_released_frozen_parallel.sh /home/yangchengyu/asg_oldfirst_20260923_release5_final
```

脚本自动读取仓库 `.env`，校验放行与版本，打印三个监督 PID。seed42 三轮顺序执行，seed43/44各一轮与其并行。当前本地代码已是验证版，不必 git pull。

监控：

```bash
tail -F /home/yangchengyu/asg_oldfirst_20260923_release5_final/seed{42,43,44}.log
```

题间暂停：

```bash
bash scripts/request_released_stop.sh /home/yangchengyu/asg_oldfirst_20260923_release5_final all
```

协作暂停后需明确取消对应 control/stop 请求，再执行同版本 `run_released_frozen_parallel.sh ROOT all --resume`。不要对旧 Release4 使用新代码续跑，不删除 attempt 或已完成结果。

以上是固定训练侧开发放行证据，不是Test134的成功率承诺，也不证明所有长推理开销均由本次修复消除。
