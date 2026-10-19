# DocVQA 人工 Bank 策略迭代

当前 v1.8 是待验证的策略候选，尚不能宣称稳定提分。将多个 Skill 的答案范围决定权
集中到“问题目标→证据单元→所求分量”的流程；专用 Skill 只负责读取过程，收尾 Skill
执行已经确定的范围，移除相互竞争的最短答案指令和虚构示例。
v1.7 实跑源码为 `ffc5323a11e18f26d47706684452b63cca091722`，其固定复测继续使用原版本。
两版模式均为 `guidance`。本目录是独立人工 Bank 对照，不改变 Ours 的自然学习、晋升或评分。
来源为用户提供的 `DocVQA_HumanIdeal_ProgrammaticAtomic_Bank_v1_20261019.zip`；
原包哈希见 `source_manifest.json`，公开资料见 `sources/SOURCES.md`。

v1.7 保留 v1.1 的简短逐字回答指导，将视觉辨认提示合入常驻 Skill，
补充虚构的字段边界示例、字面时间抄录及表格极值比较。
裁剪最多执行两次，保留原图和经哈希验证的真实裁剪图；小区域最多放大3倍，优先 PNG。
修正裁剪二进制脱敏误改输出 Schema 的问题，并声明不同动作的必填字段。
`grounded` 模式的第一次裁剪仅接受粗区域；该消融没有证明提分，当前确认使用 `guidance`，
保留可选 bbox 裁剪。插值放大不产生新证据，文字核对 Program 不能证明读图正确。

10 Skill、3 Program、3 Implementation、3 Workflow 均通过生产 Bank.put 导入并重算 ID。
Program 保持 candidate，独立只读人工视图显式授权调用；不伪造 usable 或输出消费收据。
单答案执行器使用可选 Program；Workflow 仅保存，不能宣称执行了整张图。
使用生产 AnswerExecutor、ProgramWorker、Provider、checkpoint、日志与预算账本。

开发仅使用 seed42 固定的 Train36；所有改写和失败结果保留。
冻结后用不重合的 Train144 确认，再按预先记录的门槛执行成对 Val22 和 Test201 复测。
v1.7 在 Train36 为34/36，未达到原定35题门槛；增加独立确认不改写这项失败。
重复响应不是新独立题目，Test 复测也不属于未接触的留出评估。
自然学习 Ours 的原结果、旧人工 Bank 和费用均保留，不把人工结果混入自然学习结果。

历史：v1.1 的原 Test 为180/201（89.55%完全正确率），ANLS 为94.33%。
v1.2 固定 Val 为20/22，对照 v1.1 为21/22，未进入 Test。
本轮 v1.6 两次 Train36 配对分别为32对34、32对31，也未证明稳定收益。
95%完全正确率对应 Test 至少191/201，不能用 ANLS 代替。

本轮使用 `evaluate.py`，显式传入源码、题单、只读 Bank 与累计账本。
不要把旧 `run.py` 的默认 `grounded` 模式当成受测的 v1.7 guidance 配置。
在固定的 clean WSL checkout 中，按已冻结计划运行，例如：

```bash
python experiments/docvqa_human_bank/evaluate.py \
  --repo "$PINNED_REPO" --sha "$EXPECTED_SHA" \
  --config "$SOURCE_CONFIG" --selection "$FROZEN_SELECTION" \
  --output "$NEW_OUTPUT" --campaign "$CAMPAIGN" \
  --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env \
  --crop-mode guidance
# 候选不传 --bank，按固定源码重新导入同一人工资产。
# --bank 仅用于复制旧版对照；它使用旧视图构造器。
# --check 为零 HTTP 预检；同版本、同目录恢复用 --resume。
```

每题96,000 tokens，人工对照累计上限19,488,000 tokens /9,135 HTTP，
本轮账本结转旧人工对照实际费用819,414 tokens /270 HTTP。
排他锁阻止同一账本并发写入，增加复测不扩大额度。
run_seed42 控制题单；Provider 不支持 generation seed，重复调用并非确定性复现。
模型为可变别名 `deepseek-flash`，响应 model、fingerprint、时间和 raw_usage 保存在原始请求中。

各 run 的 `plan.json` 固定身份，`result.json` 提供轻量汇总，`episodes.jsonl` 可监控进度。
对应 split 下的 `summary.json`、`traces/`、`requests/` 保存原始结果；
`final_frozen_manifest.json` 及 result 的前后 hash 核验只读 Bank。
