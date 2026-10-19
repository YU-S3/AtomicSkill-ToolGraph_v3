# DocVQA 人工 Bank 独立对照 v1.3 候选

v1.3 尚在评估，未宣称提分。合并常驻指导、让视觉阅读指导始终可见，
加入可选粗区域裁剪及动作必填字段。`evaluate.py` 使用明确的源码、题单与
不可变 Bank；所有新请求共用一份有排他锁的累计账本，旧费用保留。
guidance/grounded 两种模式仅用于区分粗区域接口的作用，前者隐藏 region；
两者共用新指导与 v1.2 的无损裁剪。v1.1 对照使用原源码及原 Bank。
开发比较使用固定 Train 子集，Bank 始终只读；不是自然学习 Ours 的新成绩。
最终选择需冻结版本并重复验证，Test 结果标记为探索性复测。

以下保留 v1.2 的失败历史：

**状态：候选版本，未通过提升验收。** 固定 Val22 对照中，v1.1 为21/22、75,796 tokens；
v1.2 为20/22、84,187 tokens。没有启动新版 Test201，不能宣称达到95%完全正确率。
原 v1.1 的180/201 Test结果和独立固定checkout保留。受测提交为 `8905bc6`；
后续只给启动入口加上了默认停止于Val的保护，没有追加模型调用。

来源：用户提供的 `DocVQA_HumanIdeal_ProgrammaticAtomic_Bank_v1_20261019.zip`。
原包与资产哈希见 `source_manifest.json`，公开来源见 `sources/SOURCES.md`。
本目录只接入独立人工对照，不更改 Ours 的自然学习、晋升、评分或已有实验。

审查后修正：简短逐字回答始终可见；不强制复核、扩写缩写或附加单位；
裁剪工具不按题干关键词隐藏；算术参数明确先后顺序。v1.2 修正 thumbnail 不放大的问题，
将小区域按比例最多放大3倍，优先无损 PNG；两次裁剪同时可见，保留表头与目标行。
修正 handwrit 词干无法命中 handwritten 的路由，并覆盖最大/最小比较问题。
明确角色、分类、完整名称的不同答案范围以及单响应仅一个动作，不增加 repair 次数。
所有改动只依据接口和通用答题约束，
不使用具体 Test 答案。Skill、Program、关系均通过生产 Bank.put 重新计算 ID。
Program 保持 candidate，只读人工视图显式授权，不伪造 usable 收据。

使用生产 AnswerExecutor、ProgramWorker、Provider、checkpoint、正式日志和预算账本。
10 Skill、3 Program、3 Implementation、3 Workflow 全部导入；
当前单答案执行器只运行可选 Program，3 Workflow 仅保存、不计作整图执行。
裁剪图仅在指定 Program 真实执行成功、源图 SHA 和输出 SHA 核验后发送到下一轮。
仍发送原图，base64 不充当文本证据；文字核对程序不能证明读图正确。

固定 seed42，默认使用完整 canonical Val22并停止；`--validation-count 2` 仅作接口检查。
只有本版本 `validation_gate.json` 明确通过且显式给出 `--allow-test`，才允许继续 Test201。
不学习、不按分数选 Bank，不重切数据；Test 开始后不更改提示和资产。
96,000 tokens/题，总上限仍为19,488,000 tokens /9,135 HTTP；增加验证数量不自动扩大总额度。
实际费用以 raw_usage 为准。
这是独立人工对照的新预算账本，旧 Ours 费用与成绩保留在原目录，不改写、不重复计入。

在独立 clean WSL checkout 中：

```bash
python experiments/docvqa_human_bank/run.py check --expected-sha "$EXPECTED_SHA" --output "$OUTPUT_ROOT"
python experiments/docvqa_human_bank/run.py run --expected-sha "$EXPECTED_SHA" --output "$OUTPUT_ROOT" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
# 同版本、同目录恢复：把 run 改为 resume。文件锁阻止重复启动。
```

`test/summary.json`、`test/traces/`、`test/requests/` 保存逐题原始结果，
`test_result.json` 保存 hard 与 ANLS，根目录保存预算、身份与前后 Bank hash。
请求模型为可变别名 deepseek-flash，实际响应 model 和时间保存在物理请求里。
