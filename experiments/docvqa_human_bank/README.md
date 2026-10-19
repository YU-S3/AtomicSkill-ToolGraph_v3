# DocVQA 人工 Bank 独立对照 v1.1

来源：用户提供的 `DocVQA_HumanIdeal_ProgrammaticAtomic_Bank_v1_20261019.zip`。
原包与资产哈希见 `source_manifest.json`，公开来源见 `sources/SOURCES.md`。
本目录只接入独立人工对照，不更改 Ours 的自然学习、晋升、评分或已有实验。

审查后修正：简短逐字回答始终可见；不强制复核、扩写缩写或附加单位；
裁剪工具不按题干关键词隐藏；算术参数明确先后顺序。所有改动只依据接口和通用答题约束，
不使用具体 Test 答案。Skill、Program、关系均通过生产 Bank.put 重新计算 ID。
Program 保持 candidate，只读人工视图显式授权，不伪造 usable 收据。

使用生产 AnswerExecutor、ProgramWorker、Provider、checkpoint、正式日志和预算账本。
10 Skill、3 Program、3 Implementation、3 Workflow 全部导入；
当前单答案执行器只运行可选 Program，3 Workflow 仅保存、不计作整图执行。
裁剪图仅在指定 Program 真实执行成功、源图 SHA 和输出 SHA 核验后发送到下一轮。
仍发送原图，base64 不充当文本证据；文字核对程序不能证明读图正确。

固定 seed42，先用 canonical Val 前2题检查真实接口，再运行 canonical Test201。
不学习、不按分数选 Bank，不重切数据；Test 开始后不更改提示和资产。
96,000 tokens/题，总计19,488,000 tokens /9,135 HTTP（含验证2题）；实际费用以 raw_usage 为准。
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
