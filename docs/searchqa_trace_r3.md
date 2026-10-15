# SearchQA R3

方法身份为 `searchqa.train-trace-capability.r3`。原 R2、旧 Unified_R1 Train300、人工 Bank Test 各自保留原源码和费用。R3 不重解 Train300，不导入人工 Bank，也没有 Test 入口。

## 已实现的固定合同

- 归纳使用完整 candidate Schema；来源、引用、父版本、正文 600 字符和卡片 900 字符的检查进入一次 repair。失败与正确的历史回答分别标为需求和 advisory，均不称为已执行证据。
- Builder 使用真实生效的 `searchqa_builder` purpose：disabled thinking、4096 completion、零 Provider 重试；每个 Job 最多初次生成加一次修复。已付费的成功或拒绝响应均不在恢复时重新请求。
- 六种类型分别检查片段、显式别名、同主体/单位/时间的数值比较、窄字面关系、显式标识符跨文档连接、来源支持。默认全为 intermediate。关系合同仅验证列出的字面语法，不宣称覆盖全部自然语言关系或保证召回。
- Program 需要所有声明来源（至少两题）、独立 Train 迁移以及空输入、Unicode、参数变化、歧义、单位和偏移边界的真实 Docker 收据。单个来源成功只记 partial；Candidate 不能直接设 usable。
- 默认 model_first、每题最多一张适用短卡，无必要公开条件则零注入。未启用程序先行、答案审阅和 final_answer Program 变体。
- Worker 原始收据保留在 Host；R3 模型可见结果及 read_result 都不能读取该审计元数据。调用、展示、读取、声明消费和配对效果分别记录。
- 只有单个合法 finish 的 `unit/end` 注释可规范化。R2 真实故障是带这些字段的 **read_result**，回放仍拒绝。真实 length 空答最多一次 disabled/2048 收尾；非 length 不补答，父预算仍为 96K。

## 固定 Train 来源

`verify` 先进行独立 Host 审计，再在权限隔离中只读取 Train 来源。历史审计包含 8,700 个文件、R2 的 19 道实际曝光来源、4 个归纳拒绝、6 次 Builder 截断和 7 条结果收据泄漏请求。

从未暴露于 R2 学习的 Train ID 中按 `SHA256(R3_version + physical_key)` 选择新 utility30，再预留独立 Worker30，其余 240 为归纳池。源提示只包含归纳池的历史 advisory，不包含新 utility/Worker 探针的历史回答或评分。Host 审计的 Val/旧 probe 诊断文件在学习沙盒中不可读。

D1 保留三份失败 R2 ProgramSpec、Atomic、来源和失败响应的完整父材料；新建 R3 候选并明确记录 `legacy_parent`，使用新 Builder/Worker。其定位操作映射为 R3 evidence_locate 的中间输出合同；这不计作新的自然归纳成功。

## 阶段命令

先在 WSL 中使用固定、干净的原生 Git checkout；Windows worktree 的 Git 链接不能作为 WSL 源码身份。

```bash
python -m atomic_skillgraph.experiments.run_searchqa_trace_r3 --help
python -m atomic_skillgraph.experiments.run_searchqa_trace_r3 \
  --stage verify --output /path/to/new_r3_run \
  --source-run /path/to/original_unified_r1/searchqa/seed42 \
  --r2-run /path/to/finished_r2 --datasets /path/to/materialized/searchqa \
  --historical-manifest /path/to/readonly_audit/source_manifest.json \
  --deny-path /path/to/manual_bank --deny-path /path/to/manual_test
python -m atomic_skillgraph.experiments.run_searchqa_trace_r3 \
  --stage prepare --output /path/to/new_r3_run \
  --external-reservation /path/to/concurrent_allocations.json
```

外部分配文件必须记录 `tokens/http/runs`；并行人工 Test 的整段额度从可用累计额度中预留，不伪装成实际已花费用。历史费用只结转一次。

付费阶段为 `reuse_spec`（D1）、`induce/build`（D2）、`measure_train --probe-size 12`（D3a）、`measure_train --probe-size 30`（D3）、`freeze`、`val`（D4）。`worker` 汇总每个 build Job 已完成的真实资格检查，`report` 输出审查指标。各阶段独立启动；完成阶段使用 `--resume` 校验，不重跑。utility 开始后禁止继续学习。

**prepare 默认没有付费授权。** 获得用户批准后，授权 JSON 需包含 `approved: true`、授权文字、实际 `method_lock.json` 的 SHA256，以及相应 `phases` 的精确 token/HTTP 上限，再通过 `--authorization` 提供。文档预算仅为提案：D1 250K/12、D2 750K/60、D3a 2.4M/50（包含在 D3 总计 5.8M/110 内）、D4 7M/150，合计 13.8M/332；阶段间不能借用。

Train12 的原始结果和费用在 Train30 中保留，后续只跑剩余 18 题。Val 使用最终冻结库做相同公开输入和参数的三臂对照。Val24 已多次用于开发的事实写入 method lock；未提供 Test 命令。

## 零 HTTP 验收边界

`tests/test_searchqa_trace_r3.py` 包含真实旧材料回放、生产请求拦截、真实 Docker、Bank 资格/冻结、原始 canonical Train30/Val24 的完整生产运行、日志和恢复检查。拦截响应仅证明接口与工程路径，不是模型自然学习成功，也不证明准确率或节省 token。完整结果和日志随本地交付报告保存。
