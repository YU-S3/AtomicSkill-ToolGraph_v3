# LiveMath P5：从成功与失败 Train Trace 合成 Atomic + Program 的轻量学习实验

**性质**：可交给 Codex 的待实施方案；不是已经跑通的 P5 付费结果。**本次仅负责自动学习改进，不使用、不导入用户的人工理想 Bank**。证据基线：上传的 `LiveMath_P4_完整运行产物审查包_20261014.zip`，实跑提交 `57ae805886e3902dc9c82bce918fc818bb244195`；在审查时仓库 `main=a418866a03309c74bd61e619178c72509d6e8f42`。实施前记录实际 HEAD/diff，使用独立工作树或分支，不修改 P4 原件，也不影响 SearchQA R2 / OfficeQA 运行。

## 0. 总体决定

- **可以从成功或失败的 Train 经验提出新的可执行 Program 假设**。不要求旧 trace 已成功运行同一个 Program；但是“提出”≠“已经完成操作”≠“可发布 usable”。
- 首轮做**小规模、真实模型自动生成**：由项目的 Extractor 从冻结的 Train60 真实题目／旧过程候选归纳 Atomic 操作与 ProgramSpec，再由同一锁定模型的 Builder 自主写 `def run(ctx, inputs)`；随后用现有 Docker Worker 实际执行。**不把本文附带的人工原型代码喂给模型，也不手工塞入 Ours Bank**。
- 首轮不重解 Train60，不跑 Test100，暂不重跑完整 Val17，不修改 P4 `+2/8次暴露` 等历史晋升规则。减少*候选生成之前*的证据门槛，而不是取消 Worker、Host 权限或正式评分约束。
- 本次试验的目标是证明“可以**自主**从历史有成功／失败的经验形成 Atomic + Program + Implementation 绑定，并在至少一个非原始输入上执行正确的**局部数学操作**”。不承诺能自动证明任意高级 PDE、几何或表示论命题；不得用有限枚举假充普遍性证明。

## 1. 审查定位（严格区分事实和推断）

### 1.1 P4 实际数据

- 原始 Train60：最终答案正确 **13/60**，错误47；格式有效41，`truncated_empty`18，另有其他空答1。原 trace 仅3次临时计算尝试，其中2次失败、1次成功却未被最终回答实际消费。因此“成功工具执行经验不足”属实，但不是“完全没有可学操作”。
- P4：30份来源检查合格提案中，15份通过后续内容检查；跨组整合留5，4进Val初筛、2进确认，最后**1条 `guidance_only` Skill**晋升，**Program=0**。进入Val的四个候选对应：量词10组配对4→4；存在／唯一性13组2→4（+2、无回退但额外耗 Token）；边界0组；渐近4组2→1。
- P4 Val54次实际 Runtime 均有合法答案、空答0；主要错误是数学结论与官方答案不一致，非完成协议截断。已知“表示论 epsilon factor”任务被错误匹配给 `asymptotic_scale`，说明路由适用前提不足。
- **重要的代码级根因**：`src/atomic_skillgraph/empirical/math_trace_learning.py::trace_records` 的来源审计直接给出 `program_learning='guidance_only; no Program promotion...'`；`math_capability.py::validate_candidate` 将候选固定为 `knowledge_type='procedural_advisory'`、`execution_intent='guidance_only'`；`run_math_p3.py` 的 P4 分支是`原子学习→合并→文字 Guidance Val 选择`。**P4 根本没有“经验→ProgramSpec→Builder→Worker”生产链**，因此Program=0不能仅归因于模型能力。
- P4 的15条已检查 Atomic 中有2条 `finite_computation`，但5条合并候选里为0；这既是能力覆盖损失，也意味着必须将 Atomic 原子保留与跨组摘要分开。

### 1.2 一次已执行的本地可行性探针（非自动学习成绩）

独立探针 `trace_to_program_pilot_20261015.zip` 中，依据真实失败 Train `livemath:202602:12` 的**公开**模同余选项，辅助合成一个接受参数化谓词的有限同余比较 Program。本地测试 n=6..60 得到B⇒D在该有限范围成立，D⇒B有反例 `(n=6,m=2)`，非法范围返回 `needs_input`。**这项程序由审查者辅助编写，不是仓库 Learner/Builder 自主产物；也未证明 `H(n,m)^ab ≅ Z²` 的条件或选项正确性，更不是正式 Worker 资格。**它只证实存在可形式化的小数学操作，并提供协议回归思路。

## 2. 源数据与严格边界

1. 父来源固定为 P4 包内原始 Train60 的任务和 Runtime trace（`source_train60/`、`p4_run/source_records.json`、`source_cases.json`；真实路径按包的 `SOURCE_MAP.json`/manifest）；逐条验证60/60 task identity、原始 trace SHA、公开输入与记录评分一致。
2. **可以复用全部15个在 P4 Val 之前就已检查合格的单元级 Atomic 候选**作为 Train-only 来源摘要。它们只是“候选操作描述”，不继承 P4 后续 Val 的 `screen_ids`、`passed_ids`、`utility_receipts` 或训练选择排序。按固定原单元次序提供所有15条；不因已看过的 Val 表现选择或排除某条。新 P5 Bank 从空开始，不继承 P4 已晋升资产。
3. Train `hard` 可供 Host 标注 `final_correct/final_wrong/empty`，但不得给模型伪造“错误原因已知”；公开题目、选择项、已保存的模型答案、合法局部结果可被索引。**绝不提供 `correct_choice`、评测答案、Val/Test 输出或人工理想 Bank**。
4. 真实执行成功的局部结果标 `observed_operation`；正确终局、但没有执行证明的局部步骤标 `answer_supported_hypothesis`；错误/空答中的操作需求标 `problem_derived_hypothesis`。后二者允许生成 Program 候选，**不能回填成 `verified_execution`**。
5. 旧P4所有文件、Bank/Frozen、评分、Token账本只读。新目录按 `source_sha + method_version + code_sha + policy_sha` 固定，已付费请求不得跨内容身份重用。

## 3. 实际实现：从真实 Trace 到完整 Program Bundle

### 3.1 新模块与最小改动范围

新增：

- `src/atomic_skillgraph/empirical/math_program_induction.py`：候选操作归纳、`ProgramSpec` Schema、Train-only 证据引用、能力去重与生成计划。
- `src/atomic_skillgraph/empirical/math_program_probe.py`：仅用于新方法的 Program 试用、变参检查和收据、Atomic↔Program 关联；复用原 `ProgramWorker`。
- `src/atomic_skillgraph/experiments/run_math_program_p5.py`：`prepare/learn/probe/report` 四个阶段的耐中断薄控制器；**首轮不增加 Test/Val 自动运行子命令**。
- `configs/livemath_p5_program_probe_seed42.yaml`：独立方法/预算/模型/来源身份。
- `tests/test_math_program_p5.py`：零HTTP的真来源与模拟 Provider 工程回归。

按需**极小范围**改 `math_trace_learning.py` 输出的 Train-only catalog 投影，及 `system.py::validate_config` 的新方法身份/`purpose_overrides` 授权；绝不重写 P4 的现有选项学习器、`Bank.freeze` 或 `AnswerExecutor`。P5 Program 创建使用已有 `ProgramWorker`、`contracts.validate_program`、`program_submission`、`Bank.put` 原生结构；现有 `local_validation`/`program_eligible` 对**正式发布**仍维持原样。

### 3.2 Host 先构造候选机会，而非一开始就要求存在历史 Program

从60条 Train + 15条已检查的 Train-only 原子候选，筛选“可能具有有限、可判定子操作”的**公开需求**，如：

- 模同余与有限反例：给定正整数、范围和明确条件，枚举反例或关系包含；只输出有限结论。
- 显式有限量词：给定有限集合和布尔关系，检查 `∀x∃y` 与 `∃y∀x` 的依赖区别；不得扩展到无限域的一般命题。
- 形式化强弱检查：给出结构化的蕴含／数值上下界前提，报告可演算比较；不知道外部定理则 `needs_input`。
- 有理数、有限序列、矩阵的确定性比较；不可把数值实验冒充 PDE 等领域定理证明。

模型可以基于**失败任务**提出新的操作，如“有限反例检查器”，也可以基于成功任务提出操作，如“可直接核对的代数计算器”；但每项都需明确自身不能证明的部分。不要预设必须每个数学类别生成代码，无法明确操作时合法返回 `no_change`。

Host 形成 `math_operation_opportunity.v1`：

```json
{
  "opportunity_id": "op_<host_digest>",
  "source_case_ids": ["<真实Train physical_key>"],
  "origin": "observed_operation | answer_supported_hypothesis | problem_derived_hypothesis",
  "public_source_refs": [{"case_id":"...","span_id":"..."}],
  "requested_local_operation": "finite_congruence_comparison",
  "limits": {"finite_only":true,"max_enumerated_cases":100000},
  "source_trace_hashes": ["<原始记录sha256>"]
}
```

这个机会目录是 Host 对**真实公开材料**的投影，不能带 gold，也不能声称模型已执行新操作；`requested_local_operation` 可以是待验证归类而非真值标签。

### 3.3 先 Atomic/ProgramSpec，再 Builder 写代码（只准自然产生）

Extractor 输出一个结构化 `atomic_program_spec.v1`；不允许在首次提交阶段用任意自由文字直接发布：

```json
{
 "decision":"propose_program | guidance_only | no_change",
 "source_opportunity_ids":["op_<host_digest>"],
 "atomic":{"goal":"...","applicability":"...","prerequisites":["..."],
           "operation":"...","outputs":["..."] ,"stop_condition":"...",
           "result_role":"intermediate"},
 "program_spec":{"input_schema":{"type":"object","properties":{},"required":[]},
                 "output_schema":{"type":"object","properties":{},"required":[]},
                 "allowed_tools":[],"finite_scope":"...","negative_cases":["..."]},
 "limitations":"Finite checks do not prove universal theorems"
}
```

- `allowed_tools=[]` 是首轮数学纯计算候选的默认规则；可读取公开字符串或有限显式参数，不得在代码中读取评测器、网络或隐藏答案。
- Host 验证 source IDs 真存在、I/O 结构及 `result_role=intermediate`；Builder 必须通过现有 `tool_builder` 请求从此 Spec 生成完整 `def run(ctx, inputs)`，返回 `{'status':..., 'outputs':...}`。至少一个明确的结果字段可以被 Runtime/后续节点消费；禁止把某个 Train 题的正确选项硬编码在代码里。
- 新的程序是“**有来源的合成代码**”，`generation_basis='hypothesis_from_train'`，而不是“历史执行过的程序”。若来源为失败，明确附带 `parent_outcome=wrong_or_empty`。没有成功执行操作时**仍可提案**；不得伪造过去的 `program_calls`/`result_store`。
- 单项拒绝不连带整批；本轮每个机会最多1个候选、最多1次真实 Builder 修复；结构化错误按候选索引归档。不要因生成量不达标自动循环追加 prompt。

### 3.4 真 Worker 轻验证，不做整题证明

首轮从已有代码中复用 `ProgramWorker.execute()` 的**锁定 Linux Docker**，不允许在宿主上直接 `exec` 生成代码；执行前调用 `validate_program`、锁定 `program_environment`、检查 schema，禁止网络和 evaluator。

每个 Program 最多做以下低成本检测：

1. 对一个已定义的**Train 公开局部输入**实际运行；输入不能包含 correct_choice，输出不能从旧正确最终答案硬编码。
2. 至少一个合法变参输入：由 Host 在同类有限定义域内产生，独立检查不变量/小范围穷举；代码不应依赖 source task_id。
3. 至少一个非法或资料不足输入：必须返回 `needs_input`、`not_found` 或 `blocked`，而不是猜测值。
4. 记录源结果、变参结果、`output_contract_status`、Docker exit、RPC调用、耗时和失败反馈；**不能把一次正例通过叫做一般定理证明**。

只进行**局部数学操作**正确性检查，不要求完整题目答对、不要求旧 Trace 中本来就有相同执行、也不要求第一轮即得到8题 Val 暴露。这正是本次简化验证的地方。

候选生命周期：`proposed → code_generated → probe_passed | probe_failed → train_trial_only`。`probe_passed` 仅表示局部执行通过，**不是现有 `Bank.program_eligible` 的 usable**。对 `probe_passed` 输出完整 Bundle：Atomic 元数据、Program源码、Implementation候选关系、实际收据、代码 hash、来源 IDs。使用独立开发试用视图调用 Worker，并将返回的**真实结果**交给模型，不能把候选改成旧正式 Bank 的 `usable` 绕过资格检查。

### 3.5 仅在 Train 做一次消费者验证（可选）

若有 `probe_passed` Program，最多选择预先固定的两道**未用作该候选来源**的 Train 实例，在相同模型与预算下做`without_program`/`with_actual_program_result`小对照。程序在真实 Worker 里执行，Host 将 `result_id`/字段交给模型，模型要显式引用实际字段才能记 `consumed=true`；无法形式化的题目不要求强制调用。

输出三种结论：`executed_and_consumed`、`executed_not_consumed`、`not_applicable_or_failed`。记录是否答对、总 tokens、模型 HTTP（含学习和 Builder），但不得把这种小规模 Train 诊断当作 Val/Test 泛化收益。**第一轮不自动运行完整 Val17**，不修改旧P4晋升门槛。下一步是否开展正式新版 Val，先看真实自动代码数量、局部检查情况和结果消费。

## 4. 小规模试验的预注册设计与预算

- 先离线固定一个 `pilot_manifest.json`：Seed42、Train60来源SHA、15条旧 Train-only Atomic 目录SHA、源码SHA、模型响应真实身份、操作类别、抽样次序。优先有限计算/量词/同余类，来源中既有成功也有失败；固定至多6个操作机会，不因某次结果不好而追加新组。留出非来源 Train 只作为可选 probe，绝不使用 Val17/Test100 真值决定候选。
- 允许6次 Extractor 决策、每个最多1次 Builder及至多1次受控修复，最多按真预算约束执行；没有合法 Program 也正常结束。先跑0 HTTP假响应回放和真正 Docker fixture，再单独放行真实付费生成。
- **建议但尚未授权**：付费学习/Builder最多 **800,000 tokens / 60 HTTP**；可选 Train-only 消费者配对最多 **1,200,000 tokens / 24 HTTP**，总计新增不超过 **2,000,000 tokens**；两个池不可互借。只有工具完全合规/公开材料齐全才允许执行。实际接受的 Governor 仍是45M总额、9,165次HTTP，原 P4 报告显示855个已计费请求合计13,904,577 tokens，另有一笔历史39,819 token未决预留，累计保守占用**13,944,396**。新探索必须在同一个累计账本中结转，不能“新run刷新45M”，未决项不得按零计算。
- 真实首轮如果出现1个 `probe_passed` 且有一次合法变参执行，即可判断“自主生成可能可行”；只有非来源实际消费才支持复用；若全部失败，报告模型提案错误、Schema、代码、Worker、域外输入等分类并停止，不再强行造出Program。

## 5. Prompt 的具体强化（比单纯加长更重要）

Extractor 在现有 P4 指令之外加入：

> Identify a **finite, checkable local operation** in these public Train experiences, including failures. State explicitly whether the operation was observed or is only hypothesized from a problem's demand. Produce one atomic capability and an executable input/output contract. Do not solve the complete theorem or reuse the original option letter. Use no_change when no finite operation is meaningful. Output only the specified schema.

Builder 加：

> Produce a complete, parameterized `def run(ctx,inputs)` satisfying the frozen contract. Accept explicit JSON values; use no grader, no network, no prefilled answers. For a finite-domain operation, return the tested domain, an actual counterexample when one exists, and a clear limitation; return `needs_input` for missing premises. Never label numerical checks as a general mathematical proof. No unexplained literals from the source task.

上述是新增模型调用的**目标**，不能把这两段文本本身当成实现。Host仍负责真实参数/来源目录和 Schema；Producer/Consumer 记录可核对后才允许后续正式接入。

## 6. 验收测试（先零HTTP，再真实生成）

1. `P5-L01`：读取60份 Train 实录、公开任务及P4学习阶段15条 checked candidate；来源哈希、角色和原始比分准确；对 Val/Test 目录进行可检测的禁止读取测试。
2. `P5-L02`：失败 Train 可以产生 `problem_derived_hypothesis`，但所有输出中 `observed_execution=false`；不存在“错答直接反推出正确证明”的字段。
3. `P5-L03`：原子来源候选全部保留，`finite_computation` 不会因跨组整合自动消失；新Program必须有明确Atomic链接和I/O Schema。
4. `P5-L04`：模拟 Provider 产出一个有效有限计算 Program、一个引用非法 source 的 Program、一个强制选项字母 Program；仅第一项进入probe，其他逐项拒绝；不发真实 HTTP。
5. `P5-L05`：真实 Docker Worker 在正例／变参／无效输入各执行一次；不支持 Docker 时 `blocked` 且不得宿主 fallback。完成后输出可复核源码和收据。
6. `P5-L06`：新目录旧P4冻结库哈希不变；即使程序 `probe_passed`，旧 `Bank.program_eligible` 和官方 `checked_asset` 仍不能将它视作正式 usable；候选在开发试用中可由独立 runner 安全执行。
7. `P5-L07`：checkpoint 在已收到付费响应后重启不会重复 Builder HTTP；失败有明确 `model_invalid/worker_failed/probe_inconclusive/budget_stopped`；所有 HTTP usage、未知账单和阶段池留痕。
8. `P5-L08`：若没有合法 Program，完整流程输出负面审查报告并终止；不启用从 Test 反向学习或降低数学正确性条件。

## 7. 产物、命令与停机点

**以下是 Codex 实施完成后应提供的 CLI 合同，当前 main 尚无此命令，不能把示例写成已经可直接运行。**

```bash
python -m atomic_skillgraph.experiments.run_math_program_p5 prepare \
  --source <P4_原始Train60目录> --output <NEW_P5_DIR> --config configs/livemath_p5_program_probe_seed42.yaml
python -m atomic_skillgraph.experiments.run_math_program_p5 check --output <NEW_P5_DIR> --zero-http
# 仅明确通过零HTTP审查和额度放行后：
python -m atomic_skillgraph.experiments.run_math_program_p5 learn --output <NEW_P5_DIR> --env-file <ENV>
python -m atomic_skillgraph.experiments.run_math_program_p5 probe --output <NEW_P5_DIR>
python -m atomic_skillgraph.experiments.run_math_program_p5 report --output <NEW_P5_DIR>
```

至少产出：`pilot_manifest.json`、`opportunity_catalog.jsonl`、`spec_proposals.jsonl`、`programs/*.py`、`program_bundles.jsonl`、`worker_receipts.jsonl`、`train_probe_pairs.jsonl`（若实际执行）、`usage_ledger.json`、`pilot_review.md/json`。报告逐条标注程序的来源（成功／失败）、是否自主生成、是否真实运行、是否变参通过、是否被消费、总费用。

**停机点：先实现 A（来源与候选目录）→ B（真实 Builder）→ C（Worker有限 probe）→ 产物审查，**不要以“还没有训练出足够usable”为由继续无限调用；只有探针确实产出有价值的 Program，才考虑另一份正式新版 Val 实验方案。

**禁止事项**：复用人工理想Bank作为Ours学习产物；将本助手离线示例代码复制为模型生成；重解Train60；使用P4的Val17/Test100反馈给P5 Learner；把有限计算宣称成高阶定理证明；绕过Docker/禁止调用评测器；未经许可超支或自动启动Test。
