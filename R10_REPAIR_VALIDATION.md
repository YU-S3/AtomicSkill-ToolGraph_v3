# R10 修复与真实环境验收

基线：`4646c8c1db3ac9b5faabe0251a3f661712684d78`。实现提交：`26b07793454ae3d375ec857b46b1efd42302a99a`。仅修改 v3/main；未修改 baseline，也未启动或恢复正式实验。

**当前结论：T5 已按用户确认修订并 resolved；success-authority 必修项已完成，最终全量测试、真实 promotion/reuse 与 provider probe 均通过，可放行 fresh R10 seed42 Full-120 → Frozen-134。最新代码指纹与收尾证据见末节。此结论是机制/协议放行，不保证正式任务全部成功。**

## 实现范围

- P0 complete eligibility 先于 Candidate bootstrap 排名执行；terminal-empirical 不参与竞争。保留 lexical 排名与 exact-contract hard filter。
- Preparation / Seeded 共用 fresh、单 semantic ToolCall 的 RuntimeStep；一次图 bootstrap。Full Dynamic 保留原路径。
- 同一 occurrence 的 Preparation / Seeded / AutomationDraft 共用 100K；Runtime 与 Runtime ToolBuilder 共用任务 300K。Builder 不额外占用 occurrence 配额；失败及 rollback 不退款。
- typed RejectedRuntimeCandidate 保留角色/值/失败码/证据与精确参数组范围。缓存键包含 world、revision、binding、semantic anchor、grounding evidence、Repeat state、Atomic identity；仅缓存确定性 validator/preflight 拒绝，同状态跨 fresh Step 复用，状态改变后失效，rollback 不抹去失败记忆。不同参数组合不相互封禁。
- Stored Composite / Atomic Composition 共用 VerifiedCompositeExecutor：effect-first、现有数据流与证据校验、precondition gate、唯一或唯一 preferred implementation；歧义交给 Agent。
- 普通 Atomic 的 typed binding / predicate Support 递归闭包。只有 Active 可自动执行，Candidate 仍由 Agent 选择并经过普通策略。未知角色不从无关 affordance 偶然绑定；semantic discovery input 保留正式语义锚点，不伪装成 concrete。
- 自动执行与 task-local trial 的 checkpoint / reset + canonical accepted-prefix replay；校验恢复 digest，保留失败分支审计与负向修复证据，正向学习排除回滚分支。
- Automation DSL 按需加载。跨任务 staging exactly-once；两任务成功后按 formal signature 去重、交叉重放，再经过普通 Admission / Aligner / Credit / Lifecycle 注册 Candidate。无伪造 Direct credit、无强制 Active。
- 新增 R10 计数、报告、隔离验收入口与 seed42 train120 / frozen134 配置；保留原模型、reasoning、预算、任务 manifests。

## 首轮测试与代码指纹（26b0779）

- 最终仓库内全量 pytest：**1335 passed in 38.30s**，`release_pytest.log`（R10 专项 28 passed）。
- `git diff --check`：通过。
- 最终真实 provider A/B/C probe：通过。
- 最终 code hash：`2a02a460ea798c12c2091f57cf63bf06465eea2b3fd32a3bd89e82ef39719b39`。
- train config hash：`5387e563b8a3d264086ffd4286ae7e7f8446a5455e062428b1959e6429e5c11d`。
- 独立 ext4 代码副本与仓库的上述代码指纹一致。早期一次不含 `.git` 的副本 pytest 因旧验收脚本读取 HEAD 失败（1332 passed / 1 failed）；未修改该正常行为，改回实际 Git 仓库运行全套通过。后来补齐 typed negative memory 后再跑全套，得到上面的 1335 passed。

## 真实 ALFWorld 验收（不是正式成功率实验）

| 项目 | 实测证据与限制 |
| --- | --- |
| T1 learned Stored Composite | 最终 `release_stored`：复制旧 v3.2 已训练 Frozen bank，真实 task0 严格成功，47.28 秒；bootstrap=1，全部 RuntimeStep=1，后续无 Runtime LLM 节点=2，三个节点均通过，bank digest 不变、code_unchanged=true。Trace `trace_70fa74a783234c49b7c83ed2e96fa486`。 |
| T2 Atomic Composition | `look_frozen_01`：真实 P1/P2 产生 atomic_composition 并进入同一短步执行器。任务未成功：计划包含 acquire-lamp，运行后节点预算耗尽。此证据只能证明路径，不能当成功率通过。另有 promotion/reuse 的显式测试父计划，不能冒充真实 P1/P2。 |
| T3 recursive Support | `final_world` / 最终复测 `release_world`：真实 ALFWorld、显式通用 fixture Atomic，GO_TO→OPEN 两层递归成功，0 provider calls。fixture 不进入正式 bank。 |
| T4 rollback | 同上：Tool 两个实际接受动作后失败；reset + replay 两个先前 canonical 动作，恢复 digest 完全一致。失败分支两个动作保留在 Trace、排除 canonical，预算不退还。 |
| T5 promotion / reuse | `promotion_05` 与固定版本复测 `final_promotion`：真实 API 在 task0 / task1 独立生成草案和程序，相同 formal signature，两个来源 cross-replay 均通过，普通 Candidate 晋升。最终 `release_reuse` 复用 `promotion_05` 的产物：task3 / task4 经 Agent 选择 Candidate，实际成功后普通生命周期转 Active；task7 自动 SupportClosure 成功，Runtime ToolBuilder=0、自动 Support=1。三个后续任务均严格成功，code_unchanged=true。不是“两来源任务后第三个任务立即自动”。 |
| T6 complete Candidate bootstrap | `look_online_01`：复制旧 bank，10 个 terminal-empirical 在排名前排除，7 个 complete eligible，complete Candidate `composite_5e43dfb30d5487183ad794fd` 获得真实 stored deployment。最终任务未成功，不能当作 look-at 成功率通过。 |
| T7 provider | 最终实现提交后的 `release_provider_probe`：真实 DeepSeek provider capability passed，绑定上述最终 code/config hash；未调低 reasoning 或更改模型。 |

隔离真实调用入口会明确声明 fixture：世界回滚验收使用通用手写合约；晋升验收限定单个父 Atomic 与初始工具菜单/契约边界，草案和 Tool IR 由实际模型生成。它们仅用于机制验收，不作为正式 benchmark 数据。历史复制 bank、原实验 Trace 与输出均未修改。

### 保留的反例，不能用成功案例覆盖

- `final_promotion` 新生成的另一个 Tool，虽也通过两个来源 cross-replay 和两个 Candidate 真实部署，但在 `final_reuse` task7 自动复用时出现 `tool_ir_action_unavailable`：程序 OPEN 其他 cabinet 后，旧循环项的 `GO_TO cabinet_2` 已不在当前 catalog。系统拒绝执行并回滚，随后 Agent/新 trial 救援，最终任务成功；该次 **support_auto_execution_count=0、Runtime ToolBuilder=2**，不算零 Builder 自动复用通过。
- 上述属于已生成程序在新状态上的泛化失败，不能以修改 action-catalog authority、跳过合法性校验或补 benchmark 流程来“修绿”。生产代码保留失败信用、回滚和 Agent breakpoint；报告保留失败证据。
- `look_frozen_01` / `look_online_01` 两个真实任务失败均保留。P0 排名修复和统一 executor 路径的证据，不等价于已证明 look-at 任务成功率或所有新 Tool 的泛化稳定性。
- `final_*` 是补 typed negative memory 前的固定快照（code hash `c221dffc86265f24e6389c52c04fa6b820b0141ee81b807ab3a398826e4de5c5`）；最终版本证据用 `release_*` 命名。没有覆盖或改写旧实验 Trace。

## T5 验收修订：Resolved

R10 第 9 节 / Support-6 要求 **Active-only 自动闭包、Candidate 必须由 Agent 选择**；Promotion-2 又要求晋升为普通 Candidate；T5 却要求两来源任务后第三个任务直接自动复用。现有 implementation/tool 生命周期要求两个真实独立 started deployment successes，不能把 R1/cross-replay 冒充 Direct。

用户已确认按现有生命周期验收，原冻结文档的“2→3 immediate auto”要求撤销，不以放宽 Candidate 或伪造 deployment credit 满足错误条款。正式 T5 名称修订为 **Runtime Support Promotion → Ordinary Activation → Deterministic Reuse**：

1. **T5-A Promotion**：两个独立成功任务的有效 R1、父节点完成与可审计学习证据；cross-source replay 通过；创建普通 Candidate Atomic / Implementation / Tool。
2. **T5-B Activation**：按普通 ONLINE policy 由 Agent 真实选择和部署，达到现有生命周期阈值后转 Active；R1/replay 不冒充 deployment success。
3. **T5-C Reuse**：Active 后合适的独立任务由 SupportClosure 自动调用并验证输出；该确定性 Support 路径 LLM=0、Runtime ToolBuilder=0。不是整个任务所有阶段 LLM=0，也不要求固定“第三任务”。

已有 `task0/task1→Candidate→task3/task4 真实部署→Active→task7 自动复用` 作为这条链的正式机制验收证据。保持 Active-only、CandidateUsePolicy、生命周期阈值、预算和工具合法性检查全部不变。

原始验收目录：WSL `/home/yangchengyu/asg_r10_validation/`。便于 Windows 核查的副本：`runs/r10_validation/`（本地实验产物，不入 Git）。

## Success-authority 最终收尾

针对用户确认后的唯一代码必修项，实现提交 `a0142a4`：`collect_observations()` 移除 `task_contract_success` hard gate，保留 `benchmark_success`、`learning_eligible`、非 infrastructure failure、非 Frozen，以及有效 R1 / parent completion 等要求。`task_contract_success` 原字段和报告诊断含义不变。与现有 `apply_terminal_outcome()` 的 `strict_task_success = benchmark_success` 对齐。

新增回归用例先在旧代码上失败（预期 1 条 observation，实际 0），修复后通过；覆盖 benchmark won=true、strict=true、learning_eligible=true、task_contract diagnostic=false 的合法场景，并检查 benchmark 失败 / learning 不合格 / infra failure 仍拒绝学习。

- 全量 pytest：**1336 passed in 40.91s**。
- code hash：`89277718c5bbebd94294269d24a2d55a722bb793497191009bb747e842491741`。
- train config hash 不变：`5387e563b8a3d264086ffd4286ae7e7f8446a5455e062428b1959e6429e5c11d`。
- 实现提交后的真实 provider capability probe：passed。
- 新版真实 reuse 回归：使用已验收 `promotion_05` 的 Candidate 产物独立复制；task3/task4 普通 Agent 选择、部署后转 Active；task7 自动 Support 成功。3 个任务均成功，Runtime ToolBuilder 均为 0；不是将历史 Trace 当成本轮运行结果。
- 新版真实 promotion 回归：新的空验收 bank、实际 API 独立生成草案和 Tool；task0/task1 均成功（132.61 / 103.95 秒），formal signature 一致，第一任务仅 observation、第二任务两个 cross-source replay 全通过后创建普通 Candidate。Trace 分别为 `trace_41ec39fd04454fcab09254d2208209d0`、`trace_18fa5786e315416ea54f5a3379d43af7`。
- 新版 reuse Trace：task3 `trace_78bd6e2e09614a4f9ca5e619eeca419b`、task4 `trace_3633a2f996b84ff584e971fd7dd3a867`、task7 `trace_cc8ce82bbd124a7b93715cabfce61705`；耗时 20.31 / 17.21 / 22.75 秒。此复用回归明确使用原已验收 Candidate bank，不声称复用的是本轮新生成的另一份 Tool。
- 两个真实验收进程均正常结束，code_unchanged=true；仓库与隔离代码副本的实验相关文件指纹一致。`git diff --check` 通过。

结论：按用户确认后的 T5-A/B/C 接受已有完整链，并在本轮分别复测 promotion 与普通 activation/reuse。唯一新增代码 blocker 已修复，无需扩修 Candidate exposure、P0、Tool IR、action-catalog、模型、reasoning 或预算；先前真实泛化失败继续保留。允许新的空 bank Full-120，不允许拿 R9.2 的旧正式 run resume 冒充本次 R10。

本轮结果目录：WSL `/home/yangchengyu/asg_r10_authority_validation/`；Windows 本地副本 `runs/r10_authority_validation/`。

## Budget-exception rollback 最终补丁

基于 `5fe7922` 核对后，确认自动 registered Tool 在部分 ACTION 成功、下一 ACTION 抛出 `BudgetExhausted` 时，会绕过只处理返回结果的回滚分支。实现提交 `29e63ef` 仅在 `NodeExecutor.try_autonomous()` 捕获该异常：累计已发生的 LLM-free 环境动作，按原 checkpoint 恢复世界与逻辑，然后原样重新抛出。正常失败返回路径不变；没有修改 ToolRunner 的异常语义，没有把预算耗尽转成 Tool intrinsic failure，没有退款。

- 新增 node/global 两个参数化回归，使用真实编译的两 ACTION registered Tool 和确定性测试 harness（不是正式 ALFWorld 成功率实验）。旧代码两例均在 world digest 恢复断言失败，修复后通过。
- 验证原 exhaustion code 保留；world、logical、binding、evidence 恢复；原始 Trace 保留一个已接受动作、canonical 排除该动作；node/global action 与 token/turn 计数不退款；rollback 和 LLM-free action 各计数一次；`TraceBuilder.finish()` 关闭异常遗留 span。
- 全量 pytest：**1338 passed in 40.92s**；`git diff --check` 通过。
- 最终 code hash：`d8b28af841e889bda11041f4ed90c34a5d98496d003b0d69b3ac27907cbc1da9`。
- train config hash 保持 `5387e563b8a3d264086ffd4286ae7e7f8446a5455e062428b1959e6429e5c11d`。
- 真实 provider probe 首次 HTTP 200，但 Probe A 缺少 reasoning_content，明确失败；未修改配置/门槛，独立复测 **passed**。首次失败保存在 `provider_probe/`，通过记录在 `provider_probe_retry/`，没有覆盖失败证据。
- 本轮没有重跑 promotion/reuse，没有更改 P0、Support、Candidate 生命周期、RuntimeStep、模型、reasoning、100K/300K 或 Tool IR，也没有启动正式 train/test。

本轮验收目录：WSL `/home/yangchengyu/asg_r10_budget_validation/`；Windows 副本 `runs/r10_budget_validation/`。T2 按真实 P1/P2 → atomic_composition → 同一 executor 的路径验收，不把目标任务 won 追加为该实现门槛；已有任务失败记录继续保留。

结论：本次确认的 budget-exception rollback blocker 已消除，可按既有指令启动新的空 bank R10 seed42 Full-120 → Frozen-134。放行不代表正式任务全部成功或 provider 永不波动；正式入口原有 provider gate 继续生效。
