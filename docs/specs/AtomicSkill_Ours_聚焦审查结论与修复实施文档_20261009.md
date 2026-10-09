# AtomicSkill Ours 聚焦审查结论与修复实施文档

**审查日期：2026-10-09。状态：修改规格已明确；本次未修改生产代码、Bank 或运行目录，未发模型请求。**

## 1. 交付结论与适用范围

**现有实现需要修改，不能原样继续扩大为多 benchmark × 3 seed 的新正式实验。** 这次材料已经足够确定主要问题，不需要再用正式实验来定位，也不需要重新索取整套审查材料。

修改分成两类，实施和报告中必须区分：

- **工程修复**：历史案例结果来源串线；Spreadsheet 整列范围评分及执行后恢复；预算结束后的学习状态；角色响应协议矛盾；重复材料与失控的失败材料展开。
- **实验协议或方法调整**：LiveMath 补齐固定上游的选项打乱、重标；缺少纠错证据时停止猜测性 guidance 学习；短 guidance 专用生成配置；首次 Builder 等两个合法独立 Train 案例，并配套补例队列。

本文件给出明确的修改位置、行为和验收要求。新配置是待实施、待有限验证的配置，不能声称已修复或已经取得新方法成绩。本轮保留 Program 两独立 Train 正向试用后的接管规则；不以取消 Program、减少资格试用或清空全部 Bank 代替修复。

审查对象是上传的“本轮正式运行聚焦审查包”，不是 10 月 7 日的旧进度快照。本包已经包含 LiveMath 的 300 个 Test 运行结果。本文报告的是包内快照，未查询用户运行机器的实时进程状态。

### 1.1 来源与证据边界

- 审查包 SHA256：`517b386a73c2ed80bd306e9916394f53d285b8eee9099354c1d3b2b7767471e9`。249 个清单条目全部通过大小及 SHA256 校验；包内 9 份源码与已钉住的 eb6d652 源码一致。
- 主要正式运行版本：`eb6d6522ba630d6643f25f90966011c595e1b327`；Office43/44 的恢复子运行使用 `85d3e633b1fc6a2eb2d580857e8eef2e88a6b934`。
- 本次重新核对的仓库 main 为 85d3e63。本文涉及的来源串线、评分器、运行恢复入口和主要成本逻辑仍存在，不能将它们说成“旧代码已经修好”。
- 包内有全量 HTTP 费用元数据、学习/试用/资产元数据及 24 个定向原始请求样本；没有每次请求的完整正文。Test 的逐题答案正文有意省略，pending checkpoint 也经过白名单裁剪。
- LiveMath 的 231 条 Train+Val 保存预测已重新用冻结 parser 核对；Test 的 300 条仅核对分类账本与请求状态，未宣称重新读取全部 Test 原始答案重评分。

本文的证据编号与附带 JSON 一致：

| 编号 | 对应证据内容 |
|---|---|
| E1 | 包完整性、源码版本和官方/上游源码身份 |
| E2 | HTTP 总账、用途分桶、完成题成本、repair 与 retry |
| E3 | LiveMath 分类、输入核验、学习来源、原始请求 17–20 |
| E4 | 367 条案例结果错配、19 个实际 Builder 请求和关联 Program |
| E5 | Spreadsheet 第 91 题、范围纯函数复现与恢复边界 |
| E6 | 资产/job/试用/在线调用、单案例提前 Builder、预算结束跳过学习 |
| E7 | 原始请求材料组成、重复字段和 Builder 失败材料 |
| E8 | 本次执行的固定上游选项置换纯函数检查 |

## 2. 已确认的问题与优先级

| 修改项 | 现场证据 | 优先级及性质 |
|---|---|---|
| M1 LiveMath 选项投影 | 177/177 正确标签都是 A；上游启用 shuffle/relabel，本地漏掉该步骤 | P0，实验输入协议修复 |
| M2 历史案例结果来源隔离 | 367 个事件取到了当前题同索引结果；19 份材料已真正请求 Builder | P0，工程修复 |
| M3 Spreadsheet 范围评分与恢复 | `spreadsheet:283-32` 的 A:G 触发 int('')；执行产物已存在，尚未评分 | P0，评分器及恢复修复 |
| M4 预算终止后的学习边界 | Sheet 80 道已评分题 learning_status=not_started，其中 41 道正确 | P0，控制流与状态修复 |
| M5 响应协议及输入材料 | 已见 Planner 直接调用环境工具、Runtime wrapper 错误；大量合同/job 重复和嵌套全轨迹 | P1，工程输入/响应协议修复 |
| M6 有证据的短 guidance 学习 | 180 次全 upsert，125 个错误/空答来源也学；已保存错误的替代数学规则 | P1，学习策略与角色配置调整 |
| M7 两例后首次 Builder | 401 个已有 Program 的 job 只有 1 个 binding，仍在等第二例 | P1，方法调度调整 |

P0 表示先消除已确认的正确性或状态问题。P1 不是要求等正式跑完再处理；准备新的低成本实验版本时一并落地，但要记录它们改变了哪些模型输入或 Bank 演化过程。

### 2.1 现在可以排除的解释

LiveMath 不是主要被 parser 格式误杀；试用模型续解不是训练费用大头；最后几次 402/评分退出不是 312.75M Train tokens 的主要来源；Program 堆积也不是普遍的同一 job 无限多 epoch 重建。[E2、E3、E6]

这些结论替代此前材料不足时的怀疑项。后续不再围绕这些已排除的解释反复安排收费实验。

## 3. M1：修复 LiveMath 的选项投影

### 3.1 实际缺陷及结论限度

本地 `src/skillcompiler_bench_contracts/livemath.py` 会从独立的 correct_choice 补回缺失的正确候选，随后排序。本批 177 条原始记录均是 choices 列 B–E、correct_choice 单列 A；补齐候选本身是必要的，但本地停在了这里。

项目声明钉住的 SkillOpt 上游，在规范化之后还执行以下操作：

1. 默认 `shuffle_choices=True`。
2. 用 SHA256(`seed:item_id`) 的前 16 个十六进制字符构造局部随机种子。
3. 对完整选项做 shuffle。
4. 按 A、B、C……重新标记，并按原正确候选的身份同步正确标签。

来源：[固定上游 dataloader](https://github.com/microsoft/SkillOpt/blob/fa4ca184573e42ec11472959dd57422381418096/skillopt/envs/livemathematicianbench/dataloader.py)。本地 prepare/materialize 路径未完成这一步。[E1、E3]

因此，“531 次实际输入都等于冻结清单”只证明运行忠于清单，没有证明清单投影正确。恒定正确位置提供了与数学能力无关的捷径。**修复的目标是恢复有效的比较协议，不能承诺仅靠 shuffle 提升 Ours 准确率。**

### 3.2 必须实现的行为

接点：`normalize_livemath_item` 之后、公共 task 与 evaluator record 分流之前；同步调整 `prepare_benchmarks.prepare_resources` 和 `canonical_manifest.materialize` 的版本/一致性检查。

实现纯函数 `permute_livemath_choices(normalized, *, choice_seed, stable_item_id)`：

- 只从规范化后的原始完整候选构造新对象，不能原地修改 canonical 输入。
- 本版明确使用独立固定的 **choice_seed=42**；与运行 seed 42/43/44 分开。三个运行 seed 和所有比较方法，对同一物理题使用同一排列。上游提供算法依据；固定排列 seed 是本项目明确声明的控制选择。
- stable_item_id 使用上游规范化记录的 `id=month:no`，本批可从 task_id 去掉 `livemath:` 前缀得到。physical_key 是另一份物理身份 hash，不能把它误当上游 item_id 来取随机种子；不用全局 RNG、Python hash 或当前列表序号。
- 按原候选唯一身份 remap gold，不能依赖“找到同文本就是正确选项”，以免重复文本产生歧义。
- 新 public choices 和 evaluator choices 必须来自同一个变换结果。只给模型新标签与文本，原正确标签、变换映射和 evaluator 元数据不进入模型输入。
- 增加内部 projection version 和 source fingerprint。相同版本、相同输入重复物化必须幂等；不能把已经重标的列表再 shuffle 一次。不同 seed/version 从 canonical 原输入重新生成。
- 保留 Train/Val/Test 的物理题成员；不按标签平衡或 Test 成绩选 seed。

记录新的 benchmark contract、public materialization 和 implementation identity；各 baseline 若使用同一任务协议，也必须消费相同公开投影。这里要求的是相同配置内容和输入语义，不是再索取一套文件清单。

### 3.3 本次已经做过的零调用检查

从钉住上游 AST 中只抽取纯 shuffle 方法，在本包 177 个冻结公开任务上使用一次固定 seed=42：

- 177/177 原对象不变、变换确定、文本多重集不变、正确选项内容不变、标签唯一且 gold 同步。
- 对另外两个运行 seed 的 354 个任务映射检查一致。
- 原正确标签分布是 A=177；该次固定排列总计 A=33、B=30、C=45、D=38、E=31。没有搜索其他 seed，也不要求人为均衡。
- 未改生产 manifest，未重标旧预测，未生成新答案。本检查证明算法适配实际候选格式，不代表生产接入已验收。[E8]

### 3.4 旧结果如何处理

保留旧协议原始成绩、费用与 Bank。旧 LiveMath Test 是旧 projection 的结果；不能只改旧预测标签就变成新协议成绩，也不能把旧 final Bank 当作新协议从头训练所得。

需要新的正式 LiveMath 比较时，在新投影和声明的学习策略下建立新的 Train Bank，再冻结执行后续评估。**这项未来重新生成的必要性不能用于要求现在开三 seed 正式实验查工程问题。** 当前先按第 10 节完成有限验证。

固定上游 rollout 默认也不注入 theorem/sketch。本包部分公开题有未展开的前提引用，这是数据可解性限制；目前不能据此直接加入全文定理/证明，形成另一套未声明的输入协议。

## 4. M2：历史案例结果来源隔离

### 4.1 缺陷、影响与保留范围

`Learner._view` 复用当前 `TaskContext`，将历史事件按 `str(index)` 注册。`TaskContext.register` 遇已有键就返回旧 result ID。结果是历史题的题面、工具名、参数仍属历史题，工具结果却来自当前题相同索引。[E4]

已确认 367 个错配事件、19 份实际发送的 Builder 材料，按生成源码可关联 9 个 Program。9 个 Program 共 13 次独立 Train trial：5 positive、6 normal、2 execution_failure；不能说它们都已合格。

其中两个 Spreadsheet Program 各有两道不同物理 Train 的正向试用，现为 usable，**这两份资格证据继续有效**。只有 Sheet44 的关联 Program 后来有 6 次在线调用，3 次 not_found、3 次 needs_input，没有已记录的输出消费或终局贡献。证据不支持清空整个 Bank或强制重建全部 Program。

### 4.2 实现修改

- 将 `_view(experience)` 改为显式接收来源身份，例如 `_view(experience, *, source_id)`。
- 修改所有入口：当前学习经验、Builder 固定案例、guidance 经验，以及验证失败的 `completed_train_case` 反馈材料。不能仅修 Builder examples 的一处调用。
- 使用专用键 `learner_case:{physical_case_id}:{events_snapshot_sha256}:{event_index}`。该 hash 对应本次实际引用的不可变事件快照。
- 当前 Runtime 的索引与 result_ref 规则保持原有语义。学习键独立命名空间，同一案例重复显示一致，不同案例/不同快照互不覆盖。
- 若同一完整来源身份出现不同内容，应报宿主数据一致性错误，不返回旧值蒙混过去。
- 在学习材料保留紧凑 case_source 标记；仍使用现有 preview/ref 能力，不增加模型审批或新的资格凭证。

### 4.3 缓存与恢复

`Learner._receive` 先读取 checkpoint 中旧的 `*_request_material`。只改 `_view` 而不处理这里，会继续使用已经缓存的串线材料。

新材料必须带版本，并进入 request semantics。对于尚未发出的待处理材料，在 recovery child 中生成新版本键；已支付的请求、响应和旧材料都保留。不能把旧响应挂到修复后的材料下，也不能通过清空 checkpoint 触发整段重新收费。

建立按 run/material/request/program 索引的定点暴露记录。旧 Bank canonical Train case 未被本次证据证明覆盖，不重建案例池。仍需消费错误材料的未完成 job 换新材料版本或明确 deferred；没有必要立即再生成这 9 个 Program。

## 5. M3：Spreadsheet 评分修复与执行后恢复

### 5.1 真实边界

目标 `spreadsheet:283-32`，seed44，order_index=90，即此前已有 90 道完整提交。answer_position 包含 `Sheet3'!A:G,'Sheet4'!A:G`。评分器试图把空行号传给 int()，触发 ValueError。[E5]

这道题已有 executor_finished、solution.py 和 case1_result.xlsx；已发生 **12 次 HTTP、295,093 tokens**，usage 无缺失。没有保存 score，学习尚未开始。模型输出的 VERIFY_OK 不能代替原生评分。

包内 pending state 主动裁掉了 executor_finished_workspace，**不能从裁剪视图缺键推断原 checkpoint 没有 receipt**。恢复时从原运行目录读取完整 state/manifest 并校验即可，不需要用户再上传一整包。

### 5.2 评分器修改合同

修改 `harness/scorers/spreadsheet.py` 的范围解析、单元格枚举与比较入口：

1. 严格识别单格、有界矩形和本次实际出现的整列范围 A:G。可支持既有合法 $ 绝对引用，保留当前数据 sheet-name 外围引号的兼容规则。
2. 整列覆盖 row 1 到 `max(ws_gt.max_row, ws_proc.max_row)`，防止漏掉预测文件额外行。按迭代器比较，不能构造数百万个地址字符串后一次展开。
3. 检查空范围、缺端点、反向范围、非 ASCII 地址字符、Excel 行列边界；非法评分元数据产生结构化 evaluator/data 合同错误。
4. 已有纯函数检查发现：空 answer_position 当前可能不比较任何单元格直接 True，反向 A2:A1 当前空枚举。这两项纳入同一严格校验；**尚无证据表明本轮真的触发过，不能计成已发生的虚假成功**。
5. 保留现有数值两位舍入、日期/时间、空字符串与 None 的比较规则。预测文件缺失/损坏、缺目标 sheet、合法格值不符按原任务失败合同处理；gold/评分元数据错误单独归类。不能 catch-all 一律记模型答错。
6. 本次不无依据扩展整行 3:9。先离线扫固定范围元数据；只有真实合同要求时才扩展并声明。

本次读取的官方 SpreadsheetBench evaluator 也对整列执行 int('')，并没有现成的上游整列支持可以直接声称继承。本修复应标记新的 scorer 版本，不伪装原评分器完全未变。[官方参考](https://github.com/RUCKBReasoning/SpreadsheetBench/blob/main/evaluation/evaluation.py)，读取到的 blob 为 `d8109e9509f06fc67e9458fbbd4d494e086e48fc`；该参考身份不冒充本实验锁定 revision。

### 5.3 恢复入口与幂等要求

当前 `experiments/run_empirical.py` 在 Executor 恢复前，看见 task_started 中存在写操作就拒绝；`recover_empirical` 的旧分支主要面向已保存 score 的学习失败。两者都不能直接套用到本题。

新增窄边界 `executor_finished` 恢复：

- 从原目录校验完整执行状态、无未确认的 Program/工具副作用、已封存 workspace receipt、文件 hash 和原 task/config/Bank/prefix。
- 只有已确认执行完成且快照完整时，允许 bypass 过早的写事件拒绝并恢复已完成产物。in-flight、未知副作用、缺 receipt 或 hash 不符仍拒绝盲重放。
- 创建带新源码/新 scorer identity 的 recovery child；普通 resume 的身份检查保持严格。
- 复用 execution 和产物，完成实际原生评价，不调用 Runtime/Planner/Extractor/Builder，不重做原求解。
- 第一 case 可复用已生成文件；额外 case 若没有评价 receipt，仍可能需要既有隔离 scorer Worker 执行保存的解法或 recalc。**零 LLM 不等于不执行任何原生评分操作。** 本包未提供完整 evaluator case 数，不能宣称只读即可完成所有 case。
- 每 case 评价 receipt 绑定真实 case_id/index、input/gold/prediction 内容 hash 与 scorer version，防止不同 case 误复用；全任务评分必须由本次真实 cases 列表对应的完整 receipt 集合聚合。持久化 raw score 后，经 M4 共用完成边界保存完整 trace、原 execution/tools/request/usage 身份，以及 child 中有效的 learning_workspace receipt；可显式复用/映射已经验证的 executor workspace，但路径必须在 child 可解析且 hash 通过。然后才持久化 task_execution_finished。当前后续学习恢复会检查 learning_workspace，不能省掉这一步。推荐在此停止，明确标记学习 pending/deferred；以后从学习边界正常继续。
- 没有完成真实学习状态处理前，不能伪造 learning_status=completed 或宣称已有 91 道完整提交。
- 幂等身份包括原 checkpoint、任务/attempt、执行/workspace manifest、scorer version。重复恢复返回已有 receipt；在评分后、提交前中断，应继续提交已有评分，不能再执行评分或重计请求。
- 原 12 次请求与 295,093 tokens 继承一次，90 道历史完整提交保持原样。本次未判断第 91 题最终对错。

## 6. M4：预算耗尽后的评分与学习共用完成边界

Sheet 已完成 391 题中有 **80 道 token_budget_exhausted、learning_status=not_started**，其中 41 正确、39 错误，整题链路成本为 53,370,479 tokens，占 Sheet Train 35.39%。文件型任务可以依赖已生成产物得分，因此预算耗尽不等于产物必错。[E6]

`EmpiricalSystem.run_task` 普通结束会保存 score 和 task_execution_finished，再调用 learn_trace；BudgetExhausted 分支却在评价后直接把 checkpoint 推到 learning_finished，未执行学习。此状态不应被视为正常走完同一条训练链。

最小修改为共用一个完成函数，负责：

1. 固定实际 execution、partial attempts、工具事件和 workspace。
2. submit/evaluate，保存真实 score 与 episode result。
3. 持久化 task_execution_finished 及学习 workspace receipt。
4. 对 learn=True 的 Train 进入原独立学习预算；Frozen 不学习。
5. 学习成功、无变更、策略跳过或失败均记录明确状态，才进入对应结束边界。学习异常不得改写已保存 solve 分数。

普通角色预算耗尽与全诊断预算耗尽必须区分。`diagnostic_budget_exhausted` 仍立即硬停，不能借“补学习”绕过整批限额；共享 Extractor/Builder 的学习预算仍共享，换 purpose 不重置。

若没有足够学习预算，保存明确的 budget-unavailable/deferred 状态，不能把 not_started 当 completed。**这项修复可能增加未来学习费用，它不是单独的降本措施。** 历史 80 题不自动追补学习，也不按新版本回写旧结果。

## 7. M5：先修实际协议冲突，再减少重复输入

### 7.1 成本分解：降本必须对准实际用途

全量 18,297 个唯一物理 HTTP 去除继承重复后，已知总量为 342,112,940 tokens；Train 为 13,143 次、312,751,102 tokens。435 条 copied_to 引用没有再收费计数。[E2]

| Train 用途 | HTTP | tokens | 占 Train |
|---|---:|---:|---:|
| 主 Runtime | 9,165 | 191,351,500 | 61.18% |
| Extractor | 2,114 | 66,232,211 | 21.18% |
| Builder | 657 | 28,935,045 | 9.25% |
| Planner | 1,192 | 25,897,581 | 8.28% |
| trial 模型续解 | 15 | 334,765 | 0.11% |

Office 完成题整链均值约 485,461 tokens，Sheet 约 384,160；不是只把挂起任务平均到完成题后的假象。5 个挂起 Train 题共 1,587,536，仅占 Train 0.51%。

补充指标均为前述总量的子集，不能再次相加：

- Train 结构 repair：1,688 次、39,668,828 tokens，占 12.68%。
- HTTP retry：3 次、41,240 tokens，不是大头。
- Builder 第二次生成：183 次、10,488,447 tokens；其中 140 次源于 length。恢复后 tool_calls 不等于实际合格，也不等于全部无效。
- trial 续解只来自 Office 的一个 trial；Sheet 为 0。纯 Worker/评分的 CPU 与 I/O 不计成模型 tokens。

已知费用不等于货币账单：cached 已包含在 input，reasoning 已包含在 output；8 次 usage 缺失（Train 7 次）仍未知。Train repair 的 input 大部分命中缓存，不能把减少某一 token 比例直接换算成同幅度货币节省。

### 7.2 当前实际配置

28 份 LLM 配置一致，实际请求与其相符：

| 内容 | 本轮配置/观测 |
|---|---|
| 模型 | 请求 deepseek-v4-flash，成功响应记录 deepseek-flash |
| 协议 | deepseek_v4_chat，thinking enabled，reasoning_effort high |
| completion 上限 | Runtime/Planner/Builder 32,768；Extractor 131,072 |
| Builder 截断恢复 | 65,536，仍受共享学习剩余额度约束 |
| 累计预算 | Runtime 600,000；Planner 120,000；Extractor+Builder 共享 262,144/学习 scope |
| purpose override | 当前没有；Office finish_only 仍用高预算思考配置 |
| 温度、top_p、生成 seed | 实际未发送，不推断服务端默认值；运行 seed 不是模型生成 seed |
| HTTP 重试 | max_retries=4，402 不重试 |
| 结构 repair | 主要角色最多 1 次；LiveMath single-answer 不二次求解 |

本轮先不提高 Runtime 上限，也不同时搜索 solver 的多组参数。

### 7.3 响应协议的最小修复

`empirical/prompts.py` 的 Planner 提示说 current_tools 可现在调用，实际原生工具却只有 submit_plan。样本 03 直接 glob/grep，样本 12 直接 execute_python，随后付费 repair。Runtime 对当前工具、batch 和 runtime_step 的描述也含混，样本 02/10 把 action 直接写成 read_result。[E7]

明确修改：

- Planner：当前响应只提交一个 submit_plan；current_tools 是未来执行计划时的环境能力描述。
- Runtime：单次或批量都使用 runtime_step ToolCall；批量是多个 runtime_step，不是直接调用环境函数。
- read_result 的 runtime_step 参数示例为：
  `{"action":"call_tool","name":"read_result","arguments":{"result_id":"已存在的ID","path":[]}}`。
  示例中的 ID 在提示中明确是占位，不是可发明的真实引用。
- REF 示例使用 `{"task":"字段名"}`、`{"from":"节点ID","field":"输出字段"}`、`{"literal":值}` 或 `{"unresolved":"原因"}`；禁止裸字符串让模型猜解释。
- guidance_skill 内仅 goal/guidance；existing_skill_id 在外层，且必须是实际已有 ID。不可把整条 Bank Skill 连同 execution_intent 原样复制进提交合同。
- 新 prompt 示例先通过实际 schema 和 validator；正式最多 1 次 repair 的机制保留。不要以关掉全部 repair 掩盖格式问题。

本版先不加入猜测性通用 JSON 修复器。若本地实现确定性规范化，只接受语义唯一且无冲突的包装形式，并完整保留原响应和转换记录；未知字段/工具、冲突参数、裸 REF 以及普通坏 JSON仍按现有失败合同处理。

24 个样本中只有 10 个提供完整 repair 上下文，足够证明上述冲突，但不足以给 1,751 次 repair 做完整原因分布。原机器可离线扫描已有 repair 消息中的 tool error、user 修复反馈及原 request.error，派生原因分类；不假定日志已提供统一 validation_error 字段。不需要重发请求，也不是等待所有分类完成才允许修已确认错误。

### 7.4 同信息去重与失败视图

Runtime 中位样本中，Office calls 的 34,180 字符有 30,740 来自 Program 卡片；Sheet 的 28,624 中有 26,719 来自 Program 卡片。候选 Program 完整合同是重要输入负担，单纯压 native tools 名称不能解决。

4 个 Extractor 原样本合计 32 个 related candidate，全部存在 current_program 的 schema/result_role/entry_constraints 与外层 Skill 重复，以及 current_job 在 pending 中再次出现。仅这些重复值占整份 user 材料的约 **17%–21% UTF-8 bytes**。[E7]

实现要求：

- 在同一请求里用合同表、job 表只序列化一次相同内容，以明确引用表示所属对象。
- 保持候选集合、顺序、字段值、权限和资格，反向展开后应与原有信息一致。不同来源的同索引结果不是可合并的重复，先完成 M2。
- Learner/Builder 只能原生提交学习/源码，不能调用 read_result；因此必要证据仍在本次请求内可读，不把它们替换为不可访问的外部指针。
- 使用新的 model-view/材料版本和简短引用解释，进入实际请求身份。同步更新 empirical.__init__ 中的 IMPLEMENTATION_REVISION、POLICY_DEFAULTS 及 validate_config 的身份校验；当前这些值按严格相等检查，不能只改 YAML。现有 provider 需要的私有推理重放保持其协议，不拿删除它当无损压缩。
- 对 Runtime 只合并真实相同内容；不暗中把候选改为 usable-only、不降低 top-k。那会改变在线探索，不能装作无损去重。

Builder 最大 Office 样本输入 122,101 tokens。一次工具预算耗尽被包装成 previous_failure 中的 24 条完整工具历史，紧凑 UTF-8 的 tools 约 246.5KB。修 `Learner.builder_material` 和构造 previous_failure 的入口：

- 源码只给一份完整待修版本。
- 失败视图保留 program/version、case/binding、错误类型/阶段、工具预算、失败调用和必要的有界结果。
- 循环类问题保留操作/参数序列摘要和关键状态，不每次重放每条巨型结果正文。
- 完整 trial 仍保存在审计记录，不能删除；材料中必要的错误证据不能藏在该角色无法读取的位置。

**取消此前统一“至少压缩 25%”的验收目标。** 新证据只量出上述重复值占比，实际 wire/token 降幅还要扣除引用开销，并受其他消息影响。验收以信息可还原、修复样本结构正确、真实字节数下降为准；不为追一个人为百分比反复追加优化与测试。

## 8. M6：有证据才提取短 guidance

### 8.1 已确认的学习问题

LiveMath 三 seed 合计：

| 分类 | Train 180 | Val 51 | Test 300 |
|---|---:|---:|---:|
| 正确 | 55 | 11 | 64 |
| 合法标签但错误 | 104 | 33 | 188 |
| 空正文且 length | 21 | 7 | 48 |
| 格式拒收/其他空答 | 0 | 0 | 0 |

Test 为 64/300=21.33%。即使假设 48 个空答全部变正确，也只有 37.33%；这是上界说明，不是可实现改进预测。不能把主要问题简化成“再给更多输出 tokens”。[E3]

180 次 Train 学习全是 upsert，176 新建、4 修订，no_change/reuse/rejected 均为 0。125 个错误/空答来源也产生了可检索规则，占 Extractor 历史费用 4,188,806 tokens，即 85.19%。

两个完整 Train 请求已证明问题：

- seed43、202601:38：原答案 B 被判错；后续 guidance 却宣称应排除 meta-option，而该题正确 A 正是 meta-option。
- seed42、202511:25：原答案 E 被判错；提取器把 A 写成陷阱，建议 C 类结论，随后结构 repair 仅改字段位置，错误知识仍入库。

系统没有把 gold 传反；问题是只有“当前答错”的反馈时，允许模型再猜一个正确结论，并持久化为指导。注入记录证明这些规则后来被使用，但不能直接量化它们造成了多少错误。

### 8.2 host 证据分流

在 `Learner._learn_guidance` 调用 `_receive` 之前增加纯函数 `_guidance_evidence`，并让 `_experience` 保留已存在的 answer_status、empty_answer、completion_truncated、provider_finish_reason。

本版明确处理当前 LiveMath 这种“只交选择标签、没有公开解题过程”的 single_answer：

| 可用证据 | 处理 |
|---|---|
| 空答或 length 截断 | host no_change，reason=insufficient_reusable_evidence，0 Extractor |
| 合法错标签，只有标量失败反馈 | 同上；不能凭“B错”自行宣布“A错/C对” |
| 完整正确提交 | 可进入原提取入口；仍是 advisory，不能标为已验证数学证明 |
| 有真实公开且可核对的失败过程/纠错依据 | 保留学习入口，不一刀切禁止失败经验学习 |
| Frozen/Val/Test | 不持久学习 |

正确性读取当前实际 `score["hard"] is True`，不能用非空 score dict 的真值判断；不把 soft 分数或任意非空反馈当成功证据。证据类型、来源 task 和创建顺序由 host 写入，不由模型自证。

保留 Train case、原分数、学习日志和 checkpoint；host 跳过不是“模型返回 no_change”，两者记录来源不同。历史错误 guidance 原字节保留在旧 Bank；新策略不把它们自动导入新诊断/新协议 Bank。

此规则以证据形态为边界，不扩大成“所有失败工具任务都不能学”。本包对 LiveMath 的影响已量出；对其他回答形态不能机械套用它的 125 条和 85.19% 比例。

### 8.3 短 guidance 的单一候选配置与实现接线

新增用途 `decision_purpose="guidance_learning"`，保持角色仍为 extractor，费用仍进 extractor_e1，且和 Builder 共用原学习 budget_scope。供本次有限验证的唯一配置为：

```yaml
llm:
  purpose_overrides:
    guidance_learning:
      protocol:
        thinking_type: disabled
      max_completion_tokens: 4096
```

这不是现有 YAML 已可直接生效的字段：`validate_config` 目前只允许 finish_only，必须加上 guidance_learning 的允许项，并连通 `_learn_guidance → _receive → system.agent → resolve_call_settings → provider`。现有 provider 仍会发送继承的 reasoning_effort 字段；审计应如实记录，不能把 disabled 之外没有改的字段报告为另一个配置。

同时实现：

- `_receive` 的语义 digest 白名单加 decision_purpose 和新学习策略/材料版本；旧响应不能跨 purpose/政策复用。
- GUIDANCE_LEARNING 的 string 合同：goal 最长 256 字符，guidance 最长 1,600 字符，rationale 最长 512 字符；非空要求继续生效。
- 内容要求是适用条件、可执行思路和局限，不是题目答案查表或没有推导依据的结论。单次成功标签不能证明生成 guidance 的所有数学断言。
- 超长或非法输出不静默截断后入库；正式最多一次已有结构 repair 或明确 rejected/no_change。诊断批将此 repair 设为 0，失败也记入槽位。
- 将 repair 上限从 _learn_guidance 的写死值显式传入可审计的学习配置/诊断覆盖；不能误以为仅改 planning.structural_repair_limit 就关闭了 guidance repair。
- 若 provider 不接受该配置、输出截断或合同失败，按批次规则记录并停止对应链，不自动试 8K/16K 或切回其他思考档位。

LiveMath Runtime 的思考模式、32,768 上限和单次作答协议保持本轮参数。短 guidance 配置先检验学习成本与合同，不能宣称解决了 188 个合法错答。

### 8.4 检索适用性暂不叠加新算法

当前 Bank 使用 goal+guidance 的词重合，长规则容易取得重合；两个原 Runtime 样本注入了明显不同数学领域的长 guidance。立即记录匹配词、排序、注入字符数与来源证据即可。

本次不同时换 embedding、选择新相似度阈值、扩大 solver 上限。这样第 10 节的小配对还能解释它在测什么。若后续检索仍无覆盖或产生负效果，记录为方法结果；不靠继续加题让结论变好。

## 9. M7：等两个合法案例后才付首次 Builder

现有 Office/Sheet 合计 461 个 Program：455 candidate、4 usable、2 disabled。401 个 job 已有 Program、只绑定 1 个案例，仍处于 waiting_example。最终仍单 binding 的 job 对应 Builder 历史费用 **27,692,423 tokens，占全部 Builder 95.70%**。[E6]

这说明主要结构是“很早付钱构建，后面没有凑齐第二例”，不是普遍多 epoch 无限再生。新策略可减少提前支出，但 27.69M 是历史归因量，不能直接当作反事实节省承诺。

### 9.1 成对修改两个接点

1. 在 `_merge_request` 和 `_realize` 中，首次尚无 Program 的 job 只有 0/1 个合法独立 Train binding 时保持 waiting_example，不调用 Builder。已有两个不同物理案例、输入与 reset/prefix 合同都有效时才 ready。
2. 同时修改 `related_candidates`：目前优先项要求“已有 candidate Program 且恰一 positive”。延后首次构建后，没有 Program 的单例 job 必须以明确“未生成、未验证”的状态进入补例候选队列，否则可能永远拿不到第二例。

第二案例的适用性仍由原公开经验、输入 schema、真实 binding 与 prefix/reset 校验决定。不能按标题相似自动合并 Skill，也不能随机凑第二例。

### 9.2 行为约束

- 每个 Train 保持原单次学习决策、至多一个 realization；不新增专门收费的“寻找第二例”阶段。
- 已有单案例 Program 如果获得第二例，优先做该版本真实 trial，不先重建源码。
- 两个已绑定案例不是两个 positive。候选仍需两道不同物理 Train 的真实正向试用，才能按原规则取得自动接管资格。
- 同一物理题重复调用、needs_input/not_found 或未满足结果合同的运行，不能制造新的独立正向资格。零 native tool calls 本身既不证明通过，也不单独构成不合格：纯计算 Program 只要输出符合既有合同，并在两个不同物理 Train 上取得真实 positive，仍按原规则处理。
- 没有合适第二例就继续等，并报告覆盖不足；不能设“每批必须生成一个 usable”的成绩门槛。
- 新源码形成新 Program 版本，旧 trial 不能无条件转给新源码。

将首次构建阈值记录为显式学习配置（本版 min_distinct_train_cases_before_first_build=2），在 validate_config 校验合法值，并让 _merge_request 与 _realize 读取同一配置，进入 implementation identity。此调度会改变在线探索和 Bank 演化，必须作为方法版本写清，不能称其为完全无行为变化的缓存优化。

### 9.3 目前方法证据应该怎样报告

已记录在线 Program 调用 26 次，全部在 Train：ok 9、needs_input 6、execution_error 4、not_found 4、blocked 3。26 次的 outputs_consumed、submission_by_program、terminal_by_program 均为 false。[E6]

因此当前证据说明“只有 4 个 usable，且尚无已记录的可归因输出复用或终局贡献”。不能把调用数当作方法有效性，也不能据此断言未被指标覆盖的工具副作用贡献绝不存在。新的调度需要观察补例与实际消费，不能只看 Program 数量涨了多少。

## 10. 验证计划：一轮离线验收，加一批有硬上限的真实诊断

本计划不重新运行原固定 12+6，不开多 seed 正式矩阵查 bug。验证目的和方法成绩分开：工程项要满足确定合同；小配对只给局部机制证据，不要求必须涨分。

### 10.1 零 LLM 的定点验收

| 修改 | 固定验证规模/材料 | 应看到的结果 |
|---|---|---|
| M1 选项协议 | 177 个现有物理题；3 个运行 seed 的物化映射 | 候选文本与正确内容不变；公私投影一致；确定且幂等；split 不变 |
| M2 来源隔离 | 包内 6 个完整错配例＋当前/两历史案例同索引、同案例不同快照、缓存恢复 fixture | 每个结果属于自己的真实案例；不复用错来源旧材料；同身份冲突显式失败 |
| M3 范围合同 | 单格、矩形、A:G、预测额外行、多 sheet 引号、空/反向/越界/缺端点 | 合法范围得到真实比较；非法元数据不空迭代成功；值比较规则不回归 |
| M3 恢复 | 已完成快照/坏 hash/in-flight/评分后中断等惰性 fixture；原第91题执行边界一次校验 | 不调用求解模型，不重放未知副作用；已评价 case 幂等；学习停在真实状态 |
| M4 预算边界 | 成功/失败产物×普通角色预算终止；再加 Frozen、整批诊断预算、学习失败 | 分数只记一次；应学习的 Train 可进入独立预算；诊断总限额不被绕过 |
| M5 协议与材料 | 已有 10 条 repair 完整样本；4 个重复材料样本；最大 Builder failure 样本 | 新示例通过真实合同；去重可反向还原；失败证据仍可读；记录实际字节变化 |
| M6 学习政策 | 空 length、错误标签、正确标签、可检查失败过程、旧缓存/重启、Frozen | 证据不足 0 Extractor；正确入口可达；专用 purpose/cap 真正进入 payload 和身份 |
| M7 调度 | 0/1/2 个 binding、同物理重复、已有 candidate 补第二例、不适用/失败/两 positive | 首次构建条件和补例队列一致；原正向晋升与接管规则保持 |

这是一组与实际变更对应的回归，不扩成新的全项目验收框架。已运行过且与修改无关的测试不反复跑。若 fixture 失败，先修对应代码；不要去真实环境碰碰运气。

整列恢复的实际评分需从原完整目录校验后做一次。若涉及额外 case 的原生隔离评价，应明确记录这些操作；本审查未执行它们。

### 10.2 一次最多 8 个物理 HTTP 的 LiveMath 局部诊断

只读 seed42 的 60 个公开 Train 题面后，没有为了凑数扩充成 4 对。固定以下 **2 个来源 Train＋2 个不同物理迁移 Train**；不读取或依据历史/本批 score 来选择，不用 Val/Test。

| 对照组 | 来源 Train（原序号） | 迁移 Train（原序号） | 公开主题与边界 |
|---|---|---|---|
| H | `livemath:202602:28`（16） | `livemath:202602:29`（19） | 闭双曲三维流形、体积/闭测地线长度、统一常数的几何界；曲线复形距离的结论不能照搬为钻孔体积公式 |
| F | `livemath:202602:38`（30） | `livemath:202511:15`（44） | 分数阶非局部正则性、远场尾项、尺度与指数；Heisenberg 群内点估计不能照搬为欧氏边界估计 |

原序号均为 1-based；精确 physical hashes 随证据 JSON 固定，启动时核对 task_id 与物理 hash。它们是有明确适用边界的主题候选，不是已经验证能迁移的规则。

固定顺序为来源16→迁移19配对→来源30→迁移44配对。H 只能使用来源16学习后冻结的快照；F 使用来源30学习后冻结的快照，不能把最终快照倒灌到 H。迁移题不学习，不充当下一组来源。

| 步骤 | 数量上限 | 物理 HTTP 上限 |
|---|---:|---:|
| 从空诊断 Bank 真实求解来源题 | 2 个 Train | 2 |
| 按 M6 判断，有证据才提取短 guidance | 每来源题最多 1 次 | 2 |
| 在各自时点冻结 Bank，对迁移题做 guidance on/off | 2 题×2 臂 | 4 |
| 合计 | 4 个物理 Train 题 | **8** |

整批共用现有 BudgetGovernor：

```python
BudgetGovernor(
    diagnostic_ledger,
    token_limit=500_000,
    finish_reserve=0,
    request_limit=8,
)
```

这是诊断器构造参数，不是给正式配置随意增加顶层键。所有调用逐次以实际 payload 的保守输入上界＋completion cap 预留；usage 缺失保留未知计费并停批。上述 8 是物理 HTTP，不能把 retry 记成额外免费调用。

诊断约束：

- 单个运行 seed=42；新 choices 使用 M1 固定映射。初始 Bank 为空，禁止导入旧 final Bank/旧 guidance。
- 来源题使用当前 Runtime 单次解答配置；按新学习策略最多一次提取，Extractor 使用 disabled/4096。诊断 max_retries=0、guidance repair=0，失败也占名额。
- 每个来源与迁移题是不同物理任务。迁移题 learn=False、Bank readonly；每对的两臂共用其时点同一冻结快照，off 只通过诊断只读 Bank 视图屏蔽 guidance 注入，不改题目、solver 或评分。
- 两臂共享只读 Bank 快照和整批 BudgetGovernor，但必须使用不同 attempt_id、checkpoint 目录及 requests 日志。诊断身份包含 pair_id、arm、冻结 snapshot digest、projection 和学习策略版本；不得把 on 臂已完成的 task checkpoint 作为 off 臂恢复入口。现有 run_task 对已完成 checkpoint 会在检索 guidance 之前直接返回旧 trace，必须阻止这种缓存串用。
- A=guidance on，B=guidance off；H 调用 AB，F 调用 BA；不把运行 seed 伪称生成 seed。
- 某来源没有生成符合合同的 guidance，其对应迁移对记 skipped，不更换来源/目标；固定的下一来源仍按顺序执行。两来源都未形成 guidance 时，本批只保留已实际发生的来源调用，没有付费迁移对照。不得用空 Bank 的两次相同输入凑出方法对照。
- 采用原检索规则并记录实际注入；不为了本批让 on 臂强制拿到人工挑选的正确规则。
- 批次或槽位到限立即结束，不补抽“容易成功”的题、不做另一轮参数 sweep，不再加回 12+6。

### 10.3 希望看到什么，怎样判定

工程层面应看到：实际 payload 的 purpose/thinking/cap 与新配置一致；不再把无依据失败猜测入库；每项调用、repair、usage 和跳过原因可解释；迁移两臂运行前后 Bank digest 一致。

方法层面报告每个已执行配对的：对→错、错→对、同对、同错，注入了哪些规则、来源是否覆盖、各臂 tokens/length/空答。来源端报告 guidance 生成数、拒绝/跳过数、单次学习成本和正文长度。

**两个配对只能观察局部机制，不能证明总体方法优势，也不设置“必须提升某百分比”的放行门槛。** 没有 guidance、没有覆盖、零收益或负收益都是可接受的诊断结论，应结束该固定批并记录具体原因。只有明确的新工程失败需要对应代码修复；不自动追加收费实验来获得正结果。

M7 的两例调度用历史 job 状态和有限状态转移验收确认可执行性；不能用 LiveMath 这批 guidance 对照冒充 Program 方法收益。Program 当前复用不足已由 E6 给出，不另开大批 Office/Sheet 来“先看看”。

## 11. 实施顺序、历史边界与实验启动条件

### 11.1 建议提交顺序

1. **正确性修复提交**：M2 来源与缓存版本，M3 范围及窄恢复，M4 统一完成边界；配对应的离线回归。
2. **输入协议提交**：M1 固定 choices 投影与全方法输入一致性；M5 响应 prompt 和同信息材料投影，记录各自版本。
3. **学习配置提交**：M6 证据分流/用途/短合同，M7 两例后首次 Builder＋配套队列，明确方法变化。
4. 固定诊断版本和选择，按第 10 节仅运行一批。将失败归因于实际边界后结束，不展开参数搜索。

可以在一个开发分支完成这三组提交，不要求每组都跑一次付费小实验。

### 11.2 历史运行处理表

| 历史对象 | 处理 |
|---|---|
| 已完成 SearchQA 及其他已完成记录 | 保留原始成绩、费用与版本，不因共享代码将要变化而自动重跑 |
| LiveMath 旧投影结果 | 标为旧协议；新协议正式结果使用新生成轨迹和新 Bank |
| 9 个来源材料受影响 Program | 定点标记；保留真实独立试用，尤其两个 usable 的资格，不清空全 Bank |
| Sheet44 第91题 | 恢复 executor_finished 后实际评分；已有12请求继承一次；学习保持真实 pending 边界 |
| Sheet 80 道预算后未学题 | 保留历史事实，不自动追补成新策略训练结果 |
| 新 prompt、学习策略、首次 Builder 调度 | 新 identity；不能在旧 run 中伪装成未变配置续写并合并为同一个干净正式方法 |
| 402 中断项 | 保留已收费证据，从已确认边界处理；不为重现402额外探测 |

5 条 402 均无重试、正文未保留，只能确认 provider 拒绝，不能凭状态和 body hash确定余额或并发原因。可以在 provider HTTP 错误分支补受限、去凭据的 error.code/type/message 与截断标记，继续保存字节数/hash；不重造旧正文，也不要求再上传无法恢复的正文。

### 11.3 是否可以启动大规模并行

**当前还不能按原版本扩大运行。** 下一步是落地上述已经定位的修改、完成相关离线验收，并执行一次有上限的诊断；不是先开多 benchmark × 3 seed 再靠日志找问题。

工程准备通过的含义是：已确认缺陷有可验证的补丁；状态/计费/恢复一致；新数据和学习协议身份明确。小配对没有正增益，不等于必须继续补测；它限制的是“已经证明方法有效”的表述。

后续若启动正式比较，应提前锁定新协议/学习配置和所有方法共享的公开输入、求解配置及评分合同。正式评估可以检验方法优劣；工程缺陷不应继续依赖正式评估来发现。Ours 包能定位本文件问题，不能替代 baseline 自身配置是否一致的核验，但当前不再向用户泛化索取额外审查包。

## 12. 本次审查实际完成与交付

已完成：包和源码完整性校验；费用账本与用途分桶独立重算；学习/job/试用统计交叉核对；原请求和代码控制流对应；来源碰撞和范围的纯函数复现；177题固定上游 shuffle 算法检查。

未声称完成：生产补丁、原机器恢复、真实评分、真实模型诊断或新正式实验。

配套 JSON 收录证据、统计、纯函数检查结果和其适用限度。本文的实现规格足以交付本地修改；后续审查只需要对照实际补丁和这组有限验收的结果，不需要再次从头构造完整大审查包。
