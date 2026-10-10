# AtomicSkill 多 benchmark、多 model 统一方法适配修订实现文档

**版本：2026-10-10，统一方法适配修订。状态：代码审查与待实现规范；本次没有修改生产代码、启动模型请求或重启正式实验。**

本次核对的 GitHub main 为 `993d6cd0c21a297e55343aab4a65009494f0df16`，其父提交为 `0099b4a44f415a8e7e418a003ef36d95028ed9b1`。后一个提交包含被审查的运行实现，前一个提交新增 LM1 报告。本文件中的“当前行为”指这个已固定的源码版本，“修改要求”指下一版需要实现的行为；不能将本文作为修复已经合入的证明。

## 1. 本次结论与覆盖范围

需要解决用户指出的问题。当前 SearchQA、LiveMath、DocVQA 的接入将“最终只提交一个答案”扩展成了“不能学习、试用或消费 Program”。同时，其他接入将第一次构建和可用资格都绑定到两个不同物理 Train 来源。两项限制共同造成部分 benchmark 实际运行文本 guidance 方法，部分 benchmark 即使有局部程序也很难进入冻结评估。这与本次明确的统一 Atomic 方法目标不一致。

修订目标是：公开任务输入进入适配器，真实执行产生 trace，从其中学习可复用的原子操作；适合程序化的操作经真实局部验收后成为 Program，其他经验保留为短 guidance，已验收程序可直接执行或组合；Train 后冻结，在相同公开输入和求解权限下测准确率、token、调用与训练摊销成本。不能先按 benchmark 名称决定某项实验永远没有 Program，也不能要求每道题必须生成程序。

**本次取消通用的“两来源准入”要求。一条真实局部经验即可首次构建；通过独立参照的局部验收即可成为 usable。这里“独立参照”指参照不是待验收程序自己生成的答案，不是要求再找一道独立 benchmark 题。跨来源成功数量继续作为复用和泛化的统计量，不再决定能否构建、晋升或进入 Frozen。**

本文替换 CF4-R1 中 single_answer 禁止 Builder/Program 的规定，替换当前通用两来源构建和晋升规定，并补充成本与多模型配置的实现要求。LM1 的公开输入隔离、选项投影、数学 guidance 来源检查、冻结只读、真实计费等已正确约束继续保留；不能为恢复 Program 而撤销这些约束。[S1][S2][S3]

上一轮 LM1 文档解决的是 guidance 可靠性、有限费用验证及少量异常处理，并没有恢复上述完整主链。其“只剩两个小修即可完成 LM1 收尾”的结论不能外推为“多 benchmark 完整 Atomic 方法已经可正式运行”。本修订对统一方法范围作出更正。

## 2. 代码审查发现：真正的问题是什么

| 项目 | 当前代码事实 | 实际影响 | 必须修改 |
| --- | --- | --- | --- |
| 单答题入口 | `system.run_task` 的 single_answer 分支一次无工具回答，`attempts=[]`；`Learner.learn` 保存经验后提前返回 guidance 学习 | SearchQA、LiveMath、DocVQA 不执行完整 Program 链 | 增加轻量答题执行路径，学习按真实操作证据选择产物 |
| 首次构建 | `POLICY_DEFAULTS`、`validate_config`、`_merge_request`、`_realize` 均落实两物理 Train 来源 | 不是改一个 YAML 数字即可解决；很多局部操作连第一次 Builder 都等不到 | 全部改为一个有效源绑定即可启动 |
| 晋升与冻结 | `Bank.record` 要两个不同 task_key 正例才 usable；`freeze` 删除其余 Program、implementation | 一源局部合格程序不能在 Test 复用 | 用版本级局部验收决定 usable；Frozen 仍只保留 usable |
| 局部试用 | 已支持 local_check，但 Answer/File/Office/Spreadsheet 没有实现；ALFWorld 仅有窄范围 TAKE/库存检查 | 大部分 intermediate 程序回到付费整题 continuation | 增加源局部重放与参照检查，默认不为局部验收重解整题 |
| 程序输出消费 | dynamic 节点的 required_fields 可以为空，显式 call_program 后的子集判断可能错误 complete 整题节点 | 只删除 single_answer 早返回，会把中间计算误当完成，甚至空答退出 | 按绑定节点合同和 result_role 完成；未绑定整题节点的 intermediate 只返回结果 |
| 学习材料 | `_experience` 保存 Program attempts，未完整纳入 `trace.result_store` 的实际结果 | 纯 Python 无 native events 时，Extractor 可能仍看不到有效局部过程 | 用真实结果引用补齐局部代码、输入、输出和消费关系 |
| 成本边界 | canonical runner 未传入全批 Governor；trial 使用独立 scope，可能获得另一份 Runtime 额度 | 原题预算不等于“原题加学习”的实际总开销上限 | 父任务与全批统一准入，所有角色、trial、retry 计入同一账本 |
| 多模型设置 | formal model 多数仍未配置；部分 reasoning/protocol 设置未实际生效；snapshot 的 token 字段可与请求不一致 | 不能认为替换模型名就完成了多模型适配 | 显式 provider 能力配置、有效设置合并、实际 payload/usage 回放与接入验证 |

源码位置详见第 12 节。以上是工程入口和策略实现问题，不证明 Atomic 方法无效，也不证明恢复入口后一定涨分。特别是 LiveMath，程序适合确定性计算、有限检查、参数化转换等局部工作，不能自动补足困难数学判断本身。[S2–S8]

需要避免另一个错误归因：**OfficeQA、SpreadsheetBench 等完整 Learner 并未在 host 侧丢弃所有整题失败经验。** `learn_trace` 会将它们送入 Learner；已有 local_check 正例也不要求整题成功。真正要补的是可用的局部证据与验收路径，而不是再增加一个“失败题也进入 Extractor”的重复分支。[S3][S5]

## 3. 统一方法合同与 benchmark 适配

### 3.1 分离最终提交、公开资源与内部计算

`single_answer` 只决定最终如何提交。公开输入、可调用的原生工具、文件/图片映射、状态恢复能力、内部局部计算能力分别表达。修改 `Capabilities` 时保留原字段含义，新增内部执行及公开资源声明；配置和运行身份中记录其版本。

原生工具权限必须来自 Adapter。内部 Program 使用同一个锁定 Python Worker；无原生工具的任务可以 `allowed_tools=[]` 做纯 JSON 计算。已有 `PROGRAM_FORBIDDEN_TOOLS={'execute_python'}` 禁止的是 Program 递归 RPC 调用另一个 Python 执行工具，不是禁止 Program 自己执行 Python。

临时代码执行与已学习 Program 复用分别记录。NoSkill/对照也应具备相同的公共内部计算能力、数据权限、CPU/内存/调用额度；Atomic 的增量是学习到的代码、接口、适用条件和组合。不能用新增网络、隐藏数据或对照无法使用的计算能力制造方法收益。[S6][S9]

### 3.2 六项实际适配范围

| Benchmark | 必须保留的公开输入与最终合同 | Program 可以承担的局部操作示例 | 本次适配要求 |
| --- | --- | --- | --- |
| SearchQA | 问题、当前固定的给定 context；单一答案文本；保留 EM/F1 评分 | 给定文本内的候选抽取、原文位置、去重、日期/数字规范化和明确计算 | 打通纯本地程序和真实结果消费；不给联网搜索，也不恢复未公开的截断前材料 |
| LiveMath | 公开问题、完整固定投影后的 choices；最终选项标签 | 精确算术、有限枚举、明确表达式检查、局部结果到本题候选文本/标签的绑定 | 中间计算回到 Runtime 判断；禁止把有限数值例子当普遍定理证明或缓存源题字母 |
| DocVQA | 原问题与真实图片；既有 ANLS 合同 | 对已观察字段计算/规范化；授权原图的裁剪、旋转等图像处理 | 模型与 Learner 需要图像证据时真实传图；只读图片映射到 Worker；文本模型仍 unsupported |
| OfficeQA | 全部授权离线 corpus；文本答案 | 公共 glob/grep/read 的局部组合、读到的字段/表格/数值处理 | 不添加 source_docs/oracle；程序用声明的读取 RPC，局部重放避免整题续跑 |
| SpreadsheetBench | 首变体 workbook、指令和 answer range；既有 solution/result 文件合同与全部变体评分 | 参数化表格定位、范围处理、计算、格式保持、文件发布 | 用真实输入文件/变更范围验收；隐藏变体不成为 Runtime 输入或“多个来源” |
| ALFWorld | 当前公开观察和真实 action catalog；环境 won | 在公开状态下执行取物、搬运、条件分支等局部动作序列 | 保留动态动作合法性和真实状态恢复；局部成功可独立于最终 won |

这些是能力范围示例，不是要人工写入 Bank 的 benchmark 解题模板。是否产生这些资产由真实 Train trace 与 Learner 决定。没有可程序化过程时允许 guidance 或 no_change；“本次自然没有产生程序”和“代码禁止产生程序”必须分开报告。

当前锁定镜像没有声明 sympy 或 OCR 引擎；不能在文档或提示中假设已经可用。DocVQA 不得通过 OCR 文本替代原图让纯文本模型伪装成支持视觉。当前 canonical 名单只有上述六项；ScienceWorld 尚未接入这一生产矩阵，本文不静默增加第七项。[S6][S9][S10]

## 4. Program 资格：单源验收，不再等待跨题晋升

### 4.1 保留现有三态，修改 usable 的依据

| 状态 | 新版含义 | Train | Frozen / Val / Test |
| --- | --- | --- | --- |
| candidate | 已生成版本，但未通过局部执行验收 | 主机安排有界隔离试用；不凭模型声称通过 | 不导出、不调用 |
| usable | 源局部验收已通过，代码、接口、权限、环境和适用范围有确定身份 | 可正常调用与组合，继续记录效果 | 只读调用，保留验收记录，不能在线晋升或改写 |
| disabled | 当前版本有已确认执行/合同问题，或被隔离 | 停用；修复得到新版本后重新验收 | 不调用 |

`usable` 表示“具备经实际检查的可执行实现”，不表示“已证明在任意新题上泛化成功”。跨任务复用和收益正是后续实验要测的内容。当前两个整题成功也不是普适正确性证明，没有理由将其当作所有局部工具必须满足的存在条件。

修改 `Bank.record`：保留原始 attempt 和 task_outcome/local_check 统计，但删除 distinct task_key 数量触发晋升的逻辑。增加主机专用的 `record_validation` 或等价入口；只有实际验收过程可写入版本级通过记录并将 candidate 置为 usable。不能接受 Extractor/Builder 在 JSON 中填写 `validated=true` 后直接晋升。

### 4.2 局部证据包：保存真实发生过的工作

在现有 Train case binding 增加 `local_evidence_ref`，由 host 解析为局部证据包。建议新增 `empirical/local_validation.py` 集中管理下面的待实现结构，而不为每个 benchmark 建一套资格状态机。

```python
LocalEvidence = {
    "source_physical_key": "...",       # 真实 completed Train
    "source_trace_sha256": "...",
    "event_ids": [...],                 # 真实执行片段；纯计算可引用结果事件
    "operation_contract": {...},        # 局部目标、输入、输出、前置条件、允许副作用
    "input_bindings": {...},            # 公开任务字段/真实工具结果的引用
    "reference": {...},                 # 原始执行产物或主机可重算的预期
    "workspace_or_state_before": {...},
    "observable_effects_after": {...},
    "environment_identity": {...}
}
```

引用的原始内容必须存在，输入值能从公开材料或真实执行结果追溯。动作前缀继续检查是该任务实际发生的前缀。旧 single_answer 记录如果只有最后字母或答案文本，就没有已经执行过的局部计算，不能补写一段“历史程序轨迹”。

可用参照主要有两类：原始动态代码/原生工具实际执行形成的局部输入输出及公开状态变化；或由独立、确定性的局部检查函数重算出的预期。前者检验编译前后的局部行为一致性，后者可以进一步验证明确的局部语义。两者均不要求第二道题。

参照不能只是待验收 Program 自己打印的“正确”、模型的未执行推理、缺少来源的常量答案、文件存在或 schema 合法。若原始局部策略本身只是未经支持的数学结论，不能将复述该结论叫作数学验证。对已有可执行转换，编译等价验收可以支持经验型工具入库，但应保留其适用条件和参照限度，不将它包装成正确解题策略的证明。

### 4.3 验收流程

一次候选验收执行：检查源码与权限、I/O 声明和局部合同；在源公开输入/起始状态上执行 Worker；按独立参照比较返回值和允许的副作用；检查实际输入参数绑定和输出消费接口；全部通过才写验收记录。

记录至少绑定 `program_id`、源码摘要、I/O 与局部合同摘要、环境/权限身份、源 trace 摘要、参照来源、检查器版本、实际输入输出摘要、通过/失败原因。源码、接口或检查器语义变化后原记录不能沿用。判断函数由主机维护，调用前不让模型修改。

防止“返回源题常量就算复用”的措施应放在参数化和局部回归上：检查声明输入确实绑定到程序需要的值，不允许按 task_id、题目 hash 或正确标签查表；存在可重算的独立参照时，增加一个合法变化输入的本地对照，检查编译版是否与原操作一致。**这个变化输入是零模型局部回归，不是第二物理来源，不另算跨题成功；没有合适变体时也不能再次用它制造一个通用两例门槛。**

对于文件，比较声明的单元格/区域/文件变化和应保持的结构，优先使用现有 workspace stage。若某项检查决定文件能否发布，检查必须在 stage→publish 之间完成。对于环境动作，使用当前公开反馈；不能承诺所有副作用都可回滚，未知副作用继续触发现有停止与恢复处理。

### 4.4 修改 trial 默认行为

`_realize` 首次有一个合法局部绑定即可调用 Builder。每个完成 Train 最多新建一个 Program；同一构建任务的首次版本及后继版本合计最多一次源码修复，创建新版本不重置修复额度。

`test_program` 增加明确的局部验收模式。intermediate 程序源局部验收结束即返回，不再默认构造一个新的 Executor 重解后半题。已有 final_answer/final_files 的封存提交与评分路径保留；确实要测试整个任务程序时应显式声明并计入父任务总预算，不能作为每个局部原子操作的隐形固定开销。本次第 9 节有限验证禁止额外的模型整题 continuation。

`job.done` 仅代表该次构建/验收工作结束。验收成功就是 usable；失败在固定修复上限内处理，之后停止。不再将 done 的候选重新排队，试图找到第二道整题来凑晋升数。

### 4.5 Frozen 与调用共用同一资格

Frozen 继续只导出 usable，无须增加“Frozen candidate 可用”功能。随 Program 导出验证记录和必要的公开局部合同，完整 Train case/jobs 仍不进入 Frozen。调用资格应统一检查：当前只读视图是否可见、状态、版本/环境身份、输入 schema、声明的前置条件与资源权限。

`routes`、`program_options`、`planning_cards` 和直接按 ID 的 dispatch 使用一致判定。若增加 Bank 方法，必须同步修改 `BankView`；其默认委托行为不能让 learned_assets_off 通过新方法访问真实 Bank。更简单的实现是资格函数只接受当前 view 返回的 Program。

Val/Test 调用日志写入实验输出，不能更新 Frozen 的 attempts、资格、检索统计或排序权重；任务内仍可临时屏蔽刚失败的程序，下一题不将它变成一次训练更新。[S3–S5]

## 5. 打通 single_answer 的最小运行闭环

### 5.1 使用轻量答题调度，不前置完整 Planner

新增 `empirical/answer_executor.py`，从原 Executor 抽出或共用 Program invocation、输出合同、结果注册、预算和最终封存辅助函数。两种调度使用同一 Bank、Worker、权限检查和 scorer。

答题过程只需保存公开任务、可用资产卡片、已执行的局部结果、预算及是否最终提交。原本负责解题的 Runtime 可直接 finish，也可请求临时局部计算或调用 Program；调用后真实结果进入同一题的后续 Runtime。没有调用时保留一次回答路径，不新增一次专门的“是否要用程序”模型路由。

对已经有明确公开参数绑定、适用范围匹配的 usable Program，可在相应已绑定局部节点直接执行；不为每项程序再调用一个确认模型。需要语义选择或参数定位时，由本来要解题的 Runtime 完成。仅凭检索相似度不能让程序代替整个最终回答。当前 entry_constraints 是文本，不能假定 host 已能理解并验证任意自然语言条件。可机器检查的条件在程序产生副作用前显式执行；未能确定的语义条件由当前 Runtime 作参数绑定与选择，不加一个独立付费检查模型。

第一版单答调度最多两次局部执行；每题总模型额度与最终提交预留共同约束后续回合，不为每一次局部调用重新发一份解题预算。单答 native_calls 仍为 0；内部计算、Program 调用与公开结果读取有独立计数。具体首轮限额见第 9 节，是待验证配置，不是已证明最优的参数。

### 5.2 三种返回情况必须分清

`intermediate`：注册真实结果和引用，继续解题。未绑定的 dynamic 整题节点不能因为 required_fields 为空或输出字段齐全而 complete。对已经显式绑定的局部 Skill/Program 节点，则按其真正的局部合同完成并传给下游。

`final_answer`：只有声明与当前提交合同匹配、必需 answer 字符串存在、输出检查通过时，才可进入原有封存提交。程序输出最终候选与 Runtime 自己最终 finish 的来源分别记录。不能把未知局部输出改名为 answer 以通过验收。

`final_files`：继续使用已有文件发布、封存、solution 重放和评分合同，不改成文本答案替代。

程序失败、参数不适用、局部检查失败要形成结构化结果；从当前真实状态继续有界求解。确定的输入不匹配不等于程序源码错误，不应因两次正常拒绝禁用版本。未捕获 host 异常和未知副作用也不能被吞成普通模型答错。

### 5.3 真正记录消费，才有资格讨论降成本

Trace 应包含临时代码执行与 Program 调用的不同类型、程序版本、实际参数/引用、结果 ID、输出合同、局部验收状态、native 操作区间、发布记录、Runtime/下游读取或使用、替换/丢弃、最后提交来源和实际成本。

保留 M4 已修正的 consumed/replaced 快照。单纯展示结果或传入一个后来失败的下游调用，不等于得到正面任务贡献；完整题答对且程序输出未使用，也不能算程序帮助答对。使用日志可以证明工作路径，因果收益仍由配对消融估计。

Program 运行不自动降低 token。如果它执行完之后，Runtime 仍将同一长上下文重新读一遍、重写同一段代码，就必须报告没有节省。新路径优先给后续 Runtime 发送简短结果和必要源引用；完整公开输入仍可访问，不能通过静默删除选项、问题条件或原图制造低成本。直接路径不强迫把短题也改造成多轮按需读取。[S2][S5][S6][S9]

## 6. 学习路径与材料：一次决策，面向实际操作

删除 `Learner.learn` 中将 single_answer 永久分流到 guidance 的早返回，以及 `_proposal_skill`、LEARNER_PROMPT 中相同的禁止性默认。

以真实证据选择已有学习协议。没有局部可执行事件的单答 trace，可继续进入轻量 guidance/no_change；存在真实局部执行或具有可独立检查输入输出的操作时，进入通用 `LEARNING` 提案。沿用 `no_change`、`reuse_existing`、`propose_skill_and_program_spec`、`propose_or_revise_workflow`，Program 请求由 `execution_intent=program_requested` 和真实局部绑定表达，不再由 benchmark 类别决定。

主机选择证据路径不增加一个付费 Router；一次任务不无条件执行“通用 Extractor＋guidance Extractor＋grounding”三套提案。对于没有执行轨迹但可验证的公开计算，必须先产生本次真实局部执行结果再据此验收，不能标为旧轨迹复用，也不要求额外重解完整题。

LM1 既有九项 guidance 在固定 Val17 上零曝光，说明那轮没有形成指导方法干预，不能说指导收益已检验或检索已经有效。新日志应分别记录无候选、范围不符、输入未绑定、程序未验收、调用失败、结果未消费，便于用已有记录定位。不能为凑曝光强塞无关数学结论，也不将 guidance 的词项覆盖阈值复制成 Program 的通用准入条件。[S12]

LiveMath 的数学文字 guidance 继续执行 LM1 已有来源和范围检查。不能从一个错误标签或失败分数倒推出替代定理。局部程序的可执行参照不需要整题正确；涉及数学结论的文本资产则不能借“Program 已执行”跳过语义证据要求。

`_experience`、`_view` 需解析 `trace.result_store` 中与选定操作相关的实际输出，形成第 4 节证据包。DocVQA 若需要图片理解支持一个新资产，必须把对应公开图片内容传给该学习角色；图片路径字符串不能当成模型看过图像。模型不支持相应模态时，明确拒绝该类学习请求，不自行转换成另一套数据条件。

控制材料大小应在整个 request 层面实施，而不仅逐字段截断。保留原始完整 trace；投影优先级为局部目标与合同、输入绑定、实际操作及结果、必要上下文、至多两个相关失败。每次请求记录原始/投影字节与事件数量；缺少必要输入时 defer，不能截断后编造证据。

现有 Runtime 已有 preview/read_result，材料也有部分去重，不需要重建另一套全量摘要系统。Learner/Builder 目前不能读取任意历史 Runtime 的 episode-local read_result；第一版直接带齐选定局部片段。不要为了压缩 trace 又付费让模型重述所有历史。[S3][S7][S9]

## 7. 降低 Train 成本：先去除重复工作，再讨论模型降配

### 7.1 当前成本结构与修订目标

默认 Runtime 每个 scope 的累计额度是 600,000 tokens，通用 Extractor 单次 completion cap 为 131,072，Extractor/Builder 使用较大的共享学习额度；Builder 截断恢复 cap 为 65,536。trial 切换 scope 后，原题额度不能约束全部 continuation。因此当前高成本不是只看“Train 题数×一次回答长度”就能解释。[S2][S3][S7]

本次直接减少三类开销：局部验收不再重解整题；单答题使用同一 Runtime 的有界局部操作而不增加完整 Planner；Extractor/Builder 只看相关操作片段并有明确输出上限。已有 Program 对重复操作的替代也应减少之后的动态代码生成和重复模型判断。

不能用禁建 Program、对某类 benchmark 只留 guidance、全部关闭推理或削掉关键公开输入来达成成本目标。LM1 小批中 low 的 token 较低但答对数也更低，不能把它当作已验证无损降本方案；本轮保留当前 Runtime reasoning 设置，将方法接通与求解能力降配分开。[S12]

### 7.2 一份总账，包含所有真实请求

在 canonical `run_formal → run → System → Provider` 全链路传入 Governor。所有 Planner、Runtime、Extractor、Builder、finish、repair、trial 和每次 HTTP retry 都在请求发出前准入，在返回后以真实 usage 入账。

账单以 actual HTTP attempt ID 去重；parent_train_task、program_version、stage、trial_id 是归因维度，不是创建额外额度的依据。父 Train 的求解、学习和程序验证共享父任务上限，全批还受同一总额限制。预算不足时停止新的学习/执行请求，使用已经预留的最终提交机会；不启动另一个满额 task 来“恢复”。

预留根据当前真正的 finish payload 与其输出 cap 计算。不能把现有固定 65,536+32,768 预留机械套在更小的总任务预算上，否则新预算下可能第一条正常请求就无法发出。未知计费保持已占预留额度并停止继续派发，不能按零成本处理。

当前 Governor 的 RLock+JSON 只适合单进程。并行阶段采用单一 admission 调度者，或每 cell 独立账本且静态子预算之和不超过全批上限；不能让多个进程各自读写同一个 JSON 就声称全局封顶。首版可以选后者，不必先建分布式预算服务。

### 7.3 成本指标与摊销

同时报告实际 prompt/completion/total tokens、请求数、缓存字段、已知美元费用（有冻结价格时）、模型/角色、CPU 时间、Program 调用和原生动作。不同模型的价格或 token 定义不同，不能用一个统一单价倒算。

对同一个模型和 benchmark，令额外学习成本为 $C_{learn}$，每个评估任务的平均在线节省为 $\Delta C$。只有在 $\Delta C>0$ 时，报告估计回本题数 $N=C_{learn}/\Delta C$；否则报告当前条件下未回本。保留总 Train 成本与仅编译/学习增量两种口径，说明对照是否需要同样的源执行，避免把历史已花费用当作新方法免费。

小批优先检查实际被替代的工作和新增调度开销，不预设必须节省 20% 或某个准确率百分比。数值阈值必须来自后续观测和正式预算，不能为通过验收临时选一个有利数字。

## 8. 多模型：修正真实配置传递，按能力接入

当前 `models.lock.json` 的六个 formal model 是未配置占位；唯一完整 current_test 是文本 DeepSeek 配置。这只能支持已配置模型上的实验，不能称为六模型已经就绪。[S8]

需要模型配置内容，而不是要求用户提供某个特定文件。每项包括实际 provider model ID、endpoint、dialect/API 形式、凭证环境变量名、文本/图片能力、工具调用能力、reasoning/thinking 的支持与映射、输出 token 限额字段、usage 字段及缓存/推理 token 口径、generation seed 支持、超时/重试约定。目录用独立安全 run_label，不能因为 API model ID 含斜线就改写请求 ID。

修正 `model_settings` 漏传有效 reasoning/thinking 设置；修正 `resolve_call_settings` 的 protocol 合并为 global→stage→purpose；最终 provider 由解析后的实际设置构造。当前实验如果坚持所有角色使用同一 backbone，应验证这一约束，而不是假装 stage.model 已支持。

修正 provider snapshot 的 `http_token_limit_field`，使其等于实际 payload 字段。特定 provider 的 reasoning_content、thinking 对象和 reasoning_effort 不能无条件发送给其他 dialect。使用 capability profile 明确可发送字段和响应解析；不支持的必要能力在请求前报 unsupported，不静默回落到另一个模型或另一种推理语义。

每个实际接入先对冻结请求/响应 fixture 检查序列化、工具参数、usage、finish_reason 与重放。只有协议需要远端确认时做有限真实调用。纯文本模型对 DocVQA 标 unsupported；只有第二个模型实际完成接入后才能说具备多模型运行能力，协议接入本身不证明方法在该模型上有效。

Train 在同一个 model×benchmark×seed 内按固定顺序更新一份独立 Bank，不能并行打乱其在线学习历史。不同 cell 可并行；Bank、checkpoint、workspace、输出目录和预算身份独立，调度器控制 provider 并发与总额度。三 seed 是独立训练运行；provider 不支持 generation seed 时如实记录，不虚构模型采样已经受 seed 控制。

## 9. 验证安排：一次有限真实批次，不循环重跑

### 9.1 先完成针对真实改动的零模型验证

这些检查是离线代码/Worker/固定环境测试，不消耗模型 token，也不再重放完整 Train 解题。覆盖下表中的不同风险即可，不为每个小字段叠加大量重复测试。

| 验收边界 | 必须看到的结果 |
| --- | --- |
| 单来源资格 | 一个实际源局部验收通过可成为 usable；不要求第二 task_key；同源变体仍只记一个来源 |
| 无效证据 | 仅 schema/文件存在/模型声称通过不能晋升；源码、环境或局部合同改变使旧验收失效 |
| 构建与冻结 | 第一个有效绑定触发一次 Builder；通过后 Frozen 保留代码、implementation 和验收身份 |
| 单答闭环 | 无需程序可一次 finish；临时代码产生真实 trace；可编译、冻结、调用学到的程序；intermediate 后继续回答 |
| 图执行 | usable 可以完成绑定的局部节点；dynamic 整题节点不因空 required_fields 提前结束 |
| 公共输入 | SearchQA 不联网；LiveMath 选项不丢失；视觉路径真实传图；隐藏变体/gold 不进入 Runtime/Builder |
| 消费与异常 | 输出读取、使用、替换、丢弃分别记录；坏输出不提交；文件发布前检查；恢复不重复已接收的付费响应 |
| 干预隔离 | NoSkill 即使知道 Program ID 也不能绕过 view；Guidance-only 不暴露 Program/implementation/可执行 workflow |
| 冻结 | Val/Test 前后 Bank hash 一致，不写资格、检索统计或训练信用 |
| 模型与费用 | 有效设置等于实际 payload/snapshot；所有 retry 和 trial 归同一总账；并发不会超配或相互覆盖 |

机制成功用固定 StubProvider 与真实受控 Worker/Adapter 检查；其中人工构造的技能只作为工程正向 fixture，不计入“自然学得 Program”的方法结果。对暂不可用的视觉/环境运行时记录未验证，不能用纯文本 mock 宣称完整通过。

另将上一轮已定位的两个轻量问题一并修复：choice guidance proposal 的非字符串/unhashable 叶子字段统一形成受控拒绝；learning/runtime choice 配置不一致在启动前拒绝。无需为这两个叶子问题另开真实模型批次。

### 9.2 一次固定真实规模

已配置 DeepSeek、seed42，覆盖当前实际运行的 SearchQA、LiveMath、OfficeQA、SpreadsheetBench。每项只做 **3 个 Train 主执行，以及 4 个 Val 物理题的三臂评估**。四项合计 **12 个 Train 主执行、16 个 Val 物理题、48 次 Val 执行，共 60 个主 episode**。这是四项加起来的总量，不是每个 benchmark 重跑 12+6，也不是可滚动重复的模板。

试验前固定 task IDs、顺序、三臂顺序、模型有效设置、公开投影、scorer、预算和代码身份。Train 用各 benchmark 固定 seed42 顺序的前三题；Val 在既有 Val manifest 上按 `sha256("atomic-unified-20261010|" + benchmark + "|" + physical_key)` 排序取前四题。碰到输入本身不可用应在任何付费请求前按公开可用性一次性固定排除及顺延，并写出原因；不能在看到正确率或 Program 暴露后换题。

每个 benchmark 从独立空 Bank 运行这三个 Train，学习只进行一次，之后冻结。已有正式 Bank/trace 可用于零模型回归和定位，不将手工补写资产或旧 guidance 冒充本批从空 Bank 自然学到的程序。这个小 Bank 测的是端到端学习与执行的机制和局部方向，不估计完整 Train 规模的最终性能。

| 臂 | 冻结资产视图 | 求解能力 |
| --- | --- | --- |
| NoSkill | 不可见任何学习资产 | 同样的公开输入、原生工具、临时代码能力、Runtime 和总预算 |
| Guidance-only | 只见真实学到的文字指导，不见 Program、implementation 或可执行 workflow | 与其他臂相同 |
| Atomic-full | 见完整验收后的资产，可调用与组合 Program | 与其他臂相同 |

Guidance-only 需要正确的只读视图：当前 `guidance_off` 是“关闭指导”，不能拿它冒充“只有指导”。Skill 附属文字可作为文字资产，但不得泄露 Program 源码或经新资格方法绕过屏蔽。三臂各题使用独立 workspace，不能复用上一臂留下的结果。

Full–NoSkill 测完整学习资产的增量；Full–Guidance 测可执行资产及其组合的增量，不能将后者称为单独改变一段程序源码的纯因果效应。三臂不分别重训，也不为每一臂再跑不同数据条件。

局部 Worker 验收最多 **24 次**，含源重放、必要回归及修复后的验收；它们不算新的物理题，不允许另开模型整题 continuation。达上限就停止新增编译，不靠补题争取出现理想 Program 数量。

### 9.3 统一预算

| 约束 | 本次固定上限/设置 |
| --- | --- |
| 全批实际 total_tokens | 6,000,000；所有角色、repair、retry、trial 和必要协议接入合计 |
| 实际 HTTP 尝试 | 最多 600，作为防循环上限；不是另一份可花 token |
| 每个 Train 主题 | 200,000 tokens，含求解、学习、Builder 和试用 |
| 每个 Val 题每臂 | 96,000 tokens，含全部在线角色 |
| Runtime 单次 completion | 保留 32,768 与现有 reasoning 设置 |
| Planner / 通用 Extractor 单次 completion | 各 8,192；不对没有 Planner 的单答路径新增 Planner |
| Builder 单次 completion | 16,384；同一构建任务的首次及后继版本合计最多一次源码修复，修复 cap 32,768；新版本不重置次数 |
| 最终序列化 | 已获证据的 finish_only cap 2,048；当前 provider 支持时关闭该次思考；不让此步骤重新解题 |
| 单答内部局部执行 | 每题最多两次；始终计入同一任务及全批额度 |

这些是本次工程集成的费用边界，不是已经验证的最优配置；全部写入运行身份。全批上限优先于子上限，不能保证每项都花到子上限。对普通 guidance 和 LM1 grounding 使用既有更小专用限额；不能用通用 8,192 覆盖其更严格的合同。

第二实际模型若配置可用，可在同一额度内做最多两个必要的真实工具对话回合，加上零调用序列化/响应回放。这不计入 60 个 benchmark 主 episode，也不等于该模型的方法效果验证；没有配置就记录未接入。暂缓的 ALFWorld、尚无视觉模型的 DocVQA 不为凑六项加入付费批次。

按 benchmark 和 Val 三臂均衡轮转，防止预算耗尽时只留下便宜任务或单侧结果。达到任一全批边界即停止，不增加备用题、不追加 seed、不另开“工程调试”费用。已开始的任务触及预算边界时，仍将实际封存的预测交给原 scorer，并另记 budget_censored；未形成可评分答案时按原空答或失败规则处理，不得因预算耗尽覆盖已获得的正确分数。未启动任务记 missing，不伪造得分。三臂都有真实最终记录才进入完整配对，缺失数量和原因一起披露。

### 9.4 希望看到什么，以及没有看到时如何结束

工程上应看到真实的输入→局部执行→trace→提取→构建→源验收→usable→Frozen→调用→结果消费→原 scorer 这条链。已知零来源/两来源阻断、提前 complete、字段异常、预算漏账和设置不生效不得再次发生。

方法上报告每题正确性、软分、在线 tokens/调用、Program 次数、真正消费次数、被替代的操作、失败回退以及额外 Planner 成本。Full–NoSkill 和 Full–Guidance 各最多 16 个完整物理题配对，每 benchmark 各 4 对；报告胜/负/平、总 token 差和中位数，不用这种规模推断稳定排行榜优势。正常结束的完整配对与含 budget_censored 的完整配对分别报告成本差，防止把提前截断产生的少耗 token 计作正常复用收益；主准确率仍保留所有已评分任务的原始分数。

有自然学得 Program 在非源 Val 任务被真实消费，并减少重复模型工作，是本批最直接的积极证据；只生成代码、只调用未消费、或 Runtime 随后全盘重做，都不是所要求的机制收益。四个 benchmark 不必每项都产生 Program，程序也不能修复全部数学能力不足。

如果出现实质工程错误，用此次已经记录的输入与响应离线复现并修复；验证确需远端时，只能使用本批尚未消耗的预算和原任务，不能重新开一轮同规模试验。若链路正常但没有自然复用，报告本批未观察到复用；若 Full 准确率和成本都变差，报告负结果并停止扩大。不能补题、调阈值、增加 Train 直到结果变好。

**工程放行不要求 accuracy 必须上涨。** 但若自然程序始终未被实际消费，就不能宣称完整方法收益已检验充分，也不应立即扩大为多模型×三 seed 的高成本批次。此时依靠现有 trace 分清任务缺少可复用操作、提取不当、检索/绑定失败或收益不足，提交这一轮完整结论；任何新的研究实验应有新问题和独立预算，不自动重复当前模板。

## 10. 实现顺序与完成定义

| 修改包 | 主要文件 | 完成定义 |
| --- | --- | --- |
| 统一程序资格 | `empirical/__init__.py`、`system.py`、`learner.py`、`bank.py`、新增 `local_validation.py` | 一源构建、源验收晋升，跨来源只统计，Frozen 保留验证身份 |
| 通用执行与单答调度 | `harness/simple_protocol.py`、`benchmarks.py`、`executor.py`、`program_worker.py`、新增 `answer_executor.py` | 无原生工具也能内部计算；中间结果不会结束整题；共用真实提交 |
| 学习与材料 | `learner.py`、`prompts.py`、`model_view.py`、`task_context.py` | 局部操作证据可追溯；无单答程序禁令；不重复支付多套提案 |
| 成本与模型 | `budget_governor.py`、`agents/provider.py`、`experiments/run_empirical.py`、`run_formal.py`、模型/默认配置 | 全链路预算封顶；有效设置、请求、快照和 usage 一致；并行隔离 |
| 评估与报告 | `bank_view.py`、新的一次性验证 runner、现有 formal log | 三臂同条件；只读不绕过；60 主 episode 和全批额度锁定；输出完整配对和缺失记录 |
| 规范同步 | README、CF4-R1 相关条款、POLICY/model-view/material/运行身份 | 不再声明 single_answer guidance-only 或通用两来源；旧结果不被新语义覆盖 |

新增模块名是实施建议，允许将同一职责放在现有模块，但不能漏掉表中行为。禁止只改配置、只删一个 if、只放行菜单或只改 Bank 状态就宣布完成。

运行 revision、学习材料、局部验收政策、答题协议、provider 有效设置和预算身份必须版本化并进入 checkpoint/Frozen/run manifest；建议新 implementation revision 使用 `empirical-v3.2-atomic-unified`。这是待合入版本标识，当前仓库还没有实现该版本。

## 11. 旧实验、后续并行与需要交付的结果

旧 SearchQA、LiveMath 成绩继续保留，标明当时是 guidance-only 接入，不能改标签声称已经验证 Program 复用。OfficeQA、Spreadsheet 的旧运行也按原版本和原预算保留，不能将新增验收规则静默写回旧 Frozen。

修订中的最小验证先完成，不立即重跑全部旧 Train。需要利用旧 Program 排查时，在独立派生目录用真实原始证据做重放，记录父 Bank 和新验收来源；这样的派生结果不是“空 Bank 新训练”。旧单答记录更不能直接生成一个假称旧时已经执行的 Program。

正式新方法若要比较三个独立 seed，应使用同一新版本、同一能力/预算合同和空 Bank 训练条件。旧 seed42 guidance-only 与新 seed43/44 完整 Program 不能直接合并成三个同方法 seed。是否复用旧源执行进行离线学习是另一种明确训练协议，必须单独标识，不能在成本和实验命名上隐藏。

达到第 9 节工程条件、获得可解释的有限真实结果，并完成实际要使用的 model/adapter 接入后，才启动对应已就绪 cell 的正式并行。无需所有模型同时上线，但不能把占位模型或未验证视觉路径放进“全部准备完成”的结论。原有暂停状态不由本审查自动解除。

实施完成后需要的是一份紧凑结果包：固定 commit 与 resolved 配置/公开输入身份；受影响分支测试结果；逐题三臂结果和预算总账；生成/验收/Frozen/消费的数量与可追溯实例；失败及未触发路径；模型接入覆盖；正式启动或不启动的理由。沿用现有 trace、request、usage、checkpoint 产物导出即可，不再要求重新拼装庞大、重复的全量审查材料。

## 12. 证据索引与本次验证边界

以下为固定提交链接；行号只用于定位当前版本，不保证修订后不移动。

| 编号 | 已核对的源码或材料 |
| --- | --- |
| S1 | [CF4-R1 第 5 节](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/993d6cd0c21a297e55343aab4a65009494f0df16/docs/specs/SkillCompiler_CF4_R1.md#L219-L235)：single_answer 明确禁止 Builder/Program |
| S2 | [empirical/system.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/system.py)：validate_config、178–205 设置合并、507–528 单答入口、615–629 学习、654–825 试用 |
| S3 | [empirical/learner.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/learner.py)：21–45 材料、106–165 提案、196–278 学习、441–585 构建与试用 |
| S4 | [empirical/bank.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/bank.py)：79–134 路由/晋升、202–222 菜单、276–319 Frozen |
| S5 | [empirical/executor.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/executor.py)：70–79 消费快照、294–319 invocation、335–345 提交、477–485 完成快捷分支 |
| S6 | [simple_protocol.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/harness/simple_protocol.py)、[benchmarks.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/harness/benchmarks.py)、[alfworld_simple.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/harness/alfworld_simple.py)：能力、公开输入、局部检查 |
| S7 | [configs/default.yaml](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/configs/default.yaml)、[budget_governor.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/budget_governor.py)、[run_formal.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/experiments/run_formal.py)：预算与正式入口 |
| S8 | [models.lock.json](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/models.lock.json)、[agents/provider.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/agents/provider.py)：模型配置、payload、snapshot 与响应 |
| S9 | [program_worker.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/program_worker.py)、[program_submission.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/program_submission.py)、[model_view.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/model_view.py)：底层执行、结果合同和材料投影 |
| S10 | [benchmark_profiles.json](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/benchmark_profiles.json)、[canonical_manifest.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/experiments/canonical_manifest.py)：六项实际名单和权限预算 |
| S11 | [bank_view.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/bank_view.py)：当前只读干预行为 |
| S12 | [LM1 报告](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/993d6cd0c21a297e55343aab4a65009494f0df16/reports/LM1_20261010.md)：既有有限验证结果，不能代替完整 Program 路径验证 |

本次实际完成了三组零模型检查。真实 Bank 类的临时数据库探针复现：零证据 candidate 可出现在当前 Train 显式菜单；同来源两次正例仍不能晋升，冻结后 Program/implementation 消失；第二物理来源才触发晋升。六类真实 Adapter/声明合同探针确认，三个 AnswerAdapter 已允许 intermediate/final_answer，现有 Worker 支持无 workspace 的纯 JSON 计算。三个源码函数的 AST fixture 复现模型设置漏传、stage.protocol 丢失和 snapshot token 字段错误。

这些检查没有执行生成的 Program、没有调用远端模型、没有启动 ALFWorld 或运行完整 benchmark scorer。它们证实本次定位的分支和合同问题，不等于修复后的完整集成通过。第 9 节的测试与真实批次是交付给实施者执行的验收要求，尚未运行。

