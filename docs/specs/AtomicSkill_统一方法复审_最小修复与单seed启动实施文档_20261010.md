# AtomicSkill 统一方法复审：最小修复与单 seed 启动实施文档

版本：2026-10-10 复审版  
审查对象：本次统一方法交付包、实际收费批次、GitHub 最新主分支对应运行源码。  
文档性质：代码审查结论与待实施修改规范。本次复审没有修改生产代码，没有调用真实模型，没有启动或恢复正式实验。

## 1. 执行结论与本轮范围

统一 Atomic 主链已经落地，不需要整体返工。单答题不再被永久限制为 guidance-only；一条真实局部来源即可构建并通过局部执行取得 usable 资格；候选、资格、冻结和测试消费的边界也已经接通。不能再将当前零 Program 归因于“必须跨来源才能晋升”。

当前版本仍不宜原样放大到完整文件类 Train。新的真实记录已经暴露了学习材料超大、学习额度被挤占、来源绑定合同不清晰的问题；另有本地可复现的 XLSX 效果哈希误判，以及正式启动配置与有限验证配置不一致。这些问题应在开始新单 seed 前修正，全部可以利用现有真实记录及本地文件进行工程验证。

本轮只完成四组修改：学习投影与预算预留、局部提案与结构修复、XLSX 效果身份、正式入口配置。不得借此重新设计整个方法，也不增加“必须自然生成 Program”“前几题必须涨分”这样的启动条件。修完并通过针对性离线回归后，直接进入一个连续 seed 的 Train→冻结→Val→Test；不再重做三臂 60 次验证，也不另开一轮 12+6。

本文件完整规定这四组补丁、验证方法和启动安排；此前统一方法文档的其他原则继续有效。本文件对“必须先在小批观察到自然 Program 消费才允许扩大”的表述作出明确修正：自然复用与收益属于实验结果，工程放行只要求真实链路可执行、合同正确、费用受控、记录可核验。

## 2. 复审身份与已有验证的有效范围

| 项目 | 已核实的身份或结果 | 解释 |
| --- | --- | --- |
| 本轮收费批次源码 | `a745f569dfaca4b4bd7fb1100e875366005f45d2` | 60 个执行的实际代码身份 |
| 最终交付运行源码 | `79aa1c006e39987d88b664cb2ae75d8c1e49be5d` | 已包含收费后局部资格等修复 |
| 本次读取的最新 main | `2e2127322d96486d2299606bd3f593f737042a4b` | 相对 79aa1c0 只更新 README／报告，运行代码一致 |
| 审查包校验 | 573 个 SHA256SUMS 条目全部匹配 | 包内容完整性已验证 |
| 交付源码复核 | 由受测 source.tar 加 post_pin.diff 重建；234 个文件逐一匹配 GitHub blob 身份 | 结论来自实际代码，而非仅阅读报告 |
| 交付方最终全量回归 | 在 `8bb36b6b08d0c8bf002672b4916e18beea3a808b` 上 372 passed、2 skipped | 不与其他重复执行次数累加 |
| 79aa1c0 身份补丁回归 | 受影响回归分别 27 与 69 passed | 不是另一次完整收费实验 |
| 本次独立复审 | 费用／分数重算、配置及准入纯函数探针、真实 XLSX 本地操作 | 未在本环境重跑全量 pytest 或 Docker Worker，不将本地探针冒充完整集成验证 |

主要代码依据：[learner.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/learner.py)、[local_validation.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/local_validation.py)、[system.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/system.py)、[run_formal.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/experiments/run_formal.py)。以下提到的源码相对路径均相对于该仓库根目录。

## 3. 这次实际结果应当如何解释

### 3.1 分数、费用与冻结记录可以对账

本批完成 12 次 Train 和 48 次 Val 执行：每个 benchmark 3 个 Train、4 个固定 Val 题，每个 Val 题执行 NoSkill、Guidance-only、Atomic-full 三个臂。总计 60/60，原始 267 条唯一 HTTP 回执合计 2,147,418 tokens，全部能与持久费用账本逐条匹配，unknown billing 为 0。四份 Frozen 的内容身份匹配，48 个 Val 执行前后的知识身份没有变化。

| Benchmark | Train 正确 | NoSkill Val | Guidance-only Val | Atomic-full Val | Train tokens | 三臂 Val tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SearchQA | 3/3 | 2/4 | 2/4 | 2/4 | 27,643 | 43,076 |
| LiveMath | 2/3 | 0/4 | 0/4 | 0/4 | 70,170 | 301,274 |
| OfficeQA | 0/3 | 0/4 | 0/4 | 0/4 | 420,954 | 526,425 |
| SpreadsheetBench | 1/3 | 3/4 | 4/4 | 3/4 | 381,489 | 376,387 |

Train 总费用 900,256 tokens，其中原题求解 781,888、学习 118,368。学习费用不是 Builder 费用：本批实际 Builder 请求为 0，Program 生成、usable Program、自然 Program 消费、单答临时代码执行也都是 0。SearchQA 和 LiveMath 各留下 2 条 guidance；OfficeQA 和 SpreadsheetBench 的 Bank 为空。

NoSkill 的 Val 总费用为 409,761，Atomic-full 为 445,367。当前没有观察到总体在线降本，不能把若干预算截断题的低消费描述为 Program 节省。金额没有冻结价格身份，因此仍不推算美元金额。

来源：本次交付包 `finite_batch/` 中逐题 trace、requests、Frozen、budget.json，以及 `verification/learning_and_execution_audit.json`；本次独立重算与交付报告一致。

### 3.2 有三个容易误读的结果

LiveMath 的 4 个 Val 题在三个臂中均为 0/4，12 个响应都正常 stop，没有本批预算截断造成的空答案。对应的 guidance 暴露次数为 0，因此不能将这些错误解释为“技能注入导致答案变差”。它表明这四题仍未解对；样本不足以估计稳定准确率，也没有检验到技能使用的效果。

SpreadsheetBench 的 Guidance-only 为 4/4，但该 benchmark 的 Bank 是空的，根本没有 guidance 注入。这个 100% 不能解释为 guidance 有效，也不能直接与历史另一个 baseline 的 100% 混为一谈。三臂各自调用模型，空 Bank 情况下仍可能出现独立生成差异。

收费批次没有自然 Program，不能据此宣称 Program 方法有效或无效。尤其文件类六个 Train 中四个 Extractor 请求没有真正发出，另外两个出现截断或提案合同问题；“有程序化机会但学习链未正常利用”与“充分执行后没有收益”必须区分。

### 3.3 预算删失口径保持准确

60 次执行中有 20 次标记 budget_censored，其中 Train 4/12、Val 16/48。OfficeQA 两次 Train 的该标记来自学习请求被拒，原题求解已经进入动作次数结束后的收尾；不能笼统说全部是求解 token 用完。SpreadsheetBench Atomic-full 的四个 Val 都带该标记，但其中三题实际输出按既定 scorer 得分成功；保留原始分数。

当前准入检查使用输入 UTF-8 字节界加最大输出额度，可能在实际 token 尚未到达 96,000 时拒绝下一请求。它说明该结果受预算策略约束，不说明已经实现等能力降本。

现报告配对分层采用“三臂中任一臂 censored”的共同题集。保留该口径时应明确写出；若另报双方独立口径，Full–Guidance 的 normal/censored 是 9/7，而共同口径是 8/8。此解释修订不改变原始评分，也不构成实验启动阻断。

## 4. 已完成的主链应保留

当前单答分支通过 AnswerExecutor，允许同一个 Runtime 直接回答、执行最多两次临时代码、调用已验收 Program、读取并消费中间结果。只有缺少真实局部执行证据时才退回文本学习；并不存在按 SearchQA 名称永久关闭 Builder 的分支。

首次构建所需独立 Train 来源数已经是 1。局部资格需要真实来源、隔离环境重建、实际 Worker 执行、输出／效果合同和版本身份；跨来源复用作为后续证据统计，不是首次 usable 的必要条件。默认不付费续解整题来验证一个局部 Program。整体任务失败也不自动抹掉其中已成功、可核验的局部操作。

candidate 不进入 Runtime／Frozen，资格绑定程序、环境和政策。动态中间节点不再因空 required_fields 提前结束；实际消费、替换和丢弃可以区分。这些正确机制应继续使用。

原生操作 v2 增加真实动作核对，已经修复“来源执行 GO_TO，候选只 LOOK，却因为 accepted=True 被判通过”的问题。当前按单个真实操作检查同名同参，尚未覆盖任意多动作等价优化；这是能力边界，本批 Builder 为 0，不能把它列为此次零 Program 的原因。没有新的真实失败证据前，本轮不扩展到通用状态等价验证器。

代码依据：[answer_executor.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/answer_executor.py)、[executor.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/executor.py)、[bank.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/bank.py)。

## 5. 修改 A：学习材料去重、有界投影和学习额度

### 5.1 真实断点

以下四个请求用当前 Governor 重建，均在 HTTP 发出前被父任务预算拒绝。表中“输入界”是实际序列化请求的 UTF-8 字节界，不是供应商已经收取的输入 token。

| Train 任务 | 请求前实际已花 tokens | 200,000 父上限内余量 | Extractor 输入界 | 加 8,192 输出后的准入预留 |
| --- | ---: | ---: | ---: | ---: |
| officeqa:UID0170 | 134,339 | 65,661 | 293,079 | 301,271 |
| officeqa:UID0221 | 151,848 | 48,152 | 225,672 | 233,864 |
| spreadsheet:44296 | 139,787 | 60,213 | 80,307 | 88,499 |
| spreadsheet:58032 | 126,715 | 73,285 | 151,782 | 159,974 |

前两个输入界自身就超过父任务上限，即使之前求解零费用，也不能发出请求。此时提高整批额度没有帮助，本批只用了原 600 万上限中的约 214.7 万。

`Learner._view` 原样复制最近三个 local_evidence；每条又携带完整 public_task、prefix、operation、reference。`learn` 的 completed_train_cases 还会复制历史案例的最近三条证据，并将当前题重复放入其中。UID0170 当前 experience 为 137,563 bytes，历史块又占 126,442；UID0221 当前 experience 为 40,429，历史块却占 157,446。“最后三条”限制了数量，没有限制材料体积。

### 5.2 必须实现的材料合同

修改 `src/atomic_skillgraph/empirical/learner.py` 的 `_view`、`learn` 和 Builder 材料构造；共用 `model_view.py` 中的投影设施，避免再出现两套互不一致的限制。

Host 保留完整、不可变的原始 trace、真实操作、前缀、环境、原始结果和文件身份。模型收到紧凑的操作目录：case_id、local_evidence_ref、真实操作类别、读／写效果、成功状态、已发布文件名、可用参数来源、可选输出引用。目录明确区分“检查已有文件”和“本次产生并发布文件”。

当前题只出现一次。历史默认只传必要的紧凑描述和引用；需要选中其他 Train 来源时才附其必要局部操作内容，不自动展开八题全部历史。结果中的长文本、完整文件内容和重放前缀不因被多个字段引用而重复内嵌。

Builder 收到被选中的真实局部操作所需完整代码／参数化信息、短合同及 Host 绑定身份；主机重放所用的完整前缀留在 Host。不得截断代码中间、伪造摘要中的成功事实，或给模型只有一个实际上无法解析的 ref 然后期待其自行获取内容。首次 Extractor 的目录需足以选择操作，Builder 请求必须携带构建该操作需要的真实内容。

在最终 provider payload 组装后，统一检查整个序列化请求，包括 system、messages、tools、schema、转义开销。超过界限时按确定顺序移除重复项和可选历史，或选择另一个真实、完整且适合预算的局部操作；不能随机裁掉必需输入。若确无可在界限内完整表达的操作，记录 `material_too_large`，保留来源和原因，不发出注定无法准入的请求。

无需增加一个付费摘要 Agent，也不增加一个新的 Planner。所有被省略的原文仍可供 Host 验证，是否能程序化仍由现有学习角色决定。

### 5.3 初始预算方案与实现点

采用一个可直接实施的保守初始方案：Train 父上限保持 200,000，求解池 120,000，学习池 80,000，两池都从同一个父任务及整个运行账本扣费。120,000 包括 Planner、Runtime、收尾和它们的重试；80,000 包括 Extractor、Builder、结构修复、源码修复以及学习 trial 中任何模型调用。

| 配置内容 | 初始值 | 状态 |
| --- | ---: | --- |
| `budget.train_task_tokens` | 200000 | 现有父上限 |
| `budget.train_solve_tokens` | 120000 | 本补丁新增，求解含 finish |
| `budget.train_learning_tokens` | 80000 | 本补丁新增，学习统一扣费 |
| `budget.eval_task_tokens` | 96000 | 本次保持，调整需另行明确 |
| `learning.extractor_payload_max_bytes` | 16384 | 本补丁新增，完整请求界 |
| `learning.builder_payload_max_bytes` | 16384 | 本补丁新增，完整请求界 |
| `learning.structural_repair_payload_max_bytes` | 8192 | 本补丁新增，仅短错误及待修提案 |
| 首次 Extractor 最大 completion | 8192 | 保持当前量级 |
| 首次 Builder 最大 completion | 16384 | 保持当前量级 |
| 结构修复最大 completion | 2048 | 新专用 purpose，disabled thinking |

容量算术为 `16384 + 8192 + 16384 + 16384 + 8192 + 2048 = 67584`，为一次提取、一次构建和一次短结构修复留下准入空间。它是待实现的预算设计，不是宣称当前投影器已经达标，也不是最优值证明。应以本包四份真实材料进行投影测试，证明目录、选中操作和构建合同完整且最终 payload 达标。不能只缩小一个 user 字符串后宣称通过。

源码修复仍遵守原有每 job 最多一次，并且只有剩余学习／父任务／整段运行额度足够才发出。不能因 job 进入 repair 或生成新版本而重置额度；本方案不保证每次提取、构建、结构修复和最大源码修复全部都能执行。未使用的预留不产生费用；本轮先使用固定两池，避免自动借额掩盖预算语义。

修改 `empirical/system.py::validate_config`、请求上下文和 `empirical/budget_governor.py::admit`。当前 schema 会拒绝这些新键，必须先实现后配置。增加明确的 `budget_pool` 归属，不能只看 stage 名称：trial 中 Runtime 也属于 learning。每次准入同时满足 pool、parent、whole-run 和 request 数限制；重试保留归属，未完成尝试按预留记账，unknown billing 继续阻断后续 HTTP。

求解临近池上限时用现有 bounded finish，finish 本身留在 solve 池内。普通 Runtime 准入还必须满足 `solve_used + request_bound + finish_reserve <= train_solve_tokens`，真正的 finish 请求不再重复预留；不能只保留当前对 200,000 父上限的 parent_finish_reserve 检查，否则 solve 池耗满后收尾仍会被拒。若新增 solve 池专用 BudgetExhausted 错误码，必须接入 Executor／AnswerExecutor 当前对 parent_task_budget_exhausted、runtime_finish_reserved 的受控收尾分支，也可沿用已识别的错误码。文件和环境任务保持既有输出封存、终止与评分流程，不额外增加一个文字 final 请求。

角色自身上限继续生效，不能用旧 Runtime 600,000 角色上限绕过新的 120,000 solve 限制。新字段、有效值和预算池归属进入 resolved config、请求日志与身份摘要。

输入 UTF-8 界继续按真实 payload 计算，不通过除以四或伪造 tokenizer 数值来准入。保持 Eval 当前策略意味着仍可能发生保守提前收尾，应如实记录，不将其当方法降本。后续若要调整 Runtime 的动态 completion cap，应作为一个独立、冻结的推理预算选择，不在本补丁中隐式混入。

## 6. 修改 B：Host 固定来源绑定，模型只提交必要决策

### 6.1 当前 Spreadsheet 提案不能只补 prefix

真实 `spreadsheet:382-10` 首次 Extractor 已提出“清空指定列中不匹配目标文本的单元格、保留格式并保存文件”的 Program。第一次响应可解析，并非 no_change。

它引用了一条真实 local_evidence_ref，却填写 reset 和空 prefix；实际所选操作前面有一个动作，因此先被前缀校验拒绝。随后修复响应 8,192 completion 全用于 reasoning，没有工具调用。

进一步对同一首提案离线检查还发现：所选 ref 对应第二次“读取并打印输入／输出样式”的检查操作，该事件没有本次发布文件的身份；真正的写入发生在第一、第三个事件。提案把 `case1_result.xlsx` 当 JSON 引用路径；final_files 合同补齐后需要 files 引用，提案没有；Sheet1、A、comments 等参数在目前“整 JSON 叶值相等”的来源检查中也不被接受。

因此不能由 Host 偷换成另一个写入事件、补一个 prefix，再把旧提案记成自然成功。必须修正合同和反馈，让新提案明确选择真实来源，仍经过实际 Worker 验收。

### 6.2 提案和 Builder 接口修改

修改 `empirical/prompts.py` 的 LEARNING／BUILD schema 与提示，联动 `learner.py`、`local_validation.py::resolve_evidence`。模型提案的 case binding 只需提交现有 case_id、local_evidence_ref、参数值及来源引用、输出参照选择。start_mode、prefix、source_trace_sha256、environment 等权威字段由 Host 从选中证据解析并冻结。

Host 建立 canonical binding 和 binding_hash。内部仍保留执行所需的完整字段，但不要求模型重新抄写，也不相信模型回传的这些权威字段。Builder 输出 source，可附短 binding_hash；不再输出并逐字复核包含完整历史的 trial_inputs。执行只使用 Host 固定的 bindings，原有“不能更换被验收来源或起点”要求继续成立。

操作目录提供 typed reference 目录。对于文件发布，从真实成功发布记录及已有文件身份派生 publication reference，明确其规范化 files 值；不能把 operation.arguments 中声明了某文件名当成已经产出该文件。final_files 先统一到现有输出合同再展示给模型。若提案另外声明 output_file 等必需字段，也必须有真实可解析的参照或明确定义、可审计的发布结果投影；不要只补 files 后继续留下一个不存在的路径。

输入绑定支持三种有出处的值：公开结构化字段、公开字符串的精确位置和值、已实际执行源码中可定位的字面量。字符串 span 只证明出处，不能因为单字符 A 在长文本中出现就声称列语义已经确定；参数含义及它与被选局部操作的关系仍需由提案和实际源验收共同成立。对未执行过的可选分支，不得把一个新默认值描述为来源已验证事实。禁止从 gold、correct_choice 等私有评分字段绑定答案。

不使用任意 eval 执行输入来源表达式。AST 路径／文本 span／JSON 路径均由 Host 验证位置与值。源选择错、没有写入效果或合同不满足时，继续受控拒绝，不降低资格条件。

### 6.3 一次结构修复应修结构

预检聚合当前能确定的错误：来源不存在／没有发布效果、参数无来源、required 输出缺引用、引用路径错误、类型错误。一次短反馈给出错误码、字段、期望和可选合法引用，避免逐个字段产生昂贵修复链。

新增 `learning_structure_repair` purpose，针对 JSON／绑定修复使用 disabled thinking、明确的 submit_learning 工具提交和 2,048 completion 上限。修复材料只带紧凑待修提案、错误集合及所需操作目录，完整序列化输入不超过 8,192 bytes；不重传全部历史，也不再次求解原题。首次算法提取与 Builder 的推理配置保持独立。

`system.agent` 目前在 repair 循环前只解析一次 settings 和 provider。必须把修复轮的有效 purpose、settings、provider、tool_choice 按轮解析，否则仅在 YAML 加一个 override 不会改变实际请求。将实际 effective settings／purpose 纳入 response_key 与日志，保留原请求、修复请求、HTTP retries 的不同身份及同一父预算。

当前 purpose whitelist 及强制工具选择仅覆盖既有部分分支，也需同步支持该新 purpose。不能对不支持的 provider 静默发送无效 thinking 或 tool_choice 参数；本次只验收已配置 DeepSeek 的现有协议。

OfficeQA 的首次 Extractor 出现 8,028 reasoning 后 JSON 截断。对这种返回，能够可靠恢复的短候选按同一规则修一次；无法恢复完整候选时记录明确 truncation／contract rejection。不得凭缺失响应补造提案。结构修复受控成功或受控拒绝都是工程结果，不要求每个模型错误都被强制修成资产。

## 7. 修改 C：XLSX 效果身份排除保存时间造成的假差异

### 7.1 已复现的问题

`local_validation.artifact_identity` 对 ZIP 各成员的原始内容计算身份，虽然忽略了 ZIP 容器元数据，但仍包含 `docProps/core.xml` 中的 modified 时间。

在项目锁定的 openpyxl 3.1.5 下，本次用同一输入执行两次相同“清空 A2”操作，仅相隔约 1.2 秒。结果单元格完全一致，ZIP 成员只有 core.xml 不同；现有效果身份不同，导致合法 Program 被误判不等价。当前正向文件夹具使用 txt，没有覆盖这种真实 XLSX 保存行为。

本次还做了纯本地候选规范化探针：相同操作跨保存时间相等；改动单元格值、公式、样式分别不相等，四项均符合预期。该探针没有修改生产函数，也未经过 Docker Worker，因此生产回归仍需补齐。

### 7.2 精确修复范围

修改 `empirical/local_validation.py::artifact_identity`，并确保 `harness/benchmarks.py` 的来源文件身份采集调用相同政策。

仅对明确识别为 XLSX OOXML 的工作簿启用规则，至少核验内容类型与 workbook 成员。解析 core.xml 后只规范化命名空间 `http://purl.org/dc/terms/` 的 modified 保存时间；保留其他元信息及其他全部 ZIP 成员内容。成员名和规范化后的内容按确定顺序计算效果身份。

原始文件 SHA-256 继续保留用于审计；新增／明确版本化的 normalized effect identity 用于局部输出等价。不得删除整个 docProps，不得只比较目标单元格，也不得默认所有 ZIP 都适用这项规则。单元格、公式、样式、sheet、关系、成员缺失／新增等差异仍应被发现。

更新局部资格政策及文件身份版本，新记录同时标明算法。新政策不能给旧 digest 直接改标签：只有读取经原身份确认的真实源产物，或在隔离环境重放原始真实 source，才能建立新的参照。候选自己的输出不能成为它自身的参照。旧 Frozen、历史评分不改写。

保留原生操作 v2 的真实效果检查，在其基础上升级，不回退为仅判断 accepted=True。Program before_publish 与 source capture 必须使用同一个文件规范化入口。

## 8. 修改 D：统一正式入口的有效配置

### 8.1 当前默认入口并不能直接复现实验设置

`configs/default.yaml` 和 `configs/main_experiment_v1.yaml` 没有完整 budget，`run_single_cell` 也没有预算 override 参数。正式 campaign 明确要求 budget；本次零 HTTP 探针已经复现启动前 ValueError，未创建 run_manifest。

有限验证入口 `run_atomic_unified_validation.prepare` 自行注入预算和 LiveMath choice_guidance 设置。正式 `run_formal.resolved_config` 没有同样注入，因此只加 budget 仍会使 LiveMath 回到另一套通用 guidance 配置。已验证的 learning/runtime choice 策略、2,048／1,536 disabled-thinking 设置必须进入正式共用配置，不能只留在诊断脚本。

### 8.2 实现和交付要求

在正式与有限验证入口之间共用一个 benchmark 配置合并函数；它负责 benchmark 特有的已确认适配设置，不负责改变统一 Atomic 方法。LiveMath learning.choice_guidance 与 runtime.choice_guidance 一起开启，guidance_learning／guidance_grounding 分别为 disabled thinking、2,048／1,536 completion。SearchQA 不继承不适用的多选题专用配置。

新增受版本控制的本轮单 cell spec／base。新学习材料、绑定、局部验证和预算政策写入 implementation_revision、resolved config、checkpoint 和 Frozen 身份。用同一组输入离线解析正式／有限验证配置，对照方法、模型、reasoning／thinking、purpose、token 字段及投影值；仅允许运行目录、phase、固定样本路径和声明预算范围等预期差异。

每个将运行的 cell 必须显式提供整段 Train+Val+Test 的 token_limit、request_limit、finish_reserve，以及每题 Train／Eval／两池上限；若使用 validation_limit，也应覆盖计划中的局部执行。当前 600 万／600 请求／24 次局部验收属于已关闭的有限批，不能直接当成完整四 benchmark 的总额度。整段运行额度应从已有预算安排明确填入；本复审没有新授权一个更大的费用总额。

四个 cell 并行时，每个 cell 使用独立预算文件和输出目录，提供各 cell 硬上限的分配表，合计不超过声明总额。现有 JSON Governor 的锁只有进程内作用，不能让四个独立进程同时写同一文件并宣称已经实现全局并发锁。现有矩阵静态分配逻辑可以复用，但不要为启动单 seed 直接调用全矩阵入口。

`run_seeds` 元规范继续为 [42,43,44]，六 benchmark 的规范顺序继续保留；现有 single_cell 会校验这些字段。实际只由 CLI 的 `--seed 42` 和 `--benchmark` 选取一个 cell，不能为了本次单 seed 把 spec 改成 [42] 而触发校验失败。

本轮使用已完成真实调用的当前 DeepSeek 配置和四个 benchmark。其他模型为空不阻止这一模型的单 seed；不将其称为已经通过多模型验证。ALFWorld 付费暂缓，当前文本模型的 DocVQA 继续标 unsupported，而非零分。

`run_formal.campaign` 建议同步将整段预算耗尽记为 `budget_stopped`，保留已完成题和被拒请求；不要统一写成 execution_failed。达到硬上限正常停止，不自动追加预算，不把未执行余题补成错误答案。

代码依据：[run_single_cell.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/experiments/run_single_cell.py)、[run_atomic_unified_validation.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/experiments/run_atomic_unified_validation.py)、[budget_governor.py](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/79aa1c006e39987d88b664cb2ae75d8c1e49be5d/src/atomic_skillgraph/empirical/budget_governor.py)。

## 9. 针对性验收：无需新增收费小批

以下是本轮必须覆盖的具体场景，不要求另造一批随机 benchmark 问题。测试中固定响应、手工候选都标注为工程夹具，不计自然学习和方法收益。

| 验证组 | 固定规模与材料 | 预期结果 |
| --- | --- | --- |
| 学习投影与准入 | 本包四个被拒的真实 Train 材料各一次 | 当前题不重复；保留可学习局部操作和有效引用；最终 payload 达界；在配置的学习池内一次提取与构建可准入 |
| 来源绑定 | spreadsheet:382-10 原提案一次错误回放；选择真实写入事件的明确人工夹具一次 | 原提案完整报告独立错误；不自动偷换源；合法来源由 Host 生成起点，能走到真实局部验收 |
| 结构修复 | Office 截断响应及 Spreadsheet reasoning-only 修复各一次回放 | effective purpose／thinking／tool_choice 正确；最多一次结构修复；未恢复结构受控拒绝；不重复原题求解 |
| XLSX 文件效果 | 相同操作跨时间、改单元格、改公式、改样式四种条件 | 仅保存时间不同接受，其余拒绝；在生产 Worker 增加真实 XLSX 发布正向用例 |
| 已有资格主链 | 现有 source→Extractor→Builder→usable→Frozen→非源消费正向夹具，以及 wrong-native-operation 反例 | 保持正向可执行；错误动作不能以 accepted=True 通过；candidate 不可被测试消费 |
| 有效配置 | 当前四 benchmark 各一份正式解析结果 | 正式／已确认适配设置相同；LiveMath 成对开启；预算与模型能力字段完整 |
| 两池及恢复 | solve 到界、learning 准入／拒绝、同身份重启、预算身份改变拒绝四个确定性边界 | 收尾不侵占学习；所有 retry 计入同一池／父／总账；已完成请求不重发；改限额不能冒充原样 resume |

这些回归应禁止网络，或使用现有 StubProvider 截获请求。需要真实文件／Worker 的用例在已有依赖与容器环境执行；不调用付费模型。已有全量测试可作为仓库规定的提交门禁，但不因某个小样本没有自然 Program 而反复扩大全量回归或收费测试。

上述“待补丁验收”与本次“已经执行的复审探针”分开汇报。已有探针已证明缺陷与修复方向；只有实现提交后的对应回归通过，才能写“修复完成”。

## 10. 单 seed 的真实执行安排

### 10.1 使用同一次 Train 的启动前缀

补丁提交且上述工程回归通过后，启动新的 seed42。每个拟运行 benchmark 的前 5 个 Train 作为该 seed 的正常任务前缀，四项最多 20 个真实 Train，均计入既定训练总量。没有额外 Val、三臂或重复抽样，不设“首五题必须生成几个 Program”的门槛。

前缀仅用于确认新配置确实被真实入口采用、费用与日志正常、没有确定性合同或账本错误。通过后在同一源码、配置、输出目录和 Bank 上 resume，直接完成剩余 Train，不重新跑这五题。可以在任务边界检查后连续推进，无需每五题再召开一次方法效果验收。

如果前五题没有可程序化的真实操作，或者模型提出 no_change、局部验收失败、暂未复用，只要记录正确且属于正常方法结果，就继续该 seed。若观察到可确认的相同工程错误，例如所有已有小型写入证据都被投影丢失、Host 必填字段仍需模型猜、budget_pool 归属错误，应停止受影响 cell 修该错误，而不是继续花完整 Train 的费用验证它是否偶然消失。

本轮不以某个任意成功率作为继续门槛。学习请求已获正常机会与“学习一定成功”不是同一要求。

### 10.2 使用当前真实 CLI，不虚构开关

以下命令是实施后的用法模板。文中 `configs/atomic_unified_seed42_livemath.yaml` 是本补丁要交付的 spec 名称，当前仓库尚无该可直接启动的新文件；`<DATASETS>` 需要替换为已使用的实际 canonical 数据材料目录。

第一次执行正式 Train 的前五题：

```powershell
python -m atomic_skillgraph.experiments.run_single_cell --config configs/atomic_unified_seed42_livemath.yaml --benchmark livemath --seed 42 --model-key deepseek-v4-flash --datasets <DATASETS> --output runs/atomic_unified_v2/deepseek-v4-flash/livemath/seed42 --env-file .env --max-new-tasks 5 --stop-after-val
```

继续同一 Train 并完成只读 Val：

```powershell
python -m atomic_skillgraph.experiments.run_single_cell --config configs/atomic_unified_seed42_livemath.yaml --benchmark livemath --seed 42 --model-key deepseek-v4-flash --datasets <DATASETS> --output runs/atomic_unified_v2/deepseek-v4-flash/livemath/seed42 --env-file .env --resume --stop-after-val
```

进入同一 Frozen 的 Test：

```powershell
python -m atomic_skillgraph.experiments.run_single_cell --config configs/atomic_unified_seed42_livemath.yaml --benchmark livemath --seed 42 --model-key deepseek-v4-flash --datasets <DATASETS> --output runs/atomic_unified_v2/deepseek-v4-flash/livemath/seed42 --env-file .env --resume
```

其他 benchmark 换成各自配置、benchmark 和独立输出目录；OfficeQA 加已有的 `--corpus-root <CORPUS_ROOT>`。不使用会展开全部 seed 的矩阵命令。若希望 Train→Val→Test 一次连续执行，可在前缀工程确认后直接使用不带 stop-after-val 的 resume；保留 stop-after-val 只是分阶段观察，不是按 Val 分数筛选或反复调参。

旧 guidance-only seed42 和当前 3 题有限批的 Bank 不作为新完整 Atomic seed42 的初始 Bank。新实验使用新目录和明确新身份。旧结果完整保留，不能用后续补丁覆盖原收费版本，也不能修改旧 manifest 来绕过身份检查。

### 10.3 该 seed 希望获得的结果

工程层面，应看到每个有合法局部材料的 Train 都有明确学习决定、预算准入或精确拒绝原因，完整请求界与实际费用可对账；真有合法提案时 Builder、局部 Worker、usable、冻结和消费可以自然发生。任何一个阶段为零，都能从计数与原因解释，而非只看到最终 Program=0。

方法层面，报告有局部证据题数、提案数、Builder 数、局部验收通过数、usable 数、检索／暴露数、实际调用数、结果消费数、跨来源调用数及其分母。自然 Program 的可用性和实际消费属于所要观察的结果；没有出现时如实记录，不通过插入工程夹具填补。

准确率按固定 manifest 和 scorer 报告，所有正常失败、预算结束和工程失败保留原类别。成本拆为 Train 求解、学习、局部验证，以及 Frozen 推理；HTTP retries 与修复不得漏算，Worker 时间与模型 tokens 分开。

单次 Atomic-full 运行能给出新方法版本的准确率、成本和链路利用率。若要声称 skill 带来增益，仍需同题、同模型与推理设置、同输入和 scorer 的有效 NoSkill 对照；仅与旧异配置 seed42 或不明实现的 100% baseline 相比不足以归因。对照按既定正式实验计划安排一次，不额外加入本轮启动前的三臂收费小批。

有可比对照时，同时给整体效果和有实际技能消费的描述性子集；后者受选择影响，不单独当作因果增益。只有同口径在线平均节省为正时才估算训练摊销／break-even，当前批次不存在已证明的在线总体节省。

Val 在当前正式入口是冻结后的只读评估，代码注明 no learning selection。它不承担 Program 首次晋升、不创建新技能、不按标签修提示，不因 Val 低分而回流训练。Test 继续使用同一 Frozen；若看过 Val 后更换方法或配置，应明确作为新版本，不能把它记成同一锁定实验。

## 11. 本轮交付要求与放行边界

实现方交付一个包含上述局部修改的干净提交、正式入口实际解析的配置和有总额度分配的单 seed 启动命令，以及第 9 节针对性回归结果。现有审查包已经足以定位本轮问题，不再要求新增大审查包或重新导出完整 corpus。

本轮放行所需证据是补丁实际生效、四类问题关闭、配置和费用身份完整、现有正向链及反例仍通过。没有“自然 Program 至少几个”“准确率至少多少”“成本必须下降多少”的额外工程门槛。

当前状态应表述为：统一方法主要修复已完成；本次复审又发现少量、可本地验证的实现与配置收尾问题；完成这些补丁后可启动一个连续 seed。不能表述为“当前版本完全无问题，直接开全量”，也不能表述为“方法没显示增益，所以必须再开一轮小实验才准开始”。

本次复审没有改变线上代码或发出收费请求。文中所有新增字段、purpose、政策版本与配置文件均属于待实施规范，不能把本文件交付等同于修复已经落库。

