# AtomicSkill CF4 R2 多基准故障修复与有限验证实施文档

版本：**CF4 R2 / Revision 2，GitHub 完整生产源码复核修订版**。日期：2026 年 10 月 8 日，北京时间。适用起点：`empirical-v3.1-CF4-R1`、提交 `3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3`。

本文件是可以独立交付实施方的完整修订版，替代上一版 CF4 R2 实施文档，不能只把本轮改动清单附在旧版后继续实施。以下“新增接口、修改、验收”均是待实施要求；本次完成材料审查、GitHub 生产源码复核和零 API 探针，没有修改生产源码、重启实验、调用实验模型或执行归档中的生成程序。

### 本次仓库复核改动

| 项目 | 本版对旧文档的修正 |
|---|---|
| 已有输出校验与发布 | 明确 Worker 已有返回校验、Schema 校验和 Workspace 原子发布；复用现成位置，补上 Executor.complete 的第二次 Schema 校验 |
| 终端文件与正向归因 | 对齐隔离试用和在线提交的必需文件/变化要求；补上直接最终提交的生产者记录，保持环境终止字段原义 |
| 已生成程序与学习恢复 | 保留原已证实缺陷；把恢复时 workspace 能力视图失真落实到真实函数 |
| 第二例调度 | 复用当前 Skill 唯一 job 的当前 Program 和剩余试用槽；不再暗示现有 action=trial 能选择任意历史版本 |
| Office 收尾 | 明确实际支持的 thinking 参数、按 purpose 选择 provider/cap 的缺失接线，以及已有结果引用在无工具收尾时的有界物化 |
| 诊断入口 | D2 使用新诊断决策而非恢复已结束任务；D3/D4 增加只读 Bank 视图的明确接线，防止只清菜单却仍可直接调用 Program |
| 成本准入 | 对接 provider 内部实际 requests.post 尝试；保留历史配对 completion cap 和全部重试计数 |
| 验证范围 | 固定一轮 D0–D4、5M token 停止预算和既定请求上限不扩大；新增源码探针均为离线复核 |


## 1 结论与本轮范围

当前需要处理的不是单独一个 `KeyError`。已经证实的问题集中在四条链路：Program 输出与文件发布契约不一致；学习中断后的状态恢复和结果记录不一致；已有候选很少获得第二个真实兼容任务的验证；单答及强制收尾的空截断、阶段成本和训练曝光口径不够明确。

同时，有几项前期整改已经落实：所附 Train/Val 的成员、seed42 顺序、公共材料哈希与实际执行一致；SearchQA、LiveMath 的 Frozen 身份及 Val 只读状态通过核对；原始调用 usage、资产父子版本与训练事件已经存在。不要重新要求这些已完成事项再做一轮付费验收，也不要重新洗牌、清空 Bank 或把旧 pilot 回填成正式数据。

本轮按以下顺序推进：

1. 用现有记录完成确定性修复与离线回归。
2. 在明确继承来源的恢复分支中，零模型调用收尾 Spreadsheet 第 57 题。
3. 对两个已有候选做一次有上限的定点验证，对两个已有 Office 空收尾做单次接口验证。
4. 固定一次有无学习资产的比较，按预注册的样本、请求数和总预算结束；结果没有改善也必须结束。
5. 再按第 10 节决定哪些正式运行可以继续、哪些只保留当前方法范围。正式续跑开始后的两个任务边界检查属于正式进度，不另开一套重复训练实验。

这轮不新增 ALFWorld、DocVQA、其他模型或其他 seed 的收费实验，不运行 Test，不以大规模正式任务继续收集同一类缺陷。

## 2 快照事实与证据边界

### 2.1 上传材料对应的进度

采集窗口为 **2026-10-07 20:10:02 至 20:10:18，Asia/Shanghai**。OfficeQA 在采集时仍在运行，各文件是相近时点的独立一致性副本，不是四个 Run 的原子快照。[E01]

| Benchmark | 已提交 Train | Train 成功 | 已提交 Val | Val 成功 | 归档状态 |
|---|---:|---:|---:|---:|---|
| SearchQA | 300/300 | 250 | 24/24 | 19 | awaiting_test |
| LiveMath | 60/60 | 17 | 17/17 | 2 | awaiting_test |
| OfficeQA | 89/120 | 21 | 0/24 | — | 第 90 题 UID0105 尚未提交 |
| SpreadsheetBench | 56/200 | 43 | 0/20 | — | 第 57 题学习试用异常，execution_failed |

OfficeQA 的 75/120 是 17:16 的旧进度，不能代表此包。第 90 题虽已有 `task_execution_finished` checkpoint，仍不能并入已提交分母。Spreadsheet 的第 57 条 episode 是 `success=null` 的异常记录，也不能当作第 57 道已完成失败题。

四个 Run 的源码均指向上述清洁提交，归档 `tracked_diff` 与 `status` 为空。主审查包与专项包的 308 个共有内容文件完全一致；只有两个打包索引文件不同。[E02]

### 2.2 数据与冻结检查的实际结论

所附公开 Train/Val 数量分别为 SearchQA 300/24、LiveMath 60/17、OfficeQA 120/24、SpreadsheetBench 200/20；各自无重复、无交集。四个 Train 执行列表都精确等于 `random.Random(42).shuffle(固定公开 Train 列表)`，已完成任务是其正确前缀。SearchQA、LiveMath 的 Val 顺序与公共清单相同，`readonly=true`，Frozen 文件与树哈希一致。[E03]

沿用既定完整划分：SearchQA 300/24/1400；SpreadsheetBench 200/20/180；OfficeQA 120/24/102；LiveMath 60/17/100；ALFWorld 120/24/134；DocVQA 沿用已接受的 180/22/201/131 口径。本包没有 Test 正文，本次不读取或重算 Test 个体答案，也不把“未读取 Test”说成划分有问题。

### 2.3 GitHub 源码身份与实际覆盖

本轮通过用户指定的 GitHub 连接核对 `YU-S3/AtomicSkill-ToolGraph_v3` 的 main，锁定提交 `3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3`，提交说明为 Record CF4-R1 production verification and launch readiness，提交时间为 2026-10-07 04:45:24 北京时间。main 与四个 Run 的归档提交相同。[E21]

共取得并逐个校验 **103 个源码、测试、配置与相关说明文件的 Git blob SHA**。其中包含仓库全部 **64 个生产 Python 文件**（src 与根 experiments 入口）；按真实 `code_identity()` 的源码散列算法重算，得到 `3ab48c8b3699775f6fccf351e159725d79b9c43c2d9bc2a3f52709a061b13391`，与归档完全一致。原包八个生产模块逐字节一致。这里不是将某个最新版本的实现套用到旧运行。

重点沿真实调用链补读了 `empirical/executor.py`、`contracts.py`、`program_worker.py`、`workspace.py`、`model_view.py`、`planner.py`、`prompts.py`、`task_context.py`，以及 agents/provider、protocol、usage，simple_protocol、benchmark contracts/scorers 和正式入口。旧版“完整 Worker/Executor/provider 未附，待实现方核对”的悬空接点已被本版的具体位置替代。

没有读取 GitHub Test 个体材料，也没有重新跑任何 benchmark。下载完整生产源码不等于对仓库所有历史报告、所有环境和真实 Docker 部署作全面验收；本次代码核查与测试范围是当前四 Run 及其共享的学习、执行、提交、恢复和对照链路。

### 2.4 当前项目的真实生产链路

| 位置 | 当前实际行为 | 本轮应保持的边界 |
|---|---|---|
| run_single_cell → run_formal.campaign → run_empirical.run | 选定 cell；固定 Train 顺序；Train 完成后冻结，再只读 Val/Test | 沿用 canonical authority，部分 Train 暂停不进入评估 |
| EmpiricalSystem.run_task | single_answer 直接一次 solver；tool_loop 经 Planner 和 Executor；最终统一 submit/evaluate 后再学习 | QA 不增加规划求解；独立 scorer 不是模型工具 |
| Planner / Bank.planning_cards | 读取实际 Skill/Workflow 及 usable Program 信息，选择或组合当前题计划；无相关资产时返回 Dynamic | 参考 guidance 不授予执行绑定；实际绑定为 dynamic/skill/program |
| Executor → ProgramWorker → Broker | 参数 ready 的 usable Program 自动执行；Broker 是原生工具桥；支持局部 patch、一次整图重规划和一次剩余 Dynamic | 保留自动接管，不增加逐节点模型批准 |
| Learner → realization job → test_program → Bank.record | Train 经验生成/复用 Skill，构建 Program 与 Implementation，固定真实 case 试用；同版本两个不同物理任务 positive 才 usable | 复用已存在的二次试用机制，不额外发明整图资格层 |
| Bank.freeze | 去掉候选 Program、对应 Implementation、job 与 Train cases，保留 usable 与指导/工作流；评估 Bank 只读 | 候选数量与 job done 不是方法效果；诊断快照不冒充完整正式 Frozen |

当前资产类型为 `skill/implementation/program/workflow`，唯一生产入口是 `skillcompiler.empirical.v1`。不要按已移除旧实现中的类名、证书或其他历史四层组件写补丁。[E22]

## 3 不得改变的已有原则

以下要求来自此前最终约定，并结合本次源码解释其实际含义。

- Planner 编排的是本题待检验策略。局部错误选择、参数、引用和交接可以修正；它们不能因为写进了尚未执行的计划就强制覆盖真实环境。
- 已验证 Skill/Program 保留程序优先、连续执行、自动接续与整题接管。不能借输出检查改回“每节点都问一次 LLM 是否批准”。
- 局部解绑与参数纠正不占整图重规划额度；整图重规划用尽后仍保留一次从真实状态退出失效计划、进入剩余任务 Dynamic 的退路。
- 新组合不需要新增整图证书、witness/owner 证明链或两题预认证。**既有 Program 版本晋升规则仍保留：同一版本、两个不同物理 Train 任务的独立 positive 才能成为 usable。**不能混淆这两件事。
- 原任务解对、Program 自报 `verification`、某个 job 为 `done`，都不能替代独立的 Program 正向证据；不同源码版本各成功一次不能合并晋升。
- 学习和资格更新只使用 Train。Val/Frozen/Test 不修改 Bank，不按 Val 分数选择程序版本、Bank、阈值或预算；Test 继续隔离。
- single_answer 每题只作一次正常求解，不追加“截断再解一次”的隐形多次尝试。
- 不修改独立评分器以掩盖错误，不向模型提供 evaluator 的正确答案；模型普通答错不被包装成工程故障，也不成为无限重试理由。
- 不热改仍在执行的共享源码；不删旧错误、伪造旧时间戳、替换旧 commit/hash 或把恢复继承的 Bank 写成空 Bank 独立训练。
- 每个修改必须有“已有证据 → 文件/函数 → 行为变化 → 失败边界 → 定点验收”。不增加与已观察问题无关的新工程门槛。

## 4 第 57 题的真实控制流与恢复目标

### 4.1 已经发生的事实

任务为 `spreadsheet:51-12`，0-based 执行位置 56，attempt 1。`state.json` 保存阶段 `task_execution_finished`，原评分 `hard=true, raw_score=1.0, case_results=[true]`。原 `solution.py`、`case1_result.xlsx` 和 sealed manifest 已归档。该题归档评分只含一个 case，不扩写成多个变体全部通过。[E04]

后续 Extractor 新建的 Skill 声明 `result_role=final_answer`，其 `output_schema.required` 却是 `output_file/count/verification`，没有 `answer`。Builder 生成的 Program 继承该角色和 Schema，源码返回文件、计数和发布清单，仍没有 `answer`。

真实调用顺序是：原题执行和独立评分完成 → 保存解题 checkpoint → 保存 Train case、Skill、job、Program、Implementation → 隔离试用 → `submission_ready` 因工作区文件齐全返回 true → `system.py:558` 按 `final_answer` 索引 `outputs['answer']` → KeyError。[E05]

崩溃时 Bank 已有上述部分写入；job 为 `kind=build,state=ready,program_id已存在,generation_count=1`。trial 的完整结果、临时工作区和输出没有持久化，甚至没有 trial 子目录。**可恢复的是原解题成功事实，不能声称已原样恢复那次丢失的试用。**

该题已记录 12 个请求，共 **397,659 tokens**：原解题 313,263，Extractor 54,889，Builder 29,507。全部原 call/usage ID 必须保留一次；不能为恢复再付一次原解题费用，也不能把已付学习费用归零。

### 4.2 推荐的最低成本收尾

在通过第 5 节恢复修复之后，将已有成功解题提交为最终任务结果，学习收尾为 `failed_engineering`，错误 job 置 `deferred`，通过有审计原因的 Bank 隔离方法禁用错误候选，不给 trial positive。这样可形成 **57 completed、44 solve successes**，同时保留学习异常及历史费用。

这一步要求 **0 次 Planner、0 次 Runtime 重解、0 次 Extractor、0 次 Builder**。它不是假装原训练完整成功，而是分别保存“解题已成功”和“学习未成功”。本次审查没有实际执行此收尾，原 Run 仍为 56/43。

## 5 必须实施的工程补丁

### R01 统一 Program 输出和文件发布契约

**证据和位置：**`learner.py:60–115,149–169,349–357`；`system.py:527–568`；`benchmarks.py:242–251`。[E05]

新增窄模块 `empirical/program_submission.py`，共用现成的 `validate_schema_instance` 与 Adapter 公开合同。当前 `Capabilities.final_submission_kind` 已有 environment/text/files/single_answer，**publication_contract 还没有现成字段或方法，必须显式新增**：在 `FileAdapter` 提供公开的 `program_submission_contract()`，说明是否支持文件发布、保留字段类型、终端必需发布名；`SpreadsheetAdapter` 返回 solution.py/case1_result.xlsx，文本类不要求这两个文件，非文件 Adapter 禁止文件发布。通过同一公共对象提供给声明检查、Worker、Executor 和学习/构建材料，核心层不硬编码 benchmark 名称。该方法与下面的接口均为待新增 API：

```python
def validate_program_declaration(skill_or_program, capabilities):
    """确定性检查 result_role、输出结构与 Adapter 最终提交类型。"""

def effective_output_schema(declared_schema, *, result_role, publication_contract):
    """生成用于 Worker 校验的 Schema；纳入系统定义的发布字段。"""

def prepare_program_submission(adapter, program, result, *, workspace_before):
    """返回 ready / not_applicable / contract_error 及已验证 payload。"""
```

返回对象至少有 `status`、`payload`、`error_code`、`repair_target`、`diagnostics`。`repair_target` 只用于区分声明修复、源码修复和宿主错误，不是能力接管的新证明体系。

| Program 角色 | 必须满足 | 不满足时 |
|---|---|---|
| final_answer | Adapter 最终支持 text/single_answer；outputs 为 object；answer 是符合原提交合同的非空字符串 | 返回明确合同错误；不读缺失字段、不调用评分器 |
| final_files | Adapter 最终提交为 files；发布名合法；必需文件齐全；有本次 invocation 的发布证据 | 返回文件发布错误并进入有界修复 |
| intermediate | 依原业务输出合同和局部检查；允许为后续节点提供数据 | 保留原真实 continuation/消费归因，不强制生成最终答案或两个文件 |

实施分为三处：

1. `Learner.validate_learning_proposal()` 在任何新 Skill/job 入 Bank 之前检查声明。Extractor 材料提供 Adapter 的公开 `final_submission_kind` 和必需发布项。只对可执行声明约束 terminal role；不要把 guidance_only 的历史空 object Schema 误判成 Program。
2. `_realize()` 组装 candidate 后、`bank.put('program', ...)` 前再检查。错误若来自 Skill 的 role/Schema，反馈给声明修复路径，至多消费既有一次结构修复；不能让只能改 source 的 Builder 反复尝试修 role。
3. `test_program()` 与 `Executor.run()` 的 Program 终端提交分支共用 `prepare_program_submission()`。在最终提交路径，只有 `ready` 才执行 `adapter.submit(payload)` 和原评分器；合法 intermediate 的独立局部检查和后续消费仍走原分支。校验成功后读取必需字段；不得用 `.get('answer','')` 掩盖错误。

**文件字段采用一个确定的兼容方案：**保留当前 `outputs.files` / `outputs.deleted_files` 传输形式，由宿主生成 effective Schema，将两个保留字段按统一类型并入允许字段；其余业务字段继续遵守原 `additionalProperties`。函数只处理 `result.outputs` 的 Schema，并返回深拷贝。两个保留字段均为相对发布名字符串数组，deleted_files 可缺省；final_files 缺 files 由发布合同明确判失败，intermediate 不被要求生成最终文件。是否允许这些字段由 Adapter 的公开 `publication_contract` 决定，不能只看 result_role：`FileAdapter._python()` 创建的临时 Program 也需要合法发布，即使未显式标 final_files。

如果业务 Schema 自己声明了不兼容的同名字段，则拒绝声明。对项目实际支持的 Schema 子集实现规范化；存在 allOf/oneOf 等组合约束时不能只加顶层 properties 就声称全部兼容。新 Skill 必须在 `_proposal_skill()` 计算 ID **之前**完成 canonical Schema 规范化；新 Program 必须在 program_digest 之前使用同一规范声明，保持 `$new`、job 与 Implementation 关联一致。旧资产只改变宿主校验视图，记录兼容合同版本，不回写原 payload 或 ID；源码或 solution 模板一旦改动，必须生成新 Program ID。禁止通过全局 `additionalProperties=true` 解决问题。

**已核实的调用位置与改法：**

- `program_worker.py:225–241` 已按返回 envelope → status=ok 的原输出 Schema → `Workspace.publish()` 执行；失败时丢弃 staging。把第 228 行改接 effective Schema，并在合适的执行边界加入角色/返回合同检查。不能写成“Worker 原本没有输出检查”。
- `workspace.py:40–92` 已校验发布路径、文件存在、未声明修改、删除项与 symlink，并通过版本目录和原子 manifest 指针发布。继续复用，不重写一套发布/回滚框架。结构、Schema 和发布名/路径检查仍在有效 manifest 更新之前；终端适用性与正向资格检查使用已经由宿主核验的发布结果，二者不要混为一层。
- **`executor.py:128–150` 的 `complete()` 还会按节点所绑定 Skill/Program 的原 `output_schema` 二次校验。** 这里必须使用对应绑定接口的同一 effective Schema，并保留实际 Program 自身输出校验。否则合法 files 已通过 Worker，仍会在第 139 行变成 HandoffError。不能用只改 Worker 的 fixture 作为完整验收。
- `complete()` 的 handoff aliases、required fields、合法业务输出约束继续保留。由 Program 生产的结果必须按真实 producer 及绑定接口选择兼容视图；普通 Dynamic 不凭参考 Skill 获得新的输出限制，显式中间业务 Schema 不被全局放宽。

`files` 仍只是发布请求，不能因模型自报存在就通过。第 7665 题“删除 files 后结构通过但无产物”应覆盖到 Worker→节点完成→最终提交，而非只测 JSON 校验。[E23]

返回结构和角色合同还要在 `terminal_by_program`、`local_check=passed` 等任何 positive 归因前校验，不能只替换第 558 行所在的末尾分支。final_files 漏发布不能被 local passed 覆盖；intermediate 的合法局部检查、真实 continuation 和环境终态证据仍保留，不将“最终提交 ready”变成所有中间能力正向归因的新条件。

`workspace_before` 和本次 Worker 的宿主发布记录必须进入文件 Program 的准备提交检查。现 Worker 的 `result.workspace` 是完整 manifest，只有 version/outputs/hashes/content_hash；`Workspace.publish()` 在 no-op 时直接返回旧 manifest，**它不是本次贡献证明**。

最小补充 `publication_receipt`：由 Worker/Workspace 宿主填写 invocation_id、经 publisher 核验的 declared/deleted、发布前后 manifest/version/必需文件 hashes，并与该次 attempt 绑定；不信任模型回传的同名字段。

**保留现有隔离终端规则，不静默放宽：** terminal final_files 的 files 声明须覆盖 Adapter 的必需发布名，最终 manifest 必须齐全，且至少一个必需文件 hash 相比 invocation 前变化；不是要求两个文件字节都变化。不能仅比较整个 workspace content_hash，因为只写 debug 文件也会使其变化。显式重申同内容文件可以是合法 publication，但在该终端试用规则下不能取得新的 positive。

合法 Dynamic/intermediate 的增量子集发布和 no-op 仍允许；它们没有通过 terminal 资格检查，不等于发布器本身出错。在线 `Executor.submission()` 第 83–84 行及终端分支第 265–268 行要读取相同的真实 attempt/receipt，不再遗漏 previous_workspace 后仅凭旧文件齐全就提交或归因。[E24]

**兼容边界：**不要直接将 `SpreadsheetAdapter.submission_ready(result_role!='final_files')` 改成 false。这个方法默认角色是 intermediate，完整 Executor 的普通 Dynamic 完成路径可能使用它。先执行全仓 `rg 'submission_ready|result_role|prepare_program_submission' src tests` 列出调用点，只收紧 Program 提交通路，并回归原动态文件提交。

### R02 使漏发布文件成为可修复失败，保存每次试用状态

**证据：**Spreadsheet 11 次 `status=ok,outcome=normal` 中，7 次只发布 xlsx、3 次未发布任何文件，只有 1 次真正完整提交后评分失败。现 `_realize()` 只对 `execution_failure` 修复，因此 10 次发布缺陷没有进入应有修复通路。[E06]

修改 `system.test_program()` 与 `learner._realize()`：

- 声明 final_files 且 status=ok，但缺必需文件/本次发布记录：`outcome=execution_failure`、`basis=null`、明确 `program_publication_incomplete`，将缺失发布项送入原一次 Builder repair。
- 声明 final_answer 却缺字段/类型错误：同样形成结构化失败；不抛 KeyError。
- 完整提交但独立评分为 false：保留普通任务失败，附 `evaluation_status=failed`。不要改成工程崩溃，也不要自动无限重做。
- 合法 `blocked/needs_input/not_found` 和不适用输入，保留独立状态。不能为了凑 usable 把它们改成 positive。
- datetime 非 JSON、非法输出类型、导入失败等可归因到生成程序的错误，由 Worker/合同边界记录为已知执行失败；不要全局捕获所有 Exception 后继续。
- 评分器、文件系统、未知宿主异常：在清理临时目录前保存 exception、已有 worker result、工具事件和公共产物索引，再重新抛出以停止该任务；不能把未知基础设施状态当普通模型答错。

在隔离 Adapter reset、prefix 重建和 Worker 执行之前，先持久化 `trial_started` 与 trial_execution_id；prefix 本身可能发生真实调用，不能等到进入 Worker 才记开始。即使程序不调用 ctx.call，也必须有开始标记。成功执行后先保存 worker_finished 和验证所需产物/摘要；评分与归因结束后保存完整 record，再幂等写 Bank，最后清理临时目录。

真实 Worker 已在子进程处理生成程序异常，并在宿主 execute 的多个 catch 中把 ValueError/OSError/RuntimeError/TypeError/KeyError 转成 execution_error。R02 的分类不能只包住 test_program：对生成程序导入、非 JSON 返回、声明文件缺失/路径非法，继续给明确的生成/合同失败；对宿主 staging、原子版本提交、Docker 控制及未知副作用失败，保存实际阶段与基础设施身份并向外传播，不能消耗 Builder 修复来掩盖。不得把所有 OSError 统一改成宿主故障，合法路径检查中的“声明文件不存在”仍是已知产物错误。[E23、E24]

```text
trial_started → worker_finished → trial_finished(record) → bank_recorded
```

逻辑 `trial_id` 继续绑定 Program 版本、物理任务和固定 binding，用于证据去重；另设 `trial_execution_id` 标识一次真正执行。缓存读取不增加执行；在隔离环境重新执行必须有新的 execution ID 和真实费用记录，不能被旧 exception 的同 ID 去重。只有验证所需文件已封存、哈希可核验，才能从 worker_finished 继续评分；只有摘要而文件已丢失时，标记该执行不可恢复，不授予 positive，也不声称已恢复全部阶段。

### R02.1 补齐“直接最终提交”的真实 Program 归因

完整源码确认一个旧文档未能确认的缺口：`Executor` 第 265–268 行允许用 `last_output` 直接完成 text/files；第 477–483 行的消费归因主要依据显式 plan.outputs 引用和后续消费者。单 Program 已自动完成、plan.outputs 为空时，可能得到 `plan_submitted`，却仍为 outputs_consumed=false、terminal_by_program=false。后者在文件/文本 Adapter 上通常正常，因为 Broker 并没有发生环境 done 转换。此时 `learn_trace:449–452` 会漏掉真实的直接提交成功。[E25]

另一个方向也必须堵住：旧完整文件留在工作区、Program 本次 files=[]，在线路径仍可能 plan_submitted；显式 outputs 引用甚至会使 consumed=true。这不能靠给所有终端程序补一个 true 修复。

最小修改如下：

1. 保持 `terminal_by_program` 的原义：本次 Program 使环境由未终止变为终止。新增 `submission_producer_attempt_id`，并在对应 attempt 上记录 `submission_by_program`，用于文本/文件直接最终提交。
2. 在真实选定最终提交 payload 的分支，依据已有 `producers`、节点来源和实际字段引用设置来源；last_output 路径显式保留其 producer。不能仅凭结果值相等、Program 自报或“最后调用过哪个 Program”反推来源。
3. 文本若被 Agent 重新提交/重写，不自动归给旧 Program；文件来源须绑定已核验 publication receipt 和最终封存所用版本/必需文件 hashes。后续写入或覆盖必须重新判定，不能沿用旧归因。
4. 保存新来源字段到 `Executor.save()`、executor_finished、Trace/episode_result；普通恢复不能丢失。最终 `adapter.submit/evaluate` 后再按真实 sealed payload/文件 hashes 核对来源。
5. 统一记录原 `worker_status` 与规范后的 `output_contract_status=valid|invalid|not_checked`；新调用只有 valid 才能参加任何 positive 归因。合同失败不能仅将 outcome 改为 execution_failure 后保留无保护的 status=ok：现 learn_trace 会在后续 Agent 解题成功时再次把它覆盖成 positive。已知合同失败将规范结果置 execution_error，并保留原 Worker 返回、失败原因与发布记录；不得覆盖原始证据。
6. `learn_trace()` 的 task_outcome 重判、Executor.invoke 的 local positive、test_program 的环境终态/local check/final submission/continuation 各分支，都复用该合同门。直接提交还要求对应真实 producer、独立 score.hard=true、原 local_check 规则允许。合法 intermediate 不被要求生成最终文件；它先通过自身结构/角色合同，再按原局部或消费规则计证。
7. 新来源和合同状态只用于新调用；不重写旧 Bank 的历史 positive 或把旧缺字段记录自动补成 valid。Val 只记录诊断，不更新资格。

本次零 API fixture 证明此控制流可触发，不证明旧四 Run 的六条 online attempts 都遇到了它。**不追改旧 false、不给历史记录补 positive。** 新代码必须同时通过“有效直接提交可归因”和“Agent 覆盖、旧文件、no-op、仅 debug 变化不可冒领”的反例。它是记账缺陷修复，不是新的接管批准或资格证明层。

### R03 修复已生成 Program 的恢复与学习样本池恢复

**已复现：**第 57 题有 program_id，但 `_realize()` 第 330 行只在 `job.kind=='trial'` 时加载它，恢复会请求第二次 Builder generation；既有 `_0_0` 缓存不会命中新的 `_0_1`。[E07]

将 job 中的“请求意图”和“已完成执行阶段”分开。最小增加 `program_ready` / `next_trial` 等阶段字段即可，不重写全套状态机。Program 与 Implementation 落库后，立即保存关联 program_id、当前 generation、待试 case 和阶段。恢复优先加载已持久化 Program，除非有明确的 source repair 指令；不能仅因 `kind=build` 再生成一次。

`_receive()` 的缓存响应只有在请求语义和所属状态仍匹配时复用。改了 Schema、代码/协议版本或 binding 的响应不得伪装成原调用。已存 Program 的复用属于恢复已完成阶段，不依赖在新目录重新计算旧 HTTP response_key。

另一个必须修复的实际控制流：`run_empirical.py:130–135` 把所有 completed trace 装进 `learner.cases`；`Learner.learn:121–124` 又把它们写入 Bank。Spreadsheet 原 Bank 只有 **42 个 train_cases，含第 57 题**，前 56 个 completed 中有 15 个预算耗尽、原本未进入学习。如果按当前逻辑恢复，会将这 15 题静默补入学习池，而且采用缺少 `submission/termination_reason/program_calls` 的另一种 experience 格式。[E08]

**修改：**恢复学习池以继承的 `Bank.train_cases()` 为唯一已发生学习暴露来源，保持其成员和内容；completed 列表只用于汇总和跳过已完成任务。不得从所有 completed trace 自动补灌。确需重建缺失记录时，必须是显式恢复动作、复用 `Learner._experience()`，列出新增暴露来源，不能隐形改变已运行协议。

如果选择恢复尚未完成的学习而非第 4.2 节的零成本收尾，还需显式初始化当前任务的公共 Adapter/TaskContext，并按 checkpoint 恢复可用结果引用。完整源码给出了具体原因：`Learner._view:23–25` 可以惰性创建 TaskContext，不能声称必然 AttributeError；但是 fresh FileAdapter 在 reset 前没有 workspace，`Learner._realize:315–317` 传入 None，`program_permission_view:43` 就把工作区报告为 unavailable/root=None，向 Builder 提供错误能力视图。[E26]

第 57 题继续采用专用零调用收尾，完全跳过这一分支。对将来确需恢复的学习，仅恢复已封存且可核验的公开材料/结果引用，再创建独立学习/试用上下文；不通过 reset 原工作区覆盖已完成产物，也不在本轮扩展为任意环境重放器。隔离试用始终使用自己的 `adapter.reset(trial_task)`。

### R04 分离解题结果 学习结果和最终 episode

修改 `system.run_task/learn_trace`、`FormalLog.task_error/end_task`、`run_empirical.run`。

Trace 增加明确字段：

```json
{
  "solve_status": "completed",
  "learning_status": "failed_engineering",
  "learning_error": {"code": "...", "stage": "...", "repair_target": "..."}
}
```

`learning_status` 的允许值为 completed、rejected、failed_engineering、not_started。评分成功保存后，学习失败不能覆盖 `score` 或把原解题改成空提交。对已知合同拒绝记录 rejected/failed 的学习终态；对未知工程错误保存学习失败 checkpoint 并停止，保留解题结果供显式恢复。

异常事件只追加到 errors/阶段事件，不提前占用最终 episode 的幂等 ID。当前第 57 题已由 `task_error()` 用 `attempt_id` 写了 null episode，`end_task()` 同 ID 会被 `emit()` 丢弃；这已做零 API 复现。[E09]

对旧 null 行采用追加的 `episode_revision/recovery_resolution` 与 `supersedes_event_id`，由一个确定性的 reducer 生成终态投影。不要删除旧行、随机增加 task attempt 来绕过去重，也不要让多个 JSONL 行被直接当多道题。最终应满足：

- 每个物理任务/正式 attempt 只有一个权威终态投影。
- `run.sqlite3`、Trace、终态 episode、summary 的完成数和评分一致。
- 恢复事件和旧异常仍可追溯；同一真实 HTTP 调用只计费一次。
- training 的 learning_start 必有 completed/rejected/exception 结束事件；当前未完成学习不能因 result=None 消失。

### R05 用显式恢复子 Run 保留版本身份

现 `run_formal.py:71–87` 和 `FormalLog.__init__`、`run_empirical.py:103–109` 锁定完整源码与配置。补丁后原目录普通 `--resume` 应继续拒绝身份不匹配；这不是要删除的障碍。[E10]

新增 `experiments/recover_empirical.py`，分为只读 dry-run 与创建恢复子 Run 两步。只针对已证实的完成解题/学习失败边界实现，不在本轮泛化为任意环境状态迁移器。

恢复清单至少记录：`parent_run_id`、旧/新 source_commit 与 source_sha256、parent_bank_digest、继承完成任务、待收尾任务、checkpoint 和 sealed bundle 哈希、原始 call IDs、路径映射、模型/材料/评分/预算身份、恢复动作与时间。

流程要求：

1. 验证来源不可写、任务与文件哈希、前缀顺序、原调用去重、Test 未开始；输出拟继承内容和第 57 题收尾报告。
2. 复制到新目录；不修改原 Run。已完成历史仍标注原版本，子 Run 是 recovery continuation，不能标成从空 Bank 独立训练。
3. 按第 4.2 节走“已评分任务收尾”专用路径，不调用原 `run_task()` 的学习恢复分支；不 reset 原工作区、不重新 evaluate、不调用 agent。job deferred，错误候选被隔离，不产生虚构试用结果。
4. 复用共同的 trace/episode 最终化函数，原 12 个请求和 397,659 tokens 保留一次；继承费用与新增费用分别列出。
5. 正常加载子 Run 后普通恢复仍做新身份校验；不能允许任意源版本、任意 Bank 通过一个泛化 `ignore_hash` 开关。

新目录中的路径只能做带哈希的定位重映射；不能因为文件路径变化而重发已完成请求。`agent()` 的 scope 包含 checkpoint root，响应键还包含 implementation revision；本次收尾无需迁移或重写这些键。旧响应作为原始证据保留，新任务生成自己的新请求标识。丢失的 trial 不恢复为成功，不复执行未知副作用。

候选隔离必须落到代码：新增 `Bank.quarantine_program(program_id, reason, recovery_id)`，在一笔本地事务中将该版本设为现有 `disabled` 状态并保存独立的合同隔离事件。不要伪造两条 execution_failure 以触发禁用。`routes/program_options` 继续排除 disabled；对应非法 Skill 的 job 不得因 `_merge_request` 收到新样例而重新激活，必须先产生并验证新的声明版本。只设 job deferred、保留 candidate 不够，因为在线 `allow_candidate=True` 原本允许提供 candidate。

恢复收尾事件标记 `is_recovery=true,new_model_calls=0,new_train_examples_consumed=0`，不能创建第二次 trajectory/learning 曝光。缺失的原 scoring_audit 标为不可用并保留已存 score 的来源，不能伪造原评分器返回详情。第 58 题起使用新建的 TaskContext，不继承第 57 题 scope 或结果引用。

恢复工具先写独立 staging 子目录和确定的 recovery ID，再发布成可运行子 Run；重复运行同一恢复请求要么幂等完成，要么对不一致来源明确拒绝。覆盖复制后、提交前、提交后三个中断点即可，不需要引入跨数据库的重型事务框架。

增加 `max_new_tasks` 或 `stop_at_task_id` 的任务边界控制，并由 `run_formal.campaign()` 透传。现 `stop_after_tasks=N` 包含历史 completed 数，必须保持其含义或明确弃用；继承 57 题后不能用 `stop_after_tasks=2` 表示再跑两题。`newly_completed` 只统计本次真正新增求解并提交的任务，不包括继承历史和离线收尾。Train 返回 `complete=false` 时，campaign 必须直接返回 `stopped_after_task_boundary`，不能继续 Val、生成正式 Frozen 或标记 awaiting_test。暂停只在安全边界进行，不把有在途副作用的中断当安全暂停。原 `STOP_AFTER_TASK` 控制文件不能不加辨别地复制并删除。

## 6 方法链路的确定缺口与有限修改

### 6.1 当前确实没有 usable Program

| 指标 | OfficeQA | SpreadsheetBench |
|---|---:|---:|
| Skill | 80 | 39 |
| Program | 62 | 42 |
| usable | 0 | 0 |
| 已持久化隔离试用 | 63 | 42 |
| positive / normal / execution_failure | 27 / 29 / 7 | 25 / 13 / 4 |
| 仅有一个物理任务 positive 的 Program | 27 | 25 |
| realization job | 72 | 39 |
| 只有一个 case binding 的 job | 70 | 38 |
| 有两个 case binding 的 job | 1 | 1 |

OfficeQA 另有一个零 binding job。Spreadsheet 第 57 题另有 trial exception 事件但没有 Bank attempt，不能加进 42 个已持久化结果。[E11]

两边均没有同版本在两个不同任务上的 positive。Office 唯一 done job 的两个结果都是 not_found；Spreadsheet 唯一 done job 是一题评分失败、一题成功。`done` 表示试用槽位已用，不表示能力可用。

两边 online attempts 各只有 3 条，全部 `outputs_consumed=false,terminal_by_program=false`，没有在线 positive。Spreadsheet 3 条还来自同一题、同一程序的三个节点。不能据此证明工具接管降本已经发生。本轮完整源码虽确认新调用存在直接提交归因缺口，仍没有足够历史执行证据将这六条 false 改成 true。

### R06 在既有单 job 调度内优先验证当前 Program 的第二例

现代码已经有两槽 case_bindings、物理任务去重、固定 prefix 校验和第二例合并；应改的是候选优先级与材料，不重新实现这些机制。`prompts.LEARNING:25–29` 的 realization_request 只有 skill_id/action/case_bindings，没有 program_id；`_resolve_realization_request:78–83` 和 `_realize:330` 只取该 Skill 唯一 job 当前的 program_id。因此不能写成“把任意历史版本列入提示词，就能用现有 action=trial 直接选它”。[E27]

本轮采用最小范围：

1. 在 `Learner.learn:128–157` 同一次 Extractor 的 related/pending 材料中，优先提供 **Skill 对应唯一 job 的当前 Program**：Skill/Implementation 关联一致、state=candidate、同版本已有一个物理 Train 任务的合格 positive，且还有未试的固定 case 或未填第二槽。已用完两槽的 done job、disabled/隔离版本不进入该集合；deferred 原因与原预算限制不能由排序解除。提供实际 Skill/job/Program ID、已用物理任务、固定 bindings、公开输入 Schema、独立结果与结构化失败。
2. 复用当前一次最多调度一个 job、trial 优先于 repair/build 的控制流。related 总条目沿用 `Bank.retrieve(limit=8)` 的上限；优先集合按当前公开 task.goal 与 Skill 的词项重叠降序、job ID 升序排列，余量补普通 related；completed_train_cases 仍沿用现有最多 8 项。排序只决定展示，不证明适用性。不新增 Extractor 或资格评审请求。历史正向但已经不由该 job 指向的版本，不混入可由现有请求选择的集合。
3. Extractor 仅在公共信息确实可绑定时输出已有 `action=trial`。保留 validate_learning_proposal、_preflight_binding、_merge_request 及 _realize 的槽位/来源/已执行绑定校验；分派前核对实际 Program ID 仍为该 job 当前版本，只试尚未执行的不同物理 case。版本 ID 用于材料审计与宿主一致性检查，不擅自加进现有严格 request Schema。
4. 已持久化当前 Program 必须按 R03 直接加载，不能又触发 Builder。尚未填满两槽不代表允许覆盖已执行 binding；预算/deferred/隔离限制照常生效。隔离版本必须先有合法声明修订，不能用新样例悄悄重新激活。
5. 第二例暴露真实源码问题时才消费原有一次 source repair；新版本两题重新计证，不继承旧版本 positive。正常正式学习沿用已定义的真实新经验触发 epoch 规则；诊断不清零旧 job 计数来伪装该触发已经发生。
6. 记录建议候选、适用/拒绝理由、实际 job.program_id、绑定来源、独立结果与费用。缺兼容第二题时保留 candidate/waiting_example，不扩样追正结果。

D1 中旧 SUMIF fc9a 已不是该 job 当前版本，必须使用显式 Program ID 的独立诊断入口，不能作为上述自动历史版本选择的证据。本轮不新增全 Bank 历史版本路由及按版本的 job 账本；那将是另一项方法修改。

### 6.2 两个定点候选的具体实施对象

#### Spreadsheet SUMIF

Skill：`skill_96aa5b3454600e8c5656c7b39bb8c03688a7edac58ad2bc7e685951f26554061`。

旧 Program：`program_fc9a9078cbe86007e2f23a20044d72a45d921021f709780e8c4d9df8a76ea800`，已在 `spreadsheet:36764` 获 positive。新 Program：`program_af31f99c56cdb4b2b787f8fc9eb693e13b765a9d1ddd336e08ddd2ad443be29e`，在 `spreadsheet:46167` 成功，却在旧 36764 评分失败。[E12]

旧题公开要求 key 列 A、criterion A2、target C2；新版写成 key 列 S、criterion C2。具体错误公式和绑定保存在已有 trial 中，是实际跨题修订退化，不是仅凭源码猜测。

固定步骤：先让旧 fc9a 在此前未试过的 46167 做 **一次**纯 Program trial，参数只取公开题意和输入工作簿；不输入标准答案。必须显式将旧 Program ID 传给 `test_program()`：该 Skill 当前 job 已指向 af31，不能调用 `_realize(kind='trial')` 然后误测当前新版本。若失败，至多一次正常 Builder 修订，再在 36764 和 46167 各做一次新版本 trial。原已执行 binding 不改写。

同时核查生成的 `solution.py` 是否保留运行时公共参数：旧程序 MAIN_SRC 的调用 `process_workbook(_in,_out,{})` 丢掉 opts，是独立变体执行必须覆盖的接点。允许持久化本题合法列名/范围等绑定，不能持久化 evaluator 输出或当前工作簿全部数据当通用算法。

#### Office OASI 行匹配

候选：`program_a6f6161109ed6fa341806fa10f60bae09f85590474d7f11baca869abfb5fc2f0`；已绑定 Train `officeqa:UID0060` 与 `officeqa:UID0059`，两次 not_found。[E13]

源码匹配后缀 `appropriationstoffederaloldageandsurvivorsinsurancetrustfund` 多了一个 f。用原 234 条 grep 结果作纯字符串核对，错误后缀匹配 0 条，正确规范化后缀匹配 14 条。现源码还没有消费声明的 `line_item_label/table_title_pattern`。

把真实失败和公共输入交给一次正常 Builder repair，要求从 `line_item_label` 等当前参数派生匹配，并明确处理表标题和单位；不得由审查者手写一个答案程序再计入自主学习成果。修订后只在上述两个 Train 任务各试一次。

两题分别涉及 appropriations 与 expenditure transfers。实施者先做零模型的公共语义/表列核对：若实际不是同一可参数化能力，应返回不适用并停止该候选，不能为凑两个 positive 强制映射同一会计列。本轮不接着寻找更容易成功的替代候选。

两个候选的付费修订上限为 **2 次 Builder HTTP 调用**，纯程序试用合计最多 **5 次**。这两个当前 job 都已 `generation_count=2,repair_used=true`；新修订必须创建独立 diagnostic revision，记录 parent_job_id/parent_program_id 和本轮一次调用额度，不能清零旧 generation/repair_used/epoch 冒充原预算未用。新 Program 不继承父版本的 positive。

固定 binding 由既有记录和公开任务/工作簿信息构造并离线校验，不输入答案；不新增 Extractor 轮次。每个 Builder 调用的结构修复、截断恢复和传输重试均不能绕过这个调用上限，输出非法或调用失败即结束本候选。整个 D1 标记 `human_assisted_diagnostic`：人工预选对象、指出错误、构造公共绑定后得到的修订，不能报告为“自主发现第二例”或“R06 自动调度已被真实验证”。它检验条件性的程序修订/跨题执行；D4 才检查新任务参数是否由系统自己取得。

## 7 单答 收尾 检索与费用的修复边界

### R07 明确空截断与实际错误阶段

LiveMath 全部 77 个实际 Runtime HTTP 输入都包含与公共材料一致的 5 个选项；没有发现缺选项、标签提取错配或 gold 字段注入。Val 17 题应分为：2 个正确标签、10 个完整但错误标签、5 个 `finish_reason=length` 且空 content。Train 60 题另有 14 个空截断。现 19 个空截断被统一记录成 single_answer 且 error=null。[E14]

在 `system.agent()` 和 single_answer 结果组装增加 `provider_finish_reason`、`answer_status`、`empty_answer`、`completion_truncated`；保留原官方分数。`answer_status` 至少区分 valid、empty、truncated_empty、invalid_format。分类由已有 response 和合同得到，不新增求解。`completion_truncated` 与答案可评分性分开：非空且可按原合同解析的 length 响应仍交原 scorer，不能一律判无效或剔除。已完成 77 题可以离线生成诊断侧表，不能改写旧正式分数。

OfficeQA 31 条 `runtime_agent_schema_error` 的实际 stage 为 Planner 9、Extractor 15、Builder 7、Runtime 0。日志新增明确 `stage/purpose/logical_decision_id/repair_index`，不再凭错误名将全部错误归因到 Runtime。[E15]

### R08 将已实现的 Office 收尾接入可用证据、独立用途配置和预算预留

89 个已提交任务中，68 个走 finish_only，其中 53 个非空、15 个空；另一个重复失败终止也是空。UID0115、UID0148 的收尾均耗尽 32768 completion，content 为空。最终文本通路已经存在，原 `executor.py:311–331` 是唯一收尾入口，repair_limit=0、无 tools，原逻辑不应重建。[E15、E28]

#### R08a 物化已经取得且当前收尾需要的证据

`recent()` 只展示 history 最后三项的预览；`TaskContext.model_memory()` 保存 name/arguments/result_id/accepted/status，没有结果正文。finish_only 不允许 read_result，一些已经取得的证据在最终材料里只剩不可解引用 ID 或 truncated preview。因此不能只写“压缩历史”，然后仍把证据引用交给一个没有读取工具的请求。

新增纯函数 `build_finish_evidence(...)`，输入当前只读 TaskContext/执行快照、completed/pending 显式引用、recent、原结果来源索引与固定输入上限；输出 `materials/included_refs/omitted_refs/unresolved_refs/evidence_digest/serialized_input_bytes`。在线在唯一 finalize helper 内调用；D2 由独立诊断入口用原快照调用同一个材料构造器。

材料按以下固定优先级去重与裁剪，并记录算法版本：

1. 保留原问题、答案合同、已存在的候选答案与明确未决项。已有符合合同的最终答案按原路径提交，无需额外 finish 请求。
2. 解引用当前 completed/pending 明确指向的值与当时已记录的 local_reads 窗口；保留原 path/offset/window。只能取原任务在收尾之前已取得的值，不能从 evaluator、learning、trial 或后续审查材料添内容。
3. 从已有执行结果索引中的成功读取/计算结果按原事件顺序逆序补入不同来源及窗口，最后补有界搜索定位 metadata。使用工具、规范化参数、scope、文档版本、offset/window 与内容 hash 去重；不合并同文档的不同窗口，不让重复目录列表或失败结果挤掉正文。
4. 每个片段附原 result/event ID、path/offset/window 和内容 hash，并与当前已取得结果核对来源；D2 以原 finished native events 和当时 local_reads 为白名单。52/51 个 context 条目含别名，不能当作独立工具调用数。

不调用 Broker、Adapter 工具或有副作用的 context.read；已有窗口用只读 resolve 和纯切片恢复，不重新读 corpus、不新增 read_result 日志或原生工具次数。引用失效明确列入 unresolved_refs；预算裁剪列入 omitted_refs，不能伪造正文或只交需要模型再解引用的 ID。

本轮固定 `finish_input_max_bytes=65536`，按最终 system/user 消息与 tools 的规范化 JSON 合计 UTF-8 字节计算；每个窗口最多沿用 `result_read_window_chars=12000`。超额按上述优先级截取并记录，基础问题/合同都放不下则本地拒绝，不临时放大上限。这个字节封顶是工程边界，不能替代对最终实际 wire 的完整输入 token 估算与 R09 准入。模型空输出仍按原 scorer 记分。

#### R08b 按真实 purpose 同时解析 provider、completion cap 与缓存身份

本仓库已支持 `OpenAICompatibleConfig.thinking_type='disabled'`，实际 DeepSeek 方言会发送 `thinking: {type: 'disabled'}`；这在本次本地序列化探针中已经确认。但 `EmpiricalSystem.provider(stage)` 当前只按 stage 缓存，thinking_type 读取全局 llm.protocol；agent 在 purpose 确定之前就选 provider。只在 YAML 加一个 finish_only 小节不会生效。[E28]

新增 `resolve_call_settings(stage, purpose)` 并将 purpose 的计算移到 provider 选择之前；默认保持所有普通阶段原设置。新增 `llm.purpose_overrides.finish_only` 配置域并在 validate_config 中验证允许字段；本轮仅在 Office 诊断配置启用，候选设置固定为 `protocol.thinking_type=disabled`、`max_completion_tokens=512`。provider 缓存键至少包含 stage、purpose 和有效设置 digest；同时保留现有 provider_override 的测试注入接口。

`agent:184` 的 completion_cap、`:218` 显式传给 provider.complete 的 override、`:186–192` response_key，以及实际 request audit 必须采用同一有效设置。只把 provider 默认 cap 改成 512 会被 Runtime 的 32768 override 覆盖。记录有效 purpose 设置和版本进入诊断/运行身份，旧响应不能被误作新协议响应；普通 solver、Planner、Extractor、Builder 的 high reasoning 和 cap 保持原值。

本地支持 payload 不等于远端必然接受。D2 正是固定的两次真实验证；若远端拒绝该合法候选设置，记录 provider 不兼容并结束该名额，不自动改参数再发，不扩大上限。未来是否采用该配置按 D2 的实际结果决定，不依据 Val 分数调参。

#### R08c 在预算耗尽之前进入同一个收尾入口

现 finish_only 由 `broker.remaining_calls()<=0` 触发，并非 Runtime token 即将耗尽时触发。真实路径探针确认：Runtime token 先耗尽时，收尾 HTTP 可能根本无法发出。把仅有的收尾分支提取为共用 finalize helper，由原生工具预算耗尽或下一次普通 Runtime 请求无法在保留收尾额度后准入时调用。

预留范围包括收尾的有界输入和 512 completion，计入原 Runtime scope 与本轮总预算，不是额外账外预算。通过 R09 的同一准入器核算，普通调用不得消费这部分预留；若连预留也不足或当前上下文无法合法构造，结束并记录原因，不重解、不自动增加 cap。single_answer 不进入该补救分支。

现有 scoped grep 已在样例中实际生效，不重做该旧补丁。重复只读请求的优化是可选的 task-local 确定性缓存：须带工具、规范参数、corpus hash、scope、offset/window，保留原结果来源及日志语义；不能误合并不同范围或改变正式调用预算口径而不记录版本。read_result 引用和批量上限先用八份存量样例回放，不付费重采相同轨迹。

### R09 正确计算总成本并实现整轮准入控制

按 `(run_id,call_id)` 去重的 3,465 条 HTTP 记录全部有 raw_usage，合计 **75,771,494 tokens**。[E16]

| Benchmark | Runtime | Planner | Extractor | Builder | 合计 |
|---|---:|---:|---:|---:|---:|
| SearchQA | 1,131,041 | 0 | 3,050,545 | 0 | 4,181,586 |
| LiveMath | 1,717,365 | 0 | 1,683,807 | 0 | 3,401,172 |
| OfficeQA | 26,662,977 | 2,861,734 | 6,273,741 | 7,988,772 | 43,787,224 |
| SpreadsheetBench | 17,831,688 | 2,172,461 | 2,137,984 | 2,259,379 | 24,401,512 |

Office 包含在途调用，Spreadsheet 包含第 57 题。Office UID0034 的关联成本 1,169,472 中，521,357 来自隔离试用的后续模型求解，不能把全部关联成本说成原 Runtime 超额。

`system._budget()` 现在按角色和 scope 分开；调用前只检查已用额度，调用后才检查新总量。实测 Spreadsheet 有 15 个 Runtime scope 超过 600k，最大 645,011；Office 有 5 个 Extractor+Builder scope 超过 262,144，最大 330,579。现配置不是精确硬封顶。[E17]

新增诊断执行器共享的 BudgetGovernor，在 **provider 内部每一次真实 HTTP attempt 发出之前**执行准入；具体接点为 `OpenAICompatibleProvider.complete()` 第 262 行开始的内部重试循环、第 270 行 requests.post 之前。该仓库使用 requests，没有另一个隐含的 OpenAI SDK 自动重试层；D1/D2/D3 的诊断 provider 设置 max_retries=0，对应一次实际发送。所有重试循环仍必须经过同一准入 hook，不能只包 complete 外层。[E29]

新增 `before_http_attempt(attempt_id, request_context, payload_fingerprint, estimated_input_tokens, max_completion_tokens)` 返回 reservation；放在每次循环生成 audit ID 后、transport try **之外**，再进入 requests.post。准入拒绝只写 admission decision，没有发送就不冒充未知计费 HTTP。进入真实发送后即占用一个 HTTP 名额，成功、400、超时、解析错误均不能退回名额后重用。

在 `_append_request_record:604–605` 完整 attempt 记录落下后，以 `after_http_attempt(reservation, request_record)` 对账，覆盖成功/所有错误出口并先于 retry 决策。按真实 audit/attempt ID 幂等，外层逻辑 request ID 不能去重内部多个 HTTP。回调故障保留已发生的费用与响应，不能掩盖或删去原记录；usage 未知保留 reservation/unknown 状态，下一次准入暂停，不能释放当成 0。

当前 provider 虽有 set_request_context，但 empirical.agent 的直接 complete 路径没有设置完整上下文。应在 system.agent 的真实发送分支显式传入或设置 request context，带 campaign_id、session_id=outer_request_id、stage/purpose/logical_decision_id/repair_index、parent_task_id、trial_id/trial_execution_id、budget_scope/episode_id；每次 provider attempt 继承它并独立准入、结算。单靠 provider thread-local 猜阶段会丢标识。恢复缓存分支只读旧记录，不调用准入发送或再次计费。累计所有阶段、所有试用/续跑、所有 HTTP 重试、未知计费尝试，为本次输入估算与最大 completion 预留额度。恢复读取旧请求不重复消费预算，历史费用与新增预算分别统计。未知 usage 不能当 0 元，出现后暂停新增请求并对账。

预算记录包含 `parent_task_id`、`trial_id`、`trial_execution_id`、`stage`、`purpose`、`actual_input/completion/total`、`reserved_tokens`、估算方法及误差。所有阶段都要进行输入和 completion 的预算检查，只管 Builder 不够。**历史配对的 D3 必须保留原 completion cap：不足以准入就不发请求，记 not_run_budget_limit，不能缩小 off 的 cap 后仍称仅移除了 guidance。**其他阶段的 cap 策略在运行前固定，两臂一致，不能根据前一臂结果调整后一臂。本轮串行调用，最多一个在途请求；达到停止预算时不再发新请求，并报告任何估算误差导致的超额。不得把“调用后报 BudgetExhausted”描述成 provider 账单绝不会越线。

成本报告分解为 solve、learning_update、program_trial 含 continuation、frozen_eval；reasoning 已包含在 completion，不能再次累加。没有价格表，不填人民币成本；需要账单金额时按 cache_hit/cache_miss/completion 各自单价计算。

另将 `unique_completed_train_tasks`、`unique_learning_source_tasks`、`learning_updates`、`program_trial_executions`、`sample_exposures` 分列。Search 当前 cumulative_train_examples_seen=600，是 300 trajectory 加 300 learning_update；LiveMath=120，同理，不是独立训练题数。保留旧字段原义，不悄悄重解释历史值。[E18]

### 7.1 Guidance 检索改进作为明确分离的方法修改

SearchQA/LiveMath 分别有 296/60 个 guidance Skill，Program 和 workflow 都为 0；Val 全部注入了 guidance。它们当前评估的是声明式指导学习，不能代替四层可执行技能图的实验。[E11]

现 `Bank.retrieve_guidance()` 使用原始词项集合重叠，存在不相关召回：Search 的 894 个 Train 注入 pair 中有 24 个重叠为 0；地点题注入人物/电影卡等已有实例。不由此推断低分必由 guidance 造成。[E19]

本轮主对照优先保持现有 Frozen 内容、检索策略与单答协议，取得一次旧方法的 paired 诊断。**下面的改法可以单独实现为未启用的版本，不在对照运行中途切换：**

1. 检索日志输出 score、命中的有效词项、过滤理由；top-k 是上限，不足时允许空。
2. 用固定 stopword 过滤、训练 Bank 词项 IDF 与长度归一替代纯词集合交集；只按公开 query 与能力描述检索，不把原题答案例子作为主要索引内容。
3. 所有参数仅以 Train 和确定性检索 fixture 固定，不根据这 24/17 题的正确率调参。
4. 不增加 LLM reranker，不在失败题上多次重解。失败轨迹可以产生避错指导，但具体训练事实与抽象规则要在新资产元数据中区分，不能把每题一张事实卡都称为泛化 Skill。

若执行方决定本轮直接启用新检索/新单答 prompt，必须在任何新调用前改用第 9 节的 fresh 双臂方案；不得先跑旧对照，再加跑新版本对照，也不得把旧有指导响应当成新策略的结果。

### 7.2 LiveMath 全 A 的解释边界

所附 Train/Val 77 个 evaluator 的正确标签均为 A；完整性元数据报告池 177 题全部为 A。这是确定的答案位置偏置。现已补读本仓库 `skillcompiler_bench_contracts/livemath.py:11–52`：它读取原 correct_choice 的 label/text，在该项缺失时补回原文并按标签排序，没有把所有正确标签改写成 A 的代码。原上游数据的标签分布不因此变成已验证的数学质量结论。本地用实际解析器/评分器重放所附 77 个响应，评分与原记录一致，19 个空截断保持原义。[E14、E32]

不向 Runtime 或 guidance 注入“总选 A”。本轮不因低分重排旧材料、不改分数、不偷偷修改统一 manifest。可以零 API 报告恒选 A 在已附 Train/Val 的退化基线，用于说明此任务协议存在位置捷径，不能把它当数学推理能力。

若后续需要修正位置偏置，采用独立版本的固定、task-ID 驱动的选项置换，公开选项、label 映射和 evaluator 同步，所有方法相同；这属于新的输入协议，旧 Run 保留原口径。当前有限诊断不再叠加一轮置换收费实验，也不要求因此重跑其他 Benchmark。

## 8 零 API 验收集合

这部分先完成一次。修复失败只重跑对应 fixture 和受影响回归，不触发额外收费试验。不能仅以源码能 import 或几个 stub 成功代替真实合同检查。

| 编号 | 输入或故障点 | 必须观察到的结果 |
|---|---|---|
| Z01 | 第 57 题真实声明、已存源码返回结构 | 入库/Program 提交前给出类型化合同拒绝；没有 KeyError、没有伪 positive |
| Z02 | final_answer 缺 answer、非字符串、空字符串、Adapter 类型不符 | 不进入独立评分；记录各自错误类型 |
| Z03 | final_files 合法两文件、只 xlsx、零发布、旧文件残留 | 仅真实齐全本次发布可提交；漏发布可反馈修复 |
| Z04 | 严格业务 Schema 加合法 files；非法发布路径；datetime | 保留严格业务校验，合法发布通过，非法数据被明确拒绝 |
| Z05 | Worker→complete 的 strict Schema；原 Dynamic/intermediate；文本/文件直接 Program 最终提交及旧文件反例 | 合法 files 不在节点交接被二次拒绝；自动接管保留；直接提交来源可核验，旧文件/no-op/debug 变化无虚假 positive |
| Z06 | 真实 57 checkpoint 加 ready/build/program_id job | 收尾不调 run_task/agent/evaluate；原 request IDs 和成本不增加；隔离候选确实不能路由 |
| Z07 | Program 保存后、trial_started 后、worker_finished 后、record 后故障注入 | 恢复不重做已完成阶段；未知试用无 positive；真正新执行单独计费 |
| Z08 | 旧 null episode 后完成恢复 | SQLite/Trace/终态投影一致，旧错误可追溯，一题不重复计数 |
| Z09 | 56 completed、Bank 42 learning cases 的真实组合 | 恢复不补入原来未学习的 15 题；experience 格式一致 |
| Z10 | 同题/跨版本正向；当前 job 第二槽；D1 显式历史版本；D3/D4 只读视图 | 仅同版本两物理任务满足晋升；现 action=trial 不选错历史版本；off 直接 Program ID 也被拒绝，Bank 不变 |
| Z11 | 77 个 LiveMath 真实响应、8 个 Office 存量样例；已得结果引用、已终结 finish checkpoint | 19 个空截断正确分类且分数不变；阶段正确；收尾证据只从已得结果物化；D2 为新诊断决策，不改旧终态 |
| Z12 | scope usage、长输入、HTTP 重试、purpose provider/cap/response_key、token先耗尽 | 每次实际发送前准入；收尾预留生效；purpose cap不被Runtime cap覆盖；旧请求不重复计费，未知计费不按0 |
| Z13 | 原身份普通 resume、显式 recovery child、Frozen 读写 | 错身份仍拒绝；合法继承有来源；Val/Test Bank digest 不变 |
| Z14 | 新任务边界暂停和恢复 | max_new_tasks 只数新提交；历史任务不重解/重学；未完成 Train 不进 Val/正式 Frozen |

上一轮五个探针确认了错误 role/KeyError、旧文件检查绕过、重复 Builder、episode 去重和保存阶段。本轮另按真实源码完成三组零 API 核验：合同/Executor 发布与直接提交 9 项，provider/预算/历史 wire 与响应回放 9 项，当前 job 版本选择及只读对照视图 4 项。原始结果、范围和替身说明写入配套证据 JSON。

这些核验没有实验模型请求、Docker 运行或归档 Program 执行。provider 组使用 pip 自带 requests 作为禁止联网的导入兼容件、拦截 POST 并封禁 socket；它验证实际源码逻辑，不冒充生产依赖环境。77 题回放只调用纯文本 LiveMath 解析/评分函数，未重新解题。只读视图是候选接线原型，不能写成已合入生产。

没有在本机运行原仓库完整 pytest/Docker 套件，也不把 README 的历史 252 项通过冒充本轮结果。实施补丁后的验收按 Z01–Z14 和相关原有回归执行；不用重复收费获取这些已知边界。

## 9 一轮有限真实验证的执行清单

### 9.1 默认方案及总量

**这是一轮固定诊断，不是无限重复的 12+6，也不是从空 Bank 再训练。** 开始前写 `diagnostic_manifest.json`，锁定源码、模型、既定 Train/QA 样本与候选、来源 Bank hash、D4 适用性谓词/代码 hash/破同分规则、预算、对照策略与停止条件。D1 冻结后、任何 Val 求解前，按既定规则补记 D4 唯一实际 ID 及诊断 Bank hash；这不是依据 Val 结果二次选题。后续不得因为结果不好临时换题、换版本或加预算。

| 阶段 | 固定内容 | 新增规模上限 | 它能回答什么 |
|---|---|---|---|
| D0 | 第 8 节全部离线检查；第 57 题收尾 | 0 模型决策 | 工程合同、日志、恢复是否正确 |
| D1 | 第 6.2 节两个既存 Train 候选 | 最多 5 次纯 Program trial；最多 2 次 Builder HTTP | 现有程序能否按真实参数跨题，修订是否退化 |
| D2 | Office UID0115、UID0148 的保存收尾状态 | 各 1 次 finish_only，共 2 次 HTTP | 真实 provider 的最终文本收尾是否接通且有界 |
| D3 | SearchQA 全部既定 Val24、LiveMath 全部既定 Val17 | 默认仅补 no-guidance 臂，共 41 次 single_answer HTTP | 现有指导的初步质量/成本配对信号 |
| D4 | Office、Spreadsheet 各最多 1 道预先锁定的适用 Val | 每题有/无 learned assets 两臂，共最多 4 次多轮 episode | 已验证 Program 是否在未参与学习的任务上真实接管并节省 |

D1 的程序试用只执行明确 terminal Program，不自动追加 Dynamic 求解来获得正分。诊断入口显式禁止 continuation，并复用正常 Builder prompt、BUILD schema、program_permission_view、固定 binding 校验和 Program/Implementation 入库规则；不直接调用可能继续生成第二版或追加 Dynamic 的旧 _realize 全循环。需要 continuation 才能测的其他候选不在这一轮增加。

D2 的两份材料已核实可恢复：UID0115 的 executor_state.context.results 有 52 项，原 finish 材料的 16 个唯一 result_id 全部可解引用；UID0148 有 51 项，23 个引用全部可解引用。两题 native_events 各有 24 个 finished 事件，结果与对应 context 条目相等。构造诊断材料时优先使用这些已完成 native events 的结果白名单及当时 local_reads 窗口，只取原收尾前已得到的内容，不混入后续审计或评分数据。

D2 从两题原 finish 请求和当时的 TaskContext 已得结果派生 **新的诊断决策**。不得普通 resume 两个已完成任务：当前 executor_finished 和 reason=finish_only 的恢复分支会直接返回，清空旧标记则伪造历史。独立入口只装载当时已得证据，使用新 checkpoint/decision ID 调用一次 finish_only，不重新执行整题，不把新答案替换旧正式分数。若无法按 result_id/path 从原 checkpoint 或对应已完成 native events 恢复某份必要证据，标记该材料缺失/跳过，不付费重采整条轨迹。[E30]

D4 仅在选定同版本 Program 已依据 Train 证据达到原 usable 条件后进行。若未达到，输出“当前候选未形成可冻结执行能力”，该基准 D4 为 not_applicable，不改成零分、不扩大训练或另找有利例子。流程固定为：D1 结束 → 冻结有父来源的 diagnostic_snapshot → 用预先写好的公开适用性谓词生成 eligible IDs → 按 `SHA256("cf4-r2|" + benchmark + "|" + task_id)` 取最小项 → 保存选择代码哈希与题目 ID → 才开始 on/off 求解。适用性只用公共任务/文件结构与合法 Schema 绑定，不使用答案、评分或先试 Program 的结果。没有适用 Val 就报告覆盖缺口；已选题不适用或失败也不补选。两个物理 Train task 必须是不同 canonical task/physical key，同一题两个工作簿变体不能冒充两道任务。

### 9.2 对照的实施要求

D3 默认保留现有 on 臂原响应，新 off 臂只清空 `guidance`，不改变任务、公开选项、single_answer 提示、模型生成参数、预算与评分器。发送前比较实际 wire 生成参数；允许的求解干预只有 guidance 清空。旧 on 只读复用，不重新收费。D1/D2/D3 的诊断 provider 实例明确使用 `max_retries=0`，每个固定名额最多发送一次真实 HTTP；当前传输实现就是 requests.post 的内部循环，不虚构另一层 SDK 参数。若实施方另引入传输包装器，其重试也必须关闭或进入同一计数。仅改变诊断实例的失败处理，不修改正式默认重试。

D3 已返回的空答案、截断或非法答案格式，仍按原评分器记分并保留在配对分母，不能比历史 on 更宽松地排除新 off 失败。只有未取得可评分输出的基础设施错误，才标记 pair 不完整；该 ID 仍保留在固定清单及覆盖率报告，不补请求、不换题、不隐去。D1 非法 Builder 输出结束候选，D2 非法收尾记诊断失败，三者不能混用删除规则。

这属于**历史配对诊断**：run_seed 控制了任务顺序，但当前配置 `generation_seed_supported=false`，模型采样与调用时点不同，不能包装成严格同批随机对照或确定因果估计。其目的只是用低成本判断是否存在值得正式评估的信号。[E20]

若检索、单答 prompt、模型协议或输出预算已改变，旧 on 不再可比。必须在首次调用前废止默认清单，改为 **fresh 两臂各 41 次，共 82 次 single_answer HTTP**，所有样本、Bank 和配置固定。默认 41 与 fresh 82 是二选一，不得顺序叠加；fresh 仍使用同一总预算，不启动新的训练。它是兼容性不满足时的预先替换规则，不是看完默认结果再增加的试验。

D4 两臂均使用相同修补后的宿主、Adapter、公开工具、环境重置、模型、预算与原评分器。on 使用预先冻结的诊断 Bank；off 禁用其 learned Skill/Program/workflow 检索和执行，其余合法 Planner/Dynamic/native tools 不降配。这比较的是整个 learned-assets 干预，不能把全部差异命名为某个 Program 的独立因果收益。每个多轮 episode 最多 40 次实际 HTTP 尝试，Planner/Runtime/修复/传输重试全部计入，两臂相同；原有更紧的限制仍有效。Val 读取期间禁止学习、program trial、资格更新。保留同一 Bank 的内容哈希，不按两臂结果选 Bank。

诊断快照记录原 Run/Bank/源码哈希、修补版本、候选新旧 ID、来源 Train ID、D1 新增 usage、冻结时间和文件哈希。Office/Spreadsheet 的 Train 尚未完成，这个诊断快照及其两道 Val 个案不得写成原正式 Run 的最终 Frozen/完整 Val 成绩。D4 两臂都从同一个标准 PublicTask 开始；选样器为判断适用性取得的 layout、series、列名定位、数据值等不得额外注入 on 臂。selection artifacts 与 runtime inputs 分开保存，参数由两臂系统各自取得。

若 on 臂没有实际执行已验证 Program，只记录 guidance/计划路径贡献；不能把它写成 Program 降本。少量适用任务只提供条件性的机制证据，不能推算整个 Benchmark 的总体提升。

### 9.2.1 有无学习资产对照的实际代码接线

当前生产代码没有现成的 off 开关。新增诊断专用 `ReadonlyBankView`，通过 `EmpiricalSystem(..., bank_view_factory=None)` 在创建底层 Bank 后、构造 Planner/Executor 前包装一次；默认 None 完全保留正常生产行为。仅允许 readonly 的独立诊断实例使用；若复用 `run_empirical.run()`，增加同名可选参数向构造器透传。不得仅在启动后替换 system.bank 而留下 planner.bank/executor.bank 指向旧对象。[E31]

| 模式 | 真实行为 |
|---|---|
| on | 底层冻结 Bank 原样读取 |
| D3 guidance_off | 仅 retrieve_guidance 返回空；其余读取与完整 single_answer wire 参数不变 |
| D4 learned_assets_off | get 返回 None；all/retrieve/retrieve_guidance/planning_cards/routes/program_options/attempts/jobs/train_cases 返回空，显式阻断直接 Program ID 调用 |

digest/close/必要的词项工具代理底层；所有 mutation 均拒绝。记录 source_bank_digest 与单独的 view_mode/view_version/view_hash，不能把“底层 Bank 相同”冒充干预模式相同。两个视图各用新的干净 checkpoint，不复用 on 的 initial_plan、executor_state 或响应。

`Planner.plan:23–25` 在 off 没有相关资产时会自然返回 Dynamic，可能少一个 Planner HTTP。这是 learned-assets 整体干预的真实结果，不为凑调用次数增加一个空 Planner 请求；公开工具、Dynamic 权限、模型和预算仍保持相同。

本次本地 wire 复核用实际 provider._build_payload 重建 Search24+Live17，共 **41/41 payload fingerprint 与归档一致**，因此现有默认历史配对方案有代码依据。修补后在发送 off 前再次以固定材料做同类离线比较，允许的求解差异仍只有 guidance 清空；本地对齐不证明远端模型后端或采样时点相同。[E29]

本次只读视图的最小接线原型已用惰性 agent/worker 验证，包含绕过菜单直接给 Program ID 的拒绝分支；它不是 R2 生产补丁已经实现的报告。

### 9.3 预算与停止条件

本轮固定新增 **5,000,000 accounted tokens** 作为停止预算；其中 **500,000** 预留给已开始 episode 的既定有界收尾，计入总额且每次请求仍须准入。CPU 状态落盘本身不消耗模型 tokens。这个数是根据既有消耗制定的执行限额，不是保证结果能提升的阈值，也不是人民币报价。历史 D3 on、原训练和第 57 题旧费用不占新增额度，但必须在总账中单列。

所有新增阶段共用一个预算台账，最多一个模型请求在途；先准入再发请求。累计新增用量达到 4,500,000 后不启动新 episode 或新的修订；预留部分不用于换题、重解或再建一轮。若 provider 计费/输入估算产生超额，记录 planned/actual/overrun，不隐去超额。

固定上限：默认方案 **45 个单次 HTTP**（2 Builder + 2 finish + 41 QA）加最多 **4 个多轮 episode，每个最多 40 个 HTTP**，合计最多 **205 次实际 HTTP 尝试**；若预先替换为 fresh QA，则为 86 个单次 HTTP 加同样四个多轮任务，整轮最多 246 次。全部还同时受 5M token 停止预算约束。不能把结构修复、截断恢复、传输重试放到另一个不计数的预算里。

满足任一条件，停止相应阶段或整轮新增请求，生成已完成部分报告：

- 同一补丁后再次出现未捕获合同错误、成功评分被学习异常覆盖、恢复重复解题/Builder。
- 任务/代码/模型协议/评分身份漂移，或出现未知计费且无法从原记录核算。
- 预算或请求/episode 上限达到；不因尚未得到正向效果而自动追加。
- D1 固定修订和试用用尽仍无兼容跨题能力；按当前不足结束。
- Val Bank 写入、不同版本 positive 合并、对照工具或预算不等价。

### 9.4 必须交付的结果与预期

工程通过必须看到：第 57 题能零重解收尾；文件/文本合同错误有确定终态；无上述未捕获异常；历史成绩和费用守恒；恢复不补入未学习历史题；Val/Frozen 的 Bank 不变。**普通模型答错不使工程验收失败。**

方法只报告观察值：

- D1 同版本 old/new Train outcome、是否达到 usable；无回归、真的参数化与真实文件发布，都是可解释证据。
- D3 逐题 on/off 正误、wins/losses/ties、总 tokens、空截断率。原 Search19/24、Live2/17 是 on 臂事实，off 结果本次尚未知，不预设必须赢几题。
- D4 当前题参数来源、Program hash、实际调用、输出消费/直接提交、官方评分和相对成本。两边都答对时再重点比较成本；不能把快速答错当降本成功。
- 同时报告一次性学习/修订成本，避免只展示评估期省 token。可计算 `C_learn / (平均C_off - 平均C_on)` 的观测摊销点；分母≤0 时报告没有观测到可摊销节省，小样本不外推。

不设“必须 80% 成功”“必须省 50%”“必须超过某论文”等临时放行门槛。希望看到的是可追溯的跨题复用与质量/成本方向；没有信号就是本轮有效结论，不继续换样本追正结果。

## 10 诊断后怎样继续正式研究

| 结果 | 下一步 |
|---|---|
| 工程未通过 | 只修具体失败点并重跑对应离线 fixture；本轮收费名额不重置。确需额外真实 provider 复核时，先说明离线不能解决的边界、所需唯一调用及单独额度，再安排后续限定验证，不重开整个 D0–D4 |
| 工程通过，D1 仍无可用跨题 Program | 保留已完成运行与诊断，明确当前可执行复用尚未成立；先修改公共绑定/实现调度或收窄方法主张，不通过继续跑剩余大 Train 期待自然解决 |
| 工程通过，候选可复用，D4 没有匹配任务 | 报告覆盖不足；不擅自更换 Benchmark 或添加有利 Test 题，现有成功也不冒充总体效果 |
| 工程通过，有真实复用证据，但小样本质量/成本无优势 | 如实记录；这不证明方法无效，也不需要自动重试。是否做正式主实验取决于既定研究目标，而非新增主观分数线 |
| 工程通过且有可解释信号 | 固定代码、配置、算法与数据身份，按原规划继续所需 Train/Frozen/Val，之后才开始 Test；不新增另一轮均衡 12+6 |

SearchQA 已完成的 Train/Frozen 不因 Spreadsheet 文件输出补丁作废。LiveMath 已完成成绩也保留，但必须同时披露空截断、位置偏置和 guidance-only 机制范围。它们的低分本身不是工程阻断，也不能被写成指导已经导致退化。

OfficeQA 的后续当前真实进度必须在执行机器上重新读取；本包只证明 20:10 的状态。若需要暂停以应用补丁，使用任务边界暂停并等待实际停稳，不能热修改。Spreadsheet 通过显式 recovery child 续接，不从头重训前 56 题。

本轮有限诊断在 D0–D4 结束，**不自动启动正式尾部**。后续正式续跑头两个新任务可分别在提交后做一次状态核验，再继续原计划；它们直接进入该继承运行的正式进度，不另做两题收费副本。其真实新增费用属于后续正式运行，不能谎称包含在本轮 5M 已结束的诊断额度或是不收费检查。

新代码混合继承的运行应明确报告 R1 前缀与 R2 后缀；若论文主表要求从首题起同一最终算法，则该继承运行保留为开发/恢复证据，最终版本只在诊断结束后进行一次预定正式训练，不把每个补丁都变成重跑理由。

诊断 Bank 不因 Val 表现好就导回生产 Bank。Train-only 的候选修复可作为有版本记录的工程开发产物；正式算法是否采用对应公共修复必须在下一阶段开始前固定，不依据某个 Val 分数挑选特定 Bank。

## 11 实施顺序与交付验收

建议按三个可审查提交组织，而非并行热改同一运行目录：

1. **合同与试用提交**：R01/R02、对应生成材料、Worker/Executor 接点和 Z01–Z05/Z07/Z10。
2. **恢复与日志提交**：R03–R05、R07/R09、Z06–Z09/Z11–Z14，生成第 57 题 dry-run 报告。
3. **有限诊断接线提交**：R06/R08、独立诊断入口、固定 manifest、配对模式与预算控制。检索算法若修改，单列开关与版本，运行前选择旧配对或 fresh 配对。

实施方交付以下可核验产物，不只写“已修复”：

- `patch_manifest.json`：起止 commit、具体函数、协议/配置变化、旧资产兼容策略。
- `offline_regression.json`：每个 Z 编号、实际 fixture、结果、原有回归影响范围；明确 0 API。
- `recovery_dry_run.json` 与 `recovery_receipt.json`：继承身份、57 收尾前后计数、原 call IDs/费用守恒、学习样本池不变。
- `diagnostic_manifest.json`：固定候选、Train/Val IDs、on/off 差异、Bank hashes、上限、停止规则。
- `diagnostic_results.json`：工程结果、逐题比较、Program 消费链、失败类别、预算 planned/actual、未知计费、未执行项及原因。

建议新增诊断命令入口 `python -m atomic_skillgraph.experiments.run_cf4_r2_diagnostic --manifest ...` 和恢复入口 `python -m atomic_skillgraph.experiments.recover_empirical --source-run ... --output ... --dry-run`。**这些是实施目标，当前代码中不存在，实施前不得当现成命令运行。** 新行为配置必须纳入 identity/config 校验，不绕过 `validate_config()`，不把新 YAML 字段直接塞进当前白名单导致启动失败。

CLI 不允许 `retry_until_success`、`ignore_identity`、自动扩样或读取 Test 的选项。对每个阶段要能在耗费预算之前打印拟执行数量、版本、成本上限及继承来源。

## 12 证据索引

E01–E20 路径相对主审查包 `CF4_R1_四个正式Run审查包_20261007_201002/`；E21–E32 指向本轮固定 GitHub 提交。源码行号按原始 3e1f6fb 文件，补丁后以函数名定位。所有统计、下载哈希和新探针结果见配套 `AtomicSkill_CF4_R2_审查证据.json`。

| 索引 | 原始证据或代码位置 | 支持的结论 |
|---|---|---|
| E01 | `README_审查入口.txt`；`derived/run_progress_at_collection.json`；各 `episodes.jsonl` | 20:10 进度、在途边界 |
| E02 | `source/source_identity.json`；各 `run_manifest.json`；两个上传 ZIP 的逐文件哈希比较 | 清洁固定提交；两包主体一致 |
| E03 | `data_identity/materialization.json`；`public_material/*/{train,val}.json`；对应 execution_manifest；final_frozen_manifest | 所附数据/顺序/Frozen 核对 |
| E04 | `failure_scene/spreadsheetbench/failure_scene_index.json`；`runs/spreadsheetbench/seed42/train/checkpoints/spreadsheet:51-12/1/state.json`；sealed_bundles | 原题成功但学习未提交 |
| E05 | `source/src/atomic_skillgraph/empirical/{system.py,learner.py}`；`harness/benchmarks.py`；57 原始请求与相关资产 | role/Schema/Adapter 错配 |
| E06 | Spreadsheet Bank attempts；`system.py:551–568`；`learner.py:371–389` | 10 次不完整发布未进入有限修复 |
| E07 | `learner.py:330–357`；57 job 与 checkpoint builder key | 恢复重复 Builder |
| E08 | `run_empirical.py:130–135`；`learner.py:17–21,117–124`；Spreadsheet Bank train_cases | 恢复补灌 15 个原未学习任务 |
| E09 | `formal_log.py:70–76,226–258`；57 null episode | 成功恢复终态被旧 ID 去重 |
| E10 | `run_formal.py:71–97`；`run_empirical.py:103–109`；FormalLog 初始化 | 普通 resume 不允许换源码 |
| E11 | 四个 `train/bank/bank.sqlite3` 的 assets/attempts/realization_jobs；`bank.py:79–135,237–281` | 0 usable、单例验证、冻结资格、机制范围 |
| E12 | Spreadsheet 上述 SUMIF 两个 Program、所属 skill/job、36764/46167 trial；公开 Train | 参数化及跨题修订退化 |
| E13 | Office a6f616 Program、UID0060/UID0059 trial 的 grep 返回与公共任务 | 后缀错误、公共参数未消费、语义边界 |
| E14 | LiveMath 77 原始 Runtime HTTP、Trace、scorer_outputs；evaluator Train/Val；livemath_integrity | 选项完整、19 空截断、答案位置偏置 |
| E15 | Office errors/episodes；8 份 selected requests/traces；`derived/officeqa/sample_selection.json` | stage 分布、收尾空截断、现有 scope 有效 |
| E16 | Search/LiveMath `llm_calls.jsonl`；Office/Spreadsheet `derived/*/all_call_usage.jsonl` | 3,465 HTTP、raw_usage、分阶段成本 |
| E17 | `system.py:101–112,179–185,263–274`；真实 scope usage | 调用后超额、整轮预算要求 |
| E18 | `formal_log.py` training/learning_end/end_task；四 Run training_events | 曝光数与独立任务数不同 |
| E19 | `bank.py:146–152`；guidance 资产；已完成 Runtime 请求 | 不相关召回、学习资产范围 |
| E20 | resolved_config 的 generation_seed_supported；Run manifest 的 generation_seed_status | 历史配对采样/时点限制 |


### 本轮 GitHub 补充证据

所有下列仓库路径均锁定 [提交 3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/commit/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3)。

| 索引 | 仓库路径与位置 | 支持的结论 |
|---|---|---|
| E21 | main branch metadata；Git tree 1606a1e1860d8ebe5438935f2243627b4a88ad8d；103 blob 校验；run_empirical.code_identity | 当前 main 与运行提交一致；64 生产 Python 的总 source hash 一致 |
| E22 | README；system.run_task；planner.plan；bank.planning_cards/freeze；contracts.resolve_node_interface | 当前唯一生产链路、资产类型、自动接管与 Train/Frozen 边界 |
| E23 | [ProgramWorker / Workspace / Executor](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/program_worker.py#L225-L241)；empirical/program_worker.py:225–241；workspace.py:40–92；executor.py:128–150 | 已有原子发布、原始 Schema 双重校验、files handoff 缺口 |
| E24 | [终端提交与文件资格](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/harness/benchmarks.py#L242-L251)；executor.py:83–84,265–268；benchmarks.py:242–251；workspace.py:73–92 | 在线旧文件/no-op 被当ready；隔离terminal原规则与真实增量发布不同 |
| E25 | [直接提交归因](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/executor.py#L263-L268)；executor.py:231–250,265–268,477–489；system.py:449–452；合同组9项探针 | 直接最终提交的缺失归因与旧文件误归因风险 |
| E26 | [学习恢复上下文](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/learner.py#L305-L330)；system.py:345–356；learner.py:23–25,315–317；program_worker.program_permission_view；FileAdapter.reset | fresh学习恢复的真实workspace能力视图缺口 |
| E27 | [当前版本与第二例](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/learner.py#L71-L157)；prompts.py:25–29；learner.py:78–114,128–157,269–303,330,371–389 | action=trial只能选当前job版本；既有第二槽与绑定约束 |
| E28 | [provider 有效参数](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/agents/provider.py#L165-L191)；provider.py:48–90,183–187；system.py:131–192,218；executor.py:86–88,311–331；task_context.py:55–64,109–110 | purpose配置未接线、实际thinking参数、finish证据引用与触发条件 |
| E29 | [真实 HTTP attempt](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/agents/provider.py#L262-L378)；provider.py:194–380；system.py:179–219；provider组9项探针 | 每真实HTTP准入接点、无SDK层、41旧wire重建一致、预算可超额 |
| E30 | [已结束执行恢复分支](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/executor.py#L24-L49)；executor.py:31–32,255–256；system.py:345–354；两题finish请求和对应存量结果 | D2必须新诊断决策，不能普通resume已完成任务 |
| E31 | [System / Bank 注入](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/atomic_skillgraph/empirical/system.py#L61-L86)；system.py:72–86；planner.py:23–25；executor.py:334–340及call_program；方法组4项探针 | 对照视图必须统一接线并挡直接调用；off自然Dynamic、源Bank只读 |
| E32 | [LiveMath 正常化](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/3e1f6fbe68fa97bbd8c2d89dd2cd8a045745b3a3/src/skillcompiler_bench_contracts/livemath.py#L11-L52)；skillcompiler_bench_contracts/livemath.py:11–52；agents/provider._parse_response；harness/scorers/livemath.py | 正常化不强制所有答案A；77保存响应重评一致 |

### 原始附件哈希

| 附件 | SHA256 |
|---|---|
| CF4_R1_Spreadsheet第57题失败现场_20261007_201002(1).zip | `3a5eef2aef17a17dffe117a3f8f0ba6d42daafaa755e2dd836364f90d82f3d8c` |
| CF4_R1_四个正式Run审查包_20261007_201002(1).zip | `b58ce02884b2b912129bcc4acc51beecd0dc877ed9ad0dc4b9da4c679002df7e` |

## 13 随附零 API 复核脚本

`AtomicSkill_CF4_R2_旧缺陷零API复核.py` 可在保留原审查包目录结构的 Linux/WSL 环境运行，只依赖 Python 标准库。`--snapshot-root` 指向四 Run ZIP 解压后的根目录，里面应有 source 和 runs；`--output` 必须在原包目录外。示例中的路径需替换为实际解压位置：

```bash
python AtomicSkill_CF4_R2_旧缺陷零API复核.py \
  --snapshot-root /path/to/CF4_R1_四个正式Run审查包_20261007_201002 \
  --output ./cf4_r1_zero_api_report.json
```

脚本先验证四个被调用宿主源文件的固定 SHA256，拒绝读取已打补丁或身份不符的源码。它在临时目录用安全替身复现旧合同/恢复/日志缺陷，不执行 Bank 中生成的代码，不接触模型 API，不写原审查包。运行结果已经复核为五项探针成立。该脚本的用途是让实施方确认原始问题可复核，**不是 R2 新补丁的自动验收器**；修补验证按第 8 节实现。
