# SkillCompiler：六 Benchmark 适配闭环与复用降本——修改实施文档

**版本：v3.1-CF4｜日期：2026-10-06**  
**Ours 实施基线：`YU-S3/AtomicSkill-ToolGraph_v3@bc63d02bc3bb0988cdbd15a203a068fc162f0b9b`。**  
**已核查 baseline 分支快照：`baseline-skillopt@f85c1810eaf2fcaa5c3e85f09dd9f154243b2f66`。该快照不等于所有 baseline 的六基准适配已完成。**  
**上游数据语义参照：`microsoft/SkillOpt@fa4ca184573e42ec11472959dd57422381418096`。**  
**交付性质：待实施规范。本文不是已合并补丁，不是新增真实模型效果报告。**

> 保留 CF3 的可纠正规划、已验证能力接管、普通参数传递和有界恢复。本轮不重建架构、不恢复证据证明链。修复确定的输入/输出和学习调度缺陷；改善程序选择信息和重复上下文；Office 的新增检索范围对所有比较方法一致提供。

本文是本轮唯一新增实施单，覆盖上一轮 D01—D07 及已批准的 Office `grep` 范围。CF2/CF3 已完成的功能是保留基线，不能为了通过新测试回退它们。下文“新增”均指待实现内容；路径与接口有明确新旧标识。实现者不得从旧 ScienceWorld、R10.x 文档恢复已经退出主线的门槛。

---

## 0. 先区分缺陷、方法瓶颈和未测量效果

### 0.1 本轮结论

**不能说剩余问题全在适配层，也没有证据要求再次推倒方法。**

| 分类 | 已观察内容 | 本轮处理 |
|---|---|---|
| 确定的数据适配缺陷 | 三个 LiveMath 实际请求只给 B/C/D/E，评分目标是缺失的 A | 恢复上游完整选项正常化，不改评分标准 |
| 答案接口遗漏 | 单轮 solver 没有明确短答案/标签输出契约，SearchQA 全句被 EM 判错 | 在同一次 solver 请求中提供适配器输出契约 |
| 确定的学习控制错误 | usable 程序仍先合并第三个试用槽，再检查“不必再试用” | 提前处理已可用的同版本 build/trial 请求 |
| 提议到试用的接线缺口 | Spreadsheet 声明必需参数，却给出缺参数的试用输入；未进入 Builder | 用现有一次提议修复处理输入一致性，不能代填答案 |
| 已批准的工具能力扩展 | Office `grep` 无法限定刚由公开 glob 选出的文件 | 新增可选 `paths`，保留全局 40 命中和原预算 |
| 方法效果瓶颈 | 有些类别没有可用程序；已有双对象程序可见但未被选择；上下文重复仍多 | 执行能力可见的计划卡片、单一模型视图、适用试用闭合；随后正常扩大 Train |
| 尚未测量 | 六基准正式准确率能否最高、冻结执行能否稳定 ≤50k/60k | 真实独立实验测量，工程测试不能代替 |

程序生成质量、合适能力的选择和未知任务覆盖，本来就是方法的一部分。不能把这部分全部命名成“工程小问题”，也不能据 6 道 Val 推断它们已解决。自动增加模型角色、降低晋升条件、强制选择某个程序、按验证题写路线，均不属于本轮方案。

### 0.2 本轮事实范围

三份最新审查包使用同一 CF3 实现。一次 fresh Train12 为 12/12、1,325,011 tokens（含学习/试用）；第一次 Val6 是 4 Look＋2普通取放，333,290 tokens；更正后的原固定六题为 **6/6、711,858 tokens、82次模型请求**。两组 Val 不得择优拼接或互相替代。

原固定六题中，Clean 为 192,123、Cool 为 257,732、双对象为 132,267 tokens；Heat 通过学出的 Program 完成，为 33,566 tokens。输入共 636,646 tokens，占约89.43%。只缩短模型输出或降低 reasoning 不是本轮主要修法。[S1]

Train12 实际为 Heat6、双对象2，其余各1，不符合上轮请求的均衡2题/类。记录此偏差，不改写历史，不为补齐均衡再支付第三轮12＋6。后续正式 Train 使用完整冻结清单及已冻结顺序。

---

## 1. 已冻结边界：实施中不得再次解释或扩张

1. 保留 `Skill / Implementation / Program / Workflow`、受限 Python、现有 Broker、独立评分器及 `dynamic/skill/program` 三种节点模式。
2. 参考 Skill 不授予自动调用权；明确选中的 usable Program 继续自动接管、执行多个动作并向后继传值。Planner 可组合已有可用程序，新图不需要先取得整图证书。
3. 局部 `args/handoff/detach` 不占整图重规划；整图最多一次，其后最多一次剩余任务 Dynamic。环境、token、原生调用、attempt 不重置。
4. 不降低同版本程序取得 usable 所需的两个不同物理 Train 正向条件；不把 Schema 合法、程序返回 `ok` 或整题文字 Skill 当作正确性证明。
5. 原 high、模型接口、各角色 token 上限、单题动作/时间/内存限制保持。DeepSeek 名称/身份不单列修改；当前视觉能力锁不变。
6. Builder 首次32,768，截断且无有效提案的恢复上限65,536。首次＋所有恢复最多两次生成，结构/代码/试用恢复共享一次机会，共享学习预算不增加。
7. Office 一次响应最多3个独立纯读调用，顺序执行，每个实际调用单独计数；不得扩展成批量环境写动作、文件写入或 Program 调用。
8. 单轮 QA 一次 solver；不增加第二次解题、回答清洗 LLM 或正确性裁判。普通无效答案如实评分。
9. Frozen 长期资产、资格、统计和排序只读；任务内记忆及计划可修正。所有 seed 独立空 Bank，不导入人工资产，不用 Val/Test 取得资格。
10. 成本目标是各支持基准的冻结执行均值争取≤50k、控制在60k以内，并争取最高主评分。它不是停止预算；失败/高成本题不能删除，训练/试用费用不能藏在平均值之外。
11. 本轮所有记录由现有调用和运行事件被动生成，不为日志额外调用模型。

---

## 2. 修改清单与实际文件

| 编号 | 修改 | 现有生产落点 | 新增内容 |
|---|---|---|---|
| C01 | LiveMath 完整选项、公共与私有记录一致 | `experiments/prepare_benchmarks.py`、`canonical_manifest.py` | 纯数据正常化函数、全池审计、materialization版本 |
| C02 | 单轮答案输出契约 | `harness/benchmarks.py::AnswerAdapter`、`empirical/system.py::run_task` | `answer_contract()`，共用文本/标签/视觉答案约定 |
| C03 | Office范围检索 | `harness/benchmarks.py::OfficeAdapter`、ToolSpec/ABI消费者 | `grep(pattern, paths=None)`，比较方法共用纯检索实现 |
| C04 | usable任务的幂等结束 | `empirical/learner.py::learn` | 副作用前的实现请求分类，避免第三槽错误 |
| C05 | 试用输入进入一次语义修复 | `Learner._receive/_skill_references/_preflight_binding/learn` | 纯提议验证＋结构化错误，先验证后保存 |
| C06 | 可执行能力清晰展示 | `empirical/bank.py`、`planner.py`、`prompts.py` | `planning_cards()`及Workflow执行模式摘要 |
| C07 | 模型视图去重、可读、可恢复 | `task_context.py`、`executor.py`、`system.py`、`learner.py`、Adapter | 单一阶段投影函数，保留实际数据和ResultRef |
| C08 | 跨方法公共契约、版本与运行 | `prepare_benchmarks.py`、`run_formal.py`、配置/打包/薄适配器 | 独立公共小包、协议指纹、单cell驱动 |

**新增公共小包目录固定为** `src/skillcompiler_bench_contracts/`，只放三类标准库纯逻辑：`livemath.py`、`answer.py`、`office.py`，另有 `__init__.py`。它不得导入 Ours 的 Bank、Learner、Executor、Provider 或私有评分器。

为避免安装 Ours 主包覆盖旧 baseline 包，新增 `tools/build_benchmark_contracts.py`：在临时标准 `src/` 项目中复制上述唯一源目录及许可证，生成最小 setuptools 描述，只构建 `skillcompiler-bench-contracts` 小 wheel，导出文件清单和SHA256。不独立维护复制品，不把整个 atomic-skillgraph 打进小wheel。baseline 只安装这个小wheel；Ours使用本仓库同一源目录。无需改变baseline算法依赖锁，也不让baseline安装Ours主包。

---

## 3. C01：修复 LiveMath 数据，不改任务成员和评分规则

### 3.1 正常化顺序

在 `skillcompiler_bench_contracts/livemath.py` 新增纯函数：

```python
normalize_livemath_item(raw_item: dict) -> dict
```

其正常路径严格参照锁定上游 `dataloader.py::_coerce_choices/_normalize_label/_normalize_item`：[S3]

1. 读取真实 `mcq.question / mcq.choices / mcq.correct_choice`，兼容上游已有的列表、字典与标签形式。
2. 取得候选的 label 与 text，不擅自重写数学内容。
3. 若正确标签已有但正确文本缺失，按已有标签查找文本；若正确标签尚未在候选中且正确文本实际存在，将该选项作为普通候选补入。
4. 沿用锁定正常化的候选排序，不新增基于得分的排序或模型重写。
5. 返回完整 choices 和私有 correct_choice，随后立刻进行公开/私有分离。公共候选的每一项只有普通 `label/text`；不携带 `is_correct`、`correct_choice`、定理或证明草稿。

**不得从正确标签推测选项内容。** 没有原始文本、同一标签对应不同内容、重复标签、空问题或不可解析候选，均报数据完整性错误。按task_id输出诊断，不静默删题、不调用LLM补题、不把所有题默认标成A。

### 3.2 三个必须一致的地方

```text
PublicTask.inputs.choices
== evaluator_record.choices
== 最终实际solver请求中展示的候选列表
```

私有 `correct_choice.label` 必须出现且只出现一次；其文本必须与该候选一致。正常化重复执行结果不变。

本轮不另引入选项随机打乱策略：使用当前公共材料的正常化列表和原标签。所有方法消费同一份列表；baseline loader 不得又私自做一次不同的选项排列。如果未来另行决定置换选项，必须整体版本化并同步映射，不能混入本补丁。

### 3.3 数据资源与旧结果

- 离线审计整个固定177题资源池；当前只有三道缺项已被实际请求证实，不能先写“177题全部缺A”。
- Train60 / Val17 / Test100、task_id、physical_key、split成员和任务执行顺序保持。
- **只修 materialized 输入及配套私有候选表，不重新生成公共 split authority。** `canonical_manifest.materialize()`已有按固定ID复制资源的路径，继续使用。
- 新材料写新目录，`materialization.json`重新计算实际文件hash，并记录 `livemath_normalization_version` 与上游revision。不能覆盖旧资源、手改旧hash或让旧checkpoint通过校验。
- 新增 `livemath_integrity.json`：各split题数、缺项修复数、重复/冲突数、未解决ID、标签分布、公共/私有指纹、输入文件版本。审计不作答，不把私有答案送入模型。
- 旧不完整选项产生的0分保留，标为输入缺陷诊断，不事后改为成功，也不与修复后结果混算。

### 3.4 baseline接入

公共修复后的choices由所有方法使用。已有上游正常化过的候选再处理不能多加一遍A。baseline算法、batch、epoch、验证选择和预算不变。数据正常化与最终模型输入在三种表示之间的等价性必须由fixture检查，不以“都读了相同task_id”代替。

---

## 4. C02：答案契约随适配器提供，一次求解完成

### 4.1 接口

新增 `AnswerAdapter.answer_contract()`，返回代码确定的公开格式说明；共用定义放在 `skillcompiler_bench_contracts/answer.py`。核心 `run_task()`只读取此接口，不按benchmark名在核心中拼例外规则。

| 场景 | 同一次solver必须看到的输出约定 |
|---|---|
| SearchQA | 只返回问题所要求的实体/短语，使用 `<answer>…</answer>`；不附整句解释、多个备选或背景信息 |
| LiveMath | 从**当前公开列表**中选择，只在 `<answer>…</answer>` 内给出一个合法标签；不得输出不存在的标签 |
| DocVQA | 使用 `<answer>…</answer>` 返回简短文档答案；现有scorer已支持此标签提取；保持原图像与问题，不新增OCR或视觉代理 |
| OfficeQA最终答案 | 只返回问题要求的答案；数值的单位、精度依问题，不擅自额外加等价变换；保留现有独立评分 |
| Spreadsheet/ALFWorld | 沿用文件提交/环境终态，不套文本选择题契约 |

已核读本基线 `harness/scorers/docvqa.py::extract_answer()`，它优先提取最后一对 `<answer>` 标签，未提供标签时才用最后非空行。因此上述DocVQA格式已定为标签短答案，不留待实现者在两套形式之间选择；ANLS算法、阈值和私有参考答案均不改。

### 4.2 调用与边界

- 单轮路径仍一次 `agent(runtime, ...)`，不新增Planner、结构repair或第二次答题。
- 当前公开候选和原始问题必须完整；不能为节省输入而裁选项。
- 回答格式指令来自任务接口，不含本题金标准示例或正确标签。
- 最终请求、响应和原scorer输出原样记录；旧回答不能仅因句中出现答案就改EM为成功。
- SearchQA上下文原截断策略保持，本轮不增加联网搜索；DocVQA图像字节/像素和能力锁保持。
- 模型不遵守格式时按原评分规则产生结果，不自动用另一个模型提取答案。

---

## 5. C03：Office `grep` 新增可选公开文件范围

### 5.1 唯一接口与返回

```text
grep(pattern: str, paths?: list[str]) -> 原ToolResult
```

`paths` 由Agent显式选择，通常来自 `glob` 或先前公开检索结果。只检查是否属于授权的语料路径命名空间，不新增“必须持有历史路径证书”的注册表。实现不得根据题目年份、gold或隐藏来源替Agent预选文件。

| 参数情况 | 精确语义 |
|---|---|
| 省略 `paths` | 保持原行为：搜索授权语料中全部 `.txt` 文件 |
| `paths=[]` | 返回空命中，绝不能回退全语料 |
| 非空列表 | 只搜索所列文件；去重，按规范化相对路径字典序遍历 |
| `paths=null`、字符串而非数组、空路径元素 | 普通输入错误，不执行检索 |
| 绝对路径、`..`、越界软链接、非文本语料文件、目录或不存在文件 | 普通明确拒绝，不开放权限，不回退全语料，不返回一部分成功结果 |

每次 `grep` **最多40条总命中**，不是每个文件40条。每个文件按行顺序扫描。原返回结构保持：`data`是命中数组，每项包含 `path / line / offset / text`；line用于显示，offset为与`read`一致的规范化Unicode字符偏移。无新分页、无全文重排、无自动第二次搜索。

旧程序省略 `paths` 时返回顺序和40条截断规则保持。新代码不能把scope变成glob表达式或正则路径；路径选择用现有 `glob`，文本查询用 `pattern`。

### 5.2 实现落点

在公共包 `office.py` 实现纯选择/扫描函数，OfficeAdapter负责转换ToolResult与错误类别。步骤固定为：

```text
校验pattern及paths类型
→ 编译正则
→ 解析全部路径并校验授权
→ 对最终规范化路径集合顺序扫描
→ 达到40条立即返回
→ 生成原格式ToolResult
```

调用前先完成全部路径校验；第一个有效文件不能在后面的无效路径发现之前产生部分可见结果。授权语料根不可用、真实I/O故障或解码基础设施错误保持原停止语义；非法正则、明确无效路径为 `accepted=false` 的普通输入错误。不得用 `except Exception: return []` 吞掉服务故障。

ToolSpec同源更新 `paths` 的JSON Schema、description与语义；Planner、Runtime、Learner、Builder ABI使用这份定义。RPC、`allowed_tools`中的名称仍是grep，不需要新Program操作码。

### 5.3 预算、批量和恢复

- 一次有范围grep仍算一次原生调用；文件数量不改变计数单位，保留原单题调用上限。
- 最多3项独立纯读批次保持；同一响应后续调用不能引用尚未产生的glob结果。先取得paths，下一轮才能根据结果发scope。
- 记录原始paths、规范化执行paths和命中截断状态，均为被动日志。
- 失败key必须包含真实的 `pattern + scope`，换成新有效scope不是重复失败；相同scope顺序变化去重后不制造假新进展。
- journal恢复继续按原子调用ID，不重发已完成子项。

### 5.4 所有比较方法一致

公共小wheel由Ours与基线薄适配器共用。ToolSpec表示可按各框架格式转写，但字段语义、路径排序、40条总上限、offset单位、错误类别及原始返回必须等价。

各方法的native优化/学习代码不换成Ours。NoSkill/HumanSkill也获得同样检索能力。既有方法不使用新参数是方法行为，不强迫其调用；但必须在其实际工具目录中提供这项能力。

新增 `reports/cf4_public_contract_matrix.json`，按方法×benchmark记录真实入口、上游commit、公共包hash、工具Schema/输出契约/材料hash、fixture结果和未接入原因。不能只给Ours勾选成功后声称八方法一致。未完成的baseline cell单独暂缓，不阻止已就绪Ours正常运行。

---

## 6. C04：usable程序请求先分类，再处理试用槽

### 6.1 当前根因

`Learner.learn()`先合并 `case_bindings` 并拒绝第三槽，后检查Program是否usable。本轮两个后续Heat经验触发这个顺序错误，任务本身已经成功，但学习更新未完成。[S1,S2]

### 6.2 新的无副作用判定顺序

提议验证和实际提交共用一个纯解析/分类函数（新增为 `Learner._resolve_realization_request`）：

```text
解析 $new / existing_skill_id / realization_request.skill_id
→ 读取实际Skill、job及job.program_id
→ 确认Program存在、关联同一Skill和当前版本
→ 判断本次action是不是build/trial/repair/defer
→ 最后决定需要处理哪些case_bindings
```

**同一实际Program已经usable，且请求为build或trial：**

- 记录 `realization_skipped=already_usable`。
- 原两个试用槽、历史attempt、生成次数、Program内容不变；本次第三个样本不并入固定槽。
- 对应job保持done，不新增Builder、试用或“第三份资格”记录。
- **只跳过这项实现工作，不提前退出整个 `learn()`。** 本次真实经验仍保存，合法Workflow更新仍执行，其他原规则允许的待办仍按原每题上限调度。

**以下不能被该分支吞掉：** 新Skill/新Program版本、未usable程序、明确repair请求、未知ID、被冻结写入。实际程序失败需要沿用原修复/抑制流程，不能因为曾经usable就永久跳过错误。

### 6.3 两槽与恢复规则保持

Candidate仍最多两个固定物理Train案例。同一程序已经执行过的案例参数不能事后更改以抹去失败。修订程序形成新版本后重新取得资格。不能用本次修复放开第三槽或直接把candidate改usable。

状态更新与响应逻辑身份保持幂等；恢复本次no-op不生成第二个Workflow版本、不重复调用Builder，不清除旧失败日志。

---

## 7. C05：试用参数与Skill接口一起验证，使用已有一次修复

### 7.1 把检查放到接收边界，不能等持久化后才发现

将现有 `_skill_references` 扩充/组合为纯 `validate_learning_proposal(proposal, cases, bank)`；它在 `agent(... validator=..., repair_limit=1)` 内调用，且不写Bank、不运行环境。

检查范围限定为已有接口事实：

1. `$new`有对应新Skill，已有ID确实存在；模式互斥和Workflow引用依现有CF3解释。
2. 先按第6节分类。已usable的重复build/trial不验证或合并被忽略的新试用槽，也不被其多余参数阻断。
3. 对真正需要构建/修复/试用的请求，检查0—2个唯一完成Train案例；case_id不能是Val/Test或不存在案例。
4. 校验每个binding.inputs符合目标Skill的input_schema，提供缺字段/类型路径；reset不能附虚假prefix，prefix仍必须等于该案例真实动作前缀。
5. 不在此验证“程序最终能否完成任务”；也不把Schema一致等价于语义成功。

### 7.2 错误反馈与次数

例如：

```json
{
  "code": "trial_binding_invalid",
  "path": "realization_request.case_bindings[0].inputs",
  "case_id": "<已有Train案例>",
  "missing_fields": ["column", "text"],
  "expected_schema": "<这份提议真实声明的输入Schema>"
}
```

修复材料只包括该能力、相关案例的公开任务/已执行反馈、错误路径及合法字段；不发送私有答案，不重新抄整条巨型轨迹。

- 原有一次Extractor语义/结构repair共用，不新增第二次修复。
- 有真实公开信息时，模型补齐参数；可以纠正同一尚未保存提议的参数化方式，但必须同步改Skill和binding，不能为了少填字段把必需运行参数删成可选后任由程序报错。
- 没有适用案例时，允许 `action=defer, case_bindings=[]`；保存待办，不猜表头、位置或目标值。
- 修复仍非法时，保存原始提议与错误，学习正常结束为rejected/deferred；原任务的成功分数不被改成失败，不能中断整个campaign或追加一次任务重做。

### 7.3 缺失工作不能被永久略过

保留CF2/CF3已有job状态 `ready/waiting_example/deferred/done`。后续新的适用Train经验可补齐构建或第二个试用；`reuse_existing/no_change`不能使原本ready的相关待办不可达。

本轮不建立新的能力裁判、语义证书或自动选择试题优化器。仅明确：等待案例与代码生成失败是不同状态；Skill文本存在不等于Program完成。单轮QA的guidance_only不因没有Program被反复送Builder。

### 7.4 对生成质量的界限

这些修改能使合法候选真正进入Builder和试用，不能保证代码一次正确。保留原一次Builder恢复和两个物理Train正向要求。局部搜索程序、完整任务程序都允许；优先提取可复用的重复工作，但不能硬把所有经验分解成指定九类能力或为了展示图强行拆单轮QA。

---

## 8. C06：计划选项体现执行能力，不能强制替换模型选择

### 8.1 不再把所有Workflow展示成同样“可执行”的整包资产

新增 `Bank.planning_cards(query)`；复用现有相关性、`program_options()`、`resolve_node_interface()`、`routes()`，不新增embedding服务或LLM筛选器。检索top_k仍为8，Program菜单保留原上限与权限。

卡片为只读派生视图，不保存回Bank，不新增Workflow资格表：

```text
Skill：id、goal、guidance、真实I/O、execution_intent、usable_program_ids
Program：id、state、关联目标、入口约束、真实I/O、result_role
Workflow：id、goal、各节点mode/绑定/args/依赖/输出；execution_summary
```

`execution_summary`只统计结构事实：`dynamic_nodes`、`bound_skill_nodes_with_usable_program`、`bound_skill_nodes_without_usable_program`、`explicit_usable_program_nodes`。**不是成功率，不是“图已验证”的布尔值。** Dynamic节点即使参考一个有Program的Skill，也仍计为Dynamic。

对比信息由同一解析结果生成；不通过字符串看起来相似来生成硬DataFlow或自动改节点模式。

### 8.2 提示词选择规则

在既有一次Planner请求中明确：

- 满足用户任务、数量、结果范围和实际入口条件时，优先明确选用可用程序或包含它们的短图，避免重复逐步实现已可执行的工作。
- 旧Workflow只有动态步骤不等于它比现有Program更成熟。可以compose一个正确能力节点；不必每次重写其程序逻辑。
- 目标范围不相符、入口未知或未ready时，可以选择动态准备与局部程序组合；保留CF3局部恢复。
- 不能因为Program存在，就调用它解决一个不同目标；不能自动把已选Dynamic替换成整题Program。

`mode=select|compose`和调用次数不变。保持完整的当前必需接口；不新增第二个Selector、评分模型或强制Program调用比例。

### 8.3 Runtime与Learner接线

Runtime使用同一Program卡片；已授权ready路线继续自动执行，Dynamic仅在模型显式调用后进入程序。不得为每一步新增“确认是否用程序”调用。

Learner继续从实际执行及真实Program调用学习工作流。可以用成功经验提议包含Program的新版Workflow，但其内容仍是候选复用策略；不把本题plan patch直接当长期已验证事实。旧动态Workflow不自动删除，也不使用Val结果给它降权。

### 8.4 能验收什么

确定性测试可以证明：正确程序存在、卡片没有藏掉、动态/程序覆盖如实展示、选择Program后可自动接管、新图不再额外要求整图认证。

**不能以fixture强制返回Program选择，再声称真实模型已经愿意复用。** 是否减少本轮双对象那种“有程序却不用”，看真实Planner响应与后续调用。

---

## 9. C07：减少重复输入，同时保持信息可获得与恢复一致

### 9.1 单一阶段模型视图

新增 `empirical/model_view.py`（纯投影，不执行工具、不调用LLM），由Planner/Runtime/Learner分别调用。不要分别在三处维护裁剪规则。

Runtime视图固定为以下分区：

```text
task：原任务目标＋公开输入的值/预览/引用
node：id、mode、真实node_goal、purpose、真实输入/输出接口
bindings：当前实际输入预览、来源引用、missing
handoff：需要的字段、相关消费者、别名及pending实际输出
state：Adapter当前公开状态
calls：当前工具、可调用Program卡片
memory：已做检查/查询的简短记录＋结果引用；最近相关反馈
recovery：原剩余预算、replan/escape余量、循环信号
```

删除顶层和`node_interface`内完全重复的goal、输入副本、已被ResultRef代替的长结果重复体；保留**一份**完整语义，不用`frame_rows/frame_map`让模型再解码列式压缩。

### 9.2 公开任务与内部执行载荷分开

新增/统一 `adapter.model_task()`；默认返回原公开goal与inputs。ALFWorld适配器从同一公开任务载荷展示原目标、目标角色、必要初始房间信息，不把backend env_index、宿主game_file路径、task_signature和native_task_id反复放进每步prompt。

这仅改变模型显示，不改PublicTask、环境reset、physical_key、程序实际inputs或旧资产Schema。既有程序需要完整 `environment_task` 时，`{"task":"environment_task"}` 仍由ValueStore解析为原真实对象；不得一边缩掉字段一边声称旧Program接口不变。

对文件/文档保留输入输出位置、solution contract、已有公共资源；对单轮QA问题/选项/上下文和DocVQA图像，不进行任意截断。对未知Adapter默认保留，不用全局黑名单删除所有名为version、state或location的业务字段。

### 9.3 结果与参数只保存一次，预览必须能取回

复用现有TaskContext和ResultRef，不创建第二个结果存储。对于超出原预览阈值的公共输入、长工具结果或长节点输出：

- 在本题上下文登记一次稳定来源ID；模型预览必须带有效 `result_id/path`，不能返回 `ref=null` 的不可恢复截断。
- `argument_refs/output_refs`的真实解析、Schema检查和程序入参保持。
- 图中大输入优先使用已有task/from引用，不让Planner把完整环境描述/文档列表复制进literal。
- 相同实际来源的结果在recent、completed_results和memory只保留一份内容预览，其余保留引用；不按语义近似合并两个不同结果。
- 新登记的输入引用在模型请求发出前随checkpoint持久化；恢复不改变scope/id/path，同一请求不会引用未落盘对象。
- 原始trace、请求和结果仍保存完整内容；投影不得为获得小数字而删除错误、隐藏动作或必要任务限定。

工具型任务使用原 `read_result`预算和窗口；不增加免费隐式读取。单轮QA必须在一次solver中获得原定公开信息，不用ResultRef把必要题干转为它无法调用的工具。

### 9.4 当前目录、指导和搜索记忆

- 精确动作目录保留**全部当前合法参数**，不截掉“看起来不重要”的地点或物体。
- 稳定语义在Planner/Builder仍完整提供；Runtime当前目录可去掉与同请求静态定义完全相同的重复副本，但名称、参数类型、单位、effect/batchable和调用条件不得丢失。
- 同一指导以Skill ID去重；已绑定能力的原指导优先，其他指导沿用现有相关性与数量上限。不强迫给每题凑满三个不相关Skill。
- 每题工作记忆保留已经检查的scope、查询、动作是否接受及结果引用；去过不等于已检查不存在，旧内容不等于当前状态。
- Program内部原生调用和Agent调用仍进入同一记忆/状态更新路径。
- progress_key/失败key继续读取真实公开状态，不读取压缩显示文本或材料序列化长度。显示变短不能改变无进展规则。

### 9.5 输入降本验收

对本轮82条已记录请求进行离线前后材料对照：分别报告messages/tool-schema字节量、重复字段数、保留的任务限定、工具合法调用集合和引用可解析率。不能把JSON字节减少百分比直接说成token或美元下降百分比。

正常状态、图绑定、程序调用行为不因显示投影而改变；真实模型是否减少输入token、额外read_result或错误，需要正式运行记录。若缩短显示导致模型额外读取很多信息，应如实记录，不继续人为缩窗以强凑成本。

---

## 10. 六Benchmark与比较方法接入范围

| Benchmark | 本轮必要改动 | 不改变 |
|---|---|---|
| ALFWorld | C04/C05学习闭合、C06选择卡片、C07输入投影 | 实际动作/观察、官方won、三模式接管、状态修复 |
| OfficeQA | C03范围grep、C04/C05、C06/C07、答案契约接线 | 每次40命中、原24次调用、批次/finish-only、私有评分 |
| SpreadsheetBench | C05试用参数修复、C06/C07 | 工作区权限、增量产物、相对发布路径、完整解法和全部变体评分 |
| SearchQA | C02短答案、共用学习验证的受影响回归 | 原context、一次solver、固定EM/F1算法 |
| LiveMath | C01完整候选、C02标签答案 | 不加theorem/sketch、一次solver、原题目成员 |
| DocVQA | C02视觉答案契约、图像/受影响回归 | 图像不丢、无隐式OCR、当前不支持模型仍score=null |

公共契约小包不包含检索策略、Skill选择、反思、GEPA优化或Ours专用计划逻辑。八方法公平性只要求公共输入、工具与评分条件一致；各方法如何利用这些接口仍按其原算法。

**本轮不声称已审查所有baseline六基准的完整实现。** 实现者必须把具体入口写入C03.4矩阵；连接公共接口即可的cell完成连接，有独立缺失入口的cell如实保留未就绪，禁止偷偷用Ours Runtime替代后标为原方法。

---

## 11. 数据规模、版本和启动方式

### 11.1 正式规模不变

| Benchmark | Train | Val | Test | Reserve |
|---|---:|---:|---:|---:|
| ALFWorld | 120 | 24 | 134 | 0 |
| SpreadsheetBench | 200 | 20 | 180 | 0 |
| OfficeQA | 120 | 24 | 102 | 0 |
| SearchQA | 300 | 24 | 1400 | 276 |
| DocVQA | 180 | 22 | 201 | 131 |
| LiveMath | 60 | 17 | 100 | 0 |

沿用已经接受的canonical manifest，split_seed=42，run_seed=42/43/44。不同seed成员相同、Train顺序按已冻结规则不同。不要把旧文档的DocVQA目标人数或LiveMath110 Test重新写回配置。[S4]

### 11.2 版本身份

- 将 `empirical/__init__.py::IMPLEMENTATION_REVISION` 更新为 `empirical-v3.1-CF4`，`POLICY_DEFAULTS`、两个base YAML与 `validate_config()`读取同一常量。只改YAML但保留旧常量会被现有严格校验拒绝，禁止这种半接线。原机制profile及CF3 `plan_execution_policy`保持，不增加第二套运行架构。
- 模型视图版本固定为 `runtime.model_view_version=empirical.model-view.cf4`，加入同一POLICY_DEFAULTS并由实际投影实现读取。公共协议身份放入 `experiment` 及run manifest；不要增加当前 `validate_config()`不接受的任意根级字段。版本、Schema、提示与真实handler必须一起接通。
- 在 `run_formal.campaign()`构造的identity中增加 `benchmark_contracts_sha256`、`public_materialization_sha256`，取实际包文件与材料hash；恢复时实际核对，不能只写日志不用。属于现有run身份扩展，不修改旧run。
- 补丁后的真实训练从fresh Bank及新目录开始；旧CF3检查点不跨源码resume。
- 旧Frozen可只读用于执行层诊断，不能成为新正式训练初始库，不能把新卡片展示当新的训练证据。
- 不热修改正在运行的checkout。公共包升级后，受影响cell固定新版本；不同协议的Office/LiveMath成绩不得混为同一条件。

### 11.3 避免再次自动跑全矩阵

现有 `run_formal.main()`以完整矩阵为入口，本轮**不得发明已经存在的 `--benchmark/--seed` 参数**。

新增薄驱动 `experiments/run_single_cell.py`（仅调度，不改算法），参数固定为：

```text
--config        既有main_experiment_v1规格路径
--benchmark     六个canonical名称之一
--seed          42/43/44之一
--datasets      新materialization根目录
--output        本cell的新输出目录
--corpus-root   OfficeQA必需，其他cell可省略
--env-file      可选，原凭证加载逻辑
--resume        仅同commit/config/材料身份可用
--stop-after-val  完成完整Train及Val后正常暂停，不运行Test
```

它复用 `run_formal.campaign()`、现有配置生成和FormalLog，不复制实现第二个Train循环；从当前已配置模型锁读取同一个模型，不自行修改模型设置。若锁中有多个完整模型条目，新增显式 `--model-key`选既有条目；一个条目时可自动选，不按价格或答案选择。

`run_formal.campaign()`新增调度参数 `stop_after_val=False`，默认保持旧完整流程。单cell驱动传入该参数；它只在完整Val结束后暂停，不改变任务/模型/训练配置。暂停时保存 `completion.status=awaiting_test`、`completed_splits=[train,val]` 与阶段结果；不得写completed、不得设置test_started_after_freeze、不得创建Test请求。继续时在同版本使用 `--resume` 且去掉 `--stop-after-val`，复用Train/Val已完成记录，只新增Test；重复日志和计费保持幂等。FormalLog的run_status明确支持此暂停状态，恢复时回到running。停止标记是调度状态，不写进科学config_hash从而破坏合法resume。

启动前必须确认入口确实存在并通过 `--help`与无LLM调用分派测试；最终交付填写实际命令、真实路径和commit。新脚本不得越过完整authority verify，不得自动扩大至其他seed/model。

---

## 12. 必须通过的生产回归

以下为实现后的要求，不是本文件编写时已经通过的测试。沿用现有CF3回归，新增下列差分用例；不要删除旧测试或把失败用例skip。输入完整性可离线检查全资源池，不产生Test求解、学习或费用。

| ID | 场景 | 必须断言 |
|---|---|---|
| T01 | LiveMath正确项缺失、文本存在 | 补入普通候选，公开无答案标记，评分候选一致 |
| T02 | 原候选已完整、正常化执行两次 | 无重复、无内容/顺序漂移 |
| T03 | 标签冲突、缺正确项文本、空题干 | 数据错误清晰，不静默补造/删题 |
| T04 | 固定177题重新materialize | 成员/顺序/physical_key不变；资源hash真实更新 |
| T05 | 最终LiveMath HTTP | 包含全部公共选项；不包含correct_choice/theorem/sketch |
| T06 | SearchQA同一短答案fixture | 输出格式指令在实际请求；一solver，无额外提取模型 |
| T07 | DocVQA/单轮恢复 | 原图像完整；同一solver恢复不重解；能力锁不变 |
| T08 | grep不传paths | 与旧全局顺序、命中及40上限一致 |
| T09 | 只给后部目标文件 | 不被前部文件40命中截断；不读取scope外文件 |
| T10 | 空scope、重复/乱序paths | 空就是空；去重与稳定排序；一次调用 |
| T11 | 无效正则/路径/越界 | 普通拒绝、无部分结果、无全局fallback；真实I/O错误不吞 |
| T12 | 多文件scope及CRLF/中文 | 40是总上限；offset可直接read到对应文本 |
| T13 | 纯读批次/恢复 | 1—3独立执行、逐次预算；已完成子项不重发 |
| T14 | Planner/Runtime/Builder中grep | 同一个字段与语义；ctx返回仍是原ToolResult |
| T15 | Ours与各已接入baseline | 同输入scope产生相同结果/预算；协议hash相同 |
| T16 | usable＋build/trial＋第三案例 | 不并第三槽、不生成；原两槽/资格/Program不变 |
| T17 | T16同时有合法Workflow提议 | Workflow更新不能因no-op被跳过 |
| T18 | candidate/显式repair/新版本 | 不被usable短路吞掉；仍按原试用和恢复额度 |
| T19 | 重启重复提交T16 | 不新增Program、试用或重复收费 |
| T20 | 新Skill试用缺必需字段 | 在接收前反馈，原有一次repair，不先写坏资产 |
| T21 | repair补齐合法真实输入 | 恰好进入正常Builder路径；不额外求解任务 |
| T22 | 无适用案例defer | waiting_example、零Builder；不伪造第二个来源 |
| T23 | 试用参数/前缀来自Val或不存在案例 | 明确拒绝；不写正向记录 |
| T24 | 第二个适用Train到来 | 沿现有规则试用/可用化；不放宽两物理案例要求 |
| T25 | Dynamic只参考整题Skill | 卡片如实标dynamic，不能被展示逻辑自动授权 |
| T26 | 两个usable程序新组合 | 一次规划后自动传值；不加LLM确认或整图验证 |
| T27 | 同时有纯动态Workflow与适用程序 | 实際请求清楚展示区别；程序不被菜单截掉 |
| T28 | 实际生成的旧Heat/双对象程序 | 只读Bank与源码不变，选中后仍能执行；fixture选择不冒充自然选择 |
| T29 | 模型视图去重 | 用户目标/数量/身份、必需输入、合法调用集合保持 |
| T30 | 长输入/结果预览 | 每个截断值有可解析ref；新引用在请求前落盘 |
| T31 | 普通字典含literal/from/version | 不被显示裁剪或自动拆包破坏 |
| T32 | 单轮上下文与视觉材料 | 必需输入不转成无法调用的ResultRef，不额外读工具 |
| T33 | 当前公开状态与进展检测 | 不读取显示压缩文本；原MOVE/INVENTORY、循环纠偏不回退 |
| T34 | 文件发布和最终提交 | 只读检查不注销输出；全部变体评分；不反复Dynamic重做 |
| T35 | 冻结/恢复/待办 | 长期hash不变；局部修改合法；试用checkpoint与父任务隔离 |
| T36 | 单cell驱动及阶段暂停/恢复 | 只运行选择cell；Val后awaiting_test，续跑不重复Train/Val或费用，只启动原Test；不触发全矩阵 |
| T37 | 旧结果与公共资源 | 旧run/hash保留；修复材料不能伪装成旧版本resume |
| T38 | 最终打包安装 | 公共小wheel不导入Ours；薄入口从安装环境可运行 |

T01/T16/T20至少建立“旧源码或旧数据失败→新实现通过”的反例对照。其余要求测试最终消费者或真实handler，不只测一个手写helper。

测试日志同时区分：纯逻辑、生产模块＋fixture、真实Adapter/Docker、真实模型。工程fixture的usage不纳入实验成本，手写程序不导入实验Bank。不得以T01—T38通过声称六基准精度最高或成本已低于50k。

---

## 13. 实施顺序、代码清理和交付

### 13.1 固定实施顺序

```text
保全旧run/Bank/材料和commit
→ C01公共数据正常化与C02答案契约
→ C03范围grep及公共包、baseline薄接口
→ C04/C05统一实现请求分类与一次语义修复
→ C06能力卡片和C07阶段投影
→ C08身份、材料、单cell入口和恢复接线
→ 现有CF3＋本轮差分生产回归
→ 固定干净commit和协议hash
→ 按第14节运行，不增加第三套方法
```

### 13.2 删除被替代代码，不保留两套同时生效

- 删除prepare中的LiveMath裸复制choices分支；所有入模材料由同一正常化路径生成。
- 替换单轮solver的无格式通用说明，不在结尾叠加彼此冲突的旧新模板。
- 将Office扫描实现移到公共包后，删除Adapter内重复扫描循环，保留唯一错误/权限转换入口。
- 删除先合并再判断usable的重复分支；提议验证和持久化调用同一分类函数。
- 删除Planner/Runtime重复拼接整资产/整输入的模型材料路径；实际执行数据、日志和checkpoint不删。
- 不保留“为了兼容新需求再走一次旧E1验证”的fallback，不新增legacy证据字段。
- 根README及CF3后续说明更新为CF4修改范围；旧CF3规范存档，明确其接管/恢复边界仍有效。

### 13.3 交付内容

1. 完整commit、resolved config、修改/删除清单、公共包构建hash和真实运行入口。
2. 原始pytest/JUnit、T01—T38对应函数/测试名、旧错新对对照；未运行部分写明。
3. 数据完整性报告与材料新旧hash，split authority不变的证据。
4. 最终HTTP样本：短答案、完整选项、grep scope、Planner卡片、Runtime引用、学习修复；不含密钥。
5. baseline公共契约矩阵：哪些cell真实接入，哪些尚未接入，不以Ours结果代替。
6. 已运行时提供逐题原始请求/响应/usage/trace、Bank、学习待办、试用及正式评分；未运行就不生成虚假的效果报告。

---

## 14. 下一步运行：不再重复“均衡12＋6直到满意”

### 14.1 先完成不付费的必要检查

完整资源审计、契约与控制流回归、原请求投影对照、已有成功Program接管回归先完成。这些不需要新训练，也不需要先得到50k的结果。

确有确定性运行故障时先修对应路径；普通答错、Program生成失败或高成本不是“所有benchmark都不能启动”的总门槛。

### 14.2 真实运行计划

**不再额外重跑第三轮ALFWorld12＋6，也不以旧Frozen单独跑六题来代表新学习效果。**

| 顺序 | 单cell运行 | 规则 |
|---|---|---|
| 首批 | DeepSeek × ALFWorld × seed42：fresh Train120→Frozen Val24 | 保留固定完整Train顺序；可以在正常任务边界检查日志，不重做已完成前缀 |
| 同批可独立推进 | 修复后的SearchQA、LiveMath × seed42，按正式完整Train/Val | 单轮接口已经正确后运行；旧LiveMath输入缺陷结果不混入 |
| 文件类 | OfficeQA、SpreadsheetBench × seed42，各自fresh完整Train→Frozen Val | 公共检索/试用/提交回归通过后运行；不等ALFWorld每题都低于60k |
| 当前不启动 | 当前模型锁下的DocVQA | 维持unsupported；后续有明确支持图像的已锁模型时再按同协议运行 |

首轮各cell使用 `--stop-after-val`，完成全部Train和正式Val后正常暂停。审查工程完整性、程序覆盖与费用后，同版本去掉该标志并 `--resume`，由原campaign一次性执行Test。不得用Test调选择、降权、修程序或调预算；若根据Val作了方法修改，则应建立新版本和新fresh运行，不能把原结果当同版本正式结果。

本文不自动启动收费请求；实现交付后按用户的启动指令执行。不要同时铺满所有模型×三个seed。一个已就绪cell不因另一个适配器未完成而被无关阻塞。

本轮新增阶段暂停接口须完成T36测试后才能使用；不能仅在启动指令里写一个源码不存在的flag。代码和材料一旦改变，另开新身份，不能续写旧版本。

### 14.3 扩规模时看什么，不强求小样本成熟

- 统计的是能力从提出、合法试用、usable到真实调用的链，不是只数Skill/Workflow文件。
- 普通取放/Cool已有单例正向，更多合适Train可能补第二例；Clean未生成程序则必须如实记覆盖缺口。不能把二者一律解释为“差一个样本”。
- 已usable能力是否被Planner明确选择、或在Dynamic中被显式调用，必须由实际请求证实。
- 高费用分别归入正常搜索/读取、重复输入、Planner修复、程序失败恢复、学习/试用；所有费用保留。
- 无程序但正常探索较长，不等于适配又错；反之字段矛盾、漏选项、错误恢复身份不会因样本变多而消失。

### 14.4 成本与质量怎样判定

每个benchmark单独计算冻结阶段每题总tokens（Planner＋Runtime＋真实恢复等，reasoning不重复加），包含失败题。训练执行、Extractor、Builder、隔离试用另外完整报告，不与冻结均值混用；未知计费继续记未知。

均值≤50k是目标，60k是希望控制的上界，不是提前停题的阈值。准确率/主评分同时报告；单轮DocVQA使用其既定主分数，不强制套ALFWorld成功率。方法是否优于baseline看同模型、同成员、同公共输入/工具/评分的完整结果，而不是六题最佳一次。

若工程已正确但正式成本/质量未达目标，结论就是效果未达标；分析实际程序覆盖、选择和剩余搜索，再另行决定方法实验。不能在本文的名义下自动换模型、调低high、增加修复、循环训练到成绩满意。

---

## 15. 本文件编写时已做的工程可行性检查及限制

本轮新增离线检查共 **18项通过**，记录见随附 `cf4_feasibility_results.json`：

- 使用与仓库blob `82f8449135c98a8519c1772d48982520801197b6`一致的Learner类，复现第三试用槽错误。
- 独立副本仅调整usable处理顺序：不报错、仍只两个槽、零Builder调用，**合法Workflow仍保存**；candidate仍不能超过两槽。
- 使用实际 `_preflight_binding` 检查缺必需参数会拒绝。
- 选项补齐参考实现验证缺项恢复、幂等、无公开正确性标记、冲突拒绝和不修改源数据。
- scope grep参考实现验证无scope兼容、显式scope、空scope、去重、40条总上限、Unicode offset与越界拒绝。

其中Learner依赖使用受控替身，scope grep与选项函数是方案参考实现，不是已合并的生产Adapter；没有调用真实模型、Docker或环境，也未执行完整评分。18项不代替第12节生产回归，不证明新选择提示词、程序生成质量或50k目标已经有效。

**可确认的是已知错误有具体可实现的闭合路径；不可提前确认的是六benchmark最后都拿最高分。** 本轮继续推进现有方法，不用工程不确定性掩盖方法风险，也不以方法效果尚未测量反复推翻已批准架构。

---

## 16. 来源索引

- **[S1]** 本对话最新三份只读审查包：`ALFWorld_CF3_seed42_12Train6Val_review_20261006_105336(1).zip`、`ALFWorld_CF3_Train12_originalFixedVal6_review_20261006_111528.zip`、`CF3_DeepSeek_multibench_2Train1Val_review_20261006_105336.zip`；对应核算文件 `CF3_20261006_运行结果核查与缺陷复现.json`。本轮继续使用其原始请求、Bank和失败提议，不把R10.x记录当新结果。
- **[S2]** Ours固定源码 `bc63d02bc3bb0988cdbd15a203a068fc162f0b9b`：`empirical/learner.py`、`bank.py`、`planner.py`、`executor.py`、`system.py`、`task_context.py`、`prompts.py`；`harness/benchmarks.py`；`experiments/prepare_benchmarks.py`、`canonical_manifest.py`、`run_formal.py`。
- **[S3]** SkillOpt固定上游 `fa4ca184573e42ec11472959dd57422381418096`：`skillopt/envs/livemathematicianbench/dataloader.py`，尤其正常化函数；Ours现有锁定SearchQA/LiveMath/DocVQA评分器。
- **[S4]** 已接受的公共实验划分、`main_experiment_v1_alignment`与`Baseline_多Benchmark实验冻结规范_v1.1_中文版.md`。人数以已经接受并实现的canonical清单为准，算法超参数不在本轮重定。
- **[S5]** 已批准CF3规范：参考与执行绑定分离，保留已验证接管、局部patch、一次重规划、一次Dynamic退路；CF3交付报告用于确定当前已保留回归范围。

**最终执行原则：先把真实输入、可用程序状态、调用接口和模型看到的材料对齐，再通过正常训练增加可复用覆盖；不让“只要多跑一点”代替确定性修复，也不让“必须证明每一步”重新绑住真实执行。**
