# AtomicSkill LiveMath LM1 复审：最小修复与后续实验决策实现文档

**文档日期：2026-10-10。状态：复审结论与待实施补丁说明；本次未修改远端仓库，未启动真实模型调用或恢复正式实验。**

本文件承接已实施的 LM1 规范，完整规定本次收尾的代码改动、保留行为、离线验收、成本与实验决策。现有 LM1 真实批次已经结束；本文件不新建另一轮重编译或 Val 试验。当前要修的是两处可复现的工程边界，不能把本轮低准确率、无指导覆盖和 low 降本失败一并包装成“代码还没修好”。

## 结论与实施范围

LM1 的主体修复已经实际落地。新提案走具名工具提交，成功答案与公开候选绑定，来源检查、Train-only IDF、Frozen 保存、BankView 对照、限定 repair 和整批预算都已在真实请求及保存记录中得到核对。当前没有证据表明这次分数被算低、正确答案被解析丢失，或有效指导被调度程序漏注入。[E1][E2][E3]

仍需修改 `validate_proposal` 中两个类型不安全的操作，以及 `validate_config` 对新学习和新检索开关的组合约束。前者可让错误类型的模型字段越过受控拒绝边界，后者可让未来配置误用旧检索路径。本轮实际响应没有触发前者，实际配置也没有触发后者，因此它们不是本轮 A 仅答对 4 题的原因。[E4]

本轮最主要的方法结果是：9 条通过来源检查的资产对 Val17 的有效覆盖为 0；把每项资产自己的源题排除后，对已完成 Train60 的跨来源选择覆盖也为 0。这说明当前资产与适用性规则组成的方法，在这些公共题干上没有形成可用的迁移干预。它并不证明所有可能的 Skill 方法都无效，也不证明不同检索器永远找不到迁移，但足以否定“重复同一轮 Val 就能解决问题”的继续方式。[E3]

本次实施的完成条件是两处补丁和明确的结果呈现通过固定离线验收。新增真实 Runtime 题目数为 **0**，新增真实学习或 checker 调用数为 **0**，新增真实 HTTP 和模型 tokens 均为 **0**。不以资产数增加、Val 准确率提高或必须观察到正收益作为关闭工程问题的条件。

## 审查依据与版本身份

| 对象 | 已核对的身份与范围 |
|---|---|
| 实际执行代码 | `0099b4a44f415a8e7e418a003ef36d95028ed9b1` |
| 复审时 GitHub main | `993d6cd0c21a297e55343aab4a65009494f0df16`；其父提交为上述执行代码，仅新增两份 LM1 报告，未再改运行代码 |
| 新审查包 | `livemath_lm1_review_20261010.zip` |
| ZIP SHA256 | `799eb652b21b794a395d59d288105024176dde883d4d6c53e3ffff2fcd92fb68` |
| 包内校验 | 244 个声明文件的 SHA256 全部一致 |
| 源码核对 | 本地审查快照中 133 个文件的 Git blob hash 与固定提交一致，覆盖本次相关源码、配置和测试；未将报告中的“已实现”替代源码检查 |
| 原经验来源 | run_id=`555c2e4b69e84ed4b58efe4eacebab26`；60 条已完成 Train 经验 |
| 批次 manifest hash | `258ea04e39b885bdcf0de4392996ac3b51798aac4961fe679f79dc75378610cf` |
| 派生 Frozen digest | `329a6318e0c983fe16a52973adae9ff7f05b04590b3f51f5ed3e4d43edca0db7` |

源码可直接核对 [LM1 实现提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/commit/0099b4a44f415a8e7e418a003ef36d95028ed9b1) 与 [报告提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/commit/993d6cd0c21a297e55343aab4a65009494f0df16)。本文件所称“当前代码”均指上述固定运行代码，避免 main 后续移动造成歧义。

包中的 `109 passed, 3 skipped` 是实施方保存的独立 WSL 回归日志。本次复审独立完成了原始最终回答、预算、来源、Frozen、选择规则及类型边界的零模型调用核验，没有将该日志写成复审方重新执行了同一套 pytest。复审环境缺少完整测试依赖，类型边界使用从固定源码提取的实际函数和实际 schema 执行；选择函数回放保留原始 Bank/selector 实现，并阻止任何外部模型请求。[E1][E3][E4]

## 本轮到底测到了什么

### 分数、成本与空答

| 实际条件 | 完成题数 | 正确 | 合法错误 | 空答 | Runtime tokens |
|---|---:|---:|---:|---:|---:|
| A：guidance_off / high | 17 | 4 | 9 | 4 | 365,818 |
| B：guidance_on / high | 0，按预声明分支跳过 | 不适用 | 不适用 | 不适用 | 0 |
| C：guidance_off / low | 17 | 2 | 14 | 1 | 234,951 |

这些数字由 34 份实际 HTTP 审计中的公开最终 content 重新解析得到，并与 canonical 候选身份重建的 gold 比对；实际 HTTP final、逻辑 response、trace prediction、轻量结果相互一致。A 和 C 的实际请求都包含原题干、原候选与空 guidance，没有在名字标为 guidance_off 的条件中偷偷加入资产。[E1]

A 与 C 的配对为：两组均正确 2 题，只有 A 正确 2 题，只有 C 正确 0 题，两组均错误 13 题。两组最终预测在 7/17 题不同。C 节约 35.7738% tokens，但正确数从 4 降为 2；固定的成本候选规则要求同时满足“tokens 至少降低 25%、正确数不降低、空答不增加”，因此继续保留 high。这个结论是按预声明规则选成本候选，不能上升为 high 的总体统计优越性，也不能把 low 写成准确率等效。[E1][E2]

5 次空答均为 `finish_reason=length` 且用完 32,768 completion tokens；A 的 4 次空答合计消耗 134,261 tokens，C 的 1 次空答消耗 33,904 tokens。现有记录没有显示非空正确 final 被评分器丢弃。不得从 provider-private reasoning 中补造最终答案，不得补答后覆盖旧空答，也没有依据在本次收尾中增加一次 Runtime 求解或提高 completion cap。[E1]

以 gold 候选的实际含义做事后描述，8 道“其他选项正确，但可以证明更强结论”的题，A、C 都答对 0 道；其余 9 道普通候选题，A 答对 4 道，C 答对 2 道。这些标签只用于结果分析，不进入提案、检索、Runtime 提示或候选选择。[E1]

A 已是真正无指导条件，仍出现上述失败，所以“旧错误指导是全部低分的原因”已不足以解释当前现象。当前公共任务上的基础作答质量与这一类候选的判断也存在明显困难。不能由 17 题推导远端模型的能力上限，更不能把新 Val 的 4/17 与旧 Test 的 20/100 相减，声称修复带来了某个准确率提升。

### 本轮没有测得指导收益

Frozen 中确实存在 9 项合规资产，9 项均通过实际 `checked_asset` 检查，Train-only 统计 N=60、df 与源公共题干 hash 均可重算。17×9 的 153 组匹配审计与实际 `Bank.select_guidance` 返回值完全一致。不存在资产漏载、冻结时统计丢失或 BankView 意外关闭 B 的证据。[E3]

153 组中，134 组不匹配任何 scope 词，17 组只匹配一个词，只有两组匹配两个词。这两组分别是环论题与 toric degeneration 经验的 `special/fiber`，coverage=0.4882578993，以及 p-adic representation 题与 Kummer 经验的 `finite/extension`，coverage=0.4855671089。它们既未过原定 0.5 门槛，内容上也不足以据此确认同一个数学结论适用。[E3]

因此不改为 0.48，不改为一个词即可入选，不回到强制 top3，不把“更强结论”等通用选项词加入 scope，也不根据 Val 正误挑阈值。将这两组强行放行，只能证明注入发生，不能证明方法有效。

对现有 60 道 Train 公共题干，复审又使用最终 Frozen 的九项资产做了一次离线选择，并对每个目标题排除与其 physical_key 相同的资产来源，结果跨来源入选仍为 0/60。这是**最终资产的事后覆盖诊断**，不是严格在线前缀评测，也不是额外的独立验证集；它没有模型调用，只用于判断现有内容是否具备可见的跨题使用机会。[E3]

9 项资产的主体结论未发现上一轮那种明确的答案反写，但大量内容依然是特定定理、对象与假设的条件化复述。来源相符只说明对该已验证提交的有限一致性检查通过；它不自动获得其他数学问题的适用性。LM1 已经解决了一部分“来源可能被反写后无条件注入”的风险，却没有凭空创造跨领域推理能力。

### 受控拒绝与 checker 的证据边界

16 条成功来源得到 9 次 upsert、1 次 no_change、6 次 rejected。拒绝原因已经具体定位，不能统一归类为“checker 发现了数学错误”。[E3]

| 事件索引与源题 | 实际拒绝位置 | 本轮如何解释 |
|---|---|---|
| 25，`livemath:202512:40` | scope 词不在规范化后的源题干，repair 后仍不满足 | 受控提案拒绝 |
| 29，`livemath:202602:38` | scope 词不满足源题干定位要求，repair 后仍不满足 | 原 PDE 反写问题对应来源，但新提案未进入 checker |
| 47，`livemath:202601:16` | scope 中 `blows` 仅出现在候选中，不在题干中；整批 repair 已耗尽 | 按固定边界拒绝，不能多给一次 repair |
| 56，`livemath:202602:18` | rationale 为 282 字符，超过 240 | 原极小嵌入反写问题对应来源，但新提案未进入 checker |
| 38，`livemath:202601:14` | checker 的 reason 为 411 字符，超过 400 | checker 结构不合规，未发布 |
| 48，`livemath:202602:33` | checker 的 source_quote 多了一层 JSON 转义，不是原文精确子串 | 引用合同拒绝，未发布 |

11 个真实 checker 都返回了 `supported`，其中两个又被 host 的结构/引用检查拒绝；本轮没有真实 `contradicted` 或 `insufficient_evidence` 的案例。两项已知旧反写来源均在 checker 之前被拒绝。因此不能把“那两个旧错误没进新 Bank”写成“真实 checker 已证明能识别这两项反写”。假 checker 冲突回归验证的是拒绝控制流。[E3]

上述六次拒绝没有造成进程异常，也没有让错误资产发布。它们会影响产出率，但本次不通过扩大长度、增加 repair、重发六条来源或自动改写引用来追回资产。正常终止的受控拒绝不要求追求零拒绝率。

提案材料已经提供 `normalized_source_topic_words`，但动态 schema 尚未把 `scope_terms.items` 限制为该词表 enum。未来新学习版本可以把这个约束前移到提交 schema，并显式要求从词表复制规范化 token，以减少无意义提案；仍需 host 精确验证。这属于提交效率优化，不能保证模型结构成功率，也不改变当前零迁移结论。**本次收尾不启用新提案协议、不重编译现有 60 条经验。**

## 必须修改的代码

### 提案校验：先确认类型，再执行集合或正则操作

**落点：** `src/atomic_skillgraph/empirical/choice_guidance.py` 的 `validate_proposal`，固定提交中的第 90、102 行附近。[S1]

函数先收集 schema 错误，然后继续检查字段语义。当前有两处操作未保护字段类型：

```python
old_id not in {a["id"] for a in related}
re.search(pattern, proposal.get("guidance", ""))
```

模型返回的 JSON 可以语法合法、字段类型非法。例如 `existing_skill_id=[]` 会导致不可哈希对象的 TypeError，`guidance=123` 会导致正则参数的 TypeError。它们绕过 `agent` 与 `_learn_grounded_choice_guidance` 已有的 ValueError→有界 repair/rejected 流程，最终可能被登记成学习阶段的工程失败。这与“模型提出了一个不合规资产，按合同拒绝”是不同结果。[E4][S1][S2]

最小修改如下。保留现有错误收集、字段长度、scope、答案标签、related ID 约束与返回行为：

```python
old_id = proposal.get("existing_skill_id")
if old_id is not None and (
    not isinstance(old_id, str)
    or old_id not in {a["id"] for a in related}
):
    errors.append(
        "existing_skill_id was not offered as a checked related asset"
    )

# 其余 schema 与非空检查维持原实现。

guidance = proposal.get("guidance")
if isinstance(guidance, str) and re.search(
    r"\b(?i:select|choose|answer|option)\s+[A-Z]\b",
    guidance,
):
    errors.append("guidance must not bind instructions to an answer label")

if errors:
    raise ValueError("; ".join(errors))
```

不得把错误类型强转为字符串，例如把数组变成 `"[]"` 后继续接受；不得用 `except Exception: rejected` 吞掉磁盘、数据库、源数据不一致等真正的工程错误；也不得直接降低 schema 要求。字段类型错误仍须在最终错误集合中出现，仍然属于同一逻辑提案，其 repair 受单提案一次、原批次总额四次的限制。

本次复审已用固定源码的实际校验函数与实际 schema，对 7 个字段各输入 8 类代表性 JSON 值，共执行 56 个离线案例。原实现为 10 个 TypeError、42 个 ValueError、4 个合法接受。隔离候选仅加入上述两个类型护栏后，同一矩阵变为 0 个 TypeError、52 个 ValueError、4 个合法接受，合法接受集合保持不变。此结果证明该异常边界可通过小补丁修复，不证明完整运行系统的所有异常边界都已覆盖。[E4]

### 配置校验：新学习开启时必须使用新检索

**落点：** `src/atomic_skillgraph/empirical/system.py` 的 `validate_config`，在 choice_guidance 两个配置块完成默认值和类型校验后、创建 Bank 或 Provider 前。[S2]

当前代码允许以下配置：`learning.choice_guidance.enabled=true`，同时 `runtime.choice_guidance.enabled=false` 或整个 runtime choice 配置缺失。学习端会产生新的 LM1 资产，single_answer Runtime 却进入旧 `retrieve_guidance(limit=3)` 路径，跳过本次新增的适用性筛选，且模型材料不带新版 applicability/qualification。这是未来配置误用的真实通路。[E4][S2]

增加一向约束：

```python
choice_learning_on = (
    config["learning"].get("choice_guidance", {}).get("enabled") is True
)
choice_runtime_on = (
    config["runtime"].get("choice_guidance", {}).get("enabled") is True
)
if choice_learning_on and not choice_runtime_on:
    raise ValueError(
        "Choice guidance learning requires choice guidance runtime selection"
    )
```

约束必须在 validate_config 阶段、Bank 创建/写入和 Provider 构造及任何付费请求之前生效。受支持的 runner 继续在写入正式运行身份前校验配置；外围 launcher 创建空目录或记录配置错误，不作为本补丁失败。不得在校验器里静默把 runtime 开关改成 true，也不得隐式回退到旧检索。

允许 learning 关闭、runtime 开启的只读使用方式；普通旧配置保持旧行为。A 条件继续通过 `BankView(..., "guidance_off")` 关闭指导，同时保留启用的公共选择题语义与新运行路径。**不能为了构建 A 条件而关闭 `runtime.choice_guidance.enabled`。** 当前真实 A/C 正是使用 BankView，因此本补丁不改变现有 A/C 的含义或得分。

本轮真实配置中 learning/runtime 均为 true，没有发生这种误用。新增约束用于后续启动前拒绝不一致配置，不应被描述成当前低准确率的根因。

### 已检查、无需继续扩展的恢复问题

`_learn_grounded_choice_guidance` 和 `learn_from_completed_record` 对已完成结果存在早返回，但现有 LM1 runner 已在外层绑定 code、config、source 与 manifest；普通正式入口也校验任务、配置和代码执行身份。当前受支持入口没有证据允许把不同语义的付费完成结果无条件复用。[E4][S2][S3]

本次不再增加另一套恢复版本体系，不把内部 API 的防误用加固升级为新的正式实验阻断项。实施补丁后应使用新的 Git commit；旧批次只读归档，原 `check-offline/run` 不应在更换代码后被绕过身份校验强行续跑。已结束的 LM1 批次不需要续跑。

## 报告与审计的修改

**落点：** `src/atomic_skillgraph/experiments/run_livemath_lm1_validation.py` 的 `summarize`，以及消费其结果的报告生成处。[S3]

当前 B 未运行、AB 配对为空时，空集合求和产生 `B_only_correct=0`、`A_only_correct=0`、`net_gain=0`。现有 `method_signal=no_effective_method_intervention` 是正确的，但只读取数字的后续汇总可能误写成“方法净收益为 0”。应把估计是否存在与数值分开。

```json
{
  "method_comparison_status": "not_run_no_exposure",
  "paired_n": 0,
  "B_only_correct": null,
  "A_only_correct": null,
  "net_gain": null,
  "method_signal": "no_effective_method_intervention"
}
```

B 的运行题数为 0、运行成本为 0，仍保留真实的零值；B 的准确率和 AB 方法效应显示“不适用”。若未来某批次发生部分配对，使用 `incomplete` 状态并单独保存 `observed_pairs` 的描述性计数；不能把部分结果自动当作完整方法比较。完整配对时才提供完整批次的净收益统计。

对已归档的本轮 `result.json`、manifest、budget、requests、trace、Frozen 和校验清单均不原地回填。生成单独的复审/导出结果，引用 parent manifest hash 与原文件 hash，并注明只调整解释字段、新增调用为 0。新代码的后续运行可以直接输出上述显式状态。

来源拒绝报告应分别给出 `proposal_schema_invalid`、`proposal_scope_invalid`、`grounding_schema_invalid`、`grounding_quote_invalid`、`grounding_contradicted`、`grounding_insufficient` 等统计含义，至少在离线报告中根据已保存字段明确区分。不要把六个受控 rejected 都写成语义 checker 的正确拒绝，也不要将工具名正确的 31/31 写成 schema 或数学正确率 100%。

候选“含有元选项”和“gold 属于元选项”应使用不同字段名；本次关心的 8 题结果是后者，只能在事后评分报告中计算。无需为分层统计改 Runtime 输入或采样规则。

## 不改动的实验与方法合同

| 合同 | 本次收尾要求 |
|---|---|
| 公开输入与评分 | 原题干、五个候选的实际正文与顺序、choice_seed、projection_version、scorer 均保留；不补造或覆盖旧 final |
| 学习来源 | 仅原成功提交可进入当前正向来源路径；错误/空答不被“纠正”后伪装成成功经验 |
| 来源检查 | 严格具名工具、唯一提交、完整结构与真实引用；supported 仅是有限来源兼容检查 |
| 检索 | 原词规范化、至少两词、Train IDF coverage≥0.5、最多两项、总注入≤2400 字符；无匹配为空 |
| 统计与冻结 | N/df 来自已完成 Train 公共题干、physical 去重；Frozen 只读并保持 digest |
| Runtime | 单题一次真实作答；本批次已结束；后续候选保留 high/32768 |
| 学习预算 | 已完成批次的 16 proposal、11 checker、4 repair 不增加；不为追回六条拒绝来源重发请求 |
| Program | 既有两条独立正向来源资格、候选/usable 状态和真实接管规则保持原定义；不把文字 guidance 的来源检查替代 Program 资格 |
| Seed 与血缘 | seed42 的离线重编译产物不冒充 seed43/44 的独立在线训练产物 |
| Test | 既有 Test20/100 原样保留；本次不调用 Test，不使用 Test 正误调 scope 或修正文 |
| 对照公平性 | 公共选项理解提示和推理条件必须明确匹配；共同提示变化的收益不能全部归因于学到的指导 |

这些要求足以界定本次修改，不新增“必须有多少条 Skill”“必须答到多高”“必须每题注入”等验收门槛。

## 固定的零调用验收方案

### 测试规模与预期

| 检查 | 固定规模 | 预期结果 |
|---|---:|---|
| 提案字段类型矩阵 | 7 字段×8 值，共 56 例；保留本次复现输入 | 原 10 个 TypeError 全部改为合同 ValueError；仍为 4 接受、52 拒绝；无自动类型转换 |
| 配置组合 | learning 开/关×runtime 缺失/关/开，共 6 例，另加一例完全旧配置 | learning 开且 runtime 缺失或关的 2 例在初始化前失败；其余允许；旧配置解析行为不变 |
| 类型错误的集成路径 | 5 个受控 transport 场景，见下表 | 错误进入同一逻辑提案的 repair/rejected 流程；真实 HTTP=0；真实工程异常不被吞掉 |
| 历史响应合同回放 | 原 31 份学习/检查物理响应 | 工具名、原 schema 及 host 校验结果与原记录一致；不发布原来 rejected 的资产，不新增 repair |
| 历史分数和成本导出 | 原 34 条 Val、65 条物理 HTTP | 仍 A=4/17、C=2/17、688,595 tokens；新增字段不重评分为另一结果 |
| 选择规则回放 | 原 17×9=153 对；另含 60 条公共 Train 的排除本源诊断 | Val 选择审计逐项等于原记录；仍 0/17 暴露、0/60 跨源覆盖；不能以变成非零为通过条件 |
| B 未运行的汇总 | 当前这一个 completed_fixed_batch 记录 | paired_n=0，方法效应字段为 null/N/A；B 题数和成本仍为 0；成本候选仍 high |
| 不可变性 | 原审查文件与 Frozen | 运行回放前后文件 hash 不变；所有新增报告写到独立输出路径 |

类型矩阵字段为 `decision`、`existing_skill_id`、`goal`、`guidance`、`scope_terms`、`applicability`、`rationale`；每个字段的固定值为 `123`、`1.5`、`true`、`null`、`{}`、`[]`、`["manifold", null]`、`"x"`，其他字段采用合规 fixture。这里的 56 例是合同边界案例，不是新的 56 道数学题。

| 集成场景 | 固定响应/事件 | 必须观察到的行为 |
|---|---|---|
| 非字符串 guidance | 首次提案 guidance=123，固定 repair 返回合法 no_change | 两个模拟提交；最终 no_change；不记 failed_engineering；无资产发布 |
| 不可哈希 existing ID | 首次提案 existing_skill_id=[]，固定 repair 返回合法 no_change | 同上；非法 ID 不被字符串化接受 |
| repair 额度已尽 | 使用已耗尽的模拟账本，返回非法字段；然后处理下一条合法来源 | 非法来源 rejected 且不补 repair；下一来源仍能处理；无整批异常退出 |
| 同身份重入 | 重入已完成的上述模拟事件 | 复用已有结果；模拟调用数、计费条目、资产数不再增加 |
| 真实工程错误保留 | 在 Bank 写入或明确的内部边界注入 OSError | 仍走工程失败路径；不能因为扩大 catch 而变为普通模型 rejected |

模拟 transport 必须在测试入口拦截实际 HTTP 请求，任何未登记的网络调用立即使测试失败。不要通过提供真实 API key 来完成这些案例。可以复用现有 `tests/test_lm1.py`、相关 provider/contract fixture 与 Bank 临时目录；无需为本补丁重新执行所有高成本 benchmark 或重新跑容器中的数学任务。

实施方还应运行现有受影响的 LM1 测试。修复仅涉及两个叶操作、一个配置约束和报告输出时，不强制扩展到全部历史 Program/容器套件；只有实际改动共享 Provider payload 或 Program 路径时，才针对该改动补测。本次没有这样的改动要求。

### 为什么不再安排真实小批次

这次两个代码缺口的触发条件是确定性的字段类型和配置组合，已有真实模型请求无法更有效地证明其修复。已有 65 次真实请求已经足够复核新提案结构、计费、A/C 路径、空答和零覆盖分支；重新生成相同来源或重做 Val 只会改变随机输出、增加成本，不能代替上述合同测试。

若未来要证明 checker 的语义判错率，或测试新的可迁移学习机制，必须把它作为新的、先定义干预与对照的研究问题。当前批次没有提供这两项正证据。**本文件不以“顺便再测几题”启动该研究，也不把换一个批次名字当作延续本次修复的理由。**

当前验收通过后，即结束 LM1 工程收尾。无需再向审查方提交另一整包真实运行结果来获得“准确率足够高”的放行。

## 成本结论与后续运行

| 成本组成 | 实际新增 HTTP | 实际 tokens |
|---|---:|---:|
| 新 proposal | 16 | 46,411 |
| 新 grounding | 11 | 27,867 |
| 新 proposal repair | 4 | 13,548 |
| 本轮学习合计 | 31 | 87,826 |
| A/C Runtime | 34 | 600,769 |
| 本轮新增总计 | 65 | 688,595 |

原 Train Runtime 1,247,149 tokens、原 Train 总成本 1,411,603 tokens 与本轮新增 87,826 的重编译成本应分别保存。本轮从已完成经验出发，确实避免再次求解 60 道源题；不能据此声称在线 Train60 只需 87,826 tokens，也不能把继承成本清零。当前有限验证的总成本已有完整账本，不需要再用实验估计。[E1]

现在可确定的降本动作是停止为已关闭的 LM1 批次重复付费、用已有记录完成工程验收，以及不把 low 当成已满足保准确率要求的默认配置。没有证据支持在本次收尾中再试 medium、增加 cap、减少 cap、加一轮补答或扩大 Train。它们改变的是推理策略与研究条件。

**对 LiveMath×3 seed：不建议现在直接扩大这一配置，理由是投入价值和方法覆盖证据不足，并非要求先把准确率修到某个数值。** 两个工程补丁通过后，工程侧可以按固定版本运行，但现有证据并不显示给同一做法增加 seed 会解决跨题迁移。当前最合理的记录是“该派生配置在 Val17 无有效指导干预；无指导 high/low 的成本候选比较已完成；保留 high”。

如果研究方案原本要求完整三 seed 才能报告总体结果，保留 seed42 的诊断记录并明确 LiveMath 该矩阵尚未完成，不能伪称完整实验结束。是否继续付出正式矩阵成本应由预先确定的研究问题决定，不能作为代码修复的验收工具。若未来改变模型、推理策略、学习内容或检索机制，应先冻结新的方法定义及公平对照，再另行安排有限验证；不能混入已经结束的 LM1 结果。

**对其他 benchmark：本次没有新增全项目暂停门槛。** 此包记录原正式进程均已暂停，本次复审没有操作用户机器上的进程。LiveMath 的零覆盖结论不自动外推到 OfficeQA、SpreadsheetBench 或 SearchQA，也不改变已验证 Program 的接管资格。其他 benchmark 是否恢复，继续依据它们自己的运行身份、已修问题和预算安排，不能仅因为 LiveMath 没有正收益而一律返工。

## 实施交付与关闭条件

本次交付一个包含上述最小修改的 Git commit，并附一份零调用回归报告即可。报告写清 commit、受影响函数、56 例类型矩阵结果、配置组合结果、模拟集成路径、原 31 份响应合同不变、34 题分数与 65 次调用成本不变、Frozen hash 不变，以及新增真实 HTTP=0。报告应区分“实施方执行的测试”和“读取的历史日志”。

原九项资产与六条拒绝记录无需改写。原 LM1 报告中结构成功率、来源检查成功率、资产发布数、覆盖率与准确率各自保留，不能互相替代。将 B 未执行的效应值导出为 N/A 后，方法解释即完整。

当前材料已经足够定位并实现这两处修改，不需要再索取大规模审查包、补打真实请求或重新提供 baseline 文件。baseline 的 100% 在缺少其实际模型、输入、投影、作答与评分设置时，仍不能用来规定 ours 的预期准确率；本次也不据此猜测 baseline 的具体实现。

**关闭时应得到：工程异常类型边界已补齐，配置误用被启动前拒绝，历史结果保真，方法零覆盖如实报告，保留 high，真实新增调用为零。** 这意味着有限真实实验已经完成其诊断作用，并不意味着指导方法在 LiveMath 上已证明有效。

## 证据索引

[E1] 新包中的 `actual_execution_identity.json`、`manifest.json`、`config.json`、`source_records.json`、`budget.json`、`result.json`、`val/A/*/requests.json`、`val/C/*/requests.json`、对应 trace 与 `resources/private/evaluator_records_val.json`。独立核验脚本及结果：`analysis/lm1_1630_score_audit.py`、`.json`、`.md`。只使用公开最终回答进行作答核验，未读取 provider-private CoT。

[E2] 新包 `README.md`、`offline_regression.log`、`verification.json`、`submission_structure_audit.json`；结合实际 runner 源码核对执行条件。报告记录与复审方独立重算的范围分别注明。

[E3] 新包 `scope_precheck.json`、`train/frozen_bank/bank.sqlite3`、`train/frozen_bank/freeze.json`、`train/events/*/learning.json` 和 `requests.json`。独立回放/内容核查：`analysis/lm1_1630_scope_replay.json`、`lm1_1630_host_replay.json`、`lm1_1630_guidance_audit.md`。排除本源的 Train 覆盖诊断不使用 Val/Test gold。

[E4] 固定源码的离线复现：`analysis/lm1_1630_validator_probe.py`、`lm1_1630_validator_probe.json`、`lm1_1630_validator_candidate_probe.json`、`lm1_1630_choice_guidance_candidate.diff`、`lm1_1630_config_probe.json`、`lm1_1630_code_audit.md`。候选 diff 只用于隔离验证，未修改远端运行代码。

[S1] [choice_guidance.py，固定提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/choice_guidance.py)：`validate_proposal`、`validate_check`、`checked_asset`、`retrieval_stats`、`select`。

[S2] [system.py，固定提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/system.py) 与 [learner.py，固定提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/empirical/learner.py)：配置校验、single_answer 分支、具名工具、有界 repair、重编译及拒绝边界。

[S3] [run_livemath_lm1_validation.py，固定提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/src/atomic_skillgraph/experiments/run_livemath_lm1_validation.py)：批次准备与身份核对、冻结、无覆盖分支、A/C 运行、成本候选与结果汇总。

[S4] [test_lm1.py，固定提交](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/0099b4a44f415a8e7e418a003ef36d95028ed9b1/tests/test_lm1.py)：已有的具名工具、来源绑定、冻结、BankView、预算和恢复回归。新增测试应沿用其 transport 截获方式。

