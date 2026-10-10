# AtomicSkill LiveMath LM1：准确率问题修复与有限验证实现文档

版本：2026-10-10。状态：待实现规格，审查与零模型调用复算已完成；本文件不表示补丁已经写入仓库，也不表示新的真实测试已经执行。

本文依据本次 LiveMath seed42 聚焦审查包和 GitHub main 的实际代码编写。实跑提交与本次查询到的 main 均为 **133f88ea809f7c14616641b6dca0a66bd38f2a63**，run_id 为 **555c2e4b69e84ed4b58efe4eacebab26**。审查包 SHA256 为 **d7ac60eb9fd2b90653f24cea46e37a8241016da0654e005ea9d794fa54e56f27**。下文路径以仓库根或审查包解压根为起点，修改模块与证据路径分别标明。

## 结论与修改范围

本次 Test 的 **20/100 是现有输出在现有评分协议下的真实成绩**。独立重解析 100 条最终回答、重建候选排列与 gold 映射后，仍为 20 正确、62 合法错误、18 截断空答。没有发现把已存在的正确最终答案漏读、选项排列与 gold 错位、删除失败题或错误统计分母的问题。因此不改评分规则，不重排本次数据，不把 82 个非空回答作为正式分母。

需要修改的是学习产物的内容依据和使用方式，以及学习提交协议。七项已发布 guidance 中，两项直接否定了其来源题已经答对的公开候选；当前实现只验证来源是否答对和产物结构是否合规，没有检查新写出的指导内容是否与来源一致。检索又按普通词交集取满前三项，让这些内容跨数学主题进入大量题目。这是当前代码中可以定位的质量约束缺口，同时暴露了方法对成功经验覆盖范围的限制。

当前 LiveMath 分支绕过 Planner/Executor 的图执行，原汇总中的 Program 调用为零。因此这个 benchmark 在现有接入方式下检验的是文本 guidance 的学习与迁移，不能单凭它证明或否定可执行 Program、工具图和自动工作流的收益。本补丁不为增加方法组件而给 single_answer 人为添加多次求解或工具执行。

本补丁使用 **LM1** 作为选择题 guidance 的独立策略版本。保留原 single_answer 的一次 Runtime 调用、公开题干与选项、原最终回答格式、原 scorer、Train/Val/Test 成员与 choice_seed=42。工程主修复保持 Runtime high/32768；low 只作为一次有限验证中的成本候选。新逻辑通过显式选择题 guidance 配置启用，不修改其他 benchmark 的当前运行配置和既有 Program、Workflow 接管逻辑，不改变 Program 需要两个独立正例的原资格规则。

本轮无需重新征求实现方向，也无需再索要完整审查包。实施者应先落实本文件的工程合同，再执行文末唯一一次有硬预算的真实验证。不得直接把现有 20% 当成评分错误改掉，也不得用高分 baseline 的未知实现反推 ours 应当达到的分数。

## 已核实的证据

### 成绩、输入与运行身份

| 项目 | 本次核实结果 | 证据与核实范围 |
|---|---|---|
| Test | 100 题，20 正确、62 合法错误、18 空答 | test/test100_light.jsonl；从 raw_final_content 独立解析，并从候选身份重建 gold |
| Train | 60 题，16 正确、32 合法错误、12 空答 | train/train60_learning_summary.jsonl；同样独立复算 |
| Val | 原汇总为 3/17 | identity/phase_summaries/val_summary_header.json；包内没有完整 Val 原始回答表，本次不声称独立重算了其逐题分数 |
| 候选投影 | 177 题、885 个候选全部对应 | identity/canonical_splits、resources/public、resources/private；独立按固定种子重建候选身份、显示标签和正确候选映射 |
| 数据边界 | 60/17/100 成员一致，task_id 与 physical_key 均无跨 split 重叠 | canonical、public、evaluator 与 execution_manifest 交叉核对 |
| 调用次数 | Test 每题一份 Runtime 物理请求，均无记录到的重试 | Test 100 行中的 runtime_http_attempts |
| 输入深读 | 8 个样本 payload 指纹一致；其中 4 个 Runtime 样本题干、选项与公开资源一致 | samples/01–08；不扩大为独立读取了包外所有完整 HTTP body |
| 产物冻结 | 最终快照文件及其组合哈希复算一致，原记录中的前后哈希相同 | train/frozen_bank 与 identity/original/final_frozen_manifest.json |
| 包完整性 | 53 个清单内文件逐项 SHA256 一致 | SHA256SUMS.json；不把内部哈希一致解释成远端生成真实性的外部认证 |

最终 Frozen 文件组合哈希为 **fc42024b4e94936d92fde31ac8704dc379835b1e77307b9f5c11bee9142fd99f**。该组合哈希与 Bank 内部 knowledge digest 是不同层次的身份，不应混写。本次只拿到了最终磁盘快照，前时点的冻结状态依据原运行记录。

当前 Test gold 分布为 A22、B20、C24、D19、E15，预测为 A18、B17、C18、D13、E16、空18。旧运行“正确答案全部为 A”的问题没有在本包的投影中重现。20% 恰好等于五选一均匀猜测的期望，不能据此断言模型实际采用了随机猜测策略。

### 两项可以直接确认的错误学习

| 来源题与原正确提交 | 公开正确候选 | 发布的 guidance 写了什么 | 已有 Test 中的注入次数 |
|---|---|---|---:|
| livemath:202602:18，提交 D，hard=true | 极小嵌入存在于球面 \(S^{p+q+2}\) | skill_31c1afc651f0e08f541046d84eef2be621d8fbf9664b7864420a8abff7e23d05 宣称固定 \(S^{p+q+2}\) 属于过度主张，只有未指定 N 的存在结论可靠；实际上转向了原错误候选 C | 60 |
| livemath:202602:38，提交 E，hard=true | 公式尾项为 \(\mathrm{Tail}_{p-1,sp,p}\) | skill_4f0312202e4cc6f0757898031aa1bccad7f99d45573882bc7d3eb3c4c45330e7 要求 \(\mathrm{Tail}_{p-1,sp,sp}\)，并明确否定正确的末项 p | 38 |

两项资产的注入并集为 **77/100**，交集为 21。该数字表示错误指导的暴露范围，不能写成“导致了 77 道错误”，也不能由此计算停用技能会提高多少分。现有包没有同题、同设置的 guidance_off 反事实。

这两项错误无需重新证明数学定理即可成立：它们与自身唯一标记为正确的来源候选直接相反。其余五项资产也逐一对照了来源候选，主结论没有发现同类直接反写；这不等于其补充解释、定理机制和跨题泛化均已正确。

证据位于 train/final_guidance_assets.json、train/train60_learning_summary.jsonl 和 resources/public/train.json。资产内虽然保存 qualification=advisory_not_proof，但 system.py 的 single_answer 模型材料只传 skill_id、goal、guidance，未把该证据限定直接传给 Runtime。

### 检索与训练覆盖的问题

bank.py 的 retrieve_guidance 使用 query 与 goal+guidance 的词集合交集排名。普通英语词、数字、LaTeX 片段与主题词都参与计数，没有适用条件检查，也没有无相关结果时返回空集的门槛。本次 100 道 Test 全部注入三项，共 300 次资产注入，guidance 正文总量为 **431,271 字符**。一个 Calabi–Yau 度量问题收到 Torelli 群同态、球面极小嵌入、Morse 临界点指导，已有 HTTP 样本可以直接核实。

另一个主要失败结构是模型几乎不选择文本为 “One of the remaining options is correct, but a stronger result can be proven.” 的候选。下表仅识别这一完全相同的候选文本，没有合并其他 none-of-the-above 类别。

| 阶段及正确候选类别 | 题数 | 正确 | 合法错误 | 空答 |
|---|---:|---:|---:|---:|
| Train：上述更强结论候选为正确答案 | 22 | 0 | 18 | 4 |
| Train：其他候选为正确答案 | 38 | 16 | 14 | 8 |
| Test：上述更强结论候选为正确答案 | 55 | 2 | 41 | 12 |
| Test：其他候选为正确答案 | 45 | 18 | 21 | 6 |

这个偏差在 Train 已经存在：60 道 Train 中模型从未选择过上述候选，22 道相应来源全部被 host skip，最终七项技能也全部来自其他类别。因此，正确执行的成功经验过滤没有给这类失败提供正向学习材料。不得为弥补覆盖而从失败反馈反推出标准答案，也不得按 Test 中的 55% 占比加入默认选项偏置。

该现象是当前方法与基础解题能力的限制，不是评分 bug。不能仅凭当前结果判断它来自模型本身、通用提示、技能干扰中的哪一项；需要后面的同设置对照。新增的选项语义说明必须对所有对照臂相同，不能把公共提示变化单独算作技能收益。

### 学习提交与 token 成本

16 道正确 Train 触发 16 次首次提炼和 15 次已有结构 repair，共 **31 次 Extractor 物理请求**。结果是 7 次 upsert、9 次 rejected；44 次 host skip 是现行证据规则的正常执行。

9 次最终拒绝中，5 次因为 existing_skill_id 或 rationale 被错误地放入 guidance_skill 内部，4 次因为字符长度超过约束，其中 rationale 超长一次、guidance 超长三次。没有证据显示这 9 次拒绝由 4096 completion token 上限造成。samples/07、08 还显示首轮仅返回 JSON 正文，唯一 repair 先用来改成 ToolCall，随后才暴露内部字段或长度错误。不能把这两个样本推广为全部 15 次首轮错误的共同原因。

| 成本项 | tokens | 含义 |
|---|---:|---|
| Train Runtime | 1,247,149 | 60 次真实回答 |
| Train Extractor | 164,454 | 含现有 repair |
| Train 合计 | 1,411,603 | Extractor 约占 11.65% |
| 其中结构 repair | 85,071 | 约占全部 Train 的 6.03%，不是全部训练成本 |
| Test 合计 | 1,850,954 | 100 次真实回答 |
| 其中 18 个截断空答 | 629,309 | 约占 Test 的 34.00% |

18 个 Test 空答均为最终 content 为空、finish_reason=length、completion_tokens=reasoning_tokens=32768。这是生成预算真实耗尽，不能从不存在的公开最终回答中“修复解析”。即便只把这 18 个空答全部假设改成正确、其余 82 题保持原样，总分也仅为 38/100；这只是解释范围的上界，不是增加预算后的预计准确率。

## LM1 的工程接口

### 配置与版本隔离

新增 learning.choice_guidance 和 runtime.choice_guidance 配置对象，只在明确启用且 adapter 为 single_answer、公开输入具备合法 choices 时走 LM1。默认未启用时维持 R3 的既有路径。不要为了使新配置通过而删除 validate_config 的旧约束；增加对新对象字段、类型、版本和预算的显式验证。

Runtime 的新选择与渲染分支由 choice_guidance.enabled 控制，不能以 learn=true 或非 readonly 为启用条件。Frozen Val/Test 同样使用已冻结的新指导与适用性策略，只关闭学习；否则会出现 Train 走新逻辑、Val/Test 又回到旧检索的实现错误。

参考配置如下。它是待实现字段定义，不是当前 main 已经支持的启动配置。

~~~yaml
learning:
  guidance_repair_limit: 1
  choice_guidance:
    enabled: true
    policy_version: choice-guidance.grounded.v1
    material_version: choice-guidance.material.v1
    max_related_assets: 2
    source_check_required: true

runtime:
  choice_guidance:
    selection_version: choice-guidance.scope.v1
    prompt_version: single-answer.semantic.v2
    max_items: 2
    max_total_chars: 2400

llm:
  purpose_overrides:
    guidance_learning:
      protocol:
        thinking_type: disabled
      max_completion_tokens: 2048
    guidance_grounding:
      protocol:
        thinking_type: disabled
      max_completion_tokens: 1536
  runtime:
    reasoning_effort: high
    max_completion_tokens: 32768
~~~

当前 system.validate_config 只允许 finish_only、guidance_learning 两种 purpose override，必须显式加入 guidance_grounding。不要把它转成新的无账本服务：仍走 stage=extractor，继续计入 extractor_e1 和原 learning token 预算，额外在请求及 usage 元数据中标记 decision_purpose，以区分 proposal、grounding 和 repair。

在 empirical/__init__.py 新增选择题专用策略、材料、检索和提示版本常量。不要全局改动非选择题使用的 LEARNING_MATERIAL_VERSION 来代替专用版本。为 Learner._receive 增加显式 material_version 参数，缺省仍用旧值；LM1 调用传选择题专用值。版本进入材料快照、逻辑响应恢复键和运行身份，旧已付费响应不能因路径相同而跨版本套用。

新配置完整 hash、代码 commit、source_run_id、源训练记录 hash、Bank digest、模型请求/响应身份、公共数据与候选投影 hash 都必须进入诊断 manifest。请求缓存键同时覆盖有效 tool_choice、purpose、schema、提示、材料版本、grounding 的输入提案 hash。使用新代码的测试必须在独立 checkout 和独立输出目录中进行。

### 公开来源事实必须由 host 绑定

新增 empirical/choice_guidance.py，承载选择题公开证据构造、LM1 schema 辅助校验和选择策略。模型不能自行创建“已核实来源”。

host 仍按现有规则过滤空答、截断以及只有失败分数的经验。对 hard=true 的合法非空提交，从该题原公开 choices 中按已经提交的显示标签精确定位候选。标签缺失、重复或无法唯一定位时，记录 source_choice_unresolvable 并跳过学习，不读取 evaluator 来补答案。

绑定结构至少包含以下内容。

| 字段 | 来源与要求 |
|---|---|
| source_task_id、physical_key | 原完成的 Train 经验 |
| projection_version | 原运行的候选投影身份 |
| submitted_label | 原始最终 content 解析出的标签 |
| selected_choice_text、selected_choice_sha256 | host 从该标签对应的公开候选直接复制并计算 |
| source_goal、source_public_inputs_hash | 原公开题干与公开输入 |
| outcome_evidence | 原 hard=true 与最终回答 hash；不复制私有 evaluator 的完整记录 |
| qualification | verified_submission_not_general_proof |
| source_run_id、source_record_hash | 原经验来源身份 |

此对象应当不可变，不能被提炼器输出覆盖。模型可见材料直接给出 source_goal、选中的候选正文以及必要的公开上下文，并明确：这是在该题限定条件下被评分验证的提交，不是任意新定理或泛化的证明。

唯一来源是正确字母时，提炼器只能提出与该来源相容的条件化观察和操作建议。不得根据失败题的私有 gold 修复或补造成功经验。仍然允许有真实已检查公开过程的原学习分支按其自身合同运行；本补丁不放宽它。

### 简化提炼 schema，并把单一工具提交落到 HTTP

在 prompts.py 增加 LM1 专用的平坦 schema，不复用 guidance_skill 的嵌套布局。建议字段如下，最终持久化时由 host 转换成 Bank 资产，而不是要求模型输出 Bank 内部结构。

| 提案字段 | 约束 |
|---|---|
| decision | no_change、reuse_existing、upsert_guidance |
| existing_skill_id | 可选，只允许本次实际提供的同域有效候选 ID |
| goal | upsert 时必需，最多 160 字符 |
| guidance | upsert 时必需，最多 1000 字符 |
| scope_terms | upsert 时必需，2–4 个来自源题干的可验证主题词；采用后述规范化 |
| applicability | upsert 时必需，最多 240 字符，说明需要匹配的数学对象和前提 |
| rationale | 可选，最多 240 字符，不进入 Runtime |

no_change 不携带资产正文；reuse_existing 不携带新正文；upsert 缺任何必需字段均拒绝。上下文中仅给最多两项通过适用性过滤的相关资产，并用短卡片表示，不再把八项长正文全部放入每次提炼。模型输出如果只是改写选项字母、把来源答案做成查表规则，或没有可复用操作，应使用 no_change。

具名工具提交由 EmpiricalSystem.agent 根据 purpose、有效 thinking 状态和 LM1 配置确定。仅对本补丁的 guidance_learning 与 guidance_grounding、且 thinking=disabled 时，分别设置具名 submit_learning 或 submit_guidance_check。provider.complete 与 _build_payload 增加默认 None 的可选 tool_choice 参数；检查工具名存在于当次 tools，其他调用保持原 payload 行为。

DeepSeek 官方接口说明，有 tools 时默认 auto，模型仍可返回正文；非 thinking 模式支持具名工具，thinking 模式使用 required 或具名工具会报错。因此不能全局添加 required，不能给 high Runtime 或 Builder 顺带强制工具，也不需要引入 beta strict endpoint。[官方 API 说明](https://api-docs.deepseek.com/api/create-chat-completion/)

host 仍严格验证工具数量、工具名和 schema。普通 JSON 正文不直接视为成功 ToolCall。若模型仍返回可严格解析的 JSON 正文，可在仅供 repair 的诊断中同时收集封装问题与全部字段问题；不能因此自动接受或静默搬移字段。repair 只修结构、长度和提交方式，不修数学结论；每个对象最多一次，有限验证整批最多四次。
这里的 repair 对象仅为 proposal：每个 proposal 至多一次，checker 为零次 repair；有限验证整批 proposal repair 至多四次。具名工具仍返回普通正文或不合规工具调用时，记录协议失败，不自动改成 JSON 模式或改 endpoint 再试。

不得用字符串截断让超长 guidance 通过，避免截断数学条件、否定词或公式。named tool 只能提高提交符合度，不能替代下节的来源一致性检查。

### 结构合法后，增加一次有界的来源一致性检查

在 Learner 新增 _learn_grounded_choice_guidance。其发布顺序固定为：host 绑定来源，提炼 proposal，结构校验，通过后做一次独立 grounding 检查，最后才允许 Bank.put。任何环节失败都保留可审查记录，不能先发布再撤回。

checker 接收当前提案、host 绑定的公开来源事实，以及必要的源题公开候选和前提。它不访问 private evaluator，不读取 Test/Val，不索取 provider-private reasoning，不重新求解源数学题。它的职责是判断指导内容是否与给定来源相容，以及是否扩大了来源本来没有支持的适用范围。

checker 输出三态：supported、contradicted、insufficient_evidence，同时提供提案中的原文片段、来源中的原文片段及简短理由。host 检查引用片段确实存在于对应输入、响应结构合法、版本和提案 hash 匹配。只有 supported 才能发布；冲突、依据不足、截断、格式错误、未知响应均不发布。该语义步骤没有“改到通过”为止的循环，也不使用另一个答案评分来选最好的提案。

每项新正文最多一次 checker 调用，thinking=disabled，completion cap=1536。只发生 reuse_existing 且正文和来源检查身份均有效时，不重复检查原正文；本次新来源的关联作为独立审计记录保存，不能把新增来源数当成额外数学证明。修改既有正文会产生新版本和新的检查，父版本不可原地覆盖。

在成功提交候选与新 guidance 之间，单靠 source_choice_id 相等或复制一段原文，无法排除其余自由文本里的反向结论。因此本次新增 checker 有两项明确同源反写作为工程依据。但 checker 本身也不是数学证明器：它可能误判，supported 只代表有限来源一致性检查通过，不升级为全局可用定理，更不获得 Program 接管资格。

持久化资产新增 evidence_source、grounding_check、applicability、scope_terms、guidance_policy_version。保留 execution_intent=guidance_only。拒绝资产不进入可检索集合；审计中保留 rejected proposal、拒绝类型、引用依据和消耗，便于确认是格式失败、来源冲突、证据不足还是无可复用内容。

已知两项错误正文应作为固定的来源冲突回归例。可以在派生数据处理时按原内容 hash 排除，但这只用于阻止已知错误复用，不得把 hash 黑名单冒充对新正文的通用语义验证。

### 用适用性过滤代替强制填满前三项

新增 Bank.select_guidance(task, policy)，返回 selected 和完整选择审计。LM1 的 Runtime 与 Learner 都使用该入口，避免一端过滤、另一端仍注入所有旧资产。旧 retrieve_guidance 保持旧策略调用兼容，不在本补丁中改动 Program 和 Workflow 的通用检索。

首先排除旧格式未通过新检查的资产、拒绝资产和已被合规子版本替代的父资产。然后对源题与当前题干采用固定的 Unicode 规范化和词规范化，去掉一般停用词、单字符变量、纯数字、常用 LaTeX 命令及 strongest/statement/question/options 等题型套话。词频统计只使用完成的 Train 公共题干，冻结后不从 Val/Test 更新。

scope_terms 必须是源题干中实际出现、规范化后可定位的 2–4 个主题词，不能由模型编造。当前题至少匹配两个 scope_terms，且按 Train IDF 加权的覆盖率至少为 0.5 才进入候选。按加权覆盖率排序，平分按稳定资产 ID 排序；最多返回两项，整体注入文字不超过 2400 字符。阈值是本次预先固定的工程初值，不在看过 Val 分数后反复调优。

IDF 固定为 idf(t)=1+ln((1+N)/(1+df(t)))。对规范化去重后的资产主题词集合 S 与当前题干词集合 Q，覆盖率为 sum(idf(t), t in S∩Q) / sum(idf(t), t in S)。N 和 df 只统计已完成的 Train 公共题干，每个 physical_key 只计一次；在线训练阶段在完成训练单元时幂等更新，恢复不能重复累计。空 Bank 的初始统计为 N=0，没有资产时直接空返回。

现有 Bank.freeze 会删除 train_cases，因此必须在冻结前把统计快照存入 Bank 专用 metadata 键 choice_guidance_retrieval_stats，包含 normalizer_version、N、df、源公共题干集合 hash 和统计版本，并确认该键进入 Bank.digest。它随数据库 backup 保留。Frozen 只读该快照，不能临时从已经删除的 train_cases、Val 或 Test 重建统计；非空 LM1 Bank 缺该快照时，应在付费调用前报身份不完整。新 runner 在适用性预检中验证这一点。

这些条件是可实现的最低适用性过滤，不证明全部数学前提已满足。模型视图必须同时传入 applicability、qualification、source_check_status 与明确的限制说明。对于仅靠一个正确来源得到的经验，不能在 Runtime 中表现成已证明的通用定理。最终正文不应包含原题“选 C”“A/B/D 都错”等与标签绑定的指导。

没有符合条件的资产时，selected 必须为空，Runtime 直接按公开输入回答。不能退回旧词交集检索补足两个名额，也不能用题目共有的“最强结论”触发所有数学领域的技能。

每题记录候选 ID、匹配主题词、覆盖率、入选或拒绝原因、实际注入 ID 与字符数、选择策略版本。该审计用于解释技能是否真正参与解题，不能只记录“Bank 非空”。

BankView 的 guidance_off 需要显式覆盖新 select_guidance，返回结构正确的空选择与视图标记。不能只在旧 hidden 集合中加名字后继续返回普通空列表，否则调用方拿到的结构会不一致。guidance_off 仍基于同一个只读 Frozen Bank，不删除资产。

### 共同的选择题说明与 Runtime 预算

把 single_answer 的通用 system prompt 提取成有版本的函数，LM1 下明确要求按每个选项的实际含义和题干量词判断，区分“某命题为真”与“它是可证明的最强结论”，同等审视评价其他选项充分性或强弱的候选。指导只能在适用条件匹配时参考，不覆盖题干或成为事实权威。

这段文字对 guidance_off、高推理 guidance_on、低推理 guidance_on 三臂完全相同。它属于公开任务说明修正，不是学到的技能。不得写入某个字母、Test 频率、具体 Test 答案或“优先选更强结论候选”的启发式。原题、选项正文及顺序不改，最终仍按 adapter.answer_contract 返回一次答案。

主修复维持 high/32768，不附加补答调用，不使用 reasoning_content 中的中间结论替代最终 content，不把外层 600000 token 上限误当成可见答案预留。18 个截断失败继续按原规则计入成绩。

成本候选只预先测试 low/32768。DeepSeek 当前官方说明 medium 会映射为 high，因此不能把 medium 写成有效的降推理方案。是否使用 low 由文末唯一一次配对验证决定，不能只看空答减少而忽略答对题数下降。[官方推理参数说明](https://api-docs.deepseek.com/guides/thinking_mode/)

### 具体修改落点与控制流

| 仓库模块 | 必须实现的改动 | 接入边界 |
|---|---|---|
| src/atomic_skillgraph/empirical/choice_guidance.py，新建 | 来源绑定、平坦提案辅助校验、scope 规范化、来源引用校验、结构化选择结果 | 纯 host 逻辑不得读 private evaluator 或调用模型 |
| src/atomic_skillgraph/empirical/learner.py | 新选择题学习分支；proposal→结构检查→grounding→发布；_receive 支持专用材料版本 | 原 _learn_guidance 与非选择题学习默认路径保留 |
| src/atomic_skillgraph/empirical/prompts.py | 平坦提案与三态检查 schema、短提示、通用选择题说明 | 不添加 Test 派生示例和答案偏置 |
| src/atomic_skillgraph/empirical/bank.py | select_guidance 与结构化过滤审计 | Bank.put 的不可变资产身份和 Program 资格规则保留 |
| src/atomic_skillgraph/empirical/bank_view.py | guidance_off 对新入口返回空选择 | 同一 Frozen Bank 的视图干预 |
| src/atomic_skillgraph/empirical/system.py | 新配置验证、grounding purpose、具名工具作用域、single_answer 新材料渲染、完整身份入恢复键 | 单题一次 Runtime 不变；所有学习消耗继续入账 |
| src/atomic_skillgraph/agents/provider.py | 可选具名 tool_choice 参数、发送前合法性检查、payload 审计 | 未传参数的旧调用行为不变 |
| src/atomic_skillgraph/empirical/__init__.py | 新选择题专用版本常量 | 不借此放宽原 POLICY_DEFAULTS |
| src/atomic_skillgraph/experiments/run_livemath_lm1_validation.py，新建 | 读取旧 Train、派生重编译、冻结、固定 Val 三臂、硬预算和一次性结论 | 不复用已有只支持固定两对任务的 run_cf4_r3_diagnostic.py 作为未声明的新实验 |
| 对应 tests | 原样本回放、两项同源矛盾、scope 空返回、grounding 拒绝/缓存、三臂一致性与预算 | 见下节验收；不新增大规模真实训练门槛 |

概念控制流如下，函数名为本补丁待实现接口。

~~~python
def learn_choice_guidance(task, completed_experience):
    source = build_verified_public_source(task, completed_experience)
    if not source.eligible:
        return record_host_skip(source.reason)

    related = bank.select_guidance(task, learning_policy).selected
    proposal = receive_versioned_proposal(source, related, max_repairs=1)

    if proposal.decision == "no_change":
        return record_no_change()
    if proposal.decision == "reuse_existing":
        return record_valid_reuse(proposal.existing_skill_id, source)

    checked_proposal = validate_proposal_and_scope(proposal, source)
    check = check_source_consistency_once(source, checked_proposal)

    if not admissible_supported_check(check, source, checked_proposal):
        return record_rejected_proposal(checked_proposal, check)

    return bank.put("skill", make_immutable_guidance_asset(
        checked_proposal, source, check
    ))
~~~

## 已有运行与产物怎么处理

保留原 20/100 结果、原完整运行目录和原 Frozen 文件。历史事实标记为 R3 实跑结果，不能把旧指导原地修改后仍沿用旧 hash、旧 run_id 或旧成绩文件。

此次有限验证使用 **offline_recompile_from_completed_train** 模式。按原 Train 执行顺序读取 60 条已完成经验，保存完整源引用；44 条不符合当前正向依据的经验继续 host skip；只对 16 条原成功经验运行新的提炼和必要检查。整个阶段不调用 Runtime 重做源题，也不调用 Program。

派生 Bank 从独立空目录开始。允许它读取本次重编译过程中已经合规发布的相关资产，但不能偷偷载入旧七项资产跳过新检查。对恢复已付费的新 proposal/checker 结果使用新版本的精确身份，避免断点续跑重复计费。

这个 Bank 使用的是旧在线过程形成的成功经验，不能称为“LM1 从零在线 Train 的反事实结果”。日志应同时报告 parent_train_runtime_tokens=1247149、parent_total_train_tokens=1411603、incremental_recompile_tokens、derivation_mode。论文中核算完整方法成本时必须说明继承的训练成本，不能只写重编译便宜的那一部分。

重编译入口应是显式的 learn_from_completed_record，而不是 EmpiricalSystem.run_task。它按公开资源重建 PublicTask，并从原记录映射 submission、score、answer_status、empty_answer、completion_truncated、finish_reason；这些字段属于继承经验，不写成新生成的 Runtime trace。为每个学习事件单独设置 phase=train、budget_scope、TaskCheckpoint、audit_path 和源记录 hash，沿用原学习账本起止处理。原 Runtime usage 只写继承成本，不再作为新请求计入本轮 BudgetGovernor。即使没有新资产，60 条源记录也保留原顺序和明确的处理状态。
具体还要在进入每个源记录时初始化 _task_start、_request_start、_learning_start、prior_usage 和 task_context，使预算、模型材料及恢复键都归属于本次重编译事件；prior_usage 只恢复该新事件此前已计费的请求，不从旧 Runtime 或上一源题继承。结束时把本事件新 usage 与继承经验分别写出。

此次派生 Bank 仅服务于 seed42 的固定诊断，不能复制给 seed43/44 冒充独立训练。后续如启动修复后的正式实验，再按统一定义独立训练并分别记录源身份。旧代码目录内仍运行的其他 benchmark 可以继续；本次证据不构成它们必须重启的理由。

由于原 Test 成绩已经用于问题审查，修复后若再次评估同一 100 题，应标注“修复后重评”及版本，保留原结果。方法选择只用已声明的 Train/Val 规则，不用旧 Test 的逐题对错筛技能、选 prompt 或调阈值。不同 seed 在同一 Test 上重复，不会使已经观察过的题目变成新的独立未见测试集。

## 验证方案：一次固定批次，分清工程合同与方法效果

### 零模型调用验证先完成

本次审查已经完成包内哈希、100 Test 与 60 Train 原始答案复算、177 题投影、split 边界、8 个请求样本、七项资产来源对照。实施后不用重新跑整套正式实验来验证这些问题。

新增自动化验证应覆盖下列有实际风险的边界。它们可以用旧公开样本、纯 host 构造数据和 capture transport 完成，不发真实 HTTP。

| 验证内容 | 规模与预期 |
|---|---|
| 来源绑定与评分隔离 | 回放原 60 条 Train；16 可进入提炼、44 host skip；不得从私有 gold 补来源；100 条 Test 仍按旧输出得到 20/62/18 |
| 两项错误资产 | 两项原正文及对应公开正确候选固定为冲突样例；在人工依据和假 checker=contradicted 的控制流回归中均不能进入 Bank |
| 正反与不足三态 | 合成 supported、contradicted、insufficient_evidence、引用不存在、空答、截断等返回；只允许完整 supported 发布 |
| 检索退路 | 同域正常入选、无主题匹配返回空、仅套话相同返回空、旧未检查资产排除、父子版本、字符预算；不得补满名额 |
| Provider 提交 | capture transport 验证只有 disabled 的 LM1 学习/检查带具名工具；high Runtime 无 tool_choice；普通 JSON 不被当成功 |
| 缓存和重入 | 改 source、proposal、schema、tool_choice 或版本会改变恢复身份；同身份恢复不新增调用或重复资产 |
| Frozen 与对照 | 新 BankView 关闭指导；三臂公共输入和基本提示相同；Val/Test 不调用学习；前后知识 digest 相同 |
| 硬预算 | 请求数、结构 repair 总数、tokens 预留、未知计费及断点恢复触发预期停止 |

两项冲突样例的假 checker 测试只能验证 host 不会发布被拒绝内容，不能宣称真实 checker 已经能识别所有数学错误。新 checker 的真实有效性仍要在有界重编译时检查其输出。独立 source_choice 绑定及原文引用同样不等于自由文本的语义证明。

无需为本补丁强制反复执行所有历史大套件。应运行受影响的 guidance、provider、single_answer、BankView、恢复与预算测试，以及既有相关 CF4 回归；若共享 Provider 修改使其他调用 payload 发生变化，再针对该具体差异扩展检查。已修复的 M4 Program 贡献快照问题不在本次重新定义验收标准。

### 真实调用的唯一固定方案

使用现有 **全部 17 道 Val**，不再抽新的 12+6，不调用 Test 调参，不做新的 60 道源题求解。选择三臂是为了在同一批次内同时回答“新指导有没有帮助”和“low 能否降低成本”。

| 阶段/对照臂 | 数据与行为 | 最大新调用数 |
|---|---|---:|
| 重编译 proposal | 原 16 条成功 Train，每条最多一次首次提炼，2048 completion cap | 16 |
| 重编译 grounding | 每项新正文最多一次检查，1536 cap；no_change 或已拒绝不调用 | 16 |
| 整批结构 repair | 所有新提案合计最多 4 次，每个提案仍至多一次，2048 cap；checker 无 repair | 4 |
| A：guidance_off / high | 固定 Val17，32768 cap，一题一答 | 17 |
| B：修复 guidance / high | 同 Val17、同公共提示、同模型服务，仅改变指导注入 | 17 |
| C：修复 guidance / low | 同 Val17、同 Frozen、同提示，仅改变推理强度 | 17 |
| 合计硬上限 | Runtime 最多 51 次，其余最多 36 次 | **87** |

全批次 **max_retries=0、物理 HTTP 上限 87、总 tokens 硬上限 2,500,000、finish_reserve=0**。这一数字是允许上限，不是承诺一定消耗这么多；任何阶段提前结束都不补足名额。按所有列明 completion cap 计算，最大 completion 总和为 1,736,704，剩余预算用于输入；是否能准入每个请求仍由 BudgetGovernor 的输入字节上界加 completion cap 预留判断。

应复用已有 BudgetGovernor 的物理请求入账和未知计费停止机制，不能在 runner 末尾才检查有没有超支。proposal/grounding/repair/runtime 共用同一个批次账本。结构 repair 剩余次数也要持久化，重启不重置；达到四次后，后续结构错误按拒绝处理。任何未知计费、未决请求或硬预算耗尽，都停止该批次，不自动加预算或另起一次“补跑”。

新 runner 可以复用 TaskCheckpoint、EmpiricalSystem、BankView 与 BudgetGovernor 的成熟部件，不能把旧固定两对任务脚本简单换个题单就宣称完成了三臂协议。预注册 manifest 中保存固定 Val ID、三臂配置、Bank hash、运行顺序和判断规则；不保留根据分数替换题目的入口。

新 runner 的命令接口统一为 prepare、check-offline、run 三个子命令。prepare 接收 --review-root、--base-config、--output，生成源记录绑定与固定批次 manifest，不读取密钥、不发 HTTP；check-offline 接收 --manifest，验证输入、配置、恢复身份与硬预算配置并产生离线报告；run 接收同一个 --manifest 和 --env-file，依次执行重编译、冻结、适用性预检和固定 Val。run 不接受新增题数、增加 seed、替换候选臂或临时上调预算参数。以上是待实施 CLI 合同，不能把当前仓库中不存在的命令当成已经可执行的启动命令。

run 的输出目录首次创建后绑定 manifest hash。再次运行同一目录只能按已保存的 checkpoint 恢复：完成的响应直接复用，已计费未决请求停止等待人工处理，不能自动再次发送。结构修复是预先计数的新调用，完整响应的重复评分和整理不新增模型调用。prepare 不能覆盖已存在的付费结果目录。

三臂按题交错，并固定轮转顺序 A-B-C、B-C-A、C-A-B。只控制本地顺序，不宣称供应商支持生成随机种子。每个任务在每个臂只有一个 Runtime 回答，超时、截断和错误都保留，不补答、不择优。

如果重编译没有产生任何合规资产，或对全部 Val 的公开题干离线选择结果均为空，就停止方法验证部分，报告“没有形成可在此 Val 使用的指导”。不要付费运行两个完全相同的 high 条件后声称已检验方法，也不要放松检索门槛凑出注入。若仍要执行本批次预声明的成本比较，可只执行 A 与明确标注 guidance_off 的 low 条件，两者仍各 17 次且受原总预算约束；不得增加第四种条件。

### 希望看到什么结果，以及怎样结束

工程合同的验收与分数提升分开。已知来源反写不得在新 Bank 中继续发布；拒绝、无可复用内容、证据不足应成为清楚的正常结果，而不是异常退出。结构提交成功率要单独报告，不能把“更愿意接受提案”当成学习质量提升。

本次希望至少能形成若干来源检查通过、条件范围明确的指导，并在部分 Val 题上真实使用；不预设必须得到七项、必须每题注入，或每个成功来源都必须产出资产。若一致性检查持续拒绝所有数学泛化，这说明仅凭正确字母不足以支撑当前提炼目标，不能靠降低检查标准宣称修复成功。

A 与 B 报告 17 个配对结果中“只有 B 正确”的题数、“只有 A 正确”的题数、总正确数、空答数、Runtime tokens，以及实际指导暴露。另报告元选项与普通选项两个描述性分层。净增益为前两种题数之差；不设“必须提高固定几个点”作为工程通过条件。

如果 B 有正净增益，且增益任务确有符合范围的指导参与，可以写“本次固定小样本显示正向方法信号”。17 题不能证明整体显著优于无指导。若 B 与 A 持平、变差或没有实际指导暴露，应分别报告“未观察到净收益”“出现退化”或“没有形成有效方法干预”，不追加同一 Val 的第二轮调参实验。工程修复可以完成，方法收益可以未成立，两者必须如实并列。

本方案没有旧 guidance 的同期第四臂，因此不用于估计旧错误指导把 20% 压低了多少。保留这个因果范围，能够避免为了补齐所有归因再不断增加实验。

C 仅在 **同 17 题 Runtime 总 tokens 比 B 降低至少 25%、正确题数不低于 B、空答数不多于 B** 时被选为后续候选配置。否则保留 high，结束成本选型。该规则是预先规定的工程选择条件，不是统计等效性证明。学习成本不能只混入某一个 Runtime 臂中作比较。

若发生前述“没有可用指导”的预声明分支，成本比较以 A 为 high 参照、以 guidance_off/low 为候选，仍使用相同的 25% 成本条件和正确数、空答数条件；结论只涉及推理成本，不涉及方法收益。

固定批次后生成一次结论，区分“补丁可运行”“指导有无方法信号”“是否采用 low”。不能将无方法收益自动改写为又一个未修完的工程 bug，也不能因为没有达到 baseline 的未知高分而继续试到满意为止。

## 后续正式实验的决策边界

当前 R3 已能完整跑完 LiveMath，但它确实会发布与来源相反的 guidance，继续原样增加 seed 主要会扩大已知缺陷的成本。建议先完成 LM1 工程修复及上述固定批次，再决定 LiveMath 的后续正式矩阵。其他正在运行的 benchmark 不受这个选择题局部结论牵连。

若工程合同通过且有限批次支持继续研究，可按冻结配置开启后续独立 seed；方法信号不足本身不是“工程不能启动”的新门槛。如果用户选择保留该 benchmark 作为方法局限或负结果，也可以继续收集正式结果，但需要如实报告已有负证据，不再用反复 pilot 替代研究结论。

新训练如采用 LM1 在线路径，应真实从其独立 Bank 开始，不能把本次 offline_recompile 派生实验包装成新的完整在线训练。修改后的方法、模型服务与公共提示身份要统一，旧 Test 与修复后重评分别展示，不混成同版本三 seed 平均。

## 模型身份与可比较性

包内实际 endpoint 为 DeepSeek 官方接口，请求名 deepseek-v4-flash，响应字段为 deepseek-flash。2026-10-10 查阅的 DeepSeek 官方说明明确：旧 deepseek-v4-flash 与 deepseek-v4-flash-vision-exp 名称仍可接受，但对应服务已退役，其请求由 DeepSeek-V4.1-Flash 提供。[官方模型名称说明](https://api-docs.deepseek.com/quick_start/pricing-details-cny/)

因此，报告应分开记录 request_model_alias、response_model、endpoint、provider_serving_statement 与核对日期。现有日志没有远端权重快照，不能把旧请求别名当成原始 V4-Flash 权重的保证。本文件不追溯改写原请求记录；按当日官方映射补充服务身份说明。若实验要求严格固定原始 V4 权重，现有官方旧别名不能满足该身份要求，不能靠本地改显示名称解决。

这项标注问题不等于已解释 20% 的原因，也不需要仅因别名变化丢弃现有真实输出。三臂有限验证须使用相同服务和同一时间窗口，并保留实际返回标识。baseline 当前没有可用实现与配置材料，本文件不对其 100% 作有效性认定，不以它设定本补丁的成功阈值。

## 交付与证据索引

实施交付应包含一个新代码 commit、一份冻结的实际配置、一份零调用回归结果，以及至多一份固定真实批次结果。真实批次报告给出逐题输出、原始 usage、来源与 grounding 审计、选择审计、各臂 Frozen 前后身份、全批次预算账本和最终决策即可；不要求新增跨 benchmark 大规模日志包。

本次分析使用的主要源文件是 identity/actual_run_identity.json、identity/original/run_manifest.json、test/test100_light.jsonl、train/train60_learning_summary.jsonl、train/final_guidance_assets.json、resources/public/train.json、resources/public/test.json、resources/private/gold_identity_map_177.jsonl、resources/public/candidate_identity_map_177.jsonl 和 samples/01–08。文件哈希以包内 SHA256SUMS.json 为准。

代码依据采用固定提交链接，避免 main 后续移动导致审查对象混淆：[System](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/system.py)、[Learner](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/learner.py)、[Bank](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/bank.py)、[Prompts](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/prompts.py)、[Provider](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/agents/provider.py)、[BankView](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/bank_view.py)、[BudgetGovernor](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/empirical/budget_governor.py)、[LiveMath 候选合同](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/skillcompiler_bench_contracts/livemath.py)、[LiveMath scorer](https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/blob/133f88ea809f7c14616641b6dca0a66bd38f2a63/src/atomic_skillgraph/harness/scorers/livemath.py)。

本次仅执行代码阅读、原始记录分析、哈希与评分复算；新增真实模型调用为 0。后续真实效果以实施后的固定批次为准。

