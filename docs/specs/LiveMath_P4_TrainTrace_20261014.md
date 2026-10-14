# LiveMath：P4 Trace 驱动数学能力学习与 Val17 验证修改实施文档

**性质**：待实现的新方法/材料版本；基于 `main=3c027c7d98b977def5ea5450f1453748cafec7a8` 实跑 P3（新 `main` 的 Office 部分修改不构成该 P3 负结果的重测）、完整 P3 审查包和理想四层 Bank。**复用 Train60 原始 Runtime trace，不重新求解 Train；旧 P3 learning、Val off/on、Frozen 与费用全部不可修改。**

## 0. 实际结果与本轮目标

- Train60 已全部进入 P3 的 15×4 来源学习；P3 **15 单元仅 2 组留存来源+内容合格候选**，后经整合有 6 个候选；Val17 前6题测试4个、后11题测试1个，Frozen **0 usable Skill、0 Program**。本轮 **1,242,239 tokens / 106 HTTP**，不是因总预算、单次 completion 或网络异常停止。
- 已确认阻断：**10 组普通提案夹带仅整合阶段才有意义的 `parent_ids`**；`validate_batch` 是一项出错整批拒绝；内容审查亦可能一项错拒整组；真实模型最终 `<answer>E</answer>` 因可选 `public_process=546 >512` 整条 ToolCall 被拒并计空答。
- 最好的量词候选 `662f` 在 10 个实际注入题上 **off 2→on 3，净增 +1**，未达冻结 +2；候选 `8e83` 1→0；另外两个适用0～1题证据不足。**工程误拒要修，数学收益仍需重新实测**，不能把历史 +1 改称正式达标。

## 1. 修改代码：P0 协议纠错、逐项保留

模块：`src/atomic_skillgraph/empirical/math_capability.py::{batch_schema,validate_candidate,validate_batch,learn}`、`model_view.py`、`system.py`、`answer_executor.py`、相关 `run_math_p3.py`/checkpoint。

1. **阶段独立 Schema**：普通四题学习时 `parent_ids` 不是有效关系字段；优先在生成请求/提示中只提供普通单元字段。若响应额外带该字段，新版 Host 只允许在**普通单元、精确字段名 `parent_ids`** 下按声明规则剔除其无效元数据，不把其内容用于来源/能力审核，并保留原 ToolCall、归一化前后内容哈希和错误收据；整合阶段则必须提供真实父候选 `parent_ids` 且逐个验证。绝不泛化为“所有未知字段忽略”。
2. **逐候选校验**：一个 `candidates[]` 成员缺字段、引用少于两个物理 Train 来源或纯格式超限，只拒绝该成员，**保留同批其他合法成员**；逐候选记录 `accepted / rejected / reason / source_case_ids`。限制 batch 候选数及每项两独立来源不变；保留一次既定结构修复，不为通过率自动追加 paid retries。
3. **逐候选内容检查**：按 `candidate_hash` 对齐真实独立审查项；一个错误 span 或 `domain_fact/unsupported` 只阻止该候选，其他真实合法 `procedural_only/disputed` 不受牵连。对 `disputed` 允许进入**受控开发性效用验证**，不直接晋升 usable；真实来源、数学事实禁止伪造仍保持。
4. **文字容量合同仅适度放宽 Host 容差**：建议 `guidance <=640`、`prerequisites` 每项 `<=128`、`stop_condition <=192`、完整注入卡 `<=1536` 字符；**生成 Prompt 的目标长度仍按旧短目标**。同步请求 Schema/Host validator/渲染/Runtime 注入额度，不能存在“Schema 拒绝但 Host 可接受”的失配；来源引用 ID 不做模糊匹配。
5. **答案和可选过程注释分离验收**：`answer_step(action=finish)` 的工具身份、主结构和合法 `answer` 通过后，单独核验 `public_process/public_steps`。仅可选注释超长或引用无效时，保留原答案交由同一评分器，过程标 `process_verification=rejected` 且不能供学习；主答案不合法仍正常拒绝。用历史 `livemath:202512:23` 的 546 字符注释做零 HTTP 真实协议回放，旧 off 臂分数不修改。

## 2. 改善真实学习与能力覆盖（Train-only）

模块：`experiments/capability_schedule.py`（若 P4 接独立分组）与 `math_capability.py::{public_structure,math_spans,learn}`、`experiments/run_math_p3.py` 派生新入口。

- 从**原始 Train60** 加载公开题干/选项、已执行计算与公开轨迹结果、训练评分及当时已保存的过程材料；不读取旧 Val17 gold/预测/配对得分，不将 P3 已发布（实际0）或人工理想 Bank 注入初始 Bank。原 trace 的旧 `score` 只作为训练反馈，不是数学证明或错误原因归因。
- 预先按真实数学操作需求而非纯 Jaccard 词汇组织 15 个学习单元，保证 60/60 覆盖、每组<=4、每候选真实证据>=2 个独立物理 Train 案例；公开任务分析只给量词/极限/唯一性/反例/有限可算的精确文本 Span，不决定选项真假。无法共享能力允许 `no_change`；同一能力跨组再整合一次，保留父子版本、引用和真实原始来源。
- 采用 LiveMath 专属五大类别（量词蕴含、收敛渐近、反例构造、存在唯一、有限计算），每个过程 Atomic 明确 `applicability / prerequisites / actionable guidance / stop_condition / source_refs`，避免只产出“留意假设”的泛化提醒。候选去重按“具体步骤＋适用前提＋不同能力覆盖”，不使用 Val 成绩或题号排序。
- **Program/Graph 分开实施，不冒充已自然学到**：若 Train60 轨迹确实已有成功本地 Python 计算及可绑定的真实输出，可沿原 `Learner→Builder→Worker→validate_on_source` 学习有限确定性 Program（有限布尔/量词/有理数、小矩阵等）；没有真实可重放来源则只能保留 guidance-only 或开展**另行版本化的 Train-only 合法验证实验**，不得从 gold 手工合成程序收据。LiveMath 当前 `single_answer` 不会由 Planner 接管 `empirical.workflow.v2`；完整 Composite 要单独开发并经动态执行/数据流测试后才可宣称支持，不作为本次 P4 的必过工程门槛。

## 3. 新版开发性效用验证与冻结

**Train60 trace → P4 多组学习/跨组整合 → 通过来源检查的候选 → 新版 Val17 真实 off/on → P4 Frozen → Test100（单次人工放行）。**

- Val17 明确是**已经使用过的开发集**。旧 P3 的路由/成对结果**不得作为新 P4 效用收据**；在新方法版本、冻结候选与路由后，对实际适用题分别运行新的 off/on，两臂模型/预算/工具/其他指导完全一致，只有目标候选可见性不同。Val 不参与生成或修订候选正文。不得用 Val 前几题结果临时重新划分训练来源或修改 selection 门槛。
- 选择控制：最多4个能力不同候选进入前6道初筛，最多2个进入后11道确认；不可用仅因 `coverage_insufficient` 就记成负效用。**正式 usable 晋升保持 P3 事先约定的核心门槛**：实际注入>=8、总净增正确>=2、correct→wrong<=1、空答不增加，且初筛/确认阶段条件分别满足。既有 `662f` 的 +1 是已观察开发结果，不在本版本回填为通过。
- 可以在 Bank 中单独记录 `source_checked`、`utility_inconclusive` 候选作为**非发布的审计/研究材料**；只有带真实配对收据且通过门槛的候选才能进入最终 Frozen。`math_capability.checked_asset`、`Bank.freeze` 与方法 ID 同步升级。所有 `no_change/rejected` 正常结束；不能以“必须至少一个 usable”作为工程通过条件。
- Test100 仅在固定方法、候选选择、Bank hash 和 Val 开发完成后执行；此前部分 Test 历史诊断已被查看，论文需披露，不能宣称从未接触过 Test。若要证明 Skill 因果收益，准备同参数 NoSkill/Skill-off Test 对照，**额外付费需计入明确预算**。

## 4. 预算（本次不增加授权）

P3 审查包累计占用 **11,817,657 / 已批准 45,000,000 tokens**（含一笔旧 39,819 未决保守预留），已知 709 HTTP，累计 HTTP 硬上限9,165。新 P4 的追加模型调用仍全部计入同一个 45M Governor，不因方法改版刷新历史额度。

| 阶段 | 新版上限建议 |
|---|---:|
| P4 从 Train60 重编译与跨组优化 | 5,000,000 tokens / 150 HTTP |
| 新 Val17 真实候选路由与 off/on | 15,000,000 tokens / 1,200 HTTP |
| Test100 正式单臂评估 | 13,000,000 tokens / 5,000 HTTP |
| 历史占用 + 全部未来阶段理论上限 | **44,817,657 / 45,000,000 tokens** |
| 理论未分配余量 | **182,343 tokens** |

**注意**：这是同一个累计上限下的阶段最大值总和，理论上只余182,343，不代表实际预计花到这么多；如随后产生新的诊断费用，必须重算剩余额度。不得互借阶段上限、结转未知计费为零或以版本重启“刷新”5M/15M/13M；如上界不再覆盖计划，应在付费前重新制定并请求授权，不能运行中擅自提高。

64K主解题 thinking enabled/high、600s、父预算128K；只有真实 `length` 空答才允许最多一次 thinking enabled/low、16K completion 补交，完整题目与选项必留，可选真实返回的推理尾段不超过16KiB，按实际剩余额度裁剪可选段；本轮P3无 `length`，不能靠提高32K/64K再修 `public_process` 导致的协议空答。其他 Benchmark 参数不改。

## 5. 零 HTTP 放行与阶段产物

1. 用**15组真实原始付费 ToolCall**在新生产解析器里回放：来源真实仍严检；普通单元 `parent_ids` 不再带坏合法候选；一条坏提案/坏检查不再毁掉整个 batch；只放宽指定文字字段容差。报告“新规则可留存几项候选”**不得冒充真实新版本模型生成率、晋升或得分**。
2. 用真实 `livemath:202512:23` 响应回放：主答案可提交、超长可选过程被拒且不进入 Skill 学习；伪造选项/ToolCall 类型依旧拒；不重新评分旧 trace。
3. 新版 Train60 身份固定，15×4 全覆盖，严格禁止 Val17/Test100 输入/gold/答案进入学习，去重与合并只使用 Train 来源。
4. 真实 Provider 完整 wire、候选注入卡片、checkpoint/已计费请求、预算/未知账本与 Frozen只读的零 HTTP 工程回归。`validated candidate→Val 6+11→Frozen→Test` 生产拦截模拟覆盖有候选/零候选、不同注入适用性、配对负增益、意外中断和恢复不二次付费。
5. 完成修复后建立**独立新版本**与目录：重新进行有界 P4 学习及开发性 Val 配对（会产生新的付费 HTTP，但不重解 Train60）；先审查候选质量与空答/成本，再决定是否进行 Test100。不要将人工理想 Bank 注入 Ours。

**禁止事项**：修改旧 P3 Frozen/收据使其事后通过；为追求资产数降低真实来源检验；把有限例子当一般定理证明；用旧 Val 分数当新 paid paired results；直接拿 P3 空 Bank 跑 Test 来证明 Skill 增益。
