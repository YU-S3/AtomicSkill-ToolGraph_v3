# AtomicSkillGraph v3.2：Agent-Driven Tool Evolution、运行时自动化与验证边界修复冻结文档

> 状态：冻结设计稿  
> 目标实现版本：v3.2  
> 基线提交：`7e57d9ddbcee8e7d131290ab9d6dd0aa11724907` — `Implement R3.1 runtime context observability`  
> 基线实验：R3.1 Fresh Empty-Bank ALFWorld Full-30  
> 本文性质：R3/R3.1 之后的正式方法补丁冻结文档。实现时不得重新解释本文已冻结的设计边界；如代码事实与本文假设冲突，必须先停下并重新确认，不允许用局部硬编码绕过。

---

# 0. v3.2 的定位

v3.2 不是继续给 Runtime Prompt、ToolCompiler 或 ALFWorld Validator 打零散补丁。

当前 Full-30 与关键 Trace 已经暴露出三个更高层问题：

1. **Planner 在确定“当前知识库根本不存在目标 Effect”时仍无条件调用 P1R，产生可避免的 LLM Repair 成本。**
2. **当前 ToolCompiler 把 Extractor 选择的连续 Evidence Slice 机械转换成 Tool Program，把“证明某个 Atomic 发生过”与“以后如何稳定执行该 Atomic”错误地等同。**
3. **当前 Runtime 中真正的主要 token 成本发生在 Tool 调用以前的 Grounding / Exploration；而这些操作中存在大量 Agent 能够提前预判的、重复、机械、低语义价值流程，但目前仍要求每一步重新调用 LLM。**

同时，Full-30 还揭示了任务级判断与节点级验证职责混淆的问题：

- Benchmark Environment 的 `won=true` 是唯一任务成功 authority；
- ours Validator 不再承担“任务是否成功”的第二套判断；
- ours Validator 的职责是定位 **Atomic / Implementation / Tool / Graph node** 是否满足自身契约；
- 如果 Benchmark 已经成功，后续图节点必须立即停止，不允许继续调用 LLM、Tool 或环境；
- 提前终止同时是对 Extractor / Composite 最小性的学习信号，而不是通过继续执行把原图“补完”。

因此 v3.2 的核心方法变化冻结为：

> **能力抽象与程序抽象分离；Tool 由 Agent 设计、由系统验证；Runtime Agent 可在执行前预判低价值重复工作并提出自动化 Atomic，再由 ToolBuilder 子智能体生成可零 LLM 执行的受限 Tool IR。**

系统中长期保持以下职责边界：

```text
LLM / Agent
├─ 任务能力分解
├─ Atomic 语义提出
├─ Tool 程序设计
├─ 是否值得自动化的判断
├─ 非确定性的候选选择
└─ Composite / Graph 语义组织

Code Authority
├─ Benchmark / Harness 边界
├─ Structured Evidence
├─ Predicate Vocabulary
├─ R0 / R1 验证
├─ Contract Compatibility
├─ Tool IR 静态安全
├─ Tool 执行与 Action Budget
├─ Registry / Alignment / Lifecycle
├─ Trace / Usage Ledger
└─ Benchmark won terminal authority
```

禁止重新滑回：

```text
Code:
if heat -> microwave
if clean -> sink
if look -> lamp
if target is cup -> search cabinets
```

这类 v2 式 workflow hardcode。

---

# 1. v3.2 七个闭合设计块

| 编号 | 设计块 |
|---|---|
| A | Planner Repairability + P1/P1R Bank Awareness + Support Atomic Integration |
| B | Benchmark Won Authority + Early Termination + Minimal Causal Graph |
| C | Semantic Evidence Model + Node-level Validator + Non-contiguous Evidence |
| D | Canonical Atomic → ToolBuilder Sub-Agent → Tool Validation |
| E | Runtime Self-Tooling + Automation Atomic + Task-local Trial |
| F | Unified Tool IR + Zero-LLM Execution + Compiler / Registry / Lifecycle |
| G | Anti-v2 Boundary + Observability + Formal Experiment Gates |

v3.2 **不重写**以下已经实现但本轮没有足够实验覆盖的模块：

- ColdStart；
- Failure Extractor；
- Provisional promotion；
- Atomic Composition 主流程。

这些模块保持现有代码，只在后续专项实验中验证，不为让 Full-30 指标变好而临时修改。

---

# 2. A — Planner Repairability 与 Knowledge Awareness

## 2.1 P1 仍然 Task-first

P1 **不得看到完整 Skill Bank 先验**。

P1 输入继续围绕：

```text
Task goal
Initial observation
TaskContract
Harness profile
Multiplicity / Repeat formal rules
```

P1 的职责是：

> 根据 TaskContract 说明“任务真正需要哪些 reusable state-transition capabilities”。

不能让 P1 因为当前 Bank 里只有某几类能力而扭曲任务分解。

冻结原则：

> **Task-first decomposition, Bank-aware repair.**

---

## 2.2 P1 后按 Atomic contract 检索

Planner 的主要可执行知识入口保持为 Atomic，而不是 Tool：

```text
Requirement
    ↓
AtomicRetriever
    ↓
Contract-compatible Atomic candidates
```

Tool 继续通过：

```text
Atomic
  ↓
Implementation
  ↓
Tool
```

被 Runtime 调用。

因此 Runtime 自创建的工具如果未来要被 Planner 复用，必须最终同时拥有合法 Atomic 身份，不能只在 ToolRegistry 中孤立存在。

---

## 2.3 Repairability Gate

建议新增：

```text
src/atomic_skillgraph/planner/repairability.py
```

概念结构：

```python
@dataclass(frozen=True)
class RepairabilityDecision:
    repairable: bool
    reason_code: str
    requirement_ids: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]
```

Gate 输入：

```text
P1 RequirementBundle
P1 deterministic bundle validation
Atomic retrieval results
related Composite hints
```

Gate 不调用 LLM。

### 允许 P1R

包括：

```text
planner_requirement_bundle_invalid
coverage_partial_effect_match
input_contract_mismatch
output_contract_mismatch
semantic_type_mismatch
cardinality_mismatch
role/interface mismatch
near-match Atomic exists
related Composite provides meaningful interface evidence
```

### Hard capability gap

当 required Requirement 满足：

1. Requirement bundle 本身合法；
2. desired Effect predicate 合法；
3. 当前可用 Atomic 中没有任何候选能够表达 required Effect family；
4. rejection 核心原因是 Effect predicate 根本不存在，而不是可修接口问题；

则：

```text
repairable = false
reason = planner_hard_capability_gap
```

直接跳过 P1R。

禁止把：

```text
object.observed_with
```

改成：

```text
object.heated
```

仅为了匹配已有 Bank。

---

## 2.4 P1R 的 Bank Awareness

P1R 可以看到 **P1 后未完全匹配的局部候选**，但不能看到完整 Bank。

允许：

```text
Requirement id
Requirement desired effects
covered / uncovered
near-match Atomic interface
near-match Atomic effects
near-match input/output role types
contract mismatch codes
related Composite interface hints
deterministic bundle validation
```

不提供：

```text
完整 Bank
所有 Tool body
所有历史 Composite 内容
所有 lifecycle ledger
```

---

# 3. A2 — Support Atomic 是正常 Atomic

辅助 Atomic 不新增特殊节点类型。

例如：

```text
locate compatible entity
```

仍然是：

```text
AbstractAtomicSkill
```

它可以：

- 被 Planner 检索；
- 进入 Composite；
- 作为正常 Graph occurrence；
- 建 DataFlow；
- 拥有 Implementation / Tool；
- 走正常 Lifecycle。

区别只是它的 Effect 可以属于 `evidence` domain，而不是直接完成 TaskContract world Effect。

P1 不需要为了已有辅助 Atomic 改写任务要求；Support Atomic 在 P1R/P2 或 Runtime missing-binding 阶段被考虑。

---

# 4. A3 — Runtime Support Atomic Retrieval

如果旧 Composite 已经命中，但之后知识库新增一个可以帮助参数解析的辅助 Atomic，不应等下一次 Composite 重学后才能使用。

因此 v3.2 冻结：

> 当前 Runtime occurrence 因 missing binding / unresolved evidence 阻塞时，可进行 generic support Atomic retrieval。

建议新增：

```text
src/atomic_skillgraph/runtime/support_retriever.py
```

唯一依据：

```text
Current blocked Atomic
    ↓
missing input roles / required resolution
    ↓
normal Atomic Bank
    ↓
candidate outputs/evidence effects can satisfy missing roles
```

禁止：

```text
if object == cup
if task_type == heat
```

这种检索规则。

代码只返回 formal-compatible support candidates。

Runtime Agent 决定：

- 调用 support Atomic；
- environment_action；
- 提出新的 Runtime Automation Atomic；
- cannot_resolve / plan_conflict。

如果选择 support Atomic，Runtime 可以临时插入普通 occurrence：

```text
Support A
   │ output/evidence
   ▼
Blocked B
```

Trace 记录：

```yaml
runtime_graph_augmentation:
  support_atomic_ref:
  producer_occurrence_id:
  consumer_occurrence_id:
  data_flow_roles:
  reason: missing_binding_support
```

Success Extractor 后续可决定是否把这个结构正式写入 Composite。

---

# 5. B — Benchmark Won 是唯一任务成功 Authority

对于 ALFWorld：

```text
HarnessActionResult.won == true
```

即任务成功。

ours Validator 不再维护第二套任务成功 authority。

```text
Benchmark won = true
      ↓
TASK SUCCESS
      ↓
immediate terminal
```

立即停止：

- 后续节点；
- Runtime Agent；
- Tool；
- LLM；
- environment actions。

---

# 6. B2 — Terminal Latch

需要双层保护。

## Harness

`harness/alfworld.py`

一旦 `_done` / `_won` 已成立，不允许再次进入 benchmark `env.step()`。

## Orchestrator

`runtime/orchestrator.py`

每次环境动作和 Tool 内部动作后检查 task terminal。

## ToolRunner

`runtime/tool_runner.py`

每个 Tool IR ACTION 后若 won：

```text
停止 Tool 后续 IR
返回 terminal-interrupted result
```

---

# 7. B3 — Task / Atomic / Tool / Composite 成功分离

## Task Success

```text
benchmark won
```

## Atomic Success

```text
Atomic validator passes its own Effects
```

## Tool Execution Success

```text
Tool intended path completes
+
required Tool step/final effects pass
```

## Implementation Success

```text
Tool execution + mapping + Atomic effect
```

## Composite Full Execution Success

所有必要 planned occurrences 在 terminal 前实际完成。

Benchmark won 不能反向证明未执行 Tool、Atomic 或 Composite 正确。

---

# 8. B4 — won 发生在 Tool 中

例如：

```text
Tool:
step1
step2 -> won=true
step3
step4
```

只执行 step1/step2。

记录：

```yaml
task_terminal_during_tool: true
tool_completed: false
tool_intrinsic_failure: false
executed_step_count: 2
remaining_step_count: 2
```

如果 step2 后 Atomic Effect 已满足：

```text
atomic_effect_passed = true
```

否则 false。

若 Composite 还有剩余节点：

```text
composite_full_execution_success = false
```

不能因为 task won 给完整 Composite self-sufficient credit。

---

# 9. B5 — Early Success 与最小因果图

## Atomic 最小性

一个 Atomic 的 Evidence Support 只包含证明其：

```text
precondition -> effect
```

所需的最小语义事件支持集。

但 Atomic 仍然可以需要：

- DataFlow；
- 前置节点输出；
- Runtime Agent 参数解析；
- support Atomic 输出。

“最小”不是简单最少动作数。

---

## Composite 最小性

Extractor/E2 Prompt 要求：

> 构造 causally sufficient minimal subgraph。

但这不做成强硬代码最小图判定器。

### Code 只 hard reject

- terminal 后未执行节点被当作成功证据；
- 未执行节点记为 completed；
- DataFlow 不闭合；
- Atomic 无合法 evidence；
- TaskContract/cardinality/identity 不合法；
- Graph 结构非法。

### 不 hard reject

代码不能因为：

```text
“似乎还能少一个节点”
```

就拒绝 Composite。

否则合法图容易被过审并回 Dynamic。

---

## Extractor Prompt

新增：

```text
- Only extract causal capabilities supported before benchmark terminal success.
- Do not preserve a planned node merely because it existed in RuntimePlan.
- Prefer a causally sufficient minimal capability set.
- Keep a support capability when its output/evidence is actually consumed.
- Do not minimize node count at the cost of required DataFlow or precondition support.
```

---

## Lifecycle

如果旧图：

```text
A -> B -> C
```

反复出现：

```text
A -> B -> won
```

Extractor 可学新图：

```text
A -> B
```

由现有：

```text
candidate -> active -> stable replacement
```

使旧图：

```text
active -> suppressed -> retired
```

不新增 hard-delete。

---

# 10. C — SemanticPredicate 增加 Effect Domain

当前 `SemanticPredicate` 扩展：

```python
class EffectDomain(str, Enum):
    WORLD = "world"
    EVIDENCE = "evidence"

@dataclass(frozen=True)
class SemanticPredicate:
    predicate: str
    args: dict[str, BindingExpression | Any]
    cardinality: int = 1
    distinct_by: str = ""
    effect_domain: EffectDomain = EffectDomain.WORLD
```

domain 直接属于 Predicate，不放 Atomic metadata。

---

## World Effect

例如：

```text
agent.holds
object.at_location
object.cleaned
object.heated
container.open
```

## Evidence Effect

例如：

```text
entity.discovered_at
binding.discovered
entity.available_for_action
```

具体 vocabulary 由 Harness 能力决定。

---

# 11. C2 — Predicate Vocabulary Authority

LLM 可以：

- 组合已有 predicate；
- 绑定角色；
- 设计 world/evidence effect；
- 建 inputs/outputs；

但不能自行定义系统当前无法验证的新 predicate 名。

Harness protocol 建议增加：

```python
def semantic_predicate_schema(self) -> list[PredicateSpec]:
    ...
```

PredicateSpec 至少：

```yaml
predicate:
effect_domain:
argument_roles:
argument_semantic_types:
validation_source:
```

这是环境接口适配，不是 task workflow hardcode。

---

# 12. C3 — Validator 只验证 ours

Generic Validator 只消费：

```text
SemanticPredicate
Bindings
Semantic Evidence
Revision
Identity
Cardinality
```

它不根据：

```text
task_type
task family
goal wording
```

决定一个 Atomic 如何执行。

Benchmark-specific 内容只在 Harness Evidence Adapter：

```text
raw action
actual observation
metadata
action catalog
    ↓
semantic evidence
```

---

# 13. C4 — Non-contiguous Atomic Evidence

保留：

```yaml
event_start:
event_end:
```

但定义为 **Evidence Envelope**，不再等同于因果 action slice。

新增：

```yaml
support_event_ids: [...]
precondition_witness_refs: [...]
effect_witness_refs: [...]
ordering_constraints: [...]
```

真正支持 Atomic 的是 support set。

Envelope 内无关事件：

- 不需要成为 Atomic input；
- 不需要 ToolBuilder复制；
- 不因为不连续而强行吸入 Atomic。

---

## 跨 Span

support events 可跨同一：

```text
logical occurrence / causal lineage
```

中的嵌套 RuntimeSpan / ToolSpan。

禁止跨：

```text
另一个无关 occurrence
另一个 task
无因果 lineage 的 node
```

---

## Atomicizer 改造

当前连续：

```text
selected = events[start:end]
```

改为：

```text
envelope_events
support_events
```

验证：

1. support event 位于 envelope；
2. accepted；
3. revision合法；
4. causal lineage合法；
5. precondition witness合法；
6. effect witness合法；
7. reusable contract 不泄露 episode concrete constant；
8. 非 support envelope event 不要求进入 Tool。

---

# 14. D — ToolBuilder Sub-Agent

ToolBuilder 是 v3.2 唯一 Tool Program author。

两种来源：

```text
Success Evolution:
Canonical Atomic -> ToolBuilder

Runtime:
Automation Atomic Draft -> R0 -> ToolBuilder
```

两条路径使用同一 Builder 和同一 Tool IR。

---

# 15. D2 — `create_tool` 只给 ToolBuilder

Runtime 主 Agent **不拥有 `create_tool`**。

Runtime 主 Agent拥有结构化入口：

```text
propose_runtime_automation_atomic
```

当主 Agent在 reasoning 中预判后续将是：

- 重复；
- 机械；
- 低语义价值；
- 可 bounded；
- 有确定 stop condition；

时提出 Automation Atomic。

R0 通过后由系统启动 ToolBuilder。

只有 ToolBuilder 有：

```text
create_tool
```

---

# 16. D3 — `create_tool` 是 ToolProposal Submission

`create_tool` 不直接写 ToolRegistry。

流程：

```text
ToolBuilder
   ↓
create_tool(ToolProposal)
   ↓
Schema validation
   ↓
Tool Program Static Validation
   ↓
ToolCompiler
   ↓
Task-local trial / Evolution admission
```

LLM 不拥有 admission authority。

---

# 17. D4 — ToolBuilder Context 冻结

## 必须提供

### Canonical Atomic

```text
canonical intent
inputs
outputs
preconditions
effects
effect_domain
required_resolution
identity constraints if applicable
```

### Atomic Evidence Support

```text
support_event_ids
action_type
arguments
before_revision
after_revision
ordering
witness refs
```

### Structured semantic delta

```text
before facts
after facts
effect witnesses
evidence deltas
```

### Harness Interface

只提供：

```text
harness profile
primitive action schemas
predicate vocabulary
action/evidence interface
```

### Tool IR Schema

```text
ACTION
IF
FOR_EACH
STOP_WHEN
RETURN
```

### Safety / Portability

```text
No Python
No shell
No filesystem
No network
No task id constants
No episode entity constants
No benchmark family branch
No hidden LLM call
Bounded max_actions
Evidence-backed outputs
```

---

## 少量 raw observation

默认只看 structured semantic delta。

只有 support event 直接相关且结构化 evidence 不足时，提供少量 raw observation。

---

## 局部失败事实

仅当失败：

- 与当前 Atomic support envelope 直接相关；
- 可防止 Builder 重建已知失败动作；
- 可帮助补 precondition / branch；

才提供。

---

## Near-match

先 deterministic exact/equivalent matching。

exact Tool 已存在：

```text
不调用 ToolBuilder
```

无 exact reuse 时可给 Builder：

```text
near-match interface
```

不提供完整 Bank / 完整旧 Tool body。

---

## 明确不给

```text
完整 Task Goal
完整 Trace
完整 Planner history
完整 Skill Bank
完整旧 Tool bodies
完整 Action Catalog history
benchmark task-family workflow
```

---

# 18. D5 — NO_TOOL

ToolBuilder 允许：

```yaml
decision: no_tool
reason_code: ...
```

当无法提出：

- 有证据；
- 安全；
- bounded；
- 可复用；
- 满足 Atomic contract；

的 Tool 时不强造。

Atomic 本身仍可由 Seeded Agent 执行。

---

# 19. D6 — ToolBuilder 必须声明 Step-level Effect

ToolProposal 不能只有 program。

每个关键 program node 应明确：

```text
expected semantic effects
```

并声明 final Atomic effects。

目的：

> 失败时定位到 Tool node / effect / Implementation / Atomic 层，而不是全部退化成 `atomic_effect_violation`。

示例：

```yaml
program:
  - node_id: activate_light
    op: ACTION
    expected_effects:
      - predicate: light.on
        effect_domain: world

  - node_id: acquire
    op: ACTION
    expected_effects:
      - predicate: agent.holds
        effect_domain: world

final_effects:
  - predicate: object.observed_with
    effect_domain: world
```

---

# 20. E — Runtime Self-Tooling Prompt

Runtime Prompt 增加：

> 在真正执行一串动作之前，如果你判断接下来主要是重复、机械、低语义价值，并且循环对象、条件和停止条件可以通过当前结构化 action/evidence 表达，应优先考虑提出 Automation Atomic，而不是逐步消耗 LLM turn。

关键：

> Agent 可以在重复操作发生 **之前** 预判。

不要求已经做过 N 次。

代码也不能写：

```python
if missing_object:
    auto_create_search_tool()
```

是否自动化仍由 Agent reasoning 决定。

---

# 21. E2 — Runtime Automation Atomic Draft

Runtime Agent 提交：

```text
intent
inputs
outputs
preconditions
effects
effect_domain
rationale
```

不提交任意 source code。

例如：

```text
locate entity matching semantic target
```

可能产生：

```text
entity.discovered_at(object, location)
domain=evidence
```

---

# 22. E3 — R0 / R1 两阶段验证

## R0：执行前结构验证

Tool 尚未执行，因此不要求 effect witness。

R0 验证：

```text
schema
role closure
input/output
predicate vocabulary
effect domain
precondition expressibility
semantic type
no concrete episode leakage in reusable contract
```

R0 pass 只代表：

> 可以交 ToolBuilder尝试实现。

不代表 Atomic 已正式验证。

---

## R1：task-local trial 后证据验证

验证：

```text
actual executed path
semantic deltas
outputs
bindings
effect/evidence witnesses
```

回答：

```text
Atomic effect passed?
Tool path passed?
outputs evidence-backed?
```

---

# 23. E4 — Task-local Trial

冻结流程：

```text
Runtime Atomic proposal
      ↓
R0
      ↓
ToolBuilder
      ↓
Static Tool pass
      ↓
TASK_LOCAL_TRIAL
      ↓
R1
```

Trial 期间：

- 当前 task 可调用；
- 其他 task 不可检索；
- 不进入 Planner 全局 Bank；
- 内部 action 正常 consume action budget；
- 全部写 Trace；
- 可触发 benchmark terminal。

---

# 24. E5 — Runtime 资产最终由 Success Extractor 审查

Runtime Agent 不直接 author 正式长期知识。

如果当前 task：

```text
benchmark won = true
```

Success Extractor 额外看到：

```text
runtime_created_atomic_drafts
runtime_created_tool_proposals
runtime_tool_trials
R0 reports
R1 reports
```

Extractor 可以：

- 保留；
- 修改；
- reject；
- exact merge；
- 与普通 E1 Atomic 一起 canonicalize；
- 决定是否纳入 Composite。

最后仍走：

```text
Extractor
→ deterministic validation
→ alignment
→ admission
→ lifecycle
```

如果 task 失败：

```text
Runtime Tool proposal + trial trace
```

只保留证据，v3.2 主线不正式入库，后续交 FailureExtractor。

---

# 25. E6 — Candidate Projection 标注 Evidence Level

当前 candidate projection 增加来源强度。

建议：

```text
observed_candidate
relation_candidate
affordance_certified
confirmed_binding
```

例如：

```yaml
candidate_bindings:
  station:
    - value: cabinet_7
      evidence_level: observed_relation
    - value: microwave_1
      evidence_level: invocation_affordance_certified
```

这只是帮助 Agent 理解证据强度。

唯一 assignment auto-confirm 的 R3 authority 不变。

---

# 26. F — Unified ToolAsset

不新增：

```text
SearchTool
GroundingTool
RuntimeTool
AtomicTool
```

长期资产统一：

```text
AbstractAtomicSkill
    ↓
ImplementationAtom
    ↓
ToolAsset
```

搜索 / 发现 / 参数解析类 Tool 仍然是正常 Tool，只是其 Atomic Effect 可为 evidence domain。

---

# 27. F2 — Tool IR v1

第一版只实现：

```text
ACTION
IF
FOR_EACH
STOP_WHEN
RETURN
```

IR 顶层保留：

```text
schema_version
```

和 opcode 扩展空间。

v3.2 不实现未来 opcode。

---

# 28. F3 — ACTION

ACTION 表示：

> 根据 Tool variables 编译一个 Harness primitive。

只能使用 Harness 声明的 action schema。

不能携带：

```text
arbitrary shell
Python
raw executable source
```

---

# 29. F4 — IF

Condition 只能读取：

```text
Tool input
Tool local variable
current action catalog
semantic evidence
binding evidence
```

禁止 LLM natural-language condition 和任意代码表达式。

未执行 branch 可以存在，但 path coverage 标记为：

```text
unvalidated
```

---

# 30. F5 — FOR_EACH

必须包含：

```text
collection_source
iteration_variable
body
max_iterations
```

Collection source 只能来自：

```text
current action catalog projection
semantic evidence query
Tool input array
Tool local deterministic collection
```

禁止 recursion / arbitrary generator。

---

# 31. F6 — STOP_WHEN

必须依赖结构化可验证 condition，例如：

```text
target evidence appears
binding discovered
Atomic effect passed
required output available
```

不能写：

```text
when search seems enough
```

---

# 32. F7 — RETURN

required output 必须可从：

```text
Tool input
local binding
structured evidence
```

确定性解释，并附 evidence refs。

---

# 33. F8 — Bounded Execution

所有 Tool 有：

```text
max_actions
```

每个 ACTION：

```text
consume one existing action budget
```

不新增无限内部预算。

---

# 34. F9 — Loop / Condition Evidence，避免 Runtime Trial 矛盾

这里必须区分两类来源。

## 成功历史 Trace 的 ToolBuilder

### Loop

至少出现 **2 次结构同构重复** 才能从历史 evidence 泛化 `FOR_EACH`。

### IF

实际走过 branch 为 validated；未走 branch为 unvalidated。

### STOP_WHEN

历史 Trace 中应有 stop predicate 从 false → true 的证据。

### RETURN

返回值必须有 witness。

---

## Runtime 预判 Tool

Runtime Agent 正是在循环发生之前提出自动化，因此 **R0 不要求已经出现2次历史 repetition**。

流程：

```text
R0 static safe
→ task-local trial
→ R1
```

Trial 自己产生 path evidence。

如果 Trial 真实走了多次 FOR_EACH iteration：

```text
记录 observed repetition
```

若只走一次：

```text
loop structure static-safe
generalization evidence weak
```

保持 Candidate，未来独立 task 继续积累。

这避免为了证明循环先浪费本来想节省的 LLM token。

---

# 35. F10 — Path Coverage

Tool 记录：

```text
program path identity
validated paths
unvalidated paths
observed loop iteration counts
stop-condition evidence
```

一次 Trace 不要求证明所有未来 branch。

Admission：

```text
Static safety pass
+
current executed path evidence
```

后续由既有 lifecycle：

```text
Candidate -> Active -> Preferred
```

积累跨任务证据。

---

# 36. F11 — Failure Localization

Tool execution diagnostics 至少扩展为：

```yaml
failure_layer:
  tool_ir
  tool_step
  tool_effect
  implementation
  atomic

program_node_id:
path_id:
expected_effects:
observed_effects:
missing_effects:
```

目标：

> 精确区分程序结构错、具体 Tool step 错、Implementation mapping 错和最终 Atomic effect 错。

---

# 37. F12 — ToolCompiler 降级为 Compiler

当前 `ToolCompiler` 的：

```text
Atomic occurrence action_events
→ primitive steps
```

程序合成逻辑删除。

v3.2 后：

```text
ToolProposal
→ normalized IR
→ BindingExpression
→ ToolRef
→ input/output schema
→ ToolAsset
→ ImplementationAtom
```

ToolCompiler 不决定哪些 action 应进入 Tool。

建议接口：

```python
class ToolCompiler:
    def compile_proposal(
        self,
        atomic: AbstractAtomicSkill,
        proposal: ToolProposal,
        provenance: ToolProvenance,
    ) -> CompiledToolBundle:
        ...
```

---

# 38. F13 — 推荐新模块

为避免 Runtime 依赖 Evolution 高层，推荐新增中性目录：

```text
src/atomic_skillgraph/tooling/
    __init__.py
    proposal.py
    builder_session.py
    ir.py
    validator.py
```

其中：

```text
proposal.py       ToolProposal / NO_TOOL
builder_session.py ToolBuilder Sub-Agent
ir.py             Tool IR AST
validator.py      static/path validation
```

Runtime 与 Success Evolution 共用。

---

# 39. F14 — Registry / Lifecycle 统一

Runtime-created / Extractor-created Tool 最终都进入现有：

```text
ToolRegistry
ArtifactStore
EvidenceLedger
Lifecycle
```

不新增第二套 ToolRegistry。

本补丁不改 lifecycle thresholds。

---

# 40. Success Evolution 完整新链路

```text
Benchmark-won Trace
      ↓
Trace Normalizer
      ↓
Extractor E1
      ↓
Non-contiguous Atomic proposals
      ↓
Atomicizer
      ↓
Canonical Atomics
      ↓
exact/equivalent existing Implementation/Tool?
   ┌──┴──┐
  yes    no
   │      ↓
 reuse ToolBuilder
          ↓
     create_tool
          ↓
   Tool Validation
          ↓
      Compiler
          ↓
Implementation + Tool
          ↓
Extractor E2
          ↓
causally sufficient graph
          ↓
Alignment / Admission / Lifecycle
```

Runtime-created draft/trial 同样进入 E1 可见上下文，由 Extractor最终决定。

---

# 41. Runtime Self-Tooling 完整新链路

```text
Current Atomic
      ↓
Runtime Agent
      ↓
predicts repetitive / mechanical /
low-semantic-value work
      ↓
propose_runtime_automation_atomic
      ↓
R0
  ┌───┴────┐
reject     pass
 │          ↓
diag    ToolBuilder
            ↓
        create_tool
            ↓
     Static Tool Validation
       ┌────┴────┐
    reject      pass
      │           ↓
     diag    task-local trial
                  ↓
        zero-LLM internal actions
                  ↓
                 R1
                  ↓
          outputs/evidence
                  ↓
          Runtime Agent resumes
```

---

# 42. ToolBuilder Prompt 核心约束

必须表达：

```text
You implement one already-proposed Atomic.
The Atomic contract is authoritative.
The source trace is evidence, not a program to replay.
Use the minimal reusable procedure needed to realize the Atomic.
Do not copy every event in the evidence envelope.
Do not add task-specific workflow knowledge.
Every branch, output, and expected effect must be representable in the supplied Harness interface.
Return NO_TOOL if no safe reusable bounded implementation is justified.
```

---

# 43. Extractor Prompt 核心约束

E1：

```text
- Propose minimal semantic capability occurrences.
- event_start/end are the temporal evidence envelope only.
- Explicitly select support_event_ids.
- Support events may be non-contiguous within one causal occurrence lineage.
- Do not include unrelated actions merely to make the interval contiguous.
- Precondition and effect witnesses must be explicit.
```

E2：

```text
- Use only validated canonical Atomics.
- Runtime-created support Atomics are ordinary candidates.
- Prefer a causally sufficient minimal subgraph.
- Do not retain planned-but-unexecuted post-terminal nodes.
- Keep support Atomic if its evidence/output is consumed.
```

---

# 44. Planner P2 Prompt

P2 可以得到：

```text
required Atomic candidates
support Atomic candidates
candidate input/output compatibility
edge evidence
```

明确：

> Support Atomic 不要求直接覆盖 TaskContract Requirement；只有当其 output/evidence 被 downstream required occurrence消费时才应进入 Graph。

禁止无作用辅助节点。

---

# 45. 主要数据结构改动

## `core/contracts.py`

新增：

```text
EffectDomain
SemanticPredicate.effect_domain
```

不增加 SearchAtomic / EpistemicAtomic 类型。

---

## E1 Schema

新增：

```text
support_event_ids
precondition_witness_refs
effect_witness_refs
ordering_constraints
```

保留：

```text
event_start
event_end
```

---

## ToolProposal

概念：

```yaml
proposal_version:
decision: create | no_tool
summary:
atomic_ref:
inputs:
outputs:
program:
max_actions:
final_effects:
evidence_outputs:
path_expectations:
rationale:
```

---

## Runtime Automation Draft

```yaml
draft_id:
intent:
inputs:
outputs:
preconditions:
effects:
rationale:
source_occurrence_id:
```

---

# 46. Tool IR 顶层

```yaml
schema_version: 1
max_actions: integer
program:
  - node_id:
    op: ACTION | IF | FOR_EACH | STOP_WHEN | RETURN
    ...
```

未知 opcode 必须：

```text
tool_ir_opcode_unsupported
```

不能静默忽略。

---

# 47. Tool 静态验证 Gate

至少验证：

1. Schema；
2. opcode whitelist；
3. unique node IDs；
4. input role closure；
5. local variable closure；
6. RETURN output closure；
7. predicate vocabulary；
8. effect_domain；
9. Harness action schema；
10. no episode concrete ID leakage；
11. no arbitrary code；
12. bounded max_actions；
13. FOR_EACH bounded；
14. collection source合法；
15. IF condition source合法；
16. no recursion；
17. final effects compatible with Atomic contract；
18. evidence outputs可验证；
19. Harness profile compatible；
20. program path可计算。

---

# 48. Tool 内部执行

每个 ACTION：

```text
compile against latest action catalog
consume action budget
execute Harness
append EnvironmentActionRecord
update TaskRuntimeContext
refresh R3 grounding
refresh evidence
check benchmark won
check STOP_WHEN
```

Tool 内部不调用 LLM。

---

# 49. ToolBuilder Token Budget

不增加：

```text
每task最多N个Tool
```

这种人为行为限制。

所有 ToolBuilder 调用正常计入现有 task token budget。

新增 usage bucket：

```text
tool_builder_runtime
tool_builder_evolution
```

可汇总为：

```text
tool_builder
```

正式 Runtime：

```text
node 100000
task 300000
```

不提高。

---

# 50. 任务与资产成功记录

建议 Trace 明确：

```yaml
task_success:
  won:
  terminal_revision:
  terminal_origin:

atomic_result:
  passed:
  effect_witnesses:

tool_result:
  completed:
  terminal_interrupted:
  intrinsic_failure:
  executed_nodes:
  path_id:

implementation_result:
  passed:

composite_result:
  all_required_occurrences_completed:
  skipped_remaining_occurrences:
```

任务成功与 ours asset quality 不再混在一个 boolean 中。

---

# 51. v3.2 最低 Observability

## Planner

```text
planner_repairability_gate_count
planner_repairability_repairable_count
planner_hard_capability_gap_count
planner_p1r_skipped_hard_gap_count
planner_support_atomic_candidate_count
planner_support_atomic_selected_count
```

## Terminal

```text
task_terminal_early_success_count
task_terminal_during_tool_count
task_terminal_with_remaining_occurrences_count
terminal_skipped_occurrence_count
```

## Evidence

```text
extractor_noncontiguous_atomic_count
extractor_support_event_count
extractor_envelope_event_count
extractor_redundant_envelope_event_excluded_count
```

## ToolBuilder

```text
tool_builder_call_count
tool_builder_no_tool_count
tool_builder_proposal_count
tool_builder_static_pass_count
tool_builder_static_rejection_count
```

## Runtime Self-Tooling

```text
runtime_automation_atomic_proposal_count
runtime_automation_r0_pass_count
runtime_automation_r0_reject_count
runtime_tool_trial_count
runtime_tool_trial_r1_pass_count
runtime_tool_trial_r1_reject_count
runtime_tool_internal_action_count
runtime_tool_llm_bypassed_action_count
```

## Support reuse

```text
runtime_support_retrieval_count
runtime_support_candidate_count
runtime_support_selected_count
runtime_support_success_count
runtime_graph_augmentation_count
```

## Path Coverage

```text
tool_validated_path_count
tool_unvalidated_path_count
tool_observed_loop_iteration_count
tool_stop_condition_witness_count
```

正式论文展示指标可以后续筛选，但 Trace event 至少需要存在。

---

# 52. 文件级实现落点

| 文件 | v3.2 责任 |
|---|---|
| `core/contracts.py` | `EffectDomain`、Predicate domain |
| `agents/structured_submission.py` | E1、Runtime Atomic、ToolProposal schemas |
| `agents/context_builder.py` | Runtime automation、ToolBuilder、E1/E2 context |
| `planner/pipeline.py` | Repairability Gate、P1R skip、support integration |
| `planner/atomic_retriever.py` | formal contract diagnostics |
| `planner/requirement_agent.py` | P1 blind、P1R bounded hints |
| `planner/repairability.py` **new** | deterministic Gate |
| `evolution/extractor_session.py` | support_event_ids、runtime drafts |
| `evolution/atomicizer.py` | envelope/support split、causal lineage |
| `evolution/tool_compiler.py` | ToolProposal → Asset |
| `evolution/composite_builder.py` | terminal/minimal causal graph适配 |
| `runtime/node_executor.py` | automation proposal、support invocations |
| `runtime/orchestrator.py` | terminal、runtime graph augmentation |
| `runtime/tool_runner.py` | IR v1、zero-LLM control、terminal interrupt |
| `runtime/support_retriever.py` **new** | support Atomic retrieval |
| `runtime/task_context.py` | task-local trial / runtime draft state |
| `runtime/grounding_state.py` | candidate evidence-level projection |
| `harness/protocol.py` | predicate vocabulary interface |
| `harness/alfworld.py` | semantic evidence adapter + terminal latch |
| `validation/atomic_validator.py` | domain-aware node validation |
| `validation/tool_validator.py` |新 IR适配 |
| `tooling/proposal.py` **new** | ToolProposal |
| `tooling/builder_session.py` **new** | ToolBuilder |
| `tooling/ir.py` **new** | Tool IR |
| `tooling/validator.py` **new** | static/path validation |
| `knowledge/tool_registry.py` |统一保存 Tool |
| `governance/lifecycle.py` |不改阈值，消费新 evidence |
| `system.py` | Success Evolution接线、runtime draft回流 |
| `experiments/report.py` | v3.2 metrics |
| `experiments/run_v3_smoke.py` | gates |

推荐 `tooling/` 使用中性目录，避免 Runtime→Evolution 形成循环依赖。

---

# 53. 正式参数明确不改

```yaml
llm:
  runtime:
    reasoning_effort: low
    max_completion_tokens: 32768
    max_total_tokens_per_node: 100000
    max_total_tokens_per_task: 300000

runtime:
  global_action_budget: 100
  node_action_budget: 35
```

同时不提高：

- Planner top-k；
- Planner repair次数；
- Extractor token cap；
- ColdStart限制；
- FailureExtractor限制；
- Lifecycle阈值。

---

# 54. Anti-v2 禁止项

Production 禁止：

```python
if task_type == "pick_heat_then_place_in_recep":
    search_microwave()

if object_type == "cup":
    search_cabinets()

if task_type == "look_at_obj_in_light":
    use_lamp_then_take()

if task_type == "clean":
    choose_sink()
```

也禁止预定义：

```text
ALFWorldCupSearchTool
HeatTaskSearchTool
CleanTaskToolTemplate
LookAtToolTemplate
```

允许：

```text
ALFWorld raw parser
primitive schemas
predicate vocabulary
observation -> semantic evidence
TaskContract adapter
ToolBuilder prompt说明 primitive interface
```

---

# 55. Deterministic Test Gates

## Gate A — P1 bank blind

Bank变化不得把完整 Bank 写入 P1 prompt。

## Gate B — hard gap skip P1R

目标 Effect family不存在时：

```text
planner_hard_capability_gap
P1R calls = 0
```

## Gate C — repairable mismatch仍P1R

相同 Effect family但接口/type/cardinality可修时：

```text
P1R = 1
```

## Gate D — benchmark terminal

won 后：

```text
no env.step
no Runtime LLM
no Tool action
no next occurrence
```

## Gate E — won不证明Tool

Tool step2触发won但还有step3：

```text
task success=true
tool completed=false
intrinsic failure=false
```

## Gate F — won不证明Composite

A→B→C，B触发won：

```text
task success=true
C not executed
Composite full execution=false
```

## Gate G — effect domain

旧 Predicate 默认 world；evidence Predicate正常序列化。

## Gate H — unknown predicate

Harness无法验证的新 predicate：

```text
R0 reject
```

## Gate I — non-contiguous evidence

support [3,7]，中间无关事件不得变成 Atomic/Tool inputs。

## Gate J — nested span lineage

同 occurrence parent/child span允许；无关 occurrence拒绝。

## Gate K — exact Tool reuse

已有等价 Tool：

```text
ToolBuilder calls = 0
```

## Gate L — bounded Builder context

ToolBuilder payload 不含完整 Task Goal、Planner history、完整 Trace/Bank。

## Gate M — create_tool ownership

Runtime Agent没有 `create_tool`；ToolBuilder有。

## Gate N — NO_TOOL

无法合法生成时 Atomic保留，不造假 Tool。

## Gate O — opcode whitelist

未知 opcode reject。

## Gate P — no arbitrary code

Python/shell/network/filesystem reject。

## Gate Q — bounded FOR_EACH

缺 max_iterations / max_actions reject。

## Gate R — historical loop evidence

历史 ToolBuilder 从一次 repetition 泛化 loop 不得当作已验证循环；两次同构重复可支持。

## Gate S — Runtime predicted loop

R0 不要求已有 repetition；task-local trial 产生 R1 path evidence。

## Gate T — zero-LLM internal loop

一个 Tool 执行10个 ACTION：

```text
internal LLM calls=0
action budget consumed=10
Trace actions=10
```

## Gate U — task-local visibility

创建后当前 task 可 trial；其他 task 在 admission前不可见。

## Gate V — won task admission

won task的 runtime draft 可被 Success Extractor审查；失败 task不正式 Success-admit。

## Gate W — support retrieval contract-based

只按 missing role/evidence contract匹配，不按 task family。

## Gate X — graph augmentation

Support Atomic作为普通 occurrence 插入并建 DataFlow。

## Gate Y — candidate evidence levels

policy context 区分 observed / affordance-certified / confirmed。

## Gate Z — no benchmark workflow

扫描新增 Production code 不得出现 family-specific workflow。

---

# 56. Targeted Smoke

重新 Full-30 前至少：

1. **P1 hard gap**：look-at Bank无 observed_with → 不P1R；
2. **Early terminal**：A→B→C，B won → C零调用；
3. **Non-contiguous Atomic**：support event非连续，不吸入中间 LOOK/EXAMINE；
4. **Success ToolBuilder**：Canonical Atomic → ToolBuilder → IR → execute → effect pass；
5. **Runtime predicted automation**：Agent执行搜索前提出 locate Atomic → R0 → ToolBuilder → FOR_EACH → STOP_WHEN → RETURN → R1；
6. **Support Atomic reuse**：旧 Heat→Place Composite + 新 Locate Atomic，missing object时 Runtime可检索 Locate；
7. **Tool terminal interruption**：Tool中途won，task成功但 Tool incomplete-not-failure；
8. **Failed-task isolation**：Runtime Tool trial局部成功但 task失败，不进入 Success Bank。

---

# 57. Full-30 启动条件

```text
[ ] deterministic tests全过
[ ] Repairability gates全过
[ ] benchmark terminal latch全过
[ ] task/node成功分离全过
[ ] effect_domain migration全过
[ ] predicate vocabulary全过
[ ] non-contiguous evidence全过
[ ] nested span lineage全过
[ ] ToolBuilder context bounded
[ ] create_tool ownership正确
[ ] Tool IR static validator全过
[ ] Runtime task-local trial全过
[ ] zero-LLM internal actions全过
[ ] R0/R1全过
[ ] support Atomic retrieval全过
[ ] runtime graph augmentation全过
[ ] Success Extractor可见runtime trial
[ ] failed task不Success-admit runtime Tool
[ ] Lifecycle阈值未漂移
[ ] Runtime预算未漂移
[ ] 无benchmark workflow hardcode
[ ] freeze新commit
[ ] fresh empty bank
[ ] 不resume旧Full-30
```

---

# 58. v3.2 Full-30 需要回答的问题

### Planner

Repairability Gate 是否减少：

```text
P1R calls
Planner tokens
```

且不降低 benchmark won。

### Tool Synthesis

LLM ToolBuilder 是否比旧机械 ToolCompiler：

- 程序更短；
- 少吸入无关 action；
- Effect correctness 更稳；
- 可产生真正可复用的多步自动化。

### Runtime Efficiency

重点看：

```text
Runtime Preparation calls/tokens
Seeded tokens
LLM-bypassed environment actions
Runtime Tool trial/reuse
support Atomic reuse
```

### Minimal Graph

Early terminal 是否逐渐产生更短、但仍充分的 Composite，并通过 Lifecycle 替换冗余旧图。

---

# 59. 基线说明

R3.1 Full-30：

```text
Benchmark Official won: 96.67%
旧 Strict TaskContract diagnostic: 90%
Graph self-sufficient: 63.33%
Total tokens: 3,512,685
Runtime prompt: 1,902,287
Node token exhaustion: 1
```

v3.2 后任务级正式成功率只以：

```text
Benchmark won
```

为 authority。

TaskContract agreement 可以保留为 diagnostic metric，但不再作为第二套任务成功判定。

---

# 60. 最终冻结结论

v3.2 的核心不是：

> “给 Agent 一个可以写循环的 Tool。”

而是：

> **Agent-driven executable capability evolution。**

能力抽象：

```text
Trace -> Atomic
```

回答：

> 学到了什么能力。

程序抽象：

```text
Atomic + minimal evidence -> ToolBuilder -> Tool
```

回答：

> 怎样把这项能力编译成稳定可复用执行过程。

Runtime 自动化：

```text
Reasoning predicts low-value repeated work
→ Automation Atomic
→ R0
→ ToolBuilder
→ task-local zero-LLM Tool
→ R1
```

回答：

> 当前没有现成 Tool 时，Agent 能否在真正浪费大量 token 前主动生成自动化过程。

最终长期知识仍然必须经过：

```text
Success Extractor
Deterministic Validation
Alignment
Registry
Lifecycle
```

Runtime Agent 不直接持有正式知识 authoring authority；ToolBuilder 不拥有 admission authority；Harness 只提供环境语义边界；Generic Validator 只验证 ours 的节点与资产；Benchmark `won` 是唯一任务成功 authority。

这一版不增加预算、不重开 ColdStart/FailureExtractor/AtomicComposition 主体、不写 benchmark workflow hardcode，以同一 Atomic / Tool / Graph / Lifecycle 体系完成从语义能力到可执行自动化工具的统一进化。
