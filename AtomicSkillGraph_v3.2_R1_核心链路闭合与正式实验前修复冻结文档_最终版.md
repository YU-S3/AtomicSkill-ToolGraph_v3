# AtomicSkillGraph v3.2-R1：核心链路闭合与正式实验前修复冻结文档

> 状态：**完全冻结，可直接进入实现**  
> 目标：修复当前 v3.2 实现与已冻结 v3.2 设计之间的断点，使 Agent-Driven Tool Evolution / Runtime Self-Tooling / Support Atomic / Benchmark Terminal Authority 真正形成可运行闭环。  
> 审查基线提交：`23cccd1592f490db13c225ca8d56b3d8f30383da` — `Complete runtime support Atomic invocation path`  
> v3.2 首次实现提交：`fbfc3bca4910d27b8044eb69de9bedcfda0915eb`  
> R3.1 基线提交：`7e57d9ddbcee8e7d131290ab9d6dd0aa11724907`  
> 本文不是新的方法版本设计，不改变 v3.2 已冻结的方法方向；它只修复当前实现尚未闭合的控制流、Evidence、Tool IR、Admission、Replay、Planner Support、预算和实验协议问题。  
> 本文所有设计边界均已确认。实现时不得重新解释、弱化或绕过本文冻结规则；如代码事实与本文冲突，必须先停下重新确认。

---

# 0. 当前结论

当前代码已经完成 v3.2 的主要结构搭建：

- `SemanticPredicate.effect_domain` 已存在；
- Runtime 主 Agent 不拥有 `create_tool`；
- `ToolBuilderSession` 已成为独立子智能体；
- Runtime 已有 `propose_runtime_automation_atomic`；
- R0 → ToolBuilder → Static Validation → task-local trial → R1 的骨架存在；
- `ACTION / IF / FOR_EACH / STOP_WHEN / RETURN` IR 类型已存在；
- `SupportAtomicRetriever` 和 `invoke_support_atomic` 路径已存在；
- Harness terminal latch 已存在；
- v3.2 metrics/report 字段已开始接入。

因此 **不需要推倒 v3.2 重写**。

但是当前实现还存在多处确定性的链路断点：

```text
ToolBuilder 能提出 Tool
        ↓
但 Tool IR 还无法可靠表达核心 search automation
        ↓
Admission 不识别 tool_ir_v1 replay
        ↓
Replay 也不执行 tool_ir_v1 program
        ↓
Evidence-domain predicate 没有真实 authority
        ↓
R1 对 locate/search Atomic 很难通过
```

以及：

```text
Benchmark won 已经被定义为最终任务成功
        ↓
但 Runtime 主循环仍用旧 TaskValidator.terminal().passed 控制是否继续
        ↓
TaskContract 未满足时仍可能继续节点
        ↓
然后撞 Harness terminal latch
```

因此当前 HEAD **只适合继续 deterministic/smoke 修复，不允许作为正式 v3.2 Full-30 实验提交**。

---

# 1. 修复范围总览

本轮修复合并为 12 个闭合修复块：

| 编号 | 修复块 | 优先级 |
|---|---|---|
| R1-1 | Benchmark Won 控制流 Authority 真正闭合 | P0 |
| R1-2 | Tool IR Admission + Replay 真正贯通 | P0 |
| R1-3 | Tool IR Search/Loop 执行语义闭合 | P0 |
| R1-4 | Evidence-domain 真实 Authority + ALFWorld Evidence Adapter | P0 |
| R1-5 | Runtime Automation 输入绑定 + R0/R1 结果语义 | P0 |
| R1-6 | Non-contiguous Evidence Causal Lineage + Witness Authority | P0 |
| R1-7 | Success Evolution / NO_TOOL / ToolBuilder Context 闭合 | P0 |
| R1-8 | Support Atomic 显式 DataFlow Mapping + Planner/Runtime Integration | P1 |
| R1-9 | Repairability Gate 对 Composite Hint 的真实使用 | P1 |
| R1-10 | ToolBuilder 共享预算 Authority | P1 |
| R1-11 | v3.2 Protocol Identity / Fresh Bank / Report 语义 | P0 |
| R1-12 | 新增针对真实断点的 deterministic gates 与 targeted smoke | P0 |
| R1-13 | Terminal-Certified Empirical Composite：Early-terminal 更短图学习、检索与生命周期 | P0 |

---

# 2. R1-1 — Benchmark Won 控制流 Authority 真正闭合

## 2.1 当前事实

当前已经有：

```python
trace.strict_task_success = trace.benchmark_success
```

并且 Harness：

```python
if self._done or self._won:
    raise AtomicSkillGraphError("harness_terminal_latched", ...)
```

说明“Benchmark won 是最终任务成功 authority”的数据层定义已经接入。

但 Runtime 主循环仍多次执行：

```python
terminal = self.validation.task.terminal(...)
if terminal.passed:
    break
```

而当前 `TaskValidator.terminal()` 仍要求：

```text
benchmark_won == true
AND
task_contract == true
```

才 `passed=True`。

这会造成：

```text
Benchmark:
won=true

TaskContract diagnostic:
false

Runtime:
terminal.passed=false
→ 继续下一节点
→ Harness 已经 terminal latch
→ 后续动作异常
```

这正是 task12 型问题仍然可能存在的原因。

---

## 2.2 冻结修法

必须彻底拆开：

```text
Task control authority
```

和：

```text
TaskContract diagnostic
```

### Runtime 控制流只看

```python
ctx.terminal_latched
```

或 Harness 明确的：

```python
benchmark_won
```

不能再用：

```python
TaskValidator.terminal().passed
```

决定：

- 是否进入下一 Graph node；
- 是否 Seeded；
- 是否 task rescue；
- 是否继续 Tool；
- 是否继续 Dynamic；
- 是否继续 ColdStart scaffold。

---

## 2.3 TaskValidator.terminal 的推荐兼容改法

为了避免旧调用者再次误用，建议直接修改语义：

```python
def terminal(..., benchmark_won: bool) -> ValidationResult:
    contract_result = self.validate(...)
    return ValidationResult(
        level="task_terminal",
        passed=bool(benchmark_won),
        checks={
            "benchmark_won": bool(benchmark_won),
            "task_contract": bool(contract_result.passed),
        },
        failure_codes=[] if benchmark_won else ["benchmark_failure"],
        ...
    )
```

如果：

```text
benchmark won=true
task_contract=false
```

记录：

```text
task_contract_success=false
metadata.anomaly=benchmark_goal_contract_mismatch
```

但：

```text
task success=true
terminal passed=true
```

不制造 task failure。

---

## 2.4 Orchestrator 必须逐处修改

重点：

```text
src/atomic_skillgraph/runtime/orchestrator.py
```

所有：

```python
if terminal.passed:
```

必须检查语义：

### 若用于任务终止

改成：

```python
if ctx.terminal_latched:
```

### 若只是记录 TaskContract diagnostic

保留：

```python
terminal.checks["task_contract"]
```

不能混用。

---

## 2.5 ToolRunner

Tool 内每个 ACTION 后：

```python
if result.won:
    terminal_interrupted = True
    stop all remaining Tool actions
```

已有方向正确。

需要保证外层 Orchestrator 接到该结果后：

```text
不再 Seeded
不再 next node
不再 task rescue
```

---

## 2.6 Credit

Benchmark won 不反向证明：

```text
Tool completed
Atomic effect passed
Composite full execution
```

早停节点继续使用：

```text
SKIPPED_GOAL_TERMINAL
```

并且：

```text
Composite full execution credit != task success
```

---

# 3. R1-2 — Tool IR Admission + Replay 真正贯通

这是当前 v3.2 最大的实际断点。

---

## 3.1 当前 Admission 不认新 Replay Kind

`ToolCompiler.compile_proposal()` 当前生成：

```yaml
tests:
  - kind: tool_proposal_replay
```

但：

```python
Admission.admit_tool()
```

只找：

```python
kind == "source_replay"
```

因此新 Tool IR 会直接：

```text
ToolStatus.SHADOW
admission_failure = source_replay_unavailable
```

即使 ToolBuilder 提案、Static Validation 都正确，也无法正式进入 Candidate。

---

## 3.2 当前 Admission closure 还是旧 Primitive IR

`Admission._tool_closure_failures()` 仍主要检查：

```python
tool.artifact["steps"]
```

而新 Tool 使用：

```python
tool.artifact["program"]
```

所以：

> tool_ir_v1 没有真正经过与自身结构匹配的 admission closure。

---

## 3.3 冻结修法：按 artifact_kind 分派

```python
def admit_tool(...):
    if tool.artifact_kind == "primitive_ir":
        return _admit_primitive_ir(...)
    if tool.artifact_kind == "tool_ir_v1":
        return _admit_tool_ir_v1(...)
```

不要继续在一个旧 `steps` closure 上叠 if patch。

---

## 3.4 tool_ir_v1 Admission 必须验证

1. `ToolValidator.validate_asset`；
2. `ToolStaticValidator` 对持久化 ToolAsset 的再验证；
3. Atomic contract compatibility；
4. output closure；
5. bounded program；
6. no concrete episode leakage；
7. allowed Harness primitive；
8. replay case 存在；
9. replay 真执行；
10. replay final Atomic Effect / evidence Effect 通过。

---

# 4. R1-2A — Replay 必须复用同一 ToolRunner

## 4.1 当前问题

当前：

```python
AlfWorldAdapter.replay_tool()
```

只执行：

```python
for raw in tool.artifact.get("steps", []):
    ...
```

新 Tool：

```text
artifact_kind = tool_ir_v1
artifact.program = [...]
```

所以新 replay 实际：

```text
reset
→ 不执行 program
→ 直接 validate final effect
```

这不是 Tool replay。

---

## 4.2 禁止修法

不要在 `AlfWorldAdapter.replay_tool()` 再复制一套：

```text
IF/FOR_EACH/STOP_WHEN/RETURN
```

解释器。

否则在线 Runtime 与 Admission replay 会产生两个 IR 语义实现。

---

## 4.3 正确修法

Tool IR 的唯一执行 authority 保持：

```text
ToolRunner
```

Replay 建议上移到 System / Admission service：

```text
System._replay_tool_candidate(...)
```

流程：

```text
Harness reset source task
        ↓
Replay validated prefix
        ↓
Build minimal replay RuntimeContext
        ↓
ToolRunner.run(candidate, bindings)
        ↓
Generic Atomic / Evidence validation
        ↓
Replay pass / fail
```

Harness 只负责：

```text
reset
execute_action
action_catalog
semantic evidence
```

不负责解释 Tool IR。

---

## 4.4 ToolProposal replay case 必须保存 prefix

当前 `tool_proposal_replay` 需要至少保存：

```yaml
source_task:
trace_id:
occurrence_id:
draft_id:
bindings:
prefix:
effects:
```

Success Evolution 可以使用：

```text
occurrence.prefix_events
```

中的 accepted prefix 先恢复 source preconditions。

第一版以 correctness 为主，不要求 prefix 最小化。

Runtime task-local trial 不需要 fresh replay；其本身就是 R1 evidence。

---

# 5. R1-3 — Tool IR Search/Loop 执行语义闭合

当前 IR opcode 存在，但还不能表达 v3.2 最重要的：

```text
遍历可访问位置
→ GO_TO
→ 如果可以 OPEN 则 OPEN
→ refresh catalog/evidence
→ 找到目标
→ RETURN object/location
```

---

# 6. R1-3A — Action Catalog Selector

## 6.1 当前缺口

现在：

```python
resolve_collection(source="action_catalog", field=...)
```

只能读 catalog entry 顶层：

```text
action_id
revision
action_type
arguments
```

不能：

```text
filter action_type == GO_TO
project arguments.destination
```

所以 Builder 没法写通用 location traversal。

---

## 6.2 新增通用 Selector，不新增 opcode

保留冻结的五个 opcode：

```text
ACTION
IF
FOR_EACH
STOP_WHEN
RETURN
```

只扩展：

```text
collection_source
condition source
argument source
```

表达能力。

建议统一 Selector Schema：

```yaml
source: action_catalog

where:
  action_type: GO_TO

project:
  kind: argument
  role: destination

distinct: true
```

---

## 6.3 Semantic selector

为支持：

```text
target = "cup"
catalog contains object = "cup_3"
```

允许：

```yaml
where:
  argument_role: object
  semantic_compatible_with:
    source: tool_input
    field: target
```

代码调用 Harness 已有的：

```python
semantic_value_compatible(...)
```

这是 generic Harness semantic compatibility，不是：

```python
if cup ...
```

---

## 6.4 ACTION 也需要动态 Catalog binding

例如：

```text
找到一个 TAKE action whose object compatible with target
```

Tool 应能把该 action 的：

```text
arguments.object
arguments.source
```

写入 local variable / outputs。

不允许 Builder手写具体 episode object ID。

---

# 7. R1-3B — max_actions 必须是运行时 authority

当前只 Static 验证：

```text
max_actions > 0
```

但 ToolRunner 没按它停止。

必须：

```python
ToolExecutionState.executed_action_count
ToolExecutionState.max_actions
```

每个 ACTION 前：

```python
if executed_action_count >= max_actions:
    stop with tool_ir_max_actions_exhausted
```

然后才：

```python
ctx.budget.consume_action()
```

两个 budget 都必须满足：

```text
Tool local max_actions
AND
Runtime node/global action budget
```

任何一个先耗尽都停止。

---

# 8. R1-3C — STOP_WHEN 与 RETURN 分离

当前：

```python
STOP_WHEN -> stop[0] = True
```

导致整个 program 停止，后面的 RETURN 无法执行。

必须区分控制信号：

```text
CONTINUE
BREAK_LOOP
RETURN_PROGRAM
BENCHMARK_TERMINAL
FAIL_TOOL
```

### STOP_WHEN 在 FOR_EACH 内

默认：

```text
BREAK_LOOP
```

然后允许顶层继续：

```text
RETURN
```

### RETURN

```text
RETURN_PROGRAM
```

### won

```text
BENCHMARK_TERMINAL
```

并立即停止所有 program。

---

# 9. R1-3D — IF / Condition 静态校验

Static Validator 必须验证 operator 枚举，与 Runtime interpreter 完全一致：

```text
exists
not_exists
equals
not_equals
contains
empty
non_empty
```

不能 Static pass 后 Runtime 才：

```text
tool_ir_condition_operator_unsupported
```

---

# 10. R1-3E — Local variable 作用域

当前 Static Validator 用一个全局：

```python
defined_locals
```

会允许潜在：

```text
在变量定义之前引用
```

必须改为按程序顺序/branch scope 验证。

规则：

```text
FOR_EACH iteration_variable
→ 只在其 body 内可用

branch local
→ 不能自动泄漏到另一 branch

RETURN
→ 只能读取所有实际路径上确定可定义的变量
```

第一版 fail closed，不需要复杂 SSA。

---

# 11. R1-3F — Concrete Episode Leakage 全 IR 扫描

不能只扫描 ACTION constant。

需要递归扫描：

```text
ACTION constant
IF condition.value
STOP_WHEN condition.value
FOR_EACH local_deterministic.values
RETURN constant
selector literal values
path_expectations
evidence_outputs
```

但允许：

```text
stable semantic literal
```

如：

```text
action_type = GO_TO
effect_domain = evidence
```

Concrete ID 检查仍使用 generic episode-instance pattern + source task forbidden terms。

---

# 12. R1-3G — Step-level Effect 真正执行验证

ToolBuilder 已能为 IR node 提交：

```yaml
expected_effects:
```

但 ToolRunner 当前没有逐 node 验证。

必须每个 ACTION 完成、Evidence state refresh 后：

```text
validate node.expected_effects
```

记录：

```yaml
program_node_id:
expected_effects:
observed_effects:
missing_effects:
step_effect_passed:
```

### 如果 required expected Effect 失败

停止 Tool：

```text
failure_layer = tool_effect
failure_code = tool_step_effect_violation
```

这才真正满足 v3.2：

> ToolBuilder 明确 effect，用于节点级错误定位。

---

# 13. R1-4 — Evidence-domain 必须有真实 Authority

## 13.1 当前问题

ALFWorld Schema 已声明：

```text
binding.discovered
entity.discovered_at
```

但当前 Validator facts 没有实际产生这些 predicate。

ToolRunner：

```python
binding_evidence=[]
```

也没有后续刷新。

因此：

```text
locate/search Atomic
```

即使完成了搜索，也很难得到：

```text
evidence-domain R1 witness
```

---

# 14. R1-4A — 先实现真正需要的 `entity.discovered_at`

ALFWorld action catalog 本身包含结构化 relation 时，可形成 evidence。

例如当前 catalog 存在：

```text
TAKE(object=X, source=Y)
```

则 Adapter 可以确定：

```text
entity.discovered_at(entity=X, location=Y)
```

这不是从 observation prose 猜测，而是：

> Harness 自己公开的 structured affordance relation。

Evidence 必须 revision-aware。

当 catalog revision 更新后：

- 当前仍存在的 relation有效；
- 不再存在的 relation按 Evidence Stability 规则失效或转 Historical Memory；
- 不能永远单调累积为 current fact。

---

# 15. R1-4B — `binding.discovered` 不可空声明

当前如果没有 generic Runtime Binding Authority 将它转成 Semantic Evidence：

> 先从 ALFWorld `semantic_predicate_schema()` 移除 `binding.discovered`。

或者实现真正的 generic authority：

```text
confirmed_binding
        ↓
binding.discovered(role,value)
```

其 `validation_source` 必须：

```text
runtime_binding_authority
```

不能声称来源是：

```text
alfworld_action_catalog
```

却没有生成逻辑。

第一版建议：

> 先用 `entity.discovered_at` 跑通 Runtime locate/search Tool；`binding.discovered` 等真正需要时再接。

---

# 16. R1-4C — ToolExecutionState Evidence 刷新

每次 Tool ACTION 后需要同步：

```text
state.semantic_facts
state.binding_evidence
state.catalog
```

建议 `TaskRuntimeContext` 提供统一 projection：

```python
ctx.tool_evidence_snapshot()
```

包含：

```yaml
semantic_facts:
binding_evidence:
action_catalog:
revision:
```

ToolRunner 不直接拼 Harness private state。

---

# 17. R1-4D — ALFWorld USE 不再机械 Toggle

当前：

```python
elif spec.action_type == "USE":
    if light already on:
        set off
    else:
        set on
```

仍然是 task33 已验证过会与真实环境分叉的 shadow transition。

冻结修法：

```text
ActionSpec
+
actual observation / metadata
        ↓
ALFWorld Semantic Evidence Adapter
        ↓
light.on / light.off
```

例如真实返回明确：

```text
turn on
```

Adapter 才建立：

```text
light.on
```

如果明确：

```text
turn off
```

才建立：

```text
light.off
```

若结果无法确定：

> 不制造确定的 light transition。

禁止通过 task_type 特判。

---

# 18. R1-4E — Harness Validator record API

建议从：

```python
record(spec, accepted, revision, done, won)
```

改为接收：

```python
record(
    spec,
    observation,
    metadata,
    accepted,
    revision,
    done,
    won,
    catalog,
)
```

或者定义：

```text
HarnessSemanticTransition
```

由 Adapter 构造。

目标：

> 节点级 Validator 根据真实环境返回建立 evidence，而不是只看 Action 名猜世界变化。

---

# 19. R1-4F — Primitive Action Schema 单一 Source of Truth

当前 `primitive_action_schema()` 与实际 parser 参数角色不完全一致。

不能维护：

```text
parser schema
ToolBuilder schema
ToolRunner schema
```

三份人工定义。

建议：

```text
parse_alfworld_action / canonical action signature table
          ↓
primitive_action_schema()
          ↓
ToolBuilder
          ↓
ToolStaticValidator
```

同一个 source。

必须至少确保实际 parser 能产生的 action type / argument roles 与 Builder schema一致。

不要为 look-at 临时单独补一个模板。

---

# 20. R1-5 — Runtime Automation 输入绑定闭合

## 20.1 当前问题

Runtime Automation Draft 只有：

```text
ParameterSpec inputs
```

没有表达：

> 本次 task-local trial 中，这些新 Automation input 从当前 blocked occurrence 的什么 authority 获得。

当前代码直接：

```python
ctx.binding_store.snapshot_for_node(blocked_occurrence)
```

取同名 binding。

例如：

```text
blocked Heat input role = object
Automation input role = target
```

即使语义完全正确：

```text
target <- current semantic anchor object
```

也不会自动存在。

---

## 20.2 新增 task-local input_binding_specs

`RuntimeAutomationAtomicDraft` 增加：

```yaml
input_binding_specs:
  target:
    kind: current_occurrence_anchor
    source_role: object
```

允许来源：

```text
current_occurrence_anchor
current_confirmed_binding
current_candidate_binding（不能自动 authority，需 Agent 明确选择）
data_flow
constant semantic literal（禁止 episode concrete ID）
```

不要复用长期 Atomic `BindingExpression` 直接写 task-local provenance，以免污染 canonical contract。

---

## 20.3 R0 验证 binding specs

R0 必须检查：

- source role存在；
- anchor/confirmed binding 当前存在；
- semantic types compatible；
- concrete requirement满足；
- 不跨 task；
- 不含未验证 Tool output；
- episode concrete ID只能存在于 task-local binding，不进入长期 Atomic contract。

---

## 20.4 Trial

Coordinator 使用：

```text
draft.input_binding_specs
```

得到 task-local：

```text
trial_bindings
```

而不是简单 role-name copy。

---

# 21. R1-5B — R1 不能把 terminal interruption 当 Tool 完整成功

当前 Runtime Automation R1：

```python
tool.completed OR tool.terminal_interrupted
```

会共同支持 `r1_passed`。

应拆成：

```yaml
r1:
  atomic_effect_passed:
  executed_path_effects_passed:
  tool_completed:
  terminal_interrupted:
  outputs_valid:
  admission_eligible:
```

### Task terminal interrupt

如果：

```text
task won=true
atomic effect passed=true
tool program没完整执行
```

则：

```text
Atomic local evidence = positive
Tool full completion = false
Tool intrinsic failure = false
admission_eligible = false
```

Success Extractor 可以利用实际 executed prefix 再生成更短 Tool。

不能把原 Tool 当完整成功。

---

# 22. R1-6 — Non-contiguous Evidence Causal Lineage 真正实现

## 22.1 当前问题

数据结构已有：

```text
support_event_ids
precondition_witness_refs
effect_witness_refs
ordering_constraints
```

但 Atomicizer 仍：

```python
selected_span_ids
if len(selected_span_ids) != 1:
    reject
```

这与冻结设计冲突。

---

## 22.2 正确 authority

允许：

```text
Runtime parent span
   ↓
Tool child span
   ↓
Runtime validation
```

共同证明一个 logical Atomic occurrence。

判断依据不是：

```text
span_id equality
```

而是：

```text
same occurrence causal lineage
```

---

## 22.3 Trace Span lineage

需要能够确定：

```text
span_id
parent_span_id
occurrence_id
```

对 support events：

1. 同一 occurrence；
2. 或 parent/child span 最终归属同一 occurrence；
3. event revision 顺序合法；

则允许。

不同 occurrence：

```text
reject
```

---

# 23. R1-6B — Witness refs 必须参与验证

当前不能只保存：

```text
precondition_witness_refs
effect_witness_refs
```

必须验证：

### Precondition refs

确实存在于：

```text
before-state authority
```

并与 proposal precondition predicate + binding 匹配。

### Effect refs

确实存在于：

```text
support event / after-state authority
```

并与 proposal Effect匹配。

Agent不能提交一个不存在的 ref 仅作为装饰。

---

# 24. R1-6C — ordering_constraints

需要第一版最小 Schema，例如：

```yaml
- before_event_id: event_3
  after_event_id: event_7
```

只允许 support events。

验证：

```text
event3 revision < event7 revision
```

不实现复杂 temporal logic。

---

# 25. R1-6D — Envelope overlap

当前 Atomic proposal 仍禁止 envelope overlap。

非连续 support 后，这个规则可能过严：

```text
Atomic A support = [3,7]
Atomic B support = [5]
```

它们的 envelope overlap，但 support events 没重叠。

v3.2 的真正 authority 应变成：

> **support_event_ids 不得被两个独立 Atomic 重复拥有，除非该事件被显式标记为 shared precondition evidence；envelope 可以 overlap。**

因此删除：

```text
used_ranges envelope overlap hard reject
```

改为：

```text
used_support_events
```

冲突检查。

---

# 26. R1-7 — Success Evolution / NO_TOOL 闭合

## 26.1 当前 NO_TOOL 会进入空 Tool/Implementation

当前：

```text
CompiledKnowledge(
    atomic=...,
    tool=None,
    implementation=None
)
```

本身方向可以接受。

但后续：

```python
rewrite_capability_labels(item,...)
```

直接：

```python
replace(compiled.tool,...)
replace(compiled.implementation,...)
```

会崩。

---

## 26.2 冻结修法

正式允许：

```python
CompiledKnowledge.tool: ToolAsset | None
CompiledKnowledge.implementation: ImplementationAtom | None
```

所有下游显式处理 Atomic-only。

---

## 26.3 rewrite_capability_labels

如果：

```text
tool is None
```

只 rewrite Atomic / occurrence。

如果：

```text
implementation is None
```

不调用 replace。

---

## 26.4 `_apply_evolution`

当前已有部分 Atomic-only 分支，继续保留。

必须保证：

```text
NO_TOOL Atomic
```

可以：

- align；
- register；
- evidence；
- lifecycle；

但不创建 fake Tool / fake Implementation。

---

# 27. R1-7B — ToolBuilder Context 去除 provenance leakage

当前 Builder context 投影：

```text
Atomic metadata
ToolProvenance
```

其中可能包含：

```text
task_id
trace_id
occurrence_id
task_local metadata
```

这些并不是 Tool 程序设计必需信息。

冻结 Builder context：

```text
canonical Atomic:
summary
inputs
outputs
preconditions
effects

source_kind:
success_evolution | runtime_automation

support events
structured delta
Harness interface
near-match interface
local relevant failures
```

不提供：

```text
task_id
trace_id
complete atomic metadata
source_task
```

Provenance 只保留 code-side，Compiler写入资产。

---

# 28. R1-7C — Runtime Builder 不再把 raw observation 当 semantic_delta

当前：

```python
semantic_delta={
    "observation": ctx.observation,
    "revision": ...
}
```

不符合冻结设计。

改成：

```yaml
semantic_delta:
  before/current_facts:
  current_evidence_facts:
  current_binding_evidence:
  revision:
```

raw observation：

> 只有在与当前 proposed Atomic 直接相关且 structured evidence 不足时，以独立 `support_raw_observation` 小字段提供。

默认不传整段 current observation。

---

# 29. R1-7D — Success Builder Harness interface

当前 Success Evolution 只给：

```text
primitive_action_types actually seen in support evidence
```

这会限制 Builder：

> 只能复制轨迹已有 action 类型，无法合理加入必要的 IF/conditional setup。

改成：

```text
完整但紧凑的 Harness primitive schemas
```

例如：

```yaml
- action_type:
  argument_roles:
```

不提供当前 episode full catalog。

---

# 30. R1-8 — Support Atomic 显式 DataFlow Mapping

## 30.1 当前问题

`SupportAtomicRetriever` 只匹配：

```python
producer_output.name == consumer_missing_role.name
```

这太窄。

例如：

```text
Locate output = entity
Heat input = object
```

语义类型都为 entity，但不会匹配。

---

## 30.2 新 SupportCandidate

建议：

```python
@dataclass(frozen=True)
class SupportRoleMapping:
    producer_role: str
    consumer_role: str
    semantic_type: str

@dataclass(frozen=True)
class SupportCandidate:
    atomic_ref: str
    role_mappings: tuple[SupportRoleMapping, ...]
    ...
```

---

## 30.3 Mapping Authority

基于：

```text
producer output semantic type
consumer missing input semantic type
required_resolution
evidence domain
```

匹配。

如果唯一映射：

```text
可以作为 candidate mapping
```

如果多个 mapping：

```text
Agent 必须在 invoke_support_atomic 中显式选择
```

代码验证。

不要因为名称近似自动选。

---

## 30.4 `invoke_support_atomic`

ToolCall 增加：

```yaml
output_mapping:
  producer_role: consumer_role
```

仅允许 formal SupportCandidate 暴露的 mapping。

---

## 30.5 publish

支持完成后：

```text
result.validated_outputs[producer_role]
        ↓ mapping
consumer input role
```

再写：

```text
binding_store
evidence_store
runtime_graph_augmentation
```

不能再靠 role 名相同。

---

# 31. R1-8B — Planner Support Atomic Integration

当前 Support 路径主要在 Runtime。

v3.2 冻结设计还要求：

> 辅助 Atomic 是普通 Graph node，Planner 在有意义时也能纳入。

但 P1仍 Bank-blind。

推荐接入点：

```text
P1 Requirement
↓
required Atomic retrieval
↓
P1R/P2
```

P2 在构图时得到：

```text
required Atomic candidates
support Atomic candidates
formal producer→consumer mapping candidates
```

Prompt：

> 只有 support output/evidence 实际被 downstream required occurrence 消费时才加入图。

---

# 32. R1-9 — Repairability Gate 必须真实使用 Composite Hint

当前 `related_composite_hints` 基本只是写进 diagnostics。

如果：

```text
Atomic direct effect near-match = none
related Composite = 有相关可修接口
```

当前仍可能判 hard gap。

修复：

```text
Composite hint
    ↓
extract component/interface/effect projection
    ↓
如果与 desired Effect family / required interface 有 meaningful overlap
        ↓
repairable = true
```

但禁止：

```text
任何 related Composite 只要存在
→ 一律 repairable
```

需要 deterministic relevance。

---

# 33. R1-10 — ToolBuilder 必须共享现有预算，而不是增加新容量

## 33.1 当前事实

ToolBuilder 已有：

```text
tool_builder_runtime
tool_builder_evolution
```

UsageBucket。

这是正确的。

但是 `_tool_builder_session()` 当前又创建：

```python
AgentBudget(max_tokens=300000)
```

与 Runtime Agent 独立。

同时 production Runtime 并没有把 ToolBuilder usage 通过：

```python
RuntimeBudget.consume_llm()
```

扣回原 300K。

因此：

> ToolBuilder 现在是“统计到了”，但不是“共享了现有 budget”。

---

## 33.2 Runtime ToolBuilder

冻结：

```text
runtime_preparation
runtime_seeded
runtime_dynamic
tool_builder_runtime
```

共同受：

```text
max_total_tokens_per_task = 300000
```

约束。

不能：

```text
Runtime已经用了250K
ToolBuilder再获得独立300K
```

---

## 33.3 Evolution ToolBuilder

Success Evolution ToolBuilder 属于成功学习阶段。

建议：

```text
extractor_e1
extractor_e2
tool_builder_evolution
```

共享：

```text
extractor.max_total_tokens_per_task = 262144
```

这样 v3.2 不凭 ToolBuilder 新增一个隐性学习预算池。

如果实现上需要独立 AgentSession，传入的 `AgentBudget.max_tokens` 应是：

```text
remaining shared allocation
```

而不是完整 cap。

---

# 34. R1-11 — v3.2 Protocol Identity 必须真正升级

当前正式代码仍硬要求：

```text
method_patch = 3.1
```

包括：

- `configs/default.yaml`
- `configs/alfworld_train_full_30.yaml`
- `configs/alfworld_frozen_eval.yaml`
- `run_v3_train.py`
- `run_v3_frozen_eval.py`
- System config loader
- Trace create metadata

这会导致新的 v3.2 实验仍被记录成 R3.1。

---

## 34.1 配置

正式 v3.2：

```yaml
schema_version: 3
method_patch: "3.2"
```

---

## 34.2 新 Full30 data_dir

禁止复用：

```text
runs/alfworld_train_full_30/data_v3
```

建议：

```text
runs/alfworld_train_full_30_v32/data_v3
```

或带 frozen commit：

```text
runs/v32_<shortsha>_full30/data_v3
```

正式要求只有一个：

> 必须 fresh empty，不复用 R3.1 bank。

---

## 34.3 Trace metadata

新 Trace：

```yaml
method_patch: "3.2"
code_commit: <frozen commit sha>
```

Formal runner 应 assert 当前 git commit 与 manifest 一致（若现有 protocol 已有类似机制则复用）。

---

## 34.4 ToolBuilder config 显式化

正式配置建议新增：

```yaml
llm:
  tool_builder:
    reasoning_effort: high
    max_completion_tokens: 32768
    request_timeout_seconds: 300
```

但其 shared total budget authority 仍按 R1-10。

不要依赖隐式默认。

---

# 35. R1-11B — Report 字段语义修正

当前：

```text
strict_task_success
```

已被设为：

```text
benchmark_success
```

但 Report 仍显示：

```text
Strict TaskContract success
```

会造成误导。

建议正式报告：

```text
Benchmark success
TaskContract agreement diagnostic
Graph full completion
Graph self-sufficient execution
```

其中：

```text
Benchmark success = trace.benchmark_success
TaskContract agreement = trace.task_contract_success
```

如果为了兼容保留 `strict_task_success` 字段：

> 只能注明 deprecated alias of benchmark_success。

不能继续叫 Strict TaskContract。

---

# 36. R1-11C — R0 Metrics 必须按真实 R0 结果计

当前 NodeExecutor 在 Draft schema parse 后就增加：

```text
runtime_automation_r0_pass_count
```

但 Coordinator 后续才执行真正：

```text
validate_automation_draft
```

修复：

### Agent proposal accepted

```text
runtime_automation_atomic_proposal_count += 1
```

### Coordinator R0 pass

才：

```text
runtime_automation_r0_pass_count += 1
```

### R0 reject

```text
runtime_automation_r0_reject_count += 1
```

---

# 37. R1-13 — Terminal-Certified Empirical Composite

该边界已经确认，以下方案正式冻结。

---

## 37.1 已冻结事实

用户已明确：

```text
Benchmark won=true
→ task严格成功
→ 不继续后续节点
```

如果原图：

```text
A -> B -> C
```

执行到：

```text
A -> B -> won
```

则：

```text
C 是本次任务不需要的尾部节点
```

Success Extractor 应学习：

```text
A -> B
```

这种更短实际成功路径，并通过 Lifecycle 逐步淘汰冗余图。

---

## 37.2 当前代码矛盾

Success Evolution 仍：

```python
coverage = contract_coverage_report(
    TaskContract,
    staged_occurrences,
)
```

只有：

```text
完整覆盖原 TaskContract
```

才进入 E2 Composite。

task12 型情况：

```text
Benchmark CLEAN 后 won
TaskContract 仍有 CLEAN + PLACE
```

实际成功 Trace：

```text
只有 CLEAN witness
没有 PLACE witness
```

因此：

```text
Extractor短图
→ TaskContract coverage incomplete
→ 不生成 Composite
```

与用户要求冲突。

---

## 37.3 不能采用的错误修法

### 方案 X：把没发生的 PLACE Effect 补到新 Atomic

禁止。

这是伪造 evidence。

### 方案 Y：因为单个 episode early won 就改写整个 TaskContract

不建议。

一个 episode 的 Benchmark terminal 不足以证明所有同类任务都不需要该 Effect。

### 方案 Z：直接把 incomplete Composite 当 complete P0 Composite

不建议。

这会让“没覆盖 formal goal”的单次偶然轨迹立刻覆盖 Planner authority。

---

## 37.4 冻结方案：Terminal-Certified Empirical Composite

建议新增一个**通用、非 benchmark-family 特判**的状态：

```text
terminal_empirical_candidate
```

它仍是普通 Composite Asset，不新增 Graph node type。

其含义：

> 这张图没有声称完整覆盖当前 TaskContract；但它在一个真实 benchmark-won episode 中作为实际执行因果前缀触发了终局成功。

记录：

```yaml
terminal_certificate:
  benchmark: alfworld
  source_trace_id:
  terminal_revision:
  executed_occurrences:
  skipped_planned_occurrences:
  task_contract_coverage:
  benchmark_won: true
```

---

## 37.5 生命周期

第一次：

```text
terminal empirical evidence
→ Composite CANDIDATE
```

不能直接 ACTIVE。

未来独立 task 中，如果该 Candidate 被探索执行并再次：

```text
benchmark won=true
```

积累：

```text
independent terminal success
```

达到现有 Composite active threshold 后：

```text
ACTIVE
```

再作为稳定 replacement：

```text
old longer Composite
→ suppressed
→ retired
```

---

## 37.6 Retrieval

需要增加一个受 Lifecycle / CandidateUsePolicy 管理的：

```text
terminal-certified candidate retrieval
```

但它不能伪装成：

```text
full TaskContract coverage
```

P0 audit 必须区分：

```text
complete_contract_composite
terminal_empirical_candidate
```

候选执行仍由 CandidateUsePolicy 控制探索比例。

---

## 37.7 为什么推荐这个方案

它同时满足：

1. Benchmark won 是 task success authority；
2. 不伪造 TaskContract effect；
3. 能真正学习 early-terminal 短图；
4. 单次异常不会立即替换旧稳定图；
5. 需要独立 task evidence；
6. 能通过现有 Lifecycle 最终 retire 冗余图；
7. 不需要 task-family hardcode。


## 37.7A 数据模型与 provenance

建议 Composite metadata 增加：

```yaml
completion_authority:
  kind: complete_contract | terminal_empirical
terminal_certificate:
  benchmark_won: true
  source_trace_id:
  terminal_revision:
  executed_occurrence_ids: [...]
  skipped_planned_occurrence_ids: [...]
  observed_task_contract_coverage:
    covered_effects: [...]
    uncovered_effects: [...]
```

约束：

- `kind=terminal_empirical` 时，必须有真实 `benchmark_won=true` Trace；
- `skipped_planned_occurrence_ids` 只能来自 terminal latch 后未执行节点；
- `observed_task_contract_coverage` 只记录诊断，不得补写未见 Effect；
- provenance 中保留 source Composite ref（若来源为旧 Stored Composite 的 early-terminal 执行），便于 replacement 关系建立。

## 37.7B Planner Retrieval

`CompositeRetriever` 需要将候选分为两个明确通道：

```text
complete_contract_candidates
terminal_empirical_candidates
```

优先级冻结为：

```text
active/preferred complete-contract Composite
    >
eligible terminal-empirical Candidate exploration
    >
Atomic composition / fallback
```

不能让单次 terminal empirical Candidate 抢占稳定完整 Composite。

当没有可用 complete-contract Composite，或 CandidateUsePolicy 允许探索时，Planner 可以尝试 terminal empirical Candidate。

Audit 必须记录：

```yaml
selected_composite_authority:
  kind: complete_contract | terminal_empirical
  candidate_status:
  source_ref:
```

## 37.7C Runtime 与验证

执行 terminal empirical Candidate 时：

- Runtime 不假装其完整覆盖 TaskContract；
- 正常按其实际节点执行；
- Benchmark won 仍是 task success authority；
- 若再次 early won，则记一次独立 terminal success evidence；
- 若未 won，则记录该 Candidate 的真实失败，不自动 Dynamic 补完后再给 Candidate success credit；
- 若 task rescue 发生，Candidate 不能获得 self-sufficient terminal success。

## 37.7D Lifecycle evidence

建议新增或复用 Composite evidence outcome：

```text
TERMINAL_EMPIRICAL_SUCCESS
TERMINAL_EMPIRICAL_FAILURE
```

或者在现有 EvidenceEvent metadata 中显式：

```yaml
completion_authority: terminal_empirical
benchmark_won: true|false
task_rescue_required: false
```

生命周期阈值不新增、不提高；使用现有：

```text
composite_active_self_sufficient_successes = 2
```

但只有：

```text
benchmark_won=true
AND task_rescue_required=false
AND candidate实际被执行
```

的独立 terminal-empirical success 才计入晋升支持。

## 37.7E Replacement

如果：

```text
old = A -> B -> C
new = A -> B
```

且 new 达到 Active：

- old 与 new 的任务适用 contract / semantic context 兼容；
- new 在独立任务上达到稳定 terminal success；
- old 的尾部 C 持续被 terminal 跳过；

才建立 replacement/derived relation，并沿现有 maintenance 进入：

```text
old ACTIVE
→ SUPPRESSED
→ RETIRED
```

禁止一次 early terminal 就立即 retire old。

---

## 37.8 最终冻结不变量

`Terminal-Certified Empirical Composite` 已确认采用，必须遵守：

1. **不得伪造缺失的 TaskContract Effect。**
2. **不得因为单个 early-terminal episode 就把候选当成完整 P0 Composite。**
3. **第一次真实 benchmark-win 前缀只产生 Candidate evidence。**
4. **后续 retrieval 必须受 CandidateUsePolicy 与现有 lifecycle 控制。**
5. **只有独立 task 的 benchmark-win evidence 达到现有 Composite 生命周期阈值后，候选才可转 Active。**
6. **Active 后若稳定替代更长旧 Composite，继续使用现有 stable replacement → suppressed → retired 路径。**
7. **它仍然是普通 Composite Asset，不新增特殊 Graph node 类型。**
8. **Planner / Report / Trace 必须显式区分 `complete_contract_composite` 与 `terminal_empirical_candidate`，不能混淆 coverage 语义。**

---

# 38. R1-12 — Tool IR Replay / Trial 的统一 Path Evidence

无论：

```text
Success Evolution replay
```

还是：

```text
Runtime task-local trial
```

最终都必须产出同结构：

```yaml
tool_path_evidence:
  program_path_id:
  executed_node_ids:
  validated_paths:
  unvalidated_paths:
  loop_iteration_counts:
  stop_condition_witnesses:
  step_effect_results:
  final_effect_result:
  outputs:
  evidence_refs:
  terminal_interrupted:
```

这样 Lifecycle 不需要区分：

```text
runtime-created Tool
success-extracted Tool
```

的证据格式。

---

# 39. Failure Localization

新增/保证以下 failure code：

```text
tool_ir_max_actions_exhausted
tool_ir_selector_invalid
tool_ir_selector_no_match
tool_ir_condition_operator_unsupported
tool_ir_local_scope_invalid
tool_step_effect_violation
tool_ir_return_output_unresolved
tool_ir_replay_failed
runtime_automation_input_binding_invalid
runtime_automation_r0_rejected
runtime_automation_r1_rejected
support_atomic_output_mapping_invalid
noncontiguous_evidence_lineage_invalid
evidence_witness_ref_invalid
```

Failure layer：

| code | layer |
|---|---|
| Tool IR schema/static | tool |
| Tool step effect | tool |
| Runtime automation input binding | runtime_binding |
| Atomic final effect | atomic |
| Support mapping | runtime_binding / composite depending stage |
| Benchmark terminal | not failure |

---

# 40. 文件级改动清单

## 必改

### `validation/task_validator.py`
- terminal passed语义改为 benchmark won；
- TaskContract只 diagnostic。

### `runtime/orchestrator.py`
- 所有 task termination control 改看 terminal latch；
- no post-won Seeded/rescue/node。

### `runtime/task_context.py`
- 统一 Tool evidence snapshot；
- terminal metadata。

### `runtime/tool_runner.py`
- local max_actions；
- selector；
- BREAK_LOOP/RETURN/TERMINAL control；
- binding evidence refresh；
- step-level effect validation。

### `tooling/ir.py`
- selector schema/resolve；
- condition operator共享枚举；
- scope-aware sources。

### `tooling/validator.py`
- selector validation；
- recursive concrete leakage；
- scope；
- operator；
- ToolAsset IR revalidation helper。

### `evolution/admission.py`
- artifact_kind dispatch；
- tool_ir_v1 admission；
- replay kind。

### `evolution/tool_compiler.py`
- proposal replay prefix；
- IR output mapping cleanup；
- Optional Tool/Implementation typing。

### `system.py`
- unified ToolRunner replay；
- NO_TOOL downstream；
- ToolBuilder context；
- shared budgets；
- v3.2 metrics；
- OPEN-1 实现若确认。

### `harness/alfworld.py`
- actual-result-driven USE semantic evidence；
- evidence-domain facts；
- primitive schema single authority；
- replay不再解释 Tool IR。

### `harness/protocol.py`
- structured semantic evidence interface；
- predicate schema authority。

### `runtime/automation.py`
- input_binding_specs；
- real R0/R1 metrics；
- structured semantic delta；
- R1 separated result semantics。

### `runtime/support_retriever.py`
- explicit role mapping。

### `runtime/node_executor.py`
- support mapping Tool schema；
- metrics；
- no stale support candidates after world changes（如有必要 refresh）。

### `evolution/atomicizer.py`
- causal lineage；
- witness refs；
- ordering；
- support-event overlap authority。

### `agents/context_builder.py`
- E1 search/support wording；
- ToolBuilder context去 provenance；
- Runtime automation prompt。

### `agents/structured_submission.py`
- runtime input_binding_specs；
- selector；
- witness/order schema。

### `experiments/report.py`
- Benchmark/TaskContract label；
- corrected metrics。

### `experiments/run_v3_train.py`
### `experiments/run_v3_frozen_eval.py`
### `configs/*.yaml`
### `traces/schema.py`
- method_patch升级3.2；
- fresh protocol identity。

---

# 41. E1 Prompt 的一个直接冲突也必须修

当前旧 Prompt 仍写：

```text
Do not extract:
- search or exploration detours
```

v3.2 又允许：

```text
Runtime-created evidence-domain locate/search support Atomic
```

需要改成：

```text
Do not extract incidental search/exploration detours that have no reusable,
validated evidence-domain Effect.

A bounded Runtime-created automation that has passed R1 and produces an
authoritative reusable evidence-domain Effect is not an incidental detour;
review it as an ordinary Atomic candidate.
```

这样：

- 随机漫游仍不学；
- 已验证 Runtime Tool 可以学。

---

# 42. Formal Config 不得修改的参数

继续保持：

```yaml
llm:
  runtime:
    reasoning_effort: low
    max_completion_tokens: 32768
    max_total_tokens_per_node: 100000
    max_total_tokens_per_task: 300000

  extractor:
    reasoning_effort: high
    max_completion_tokens: 131072
    max_turns: 2
    max_total_tokens_per_task: 262144

runtime:
  global_action_budget: 100
  node_action_budget: 35
```

不提高：

- Planner top-k；
- Planner repair limit；
- node/global actions；
- Runtime token cap；
- Extractor cap；
- Lifecycle threshold；
- ColdStart caps；
- FailureExtractor cap。

---

# 43. 新 Deterministic Gates

## Gate 1 — won-only terminal control

构造：

```text
benchmark won=true
task_contract=false
```

必须：

```text
task success=true
no next node
no Seeded
no rescue
no env.step
```

---

## Gate 2 — task12-style graph

```text
Clean node -> benchmark won
Place node remains
```

必须：

```text
Place status = SKIPPED_GOAL_TERMINAL
no Harness terminal error
```

---

## Gate 3 — tool_ir_v1 Admission

生成一个真实 IR Tool：

```text
Static pass
Replay executes actual program
Effect pass
→ Candidate
```

不能：

```text
source_replay_unavailable
```

---

## Gate 4 — Replay actually executes

Replay前 effect false。

Tool ACTION 后 effect true。

必须证明：

```text
executed_action_count > 0
```

否则 replay不能 pass。

---

## Gate 5 — NO_TOOL

Builder返回：

```text
decision=no_tool
```

必须：

```text
Atomic admitted
Tool none
Implementation none
no exception
```

---

## Gate 6 — max_actions

嵌套：

```text
FOR_EACH 20
```

但：

```text
tool.max_actions=5
```

实际：

```text
environment actions <= 5
```

---

## Gate 7 — STOP_WHEN then RETURN

找到目标：

```text
STOP_WHEN true
```

必须 break loop，然后：

```text
RETURN object/location
```

成功。

---

## Gate 8 — Catalog selector

Catalog：

```text
GO_TO(destination=a)
GO_TO(destination=b)
TAKE(object=x,source=b)
```

selector：

```text
GO_TO -> destination
```

返回：

```text
[a,b]
```

不依赖 benchmark name。

---

## Gate 9 — Semantic selector

target semantic anchor：

```text
cup
```

catalog concrete：

```text
cup_3
```

Harness compatible：

```text
true
```

selector可匹配。

---

## Gate 10 — Evidence-domain

Catalog relation：

```text
TAKE(object=cup_3, source=countertop_2)
```

必须生成可验证：

```text
entity.discovered_at(cup_3,countertop_2)
```

---

## Gate 11 — USE actual-result semantic state

连续两次真实结果都：

```text
turn on
```

Validator：

```text
light.on remains true
```

不能机械 toggle off。

---

## Gate 12 — Runtime automation input mapping

Blocked role：

```text
object semantic anchor=cup
```

Automation input：

```text
target <- current_occurrence_anchor.object
```

R0 pass，trial得到 target=cup。

---

## Gate 13 — R1 terminal interrupted

Task won中断 Tool。

必须：

```text
task success=true
tool_completed=false
intrinsic_failure=false
admission_eligible=false
```

---

## Gate 14 — Cross nested span evidence

parent RuntimeSpan event + child ToolSpan event：

同 occurrence lineage：

```text
pass
```

不同 occurrence：

```text
reject
```

---

## Gate 15 — Witness refs

不存在的 effect witness ref：

```text
reject
```

存在且 predicate/binding匹配：

```text
pass
```

---

## Gate 16 — Envelope overlap / support disjoint

两个 Atomic envelope overlap，但 support events不重叠：

```text
allowed
```

support event被两个独立 Atomic同时拥有：

```text
reject
```

---

## Gate 17 — Step-level effect

IR node action accepted，但 node expected Effect没发生：

```text
tool_step_effect_violation
program_node_id correct
```

---

## Gate 18 — Support role mapping

Producer：

```text
output entity
```

Consumer：

```text
missing object
```

semantic type compatible。

Agent选择映射：

```text
entity -> object
```

代码验证后 publish。

---

## Gate 19 — Repairability Composite hint

无 direct Atomic near-match，但 related Composite 中存在 relevant interface/effect：

```text
repairable=true
```

无相关内容：

```text
hard_gap
```

---

## Gate 20 — ToolBuilder shared budget

Runtime 已用：

```text
290K / 300K
```

ToolBuilder 最多只有：

```text
10K remaining
```

不能再获得300K。

---

## Gate 21 — No provenance leakage

ToolBuilder request中：

```text
task_id absent
trace_id absent
full Atomic metadata absent
```

---

## Gate 22 — v3.2 protocol identity

Formal train config：

```text
method_patch=3.2
fresh data_dir
```

Trace：

```text
method_patch=3.2
```

---

# 43A. R1-13 专项 Deterministic Gates

## Gate 23 — early-terminal Candidate creation

原图：

```text
A -> B -> C
```

执行：

```text
A -> B -> benchmark won
```

必须允许生成：

```text
terminal_empirical Candidate = A -> B
```

并记录真实 uncovered TaskContract effect；不得伪造 C。

## Gate 24 — terminal empirical 不等于 complete contract

Candidate metadata：

```text
completion_authority=terminal_empirical
```

P0 audit 不能把它计为：

```text
complete_contract_composite
```

## Gate 25 — stable complete Composite 优先

当同时存在：

```text
ACTIVE complete-contract Composite
CANDIDATE terminal-empirical Composite
```

默认命中稳定 complete-contract Composite。

只有 CandidateUsePolicy 允许探索时才可选择 empirical Candidate。

## Gate 26 — independent evidence promotion

同一 empirical Candidate：

- source task success 1 次；
- 独立 task 再 self-sufficient benchmark win 1 次；

达到现有 lifecycle 阈值后：

```text
Candidate -> Active
```

重复同一 task/trace 不得伪造独立支持。

## Gate 27 — failure does not receive success credit

Empirical Candidate 被探索执行但：

```text
benchmark won=false
```

即使后续 Dynamic rescue 成功：

```text
Candidate terminal success credit = 0
```

## Gate 28 — replacement only after stable evidence

New shorter Composite 未 Active：

```text
old Composite不得 suppress/retire
```

New 达到 Active 且 replacement 条件满足后，才进入现有 replacement 生命周期。

---

# 44. 新 Targeted Smoke

## Smoke A — Tool IR Admission

真实成功 Trace：

```text
Extractor
→ Atomic
→ ToolBuilder
→ tool_ir_v1
→ fresh replay
→ Candidate Tool
```

检查：

```text
Tool count增长
Implementation count增长
Tool不是SHADOW
```

---

## Smoke B — Runtime Locate Tool

场景：

```text
当前 Atomic 缺 object
```

Runtime Agent 在搜索前提出：

```text
locate target
```

执行：

```text
R0
→ ToolBuilder
→ FOR_EACH catalog destinations
→ zero-LLM GO_TO/OPEN
→ entity.discovered_at
→ STOP_WHEN
→ RETURN object/location
→ R1
→ Runtime resumes
```

检查：

```text
internal LLM calls = 0
```

---

## Smoke C — Support Reuse

知识库已有：

```text
Locate(entity)->entity/location
```

旧 Composite：

```text
Heat -> Place
```

Heat missing object：

```text
Support retrieval
→ explicit role mapping
→ invoke locate
→ DataFlow
→ Heat
```

---

## Smoke D — Early terminal

任务：

```text
A -> B -> C
```

B action产生：

```text
won=true
```

检查：

```text
C零执行
no LLM after won
no env.step after won
Task success
Composite full execution=false
```

同时检查：

```text
terminal empirical Composite candidate created
completion_authority=terminal_empirical
uncovered TaskContract effects仅记录诊断
不伪造Effect
```

---

# 45. Full-30 前 Gate

正式 Full-30 前必须全部满足：

```text
[ ] HEAD重新冻结
[ ] Git diff仅包含本轮确认修复
[ ] deterministic suite全过
[ ] v3.2 tests全过
[ ] Tool IR Admission真实pass
[ ] Tool IR replay真实执行program
[ ] NO_TOOL pass
[ ] max_actions runtime authority pass
[ ] STOP_WHEN→RETURN pass
[ ] catalog selector pass
[ ] semantic selector pass
[ ] evidence-domain R1 pass
[ ] USE actual-result semantic test pass
[ ] runtime input_binding_specs pass
[ ] cross-span causal lineage pass
[ ] witness refs pass
[ ] step-effect failure localization pass
[ ] support explicit role mapping pass
[ ] Repairability Composite hint pass
[ ] ToolBuilder shared budget pass
[ ] task12-style won-only terminal pass
[ ] terminal empirical Composite gates全过
[ ] method_patch=3.2
[ ] fresh empty data_dir
[ ] no R3.1 resume
[ ] formal parameters unchanged
[ ] no ALFWorld task workflow hardcode
```

---

# 46. 修复顺序

建议严格按下面顺序，避免后面的测试建立在错误底层上：

### Phase 1 — Task correctness
1. won terminal authority；
2. task validator diagnostic separation；
3. terminal no-follow-up gates。

### Phase 2 — Tool runtime core
4. max_actions；
5. control signal；
6. selector；
7. evidence refresh；
8. step-level effect。

### Phase 3 — Evidence boundary
9. ALFWorld actual-result semantic adapter；
10. entity.discovered_at；
11. primitive schema single authority。

### Phase 4 — Runtime Automation
12. input_binding_specs；
13. R0/R1 structured result；
14. ToolBuilder structured context。

### Phase 5 — Admission / Replay
15. tool_ir_v1 admission dispatch；
16. unified ToolRunner replay；
17. replay prefix；
18. NO_TOOL.

### Phase 6 — Extraction
19. causal lineage；
20. witness refs；
21. support overlap；
22. E1 support/search wording。

### Phase 7 — Support / Planner
23. role mapping；
24. P2 support candidate；
25. Composite hint repairability。

### Phase 8 — Budget / Protocol
26. shared ToolBuilder budget；
27. method_patch 3.2；
28. fresh bank；
29. report semantics。

### Phase 9 — Terminal Empirical Composite
30. 实现 terminal empirical Composite Candidate；
31. 接入 CandidateUsePolicy retrieval；
32. 接入独立 benchmark-win evidence；
33. 接入 stable replacement / suppression / retirement；
34. 跑专项 Gate。

---

# 47. 明确禁止的“修复”

本轮仍禁止：

```text
raise Runtime token budget
raise node/global action budget
raise Planner top-k
raise repair limits
hardcode cup search
hardcode microwave/sink/lamp
hardcode six ALFWorld workflow
predefine SearchToolForHeat
让 benchmark won 反向证明 Atomic effect
让 terminal interruption算 Tool full success
因为 ToolBuilder失败就回旧 mechanical ToolCompiler
把 Tool IR replay写成第二套解释器
为了让 Evidence pass直接信LLM output
```

---

# 48. 修复完成后的预期链路

## Success Evolution

```text
Benchmark won Trace
        ↓
Extractor non-contiguous Atomic
        ↓
Canonical Atomic
        ↓
exact executable reuse?
   ┌────┴─────┐
  yes         no
   │           ↓
 reuse      ToolBuilder
              ↓
         ToolProposal
              ↓
        Static Validation
              ↓
          Compiler
              ↓
      Unified ToolRunner replay
              ↓
         Admission
              ↓
 Atomic / Tool / Implementation
              ↓
        Composite E2
              ↓
        Lifecycle
```

---

## Runtime Automation

```text
Blocked Atomic / expensive grounding
        ↓
Runtime Agent predicts repetitive work
        ↓
Automation Atomic Draft
        ↓
task-local input binding map
        ↓
R0
        ↓
ToolBuilder
        ↓
Static Validation
        ↓
Task-local trial
        ↓
zero-LLM selector / loop
        ↓
real evidence-domain facts
        ↓
R1
        ↓
output bindings
        ↓
blocked Atomic resumes
```

---

## Support Reuse

```text
missing consumer role
        ↓
Support Atomic retrieval
        ↓
producer output ↔ consumer input mapping
        ↓
Agent selects support Atomic/mapping
        ↓
normal Atomic invocation
        ↓
validated output
        ↓
Runtime graph augmentation
        ↓
consumer resumes
```

---

# 49. 本轮修复成功标准

不是要求 Full-30 成功率必须立刻高于 96.67%。

第一层必须证明：

1. `won` 后没有任何多余 LLM/action；
2. ToolBuilder Tool 能真正 Admission；
3. Replay 真正运行 Tool IR；
4. Runtime 能真正创建并执行一个 zero-LLM 搜索/参数解析 Tool；
5. Evidence-domain effect 有 deterministic witness；
6. non-contiguous evidence 真正跨 nested spans；
7. Support Atomic 真正跨 role 名复用；
8. ToolBuilder 没有隐性新增 budget；
9. 新实验 protocol 与 R3.1 完全隔离。

第二层再看：

```text
Benchmark success
Runtime tokens
Preparation tokens
ToolBuilder creation cost
LLM-bypassed actions
Support reuse
Tool reuse
Graph self-sufficient execution
```

---

# 50. 最终冻结结论

当前 `23cccd1...` 已经证明：

> v3.2 的结构方向被代码接入了。

但还不能证明：

> v3.2 的核心方法链已经真实可运行。

当前最重要的问题不是继续加 Agent Prompt，而是把：

```text
ToolBuilder Proposal
→ Tool IR
→ Runtime Execution
→ Evidence
→ R1
→ Replay
→ Admission
→ Registry
```

这条链真正闭合。

同时必须把：

```text
Benchmark won
```

从“最终 report 里的 task success”提升为：

> **整个 Runtime 控制流的唯一终局 authority。**

本轮修复不得重新引入 benchmark workflow，不得通过提高预算掩盖 IR/Search/Admission 问题，也不得继续让旧 mechanical ToolCompiler 成为正式 v3.2 fallback。

本文全部内容均已冻结，可直接作为 v3.2-R1 实现要求。

---

# 51. 最终冻结状态

所有架构边界均已确认。

此前的 OPEN-1 已正式冻结为：

```text
R1-13 Terminal-Certified Empirical Composite
```

最终实现不得再把它视为可选功能。

v3.2-R1 完成后，只有在本文全部 deterministic gates、targeted smoke、formal protocol identity 与 fresh-bank 条件满足后，才允许冻结新 commit 并启动新的正式 Full-30。

任何实现中若发现：

- 需要新增 benchmark workflow hardcode；
- 需要提高预算才能让链路跑通；
- 需要重新定义 ToolBuilder/Runtime/Validator authority；
- 需要让 Benchmark won 反向证明未执行的 Atomic/Tool/Composite；
- 需要跳过 Admission/Replay 才能保存 Tool；

都视为实现偏离，必须停止并重新审查，而不是继续打局部补丁。
