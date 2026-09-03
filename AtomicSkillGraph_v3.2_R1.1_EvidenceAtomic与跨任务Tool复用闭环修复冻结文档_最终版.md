# AtomicSkillGraph v3.2-R1.1：Evidence Atomic Fresh-Output、Tool RETURN 与跨任务复用闭环修复冻结文档

> 状态：**完全冻结，可直接进入实现**  
> 审查基线提交：`1ae993cf90d94fa23758267b8a87f8f5994501ad` — `Scope shared ToolBuilder budget to current task`  
> 上一冻结文档：`AtomicSkillGraph_v3.2_R1_核心链路闭合与正式实验前修复冻结文档_最终版.md`  
> 目标：只修复当前 v3.2-R1 实现中仍阻断正式 Full-30 的最后一组旧模型假设；不扩大方法面、不提高预算、不改 top-k、不重开 ColdStart / FailureExtractor / AtomicComposition。  
> 当前结论：**允许继续 deterministic / targeted smoke；在本文 P0 Gate 通过前，不允许将新的 Fresh Empty-Bank Full-30 视为正式 v3.2 实验。**

---

# 0. 本轮为什么还需要 R1.1

当前 HEAD 已经闭合了上一轮的大部分问题：

- Benchmark `won` 已成为 task-level success authority；
- Tool IR 已有 `ACTION / IF / FOR_EACH / STOP_WHEN / RETURN`；
- `max_actions` 已进入运行时 authority；
- `STOP_WHEN` 已能 break loop 后继续 `RETURN`；
- Action Catalog selector 已支持结构化 filter / projection；
- `entity.discovered_at` 已由当前 ALFWorld Action Catalog 建立 evidence-domain fact；
- `USE` 已根据真实 observation 更新 light state，不再机械 toggle；
- Runtime Automation 已有 R0 → ToolBuilder → task-local trial → R1；
- Tool IR Admission 已区分 `primitive_ir` / `tool_ir_v1`；
- Tool IR Replay 已改由同一 `ToolRunner` 执行；
- Support Atomic 已支持 producer output → consumer input 显式 role mapping；
- ToolBuilder token 已按当前 task 剩余预算分配；
- `method_patch=3.2` 和独立 Full-30 data_dir 已配置；
- Terminal-Certified Empirical Composite 已有 Candidate 创建和独立 retrieval channel。

但是，v3.2 最核心的新能力：

```text
输入一个语义目标
        ↓
自动搜索 / 参数解析 Tool
        ↓
发现新的 concrete entity/location
        ↓
把这些 fresh outputs 交给下游 Atomic
        ↓
任务成功后固化成可跨任务复用 Atomic + Tool
```

仍被 v3 旧的 **“Atomic output 必须等于某个 input identity”** 假设阻断。

典型目标能力：

```text
Atomic: locate_target

input:
  target = "cup"        # semantic target，调用前已知

output:
  entity = cup_3        # 执行后新发现
  location = countertop_2

effect:
  entity.discovered_at(
      entity=$entity,
      location=$location
  )
```

当前系统仍有三层旧假设：

```text
Atomicizer:
output value 必须已经属于 input values

System._canonical_atomic_for_occurrence():
output 找不到等值 input → return None

AtomicValidator.validate():
effect validation 只使用 input binding，
fresh Tool outputs 没进入 Effect binding authority
```

因此：

> Runtime task-local Tool 即使搜到了对象，也不能可靠地被 Success Evolution 固化为下一任务真正可复用的 `locate_target` Atomic。

这不是小的 report 问题，而是当前正式 Full-30 仍需暂停的主要原因。

---

# 1. R1.1 修复块总览

本轮合并为 8 个修复块：

| 编号 | 修复块 | 优先级 |
|---|---|---|
| R1.1-1 | Atomic Boundary Provenance：Input Authority + Fresh Output Derivation | P0 |
| R1.1-2 | Atomic Validator：Effect 可以引用 validated fresh outputs | P0 |
| R1.1-3 | Tool IR `RETURN` 成为 Tool Output Authority，移除 self-output mapping 冲突 | P0 |
| R1.1-4 | Tool IR Recursive Safety / Action Collection / Return Closure | P0 |
| R1.1-5 | Terminal-Empirical Planner Validator：blocking vs diagnostic checks | P0 |
| R1.1-6 | Terminal-Empirical 当前 TaskContract 兼容性 Gate + Lifecycle Gates 26–28 | P1 / Full-30 前必须 |
| R1.1-7 | Runtime Self-Tool → Success Extractor → 下一独立任务复用的端到端 Gate | P0 |
| R1.1-8 | ALFWorld 精确版本冻结与实验 provenance | 实验协议 P0 |

---

# 2. R1.1-1 — Atomic Boundary Provenance

## 2.1 当前错误假设

当前 `evolution/atomicizer.py` 存在：

```python
if any(value not in inputs.values() for value in outputs.values()):
    raise ValueError(
        f"Atomic output lacks reusable input identity: {proposal.phase_id}"
    )
```

E1 Prompt 也要求：

```text
output_roles:
- each output value must equal one input value
```

这个规则适合旧 Atomic：

```text
clean(object)
heat(object)
place(object,destination)
```

因为输出 identity 通常只是把输入 identity 继续发布。

但它不适合：

```text
locate(target) -> entity, location
inspect(target) -> discovered property
resolve parameter -> concrete value
```

这些能力的 output 正是执行后才获得的新信息。

---

# 3. 冻结方向：Atomic Output 必须有“来源”，但不要求一定是 Input Identity

建议统一定义两种 output derivation：

```text
INPUT_IDENTITY
EFFECT_WITNESS
```

不要新建：

```text
SearchAtomic
EvidenceAtomic
RuntimeAtomic
```

子类。

同一个 `AbstractAtomicSkill` 即可。

---

## 3.1 INPUT_IDENTITY

已有能力：

```text
input object = cup_3
output object = cup_3
```

声明：

```yaml
output_derivations:
  object:
    kind: input_identity
    input_role: object
```

---

## 3.2 EFFECT_WITNESS

Fresh output：

```text
input target = cup

effect witness:
entity.discovered_at(
    entity=cup_3,
    location=countertop_2
)

outputs:
entity=cup_3
location=countertop_2
```

声明：

```yaml
output_derivations:
  entity:
    kind: effect_witness
    predicate: entity.discovered_at
    argument_role: entity

  location:
    kind: effect_witness
    predicate: entity.discovered_at
    argument_role: location
```

Code 必须验证：

1. `predicate` 是 Atomic declared Effect；
2. `argument_role` 在该 Effect schema 中存在；
3. R1 / Extractor source occurrence 中存在真实 matching Effect witness；
4. witness argument 的 concrete value 等于该 output value；
5. cardinality / distinctness 与 Atomic Effect 一致；
6. 一个 required output 只有一个 authoritative derivation。

LLM 不能通过 `output_derivations` 创造不存在的值。

---

# 4. Schema 落点

为了避免给 `AbstractAtomicSkill` 做破坏性字段迁移，本轮建议：

> v3.2-R1.1 先把 normalized derivation 存在现有 `validator_spec["output_derivations"]`。

即：

```python
atomic.validator_spec = {
    "validator_id": "...",
    "identity_strict": True,
    "output_derivations": {
        ...
    },
}
```

旧：

```text
validator_spec["output_identity"]
```

作为兼容输入保留，但在 load / canonicalization 时转换为：

```text
output_derivations.kind = input_identity
```

之后 Core Runtime / Validator 只读 normalized `output_derivations`。

不要同时维护：

```text
output_identity
output_derivations
```

两个独立 authority。

---

# 5. AtomicOccurrenceProposal 新字段

建议在：

```text
evolution/atomicizer.py
agents/structured_submission.py
```

增加：

```yaml
input_provenance_refs:
  <input_role>: <authority_ref>

output_derivations:
  <output_role>:
    kind: input_identity | effect_witness
    input_role: ...
    predicate: ...
    argument_role: ...
```

---

# 6. Input Provenance 也必须扩展

当前 Atomicizer 主要要求：

> concrete input value 必须能在前序/当前 action arguments 中找到。

这同样不够支持：

```text
Runtime Automation input target="cup"
```

因为 `cup` 可能来自：

```text
current_occurrence_anchor.object
```

不是环境 action argument。

---

## 6.1 Input provenance authority 类型

第一版允许：

```text
ACTION_ARGUMENT
TASK_SEMANTIC_ANCHOR
CONFIRMED_BINDING
DATA_FLOW
VALIDATED_OUTPUT
```

不要允许 Agent 自由写字符串当 authority。

---

## 6.2 Runtime Automation 必须把实际 trial input authority 写进 Trace

当前 `RuntimeAutomationAtomicDraft.input_binding_specs` 已表达：

```yaml
target:
  kind: current_occurrence_anchor
  source_role: object
```

但 Success Extractor 还需要 code-side 的 resolved authority。

`runtime_tool_trials[draft_id]` 建议新增：

```yaml
trial_bindings:
  target: cup

input_authorities:
  target:
    kind: current_occurrence_anchor
    source_occurrence_id: ...
    source_role: object
    value: cup
    authority_ref: runtime_input:<draft_id>:target
```

`authority_ref` 由代码生成，LLM 不能自定义。

---

## 6.3 普通非 Runtime-created Atomic

如果 input 来自 action：

```yaml
input_authorities:
  object:
    kind: action_argument
    event_id: ...
    argument_role: object
    value: cup_3
    authority_ref: action_arg:<event_id>:object
```

如果来自 upstream validated DataFlow：

```yaml
kind: validated_output
producer_occurrence_id: ...
producer_role: ...
```

这样以后 Input provenance 不再依赖：

```text
“值有没有刚好出现在某个 action arguments”
```

的隐式猜测。

---

# 7. Extractor E1 Context

Code 在 E1 Context 中新增只读：

```yaml
boundary_authorities:
  inputs: [...]
  effects: [...]
```

例如：

```yaml
inputs:
  - authority_ref: runtime_input:draft_1:target
    role: target
    value: cup
    source_kind: current_occurrence_anchor

effects:
  - witness_ref: alfworld_action_fact:r7:entity.discovered_at:...
    predicate: entity.discovered_at
    args:
      entity: cup_3
      location: countertop_2
```

Extractor 只能引用这些 authority refs。

---

# 8. E1 Prompt 修改

删除：

```text
each output value must equal one input value
```

替换为：

```text
Every required output must have exactly one code-verifiable derivation.

An output may be:
1. INPUT_IDENTITY: exactly the same concrete identity as one declared input; or
2. EFFECT_WITNESS: a concrete argument of one declared authoritative Effect witness.

Do not invent an output value.
Do not derive an output from observation prose.
Use only supplied boundary_authorities / effect witness refs.
```

同时：

```text
input_roles
```

不再要求只能从 selected action slice argument 复制。

改成：

> 每个 input 必须引用一个 code-authoritative input provenance ref。

---

# 9. Atomicizer 新验证顺序

建议：

```text
Envelope
→ support_event_ids
→ causal lineage
→ input provenance
→ precondition witness
→ effect witness
→ output derivations
→ ordering constraints
→ canonical occurrence
```

---

## 9.1 Input

对每个 input：

```text
proposal.input_provenance_refs[role]
```

必须 resolve 到：

```text
authority.role/value == proposal.input_roles[role]
```

---

## 9.2 Output — INPUT_IDENTITY

验证：

```text
output value == input_roles[input_role]
```

---

## 9.3 Output — EFFECT_WITNESS

验证：

```text
effect witness exists
predicate matches
argument_role exists
witness.args[argument_role] == output value
```

---

## 9.4 禁止

```text
output没有derivation
一个output有多个authority
output来自raw observation
output来自Agent prose
output来自未验证Tool return
```

---

# 10. CanonicalAtomicOccurrence 必须保存 Derivation

新增/保留：

```yaml
input_provenance_refs:
output_derivations:
effect_witness_refs:
```

否则：

```text
Atomicizer验证过
        ↓
System._canonical_atomic_for_occurrence()
        ↓
再次退回 output=input 模型
```

---

# 11. `System._canonical_atomic_for_occurrence()` 必须修改

当前逻辑：

```text
for every output:
    find input with same concrete value

if none:
    return None
```

必须删除。

改为：

```text
occurrence.output_derivations
        ↓
normalize / deterministic recheck
        ↓
atomic.validator_spec["output_derivations"]
```

### Legacy occurrence

如果没有新字段：

```text
output == input
```

时允许自动迁移为 `input_identity`。

如果 output 不是 input，且没有 derivation：

```text
fail closed
```

不能猜。

---

# 12. R1.1-2 — Atomic Effect Validation 必须能引用 Fresh Outputs

这是第二个 P0。

Runtime task-local Tool 可以：

```text
RETURN:
entity = cup_3
location = countertop_2
```

但是当前 `AtomicValidator.validate()`：

```text
plain = input bindings
validator_channel.validate_atomic_effect(
    effects=atomic.effects,
    bindings=plain
)
```

`output_candidates` 没进入 Effect binding。

因此 Effect：

```text
entity.discovered_at(
    entity=$entity,
    location=$location
)
```

中的 `$entity/$location` 无法从 Tool fresh output resolve。

---

# 13. 冻结修法：Merged Atomic Boundary Values

`AtomicValidator.validate()` 应建立：

```python
boundary_values = {
    **plain_inputs,
    **validated_output_candidates,
}
```

但不能直接信 Tool 输出。

顺序必须是：

```text
Tool RETURN
    ↓
Tool output schema validation
    ↓
output_candidates
    ↓
Atomic output_derivation validation
    ↓
只有 derivation-backed outputs 进入 boundary_values
    ↓
Harness validate Atomic Effect
```

不能：

```text
Tool说 entity=cup_3
→ 直接把 cup_3 当事实
```

---

## 13.1 INPUT_IDENTITY

值由：

```text
input binding authority
```

证明。

---

## 13.2 EFFECT_WITNESS

这里存在鸡蛋问题：

```text
Effect witness 用 output role
output 又由 Effect witness证明
```

解决方式：

Harness resolver 不先信 output 值，而是：

```text
Atomic Effect pattern
+
known input bindings / semantic anchors
+
current authoritative facts
        ↓
resolve unique matching witness assignment
        ↓
derive output candidates
```

实际上当前 `AlfWorldValidatorChannel.resolve_atomic_effect()` 已经能从 Effect witness argument 中生成 contextual outputs。

所以：

### Runtime current-effect path

继续使用：

```text
resolve_atomic_effect()
```

作为 fresh-output authority。

### Tool Implementation path

`ImplementationRunner` 不应只调用：

```text
AtomicValidator.validate()
```

来完成 generated-output Atomic。

需要增加一个统一方法，例如：

```python
AtomicValidator.validate_execution_result(
    atomic,
    occurrence,
    input_bindings,
    tool_output_candidates,
    validator_channel,
)
```

逻辑：

1. 对 input-identity outputs 做一致性校验；
2. 调 Harness `resolve_atomic_effect()` 找真实 witness；
3. 得到 authoritative `resolution.output_candidates`；
4. 与 Tool RETURN outputs 比较；
5. 只有完全一致才 Atomic pass。

这样：

> Tool RETURN 是程序返回值；Harness Effect witness 才是语义 authority。

---

# 14. Output mismatch Failure

例如 Tool 返回：

```text
entity=cup_3
```

但 current authoritative witness 是：

```text
entity=mug_2
```

必须：

```text
atomic_output_effect_witness_mismatch
```

不能把 Tool return 覆盖 Harness。

---

# 15. R1.1-3 — `RETURN` 是 tool_ir_v1 的 Tool Output Authority

当前 `ToolCompiler.compile_proposal()` 对 fresh output 构造：

```text
artifact.output_mapping[role]
    = TOOL_OUTPUT(primary, role)
```

但这在 ToolAsset 自身内部形成了类似：

```text
Tool output <- Tool output
```

的 self-reference。

Admission legacy closure 又只接受：

```text
SKILL_INPUT
CONSTANT
```

导致 fresh-output Tool 被拒。

---

# 16. 冻结语义

### `primitive_ir`

继续：

```text
artifact.steps
artifact.output_mapping
```

保持 R3.1 兼容。

### `tool_ir_v1`

Tool output authority 是：

```text
RETURN.output_sources
```

不要再用：

```text
artifact.output_mapping
```

表达 Tool 内部 output derivation。

---

# 17. ToolCompiler v3.2 IR

对于：

```text
ToolProposal.outputs == Atomic.outputs
```

生成：

```yaml
ToolAsset:
  interface:
    output_schema: ...

  artifact:
    schema_version: 1
    max_actions: ...
    program: ...
    final_effects: ...
    evidence_outputs: ...
    path_expectations: ...
```

不再生成：

```text
artifact.output_mapping
```

给 `tool_ir_v1`。

---

## 17.1 Implementation Output Mapping

仍然生成：

```text
Implementation.execution_policy.output_mapping:
  atomic_output_role
      ← TOOL_OUTPUT(primary, tool_output_role)
```

例如：

```yaml
entity:
  kind: tool_output
  source_step: primary
  source_role: entity

location:
  kind: tool_output
  source_step: primary
  source_role: location
```

这是正确层级：

```text
Tool RETURN
→ Tool output
→ Implementation mapping
→ Atomic output
```

---

# 18. Admission 对 Tool IR Output Closure

`Admission._tool_ir_closure_failures()` 不再要求：

```text
artifact.output_mapping == required outputs
```

而是调用 `ToolStaticValidator` 证明：

> 所有当前 replay / validated path 的 required Tool outputs 都由 RETURN 产生。

最低 Gate：

```text
every required Tool output
∈ at least one RETURN.output_sources
```

Path coverage 仍按 v3.2 Lifecycle 继续积累。

---

# 19. ToolRunner

`tool_ir_v1`：

```text
RETURN
→ state.outputs
```

如果 program 没有产生 required output：

```text
tool_ir_return_output_unresolved
```

不要 fallback 到 legacy：

```text
artifact.output_mapping
```

`primitive_ir` 保持原逻辑。

---

# 20. R1.1-4 — Recursive Tool IR Safety 还需补齐

当前 Tool IR 已支持 nested `IF/FOR_EACH`。

但 `ToolCompiler` 当前收集：

```python
action_nodes = [
    node for node in program
    if node["op"] == "ACTION"
]
```

只看 top-level ACTION。

所以：

```text
FOR_EACH:
  body:
    ACTION GO_TO
    ACTION OPEN
```

不会进入：

```text
tool.safety.allowed_action_types
```

这是错误的。

---

## 20.1 统一 Program Walker

在 `tooling/ir.py` 提供唯一：

```python
walk_program_nodes(program)
```

返回所有 nested nodes。

以下模块统一使用：

```text
ToolCompiler
ToolStaticValidator
Admission
ToolRunner metrics/path
Concrete-ID leakage scanner
```

不要各自写不同递归。

---

## 20.2 allowed_action_types

必须从所有 nested ACTION 收集：

```text
GO_TO
OPEN
TAKE
...
```

---

## 20.3 Concrete-ID Leakage

当前 `_concrete_ids_from_nodes(program)` 也主要扫描 top-level node。

必须改成递归 walker。

以下 nested 位置都检查：

```text
ACTION.constant
IF condition.value
STOP_WHEN value
FOR_EACH local_deterministic.values
FOR_EACH selector literals
RETURN constant
expected_effects
path_expectations
```

---

# 21. Semantic Evidence Source Schema 也必须统一

当前 Tool IR source 包含：

```text
semantic_evidence
binding_evidence
```

但：

```text
action_catalog entry:
{
  action_type,
  arguments
}

semantic fact:
{
  predicate,
  args
}
```

结构不同。

不要让同一个 selector helper 默认所有来源都有：

```text
arguments
```

---

## 21.1 Generic structured selector

建议 source 明确：

```yaml
source: semantic_evidence
where:
  predicate: entity.discovered_at

project:
  kind: argument
  role: entity
```

`semantic_evidence` 使用：

```text
entry.args
```

`action_catalog` 使用：

```text
entry.arguments
```

`binding_evidence` 使用它自己的 schema。

---

## 21.2 RETURN 同样支持 structured projection

例如：

```yaml
entity:
  source: semantic_evidence
  where:
    predicate: entity.discovered_at
  project:
    kind: argument
    role: entity
```

但如果核心 Locate Tool 通过 nested Action Catalog loop + local variables RETURN 已能工作，此功能可排在本轮 P1；**不能在 ToolBuilder Prompt 中宣称 semantic_evidence RETURN 是可用接口而实际无法表达。**

最低要求二选一：

1. 本轮真正实现 structured semantic-evidence selector/RETURN；或
2. v3.2 ToolBuilder Harness Interface 暂时不公开这条尚未完整支持的 output source。

推荐 **1**，因为它与 evidence-domain Tool 是同一方法边界。

---

# 22. R1.1-5 — Terminal-Empirical PlannerValidator Blocking / Diagnostic 分离

当前已经有：

```python
terminal_empirical = (
    selected_composite_authority.kind == "terminal_empirical"
)
```

并且：

```text
coverage=false
```

时不再 append `task_contract_mismatch`。

但是最后仍：

```python
passed = not errors and all(checks.values())
```

而：

```text
checks["task_contract_effect_coverage"] = False
```

仍会让整个 Planner Validation fail。

所以：

> Terminal-Empirical Candidate 现在能“创建”和“检索”，但不能真正通过 PlannerValidator进入 Runtime。

---

# 23. 冻结修法

PlannerValidator 输出需要区分：

```text
blocking_checks
diagnostic_checks
```

不必马上改 `ValidationResult` 数据结构，可以先内部维护：

```python
blocking_check_names = {...}
```

---

## 23.1 complete_contract Composite

以下仍 blocking：

```text
task_contract_effect_coverage
identity_cardinality_preserved
required_inputs_closed
...
```

---

## 23.2 terminal_empirical Composite

保持真实：

```text
checks["task_contract_effect_coverage"] = False
```

但这个 check 是 diagnostic。

仍然 blocking：

```text
node refs valid
harness compatibility
edge closure
DataFlow types
required inputs closed
identity/cardinality safety for effects it actually claims
Candidate lifecycle authority
terminal certificate valid
```

不能为了 empirical Candidate 把 structural safety 放松。

---

## 23.3 ValidationResult

例如：

```yaml
passed: true

checks:
  task_contract_effect_coverage: false
  terminal_empirical_authority_valid: true
  graph_structure_valid: true
  ...

metadata/diagnostic:
  incomplete_task_contract_allowed_by: terminal_empirical
```

如果暂时不扩 `ValidationResult.metadata`，可放在 `checks`：

```text
terminal_empirical_incomplete_coverage_nonblocking=true
```

---

# 24. R1.1-6 — Terminal-Empirical Current TaskContract Compatibility

仅靠 lexical similarity + CandidateUsePolicy 仍然太宽。

一个：

```text
clean prefix
```

不能因为文本像，就被拿去探索：

```text
heat task
```

---

## 24.1 推荐 Gate

Terminal empirical Candidate 的：

```text
observed covered effects
```

必须是当前 TaskContract 的：

> 非空、deterministically compatible subset。

例如：

```text
candidate observed:
object.cleaned

current:
object.cleaned + object.at_location
→ eligible
```

```text
candidate observed:
object.cleaned

current:
object.heated + object.at_location
→ reject
```

---

## 24.2 Authority 来源

不能只存 predicate string。

当前 certificate 的：

```yaml
covered_effects:
  - object.cleaned
```

信息过窄。

建议改成：

```yaml
covered_effects:
  - predicate: object.cleaned
    effect_domain: world
    argument_roles: [object]
    cardinality: 1
    distinct_by: ""
```

不要保存 episode concrete ID 到 reusable compatibility signature。

使用现有 `ContractMatcher` / predicate shape / semantic type authority做 subset compatibility。

---

# 25. Terminal Candidate Retrieval 顺序保持

继续：

```text
stable complete-contract ACTIVE/PREFERRED
        >
eligible terminal empirical exploration
        >
P1 / Atomic composition / fallback
```

单次 empirical Candidate 不能抢掉稳定完整 Composite。

---

# 26. Lifecycle Gate 26–28 必须补测试

当前已有：

```text
Gate23 candidate creation
Gate24 separate channel
Gate25 candidate policy
```

还需正式加入：

### Gate26

source task：

```text
SELF_SUFFICIENT_SUCCESS #1
```

独立 task：

```text
SELF_SUFFICIENT_SUCCESS #2
```

现有：

```text
composite_active_self_sufficient_successes=2
```

触发：

```text
Candidate → Active
```

重复相同 task_id 不算第二次 independent support。

---

### Gate27

Empirical Candidate 执行：

```text
benchmark won=false
```

随后 Dynamic rescue：

```text
benchmark won=true
```

该 Candidate：

```text
SELF_SUFFICIENT_SUCCESS += 0
```

---

### Gate28

New shorter Candidate 尚未 Active：

```text
old longer Composite 保持原状态
```

只有：

```text
new Active
+
stable replacement evidence
```

后才：

```text
old Active → Suppressed → Retired
```

---

# 27. Stable Replacement Relation

对于：

```text
old = A → B → C
new = A → B
```

需要建立 deterministic replacement eligibility：

1. new `completion_authority=terminal_empirical`；
2. new 已 Active；
3. `source_composite_ref == old.ref`，或存在确定 DERIVED_FROM lineage；
4. new sequence 是 old 实际 executed prefix / causally sufficient subset；
5. old omitted tail 在 source terminal trace 中真实为 `SKIPPED_GOAL_TERMINAL`；
6. new 已有 independent terminal self-sufficient evidence。

才允许调用现有：

```text
assign_superseded
→ stable_replacement
→ suppress / retire
```

禁止：

> 只因为“new节点更少”就替换 old。

---

# 28. R1.1-7 — 必须新增跨任务 End-to-End Gate

这是本轮正式 Full-30 前最重要的测试。

现有大量 Gate 都是单模块。

但 v3.2 方法真正要证明的是：

> Runtime 自己做出的 Tool 能在成功任务后变成长期知识，并在下一独立任务无 LLM 重复搜索地复用。

---

# 29. Gate 29 — Runtime Self-Tool Cross-Task Reuse

## Task A：产生新能力

初始 Bank：

```text
无 locate Atomic
```

某 blocked Atomic：

```text
needs object
semantic anchor object=cup
```

Runtime Agent 提：

```text
Automation Atomic:
locate semantic target

input:
target <- current_occurrence_anchor.object

outputs:
entity
location

effect:
entity.discovered_at(entity=$entity, location=$location)
```

---

## Task A ToolBuilder

ToolBuilder 生成例如：

```text
FOR_EACH reachable destination
    ACTION GO_TO(destination)

    FOR_EACH current TAKE action semantically compatible with target
        RETURN
            entity = selected TAKE.object
            location = current destination
```

或者等价 bounded IR。

要求：

```text
内部 environment action > 1
内部 LLM call = 0
```

---

## Task A R1

实际发现：

```text
entity=cup_3
location=countertop_2
```

必须：

```text
Tool RETURN valid
entity.discovered_at witness exists
Atomic fresh output derivation valid
R1 pass
```

---

## Task A Success Evolution

Benchmark 最终 won。

Extractor 必须能形成：

```text
Atomic:
input target
outputs entity/location
effect entity.discovered_at
```

而不是：

```text
input target/entity/location
```

---

## Task A Tool Admission

必须：

```text
ToolCandidate admitted
ImplementationCandidate admitted
```

不能：

```text
tool_output_mapping_not_closed
```

---

## Task B：跨任务复用

新 task，新的 concrete target：

```text
cup_7
```

当前 downstream Atomic 缺：

```text
object
```

SupportRetriever：

```text
Locate.entity
→ downstream.object
```

返回 candidate。

Agent选择 support invocation。

---

## Task B Tool execution

要求：

```text
不调用新的 ToolBuilder
不重新逐步 LLM 搜索
existing Locate Tool执行
zero-LLM 内部搜索
找到 cup_7 / new location
```

输出：

```text
entity → downstream object
```

并继续 downstream Atomic。

---

# 30. Gate29 判定标准

至少检查：

```yaml
task_a:
  runtime_automation_r0_pass: true
  task_local_tool_r1_pass: true
  learned_atomic_inputs: [target]
  learned_atomic_outputs: [entity, location]
  learned_tool_status: candidate
  learned_implementation_status: candidate

task_b:
  support_atomic_ref == task_a.learned_atomic_ref
  support_tool_ref == task_a.learned_tool_ref
  tool_builder_runtime_calls_for_locate == 0
  learned_tool_invocation_count >= 1
  llm_bypassed_environment_actions >= 1
  output_entity != task_a.output_entity
  downstream_atomic_receives_output: true
```

关键：

> Task B concrete entity 必须与 Task A 不同。

否则不能证明泛化。

---

# 31. Gate30 — Fresh Output Negative Test

Tool 返回：

```text
entity=mug_2
```

但 effect witness：

```text
entity.discovered_at(cup_3,countertop_2)
```

必须：

```text
Atomic reject
tool may have run, but Implementation/Atomic does not pass
```

---

# 32. Gate31 — Input Authority Negative Test

Runtime Draft 声明：

```text
target=cup
```

但 E1 Proposal 引用不存在的：

```text
runtime_input:fake:target
```

必须：

```text
Extractor Atomic reject
```

---

# 33. Gate32 — Fresh Output 不能从 Raw Observation 产生

Raw text：

```text
"You see cup 3 on countertop 2"
```

但没有：

```text
structured effect/evidence witness
```

Extractor不能把：

```text
cup_3 / countertop_2
```

作为 EFFECT_WITNESS output。

---

# 34. Gate33 — Recursive Concrete-ID Leakage

Tool：

```text
FOR_EACH:
  body:
    ACTION GO_TO constant cabinet_7
```

虽然 concrete ID 位于 nested body：

```text
Static reject
```

同样覆盖：

```text
nested IF
nested STOP
nested RETURN constant
nested local_deterministic values
```

---

# 35. Gate34 — Recursive Safety Action Collection

Tool：

```text
FOR_EACH:
  body:
    ACTION GO_TO
    IF:
      ACTION OPEN
```

最终：

```yaml
tool.safety.allowed_action_types:
  - GO_TO
  - OPEN
```

不能是空集合。

---

# 36. Gate35 — Terminal Empirical Planner Execution

构造：

```text
TaskContract:
clean + at_location

Terminal empirical Candidate:
clean only
```

必须：

```text
retrieve_terminal -> candidate
PlannerValidator -> passed
task_contract_effect_coverage=false diagnostic
Runtime -> executes candidate
```

不能在 PlannerValidator 处被 `all(checks.values())` 挡掉。

---

# 37. Gate36 — Terminal Empirical Contract Subset

Candidate：

```text
covered=clean
```

Current Task：

```text
clean + place
→ eligible
```

Current Task：

```text
heat + place
→ ineligible
```

---

# 38. R1.1-8 — ALFWorld 精确版本冻结

当前 `pyproject.toml` 仍是：

```toml
alfworld = ["alfworld>=0.3.0"]
```

所以正式环境版本仍然浮动。

由于当前：

- `infos["won"]`；
- admissible command；
- TextWorld terminal；
- observation wording；
- USE semantics；

都直接影响 Harness 与 Validator，正式实验必须记录精确版本。

---

# 39. 实验前要求

正式环境已确认：

```text
ALFWorld 0.4.2
```

因此必须将：

```toml
alfworld = ["alfworld>=0.3.0"]
```

修改为：

```toml
alfworld = ["alfworld==0.4.2"]
```

并将版本写入：

```yaml
RunManifest.environment:
  alfworld_version: "0.4.2"
```

formal preflight 必须执行等价检查：

```text
installed ALFWorld == 0.4.2
```

不匹配则：

```text
protocol/environment failure
→ formal run blocked
```

不能自动接受更高版本，也不能静默继续。

---

# 40. 文件级修改清单

## P0 必改

### `src/atomic_skillgraph/evolution/atomicizer.py`

- `AtomicOccurrenceProposal.input_provenance_refs`
- `AtomicOccurrenceProposal.output_derivations`
- fresh output validation
- input authority validation
- legacy output identity migration
- preserve derivation into `CanonicalAtomicOccurrence`

### `src/atomic_skillgraph/agents/structured_submission.py`

扩展 E1 Schema。

### `src/atomic_skillgraph/agents/context_builder.py`

- E1 boundary authorities
- 删除 `each output value must equal one input`
- 增加 INPUT_IDENTITY / EFFECT_WITNESS 规则

### `src/atomic_skillgraph/runtime/automation.py`

Runtime trial 记录：

```text
trial_bindings
input_authorities
R1 outputs
R1 witness refs
```

### `src/atomic_skillgraph/system.py`

- `_canonical_atomic_for_occurrence`
- Runtime trial → Extractor authority projection
- ToolBuilder success evolution context
- Terminal empirical replacement evidence
- Gate29 integration fixture 如放系统测试

### `src/atomic_skillgraph/validation/atomic_validator.py`

增加 generated-output validation path。

### `src/atomic_skillgraph/harness/alfworld.py`

保证 `resolve_atomic_effect` / output candidate 与 fresh output derivation一致；不新增 workflow。

### `src/atomic_skillgraph/evolution/tool_compiler.py`

- tool_ir_v1 不再写 self-referential artifact output mapping
- Implementation 用 TOOL_OUTPUT 映射 Atomic output
- recursive ACTION collection

### `src/atomic_skillgraph/evolution/admission.py`

- tool_ir output closure 改看 RETURN
- 不再使用 legacy artifact.output_mapping gate

### `src/atomic_skillgraph/tooling/ir.py`

- shared recursive walker
- semantic evidence structured selector / projection
- RETURN structured source

### `src/atomic_skillgraph/tooling/validator.py`

- recursive concrete leakage
- nested action safety
- RETURN required output closure
- semantic evidence selector validation

### `src/atomic_skillgraph/runtime/tool_runner.py`

- tool_ir_v1 output only from RETURN
- no legacy fallback
- structured evidence selector
- fresh output path evidence

### `src/atomic_skillgraph/planner/validator.py`

blocking vs diagnostic checks。

### `src/atomic_skillgraph/planner/composite_retriever.py`

terminal empirical current-contract subset compatibility。

### `src/atomic_skillgraph/evolution/composite_builder.py`

terminal certificate 保存 reusable covered-effect signature。

### `tests/test_v32_r1_gates.py`

新增 Gate26–36。

---

# 41. 不应修改

本轮仍禁止：

```text
增加 Runtime 300K budget
增加 node 100K budget
增加 action budget
增加 Planner top-k
增加 P1R次数
增加 lifecycle threshold
写 ALFWorld cup/cabinet search
写 heat/microwave search
写 clean/sink search
写 look/lamp workflow
让 Raw observation 成为 Atomic Effect authority
让 Tool RETURN 自己证明 Atomic Effect
重新启用 mechanical ToolCompiler 作为正式 fallback
```

---

# 42. 修复顺序

建议：

### Phase 1 — Atomic Boundary Model

1. input provenance；
2. output derivation；
3. Atomicizer；
4. `_canonical_atomic_for_occurrence()`；
5. AtomicValidator generated output。

### Phase 2 — Tool Return

6. ToolCompiler；
7. ToolStaticValidator；
8. Admission；
9. ToolRunner；
10. recursive walker / leakage。

### Phase 3 — Runtime → Evolution

11. runtime trial authority；
12. Extractor context；
13. persistent Atomic/Tool admission。

### Phase 4 — Terminal Empirical

14. Planner blocking/diagnostic；
15. current TaskContract subset compatibility；
16. Gate26–28 replacement lifecycle。

### Phase 5 — Cross-task Gate

17. Gate29/30/31/32；
18. real ALFWorld targeted self-tool smoke。

### Phase 6 — Formal protocol

19. pin ALFWorld；
20. freeze new commit；
21. fresh empty bank。

---

# 43. 新 Formal Full-30 启动条件

除 v3.2-R1 原有 Gate 外，再增加：

```text
[ ] fresh-output Atomic deterministic Gate通过
[ ] runtime semantic-anchor input provenance Gate通过
[ ] Tool IR fresh RETURN Admission Gate通过
[ ] recursive Tool safety Gate通过
[ ] recursive episode-id leakage Gate通过
[ ] terminal empirical Planner execution Gate通过
[ ] terminal empirical current-contract subset Gate通过
[ ] Gate26 independent promotion通过
[ ] Gate27 rescue no-credit通过
[ ] Gate28 stable replacement通过
[ ] Gate29 cross-task Runtime Tool reuse通过
[ ] Gate30 output/effect mismatch fail-closed
[ ] Gate31 fake input authority fail-closed
[ ] Gate32 raw observation cannot create output
[ ] actual ALFWorld exact version recorded
[ ] ALFWorld version protocol-pinned
[ ] deterministic suite全过
[ ] real ALFWorld smoke全过
[ ] frozen new commit
[ ] fresh empty v3.2 bank
```

---

# 44. R1.1 完成后方法链应是

```text
Task A
  │
  ▼
Runtime Agent
  │ predicts repetitive low-value grounding
  ▼
Automation Atomic Draft
  │
  ├─ input target
  │   authority = semantic anchor
  │
  ▼
R0
  ▼
ToolBuilder
  ▼
Task-local Tool IR
  ▼
Zero-LLM search
  ▼
RETURN entity/location
  │
  ▼
Harness Effect Witness
entity.discovered_at(entity,location)
  │
  ▼
R1
Tool output == authoritative witness output
  │
  ▼
Task A won
  │
  ▼
Success Extractor
  │
  ├─ input provenance
  ├─ effect witness
  └─ output derivation
  ▼
Canonical Atomic
input target
outputs entity/location
  │
  ▼
ToolBuilder / validated runtime proposal
  ▼
Tool Admission
  ▼
Implementation
  ▼
Lifecycle Candidate

================ NEW TASK ================

Task B
  │
  ▼
downstream Atomic missing object
  │
  ▼
Support Atomic Retrieval
  ▼
Locate Atomic
  │
  ▼
existing learned Tool
  │ zero LLM internal search
  ▼
new entity/location
  │
  ▼
DataFlow entity -> object
  ▼
downstream Atomic
```

这才是 v3.2 Runtime Tool Evolution 的完整实验对象。

---

# 45. 修复成功标准

R1.1 不要求新的 Full-30 一定超过 R3.1 的 96.67% Benchmark success。

先要求机制真实成立：

1. Runtime search Tool 可以产生执行前未知的 fresh output；
2. fresh output 由 structured Effect witness 证明；
3. Success Extractor 能保留 `input target → output entity/location`，不会把 output 倒塞进 input；
4. Tool IR `RETURN` 能正式 Admission；
5. 下一独立 task 能检索同一 Atomic/Tool；
6. 下一任务 Tool 找到不同 concrete entity；
7. 中间搜索动作不调用 LLM；
8. Terminal empirical Candidate 真能通过 PlannerValidator进入 Runtime；
9. empiric Candidate 不跨不兼容 TaskContract 滥用；
10. ALFWorld 环境版本固定。

然后才看：

```text
Runtime Preparation tokens
Seeded tokens
ToolBuilder cost
LLM bypassed actions
Support reuse count
Tool reuse count
Benchmark success
Graph self-sufficient execution
```

---

# 46. 仍需用户确认的 3 个边界

## 46.1 Output Derivation 最终冻结

已确认：

> `INPUT_IDENTITY / EFFECT_WITNESS` 是 **所有 Atomic 的统一 output derivation model**，不只用于 `effect_domain=evidence` 的 Search / Grounding Atomic。

因此：

- world-effect Atomic 也允许 `EFFECT_WITNESS` fresh output；
- evidence-effect Atomic 不新增特殊子类；
- Planner / DataFlow / Validator 统一消费 Atomic outputs；
- 代码不得通过 task type / benchmark family 判断某个 Atomic 是否可以产生 fresh output；
- 唯一判断依据是其 `output_derivations` 是否被 deterministic witness validation 证明。

---

## 46.2 Terminal-Empirical 当前任务兼容规则最终冻结

已确认：

> Terminal-Empirical Candidate 的真实 observed covered effects 必须构成当前 TaskContract 的 **非空 compatible subset**。

例如：

```text
candidate observed:
object.cleaned

current task:
object.cleaned + object.at_location
→ eligible
```

而：

```text
candidate observed:
object.cleaned

current task:
object.heated + object.at_location
→ reject
```

该规则：

- 不要求 empirical Candidate 完整覆盖当前 TaskContract；
- 不能仅凭 lexical similarity / CandidateUsePolicy 放行；
- 不得使用 benchmark task type 分支；
- 必须由 ContractMatcher / predicate shape / semantic type / cardinality authority 做 deterministic compatibility。

---

## 46.3 ALFWorld 版本最终冻结

已确认当前正式实验环境：

```text
ALFWorld 0.4.2
```

正式协议必须 pin：

```toml
alfworld = ["alfworld==0.4.2"]
```

并在：

- formal preflight；
- RunManifest；
- Trace / report environment metadata；

中记录：

```yaml
alfworld_version: "0.4.2"
```

正式 Full-30 若检测到安装版本不是 `0.4.2`：

```text
preflight fail
```

不得继续运行并把结果与 R3.1 / v3.2 正式实验混合。

---

# 47. 最终冻结状态

本文件全部设计边界均已确认。

最终冻结：

```text
Output Derivation:
INPUT_IDENTITY / EFFECT_WITNESS
适用于所有 Atomic

Terminal-Empirical compatibility:
historically observed covered effects
必须是当前 TaskContract 的非空 compatible subset

ALFWorld:
0.4.2
```

实现阶段不得再把以上三项视为可选项。

v3.2-R1.1 完成后，只有在本文全部 deterministic gates、Gate29 跨任务 Runtime Tool reuse、真实 ALFWorld smoke、精确版本 preflight 和 fresh-bank protocol 均通过后，才允许：

```text
freeze new commit
→ fresh empty v3.2 bank
→ formal Full-30
```

任何实现如果需要：

- 重新把 fresh output 塞回 Atomic input；
- 信任 Tool RETURN 而不验证 Effect witness；
- 依赖 raw observation 产生 fresh output；
- 用 benchmark workflow hardcode 搜索策略；
- 通过提高预算掩盖 Runtime Tool 失败；
- 跳过 Admission / Replay；
- 允许 ALFWorld 非 0.4.2 正式运行；

都视为偏离本冻结文档，必须停止并重新审查。
