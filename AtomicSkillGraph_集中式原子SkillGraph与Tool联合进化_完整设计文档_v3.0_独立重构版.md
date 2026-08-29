# AtomicSkillGraph v3.0：独立 ToolCall Agent 与原子 Skill/Tool 联合自进化系统完整实现设计文档

**文档版本：** v3.0 Revised Design Freeze（独立重构版）  
**日期：** 2026-08-29  
**前序设计参考：** `AtomicSkillGraph_集中式原子SkillGraph与Tool联合进化_完整设计文档_v2.0.md`  
**历史代码参考点：** AtomicSkill-ToolGraph v2 最终代码及其 Git 历史  
**研究定位：** 独立的集中式 Self-Evolving Agent 系统；不再以 FlowEvo 扩展、兼容层或增量修改为研究定义  
**实现策略：** v3 整体重构；v2 仅作为行为、数据结构和失败经验的参考，不作为 v3 Runtime 的依赖链  
**文档性质：** 最终目标架构 + 软件工程详细设计 + 数据契约 + 执行协议 + 最小验证规范  
**状态：** 本文冻结 v3 必须一次实现的完整设计；不把核心能力推迟到所谓 v3.1/v3.2。

---

# 0. 执行摘要

AtomicSkillGraph v3.0 是一个独立的长期自进化 Agent 系统。它保留 v2 中最重要的知识表示：

```text
Global SkillGraph
├── AbstractAtomicSkill       # 稳定、可验证的原子能力：做什么
├── ImplementationAtom        # Skill 到可执行实现的绑定：如何做
└── CompositeSkill            # 原子能力的可复用完整流程

Global Tool Repository
└── ToolAsset                 # 真正可执行、可测试、可版本化的长期资产
```

v3 不再延续 v2 的核心运行方式。v2 中 Planner、Runtime、Adapter、Resolver、Validator 共同承担参数解析、位置探索、DataFlow、Tool 选择和信用更新，造成同一语义在多层重复实现。v3 将职责重新冻结为：

```text
Planner Pipeline
    负责：完整 Composite 查找、能力需求分解、Atomic 候选检索、唯一流程构建

Runtime Orchestrator
    负责：严格按 RuntimeLinearPlan 的 occurrence 顺序执行

Runtime Agent
    负责：当前节点存在语义不确定性时，探索环境、提出参数并调用当前节点允许的实现

Environment Action Tool
    负责：提供 Harness 当前允许的原生环境操作，不属于学习 Tool

Implementation Invocation
    负责：把一个 ImplementationAtom 作为 native ToolCall 暴露给 Agent，并执行其绑定的一个或多个 ToolAsset

ToolCall Preflight
    负责：Schema、Binding、Grounding、关系证据、兼容性和安全检查；未通过时 Tool 不启动

Validator
    负责：Tool、Atomic、Composite、TaskContract、Benchmark 分层验证

EvidenceLedger + CreditAssigner
    负责：唯一、幂等地记录事实并确定应该奖励或惩罚哪个长期对象

Extractor Session
    负责：成功轨迹第一轮提取 Atomic，代码验证后第二轮在同一会话中构建 Composite
```

v3 的 Planner 固定为：

```text
P0  完整 Composite 检索
    └─ 命中：实例化其 canonical control sequence

P1  Capability Requirement Proposal
    └─ Atomic 模糊召回与 Contract 过滤

P1R Requirement Repair（最多一次，仅在覆盖不完整时）
    └─ 可参考与部分命中节点相关的已知 Composite，但不能自动执行它

P2  Linear Workflow Proposal
    └─ 从候选 occurrence 中生成唯一的一条执行流程

P2R Graph Repair（最多一次，仅在验证失败时）

任一步仍不能得到完整合法计划
    └─ Full Dynamic
```

最终 Runtime 输入不是一般 DAG，而是：

```text
一个严格有序的 occurrence control sequence
+
允许跨步、允许多输入的 DataFlow / Dependency DAG
```

例如：

```text
控制序列：A → B → C

数据边：
A.output_x ───────────→ C.input_x
B.output_y ───────────→ C.input_y
```

这仍然是一条任务流程。只有控制序列必须线性；数据边和依赖边允许 fan-in/fan-out，但只能沿控制序列向前，不能形成时间逆向依赖。

Runtime 真正的执行机制只有两类：

```text
1. Implementation / Tool execution
2. Modern Runtime Agent execution
```

但保留三种语义 provenance：

```text
DIRECT
    Learned Implementation 真正 started=true。
    - direct_autonomous：参数全部已认证，零 LLM 直接执行
    - direct_agent_prepared：Agent 先探索/解析参数，再有效调用 Learned Implementation

SEEDED
    当前 Atomic 已存在，但 Direct 不可用或失败；Fresh Agent 只看 Atomic 语义自行完成节点。

DYNAMIC
    没有完整 SkillGraph 计划，或完整图执行后任务仍未完成；Agent 不接收 Atomic guideline。
```

v3 最重要的运行不变量是：

> **LLM 负责语义决策；代码负责契约、证据、合法性和信用。Agent 提出的值只是 proposal，只有经当前 Harness 证据认证后才能启动 Learned Implementation。**

---

# 1. 研究定位与代码重构原则

## 1.1 独立研究定位

v3 不再把贡献描述为“在 FlowEvo 上增加 Atomic SkillGraph”。研究对象是完整的：

```text
Atomic Skill / Implementation / Tool / Composite 的联合自进化系统
+
基于 native ToolCall 的现代 Planner/Runtime/Extractor Agent 架构
+
Evidence-driven 生命周期治理
```

其他 Skill 进化方法只作为外部对比方法。是否能够共用 Runtime、是否采用原方法的 native Runtime，由正式实验协议单独决定；v3 的内部设计不以兼容任一对比方法为前提。

## 1.2 整体重构，而不是继续修补 v2

v3 代码不得继续在 v2 的 `_run_env_nodes()`、自动位置发现、字符串 binding、Direct Gate 补丁链上累积条件分支。

固定处理方式：

1. 给 v2 最终状态打 Git tag，例如 `v2-final-reference`；
2. v3 使用新的数据 Schema、Runtime 和 Agent 协议；
3. 允许复用纯工具代码，例如 Ref、内容哈希、部分 predicate matcher、ALFWorld task loader；
4. 任何 v2 模块只有通过 v3 contract 审计后才能移植；
5. v3 Core 不导入 v2 Runtime，也不依赖 vendored FlowEvo Runtime；
6. v2 Skill/Tool bank 不进入 v3 正式训练；正式实验从空 v3 bank 开始；
7. 不实现为了读取旧错误资产而放宽 v3 Gate 的兼容分支。

## 1.3 设计完整，工程垂直交付

本文定义的是一次完整的 v3 目标架构，不把核心能力延迟到后续版本。工程实现可以按垂直闭环分段完成，但每个分段都必须符合最终契约，不能先写一套临时语义、以后再替换。

---

# 2. 研究问题、目标与非目标

## 2.1 核心研究问题

### RQ1：如何从长期轨迹中发现真正可复用的原子能力？

原子能力由稳定状态转移、输入输出、独立验证和失败归因边界决定，而不是由 action 数、函数名或 task type 决定。

### RQ2：如何让 Agent 在不知道具体实例参数时仍能可靠复用 Learned Tool？

Agent 通过 Harness 提供的环境动作自主探索；参数必须经过 RuntimeBinding 与当前 GroundingEvidence 认证后才能启动 Implementation。

### RQ3：如何从原子能力库构造完整、唯一、可执行的任务流程？

Planner 先寻找完整 Composite；失败后先提出能力需求，再检索 Atomic，最后由 LLM 构建一条唯一的 occurrence 控制序列。代码只验证，不偷偷创造不存在的新边。

### RQ4：如何准确区分 Planner、Binding、Implementation、Tool、Atomic、Composite 的责任？

所有尝试输出统一执行结果；所有长期信用只由 append-only EvidenceLedger 和确定性 CreditAssigner 更新。

### RQ5：如何让长期知识可靠晋升、替换、抑制和退役，同时避免门控死锁？

Candidate 在 online train 中可受控使用以获得真实证据；Frozen test 只使用 Active/Preferred；退役必须依赖可靠替代或强负面证据，而不是“最近没调用”。

## 2.2 首要目标

1. 原子 Skill、Implementation、Tool、Composite 四层长期知识；
2. 完整 Composite → Atomic 组合 → Full Dynamic 的严格 Planner 路由；
3. native ToolCall、多轮 Session、reasoning 支持和 provider-independent Agent 协议；
4. 节点级 Runtime，已知 DataFlow 下零 LLM 自动执行；
5. Agent 自主探索、代码认证 grounding，减少 benchmark-specific Core 逻辑；
6. started-based 信用归因；
7. 同一 Extractor Session 两轮提取；
8. EvidenceLedger 驱动生命周期；
9. 长期图与执行控制序列分离；
10. 可审计、可冻结、可断点恢复的实验系统。

## 2.3 非目标

v3 首版不做：

- 联邦 Skill 聚合；
- 跨 Agent 个性化分发；
- 训练一个神经 Router；
- 让 LLM 浏览完整 Skill/Tool 库；
- 允许任意 Python 表达式作为参数 mapping；
- 把完整 Composite 编译成不可拆分 mega-tool；
- 用 hidden world state 替 Runtime Agent 找参数；
- 为旧 v2 bank 做宽松兼容；
- 在一个 Composite version 中表达多个可替代控制流程；
- 首版支持 Parallel、Loop、Branch、Retry 等复杂 Runtime 控制结构。

---

# 3. 核心语义不变量

## 3.1 唯一事实源

| 概念 | 唯一事实源 |
|---|---|
| Task 最终目标 | `TaskContract` |
| Planner 最终执行顺序 | `RuntimeLinearPlan.control_sequence` |
| 参数值与来源 | `RuntimeBindingStore` |
| 当前可执行环境动作 | `HarnessActionCatalog` |
| 参数关系是否可执行 | `GroundingEvidenceStore + GroundingConstraint` |
| Skill→Tool 映射 | `ImplementationAtom` |
| Tool 是否真正启动 | `ImplementationExecutionResult.started` / `ToolExecutionResult.started` |
| 节点是否成功 | `AtomicValidator` |
| 图是否自洽 | `CompositeValidator` |
| 长期信用 | `EvidenceLedger + CreditAssigner` |
| 生命周期状态 | Registry 中由 LifecycleProjection 生成的状态 |

任何下游模块不得重新用字符串、命名习惯或 task type 猜测这些语义。

## 3.2 Planner 不变量

- P0 只接受完整、可用、canonical workflow 明确的 Composite；
- Atomic 覆盖不完整时不执行半张图；
- 已有 verified edge 可以复用；
- 不存在的新 edge 必须由 Planner Agent 显式提出；
- Planner 最终只交付一条 control sequence；
- 同一个 Atomic 可用多个 occurrence 重复出现；
- Requirement Repair 最多一次；Graph Repair 最多一次；
- Planner 无法形成完整计划时直接 Full Dynamic。

## 3.3 Runtime 不变量

- Runtime 按 occurrence 顺序执行，不重新设计整张图；
- 当前节点参数已充分认证时，不调用 LLM；
- Agent 只能看到当前节点和当前允许的 Implementation Invocation；
- Agent 看不到完整 SkillGraph 和 Tool Repository；
- ToolCall preflight 未通过时，Implementation/Tool 均 `started=false`；
- Direct 失败后的 Seeded 使用 fresh session，不携带失败 Tool body；
- 节点 Seeded 失败后终止计划，不增加第三个节点级 Dynamic；
- 只有整题无计划或整图结束后任务未完成时进入 Full Dynamic。

## 3.4 Binding 不变量

- `apple` 与 `apple_2` 是不同解析级别；
- `TASK` 来源不自动等于 executable concrete；
- `AGENT_PROPOSED` 永远不能直接执行；
- 关系型 Tool 参数必须满足 GroundingConstraint；
- GroundingEvidence 带 world revision，旧证据可失效；
- 只有经过 Validator 认证的 Tool output 才能发布到 DataFlow。

## 3.5 信用不变量

- Tool/Implementation 只有在相应执行真正 started 后才获得执行证据；
- Agent 参数错误不惩罚 Tool；
- Implementation mapping 错误不惩罚 Tool body；
- Direct 失败、Seeded 成功：Atomic 可以得到正证据，失败的实现层按 started/failure layer 得到负证据；
- 图内所有节点成功但需要 Full Dynamic rescue：节点正、Composite 结构负；
- Goal early terminal 只更新 Composite occurrence skip，不惩罚全局 Atomic；
- Infrastructure failure 不进入任何长期负证据。

## 3.6 Reasoning 不变量

- 系统不解析 reasoning text；
- 只消费 assistant content、structured output、native tool calls；
- reasoning tokens 进入成本统计；
- Provider 不返回 reasoning 时记录 unavailable，不估算。

---

# 4. 总体架构

```text
┌────────────────────────────────────────────────────────────────────┐
│                     Benchmark / Agent Harness                      │
│ observation / action catalog / execute / done / won / validator   │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Harness Adapter                            │
│ policy-facing channel                  validator-only channel      │
│ observation + actions                  formal facts / verifier      │
└───────────────┬──────────────────────────────┬─────────────────────┘
                │                              │
                ▼                              ▼
┌────────────────────────────┐       ┌───────────────────────────────┐
│ Planner Pipeline           │       │ Validation Engine             │
│ P0 / P1 / P1R / P2 / P2R  │       │ Tool / Atomic / Composite /   │
└──────────────┬─────────────┘       │ TaskContract / Benchmark      │
               │                     └───────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    RuntimeLinearPlan                               │
│ control_sequence + data_edges + dependency_edges                  │
└──────────────┬─────────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Runtime Orchestrator                            │
│ BindingStore / EvidenceStore / NodeExecutor / TaskRuntimeContext   │
└───────┬───────────────────────┬──────────────────────┬─────────────┘
        │                       │                      │
        ▼                       ▼                      ▼
 Environment Action      Implementation Invocation   Fresh Seeded /
 native Tool             native Tool                 Full Dynamic Agent
        │                       │
        ▼                       ▼
 Harness action          ImplementationRunner
                               └─ ToolRunner(s)

执行结束
   │
   ├─ TraceStore
   ├─ EvidenceLedger → CreditAssigner → LifecycleProjection
   ├─ success → Extractor / Tool Compiler / Admission / Alignment
   └─ failure → FailureLocalizer / RepairProposal / replay-admission
```

---

# 5. 最终代码仓库结构

v3 最终使用新的 `src/atomic_skillgraph/` 实现。v2 通过 Git tag 保存，不在运行期 import。

```text
src/atomic_skillgraph/
├── core/
│   ├── refs.py
│   ├── status.py
│   ├── contracts.py
│   ├── bindings.py
│   ├── edges.py
│   ├── results.py
│   ├── errors.py
│   └── serialization.py
│
├── agents/
│   ├── protocol.py
│   ├── provider.py
│   ├── session.py
│   ├── usage.py
│   └── context_builder.py
│
├── knowledge/
│   ├── artifact_store.py
│   ├── skill_registry.py
│   ├── tool_registry.py
│   ├── graph_store.py
│   ├── query.py
│   └── database.py
│
├── planner/
│   ├── pipeline.py
│   ├── composite_retriever.py
│   ├── requirement_agent.py
│   ├── atomic_retriever.py
│   ├── related_composite.py
│   ├── workflow_agent.py
│   ├── compiler.py
│   └── validator.py
│
├── runtime/
│   ├── orchestrator.py
│   ├── task_context.py
│   ├── node_executor.py
│   ├── binding_store.py
│   ├── evidence_store.py
│   ├── invocation_compiler.py
│   ├── implementation_runner.py
│   ├── tool_runner.py
│   └── budget.py
│
├── validation/
│   ├── engine.py
│   ├── tool_validator.py
│   ├── atomic_validator.py
│   ├── composite_validator.py
│   ├── task_validator.py
│   └── failure_localizer.py
│
├── evolution/
│   ├── extractor_session.py
│   ├── trace_normalizer.py
│   ├── atomicizer.py
│   ├── composite_builder.py
│   ├── tool_compiler.py
│   ├── admission.py
│   ├── aligner.py
│   ├── repair.py
│   └── maintenance.py
│
├── governance/
│   ├── ledger.py
│   ├── credit.py
│   ├── projections.py
│   └── lifecycle.py
│
├── harness/
│   ├── protocol.py
│   ├── action_catalog.py
│   └── alfworld.py
│
├── traces/
│   ├── schema.py
│   └── store.py
│
└── system.py

experiments/
├── run_v3_train.py
├── run_v3_frozen_eval.py
├── run_v3_smoke.py
├── protocol.py
└── report.py
```

模块边界是强约束：

- `planner/` 不执行环境动作；
- `runtime/` 不创建长期 Skill/Tool；
- `harness/` 不修改知识生命周期；
- `validation/` 不向 Agent 泄漏 validator-only 信息；
- `evolution/` 不直接更新统计；
- `governance/` 不重新解释轨迹语义，只消费标准 EvidenceEvent。

---

# 6. 持久化知识对象

## 6.1 公共引用与状态

```python
@dataclass(frozen=True)
class SkillRef:
    logical_id: str
    version: str

@dataclass(frozen=True)
class ToolRef:
    tool_id: str
    version: str

class SkillStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SHADOW = "shadow"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"

class ToolStatus(str, Enum):
    DRAFT = "draft"
    ADMISSION_PENDING = "admission_pending"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PREFERRED = "preferred"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"
```

所有语义、接口、图结构、artifact body 的实质变化产生新版本。统计和 Evidence 不进入语义哈希。

## 6.1.1 Core 语义类型

```python
@dataclass(frozen=True)
class SemanticPredicate:
    predicate: str
    args: dict[str, BindingExpression | Any]
    cardinality: int = 1
    distinct_by: str = ""

class GraphEdgeType(str, Enum):
    DATA_FLOW = "data_flow"
    REQUIRES_SKILL = "requires_skill"
    NEXT = "next"

@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    edge_type: GraphEdgeType
    source_step: str
    target_step: str
    source_role: str = ""
    target_role: str = ""
    origin: str = ""  # existing_active | candidate_hint | planner_proposed | extractor_validated
    existing_edge_id: str = ""
    evidence_refs: tuple[str, ...] = ()

@dataclass(frozen=True)
class CompositeOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    binding_specs: dict[str, BindingExpression]
```

首版 Runtime `NEXT` 不作为 Planner 独立 edge list 的必填项；`control_sequence` 是其语义来源。Compiler 只把已提出的顺序物化为 NEXT 记录，不能改变顺序。


## 6.2 ParameterSpec

```python
@dataclass
class ParameterSpec:
    name: str
    semantic_type: str
    required: bool = True
    runtime_resolvable: bool = False
    required_resolution: str = "semantic"
    description: str = ""
```

`required_resolution` 可为：

```text
semantic          # apple / spreadsheet / user account
concrete          # apple_2 / sheet1 / concrete DOM node
relation_verified # 与其他参数之间存在当前有效关系证据
```

## 6.3 AbstractAtomicSkill

```python
@dataclass
class AbstractAtomicSkill:
    ref: SkillRef
    summary: str
    inputs: list[ParameterSpec]
    outputs: list[ParameterSpec]
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    validator_spec: dict
    failure_modes: list[dict]
    guideline: dict
    metadata: dict
    status: SkillStatus
```

固定语义：

- 描述一个独立、稳定、可验证的核心状态转移；
- 不保存 Tool body；
- 不保存 benchmark command regex；
- `runtime_resolvable=true` 仅表示允许 Runtime Agent 通过正常 Harness 交互解析，不表示代码可读取 hidden state；
- `outputs` 只有在 Atomic Validator 通过后才能发布。

## 6.4 Typed BindingExpression

Implementation mapping 不再使用 `$flow.xxx` 等任意字符串。

```python
class BindingExprKind(str, Enum):
    SKILL_INPUT = "skill_input"
    CONSTANT = "constant"
    DATA_FLOW = "data_flow"
    TOOL_OUTPUT = "tool_output"
    ADAPTER_TRANSFORM = "adapter_transform"

@dataclass(frozen=True)
class BindingExpression:
    kind: BindingExprKind
    source_role: str = ""
    source_step: str = ""
    constant: Any = None
    transform_id: str = ""
```

禁止：

- 任意 Python expression；
- 未注册 transform；
- Runtime 看到字符串 `$xxx` 再临时解释；
- 以字段名后缀猜测 source location。

## 6.5 GroundingConstraint

```python
class GroundingConstraintKind(str, Enum):
    ARGUMENT_EXISTS = "argument_exists"
    ARGUMENT_CONCRETE = "argument_concrete"
    HARNESS_AFFORDANCE = "harness_affordance"
    CURRENT_CONTEXT = "current_context"
    CUSTOM_ADAPTER = "custom_adapter"

@dataclass
class GroundingConstraint:
    constraint_id: str
    kind: GroundingConstraintKind
    action_type: str = ""
    argument_mapping: dict[str, BindingExpression] = field(default_factory=dict)
    required_resolution: str = "concrete"
    verifier_id: str = ""
```

Acquire 示例：

```yaml
grounding_constraints:
  - constraint_id: can_take_from_source
    kind: harness_affordance
    action_type: TAKE
    argument_mapping:
      object: {kind: skill_input, source_role: object}
      source: {kind: skill_input, source_role: source}
    required_resolution: relation_verified
```

## 6.6 ToolBinding 与 ImplementationAtom

```python
@dataclass
class ToolBinding:
    tool_ref: ToolRef
    role: str
    parameter_mapping: dict[str, BindingExpression]
    order: int = 0

@dataclass
class ImplementationAtom:
    ref: SkillRef
    abstract_ref: SkillRef
    tool_bindings: list[ToolBinding]
    grounding_constraints: list[GroundingConstraint]
    execution_policy: dict
    compatibility: dict
    quality: dict
    status: SkillStatus
```

ImplementationAtom 是 v3 的可执行调用边界。它负责：

- Skill input 到 Tool argument 的映射；
- 一个或多个 ToolAsset 的执行顺序；
- GroundingConstraint；
- Harness compatibility；
- execution policy；
- Implementation 层信用。

Agent 调用的是 **Implementation Invocation**，不是自由编排内部 ToolAsset。这样一个 Implementation 绑定多个 Tool 时仍有确定执行语义。

## 6.7 ToolAsset

```python
@dataclass
class ToolAsset:
    ref: ToolRef
    summary: str
    signature: dict
    interface: dict
    artifact_kind: str
    artifact: dict
    tests: list[dict]
    safety: dict
    provenance: dict
    metadata: dict
    status: ToolStatus
```

ToolAsset 保存可执行内容和局部接口，但不保存：

- Runtime Agent prompt；
- 完整 Atomic guideline；
- 完整 SkillGraph；
- ALFWorld-specific validation regex；
- 生命周期统计的事实源。

生命周期统计由 EvidenceLedger projection 生成。

## 6.8 CompositeSkill

为避免 Persistent DAG 的 AND/OR 和多路径歧义，一个 **可执行 Composite version 只保存一个 canonical control sequence**。

```python
@dataclass
class CompositeSkill:
    ref: SkillRef
    summary: str
    occurrences: list[CompositeOccurrence]
    control_sequence: list[str]
    data_edges: list[GraphEdge]
    dependency_edges: list[GraphEdge]
    goal_contract: TaskContract
    guideline: dict
    insight: dict
    validator_spec: dict
    metadata: dict
    status: SkillStatus
```

规则：

- `control_sequence` 是唯一 canonical 流程；编译器可据此物化 NEXT 控制边，但这只是序列化 Planner/Extractor 已明确提出的顺序，不属于代码创造新语义边；
- DataFlow/Dependency 可跨步骤形成 DAG；
- 同一 AtomicRef 可以出现多次；
- `goal_contract` 必须使用语义角色和参数占位，不保存 apple_2、cabinet_3 等具体 episode 实例；
- 不同可替代流程必须是不同 Composite logical_id/version/variant，不塞进同一个普通 DAG；
- 全局 SkillGraph 仍然可以是 DAG，因为多个 Composite、Atomic、Implementation、语义边、演化边共同构成图；
- v2 中包含 branch/parallel 的旧 Composite 不直接迁移为可执行资产，必须离线拆为独立 canonical variants 后重新验证。

## 6.9 Global SkillGraph

Global SkillGraph 允许：

```text
structural: implements / contains
semantic: equivalent / similar / alternative / conflict
knowledge: verified_data_flow / verified_dependency
lineage: derived_from / supersedes / split_from / merged_from
```

全局图可以是一般 DAG。Runtime 不直接执行全局图，而是由 Planner 生成 `RuntimeLinearPlan`。


---

## 6.10 可复用边的来源与状态

Planner 所谓 `existing edge` 不能只是“历史某个 JSON 中出现过”。可复用边必须满足：

```text
来自 ACTIVE Composite 的 canonical occurrence mapping
或
来自经过成功轨迹验证的 Active EdgePattern
```

首版不为 Edge 单独建立复杂生命周期。边的可信状态继承其来源 Composite：

```python
@dataclass(frozen=True)
class ExistingEdgeEvidence:
    edge_id: str
    source_composite_ref: str
    source_step_ref: str
    target_step_ref: str
    edge_type: str
    source_role: str
    target_role: str
    semantic_types: tuple[str, str]
    support_trace_ids: tuple[str, ...]
```

规则：

- Candidate Composite 中的边可用于 online hint，但不能标记为 `origin=existing_active`；
- Frozen eval 只复用 Active Composite 中的边；
- Planner 新提出的 temporary edge 只有在任务成功、Extractor E2 保留、代码验证通过并进入新的 Composite candidate 后才成为持久化知识；
- 一个 edge 的 role mapping 或 Atomic version 变化时必须产生新的 edge id；
- 代码不能把“Effect 名相似”直接登记为 existing edge。

---

# 7. TaskContract 与 Harness 双通道

## 7.1 TaskContract

Planner 生成的 Requirement 不能成为“任务完整性”的唯一权威。v3 必须建立独立 TaskContract。

```python
class ContractSource(str, Enum):
    BENCHMARK_FORMAL = "benchmark_formal"
    ADAPTER_DERIVED = "adapter_derived"
    PLANNER_PROPOSED = "planner_proposed"

@dataclass
class TaskContract:
    target_effects: list[SemanticPredicate]
    cardinality_constraints: list[dict]
    identity_constraints: list[dict]
    source: ContractSource
    confidence: float
    validator_id: str

class IdentityRelation(str, Enum):
    SAME_AS = "same_as"
    DISTINCT_FROM = "distinct_from"

@dataclass(frozen=True)
class IdentityConstraint:
    left_role: str
    relation: IdentityRelation
    right_role: str
    scope: str  # occurrence | task
```

`cardinality_constraints` 表达需要多少个独立 witness；`identity_constraints` 明确 Heat/Place 是否必须作用于 Acquire 的同一对象，以及 pick-two 是否必须使用不同实例。

权威等级：

```text
BENCHMARK_FORMAL > ADAPTER_DERIVED > PLANNER_PROPOSED
```

Planner plan 的“完整”至少需要：

1. 所有 required CapabilityRequirement 被覆盖；
2. 所有可用的权威 `TaskContract.target_effects` 被覆盖；
3. 所有 required node inputs 有来源，或明确标为 runtime_resolvable；
4. cardinality / identity 约束被写入 Runtime Plan；
5. control sequence、DataFlow 和 Dependency 全部合法。

当 Benchmark 没有 formal contract 时，结果必须区分：

```text
formal_contract_complete
planner_assessed_complete
```

不允许把 Planner 自己提出、自己覆盖的 Requirement 宣称为形式完备证明。

## 7.2 Policy-facing channel

Planner 和 Runtime Agent 只可见：

```text
task goal
policy observation
current HarnessActionCatalog
accepted / done / won
已验证且允许暴露的 Tool outputs
```

禁止提供：

```text
完整 hidden PDDL state
未观察到的对象位置
validator-only facts
Benchmark oracle 的具体答案
```

## 7.3 Validator-only channel

Validator 可以通过独立接口使用 Benchmark 内部可靠信息：

```python
class ValidatorChannel(Protocol):
    def snapshot(self) -> dict: ...
    def validate_atomic_effect(self, request: dict) -> ValidationResult: ...
    def validate_task_contract(self, contract: TaskContract) -> ValidationResult: ...
```

约束：

- 返回值只进入 Validator/Trace；
- 不写入 RuntimeBindingStore 作为参数 oracle；
- 不向 Planner/Runtime Agent展示 hidden fact；
- 允许用于确认 heated、cleaned、placed、cardinality、identity 等正式状态；
- 如果 Benchmark 无此能力，Adapter 必须明确声明 validation strength，而不是伪造确定性。

这使“决策不泄漏”和“验证可靠”同时成立。

---

# 8. Harness Action Interface

## 8.1 通用接口

```python
@dataclass(frozen=True)
class HarnessActionSpec:
    action_id: str
    revision: int
    action_type: str
    arguments: dict[str, Any]
    display_text: str
    raw_action: Any
    metadata: dict[str, Any]

@dataclass
class HarnessActionResult:
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    catalog: list[HarnessActionSpec]
    metadata: dict[str, Any]

class HarnessAdapter(Protocol):
    def reset(self, task) -> HarnessActionResult: ...
    def action_catalog(self) -> list[HarnessActionSpec]: ...
    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult: ...
    def task_contract(self, task) -> TaskContract: ...
    def validator_channel(self) -> ValidatorChannel: ...
    def compile_primitive(self, primitive, bindings) -> Any: ...
    def execute_primitive(self, primitive, bindings) -> HarnessActionResult: ...
```

## 8.2 Environment Action Tool

Runtime Agent 看到的原生工具：

```json
{
  "name": "environment_action",
  "description": "Execute one currently available environment action.",
  "parameters": {
    "type": "object",
    "required": ["action_id"],
    "properties": {
      "action_id": {
        "type": "string",
        "enum": ["a001", "a002", "a003"]
      }
    }
  }
}
```

Prompt 中提供紧凑 catalog：

```text
a001 | GO_TO   | cabinet_3

a002 | OPEN    | cabinet_3

a003 | EXAMINE | countertop_1
```

规则：

- Agent 不能提交任意 raw command，只能选当前 action_id；
- action_id 只在对应 `revision` 有效；
- revision 由 Adapter 维护：只要世界状态或 action catalog 发生变化就递增；accepted action 通常会递增，rejected action 若导致 catalog/observation 变化也必须递增；
- revision 更新后旧 action_id 全部失效；
- catalog 过大时由 Adapter 做分组/分页，但不能隐式删除完成当前节点所必需的合法动作；
- action catalog 大小、分页和检索结果写入 Trace。

## 8.3 ALFWorld ActionSpec

ALFWorld Adapter 在边界处唯一一次把 admissible command 解析成结构化 ActionSpec，例如：

```text
raw: take apple 2 from cabinet 3
```

```json
{
  "action_type": "TAKE",
  "arguments": {
    "object": "apple_2",
    "source": "cabinet_3"
  }
}
```

Core 不再解析 ALFWorld raw command；Core 只处理 `action_type + arguments`。

---

# 9. RuntimeBinding、Evidence 与 Grounding

## 9.1 BindingSource、状态和解析级别

```python
class BindingSource(str, Enum):
    TASK = "task"
    DATA_FLOW = "data_flow"
    TOOL_OUTPUT = "tool_output"
    HARNESS_EVIDENCE = "harness_evidence"
    AGENT_PROPOSED = "agent_proposed"
    UNRESOLVED = "unresolved"

class BindingStatus(str, Enum):
    GROUNDED = "grounded"
    PROPOSED = "proposed"
    UNRESOLVED = "unresolved"
    INVALIDATED = "invalidated"

class BindingResolution(str, Enum):
    SEMANTIC = "semantic"
    CONCRETE = "concrete"
    RELATION_VERIFIED = "relation_verified"

@dataclass
class RuntimeBinding:
    role: str
    value: Any
    semantic_type: str
    source: BindingSource
    status: BindingStatus
    resolution: BindingResolution
    evidence_refs: list[str]
    world_revision: int
```

例：

```text
Task 说 apple：
source=TASK, status=GROUNDED, resolution=SEMANTIC

Agent 从 ActionCatalog 发现 apple_2：
source=HARNESS_EVIDENCE, status=GROUNDED, resolution=CONCRETE

当前 TAKE(apple_2, cabinet_3) 可用：
object/source 对应 binding 可升级为 RELATION_VERIFIED
```

## 9.2 GroundingEvidence 生命周期

```python
class EvidenceStability(str, Enum):
    REVISION_SCOPED = "revision_scoped"
    STATE_SCOPED = "state_scoped"
    PERSISTENT = "persistent"

@dataclass
class GroundingEvidence:
    evidence_id: str
    evidence_type: str
    payload: dict
    source: str
    observed_at_revision: int
    valid_from_revision: int
    invalidated_at_revision: int | None
    stability: EvidenceStability
    action_id: str | None
```

`GroundingEvidenceStore`：

```python
class GroundingEvidenceStore:
    def replace_action_catalog(self, catalog, revision): ...
    def add_task_evidence(self, ...): ...
    def add_validated_tool_output(self, ...): ...
    def invalidate_after_transition(self, old_revision, new_revision): ...
    def match_constraint(self, constraint, bindings, revision): ...
```

关键规则：

- 当前 action affordance 默认 `REVISION_SCOPED`；
- Agent 拿走对象后，旧 `TAKE(object, source)` 证据自然失效；
- container open/closed、当前位置等状态证据必须随 transition 更新；
- ToolCall Gate 只能消费当前 revision 有效的 Evidence；
- v3 不维护一个只增不减的“对象永远在某位置”缓存。

## 9.3 Agent proposal 的处理

Agent ToolCall 参数首先写入临时 proposal：

```python
RuntimeBinding(
    source=BindingSource.AGENT_PROPOSED,
    status=BindingStatus.PROPOSED,
    resolution=BindingResolution.SEMANTIC,
)
```

Preflight 成功后才生成或升级正式 Binding。失败 proposal 不覆盖已有 grounded binding。

## 9.4 输出发布

Tool/Implementation 声称的 output 不直接进入 DataFlow。

发布条件：

```text
Implementation started
Atomic Validator passed
Output schema passed
Output value具有 Validator/Tool/Harness 证据
```

```python
binding_store.publish_validated_outputs(
    occurrence_id,
    outputs,
    validation_refs,
    revision,
)
```

---

## 9.5 RuntimeBindingStore

```python
class RuntimeBindingStore:
    def seed_task_bindings(self, task, contract, revision): ...
    def apply_data_flow(self, plan, current_step, validated_outputs): ...
    def propose_agent_arguments(self, occurrence, arguments, revision): ...
    def ground_from_evidence(self, proposal, constraints, evidence_store): ...
    def invalidate_revision(self, revision): ...
    def publish_validated_outputs(self, occurrence, outputs, validation_refs): ...
    def snapshot_for_node(self, occurrence) -> dict[str, RuntimeBinding]: ...
```

所有修改产生 `RuntimeBindingChange` 并进入 Trace。BindingStore 是 task-local，不进入长期 Skill bank。

---

# 10. Runtime 专用图 IR

## 10.1 RuntimeOccurrence

```python
@dataclass
class RuntimeOccurrence:
    step_id: str
    occurrence_id: str
    node_ref: SkillRef
    requirement_ids: list[str]
    binding_specs: dict[str, BindingExpression]
    implementation_candidates: list[SkillRef]
    expected_effects: list[SemanticPredicate]
    status: str = "not_started"
```

## 10.2 RuntimeLinearPlan

```python
@dataclass
class RuntimeLinearPlan:
    task_id: str
    source: str  # stored_composite | atomic_composition | full_dynamic
    source_composite_ref: str | None
    occurrences: list[RuntimeOccurrence]
    control_sequence: list[str]
    data_edges: list[GraphEdge]
    dependency_edges: list[GraphEdge]
    task_contract: TaskContract
    planner_audit: dict
```

## 10.3 控制链与数据 DAG 分离

控制链必须严格是：

```text
step_000 → step_001 → ... → step_N
```

DataFlow/Dependency 允许：

```text
step_000.output_a → step_003.input_a
step_001.output_b → step_003.input_b
step_002.effect_x  → step_003.precondition_x
```

验证器：

```python
def validate_runtime_plan(plan):
    by_id = {o.step_id: o for o in plan.occurrences}
    assert set(plan.control_sequence) == set(by_id)
    assert len(plan.control_sequence) == len(by_id)
    assert all_unique(plan.control_sequence)

    position = {step: i for i, step in enumerate(plan.control_sequence)}

    for edge in plan.data_edges + plan.dependency_edges:
        assert edge.source_step in by_id
        assert edge.target_step in by_id
        assert position[edge.source_step] < position[edge.target_step]

    assert each_target_input_has_one_authoritative_source(plan.data_edges)
    assert required_inputs_are_closed_or_runtime_resolvable(plan)
    assert task_contract_is_covered(plan)
```

不再对 DataFlow/Dependency 的总 indegree/outdegree 施加 `<=1`。

## 10.4 Runtime plan 不支持的控制结构

首版 Runtime 明确拒绝：

```text
BRANCH
PARALLEL
LOOP
RETRY
FALLBACK edge
```

Fallback 是 Runtime 状态机内部策略，不是 Planner 生成的图边。

---

# 11. Modern Agent 协议

## 11.1 Provider-independent AgentClient

```python
@dataclass
class NativeToolSpec:
    name: str
    description: str
    input_schema: dict

@dataclass
class NativeToolCall:
    call_id: str
    name: str
    arguments: dict

@dataclass
class AgentTurn:
    content: str
    tool_calls: list[NativeToolCall]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    latency_ms: float
    provider_metadata: dict

class AgentSession(Protocol):
    @property
    def session_id(self) -> str: ...

    def next_turn(
        self,
        user_input: str | None,
        *,
        tools: list[NativeToolSpec] | None = None,
        structured_output_schema: dict | None = None,
    ) -> AgentTurn: ...

    def submit_tool_result(
        self,
        call_id: str,
        result: dict,
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn: ...

    def snapshot(self) -> dict: ...
```

## 11.2 ToolCall 循环

固定协议：

```text
Assistant ToolCall
→ 校验 call_id / tool name / arguments
→ 执行或返回结构化 ToolCallError
→ submit_tool_result(call_id, result)
→ Assistant 下一轮
```

规则：

- 首版每轮只允许一个 ToolCall；多个 ToolCall fail closed 并允许一次协议修复，随后按当前模式失败；
- 返回纯正文且没有 structured output/tool call 时不解析正文动作：返回 `agent_protocol_no_action`，允许一次协议修复；
- Tool result 必须回到原 Session；
- Provider 无 server-side session 时，由 client 持有并重放消息历史；
- Session 可 snapshot 用于审计和同进程 provider retry；正式实验的跨进程 resume 默认只在 completed-task 边界进行。若 Harness 未实现确定性 episode replay，进程崩溃后不得从任意中间 world revision 继续；
- reasoning text 不进入系统逻辑；
- content 中的自然语言不用于解析 action 参数，参数只来自 native tool_calls/structured output。

## 11.3 预算配置

不再有 framework 固定 8192 上限，也不把 384K ceiling 当默认消费预算。

```yaml
llm:
  planner:
    max_completion_tokens: 32768
    max_visible_tokens: 8192
    reasoning_effort: high
    max_turns: 4
    max_total_tokens_per_task: 120000

  runtime:
    max_completion_tokens: 32768
    max_visible_tokens: 4096
    reasoning_effort: high
    max_turns_per_node: 12
    max_total_tokens_per_node: 80000

  extractor:
    max_completion_tokens: 131072
    max_visible_tokens: 32768
    reasoning_effort: high
    max_turns: 2
```

这些是可配置运行预算，不是模型能力上限。Provider 可支持更大的 context/completion ceiling；v3 adapter 不人为截断到旧值。

## 11.4 Session 类型

```text
PlannerSession
    固定 P1/P1R/P2/P2R；同一 task 内共享上下文

RuntimePreparationSession
    每个节点单独创建；可 environment_action 和 invoke_implementation

SeededSession
    Direct 失败后 fresh 创建；不继承失败 Tool body

DynamicTaskSession
    无图或 task rescue 时创建

ExtractorSession
    固定两轮；Turn 2 重发 canonical occurrences
```

---

# 12. Planner Pipeline

## 12.1 Planner 总流程

```text
P0   Complete Composite Retrieval
  ├─ hit → instantiate canonical sequence → validate → Runtime
  └─ miss

P1   Capability Requirement Proposal
     → Atomic Retrieval

P1R  Requirement Repair（coverage incomplete 才调用，最多一次）
     → Atomic Retrieval again

coverage 仍不完整
     → Full Dynamic

P2   Workflow Proposal
     → Runtime Plan Validation

P2R  Graph Repair（invalid 才调用，最多一次）

仍 invalid
     → Full Dynamic
```

Planner 最多四次 LLM turn：

```text
P1 / optional P1R / P2 / optional P2R
```

## 12.2 P0：完整 Composite Retrieval

只召回：

- 当前 train/frozen 模式允许的状态；
- goal contract 与 TaskContract 兼容；
- canonical control sequence 完整；
- occurrence refs 可用；
- Harness compatibility 满足；
- 不需要 planner 创建新边即可实例化；
- 允许 concrete 参数仍待 Runtime 解析，但能力集合、control sequence、DataFlow/Dependency 和 TaskContract 覆盖必须完整。

命中后：

```text
Task binding
→ instantiate RuntimeOccurrence
→ resolve已有 binding/data edge
→ validate RuntimeLinearPlan
```

P0 不调用 Planner LLM。P0 失败才进入 P1。

## 12.3 CapabilityRequirement

```python
@dataclass
class CapabilityRequirement:
    requirement_id: str
    intent: str
    desired_effects: list[SemanticPredicate]
    expected_inputs: list[ParameterSpec]
    expected_outputs: list[ParameterSpec]
    precondition_hints: list[SemanticPredicate]
    semantic_variants: list[str]
    required: bool
    rationale: str
```

P1 只看：

```text
Task goal
Policy observation summary
TaskContract
少量 Harness profile
```

不看完整 SkillGraph/ToolRepo。

## 12.4 Atomic Retrieval

每个 Requirement：

```text
semantic recall
→ effect contract filter
→ I/O compatibility
→ precondition compatibility
→ status/harness filter
→ historical utility rerank
```

默认保留 Top-3，最多 Top-5。Embedding 只用于召回，不能覆盖 Contract 不匹配。

输出：

```python
@dataclass
class RequirementSearchResult:
    requirement: CapabilityRequirement
    candidates: list[AtomicCandidate]
    covered: bool
    rejection_reasons: list[dict]
```

## 12.5 Related Composite Hint

若部分命中 Atomic 同属于一个已知 Composite，可把该 Composite 的摘要、canonical sequence、相关节点 contract 作为 **P1R 参考**。

禁止：

- 自动执行整张 Composite；
- 自动把未选节点加入 plan；
- 把 Composite 当成“任务肯定需要这些节点”的 oracle。

P1R 需要判断该 Composite 是否揭示：

- 原 Requirement 漏掉能力；
- 原 Requirement 拆分错误；
- 有另一条完整能力表达；
- Composite 与当前任务其实不相关。

## 12.6 P1R：Requirement Repair

只在 required coverage 不完整时调用一次。

输入：

```text
Task + TaskContract
requirements_v1
候选及 rejection reason
uncovered requirements
related composite hints
```

输出完整替代 `requirements_v2`，而不是 patch 文本。

重新检索后仍不完整：

```text
planner_outcome = uncovered_capability_requirement
plan = full_dynamic
```

## 12.7 Planner 错、Retriever 错与 Knowledge Gap

运行时只记录可证实诊断：

```text
planner_requirement_revised
    P1R 删除/合并/重写了 P1 requirement

retrieval_miss
    事后 full-registry contract audit 找到本可用节点，但在线召回漏掉

uncovered_capability_requirement
    P1R 后仍无合法候选；当前只能确认“未覆盖”，不能提前宣称知识库数学上不存在

confirmed_capability_gap
    Full Dynamic 成功后，Extractor 产生与未覆盖 Requirement contract-compatible 的新 Atomic
```

## 12.8 P2：唯一 Workflow Proposal

P2 同一 PlannerSession 接收：

```text
Task / TaskContract
最终 requirements
每个 requirement 的 Atomic candidates
候选之间已存在的 verified edges
相关 Composite hint（只作参考）
```

输出：

```python
@dataclass
class PlannerWorkflowProposal:
    steps: list[ProposedOccurrence]
    control_sequence: list[str]
    data_edges: list[ProposedEdge]
    dependency_edges: list[ProposedEdge]
    requirement_coverage: dict[str, list[str]]
```

必须满足：

- 只选候选 Atomic；
- 每个 required Requirement 至少被一个 occurrence 覆盖；
- 允许同一 Atomic 重复 occurrence；
- 只有一条 control sequence；该 sequence 本身必须标记来源：P0 为 `existing_composite_sequence`，P2 为 `planner_proposed_sequence`；
- Compiler 只能把该 sequence 物化为 NEXT 记录，不能重新排序；
- 不允许独立点；
- 新边必须显式标记 `origin=planner_proposed`；
- 已有边标记 `origin=existing` 并引用真实 edge id；
- DataFlow 可以跨步、多输入；
- 不允许 control branch/loop/parallel。

## 12.9 P2R：Graph Repair

Validator 失败时把以下内容交回同一 Session：

```text
原 proposal
精确 validation error codes
authoritative node contracts
existing edge ids
TaskContract uncovered effects
```

P2R 只能：

- 改 occurrence 选择；
- 改 occurrence 顺序；
- 改 planner-proposed edge；
- 删除错误 edge；
- 重复已有 Atomic occurrence。

不能：

- 发明新 Skill；
- 改写 Atomic contract；
- 伪造 existing edge；
- 增加第二条控制路径。

再失败：`graph_compilation_failed → Full Dynamic`。

## 12.10 Planner Validator

验证项：

```text
schema valid
all node refs exist
status usable
requirement coverage
TaskContract effect coverage
control sequence complete/unique
no disconnected occurrence
data/dependency edge forward-only
edge origin valid
existing edge id exists and matches
planner-proposed edge type/role compatible
one authoritative producer per target input
required input source closure
runtime_resolvable slots explicitly marked
identity/cardinality preserved
Harness compatibility
```

代码不得自动修正语义错误；只允许规范化 step_id、排序和序列化。


---

## 12.11 PlannerAudit

每个任务必须持久化完整 PlannerAudit：

```python
@dataclass
class PlannerAudit:
    composite_candidates: list[dict]
    composite_rejections: list[dict]
    selected_composite: str | None
    requirements_p1: list[dict]
    atomic_search_p1: dict
    related_composite_hints: list[dict]
    requirements_p1r: list[dict]
    atomic_search_p1r: dict
    workflow_p2: dict
    validation_p2: dict
    workflow_p2r: dict
    validation_p2r: dict
    final_outcome: str
    fallback_reason: str
```

Audit 用于区分 Planner、Retriever、Knowledge Gap 和 Graph validation；不进入 Runtime Prompt，除非是同一 PlannerSession 的 repair input。

---

# 13. Runtime Orchestrator 与节点状态机

## 13.1 TaskRuntimeContext

```python
@dataclass
class TaskRuntimeContext:
    task_id: str
    task_goal: str
    task_contract: TaskContract
    plan: RuntimeLinearPlan
    current_step_index: int
    world_revision: int

    observation: str
    action_catalog: list[HarnessActionSpec]
    action_history: list[dict]

    binding_store: RuntimeBindingStore
    evidence_store: GroundingEvidenceStore
    validated_outputs: dict[str, dict]

    global_action_budget: int
    used_actions: int
    token_budget: dict

    trace_builder: TraceBuilder
```

Context 是代码侧状态，不是一个无限增长的 LLM conversation。

## 13.2 节点状态机

```text
ENTER
  ↓
RESOLVE_BINDINGS
  ↓
LOAD_IMPLEMENTATIONS
  ├─ none → FRESH_SEEDED_SESSION
  └─ candidates
       ↓
AUTONOMOUS_PREFLIGHT
  ├─ valid → INVOKE_IMPLEMENTATION
  └─ no
       ↓
RUNTIME_PREPARATION_SESSION
       ├─ environment_action
       ├─ invoke_implementation
       ├─ Agent直接完成Atomic effect
       └─ exhausted / give_up

Implementation started
  ↓
VALIDATE_ATOMIC
  ├─ pass → PUBLISH_OUTPUTS → NEXT
  └─ fail → FRESH_SEEDED_SESSION

Implementation未启动或preflight失败两次
  ↓
FRESH_SEEDED_SESSION

Seeded
  ├─ Atomic pass → NEXT
  └─ fail → STOP PLAN / TASK FAILURE
```

## 13.3 NodeExecutionStatus

```python
class NodeExecutionStatus(str, Enum):
    NOT_STARTED = "not_started"
    ALREADY_SATISFIED = "already_satisfied"
    DIRECT_AUTONOMOUS_SUCCESS = "direct_autonomous_success"
    DIRECT_AGENT_PREPARED_SUCCESS = "direct_agent_prepared_success"
    AGENT_COMPLETED_BEFORE_INVOCATION = "agent_completed_before_invocation"
    SEEDED_SUCCESS = "seeded_success"
    FAILED_NOT_STARTED = "failed_not_started"
    DIRECT_FAILED = "direct_failed"
    SEEDED_FAILED = "seeded_failed"
    SKIPPED_GOAL_TERMINAL = "skipped_goal_terminal"
```

## 13.4 Already Satisfied

在节点开始前，Validator-only channel 如果能确认该 occurrence 的**具体、身份一致、cardinality 合法** Effect 已满足，可以标记 `ALREADY_SATISFIED`。

禁止仅凭 class-valued 参数或相同 predicate 名判断。

Already-satisfied：

- 不算 Direct；
- 不给 Tool/Implementation 使用证据；
- 给 Composite occurrence 正常完成证据；
- 可发布由 Validator 认证的 outputs。

---

# 14. Implementation Invocation

## 14.1 为什么 Agent 调用 Implementation 而不是任意 ToolAsset

一个 ImplementationAtom 可以绑定多个 ToolAsset。若直接把内部 Tool 全暴露给 Agent，会重新引入：

- Agent 编排内部 Tool；
- 多 Tool started 边界不清；
- 参数 mapping 重复；
- Tool/Implementation 信用难分；
- N:M 关系失去确定语义。

因此 Runtime Agent 看到的是由 Implementation 编译得到的 Learned Invocation Tool：

```text
invoke_impl_<logical_id>_<version>
```

它最终调用一个或多个真实 ToolAsset，但 Agent 不浏览 Tool Registry。

## 14.2 Invocation Spec

```python
@dataclass
class ImplementationInvocationSpec:
    name: str
    implementation_ref: SkillRef
    atomic_ref: SkillRef
    description: str
    input_schema: dict
    grounding_constraints: list[GroundingConstraint]
    tool_refs: list[ToolRef]
    execution_policy: dict
```

`input_schema` 由 Atomic inputs、Implementation mapping 和 Tool signature 编译得到。

## 14.3 Invocation Compiler

```python
class InvocationCompiler:
    def compile(
        self,
        atomic: AbstractAtomicSkill,
        implementation: ImplementationAtom,
        tools: list[ToolAsset],
        current_bindings: dict[str, RuntimeBinding],
    ) -> ImplementationInvocationSpec:
        ...
```

编译时必须验证：

- Tool refs 存在且状态可用；
- mapping AST 可解析；
- 所有 Tool required args 有来源；
- grounding constraints 可被当前 Harness profile 支持；
- execution policy 只包含支持的串行模式；
- output mapping 能映射回 Atomic outputs。

不合法 Implementation 不暴露给 Agent，并产生 `implementation_compile_rejected` 审计，不立即记长期负信用；只有被真实选择/执行的错误才进入 Evidence。

## 14.4 多 Tool 串行执行

首版 Implementation 只支持串行 ToolBindings：

```python
sorted(tool_bindings, key=lambda item: item.order)
```

不支持内部 parallel/branch/loop。每个 Tool 结果独立记录；Implementation 的 `started=true` 定义为第一个 ToolAsset 真正 started。

---

# 15. ToolCall Preflight

## 15.1 PreflightResult

```python
@dataclass
class ToolCallPreflightResult:
    passed: bool
    implementation_ref: str
    normalized_arguments: dict
    binding_updates: list[RuntimeBinding]
    matched_evidence_refs: list[str]
    failure_layer: str
    failure_code: str
    message: str
```

## 15.2 固定检查顺序

```text
1. native ToolCall name / call_id 合法
2. JSON Schema 合法
3. argument semantic type 合法
4. BindingExpression mapping 合法
5. required resolution 满足
6. GroundingConstraint 满足
7. 当前 world revision 证据有效
8. Harness/Implementation compatibility
9. safety / lifecycle status
```

任一步失败：

```text
ImplementationExecutionResult.started = false
ToolExecutionResult 不创建，或全部 started=false
```

## 15.3 参数关系验证

例如 Agent 提出：

```json
{"object": "apple_2", "source": "fridge_1"}
```

Acquire Implementation 要求：

```text
HARNESS_AFFORDANCE TAKE(object=apple_2, source=fridge_1)
```

当前 catalog 没有匹配 ActionSpec，则拒绝：

```json
{
  "error": "argument_relation_not_grounded",
  "message": "No current TAKE affordance supports object=apple_2 and source=fridge_1.",
  "repairable": true
}
```

Agent 可继续探索。错误 proposal 不写入 Tool/Implementation negative evidence。

## 15.4 未来动作与当前 affordance

Preflight 只验证 Implementation 明确声明的**调用入口关系**，不能要求多步 Tool 的所有未来动作在调用前都已 admissible。

ImplementationRunner 真启动后，每个 primitive 在其执行点通过 Harness 验证。

首版不对 `started=true` 的失败 Implementation 做同节点 Direct 重试；它立即进入 Fresh Seeded。只有 `started=false` 的 preflight/协议错误允许最多两次参数 repair。

---

# 16. ImplementationRunner 与 ToolRunner

## 16.1 执行结果

```python
@dataclass
class ToolExecutionResult:
    tool_ref: str
    preflight_passed: bool
    started: bool
    completed: bool
    state_changed: bool
    executed_step_count: int
    failure_step_index: int | None
    partial_effects: list[dict]
    output_candidates: dict
    before_revision: int
    after_revision: int
    failure_layer: str
    failure_code: str
    failure_message: str

@dataclass
class ImplementationExecutionResult:
    implementation_ref: str
    atomic_ref: str
    preflight_passed: bool
    started: bool
    completed: bool
    atomic_effect_passed: bool
    tool_results: list[ToolExecutionResult]
    realized_bindings: dict[str, RuntimeBinding]
    validated_outputs: dict
    before_state_ref: str
    after_state_ref: str
    failure_layer: str
    failure_code: str
```

## 16.2 Tool primitive execution

Interactive Tool artifact 推荐使用通用 Primitive IR：

```python
@dataclass
class PrimitiveToolStep:
    action_type: str
    argument_mapping: dict[str, BindingExpression]
```

ALFWorld Adapter 把 Primitive 编译为当前 raw command 或匹配当前 ActionSpec。

迁移期可以支持旧 action template，但必须先在 Adapter 内编译为 PrimitiveToolStep；Core 不解析模板文本。

## 16.3 部分副作用

Tool 执行到一半失败时：

- 保留真实 after revision、ActionRecord、partial effects；
- Seeded 从真实当前状态继续，不 reset；
- failure localizer 记录失败 step；
- 已成功执行的准备动作不伪装成 Tool completed；
- Tool 的 intrinsic failure 依据失败 primitive 和 Atomic effect 判断；
- 不尝试通用事务回滚。

## 16.4 Output validation

ImplementationRunner 完成后：

```text
Tool output candidates
→ output schema check
→ Atomic Validator
→ materialize validated Atomic outputs
→ BindingStore publish
```

Tool 返回的任意 dict 本身不是可信 DataFlow。

---

# 17. Runtime Agent 节点会话

## 17.1 暴露的工具

RuntimePreparationSession 只看到：

```text
1. environment_action
2. 当前节点 Top-K compatible Implementation Invocation（最多3个）
3. report_runtime_status(status=cannot_resolve|give_up)
```

节点 Effect 在每次 environment_action / Invocation 后由 Orchestrator 自动验证；Agent 不需要用自然语言声明成功。`report_runtime_status` 仅用于明确放弃，避免解析自由文本。

不看到：

```text
完整 Tool Registry
其他节点 Tool
被 suppress/retire 的资产
长期 EvidenceLedger
Validator-only facts
```

## 17.2 Agent 输入

```text
Task goal
当前 Atomic summary / inputs / outputs / effects
当前已认证 bindings
仍缺失的 required arguments
当前 observation
紧凑 action catalog
本节点相关 action history
剩余 node/global budget
允许的 Implementation Invocation schemas
```

## 17.3 两次 ToolCall repair

`ToolCall preflight` 失败可 repair 最多 2 次。这里指 Learned Invocation 的参数/Schema 修复，不限制 Environment Action Tool 的正常探索轮次；探索受 node action/token/turn budget 约束。

第三次 Learned Invocation preflight 仍失败：

```text
direct outcome = failed_not_started
→ fresh Seeded
```

## 17.4 Agent 在 Learned Invocation 前完成 Atomic

Runtime Agent 可能用 `environment_action` 直接产生当前 Atomic effect，例如直接执行 TAKE，而未调用 Learned Implementation。

此时：

```text
status = AGENT_COMPLETED_BEFORE_INVOCATION
Atomic positive
Runtime Agent positive
Implementation neutral
Tool neutral
Direct count = 0
```

这是 non-Direct Agent completion，不得伪装成 Tool reuse。

## 17.5 direct_autonomous

满足：

- 只有一个明确可用 Implementation，或存在稳定 Preferred；
- 所有 required args 达到要求的 resolution；
- GroundingConstraint 当前有效；
- preflight passed。

则代码直接调用 ImplementationRunner，零 LLM。

## 17.6 direct_agent_prepared

Agent 通过环境探索获得参数并成功调用 Learned Invocation，且 `started=true`。

环境探索成本计入 Runtime Agent；Implementation/Tool execution 单独计入 Direct。

---

# 18. Seeded 与 Full Dynamic

## 18.1 Fresh Seeded

触发：

- 无合法 Implementation，此时不启动无意义的 Direct preparation session；
- Direct preflight 两次 repair 后失败；
- Direct started 后失败；
- Agent preparation 主动放弃，但 Atomic 仍可由 Skill guideline 指导完成。

Fresh Seeded 输入：

```text
Task
当前 observation / action catalog
当前 Atomic contract
Atomic guideline
经过验证的 bindings
本节点相关真实 action history
```

不输入：

```text
失败 Tool body
Tool source code/template
失败 Implementation mapping
失败 Tool 参数建议
```

Seeded 只使用 Environment Action Tool。Seeded 成功说明 Atomic 语义大概率成立，但不证明现有 Implementation/Tool 正确。

## 18.2 Full Dynamic

触发：

1. P0/P1/P1R/P2/P2R 无法形成完整计划；
2. 所有计划 occurrence 完成，但 TaskContract/Benchmark 未完成。

Full Dynamic 只看 Task、当前环境和历史，不接收 Atomic guideline 或 Learned Invocation。

## 18.3 不存在节点级第三层 Dynamic

节点 Direct 失败后只有一次 Fresh Seeded。Seeded 失败则该计划失败。这样避免 Direct→Seeded→Dynamic 三个近似 Agent 路径重复消耗和归因混乱。

---

# 19. Task 提前终止与图完成

## 19.1 五个独立结果

```text
benchmark_success
node_contract_success
implementation_direct_success
graph_self_sufficient_success
graph_full_completion
```

## 19.2 Goal Early Terminal

当：

```text
Benchmark won
且 TaskContract validated
```

可以停止后续 occurrence。

未执行 occurrence：

```text
SKIPPED_GOAL_TERMINAL
```

只更新 Composite occurrence skip evidence，不惩罚对应 Atomic/Implementation/Tool。

## 19.3 Benchmark / Contract mismatch

Benchmark won 但 TaskContract 未通过：

```text
benchmark_goal_contract_mismatch
learning_eligible = false
```

不生成成功 Skill/Composite，不给未验证节点正证据；已经真实 started 的 Tool/Implementation attempt 证据仍按事实保存。

## 19.4 图结束但任务未完成

```text
所有 occurrence validated
但 TaskContract/Benchmark pending
→ Full Dynamic rescue
```

若 rescue 成功：

- 已有 Atomic/Implementation/Tool 按节点真实结果记正；
- Composite 记 `task_rescue_required` 结构负证据；
- rescue 片段进入 Extractor，用于发现缺失能力/边/流程版本。

---

# 20. Validation Engine

## 20.1 分层验证

```text
Tool-local
    executable/interface/primitive execution

Implementation
    mapping/constraint/policy/Tool sequence

Atomic
    declared core effect + output identity

Composite
    occurrence order、DataFlow、Dependency、self-sufficiency

TaskContract
    target effects/cardinality/identity

Benchmark
    official done/won/score
```

## 20.2 ValidationResult

```python
@dataclass
class ValidationResult:
    level: str
    passed: bool
    checks: dict[str, bool]
    failure_codes: list[str]
    messages: list[str]
    witness_refs: list[str]
    before_ref: str
    after_ref: str
```

## 20.3 Atomic identity consistency

同一 Atomic occurrence 的多个 Effect 必须使用一致实体 witness。例如：

```text
heated(egg_1)
at_location(egg_2, fridge_1)
```

不能满足同一 occurrence。Validator 必须使用 realized bindings 和 identity constraints。

## 20.4 Composite self-sufficiency

Composite success 不等于 Benchmark 最终 success。Composite Validator 需要分别输出：

```text
all_occurrences_valid
all_dataflow_realized
no_task_rescue_required
task_contract_covered_at_graph_boundary
```

---

# 21. Failure Attribution

## 21.1 FailureLayer

```python
class FailureLayer(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    PLANNER_REQUIREMENT = "planner_requirement"
    RETRIEVAL = "retrieval"
    PLANNER_GRAPH = "planner_graph"
    RUNTIME_AGENT = "runtime_agent"
    RUNTIME_BINDING = "runtime_binding"
    IMPLEMENTATION = "implementation"
    TOOL = "tool"
    ATOMIC = "atomic"
    DATA_FLOW = "data_flow"
    COMPOSITE = "composite"
    TASK_CONTRACT = "task_contract"
    BENCHMARK = "benchmark"
```

## 21.2 统一归因表

| 现象 | started | 主要责任 | 其他对象 |
|---|---:|---|---|
| ToolCall JSON/schema 错 | false | Runtime Agent | Impl/Tool neutral |
| 参数只是语义类、未 concrete | false | Runtime Agent/Binding | Impl/Tool neutral |
| 参数关系无当前 evidence | false | Runtime Agent/Binding | Impl/Tool neutral |
| BindingExpression 映射错误 | false | Implementation | Tool neutral |
| GroundingConstraint 错误/过严/过松 | false | Implementation | Tool neutral |
| Harness compatibility 错 | false | Implementation | Tool neutral |
| Tool primitive 真启动后 rejected | true | Tool | Implementation 记录 downstream failure |
| Tool 完成但 Atomic effect 未达到 | true | Tool + Implementation | Atomic unresolved |
| Direct 失败、Seeded 成功 | 依情况 | Atomic positive；失败实现层 negative | Composite neutral |
| Seeded 也失败、输入已正确 | n/a | Atomic candidate negative | 需足够证据才 suppress |
| 节点均成功但任务 rescue | n/a | Composite structural negative | 节点 positive |
| Planner 漏能力、P1R修正 | false | Planner diagnostic | Skill neutral |
| 在线 retrieval 漏掉已有 compatible skill | false | Retriever | Skill neutral |
| API/网络失败 | false | Infrastructure | 全部长期对象 neutral |

## 21.3 FailureEnvelope

```python
@dataclass
class FailureEnvelope:
    failure_id: str
    layer: FailureLayer
    code: str
    task_id: str
    trace_id: str
    occurrence_id: str
    attempt_id: str
    started: bool
    artifact_refs: list[str]
    evidence_refs: list[str]
    recoverable: bool
    message: str
```

所有 FailureProcessor、CreditAssigner、RepairProposal 都消费该结构，不重新解析日志字符串。

---

# 22. Repair Proposal

v3 不复制整份 Skill/Tool bank 建复杂 failure branch。使用局部、不可变 RepairProposal：

```python
@dataclass
class RepairProposal:
    proposal_id: str
    target_ref: str
    target_layer: str
    operation: str
    proposed_patch: dict
    source_failure_ids: list[str]
    status: str  # proposed | replaying | admitted | rejected
```

可能操作：

```text
revise_atomic_contract
revise_guideline
revise_implementation_mapping
revise_grounding_constraint
replace_tool_body
add_tool_test
specialize_tool
split_tool
revise_composite_sequence
remove_redundant_occurrence
insert_missing_occurrence
```

Executable 或 binding/policy 修复必须：

```text
局部 sandbox/replay
→ validation
→ admission
→ 新 CANDIDATE version
```

失败文本本身不能直接激活修改。


---

# 23. Trace Store v3

## 23.1 Trace 必须记录的层次

```text
TaskRecord
TaskContract
PlannerAudit
RuntimeLinearPlan
AgentSessionRecord
AgentTurnRecord
NativeToolCallRecord
EnvironmentActionRecord
ImplementationInvocationRecord
ToolExecutionRecord
RuntimeBindingChange
GroundingEvidenceChange
ValidationRecord
FailureEnvelope
EvidenceEventRef
LLMUsage
```

不得再通过事后 action 文本猜测：

- 哪一段属于 Tool；
- Tool 是否 started；
- 参数是谁提出的；
- 哪一次 Session 做了 Seeded；
- 哪条 output 进入了 DataFlow。

## 23.2 NativeToolCallRecord

```python
@dataclass
class NativeToolCallRecord:
    call_id: str
    session_id: str
    occurrence_id: str
    tool_name: str
    arguments: dict
    call_kind: str  # environment_action | implementation_invocation
    preflight_result: dict
    result_ref: str | None
    turn_index: int
```

## 23.3 RuntimeSpan

每个 Agent、Implementation、Tool、Task rescue 都有明确 span：

```python
@dataclass
class RuntimeSpan:
    span_id: str
    kind: str
    occurrence_id: str
    action_start: int
    action_end: int
    parent_span_id: str | None
    learnable: bool
```

Extractor 以真实状态转移为 authority；ToolCall/Span 只提供 intent、provenance 和边界提示。

---

# 24. 成功轨迹与 Extractor 双轮 Session

## 24.1 ExtractionPolicy

为兼顾效率，不对所有普通成功重复运行昂贵 Extractor。

必须运行 Extractor 的情况：

```text
Full Dynamic success
Task-level Dynamic rescue success
Seeded 产生现有 Atomic/Tool 未覆盖的新稳定动作片段
Composite 需要结构修订
成功轨迹包含未对齐 runtime span
定期抽样维护要求
```

可以跳过完整 Extractor、只写复用 Evidence 的情况：

```text
全部 occurrence 已由现有 Active Atomic 覆盖
Direct/Seeded 结果与现有 contract 一致
没有 task rescue
没有未知动作片段
没有新 failure/novelty signal
```

`ExtractionPolicy` 的决策写入 Trace。

## 24.2 Turn E1：Atomic Proposal

Extractor Session 接收：

```text
Task goal / TaskContract
结构化 Action/ToolCall/Span
每一步 before/after state summary
positive/negative effects
accepted/rejected
origin / mode / artifact refs
已知 Atomic contract 摘要
```

输出：

```python
@dataclass
class AtomicOccurrenceProposal:
    phase_id: str
    intent: str
    event_start: int
    event_end: int
    input_roles: dict
    output_roles: dict
    preconditions: list[SemanticPredicate]
    effects: list[SemanticPredicate]
    rationale: str
```

## 24.3 确定性验证

代码完成：

```text
event range 合法
不跨不兼容 RuntimeSpan
动作已 accepted
Effect 有状态/validator witness
observed fact 不伪装成 action effect
precondition 在 before state 有证据
参数来自真实 action/tool/binding
identity 一致
causal slice 能支撑 TaskContract 或复用能力
```

输出 canonical occurrences。

## 24.4 Turn E2：Composite Proposal

使用同一个真实 Session，但必须显式重新输入：

```text
The following canonical occurrences were validated by code and are authoritative.
Discard or correct any conflicting memory from the previous turn.
...
```

E2 只能使用 canonical occurrence ids、已验证 Skill refs 和已知 edge refs。

输出：

```text
canonical control sequence
existing edge reuse
new edge proposals
summary/guideline/insight candidate
```

代码再做 Composite validation。

## 24.5 Extractor 不消费 reasoning text

E1/E2 的结果只读取 structured output。Reasoning 仅计 Token。

---

# 25. Atomic / Implementation / Tool / Composite 生成

## 25.1 Atomic Alignment

Canonical occurrence 与已有 Atomic 对齐至少检查：

```text
Effect equivalence/compatibility
I/O role compatibility
precondition compatibility
validator compatibility
atomic boundary compatibility
```

Embedding 只召回，不做最终 merge。

结果：

```text
reuse existing
new candidate
new version
split candidate
merge candidate
```

## 25.2 Tool Compilation

从 canonical occurrence 的真实 span 提取：

- 可执行 primitive sequence；
- Tool signature；
- 参数 mapping；
- required context；
- replay case；
- source provenance。

Tool compiler 不能把 Runtime Agent 的探索 detour 无条件塞入 Tool；只有稳定、必要、可复用步骤进入 artifact。

## 25.3 Implementation Generation

Implementation 从 Atomic contract、Tool signature 和真实 binding provenance 构造：

```text
ToolBinding order
Typed BindingExpression
GroundingConstraint
Harness compatibility
execution policy
output mapping
```

Implementation 静态 closure 不通过则存 `SHADOW`，不暴露给 Runtime。

## 25.4 Admission

Tool/Implementation admission 至少验证：

```text
schema/interface
artifact safety
mapping closure
GroundingConstraint supported
source replay
Atomic effect replay
output mapping
version/hash dedup
```

Admission 通过后进入 Candidate。Candidate 可在 online train 被受控选择。

## 25.5 Composite Alignment

Composite 对齐：

```text
canonical control sequence
occurrence Atomic refs
DataFlow/Dependency roles
TaskContract
```

不同控制流程不强制合并。若目标相同但流程不同，保存为独立 variant/版本，并用 `ALTERNATIVE` 或 lineage 边关联。

## 25.6 长期进化操作

正式支持：

```text
Atomic: add / revise / split / merge / suppress / supersede
Implementation: add / revise_mapping / revise_constraint / specialize / supersede
Tool: discover / parameterize / update / generalize / specialize / merge / split / add_replay / supersede
Composite: add / revise_sequence / insert_occurrence / remove_occurrence / alternative / supersede
```

### Tool generalize

候选召回使用：

```text
signature + Primitive shape + bound Atomic effect + parameter families
```

LLM 只能提出 generalized signature/artifact；代码必须在全部 source replay cases 上验证，并重新 Admission。原 specialised Tool 保留，直到 generalized Tool 获得稳定 Active 证据。

### Tool specialize

只有当 failure cluster 在输入类型、Harness context 或参数约束上形成稳定子域时生成 specialised candidate。不得把一次 Agent 参数错误误判为 Tool specialization 需求。

### merge / split

- merge 要求行为、接口、Effect 和 replay 等价；
- split 要求一个 artifact 内存在两个独立 Effect、可独立复用边界或稳定不同 failure cluster；
- 所有操作生成新版本，不原地覆盖。

## 25.7 Guideline 与 Composite Insight

Atomic guideline 来源于多条兼容成功/失败证据，不因一次 Tool 成功自动改写。Composite insight 可聚合：

```text
常见参数解析策略
高频失败模式
有效节点顺序
冗余 occurrence
不同 Implementation 的适用条件
```

Insight 是 Planner/Seeded 的可选语义上下文，不具有越过 TaskContract、GroundingConstraint 和 Validator 的权力。维护周期批量更新，避免每题调用 LLM。

---

# 26. Full Dynamic 成功后的 Knowledge Gap 处理

Planner fallback audit 保存未覆盖 Requirement。

Full Dynamic 成功后：

```text
Extractor canonical Atomic
→ 与 uncovered requirements 做 contract match
```

分类：

```text
confirmed_capability_gap
    新 Atomic 正好覆盖未覆盖 Requirement

planner_requirement_error
    新成功流程不需要原 Requirement，说明 P1/P1R 分解有问题

retrieval_miss
    新流程复用了库中已有能力，但在线 Retriever 漏召回

novel_workflow_only
    Atomic 已有，缺的是流程/边/Composite
```

这些诊断进入 Planner/Retriever/Evolution 统计，不直接惩罚 Skill。

---

# 27. EvidenceLedger 与 exactly-once

## 27.1 Ledger 使用 SQLite

Artifact body 继续使用不可变 JSON 文件；Ledger、Registry index、status projection 使用 SQLite，获得事务、唯一索引和断点恢复。

```text
data/
├── artifacts/
│   ├── atomic/
│   ├── implementation/
│   ├── composite/
│   └── tools/
├── traces/
├── snapshots/
└── state/asg_v3.sqlite
```

## 27.2 EvidenceEvent

```python
class EvidenceEventType(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    SELECTED = "selected"
    PREFLIGHT_REJECTED = "preflight_rejected"
    EXECUTION_STARTED = "execution_started"
    DIRECT_SUCCESS = "direct_success"
    DIRECT_FAILURE = "direct_failure"
    AGENT_NODE_SUCCESS = "agent_node_success"
    SEEDED_SUCCESS = "seeded_success"
    SEEDED_FAILURE = "seeded_failure"
    SELF_SUFFICIENT_SUCCESS = "self_sufficient_success"
    TASK_RESCUE_REQUIRED = "task_rescue_required"
    GOAL_TERMINAL_SKIPPED = "goal_terminal_skipped"
    CONTRACT_MISMATCH = "contract_mismatch"
    SUPERSEDED = "superseded"

@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    schema_version: int
    task_id: str
    trace_id: str
    occurrence_id: str
    attempt_id: str
    sequence_no: int
    artifact_ref: str
    artifact_kind: str
    event: EvidenceEventType
    failure_layer: str
    confidence: float
    metadata: dict
```

数据库唯一约束：

```text
UNIQUE(event_id)
UNIQUE(trace_id, attempt_id, artifact_ref, event, sequence_no)
```

## 27.3 事务顺序

```text
1. Trace 原子落盘
2. 在一个 SQLite transaction 内 append EvidenceEvent
3. 更新 projection checkpoint
4. LifecycleProjection 消费未处理 event
5. Registry status/recommended pointer 更新
```

进程中断后可从 Ledger 重建所有 statistics/status。Runtime 不直接 `counter += 1`。

## 27.4 CreditAssigner

```python
class CreditAssigner:
    def assign(self, trace: TraceRecord) -> list[EvidenceEvent]: ...
```

它依据：

```text
started
failure_layer
final Atomic validation
Seeded rescue
Task rescue
Goal terminal
Infrastructure retry
```

生成事件。它不调用 LLM。

---

# 28. 生命周期设计

阈值是配置化实验超参数，不是不可改变的语义常量。状态转移和证据类型是设计不变量。

## 28.1 Online/Frozen 可用状态

```text
Online train：Candidate / Active / Preferred 可用
Frozen eval：Active / Preferred 可用
Shadow/Suppressed/Retired：不可用
```

Candidate 排序低于 Active；只有在无可靠 Active、显式探索配额或候选明显更匹配时使用，避免 Candidate 永远拿不到证据或反过来过度干扰。默认实现可使用固定 seed 的 10%~20% candidate exploration quota；该 quota 只在 online train 生效并写入 Planner/Runtime audit。

## 28.2 AbstractAtomicSkill

```text
DRAFT
  ↓ deterministic occurrence validation
CANDIDATE
  ↓ independent compatible evidence
ACTIVE
  ↓ repeated semantic failure after impl/tool excluded
SUPPRESSED
  ↓ stable replacement supersedes
RETIRED
```

默认激活政策：

```text
至少2条独立 canonical success occurrences
或
1条source + 1次后续成功 reuse
```

Atomic 负证据只有在：

- 输入已正确；
- Implementation/Tool failure 已排除或 Seeded 仍失败；
- Atomic effect 本身反复不可达；

时成立。

## 28.3 ImplementationAtom

```text
DRAFT
  ↓ static mapping/constraint/compatibility validation
CANDIDATE
  ↓ real started Direct success
ACTIVE
  ↓ repeated intrinsic mapping/policy failure
SUPPRESSED
  ↓ stable replacement
RETIRED
```

默认激活政策：至少 2 次独立 started Direct success；Candidate 在线可用，因此不会形成门控死锁。

Implementation 主要负证据：

```text
mapping_error
constraint_error
compatibility_error
policy_error
```

Agent proposal 错误不属于 Implementation negative。

## 28.4 ToolAsset

```text
DRAFT → ADMISSION_PENDING → CANDIDATE → ACTIVE → PREFERRED
                       └────→ SHADOW
ACTIVE/PREFERRED → SUPPRESSED → RETIRED
```

默认激活政策：

```text
Admission passed
+ 至少2次独立 started execution success
+ intrinsic failures 不超过配置阈值
```

Preferred：

```text
至少5次 started execution
Beta/Wilson reliability lower bound 达标
同等功能候选中 cost/utility 更优
```

Suppress 只消费 started=true 后的 intrinsic Tool evidence。

Retire 主要依赖：

```text
稳定 Active replacement supersedes
或
强负证据 + 有可靠替代
```

“不常使用”不能单独 retire。

## 28.5 CompositeSkill

```text
DRAFT
  ↓ canonical workflow/contract validation
CANDIDATE
  ↓ independent graph self-sufficient successes
ACTIVE
  ↓ repeated task rescue / structural mismatch
SUPPRESSED
  ↓ stable replacement
RETIRED
```

默认激活政策：至少 2 个独立 task 的 graph self-sufficient success。

Composite 统计分两层：

```text
Composite global
Composite occurrence-level
```

Occurrence 保存：

```text
selected
executed
already_satisfied
skipped_goal_terminal
direct_autonomous
direct_agent_prepared
agent_completed_before_invocation
seeded_success
failure
```

## 28.6 冗余节点修订

当某 occurrence 持续高比例 `skipped_goal_terminal`，且移除后 TaskContract 仍完整：

```text
生成新的 Composite candidate version
→ replay/evaluation
→ Active 后 supersede 旧版本
```

不 retire 对应全局 Atomic。

---

# 29. 生命周期 Projection 的简洁实现

不为四类实体分别写四套散乱 counter 更新器。

```python
class LifecycleProjection:
    def consume(self, events: list[EvidenceEvent]) -> None: ...
    def stats(self, artifact_ref: str) -> ArtifactStats: ...

class LifecyclePolicy:
    def review_atomic(self, ref, stats): ...
    def review_implementation(self, ref, stats): ...
    def review_tool(self, ref, stats): ...
    def review_composite(self, ref, stats): ...
```

公共统计：

```text
independent_task_count
selected_count
started_count
success_count
failure_count
consecutive_intrinsic_failures
self_sufficient_success_count
task_rescue_count
cost_sum
latency_sum
```

维护不必每题全库扫描。建议：

```text
每个任务只更新受影响 artifact projection
每5~10个成功 episode 执行一次 candidate promotion / preferred / supersede review
```

---

# 30. 成本与效率设计

## 30.1 避免不必要 LLM

```text
P0 Composite 命中：Planner 0 LLM
所有参数闭合：Runtime 0 LLM
DataFlow 后续节点闭合：Runtime 0 LLM
普通稳定 Direct 成功：Extractor 可跳过
Requirement coverage 完整：不调用 P1R
Graph validation 通过：不调用 P2R
```

## 30.2 Context 压缩

Runtime 节点 Session 只携带：

- 当前节点 contract；
- 当前已认证 binding；
- 相关 action history；
- 当前 action catalog；
- 当前允许的最多 3 个 Invocation；
- 剩余预算。

不发送完整任务历史、完整 graph、完整 Tool body。

## 30.3 Action catalog 压缩

- schema 中只使用短 action_id；
- display catalog 结构化、去重；
- revision 更新后只发送变化或新 catalog；
- catalog 过大时可先用 `list_action_groups` / `expand_action_group`，但 ALFWorld 首版优先直接 enum；
- catalog token 单独统计。

## 30.4 Extractor 策略

只有 novelty/rescue/dynamic/repair 相关成功触发完整两轮 Extractor；稳定复用仅写 Evidence，从而控制训练成本。

## 30.5 Batch maintenance

Tool generalization、duplicate detection、Composite redundancy review 在维护周期批处理，不阻塞每个 task 的关键路径。

---

# 31. Budget 规范

## 31.1 环境动作预算

```text
Global task action budget
    所有 Environment Action、Tool primitive 共用

Node budget
    当前 occurrence 的 Direct preparation + Implementation + Seeded 共用

Full Dynamic / Task rescue
    使用剩余 global budget
```

不允许 fallback 重置已消耗动作。

## 31.2 LLM 预算

```text
Planner task token/turn budget
Runtime per-node token/turn budget
Seeded per-node token/turn budget
Dynamic task token/turn budget
Extractor trace token/turn budget
```

Preflight rejection：

- 不消耗 environment action；
- 消耗 LLM call/token；
- 两次 learned invocation repair 上限。

## 31.3 失败分类

预算耗尽必须区分：

```text
planner_token_budget_exhausted
runtime_node_token_budget_exhausted
runtime_node_action_budget_exhausted
episode_action_budget_exhausted
extractor_token_budget_exhausted
```

禁止全部覆盖成 `max_steps`。

---

# 32. Token Accounting

统一字段：

```text
prompt_tokens
completion_tokens
total_tokens
reasoning_tokens
call_count
latency_ms
```

Bucket：

```text
planner_p1
planner_p1_repair
planner_p2
planner_p2_repair
runtime_preparation
runtime_seeded
runtime_dynamic
extractor_e1
extractor_e2
evolution_repair
```

每个 Session/Turn 都产生 usage event。Episode：

```text
sum(real buckets) + unattributed == episode_total
```

正式实验要求：

```text
token_mismatch = 0
unattributed_total_tokens = 0
```

Reasoning tokens 是否包含于 completion 由 provider metadata 说明，不能再次相加推导 total。

---

# 33. 错误分类

建议固定 code：

```text
planner_requirement_invalid
planner_requirement_uncovered
planner_requirement_repair_failed
retrieval_miss
planner_graph_invalid
planner_graph_repair_failed
runtime_agent_schema_error
runtime_agent_multiple_tool_calls
runtime_binding_unresolved
runtime_binding_not_concrete
runtime_relation_not_grounded
stale_grounding_evidence
implementation_compile_rejected
implementation_mapping_error
implementation_constraint_error
implementation_compatibility_error
implementation_invocation_failed
tool_preflight_rejected
tool_primitive_rejected
tool_execution_error
tool_output_schema_error
atomic_effect_violation
data_flow_error
composite_self_sufficiency_failure
task_contract_mismatch
benchmark_goal_contract_mismatch
benchmark_failure
action_cycle
llm_error
infrastructure_failure
```

每个失败必须有：

```text
failure_layer
failure_code
started
attempt_id
occurrence_id
artifact_refs
evidence_refs
```

---

# 34. Planner / Runtime / Extractor 伪代码

## 34.1 Planner

```python
def build_plan(task, system) -> RuntimeLinearPlan:
    contract = system.harness.task_contract(task)

    for composite in system.composite_retriever.retrieve_complete(task, contract):
        plan = system.plan_compiler.from_composite(task, contract, composite)
        report = system.plan_validator.validate(plan)
        if report.passed:
            return plan

    session = system.planner_agent.new_session(task, contract)
    requirements = session.propose_requirements()
    search = system.atomic_retriever.retrieve(requirements)

    if not search.full_coverage:
        hints = system.related_composite.find(search)
        requirements = session.repair_requirements(
            requirements=requirements,
            search=search,
            related_composites=hints,
        )
        search = system.atomic_retriever.retrieve(requirements)

    if not search.full_coverage:
        return RuntimeLinearPlan.full_dynamic(
            task, contract,
            reason="planner_requirement_uncovered",
            audit=build_planner_audit(...),
        )

    proposal = session.propose_workflow(
        requirements=requirements,
        candidates=search.candidates,
        existing_edges=system.graph_store.existing_edges(search.refs),
    )
    plan = system.plan_compiler.compile(proposal, task, contract)
    report = system.plan_validator.validate(plan)

    if not report.passed:
        proposal = session.repair_workflow(proposal, report)
        plan = system.plan_compiler.compile(proposal, task, contract)
        report = system.plan_validator.validate(plan)

    if not report.passed:
        return RuntimeLinearPlan.full_dynamic(
            task, contract,
            reason="planner_graph_repair_failed",
            audit=build_planner_audit(...),
        )

    return plan
```

## 34.2 Runtime

```python
def run_task(task, system):
    plan = system.planner.build_plan(task)

    if plan.source == "full_dynamic":
        result = system.dynamic_agent.run(task)
        return system.finalizer.finish(task, plan, result)

    ctx = TaskRuntimeContext.create(task, plan, system.harness)

    for step_id in plan.control_sequence:
        occurrence = plan.occurrence(step_id)
        node = ctx.trace_builder.start_node(occurrence)

        ctx.binding_store.resolve_from_task_and_dataflow(occurrence, ctx)

        already = system.validator.validate_already_satisfied(
            occurrence, ctx
        )
        if already.passed:
            node.finish_already_satisfied(already)
            ctx.binding_store.publish_from_validation(occurrence, already)
            continue

        invocations = system.invocation_compiler.compile_candidates(
            occurrence, ctx
        )

        if not invocations:
            direct = system.node_executor.not_started(
                occurrence,
                failure_code="no_compatible_implementation",
            )
        else:
            direct = system.node_executor.try_autonomous(
                occurrence, invocations, ctx
            )
            if direct is None:
                direct = system.node_executor.run_preparation_session(
                    occurrence=occurrence,
                    invocations=invocations,
                    ctx=ctx,
                    learned_call_repair_limit=2,
                )

        if direct.atomic_effect_passed:
            node.finish_direct_or_agent(direct)
            ctx.binding_store.publish_validated_outputs(
                occurrence, direct.validated_outputs
            )
        else:
            seeded = system.node_executor.run_seeded_fresh(
                occurrence, ctx
            )
            if not seeded.atomic_effect_passed:
                node.finish_failure(direct, seeded)
                break
            node.finish_seeded(direct, seeded)
            ctx.binding_store.publish_validated_outputs(
                occurrence, seeded.validated_outputs
            )

        terminal = system.validator.validate_task_contract(ctx)
        if system.harness.benchmark_won() and terminal.passed:
            ctx.mark_remaining_goal_terminal()
            break

    if ctx.plan_boundary_reached() and not ctx.task_complete():
        rescue = system.dynamic_agent.run(task, resume=ctx)
        ctx.attach_task_rescue(rescue)

    trace = ctx.trace_builder.finish()
    return system.finalizer.finish(task, plan, trace)
```

## 34.3 Finalizer

```python
def finish(task, plan, trace):
    trace_store.save_atomic(trace)

    events = credit_assigner.assign(trace)
    ledger.append_transaction(events)
    lifecycle_projection.consume_new_events()

    if trace.infrastructure_failure:
        pass  # 不产生长期 repair/negative learning
    elif trace.benchmark_success and trace.learning_eligible:
        if extraction_policy.should_extract(trace):
            evolution.process_success(trace)
    elif trace.benchmark_success and not trace.learning_eligible:
        evolution.process_anomaly(trace)
    else:
        evolution.process_failure(trace)

    if not config.freeze_skills:
        maintenance.run_if_due()
    return episode_summary(trace)
```

成功、失败、contract mismatch 和 infrastructure failure 是四个明确分支，不能依赖一个模糊 `learning_eligible` 条件嵌套。

## 34.4 Extractor

```python
def process_success(trace, system):
    session = system.extractor_agent.new_session(trace)

    proposal = session.propose_atomics(
        system.trace_normalizer.build(trace)
    )

    canonical = system.atomicizer.validate_and_canonicalize(
        proposal, trace
    )

    composite_proposal = session.propose_composite(
        authoritative_occurrences=canonical,
        existing_edges=system.graph_store.existing_edges_for(canonical),
    )

    composite = system.composite_builder.validate_and_build(
        composite_proposal, canonical, trace.task_contract
    )

    atomic_refs = system.aligner.align_atomics(canonical)
    compiled = system.tool_compiler.compile(canonical, trace)
    admitted = system.admission.admit(compiled)
    implementation_refs = system.aligner.align_implementations(
        atomic_refs, admitted, canonical
    )
    composite_ref = system.aligner.align_composite(composite)

    system.ledger.append_transaction(
        system.credit_assigner.assign_evolution(
            trace, atomic_refs, implementation_refs,
            admitted.tool_refs, composite_ref
        )
    )
```

---

# 35. 持久化与冻结

## 35.1 Artifact Store

每个版本内容不可变：

```text
artifacts/atomic/<id>/<version>.json
artifacts/implementation/<id>/<version>.json
artifacts/composite/<id>/<version>.json
artifacts/tools/<id>/<version>/tool.json
artifacts/tools/<id>/<version>/artifact.*
```

SQLite 保存：

```text
artifact_index
recommended_pointers
status_projection
evidence_events
projection_checkpoints
run_manifests
```

### 35.1.1 最小 SQLite 表

```sql
CREATE TABLE artifact_index (
    artifact_ref TEXT PRIMARY KEY,
    artifact_kind TEXT NOT NULL,
    logical_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    file_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);

CREATE TABLE recommended_pointers (
    logical_id TEXT PRIMARY KEY,
    artifact_ref TEXT NOT NULL
);

CREATE TABLE evidence_events (
    event_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    artifact_ref TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    event_type TEXT NOT NULL,
    failure_layer TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(trace_id, attempt_id, artifact_ref, event_type, sequence_no)
);

CREATE TABLE lifecycle_projection (
    artifact_ref TEXT PRIMARY KEY,
    projection_json TEXT NOT NULL,
    last_event_rowid INTEGER NOT NULL
);

CREATE TABLE run_manifests (
    run_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    task_manifest_hash TEXT NOT NULL,
    code_commit TEXT NOT NULL,
    state TEXT NOT NULL
);
```

Artifact 文件写入采用 temp file + fsync + rename；随后在 SQLite transaction 中注册 index。索引存在但文件不存在、或文件 hash 不符时启动必须 fail closed。


## 35.1.2 运行状态与断点续跑

正式 runner 只在任务边界承诺可恢复：

```text
pending → running → completed | infrastructure_failed | task_failed
```

每完成一题，先原子写 Trace，再提交 Ledger/结果并更新 manifest。`--resume` 只跳过 manifest 中已 completed 且 task signature、config hash、code commit、knowledge milestone 都一致的任务。

任务中途基础设施失败：

- 丢弃该 attempt 的长期 Evidence；
- 从相同任务初始状态重跑，次数配置化；
- 不把 API failure 记为 Skill/Tool failure；
- 不默认跨进程恢复 ALFWorld 中间 episode。


## 35.2 Frozen Evaluation

Frozen snapshot 包含：

```text
Artifact files
Registry pointers/status
必要的 query index
不包含 train-time mutable maintenance queue
```

评估前后：

```text
knowledge_digest_before == knowledge_digest_after
```

Test 阶段禁止：

- 新建/修改 Skill/Tool；
- 更新 status/utility；
- 写 EvidenceLedger 中的长期训练事件；
- task 间传播新知识。

允许写：

- 独立 eval Trace；
- eval metrics；
- 临时 RuntimeBinding/Evidence。

---

# 36. ALFWorld v3 适配规范

## 36.1 保留的适配职责

```text
task loading / deterministic identity
raw goal / observation
admissible commands
ActionSpec parser
environment_action(action_id)
accepted / done / won
TaskContract 构造
validator-only effect verification
Primitive IR 编译/执行
source replay / admission
```

## 36.2 明确删除的 v2 正常路径

```text
framework discover_object_location()
if role.endswith("_location") 自动扫描
Core 对 ALFWorld action 做 regex 参数解析
Core 维护 ALFWorld object_at 作为 Runtime oracle
Runtime 直接读取 PDDL hidden state 找参数
```

## 36.3 Agent 探索与证据认证

Agent 决定：

```text
去哪里
打开什么
检查什么
何时调用 Implementation
```

Adapter 认证：

```text
当前 action_id 是否有效
执行是否 accepted
新的 ActionCatalog 暴露了哪些 concrete entity/relations
```

Validator-only channel 可用官方/内部状态验证最终 Effect，但不将 hidden facts提供给 Agent 或 BindingStore。

## 36.4 Action catalog 关系证据

例如 catalog 暴露：

```text
TAKE(object=apple_2, source=cabinet_3)
```

EvidenceStore 可生成当前 revision 的：

```text
entity concrete: apple_2
entity concrete: cabinet_3
relation affordance: TAKE(apple_2, cabinet_3)
```

Agent 若提交 fridge_1，Preflight 无匹配 evidence，Implementation 不启动。

---

# 37. 跨 Benchmark 边界

v3 Core 依赖的是：

```text
TaskContract
HarnessActionCatalog
Environment Action execution
GroundingConstraint matching
ValidatorChannel
Primitive compiler
```

新 Benchmark 需要实现自己的 Adapter/Harness profile，但不修改：

```text
Planner Pipeline
RuntimeLinearPlan
AgentSession
RuntimeBinding/Evidence
Implementation Invocation
CreditAssigner
Lifecycle
Extractor 主协议
```

若某环境没有 admissible action 列表，可由 Harness 提供：

- 可操作 UI element catalog；
- function/action schema；
- safe action validator；
- action rejection feedback。

不能为了“通用”而把所有环境强制成 ALFWorld 文本命令。

---

# 38. v2 代码复用准则

## 38.1 可以移植

经过 contract 审计后可移植：

```text
Ref/version/content hash
不可变 JSON artifact storage
部分 predicate/effect matching
ALFWorld task loading 与 deterministic mapping
LLM usage accounting 思路
Trace atomic write
Admission sandbox 基础
```

## 38.2 不能直接沿用

```text
v2 AtomicPlanner 主流程
RuntimeDataFlowSynthesizer 自动写边
system.py::_run_env_nodes 大型状态机
framework object location discovery
字符串 `$flow/$inputs` binding
v2 Direct Gate/Selector/Resolver 串联逻辑
Runtime 各处直接改统计
stateless generate() 伪多轮 Extractor
v2 bank status/utility
```

## 38.3 不做 v2 bank migration

v3 loader 读取到 schema_version < 3：

```text
reject_for_runtime
```

可提供只读审计/可视化工具，但正式 v3 Runtime 不调用旧资产。

---

# 39. 实现顺序：垂直闭环，而非模块堆积

本文全部属于 v3 必须交付范围。实现按以下垂直切片进行，是为了每一步都有可运行闭环，而不是推迟设计。

## Slice 0：仓库重置与 Schema

交付：

```text
v2 tag
v3 package skeleton
Core refs/contracts/results
SQLite state DB
ArtifactStore
TraceStore
config loader
```

验收：能创建空 bank、注册不可变 artifact、写/读 Trace、Ledger 幂等写入。

## Slice 1：Modern Agent + Harness Full Dynamic

交付：

```text
AgentSession native toolcall
ALFWorld ActionCatalog + environment_action
TaskRuntimeContext
Full Dynamic Agent
Token accounting
policy/validator dual channel
```

验收：无 Skill 情况可完整执行并保存 ToolCall/Action/Token Trace。

## Slice 2：单 Atomic / 单 Implementation / Direct

交付：

```text
Atomic/Implementation/Tool Registry
BindingExpression
InvocationCompiler
Preflight
ImplementationRunner/ToolRunner
Atomic Validator
direct_autonomous
```

验收：参数闭合时零 LLM 调用 Learned Implementation，started/credit 正确。

## Slice 3：Agent-prepared Direct + Seeded

交付：

```text
RuntimeBinding/Evidence revision
GroundingConstraint
节点 RuntimePreparationSession
2次 preflight repair
agent_completed_before_invocation
Fresh Seeded
partial side-effect result
```

验收：错误参数不会启动 Tool；正确探索后可调用；Direct 失败后 Seeded fresh 成功，信用分层正确。

## Slice 4：Planner Pipeline

交付：

```text
P0/P1/P1R/P2/P2R
TaskContract coverage
Related Composite Hint
RuntimeLinearPlan compiler/validator
control sequence 与 data/dependency 分离
```

验收：完整 Composite 零 LLM；Atomic 组合能生成唯一流程；覆盖不全 Full Dynamic。

## Slice 5：Extractor 与 Evolution

交付：

```text
真实双轮 ExtractorSession
canonical Atomic validation
Tool Compiler/Admission
Implementation/Composite alignment
ExtractionPolicy
RepairProposal
```

验收：Full Dynamic success 能生成候选；E2 只使用 canonical occurrence；候选可进入 online Runtime。

## Slice 6：Evidence / Lifecycle / Freeze

交付：

```text
CreditAssigner
LifecycleProjection/Policy
candidate online use
promotion/suppress/supersede
frozen snapshot/hash
```

验收：同一 Trace 重放不重复记账；冻结评估不改变知识 digest。

## Slice 7：实验入口与报告

交付：

```text
train runner
frozen eval runner
manifest/resume
per-agent token report
artifact growth/lifecycle report
```

验收：可按固定任务清单 train→freeze→test，并支持逐题断点续跑。

---

# 40. 最小必要验证策略

v3 不在初始开发阶段预先构造庞大测试矩阵。采用“一套确定性全链 smoke + 一套真实环境 smoke + 发现问题后补对应回归测试”的原则。

## 40.1 静态 Preflight

一个命令：

```bash
python -m experiments.run_v3_smoke --preflight
```

检查：

```text
imports/config/API source
SQLite schema
empty bank
Harness initialization
Agent provider supports native toolcall
Task manifest
Artifact/Trace output writable
```

## 40.2 Deterministic no-API full-chain smoke

使用 Fake Agent + Fake Harness，最少 4 个 episode 覆盖：

1. Full Dynamic success → Extractor → Atomic/Tool/Implementation/Composite Candidate；
2. 下一任务 direct_autonomous success；
3. Agent 错参数 preflight rejected、Tool neutral、Fresh Seeded success；
4. 图节点均成功但 task rescue，Composite structural negative，新能力候选生成。

验收不是单个 helper test，而是最终检查：

```text
Trace 完整
Ledger 无重复
token 守恒
started credit 正确
validated outputs 可 DataFlow
candidate 可在线使用
frozen digest 不变
```

## 40.3 Real ALFWorld smoke

为了效率，不跑 6×1 大矩阵作为开发门禁。建议固定一个易成功 profile：

```text
Cold learning：3 个 pick_and_place_simple train tasks
Warm reuse：2 个未见过的同类 train/eval tasks
可选 multi-node：1 个 heat-then-place task
```

必须证明：

```text
Environment Action Tool 真运行
ActionCatalog revision 正常
至少1条成功 trace
至少1个 Candidate 产物
至少1次 Learned Invocation preflight
至少1次 started Direct 或明确的 agent_completed_before_invocation
无 graph/ledger/token schema error
```

若没有任何成功轨迹，不能用“程序没崩”宣称 smoke 通过。

## 40.4 回归测试策略

后续每发现一个真实 bug：

```text
先定位 root cause/invariant
再增加一个最小 deterministic regression case
```

不为尚未出现的所有组合预先编写海量单元测试；但已经修过的根因必须永久保留回归测试。

---

# 41. 正式实验定位

v3 是独立研究，不要求把增益归因于 FlowEvo 的某个组件。

正式比较回答：

> 不同 Skill/Tool 进化系统在相同任务协议、模型/API 配置和 train/test split 下，完整系统的成功率、训练成本、测试成本、复用率和知识增长如何？

## 41.1 比较条件

至少：

```text
pure_dynamic
四种外部 Skill 进化方法
AtomicSkillGraph v3
```

需要训练的方法：

```text
120 train
→ freeze
→ 60 heldout test
```

Pure Dynamic 只跑 test，除非额外报告 train diagnostic。

## 41.2 公平性

统一：

```text
任务 manifest
train/test split
模型/provider/base_url/API key source
max environment steps
结果 Schema
Token 统计字段
```

但不强迫每种研究方法采用 v3 Runtime；若原方法的 Runtime 是方法的一部分，应保留并在 adaptation manifest 中注明。论文结论是 end-to-end 方法比较，不做不成立的组件因果归因。

## 41.3 v3 自身指标

```text
benchmark success
graph self-sufficient success
direct autonomous rate
direct agent-prepared rate
agent-completed-before-invocation rate
seeded success rate
full dynamic rate
Task rescue rate
Atomic/Implementation/Tool/Composite lifecycle
Planner requirement/graph repair rate
confirmed capability gap
Token/latency/cost per solved task
```

---

# 42. 配置草案

```yaml
schema_version: 3

data_dir: data_v3

llm:
  provider: openai_compatible
  base_url: "..."
  model: "..."
  api_key_env: MODEL_API_KEY

  planner:
    reasoning_effort: high
    max_completion_tokens: 32768
    max_visible_tokens: 8192
    max_turns: 4
    max_total_tokens_per_task: 120000

  runtime:
    reasoning_effort: high
    max_completion_tokens: 32768
    max_visible_tokens: 4096
    max_turns_per_node: 12
    max_total_tokens_per_node: 80000
    learned_toolcall_repair_limit: 2

  extractor:
    reasoning_effort: high
    max_completion_tokens: 131072
    max_visible_tokens: 32768
    max_turns: 2

planner:
  composite_top_k: 5
  atomic_top_k_per_requirement: 3
  max_atomic_top_k: 5
  requirement_repair_limit: 1
  graph_repair_limit: 1
  max_runtime_occurrences: 16

runtime:
  global_action_budget: 100
  node_action_budget: 35
  node_agent_turn_budget: 12
  max_implementation_candidates: 3

lifecycle:
  maintenance_interval_successes: 5
  atomic_active_independent_support: 2
  implementation_active_direct_successes: 2
  tool_active_started_successes: 2
  tool_preferred_min_started: 5
  composite_active_self_sufficient_successes: 2

extraction:
  extract_full_dynamic_success: true
  extract_task_rescue_success: true
  extract_novel_seeded_success: true
  skip_stable_direct_success: true
```

所有阈值可通过实验配置调整；Schema、状态转移和信用规则不可通过配置绕过。

---

# 43. ALFWorld 完整示例

任务：

```text
Heat an apple and put it in the fridge.
```

## 43.1 Planner

P0 无完整 Composite。

P1：

```text
R1 obtain target object
R2 change target object to heated state
R3 place same object at destination
```

Atomic Retrieval 返回 Acquire / Heat / Place。

P2 输出：

```text
control_sequence:
Acquire_1 → Heat_1 → Place_1

data_edges:
Acquire_1.held_object → Heat_1.object
Heat_1.object → Place_1.object

dependency_edges:
Acquire_1.agent_holds → Heat_1.requires_holds
Heat_1.object_heated → Place_1.preceded_by_heating
```

## 43.2 Acquire

初始 Task binding：

```text
object = apple
resolution = semantic
source = task
```

参数不 concrete，无法 autonomous Direct。

RuntimePreparationSession：

```text
environment_action(a001=go to cabinet 3)
environment_action(a007=open cabinet 3)
```

新 catalog 出现：

```text
TAKE(object=apple_2, source=cabinet_3)
```

EvidenceStore 认证 concrete + relation。

Agent 调：

```json
{
  "name": "invoke_impl_acquire_v2",
  "arguments": {
    "object": "apple_2",
    "source": "cabinet_3"
  }
}
```

Preflight passed，Implementation started。Atomic Validator 验证 agent holds apple_2，发布：

```text
held_object = apple_2
```

状态：`direct_agent_prepared`。

## 43.3 Heat

DataFlow 已得到 `object=apple_2`。若 Implementation 只要求 object，且当前 context/constraint 满足：

```text
zero LLM → direct_autonomous
```

若需要具体 heating_station，Runtime Agent 探索后调用 Invocation。

## 43.4 Place

DataFlow 得到同一 `apple_2`；Task 提供 destination semantic `fridge`。Agent 需要探索 concrete fridge instance，认证后调用 Place Implementation。

## 43.5 错误参数

Agent 错提：

```text
apple_2 + fridge_1 作为 Acquire source
```

当前 catalog 无 `TAKE(apple_2, fridge_1)`，Preflight 拒绝：

```text
started=false
Runtime Agent/Binding negative
Implementation/Tool neutral
```

Agent 可继续探索，最多两次 Learned Invocation repair。

---

# 44. 主要风险与固定缓解

## 44.1 Planner Requirement 漏能力

缓解：TaskContract 独立覆盖、P1R、Dynamic 后验 gap 诊断。

## 44.2 全局图 DAG 与执行流程混淆

缓解：Composite version 只有一个 canonical control sequence；Runtime 只执行 control sequence；DataFlow/Dependency 独立验证。

## 44.3 Stale grounding

缓解：world revision、revision-scoped evidence、catalog replace/invalidate。

## 44.4 Agent 绕过 Learned Tool

缓解：允许但明确分类 `agent_completed_before_invocation`，不计 Direct/Tool reuse。

## 44.5 多 Tool Implementation 信用混乱

缓解：Agent 调 Implementation Invocation；ImplementationRunner 固定串行执行内部 Tool；每个 Tool 单独结果。

## 44.6 Validator 信息泄漏

缓解：policy-facing 与 validator-only channel 强隔离；Validator result 不进入 Agent binding。

## 44.7 Candidate 门控死锁

缓解：Candidate online 可用、frozen 不可用；小样本激活阈值配置化；started-based evidence。

## 44.8 Ledger 重复记账

缓解：SQLite transaction、event_id/unique constraint、projection checkpoint、可重建 projection。

## 44.9 Token 失控

缓解：P0/zero-LLM fast path、bounded turns、per-node budget、Extractor policy、compact action ids。

---

# 45. Design Freeze 决策清单

1. v3 是独立研究和独立 Runtime，不再以 FlowEvo 扩展为定位。
2. v3 整仓重构；v2 通过 tag 保留，仅选择性移植纯工具代码。
3. 正式 v3 bank 从空库训练，不加载 v2 bank。
4. Global SkillGraph 可以是 DAG。
5. 一个可执行 Composite version 只有一条 canonical control sequence。
6. Runtime plan 的控制序列严格线性；DataFlow/Dependency 允许 forward DAG 和多输入。
7. P0 只使用完整 Composite。
8. Atomic 覆盖不完整时最多一次 Requirement Repair，仍不完整则 Full Dynamic。
9. 部分命中节点所属 Composite 只作为 Requirement Repair 参考。
10. 新临时边只能由 Planner Agent 提议；代码只验证。
11. Graph Repair 最多一次。
12. Planner completeness 同时依赖 Requirement coverage 与独立 TaskContract。
13. Runtime 按节点执行，不重新规划全图。
14. Runtime Agent 每节点小 Session；Direct 失败后 Seeded fresh Session。
15. Environment Action Tool 使用当前 revision 的 action_id enum。
16. Agent 只看到当前节点最多3个 Implementation Invocation。
17. Agent 调用 Implementation，而不是自由编排内部 ToolAsset。
18. Implementation 首版只支持串行 ToolBindings。
19. ToolCall Schema/grounding repair 最多2次。
20. Agent proposal 不能直接执行；必须通过 Binding/Evidence/Constraint preflight。
21. Binding 区分 semantic、concrete、relation_verified。
22. GroundingEvidence 带 revision 并可失效。
23. Tool/Implementation 只有 started=true 后才获得执行信用。
24. Agent 在 Invocation 前完成节点不算 Direct。
25. 节点只有 Direct→Fresh Seeded；没有第三个节点级 Dynamic。
26. 无计划或图后缺口才使用 Full Dynamic。
27. Tool output 只有 Atomic Validator 通过后才发布到 DataFlow。
28. Policy-facing 与 validator-only channel 严格隔离。
29. Extractor 是同一真实 Session 的 E1/E2；E2 重发 canonical occurrences。
30. 稳定普通成功可跳过 Extractor，仅写 Evidence。
31. 所有长期信用进入 SQLite EvidenceLedger。
32. Runtime/Evolution 不直接修改长期 counter/status。
33. Atomic、Implementation、Tool、Composite 四类实体独立生命周期。
34. Candidate online 可用，Frozen 只用 Active/Preferred。
35. Goal early terminal 更新 occurrence skip，不惩罚全局 Atomic。
36. Composite 冗余通过新版本 supersede，不原地删节点。
37. 退役主要依赖可靠替代或强负面证据，不依赖低使用频率。
38. 初始开发只要求 deterministic full-chain smoke + 小规模真实 ALFWorld smoke。
39. 真实 bug 出现后按 root cause 增加最小 regression test，不预先构造巨大测试矩阵。
40. 正式实验比较完整进化方法，不声称必须从 FlowEvo 组件中隔离归因。

---

# 46. 一句话定义

> **AtomicSkillGraph v3.0 是一个独立的、基于 native ToolCall 的集中式自进化 Agent 系统：它用 Planner Agent 从任务中生成能力需求并构建唯一的原子能力流程，用 Runtime Agent 在真实 Harness 中探索并认证具体参数，通过 Implementation Invocation 执行可版本化 Tool，用分层 Validator 和 append-only EvidenceLedger 精确归因，再从成功轨迹中以同一 Extractor Session 发现新的 Atomic、Implementation、Tool 与 Composite，使任务执行逐渐从反复 LLM 规划转化为可验证、可复用、低 Token 的结构化能力调用。**

