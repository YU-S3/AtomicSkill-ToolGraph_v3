# SkillCompiler：可纠正规划与已验证能力接管——修改实施文档

**版本：v3.1-CF3**  
**日期：2026-10-05**  
**修改基线：`YU-S3/AtomicSkill-ToolGraph_v3@2528f53b222dfbb79a54e37ad7514eb3f160fd59`**  
**对应最近一次真实运行：`d046c1de3c7488a30bbe1edc1e0397b695041ba4`，空 Bank 的 seed42 / 12 Train + 6 Val。**  
**状态：已确认设计边界的待实施规范；不是已提交补丁，也不是补丁后的效果报告。**

> **解除的是 Planner 临时猜测造成的错误执行约束，不是解除经过实际验证的 Skill / Program 的接管能力。**
>
> 正确匹配的可用能力继续程序优先、连续执行和自动交接；Planner 可以用已有可用节点和 Program 新组合工作流，组合后立即执行。不得另设“整个新工作流必须先经两题验证”的启动门槛。出现冲突时，修订本题选择、参数或交接，不修改已保存能力的定义，不把错误结果记为成功。

本文是本轮新增缺陷与控制策略的唯一实施单，取代最近几轮尚未落地的 Planner 松绑、节点接口和恢复策略建议。CF2 已经落地的功能是当前生产基线，不重新实施或回退；第 13 节列出必须保留的行为。无需从聊天中拼接新要求。任何标为“新增”的字段或函数均是本轮要求，不表示当前仓库已经实现。

---

## 0. 冻结决策与禁止误读

### 0.1 三件事必须分清

| 内容 | 本轮含义 | 是否允许自动接管 |
|---|---|---|
| Planner 临时提出的步骤、位置、顺序和跨节点接线 | 本题待检验的策略，不是已经发生的事实 | 可以组织执行，但冲突时可以纠正；不能覆盖真实反馈 |
| 已有 Skill 的文字指导 | 参考知识；仅有文字不等于存在可用程序 | 不因“被引用”自动取得程序执行权或强制输出权 |
| 明确选中的 Skill / Program 执行能力 | 使用资产原有目标范围和真实 I/O；可用 Program 按现有规则执行 | **保留程序优先、准备后自动接续、多动作内部执行** |

“程序在 Train 中通过过实际测试”不等于“当前 Planner 一定把它放在了正确的位置”，也不等于“任何新组合都保证成功”。系统保留尝试新组合的能力；只在实际接口、权限、预算和反馈层处理问题，不增加语义证明器或新的 LLM 审查角色。

### 0.2 用户已明确批准的恢复规则

1. 局部解除不适用绑定、修改未执行参数、修正交接，不占用整图重规划额度；仍消耗实际 Runtime 决策及原预算。
2. 整图重规划仍最多一次。显式 `revise_plan` 与系统触发的整图重新规划共用这一次额度，不各给一次。
3. 整图重规划用尽后，允许一次从当前真实状态退出失效计划、进入剩余任务 Dynamic。不得 reset 环境、token、动作额度、attempt 或原始结果。
4. Dynamic 退路仍可选择可用 Program 执行；不得把“退出错误图”实现成“禁用全部已学能力”。
5. 同一错误不能无限局部修改；明确重复失败使用原有两次上限，并进入本文件的有界恢复顺序。
6. 不把已验证 Program 全部降为 advisory，不取消正常自动执行，不增加每节点/每动作 LLM 确认。

### 0.3 保持不变

| 项目 | 固定要求 |
|---|---|
| 主线 | 现有 empirical：Skill / Implementation / Program / Workflow，受限 Python，普通值传递 |
| 验证 | 不恢复 witness、owner、历史同值硬依赖、完整因果证明或四层证书 |
| Program 可用 | 保留同版本、两个不同物理 Train 任务的实际正向条件；`ok` 不等于官方正确 |
| 模型 | 当前 DeepSeek 服务、配置与各角色 high 不变；不单独修改命名或视觉锁 |
| Builder 恢复 | 首次 32,768；length 且无可用提交时恢复上限 65,536；首次加恢复最多两次；结构/代码/试用修复共享一次机会 |
| 数据与评价 | 公共 manifest、原 Train 顺序、42/43/44、评分器与题目预算不变；不导入人工 Bank |
| Frozen | 长期资产、使用统计、排序、资格均只读；本题计划和工作记忆允许修正 |
| 单轮 QA | 一次正式 solver，不增加 Planner、节点确认、二次解题或默认 Builder |
| 计费 | 所有模型请求、修复、隔离试用如实保存；恢复响应不重复计费；未知费用不填零 |
| 成本/质量 | 每个支持的 benchmark 冻结执行争取均值 ≤50k、目标上界 60k，同时争取最强 baseline 的主指标；不是把任务硬截断到 50k |

---

## 1. 本轮修改依据与实际范围

### 1.1 已核实的生产路径问题

| 编号 | 已核实问题 | 必须修改的生产位置 |
|---|---|---|
| P01 | `skill_id` 同时控制参考指导、自动路线和 `complete_node` 输出；局部节点可能被整题 Skill 约束 | `empirical/contracts.py`、`prompts.py`、`bank.py`、`executor.py` |
| P02 | `dynamic=true` 没有阻止 `Bank.routes()` 根据 Skill 自动进入 Program | `Bank.routes()`、统一节点解析 |
| P03 | Planner 只看到当前可用工具，未必看到后续操作的稳定语义；执行节点又允许自由文本改写范围 | `Planner.plan()`、节点表示和提示词 |
| P04 | 动态生产者的输出与后继字段不一致；运行返回混用计划 `literal/from` 包装 | `validate_workflow()`、`ValueStore`、Runtime 材料和完成处理 |
| P05 | `complete_node` 校验失败直接 `continue`，绕过统一失败处理 | `Executor.run()` |
| P06 | 相同请求内容可能命中已处理失败的旧响应，导致跨逻辑决策反复 recovered_response | `EmpiricalSystem.agent()`、`TaskCheckpoint`、Executor 决策提交点 |
| P07 | 被环境接受但没有新进展的循环未进入有效纠偏，尤其错误计划与反馈冲突时 | `Executor.run()`、`TaskContext`、通用进展观察 |
| P08 | Builder 有函数名说明，但生成程序仍可能误读真实工具目录和返回封装 | `Learner._realize()`、程序 ABI 说明、`program_worker.py`、真实试用 |

最近报告记录 Train 11/12、Val 5/6；同时存在真实程序接管：后续 Train 原环境调用 Program 五次完成任务，Val 的 Heat 与双 creditcard 由 Program 终止成功。故本轮既要修复错误计划路径，也必须保护已经成功的接管路径。[S1]

### 1.2 不把旧问题重复列成新缺陷

MOVE/INVENTORY 状态更新、ToolSpec、结果引用、Office 纯读批量、文件增量发布、位置单位、finish-only、待实现作业、隔离试用题目映射复用已有代码。对这些做受影响回归，不重新写另一套实现。[S2]

旧 12+6 是开发诊断，不是正式 Train120/Test134 的效果证明。不得因本文件修改而重标旧分数、改写原冻结库或向原结果追加新版本任务。

---

## 2. 最小节点表示：参考与执行绑定分开，但不削弱接管

### 2.1 新增节点字段 `execution_mode`

节点统一保留 `id / args / after`，新增：

```text
execution_mode = dynamic | skill | program
reference_skill_ids = []       # 可选，仅供参考，最多3项
purpose                       # 可选：该节点在父任务中的用途，不是新的能力定义
```

分别约定：

| 模式 | 必需字段 | 当前目标与接口来源 | 自动路线 |
|---|---|---|---|
| `dynamic` | `goal` | 本题结果目标；交接字段由真实下游引用生成 | 不从 reference_skill_ids 自动取路线；Agent 仍可显式调用准备/可用程序 |
| `skill` | `skill_id` | **所选 Skill 的原有完整能力范围及 I/O** | 查询该 Skill 的实际 Implementation 路线，优先所有 ready usable Program |
| `program` | `program_id` | **所选 Program 的真实 I/O 和关联能力范围** | 指定 Program；当前状态不适用时可局部解除本题绑定 |

必须互斥：`dynamic` 不携带执行用 `skill_id/program_id`；`skill` 不携带 `program_id`；`program` 不携带执行用 `skill_id`。Program 的关联 Skill 由 Bank 查得，不能由 Planner 伪造新关联。参考 Skill 放在 `reference_skill_ids`，不参与路线资格和完成检查。

`goal` 在 dynamic 模式下是本题结果目标。在 skill/program 模式下，Planner 不另写一个可以缩小或扩大能力范围的 `goal`；内部运行视图由代码读取资产生成目标。上层意图写 `purpose`，它不能覆盖资产真实接口或用户任务。

**这不是禁止 Planner 选择错误能力的形式保证。** 它消除“同一节点同时声称是局部搜索和完整任务求解”的表示歧义；选择是否合适仍由本题实际执行检验，发现不适用时按第 6 节修正。

### 2.2 一份节点解析函数供所有调用方共用

在 `empirical/contracts.py` 新增纯解析函数：

```python
def resolve_node_interface(node: dict, bank) -> dict:
    """解析执行模式、目标、I/O与资产引用；不执行工具，不调用模型，不改变Bank。"""
```

返回至少：

```text
execution_mode
node_goal
purpose
reference_skill_ids
bound_skill_id / bound_program_id
input_schema / output_schema（执行绑定时来自实际资产）
args
after
```

Planner 验证、Runtime 材料、路线查找、节点完成和 Learner 工作流保存必须共用这一解释。不得分别手写“skill_id 代表什么”的逻辑。

`Bank.routes(node)`：

- `dynamic` 返回空的**自动**路线；reference_skill_ids 不授予自动调用权。
- `skill` 保留现有 Implementation 查询、usable 过滤及排序。
- `program` 返回明确指定的合法路线。
- Train 的 candidate 手动尝试权限维持原规则；不因为是新组合自动晋升，也不默认在 Frozen 放行 candidate。
- `program_options()` 仍向动态节点展示可显式选择的相关 Program；动态节点不是没有工具能力。

### 2.3 没有 Program 的 Skill 不伪称已验证程序

Skill 文本存在但无 usable Program 时，可以继续参考其指导，或在明确的 skill 模式中由 Runtime 执行该完整能力；后者若由 Agent 完成，origin=agent，不能记录为 Program 接管。实际能力不适用时允许解除到 dynamic。

仅仅存在 Skill ID、Planner 填 `validated=true`、Workflow 存在于 Bank，均不能生成新的可用资格。禁止新增由模型声明的“已验证”布尔值。

---

## 3. 保留三种接管路径

### 3.1 已有可用 Program 直接接管

```text
Planner选择能力/程序并绑定本题参数
→ 读取真实Program输入接口
→ 参数ready、状态/权限允许
→ 直接运行Program
→ 校验实际返回结构、必要局部结果
→ 发布真实结果
→ 继续后继；无需再问一次Runtime
```

Program 内部普通循环、分支和多步调用继续执行，不把内部每个动作拆回 LLM。除真实错误、环境终止、预算或隔离限制外，不因“工作流是新组合”而打断。

### 3.2 Planner 用已知节点新组合工作流

允许：

```text
已可用Find Program
→ 实际对象/资源结果
→ 已可用Transform Program
→ 实际输出
→ 已可用Deliver/Submit Program
```

只做普通结构与字段可引用检查，不要求整个新图先经历两个 Train 任务，不增加语义裁判、预执行证明或每节点 LLM 确认。新连线的正确性不能自动继承单个程序的测试结论，但这**不是阻止尝试执行的理由**。

一段不成立时，修该段的本题绑定/交接，保留其他尚可用节点及已获得结果；不得默认全图退回逐步 Agent。

### 3.3 复用已保存的 Workflow

保留一次 `mode=select` 选择与参数实例化；结构无需每次重写。新保存的 Workflow 显式包含 execution_mode。正确的 Skill/Program 绑定、节点间顺序和显式数据依赖继续有效。

“该 Workflow 曾成功”不是永久无条件执行保证；改变参数或组合后仍使用当前接口、权限和实际反馈。新鲜状态缺失或反馈矛盾时，允许修正本题实例，不原地改动长期 Workflow。

不新增 Workflow 生命周期或整图证书表。现有 Program 的测试记录继续决定其可用性；整体流程表现只如实记录。

---

## 4. Planner：结果目标、真实接口和可撤销假设

### 4.1 规划材料

`Planner.plan()` 中分别提供：

```text
original_task：原始目标与公开输入
related_interfaces：相关Skill/Workflow
program_options：现有共用程序卡片
tool_definitions：Adapter稳定工具说明（名称、语义、参数、结果与单位）
current_tools：此刻实际可用操作及参数
public_state：当前公开状态
completed_results / feedback：仅恢复时提供
```

`tool_definitions` 来自已有 Adapter / ToolSpec，不另写一套工具说明。只提供公开且被授权的工具；完整工具能力不等于当前就允许调用，实际派发始终重新读取当前接口。

不发送全部 Program 源码或全Bank轨迹。必要输入、视觉内容、题目数量、身份与输出位置不得为了缩短材料而丢失。

### 4.2 Planner输出要求

- 动态节点描述“需要得到什么结果”，操作提示只能作为可更改建议。
- 真正调用 Skill/Program 时，引用真实资产ID及参数；能力目标、I/O由代码解析，不允许 Planner 一边选整题能力、一边重写成局部搜索。
- `purpose` 不作为新的硬前置条件，不进入官方评分。
- 当前未知的值使用 unresolved 或生产者引用，不因为计划写了一个地点就把它当已发现事实。
- 稳定工具语义优先于 Planner 对现实世界操作的猜测；Runtime 可以采取达到同一目标的合法不同操作。
- 不做自然语言“语义完全相等”校验；不引入新的 LLM reviewer。

### 4.3 选择与组合的结构

保留顶层 `mode=select|compose`。新 Schema 根仍为 object，避免重现 Provider 根级 schema 问题。

新节点最小示例：

```json
{
  "id": "find",
  "execution_mode": "dynamic",
  "goal": "找到满足当前查询的资源，并返回它的可读取位置",
  "reference_skill_ids": ["skill_example_reference"],
  "args": {"query": {"task": "query"}}
}
```

```json
{
  "id": "transform",
  "execution_mode": "program",
  "program_id": "program_example_usable",
  "purpose": "处理前驱已经找到的资源",
  "args": {"resource": {"from": "find", "field": "resource_id"}},
  "after": ["find"]
}
```

示例ID不是可用生产资产。运行时仍要求ID真实存在，不注入手写Bank。

---

## 5. 结果交接：验证实际函数接口，不验证对错误计划的服从

### 5.1 分别确定节点输出要求

**Program节点：** 返回 `status/outputs`，outputs满足该版本真实 output_schema，再进行显式交接。

**Skill节点：** 对应完整能力I/O，不得因为上层写了局部purpose便改变其输出。由可用 Program 完成则保留真实执行归因；由 Agent 实现则标为agent，不冒充Program正向。

**Dynamic节点：** 不继承 reference_skill_ids 的 output_schema。代码根据未执行后继和顶层 outputs 中的 `from=this_node` 引用，生成当前需要交接的字段列表，并明确放进 Runtime 材料。类型在真实消费者调用时按其接口检查，不构造复杂跨节点类型证明。

没有后继字段需求的动态节点，不凭参考Skill强制产生 `action_sequence/completed`。节点完成仍不等于任务成功；终态、提交型收尾沿用原规则。

### 5.2 缺字段不是继续携带永久missing

`complete_node` 在发布前检查实际需要的字段是否存在。失败时生成统一 `handoff_error`：

```text
node_id
execution_mode
required_fields
received_fields
missing_fields
consumers（只包含相关的后继引用）
```

已提交内容保存在本题 `pending_outputs` 与原日志，不能丢掉然后让Agent重新搜索。它尚不是完成结果，不给正向信用。

Runtime可以：补充合法实际值；显式把已有输出映射到所需字段；或修订尚未执行后继，让它引用实际字段。不得自动猜测 `object_id == tomato_instance` 的语义；也不得填假值满足检查。

### 5.3 普通映射，不改写实际程序结果

新增可选节点 `output_aliases`，方向固定为 **实际输出字段 → 交接别名**：

```json
{"output_aliases": {"object_id": "item", "location": "source"}}
```

规则：

- 先验证原始 Program/Skill 输出，再生成交接视图。
- 原字段保留；别名只引用同一个已有实际值/ResultRef，不创造值。
- 拒绝不存在的源字段、重复目标和覆盖不同原值。
- 原始Program结果、source asset和已有测试资格不修改。
- 别名声明本身不表示两个概念语义等价，最终正确性仍由实际调用/评分确认。
- 程序schema失败不得靠output_aliases绕过；只能修提案或解除本题不适用绑定。

### 5.4 计划引用与普通返回保持分离

`args` 使用已有 task/from/literal/unresolved；`complete_node.outputs` 是普通值；大结果使用已有 output_refs。

不能全局拆解所有 `{"literal": ...}` 字典，普通业务数据也可能有该键。检测方式是具体的消费者接口类型和明确的引用字段，而不是“看到literal就解包”。Runtime材料必须直接给出当前返回示例；所有提示与Schema由同一节点接口解析结果生成。

已经完成的结果不覆写。后续更正用新结果记录/别名视图，并修改未执行消费者引用；之前发生的操作与结果原样保留。

---

## 6. 局部修正、一次整图重规划和一次Dynamic退路

### 6.1 新增单操作 `patch_node`

在现有 Runtime 协议加入：

```text
runtime_step.action = patch_node
patch.target_node
patch.kind = args | handoff | detach
patch.reason
patch.args / output_aliases / consumer_refs / dynamic_goal（依kind使用）
```

本操作不与Office纯读批次混发；仍是一项Runtime决策。不新建辅助Agent。

| kind | 允许 | 不允许 |
|---|---|---|
| `args` | 修改未执行参数、补实际值或ResultRef；对替换项显式记录 | 修改原始用户输入或已执行Program的实参记录 |
| `handoff` | 映射pending/实际输出，改未执行消费者的字段引用 | 覆写旧输出、制造不存在的资源或伪造已完成 |
| `detach` | 解除当前不适用的本题Skill/Program绑定，明确当前结果目标；保留原资产作参考与候选 | 修改Bank源码、Schema、usable状态；把未执行Program计为完成 |

`detach` 不是一般默认步骤。正常匹配、参数ready且没有实际冲突的已授权可用Program继续直接执行，不插入“是否解除”的LLM询问。

patch不能添加/删除/重排工作流节点；这属于整图修改。handoff可以调整相关未执行后继的参数引用，但不能改它们的真实能力定义。

### 6.2 patch事务与原子性

先在内存副本上应用patch，使用同一节点解析与普通引用检查；有效后一次提交本题计划、pending结果状态和恢复身份。无效patch留下结构化反馈，不产生环境动作、不发布假结果。

patch本身不能修改正在运行的Program。Program执行期间由其自身控制分支，只有返回/报错/终止后才进入局部恢复；未知在途操作仍停止，不能并行重做。

### 6.3 统一有界失败处理

将以下路径都接到同一处理函数，不再散落 `continue`：

```text
complete_node输出不满足交接
准备输出映射错误
当前执行绑定接口不适用
同状态下原生/程序调用明确失败
无进展循环信号
```

普通明确拒绝的同错误签名上限沿用2次；签名排除预算、日志序号、revision与随机ID，不使用模型长解释作为身份。一次恢复请求本身的结构repair不增加新的全局预算。

达到上限后：

```text
还可整图重规划：使用剩余那一次
否则尚未退出失效计划：进入一次剩余任务Dynamic
否则：不重复派发同一已知无效操作，保留明确反馈和原预算终止规则
```

合法新参数、实际相关状态改变或新资源可以使后续调用成为新情况。不能通过换节点ID或改一句无关detail清空同一错误的限制。

### 6.4 Dynamic退路必须保留学习收益

进入退路时：

- 原始任务目标、当前环境、已取得结果、工作记忆、剩余预算全部保留。
- 被放弃节点标为abandoned，不标completed；已完成节点不重新做一遍。
- 保留相关usable Program卡片，允许显式调用程序完成剩余子任务或剩余整题。
- 只解除失效的计划接线，不禁用所有Skills/Programs。
- 不再调用新的全局Planner，不重新实例化同一失效整图循环。
- 退路节点没有完成任务却调用complete_node时，按当前结果和终止方式继续同一剩余任务处理；不通过反复创建dynamic_1…N刷新失败计数。
- 正式任务不因此获得第二次attempt、第二份评分或额外工具额度。

### 6.5 保留普通程序失败的既有语义

`not_found/needs_input/blocked`不产生永久disabled；保留真实搜索和状态供恢复。调用前缺参数/错误绑定是计划或调用错误，不自动作为程序实现失败。

真实程序异常或必要结果检查失败沿用原处理；Frozen只做本题避用，Train才允许按既定规则更新长期状态。已经由环境实际终止成功时优先采用官方终态，不能因worker没有完整RETURN抹掉成功。

---

## 7. 无进展纠偏：通用信号，不是benchmark路线特判

在 `TaskContext` 维护本题有限窗口的公开进展观察，供Executor使用。不要读取隐藏目标、金标准或私有对象树。

### 7.1 使用现有公开数据

综合：Adapter稳定 `progress_key()`、首次出现的资源内容摘要、有效结果发布、被后继实际消费的结果和当前任务终态。排除时间戳、请求ID、访问次数、预算变化和文件随机版本目录名。

同内容产生新result_id不算新信息。Program内部动作也进入同一观察记录，但无进展规则不把正常Program循环拆回逐步LLM；Program内部仍受原时间/调用/隔离限制。

### 7.2 固定检测规则

只检测最近序列中长度1—4的动作/状态模式，在同一未完成节点内连续重复两遍，且这两遍之间没有新的公开资源内容、有效交接或其他相关公开进展。命中表示“需要纠偏”，不是判定任务不可能完成。

第一次命中把有限的循环摘要放进下一次既有Runtime材料；模型可patch、换操作或重规划。相同循环再次出现且仍无新进展，按第6.3节升级。正常搜索不断获得新检查结果、不同数据窗口或新文件内容时不得命中。

这是一项固定控制取值，不按Heat/Cool、题号或分数调参。对无法可靠规范化的新反馈保守视为可能的新信息，不能为了触发停止而删除真实差异。

---

## 8. 逻辑决策恢复：同一决策幂等，不同决策不得复用已拒绝回答

### 8.1 新增持久化逻辑决策身份

修改 `TaskCheckpoint` 和 `EmpiricalSystem.agent()`：在真正发模型请求之前持久化：

```text
scope = task attempt / isolated trial / learning job
logical_decision_id
purpose = planner | runtime | extractor | builder | finish_only
owner_state_version
repair_index
status = prepared | response_received | applied | rejected
```

response key至少包含：scope、logical_decision_id、repair_index、请求内容digest、模型/策略身份。不再单独用相同prompt内容跨逻辑决策取回答。

- 相同决策在崩溃后继续：沿用同ID及已落盘响应。
- 响应已经应用或最终拒绝：下一轮使用新ID，即使prompt相同。
- 同一次结构repair：同logical_decision_id、不同repair_index；不得因此多一次恢复额度。
- Builder：原job_id/epoch/generation和固定试用绑定继续参与身份；恢复不能重新开始generation_count。
- 单轮QA：一个正式solver逻辑ID，恢复不产生第二次解题。

### 8.2 原子提交与副作用

Executor的pending_step、输出发布、局部patch、失败计数、node_index、replan/escape计数和决策完成状态应随同一份checkpoint状态原子提交。

原生事件仍有独立journal：事件ID与逻辑决策及ToolCall子项稳定关联。已经实际完成的调用不因崩溃或批次恢复重发。若动作结果或Program在途副作用未知，沿用现有停止/显式重建规则，不声称可以无条件exactly-once重放环境。

响应收到但尚未应用：恢复执行该响应一次。响应已拒绝且反馈已提交：不能再次恢复执行它。模型请求发出但返回/费用不可确认：保留unknown记录，不伪造零成本。

### 8.3 不能只修缓存、不修失败分支

P05与P06一起交付。只给每次循环新request_id，会把原来的廉价恢复死循环变成真实付费死循环；只限制循环但复用旧答复，又会阻断模型纠错。

`complete_node`错误必须出现在该决策的结构化结果与下一次材料中；已处理失败状态参与checkpoint，而不是仅存在临时history里。

---

## 9. Program ABI：把真实返回形状交给Builder，不让它猜

### 9.1 保持已有Program后端和接口，不新造解释器

`def run(ctx, inputs)`、原RPC、隔离、允许工具、文件发布规则不变。新增一份代码生成的 `public_program_abi()` 说明，供首次Builder和所有修复共用。放在现有 `empirical/program_worker.py` 或其公共纯数据邻接模块，worker校验与帮助生成读取同一份常量定义。

说明必须包含实际方法签名、返回封装与**工具表面类型**：

```text
ctx.observe() -> 当前公开状态dict
ctx.available_tools() -> list[ToolView]
ctx.call(name, arguments) -> 实际ToolResult
ctx.remaining_calls() -> 剩余原生调用额度
ctx.read_result(result_id, offset=0, limit=None, path=None) -> 本题已获得结果
```

### 9.2 真实工具形状

exact_catalog示例：

```json
{
  "name": "EXAMPLE_OPERATION",
  "description": "公开语义",
  "input_schema": {"type": "object"},
  "current_arguments": [{"resource": "resource_1"}]
}
```

`current_arguments`中的每个字典才是一次合法调用的完整实参。顶层没有单一 `arguments` 时不能把整个工具丢掉。

named_tools示例：名称和input_schema提供可调用接口，可能没有current_arguments穷举；缺这个字段不表示工具不可调用。不得在通用程序中假设所有工具都采用同一种目录。

使用由Adapter和ToolSpec真实生成的示例，不使用验证题的隐藏信息，不把固定位置或答案放进ABI帮助。

### 9.3 唯一程序返回封装

```json
{"status": "ok", "outputs": {"resource_id": "resource_1"}}
```

正常状态仍是 `ok/not_found/needs_input/blocked`；execution_error由执行层记录。返回原任务结果字典而没有status/outputs属于普通程序结构失败，不能被静默包装成ok。

Builder输出示例明确展示封装及本次Skill的真实output_schema。错误程序在原定Train试用中失败并进入已有共享repair额度；本轮不再增加第二次代码repair、不复制人工脚本进Bank。

### 9.4 学习保存与入口一致

`Learner.learn/_realize` 保存的新Workflow使用同一execution_mode和节点解释；`$new`规则保持。已有Skill被仅作参考时使用reference_skill_ids；生成Program和Implementation的关联仍保留明确skill_id。

程序候选产生后，正常试用、两个物理Train正向、后续Online调用和Frozen可用状态沿用CF2。不新增“每个新图整体验证才可执行”的工作队列。

---

## 10. 提示词与实际字段必须一起修改

| 角色 | 必须提供 | 必须删除或替换 |
|---|---|---|
| Planner | 完整稳定工具语义；当前可调用集合；相关真实能力；三种execution_mode；结果导向目标 | reference Skill自动继承完成要求；未执行假设充当公开状态；选Program又自由改写其能力 |
| Runtime | original_task；解析后的node_goal；mode；真实执行接口或动态交接字段；current_state；参考指导；恢复余量 | “任何节点都必须服从Planner步骤”；只给Skill文本不给完成字段；参考与绑定混写 |
| Builder | 当前真实ctx ABI、两种工具表面、唯一返回封装、固定案例/输入和Skill范围 | 只有函数名而没有返回形状；使用旧IR或历史见证术语 |
| Learner | 保留已执行程序/工作流的实际复用经验；新模式与待完成工作；真实失败 | 保存歧义skill_id；把新组合描述成已整体验证；把本题patch直接升级成validated |

所有实际HTTP schema保持object根。计划引用Schema、真实运行值、动态交接字段和Program返回不能共用一个过大的自由表达Schema。

必要场景下可以给动态节点展示简短可读的字段列表，不要求每步发送完整嵌套schema。程序已经ready并被自动执行时不发送Runtime请求，所以不会为了“确认已验证能力”增加成本。

---

## 11. 兼容、冻结、归因和清理

### 11.1 新旧Workflow并存的有限兼容

新Workflow写明 `interface_version="empirical.workflow.v2"` 和execution_mode；其内容变化按现有content ID产生新资产，不原地覆盖旧内容。

旧Workflow：

- 不重写旧Bank字节、历史结果或测试记录。
- 明确program_id的节点可在本题只读投影中解析为program模式，接口来自真实Program。
- 只有skill_id、用途不明确的旧节点，不由loader猜测“已验证局部调用”。在原本就存在的首轮Planner选择/组合响应中明确其执行模式；不额外加一次确认请求。
- select旧Workflow时允许一次返回 `node_modes`，为歧义节点明确dynamic/skill；dynamic时将原skill_id保留为reference。指定skill执行时采用资产原范围与I/O，不能保留另一套冲突目标。
- 未给出所需模式时进入原有一次结构repair或本题Dynamic，不写迁移信用。
- 已可用Program保持可检索与可调用；不得为兼容而统一禁用。

新的原生v2 Workflow正常select不要求node_modes；不增加每次重写整图的开销。

### 11.2 冻结输出

`Bank.freeze()`需要理解execution_mode：不可用的绑定Program从Frozen排除后，相应本题执行投影明确降为dynamic/reference，不能删除program_id后又被遗留skill_id偷偷绑定到其他执行范围。

任何降级投影保持原始Workflow来源可查，不声明仍是原执行图的相同验证结论。已有可用Program的源码、输入输出、状态与独立Train记录不迁移、不重算。

冻结状态、读取排序、统计和程序状态在Val/Test始终不改。新运行配置/代码不与旧checkpoint强行resume。

### 11.3 正向归因

- 只有真实Program启动、实际执行及原有正向依据，才计Program信用。
- patch、解除绑定、Agent手工完成或退出失效图不算Program成功。
- Program有效输出被显式别名/结果引用传给后继，沿用原实际消费归因；不得因为增加别名丢掉成功复用记录。
- 同一结果被后继消费多次仍是同一次程序调用，不增加独立任务数。
- 放弃错误外层计划，不抹掉内部Program已经实际成功执行的记录；也不自动把最终任务成功归功给所有被检索资产。

### 11.4 删除被替代的主线代码

替换后删除：把skill_id同时当参考与执行入口的分支；无效dynamic标记；complete_node失败裸continue；仅内容digest跨决策恢复；无限生成同目标dynamic_N的恢复路径；Builder重复手写不一致ABI帮助。

保留：普通Schema参数检查、实际工具权限、环境/文件安全、官方评分、原始日志、既有可用程序、非LLM检查和最小checkpoint。不要因本次改动把旧证明引擎重新导入。

---

## 12. 生产文件落点

路径前缀均为 `src/atomic_skillgraph/`。

| 文件 | 必须完成的内容 |
|---|---|
| `empirical/contracts.py` | execution_mode解析；dynamic参考不继承接口；普通交接字段与别名；统一Workflow校验 |
| `empirical/prompts.py` | NODE/PLAN/STEP/LEARNING模式；patch_node；字段示例；新旧说明替换，不能叠加矛盾提示 |
| `empirical/planner.py` | 稳定工具说明；select/compose统一解析；新组合可直接执行；旧歧义节点一次归一 |
| `empirical/bank.py` | 按明确模式查路线；reference不自动调用；保留程序卡片；freeze正确处理降级投影 |
| `empirical/executor.py` | Program-first保留；动态交接；pending_outputs；局部patch；统一失败；一次replan/escape；决策提交 |
| `empirical/checkpoint.py` | 持久逻辑决策及response状态；恢复身份；与执行状态原子提交 |
| `empirical/system.py` | agent恢复key；planner/runtime/learner/builder/finish-only决策scope；原始usage去重；单轮QA保护 |
| `empirical/task_context.py` | pending/实际结果区分；公开进展与循环摘要；已有工作记忆/大结果引用保留 |
| `empirical/learner.py` | 新Workflow模式；Builder ABI实际传入；既有待实现/修复/试用额度不变 |
| `empirical/program_worker.py` | 真实ABI帮助与返回常量同源；不取消隔离或改变既有程序方法签名 |
| `harness/tool_spec.py`、`simple_protocol.py` | ToolView/ToolResult真实形状测试；稳定语义透传；批次与副作用journal一致 |
| `harness/alfworld_simple.py`、`benchmarks.py`、`workspace.py` | 原CF2正确行为回归；核心不加benchmark名字判断 |
| 现有实验入口和配置 | 新策略身份、独立输出目录、checkpoint版本拒绝；不虚构不存在CLI参数 |

同一节点解释逻辑必须集中，不能Planner、Bank、Executor三处独立维护。新配置仅增加策略身份，不调原科学预算：

```yaml
runtime:
  plan_execution_policy: empirical.recoverable-takeover.v1
  full_replan_limit: 1
  dynamic_escape_limit: 1
  unchanged_failure_limit: 2
```

这些值是已批准边界的显式配置；若现有项目已经有等价字段，只用一份字段并统一读写，不保留相互冲突的两套参数。

---

## 13. 六Benchmark共享策略和原修复保留项

### 13.1 按接口能力分流，不按题型特判

| 接口类型 | 适用数据集 | 本轮行为 |
|---|---|---|
| 交互环境、实际环境终态 | ALFWorld | 同一可纠正计划/自动接管/恢复；不写Heat或位置路线表 |
| 文本工具与答案提交 | OfficeQA | 同一节点/结果引用/恢复；纯读1—3顺序调用和finish-only保留 |
| 文件计算与产物提交 | SpreadsheetBench | 同一节点/Program接管；文件与完整解法ready后提交评分 |
| 单轮文本回答 | SearchQA、LiveMath | 不进入图执行；一次solver和既有指导学习保持；恢复区分同次请求与新决策 |
| 单轮视觉回答 | DocVQA | 不丢图、不偷偷OCR；当前能力锁不变，unsupported不冒充成绩 |

局部patch、明确执行绑定和决策身份都属于通用机制。所有核心分支基于execution_mode、工具表面与Capabilities，不出现Heat/Cool、任务ID或benchmark名特判。

### 13.2 必须原样保持并测试的已完成修复

- ALFWorld：TAKE→MOVE/PUT→INVENTORY的一致状态；公开语义透传；隔离试用复用题目映射而不共享世界状态。
- Office：grep/read同一文本单位；确定输入错误为普通失败；真实基础设施异常不吞；1—3独立纯读计各自额度。
- Spreadsheet：只读检查不清空输出登记；修改/删除显式发布；相对发布名与工作区绝对路径分开；真实封存解法完成全部评分变体。
- 收尾：最终产物就绪不再重复解题；原生额度用尽最多一次受原预算约束的finish-only；环境实际终态优先。
- 学习：新Skill只用`$new`或真实已有ID；文字Skill与Program待完成状态分开；两个适用物理Train案例；Builder共享一次恢复。
- 日志：stage、logical_decision_id、repair、usage、native attempt、Program实际调用及结果；不增加补日志LLM。

### 13.3 运行安排不是新增性能门槛

SearchQA/LiveMath的单轮路径不因ALFWorld图修复而被全部禁止运行；需要先运行时固定单独源码与输出，不热修改。共享恢复或学习逻辑改变后，相关run按实际版本区分，不能混写成同一版本。

ALFWorld、Office、Spreadsheet走受影响Executor，完成本轮确定性回归后再进入相应新正式运行。当前DocVQA能力锁保持。Baseline各自入口仍独立验收，Ours补丁不表示其他七方法已全部放行。

本文不启动任何任务，也不新增收费试验授权。是否运行及数据规模按用户指令；实现者不得自行扩大模型矩阵或循环pilot到成绩满意。

---

## 14. 必须完成的回归：同时证明“错误约束退出”和“正确接管不退化”

以下是**待实施的生产验收要求**，不是本轮已经通过的测试报告。不得用仅测试一个新helper的返回值替代Planner→Bank→Executor→Broker→结果提交的路径。

| ID | 场景 | 必须断言 |
|---|---|---|
| T01 | dynamic参考一个整题Skill | 不继承整题output_schema，不自动启动其Program |
| T02 | 同一Skill明确作为完整skill节点 | 保留正确能力范围、原I/O及可用Program自动接管 |
| T03 | program节点ready | 自动执行，不新增Runtime确认调用 |
| T04 | 新组合两个usable Program | 同一次规划后两节点自动接续；不要求新图预验证 |
| T05 | 准备程序返回后输入ready | 普通映射接入后继，不再多问一次相同调用 |
| T06 | 第一路线不ready，第二条ready | 选择合法ready路线；不是直接回Agent |
| T07 | 参考full-task Skill的文档搜索节点 | 返回resource_id后继续后继，不要求completed/action_sequence |
| T08 | 实际输出名与后继字段不一致 | 有界handoff_error＋局部修正；不永久携带missing |
| T09 | 普通业务字典含literal/from键 | 不自动拆包；明确ResultRef仍正常解析 |
| T10 | output_aliases | 仅引用实际值；缺源/冲突拒绝；保留程序实际消费信用 |
| T11 | 局部args/handoff/detach | 不耗整图重规划；不改已完成结果、不改Bank、不假成功 |
| T12 | 同一交接错误重复 | 进入有界恢复，不出现416次裸循环 |
| T13 | 同prompt两次不同逻辑决策 | 各自独立响应，不能恢复上次已拒绝回答 |
| T14 | 同一未完成逻辑决策进程恢复 | 复用已落盘响应，无重复请求/计费/副作用 |
| T15 | 响应拒绝后崩溃恢复 | 保留错误和已结束状态，下一决策不再次应用旧答复 |
| T16 | Office批次执行到第2项后恢复 | 已完成子项不重发，未执行项和剩余额度一致 |
| T17 | 整图replan已用尽又不适用 | 仅一次剩余Dynamic退路；同一真实世界和预算 |
| T18 | Dynamic退路有匹配usable Program | 仍可接管；不得全局禁用程序或每步都强制原生调用 |
| T19 | 已知循环与正常探索对照 | 两遍同循环触发纠偏；新检查/新文档内容不误触发 |
| T20 | Planner不在reset可调用的原生操作 | 稳定说明仍在实际HTTP请求；当前目录不伪造可调用 |
| T21 | Builder实际ToolView | exact_catalog解析current_arguments；named_tools无该字段仍可用 |
| T22 | Builder实际Program返回 | 有status/outputs正常；缺外壳按普通失败；不得包装成成功 |
| T23 | 原真实Heat/双对象usable程序 | 同一个只读Bank保留程序调用，不能因新模式全部降成Agent |
| T24 | Program正常多动作循环 | 内部零LLM；预算与每个原生事件完整保留 |
| T25 | Frozen失败恢复 | 本题避用/patch可写，长期hash、状态、排序不变 |
| T26 | Program在途未知副作用 | 停止而非重复执行；不得把恢复当第二次成功记录 |
| T27 | 单轮QA/视觉路径 | 仍一次solver；没有额外Planner；图像输入完整/能力锁不变 |
| T28 | 文件、位置、状态和试用映射 | 第13.2节所有既有修复不回退 |

关键反例T01/T07/T12/T13，以及已成功接管T02/T03/T04/T23，必须形成双向回归：不能通过把所有节点变Dynamic来修错误，也不能靠继续接受错误schema来保护接管。

需要模型验证策略效果时，先复用已有开发任务与独立输出目录，不修改题号、评分和预算。旧Frozen诊断只测执行层；新版学习效果必须使用同一新版本从fresh Bank训练，二者分开命名和计费。不得拿录制反馈回放或手写程序替代自然学习效果。

---

## 15. 实施次序与交付清单

### 15.1 固定顺序

```text
保全现有结果、代码版本和usable Program
→ 节点模式/接口解释与Planner/Runtime Schema一起修改
→ 保留程序优先和自动交接，加入局部patch及有界退路
→ 同时修complete_node失败处理与逻辑决策checkpoint
→ Planner稳定工具说明、Builder真实ABI、无进展信号
→ Learner/Bank.freeze/旧Workflow读取接线
→ 单轮QA、文件、状态、计费与恢复回归
→ 固定版本并按已授权范围做真实验证/正式运行
```

先把模式、Schema、调用方和测试一次接通再删除旧分支，不能先删旧字段让运行到中途才报错。对于基线后已有同类正确实现，核对后复用，不回退重新实施。

### 15.2 最终交付

```text
commit SHA、完整resolved config、实际新增/修改/删除路径
T01—T28生产路径结果及原始测试日志
实际请求样本：Planner / Runtime / Builder / 恢复
正常接管保留与错误绑定退出的对照
已执行真实run的manifest、完整usage与结果（没有运行就明确未运行）
已知未完成项：区分工程错误、模型普通失败和效果未测量
```

报告三类状态：

- **工程通过：** 接口、控制流、恢复、隔离与评分正确。
- **机制发生：** 实际学出的程序被接管路径调用、跨节点自动传值和继续。
- **效果测量：** 成功率/主评分、每题成本按真实结果报告，未达目标不得说达标。

不得以“全部检查通过”推导六benchmark成本必低于50k或准确率最高；也不得因效果尚未测出就重复推翻本轮架构。已知执行缺陷按本文件闭合，后续科学结果按事实记录。

---

## 16. 来源与本轮核验边界

[S1] `ALFWorld_seed42_12Train6Val_review_20261005_183647.txt/.zip`：运行版本d046c1de；空Bank、Program实际Train/Val接管、Frozen只读记录。本文件不重新将旧R10.x Trace当成本轮结果。

[S2] `CF2_交付报告.md` 与 `SkillCompiler_通用降本与一致性修复_最终实施文档_v3.1-CF2.md`：已批准预算、适用案例、只读批量、结果存储、程序资格、文件/环境/提交语义。CF3只替换本轮明确列出的歧义与控制分支。

[S3] 当前仓库2528f53快照，核读的核心文件：

| 路径 | 核查用途 |
|---|---|
| `empirical/contracts.py` | 普通值与引用、Workflow结构校验、Skill输出继承 |
| `empirical/bank.py` | 路线、usable、Program卡片、freeze |
| `empirical/planner.py` | 当前材料与select/compose |
| `empirical/executor.py` | 自动执行、complete_node、错误处理、replan/fallback |
| `empirical/system.py`、`checkpoint.py` | 请求恢复、单轮QA、试用与成本 |
| `empirical/prompts.py`、`learner.py` | 模型实际协议与Builder调用材料 |

**本轮完成的是源码与已确认要求的复核、实施规范定稿和文档内样例检查。没有声称CF3已经合入仓库，没有重新运行生产测试、真实模型、Docker或环境，也没有启动收费实验。**
