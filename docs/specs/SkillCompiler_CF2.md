# SkillCompiler 通用降本与一致性修复：最终实施文档

**版本：v3.1-CF2（本轮合并定稿）**  
**实施基线：`YU-S3/AtomicSkill-ToolGraph_v3@77959e89afce13e67402b6dd6672f3dbadff0d2e`。**  
**范围：现有 empirical v3.1 主线；六 Benchmark 的公共执行/学习机制，以及相应 Adapter 缺陷。**  
**状态：设计边界已确认、源码及原始运行材料已核查；本文规定待实施补丁，不表示补丁已经完成或六 Benchmark 效果已经达标。**

本文件取代此前所有 `ALFWorld降本`、`剩余缺陷`、`CostFix-1.0`、`costfix1` 补充实施单。实现者只执行本文件，不再将旧补充单叠加。已经实现的 empirical 架构继续保留；不重新执行 R10.x 或 v3.0 的旧证明型机制。文内“新增”文件、接口、字段是本轮实施要求，不是声称基线已经具备。

---

## 0. 已确认边界与完成标准

### 0.1 不再讨论、不擅自改变的事项

| 项目 | 本轮固定要求 |
|---|---|
| 方法 | 保留 Skill / Implementation / Program / Workflow、受限 Python、普通参数传递、真实 Train 试用和独立评分 |
| 验证复杂度 | 不恢复 witness、owner、效果证明、历史同值硬依赖、四层证书或新的模型审查器 |
| 模型 | 沿用当前 Provider、模型和能力锁；不专门修改 DeepSeek 命名/身份记录，不因别名问题延迟实验 |
| 推理 | 所有角色保持当前 `high`；本轮不做 Runtime 推理降档 |
| Builder | 首次最大输出 32,768；仅截断且无可接收提案时，使用原有一次恢复额度，恢复最大输出 65,536 |
| 恢复次数 | 同一构建作业总共最多两次 Builder 生成：首次＋一次共享恢复；截断、结构、代码、试用修复共享这一次额度 |
| 共享预算 | 不提高原 Planner、Runtime、学习或试用预算；请求恢复不清零已消费 tokens |
| 纯读批量 | 同一响应允许最多 **3** 个独立纯读调用，按顺序执行；本轮原生白名单仅 OfficeQA `glob/read/grep` |
| 程序可用 | 保持同程序版本两个不同物理 Train 任务的实际正向记录；不把 `ok` 当官方正确 |
| 训练范围 | 不导入人工 Bank；Val/Test 不写长期知识；待补构建只使用已完成的 Train 经验 |
| 实验 | 六个既有清单和 42/43/44 不变；同一 cell 三个 seed 独立空 Bank；不恢复 ScienceWorld |
| 费用 | 所有真实请求、恢复、试用都计费记录；不新增总结/补日志/证明 LLM |
| 运行安全 | 不热修改运行中的 checkout，不覆盖旧结果，不跨代码策略修改 hash 强行 resume |

“最多3个”及后文结果展示阈值是对已批准机制的固定实现取值，不是引用上游默认值；不得根据分数逐 Benchmark 调整。

### 0.2 结果目标不能替代正确性，也不能被忽略

- **冻结执行目标：** 每个支持的 `模型×Benchmark` 平均 tokens 争取 **≤50,000/题**，目标上界 **60,000/题**。这不是新的任务停止预算。
- **质量目标：** 在同模型、同 split、同评分、同交互条件下，争取达到全部 baseline 中最高的主指标。不得以删除困难题、增加重复作答、改变 scorer 或放宽正确性换成本。
- 训练主任务、Learner/Builder、隔离试用分别记录；不能将训练费用藏进测试外，也不能用含试用的 Train 均值冒充 Test 均值。
- 单元测试、录制反馈回放、真实任务效果分别报告。测试通过不表示平均成本已达标，更不表示六 Benchmark 均已超过 baseline。
- 已知一致性错误必须有旧代码失败、新代码通过的回归。文档不声称数学意义上杜绝全部未来缺陷；工程要求是本轮列举的不一致不能静默回归。

### 0.3 实施顺序

```text
固定基线和原始材料
→ 公共工具定义、结果/状态更新与终止语义
→ Runtime/Planner、结果引用和纯读批量
→ Learner待完成作业、适用试用、Builder恢复
→ 六Adapter接线与单轮QA保护
→ 删除被替代分支、完成非LLM回归
→ 有界真实机制/成本验证
→ 独立新策略正式实验
```

不在核心接口尚未接通时分六个 Benchmark 各写一套状态、提示词和恢复逻辑。

---

## 1. 修改依据：以本次新材料为准

### 1.1 ALFWorld 最新六题，不与旧 R10.x Trace 混用

来源：`valid_seen6_materials_20261005_015255.zip/pilot_terminal_rule/`。训练源码记录为 `15ede6b…`，六题验证为 `4188a7f…`；核查基线为上述 `77959e8`。报告中的旧运行身份不能重标为新补丁身份。[S1]

| 任务后缀 | 场景 | tokens | LLM请求 | 程序执行 |
|---|---|---:|---:|---|
| `_105_look_at_obj_in_light` | keychain＋desklamp | 32,837 | 7 | 无 |
| `_139_pick_and_place_simple` | candle→toilet | 33,507 | 6 | 一次，`not_found` |
| `_123_pick_clean_then_place_in_recep` | clean cloth→cabinet | 45,278 | 11 | 无 |
| `_33_pick_cool_then_place_in_recep` | cool apple→countertop | 108,323 | 25 | 无 |
| `_116_pick_heat_then_place_in_recep` | heat apple→diningtable | 460,965 | 84 | 一次，`not_found` |
| `_21_pick_two_obj_and_place` | 两张creditcard→dresser | 9,225 | 1 | 一次，15个动作后终止成功 |
| 合计 | 全部6题成功 | **690,135** | **134** | Frozen只有一个usable程序 |

输入484,762，输出205,373；reasoning包含在输出中，不再次相加。Heat占总量66.79%；其余五题平均45,834。九个Workflow都是单动态节点。因此“前8菜单截断”是真实代码缺口，但不是这六题仅一个程序的主要原因。

Heat用量的实际时序分段：Planner 10,201；目标搜索269,111；TAKE/到达microwave/OPEN 12,054；错误MOVE后恢复155,154；真正HEAT/导航/放置14,445。分段费用不是可保证回收的全部费用。

### 1.2 具体缺陷和对应改动

| ID | 已定位问题 | 直接修改 |
|---|---|---|
| F01 | MOVE放下对象后`held_objects`不更新；INVENTORY明确为空仍保留旧值 | 单一状态更新入口，覆盖MOVE/PUT/INVENTORY，局部检查读取同一当前状态 |
| F02 | `primitive_action_schema.public_semantics`在SimpleAdapter中丢失 | 注册定义直达Planner/Runtime/Builder/参数检查 |
| F03 | 搜索中间结果只在短历史里，程序内部搜索也难供后续使用 | 每题工作记忆和统一结果存储 |
| F04 | 文字Skill存在但Program缺失；`reuse_existing`不进入Builder | 可执行状态与待完成作业分开，适用新Train经验到来时补齐 |
| F05 | Builder截断没有有效提交；Look试用选择了无light_source的Heat案例 | 有界截断恢复；先固定适用输入和案例，不用无关案例凑两个 |
| F06 | 学到的搜索程序把观察文本中的对象名称当导航地点 | Builder与实际测试覆盖公开工具参数来源及中途入口，不手改验证题程序 |
| F07 | 程序卡片无目标，菜单按ID取前8；自动入口只看第一路线 | 共用相关程序卡片、当前Skill指导、全部授权路线ready检查 |
| F08 | 准备输出只支持同名字段；同一无进展失败被revision扰动 | 显式普通字段映射、稳定相关状态key |
| F09 | Spreadsheet生成产物后`files=[]`检查清空manifest登记 | 增量发布；未改动正式产物保持登记；改动/删除明确提交 |
| F10 | 文件图完成后反复进入整题Dynamic；试用产出后又完整求解 | 区分环境终止与结果提交；结果就绪直接封存评分 |
| F11 | Office grep行号被传给read字符偏移 | 统一明确的位置单位，返回可直接消费的offset |
| F12 | 多个合法纯读ToolCall引发昂贵结构repair；大列表被模型抄写 | 允许有界顺序纯读；大结果引用，不复制全文 |
| F13 | 原生额度用尽时丢失最后提交；普通正则/路径错误升级成未知副作用 | finish-only收尾；Adapter明确普通输入错误 |

文件类来源为 `file_adapter_smoke_bff5350_materials_20261005_023527.zip`，运行是开发smoke、不是正式成绩。Office 30次结构修复请求共490,956 tokens；Spreadsheet 40757的文件存在但登记被清空；353-29多次完成节点后仍反复进入Dynamic。以上以原始请求、Trace和源码核查为依据。[S2]

### 1.3 已完成核查与待验证不能混淆

已完成的历史核查包括：原始文件校验、逐请求统计、生产函数控制复现、MOVE最小修正对照、学习程序录制反馈回放。**没有完成本文件所有补丁的生产实现，也没有跑补丁后的真实模型六 Benchmark 矩阵。**

本文后续测试表是实现者必须完成的验收规范，不是已经通过的测试报告。[S3]

---

## 2. 代码落点和职责：只维护一套控制流

下列路径以`src/atomic_skillgraph/`为前缀；“新增”均为本轮新增。

| 文件 | 具体职责 |
|---|---|
| `harness/tool_spec.py`（新增） | `ToolSpec`、参数/结果结构、单位、公开说明、只读与批量属性；生成模型视图 |
| `harness/simple_protocol.py` | Broker单一调用入口；记录结果、调用额度、终态和批量执行子项；保留lease与未知副作用处理 |
| `harness/alfworld.py` | 原命令解析与公开原生语义；不增加任务路线表 |
| `harness/alfworld_simple.py` | 单一ALFWorld公开状态更新、INVENTORY解析、轻量视图、本地结果检查 |
| `harness/benchmarks.py` | Office位置引用/错误、三类QA保护、文件任务提交契约 |
| `empirical/task_context.py`（新增） | 本题工作记忆＋结果存储＋模型视图；不成为长期Bank或证明系统 |
| `empirical/contracts.py` | 普通ResultRef、运行结果与计划引用的区分；ValueStore按需解析 |
| `empirical/prompts.py` | 唯一版本化小Schema；复用选择、显式映射、只读批量、完整提交规则 |
| `empirical/bank.py` | 能力卡片查询、待完成作业和案例索引存储；Frozen禁止写 |
| `empirical/planner.py` | 选择已有Workflow或提交短计划；不重写整个已有图 |
| `empirical/executor.py` | 当前目标程序优先、准备接续、结果引用、稳定失败key、提交型收尾 |
| `empirical/learner.py` | 经验去重、程序意图、待完成工作调度、适用案例选择 |
| `empirical/system.py` | 实际Provider请求、Builder共享恢复额度、角色预算、试用收尾、QA旁路 |
| `empirical/workspace.py` | 内容一致的增量发布、只读检查不注销产物、原子提交和丢弃 |
| `empirical/program_worker.py` | 现有隔离和RPC；接收公共接口变化，传回运行/文件结果，保留终态处理 |
| `experiments/`中的现有runner | 新策略配置、诊断入口、恢复与新目录；不得新增不存在的CLI参数来“说明如何运行” |
| `tests/`（相应新增测试） | 本文件T01—T40；旧代码反例、生产模块路径、恢复与跨Benchmark契约 |

允许小函数合并进邻近模块，但不得出现两个互不一致的工具定义、状态写入口、结果发布器或Builder恢复计数器。不得复制一份Executor专供某个Benchmark。

---

## 3. 公共工具契约：工具事实只定义一次

### 3.1 `ToolSpec`的最小字段

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    result_schema: dict
    effect: str                # read_only / stateful / sandbox_compute
    batchable: bool            # 本轮只有Office glob/read/grep为True
    field_units: dict          # 例如offset=unicode_codepoint_0_based
```

handler仍由Adapter负责；Python可调用对象不进入prompt。每个Adapter创建一次定义集合，`available_tools()`只叠加动态可用性/当前合法参数，`tool_definitions()`从同一集合给出静态帮助。

- `read_only`由代码定义，模型不能声明；`execute_python`即使这次只打印，也不进入纯读白名单。
- 对ALFWorld，原生命令参数和名字来自同一解析表；已有`public_semantics`直接进入description。PUT和MOVE共享放置语义，不允许说明支持而状态处理不支持。
- Runtime菜单、Builder帮助、参数校验都使用同一ToolSpec。不得通过另一份手工枚举“简化”成不同的工具名/字段。
- `result_schema`描述实际返回：列表就是列表，不能一边声明对象、一边返回字符串/列表。统一外壳仍为`accepted/observation/data/error/done`，附加字段由定义声明。
- 这不是任务效果证明。工具结构合法不等于内容正确；最终分数仍由原scorer返回。

### 3.2 公共输入和反馈不能被策略层偷偷修饰

Broker接受合法参数后调用一次真实handler。原始请求、返回、异常分类保留。派生状态与摘要另外保存，不覆盖原始observation；不得把“模型说它完成了”写入Adapter状态。

输入Schema错误属于未执行；handler明确拒绝属于已尝试失败。每项日志写清是否发给环境、是否消耗原生调用、环境实际步数。未知在途副作用保持停止，不盲目重发。

### 3.3 统一位置单位

Office `read.offset`固定为**解码后Python字符串的零基字符索引**，不是UTF-8字节位置、token位置或行号。

`grep`每条命中返回：

```json
{"path":"relative.txt","line":253,"offset":25149,"text":"..."}
```

`line`是一基显示行号；`offset`可以直接传给read。两者由同一份解码文本计算。使用保留换行的累计长度，明确CRLF规范化；read和grep必须读取同一规范化文本，不各算一套。

保持既有每次最多40条grep命中和12,000字符read窗口。这里修的是位置语义，不扩充检索预算，不用目标答案选择结果。

---

## 4. 公开状态一致性：修MOVE，也修产生该类问题的结构

### 4.1 ALFWorld只保留一个事实写入口

在SimpleAlfWorld内设`_apply_public_result(name, arguments, result)`；`call()`执行真实动作后只调用它一次。Agent和Program都经过Broker→同一Adapter，因此共享更新。

| 事件 | 更新规则 |
|---|---|
| reset | 清空位置、发现、搜索历史与先前任务状态；按本次reset公开反馈初始化 |
| 接受的GO_TO | 设置当前位置，记录真实访问；访问不等于完整检查 |
| 接受的TAKE | 加入持有集合，更新该对象的公开移动记录 |
| 接受的PUT或MOVE | 移除被放置对象，记录实际目的地；不只处理PUT |
| 明确INVENTORY非空/为空 | 用解析出的完整集合纠正持有缓存；空返回确认为空集合 |
| INVENTORY无法解析 | 标记inventory未知/不完整，不把旧集合继续当确定当前事实 |
| 拒绝的动作 | 不套用成功动作的状态变化；保留真实返回 |
| 其他明确有状态变化的动作 | 显式处理或使受影响缓存失效；不能默默当作不改变状态 |

保留旧Program常用的`observation/current_location/held_objects/visited_locations`可访问性。增加`inventory_status=known|unknown`等必要状态标识；未知时模型视图只显示未知，不展示旧列表为当前事实。兼容列表仅表示“上次已知”，使用方必须看status。局部检查只有status已知时才能据此通过。

`check_local()`读取同一当前状态及实际调用记录；不能独立维护第二份`held`。TAKE后又MOVE并不满足“返回对象仍被持有”；不因为此前有一次TAKE就正向通过。未知时返回unavailable，不制造新的整个任务异常。

### 4.2 无新增世界真值服务

不读隐藏物体树、不增加每步INVENTORY/LOOK探测，不让LLM确认状态。更新只来自现有公开反馈与已接受原生动作。公开反馈有矛盾时，优先撤销缓存的确定性并暴露必要原文，不能猜一个值维持“已知”。

### 4.3 其他Adapter对应的状态

- Office：语料只读；读取结果对应固定语料文件和字符区间。查询失败不登记为“已读到答案”。
- Spreadsheet：当前版本来自Workspace manifest；只有原子publish成功后才更新可提交产物。失败stage不成为当前版本。
- 单轮QA：公开输入只读，没有需要模拟的物理状态；不强制添加inventory、关系谓词或局部成功验证。

公共核心保存本题资源和进度；具体状态转换留在Adapter，禁止在通用Executor写`if task_type == heat`。

---

## 5. 本题工作记忆与结果引用：少传副本，不丢事实

### 5.1 两个对象合在`TaskContext`中即可

```text
TaskContext
  result_store：每次真实工具/程序结果，只存一份，可按ID访问
  working_memory：当前目标、已检查资源、必要历史结果、未解决输入
```

它们在每题和每个隔离Trial开始时新建；下一题/其他seed不继承。TaskContext只读Adapter提供的当前状态，不再缓存一份独立可写的held/current_location；Adapter仍是本域当前状态的唯一写方。恢复同一题时从持久化原始结果重建，不新增总结LLM，也不通过评估答案补内容。

Broker在结果完成后按event_id登记一次。Program内部每个原生调用同样登记；Program最终结果补充调用区间，不重复复制所有内部动作。

### 5.2 保留内容与展示内容分开

| 原始存储 | 默认LLM视图 |
|---|---|
| 完整公开工具返回、程序返回、文件版本引用 | 相关摘要/短窗口、result_id、类型、总大小、是否截断 |
| 全部真实调用顺序与失败 | 最近3条简明反馈＋去重工作记忆 |
| 所有已完成节点结果 | 小值直接显示；大值显示引用，不完整展开 |
| 完整原生目录来源/审计字段 | 名称、参数、公开含义、当前合法选项；不反复发hash、source_ref、revision证明串 |

本轮固定展示阈值：单个JSON/text值≤2,048字符可以内联；列表预览最多40项且仍受2,048字符约束；大值原文完整保留。不得对任务原始问题、SearchQA既定context、LiveMath选项或DocVQA图像套用此截断。

提供`read_result(result_id, offset, limit)`本题只读访问；text按字符、数组按项，单位在返回中明确。它只读取已获得公开结果，不能查新外部信息。模型调用该读取仍占一次现有工具调用额度，不增加总额度；默认文本窗口12,000字符、列表40项。该工具本轮不参与多调用批量，单轮QA不暴露它。Program与自动参数解析读取本题已有值不算新的环境动作，但记录本地读取且受RPC大小、时间和内存限制。

本题`read_result`由Broker注册为保留的本地handler，在Adapter调用前分流；不得转发给ALFWorld原生动作解析器。它消耗现有总工具调用额度，但`backend_invoked=false、environment_step=0`，日志明确标记`local_result_read`。它只接受当前TaskContext签发的result_id，不接受路径或上一题ID；Program同样只能经受限公开RPC访问。

### 5.3 简单引用协议，不让模型抄写大列表

新增用于运行时引用的结构：

```json
{"result_id":"r17","path":["data",0,"offset"]}
```

`path`仅允许字段名和非负数组下标；空path表示完整结果。无表达式、脚本、通配符或外部路径。

`runtime_step`增加：

```json
{
  "action":"complete_node",
  "outputs":{},
  "output_refs":{"matches":{"result_id":"r17","path":["data"]}}
}
```

工具/程序调用同样允许`argument_refs`，由核心在调用前解析成真实参数。`outputs`与`output_refs`、`arguments`与`argument_refs`不允许写同一个目标字段；冲突是普通结构错误，不默默覆盖。

ValueStore存大结果引用；在类型检查与交给Program/工具前按需解析。Program需要普通列表时收到普通列表，不能把preview伪装成完整列表。超出既有RPC上限时使用上述受限读取，不把限制直接调大到无限。

**计划引用**仍是`task/from/field/literal/unresolved`；**运行outputs**是实际值或显式`output_refs`。不要把`{"literal": ...}`在运行输出中自动剥壳，那可能改变合法的用户数据。提示词和类型定义清楚分开，错误由普通反馈修正。

### 5.4 工作记忆的各域内容

- ALFWorld：真实检查过的scope、公开发现对象、最后已知位置、当前位置/持有状态、目标查询、未检查地点。closed、unparsed或只有访问不能写成目标不存在；移动后失效原位置的“当前”属性。
- OfficeQA：检索表达式、结果ID、命中文档与可用offset、已读窗口；不让模型重复生成完整路径列表。
- Spreadsheet：执行代码的内容ID、运行返回、已发布产物版本、已报告的工作表/表头信息。没有实际检查过的工作表不能自动标记已检查。
- QA：仅使用本次公开输入和检索到的简短指导；没有多轮环境历史。

结果存储是公开信息缓存，不是新的证据准入系统；不得要求每条记忆先经模型评价才能写入。

---

## 6. Planner与Runtime：使已有能力可见、可选、可自动接续

### 6.1 共用程序卡片查询

新增`Bank.program_options(query, *, node=None, allow_candidate=False, excluded=(), limit=8)`，Planner与Executor共用。每个Program只展示一次，内容为：

```text
program_id、state
capabilities=[skill_id、goal]
简短使用说明与入口限制
真实input_schema、output_schema
```

入口限制是程序作者显式声明的普通使用说明，不是代码推导的形式前置条件。历史程序没有声明时记“未声明”，不能自动写“支持任意中途状态”。当前节点的Skill guidance独立给一次；不把源码、试用轨迹或全Bank塞进卡片。

排序固定：当前已授权路线优先；其余按与节点goal词重叠、整体task goal词重叠降序，再按已有可用状态/Train可靠性排序，最后ID。分词使用Unicode单词、casefold；不新增LLM reranker或网络embedding。查询只读，不因Test使用次数改变长期排序。

Frozen排除candidate/disabled；Train可展示candidate。菜单最多8项，不限制内部对当前节点全部授权路线的ready检查。Dynamic提供最多3条相关Skill指导，不提供整份Workflow JSON当指导。

### 6.2 已有Workflow只选择与实例化，不要求重写

Planner一次主要响应支持互斥的：

```text
mode=select：workflow_id＋明确node_args覆盖
mode=compose：完整短workflow
```

select只允许实际检索到的ID；代码deepcopy、应用参数，再用现有普通结构验证。保留任务数量、对象身份和必要关系；需要改节点目标或结构时使用compose。不能用历史值相等自动加边，也不能代码按任务类型偷改路线。

select不新增一个selector LLM；仍是一次规划和至多一次结构修复。SearchQA/DocVQA/LiveMath保持single_answer旁路，不增加Planner。

### 6.3 所有已授权路线逐一检查ready

```text
resolve当前node.args
→ 从Bank.routes(node)获取已授权路线
→ 排除本题blocked、同状态已试过的调用
→ 按现有顺序逐条检查真实输入Schema
→ 第一条ready就执行
→ 全部不ready才交Runtime
```

第一条缺输入不能阻止第二条ready路线。节点显式指定program_id时仍只授权该程序。不能因其他程序I/O相似就扩大到其他目标。

### 6.4 准备输出的显式改名

`call_program`支持可选`output_mapping`：key为程序输出字段，value为当前节点输入字段。

```json
{
  "action":"call_program",
  "name":"program_id",
  "arguments":{"target_query":"apple"},
  "output_mapping":{"found_object":"object","found_location":"source"}
}
```

执行顺序：实际程序返回→检查正常结果与必要字段→应用映射→重新resolve→自动ready检查。

- 当前路线程序完成自身节点时按其output_schema发布；不通过准备映射重写其输出。
- 非当前路线准备程序的目标输入名必须来自当前node.args或关联Skill/Program的I/O定义。
- 没有mapping时仅同名补缺；不静默覆盖已有参数。
- 替换已给参数必须在`replaced_inputs`明确声明；只改未完成节点，不改已完成结果和原始任务要求。
- 映射错了，不回滚已经真实发生的环境动作；保存可用结果给Runtime修正，但不发布假节点完成。
- 不另发一次“确认映射”的LLM请求。

### 6.5 正常未完成与真实程序错误分开

`not_found/needs_input/blocked`是正常未完成：不发布成功outputs、不永久禁用，但保留实际搜索过程和已取得的公开信息。相同输入与相同相关状态不立即自动重试。

程序异常、必要返回字段缺失或局部检查明确失败，本题停用该版本；长期状态仍只由Train现有规则更新。环境已终止时立即采用真实终态，不能因为worker来不及RETURN改写已发生的成功。

### 6.6 稳定失败key，不用整份带revision的observe

`Adapter.progress_key()`只序列化相关公开状态、实际合法参数、资源内容/产物版本；排除时间、预算计数、revision、source_ref、随机版本目录名和日志序号。对无法结构化的新反馈保留文本变化，不能把未知文本全部忽略。

同一action/name/args＋同progress_key的明确拒绝达到2次后，不向环境发送第3次，走既有一次replan或明确终止。真实状态变化解除限制。

被接受的OPEN/CLOSE、导航或重复查询不因字面重复被一律禁止；这类无效循环首先靠完整工作记忆和正确状态解决。不得新增“所有重复动作都禁止”的策略。

### 6.7 Runtime模型材料只含决策所需内容

```text
原始任务目标（一次）
当前节点目标、当前Skill指导、实际参数与缺项
当前公开状态的可读投影
相关工具定义/当前合法选项、相关程序卡片
本题工作记忆
最近3条简明结果
已完成结果的小值/引用
剩余预算与是否还可replan
```

最新observation不在多处重复；完整discovery审计元数据留在原日志。不得恢复需要模型解码的frame_rows/frame_map。每个阶段只使用一份模板，旧版“每次必须一个调用”等说明按第7节条件生成，不能新规则追加在相反旧规则之后。

---

## 7. 同轮多个纯读调用：一次接线到实际Provider响应

### 7.1 允许集合与返回类型

当前允许：一个Runtime响应内 **1—3个** `runtime_step` ToolCall，每个`action=call_tool`且name为Office `glob/read/grep`，并由ToolSpec标记`effect="read_only"、batchable=True`。

新增内部`RuntimeDecision`包含有序actions及原始call_id。非Runtime阶段仍返回单个提交对象；Planner、Learner、Builder不能因此变成多提案。

单个Runtime action继续使用原协议。多个`finish/complete_node/call_program/revise_plan`、混合写动作、超过3个或非法结构，不自动执行其中一部分：进入原有至多一次结构修复；仍失败则按正常协议失败记录。不得静默丢弃后几个调用。

### 7.2 执行顺序和依赖

1. 在执行任何子项前检查整批形状、工具白名单和参数引用；所有参数必须在本响应发出前已确定。
2. 允许引用此前result_store的值；不允许引用同批尚未生成的结果。
3. 按ToolCall数组顺序逐个调用Broker，不并行运行环境。
4. 每个实际调用分别计费/计工具额度/记事件和结果；无额外批次费用，也不能一批只算一次。
5. 普通正则/路径输入错误只影响该子项，其余独立只读项继续；结果统一返回给下一轮。
6. 预算耗尽或基础设施异常后停止剩余项。剩余项标记`not_executed_budget`或`not_executed_infrastructure`，不伪造结果，不补发。
7. 同一个已保存响应恢复时，使用响应ID＋call_id或稳定数组索引的日志key，已完成项不重复执行、不重复计费。

### 7.3 必须修改真实阻塞点

`EmpiricalSystem.agent()`当前先检查`len(tool_calls)==1`。必须在这里针对Runtime返回完整批次，而不是只在Executor增加一个batch分支。Provider层保存整个数组、实际usage与finish_reason；不能丢失第二/第三个ToolCall。

Runtime prompt通过能力配置生成：Office说明最多3个独立只读；其他工具型任务仍说明一个操作；single_answer无ToolCall修复和批量入口。

本轮不依赖一个可能被Provider忽略的`parallel_tool_calls`开关。没有服务端强约束时，本地解析仍完整支持已批准的批次。

---

## 8. 提交型任务收尾：节点结束不是重新解题信号

### 8.1 Adapter明确终止方式

在Capabilities添加`final_submission_kind`：

| Benchmark | 类型 | 收尾 |
|---|---|---|
| ALFWorld | `environment` | 真实done/won停止；额度用尽且未won则失败，不通过文本宣称成功 |
| OfficeQA | `text` | 显式最终answer准备好则提交；不能把任意准备字典当answer |
| Spreadsheet | `files` | 完整可提交文件bundle准备好后按计划收尾 |
| SearchQA、DocVQA、LiveMath | `single_answer` | 保留一次solver回答，不增加第二次解题 |

新增纯结构的`submission_ready(...)`或等价函数，只检查所需结果/文件是否存在且属于当前题，不接触参考答案、不预判正确性。

### 8.2 正常图走完

- `environment`未终止：保留原有从实际状态继续的Dynamic机制。
- `text`：若plan.outputs显式提供最终answer，或最后节点显式产出声明为最终的answer，则直接submit。
- `files`：plan已走完，且当前已发布的完整解法bundle齐全，直接submit；不得自动反复创建dynamic_1…dynamic_8。
- 仅完成准备节点、最终结果不齐：可继续原有Dynamic，不新增固定复查/二次求解。
- 显式`finish`仍可提交不足或错误结果，由scorer给真实失败；不自动重做任务到成功。

### 8.3 用完原生工具额度

环境已终止优先处理。否则：

```text
禁止新native / Program调用
→ text已有最终answer：直接提交
→ text没有answer：最多一次finish-only请求
→ files：封存已有产物，缺完整bundle如实失败
→ environment：保留真实未完成结果
```

finish-only仍属于原Runtime预算，无结构repair，无新检索、程序调用或replan。若总token预算已用尽，不强行额外请求。single_answer不进入这个分支。

### 8.4 同样修复Train试用收尾

`EmpiricalSystem.test_program()`里，Program返回完整最终产物时，直接调用对应submit/evaluate，不再启动一遍整题Dynamic。

局部能力只返回中间结果时，仍可从实际状态运行原定续程，但只运行一次，计入Train试用成本。最终正向记录依据保持明确：独立局部检查通过，或实际提交/消费产物后的Train任务结果；不能只因`status=ok`晋升。

文件程序本次声明/生成的最终文件必须实际被封存评分；之前无关产物已存在、一个空程序直接返回ok，不获得“生成最终解法”的正向记录。此检查是普通文件关联与实际消费，不恢复形式证书。

---

## 9. Workspace增量提交：文件、manifest、封存必须一致

### 9.1 定义本次`files`的含义

`execute_python.files`及Program返回`outputs.files`：本次新建或修改、希望作为正式产物提交的相对文件路径。增加可选`deleted_files`，默认空列表，显式撤销既有正式产物。

- 未修改、仍存在的已登记正式产物，默认继续保留。
- 只读检查`files=[]、deleted_files=[]`不能清空既有登记。
- 已登记产物被修改却未重新声明，或被删除却未列在deleted_files中：返回普通`undeclared_output_change`，丢弃该stage，保持上一版本。
- 中间文件可存在但不是最终提交依据；提交和评分只使用当前登记的完整产物。

### 9.2 `Workspace.publish`的确定算法

```text
读上一个已提交manifest与登记产物hash
→ 检查stage是本Workspace所有、无越界/符号链接
→ 对比上版登记文件的存在与内容
→ 未改且未删除：继承登记
→ 修改/新增：必须出现在本次files并验证实际文件
→ 显式删除：从登记移除，验证与stage一致
→ 未声明修改/删除：拒绝本次提交并保留旧版
→ 对有效输出生成新登记
→ 原子切换manifest
```

只读检查未改变任何工作区文件时可以直接保留原manifest并丢弃临时stage；即使创建新的内部version目录，逻辑内容hash不变也不能被当作任务进展。

- 不允许把`/workspace/inputs`内文件声明为可变输出。
- 不根据Agent一句“文件正确”发布；只验证文件存在、范围、声明与内容版本一致。
- 失败stage不修改当前指针。commit前后崩溃恢复必须只看到完整旧版或完整新版。
- 保留现有只读input挂载、容器权限、无网络、CPU/内存/时间上限，不将生成代码放到宿主进程执行。

### 9.3 Spreadsheet完整解法保持原规则

最终必须有`solution.py`和`case1_result.xlsx`；固定镜像及解法hash封存。其余隐藏/独立变体使用同一封存解法，不把评分反馈回传Agent修答案。

变体的全部结果都保留，错误答案是正常实验结果。不能因为主案例文件存在就跳过其余变体，也不能借本次修复更换scorer或金标准。

---

## 10. Office普通输入错误：反馈而不是未知副作用

在Office Adapter内明确捕获：非法正则、授权语料根内不存在文件、目录当文件、越界路径等确定输入拒绝。返回`accepted=false、error_code、error、data为空、done=false`，该次调用仍计入额度。

不暴露宿主绝对路径，不为“兼容”开放越界。根语料不存在、挂载消失、编码/存储服务故障等基础设施异常仍向上抛出。Broker不得把所有异常统一吞成普通失败。

对于无返回的在途调用、超时或未知副作用，保留当前停止和恢复规则；不能盲目重发可能已执行的状态动作。

---

## 11. Learner：已有文字指导不代表已有程序

### 11.1 程序意图和作业状态分开

新Skill保存普通`execution_intent=guidance_only|program_requested`。这不是可用资格或证明字段。

- 单轮QA默认guidance_only，不自动为每道问题构建Python。
- 工具型任务中，Agent明确请求程序时记录program_requested；Builder失败不清掉这一意图。
- 旧Skill缺该字段时，不猜全部需要程序。只依据已保存的原始`generate_program`/学习决策补出请求状态，或在后续既有Learner调用中明确选择。

Bank增加一个小型`realization_jobs`表，最少保存：

```text
skill_id / skill_version
kind = build | repair | trial
state = ready | waiting_example | deferred | done
program_id（有则填）
last_error_kind
已选Train案例及其固定输入/起点
repair_used、生成次数、本次触发task
```

它是未完成工作队列，不是新的EvidenceLedger。Frozen不导出可写作业；任务内不会让Test触发Builder补长期能力。

### 11.2 一次既有Learner响应同时决定学习与实现工作

在`submit_learning`增加可选`realization_request`：

```text
skill_id = 已有ID或$new
action = build | repair | trial | defer
case_bindings = 0—2个已完成Train案例的普通输入/起点
```

不新增一个“判断是否适用”的LLM。既有Learner收到相关Skill卡片时，必须同时看到`program_requested / usable版本 / pending原因`，不能只看到Skill文本。

`reuse_existing + generate_program=true`必须进入已有Skill的实现逻辑；生成程序不能继续仅绑定在`propose_skill_and_program_spec`这个字符串上。`no_change`只表示不改指导文本，不能删除或短路既存待完成作业。若已有ready作业及完整试用绑定，调度器照常执行；若缺适用案例，则此次Learner仍需在realization_request中给出可绑定案例或defer，不能用no_change把缺Program误记成完成。不增加第二次LLM来补这一决定。

### 11.3 调度规则固定

每个完成的Train任务后最多执行一个实现作业，优先：

1. 与当前经验相关、已有合法新案例的待试用版本；
2. 当前或新Train证据暴露实际问题的待修复版本；
3. program_requested但尚无程序的待构建能力；
4. 此次明确提出的新程序。

相关性使用同一查询，不另加模型筛选器。适用输入由本次Learner/Builder的普通提交给出，代码做字段、权限和起点可建立的检查。没有合法案例则waiting_example，不为了运行队列强行选任意一题。

每个作业选择/完成都写在当前checkpoint；恢复不能再执行一遍。相同版本与相同物理案例的已完成试用不重新付费执行、不重复计入独立成功。

本题恢复额度耗尽后作业deferred；只有新的相关Train经验、未测试的适用案例或新的真实执行错误才能重新激活。单纯再次扫描同一份材料不能无限重试。Train结束不无限清空队列；未完成项保存为候选，不阻塞冻结本来已可用的资产。

### 11.4 合法试用案例，而不是按整题词重叠硬塞两题

保留最多两个**实际适用的**物理Train试用槽位。

- 必须来自当前run已完成的Train任务；不得读取Val/Test。
- 同一个Task重复两次不算独立。
- 起点只能reset或该案例真实动作前缀重放；不复制旧后验状态。
- 输入是该案例的普通参数，不照搬另一题的对象ID/路径。
- required参数不齐、必需公开工具不存在、前缀无法建立起点：在Program执行前标记inapplicable；不占实际程序试用槽位，不给正向记录。
- 已实际执行但答案错、程序正常未找到或执行失败，必须保留并占该版本已选槽位，不能不断换题凑两个成功。
- 之后出现适用案例可补原本未执行的槽位；修订source或接口生成新Program版本，新版本重新测试，不转移旧成功。

跨family局部能力测试允许，但必须确实具有所需输入/起点。禁止为了Look程序凑第二题，给不含任何可绑定灯输入的Heat案例。

### 11.5 同源、紧凑经验包

Learner/Builder只接收一次：本题公开目标/输入、真实动作与结果索引、程序调用区间、实际提交、允许的训练hard/soft反馈、相关能力及待完成工作。删除`tools`、`execution.history`和多个嵌套副本里重复的同一段经历。

大文档和文件使用已获得资源的相关窗口/引用；生成代码需要的工作表结构等不能用空摘要替代。原始Trace完整保存。公开目录名称与真实可用参数分别表述，避免把观察文本提到的对象名当工具参数。

提示词明确程序支持的入口：若设计成reset起点则显式说明；若希望中途准备复用，应先检查当前已获得结果、当前可调用TAKE/读文件等，而不是永远从头搜索。不得把开发验证题的正确位置、文件名或答案写成代码常量。

### 11.6 程序结果的用途提前明确

Skill/Program的普通接口附带`result_role=intermediate|final_answer|final_files`，由同一次Learner提议确定，Builder不得靠返回值临时提升用途。

- intermediate：可能供后继使用，不因图末尾恰好有一个dict就自动提交为答案。
- final_answer：实际输出含声明的answer字段，提交给锁定scorer。
- final_files：实际输出声明完整解法文件，Workspace与Adapter检查可提交性后封存。

旧资产缺该字段时按intermediate解释，除非旧Workflow已明确声明最终answer/文件输出；不得在Frozen原库中补写。环境型任务仍只看真实终态，不由result_role宣布won。

---

## 12. Builder截断恢复：一次共享机会，不能叠加

### 12.1 触发条件

首次Builder请求仍为32,768最大输出。只有同时满足：

```text
finish_reason == length
且没有完整、合法、可接收的submit_program提案
且本构建流程repair_used == false
且共享学习预算有剩余
```

才允许将本次恢复最大输出设为65,536。若已经得到完整且通过普通结构校验的提交，即使finish_reason为length，也正常使用，不重复生成。

### 12.2 统一处理两种返回路径

`EmpiricalSystem.agent()`既要处理正常AgentTurn返回但无有效ToolCall，也要处理`ProviderAgentProtocolError.usage_turn`携带的截断响应。不能只修其中一个分支。

恢复重新要求**完整提案**，不拼接未闭合JSON、不执行半截Python、不靠字符串截断补括号。材料包含同一Skill、同一已固定案例、必要公共接口、简短截断原因；不复制整段无效隐藏推理或整条历史长Trace。

### 12.3 一个计数器统管结构与执行恢复

```text
首次Builder生成
  ├─ length无有效提交 → 消耗唯一恢复额度，cap<=65,536
  ├─ 结构/语法错误 → 消耗唯一恢复额度，cap保持32,768
  └─ 得到程序并试用
        └─ 需要代码修复且额度尚未用 → 消耗唯一恢复额度，cap保持32,768

任意恢复已用后：不再进行第三次Builder生成
```

Provider transport retry维持原设置，但不当作新的科研采样；真实计费用量全部记录。共享学习余额检查包含首次生成和恢复，不能开新budget_scope规避上限。隔离Trial原有独立运行预算保持，报告时仍归到父训练作业的试用费用，不隐藏。

### 12.4 请求级上限不污染其他角色

为`agent()`提供请求级completion override或等价的不可变请求配置。不要直接修改缓存Provider对象，使后面的Runtime也变成65,536。

实际请求日志和checkpoint key必须包含本次completion上限、repair原因、case集合及作业代次。同一个已落盘响应resume时直接复用，不再次生成。`repair_used`在发恢复请求前持久化；崩溃后不能再获得一个新的恢复额度。

实际请求输出cap不得高于当前剩余共享额度；prompt与completion仍按原预算器计入，返回后若超过原总预算立即停止后续工作，不追加额度。若剩余共享额度不足，结束本次构建并保留deferred，不重新做Train环境交互，不要求每次必须花满65,536。所有输出上限只是上限，不是预先消耗量。

---

## 13. 提示词与实际字段统一：不仅修改自然语言

### 13.1 每个阶段只暴露需要的提交形式

| 阶段 | 内容 | 不再发送 |
|---|---|---|
| Planner | 任务、能力卡片、已有workflow概要、select/compose | 内部证据结构、全库程序源码、多套同义绑定类型 |
| Runtime | 第6.7节局部可读材料、当前允许的单调用/纯读批量、实际I/O | 全部历史正文、每步重复审计字段、相反的“必须单调用”说明 |
| Learner | 一份去重经验、相关实际程序状态、最多一个实现工作 | 相同events在多个嵌套history重复出现 |
| Builder | 单个局部能力、真实工具定义、固定适用案例、入口使用说明 | current/history两份形式合同、合法性见证、验证题答案 |
| Finish-only | 已获得的相关信息、仅answer提交 | 所有工具、程序、replan入口 |
| 单轮QA | 原公开输入、最多3条相关Skill指导、必要图像 | Planner、程序试用、第二次求解或完整Workflow JSON |

### 13.2 结构检查要有边界

保留：实际调用名称、必填参数、类型、结果引用存在、权限、资源限额。非执行`detail/rationale`过长时记录原文并截断展示，不把纯说明超长升级为整题失败；不得截断source、表达式、路径或实际参数来假装有效。

错误必须指向真实字段和原因，如`read.offset_unit_mismatch`、`missing_input.light_source`。不能统一报“验证失败”再要求模型重写全部内容。

本轮不添加LLM judge检查模型是否遵守字段；测试必须截获**实际发送的HTTP messages和tools**与真实handler数据相互验证，而不是只断言某内部变量被构造了。

---

## 14. 六Benchmark适配边界与固定数据

### 14.1 数据清单不重新生成

| Benchmark | Train | Val | Test | Reserve | 交互方式 |
|---|---:|---:|---:|---:|---|
| SearchQA | 300 | 24 | 1400 | 276 | single_answer |
| SpreadsheetBench | 200 | 20 | 180 | 0 | 文件工具＋完整解法提交 |
| OfficeQA | 120 | 24 | 102 | 0 | 文本语料工具＋答案提交 |
| DocVQA | 180 | **22** | **201** | **131** | single_answer，真实图像 |
| LiveMath | 60 | 17 | 100 | 0 | single_answer，问题＋选项 |
| ALFWorld | 120 | 24 | 134 | — | 公开环境交互 |

采用已经通过对齐的实际manifest/hash。DocVQA使用后来确认的180/22/201/131，不回到原目标人数。三个run_seed仍是42/43/44，成员固定、Train顺序按各seed，Val/Test保持canonical。[S4]

### 14.2 各Adapter具体要求

**ALFWorld**：实现第3—6节状态、动作说明与搜索记忆；局部检查只依据本次公开结果。不设Heat专用路线/物品位置常量，不直接把TAKE、HEAT、MOVE拼成测试答案工具。100个原生调用/环境步的现有限制不变，程序仍受既有32-call invocation等限制。

**OfficeQA**：固定全语料范围、glob/read/grep/受限计算；统一字符offset，三项纯读批量，已读资源引用，普通输入错误。24个工具调用上限不变。若产出的完整答案被实际提交，由原EM/F1评分；不换成宽松LLM judge。

**SpreadsheetBench**：保留Python受限执行与当前30次工具预算、完整solution封存及所有case评分；增量manifest、只读检查保留产物、程序试用产出即评分。文件写与`execute_python`不批量；输出文件可提交与输出正确分别检查。

**SearchQA**：保留当前公开question/context及其既定处理，单次solver，指导仅检索Skill。没有可用程序不是错误，不新增检索工具或多轮作答。

**DocVQA**：支持视觉的模型路径必须从原始图像贯通到实际HTTP；结果引用投影不得把图片替换成纯路径或删除content_parts。当前模型能力锁维持原状态，unsupported继续空分且不调用模型；不为本轮改名、切模型、加OCR、解禁能力。其他视觉模型接入时用同一核心单轮路径。

**LiveMath**：固定问题＋选项及原答案格式和scorer，一次solver，不以输出越短为由删除必要题目，不强行拆为图节点或Python程序。

### 14.3 Baseline边界

本次主要修Ours，保留七个baseline原生算法及既有epoch/batch/候选预算。公共工具说明、路径单位、有效反馈或提交修复若由baseline bridge共用，也必须暴露一致含义；不能只给Ours额外真实信息。

Ours的工作记忆、待完成作业和程序优先Executor不自动替换baseline方法。baseline完整六基准可用性仍按其实际适配报告，不因本次Ours通过而一并声明完成。

可用性测试与实际模型性能分开：没有支持视觉的当前模型不能让DocVQA虚假通过实测，也不妨碍其他已支持组合进行诊断。

---

## 15. 配置、checkpoint、Frozen与历史迁移

### 15.1 只引入一个本轮行为版本

保留`mechanism_profile=skillcompiler.empirical.v1`与现有schema。增加：

```yaml
experiment:
  implementation_revision: empirical-v3.1-CF2
runtime:
  read_batch_max_calls: 3
  result_inline_max_chars: 2048
  result_preview_max_items: 40
  result_read_window_chars: 12000
learning:
  builder_truncation_recovery_max_completion_tokens: 65536
```

这是在原section追加字段，不能用这段覆盖整个配置。各默认行为在代码中只有一份常量来源；启动打印resolved_config并验证新字段真正接入，不做一堆可以忘记开启的散落开关。未知枚举在启动前清楚拒绝。

其他设置继承**实际原配置**，尤其Planner/Runtime/Extractor/Builder的high、共享上限、原生/环境预算、worker隔离与最多一次replan。基线default中Planner120,000、Runtime600,000、Extractor/Builder共享262,144；实际Benchmark resolved_config如已有不同锁定值，以原运行配置为准，不借本轮抬高。

### 15.2 不动正在执行的版本

先读取本机matrix和进程；本次上传状态是过去快照，不代表此刻。已被用户停止的run不能自动恢复。需要切版本时按现有STOP_AFTER_TASK等待任务及学习事务结束；新checkout、新输出根、新cell身份。

新策略正式训练从空Bank开始。旧Frozen只读副本可用于固定知识状态的开发对照，明确标记diagnostic，不能当作新策略已训练结果。不能人工把旧candidate改usable或把记录中的旧错误“回填修正”。

### 15.3 Resume必须覆盖本轮新增状态

最少持久化：原始Provider响应、批次子call完成状态、Builder修复额度、待完成作业、选定案例/输入、结果存储索引、已提交workspace指针。

- 已完成原生调用不重发；已保存模型响应不重生成。
- 崩溃时不确定是否完成的状态动作，保留原有中止/独立重建流程，不能复用半个状态。
- 只读结果可从原日志重建；环境状态必须reset并重放真实前缀，不能只恢复缓存就假设环境也恢复。
- 不跨策略版本用旧response_key或resume state；拒绝时说明版本冲突，不改旧hash绕过。
- 用户停止状态保持，不能因为补丁完成自动启动收费任务。

### 15.4 Frozen边界

长期Bank、程序源码、状态和排序在Val/Test前后hash一致。每题TaskContext、临时避用集合、批次结果写run目录；不写Frozen use_count、disabled或新的程序意图。Trial与正式episode使用不同Adapter/工作区，状态不串。

若公开ABI新增字段/单位改变，显式记录新ABI版本并让新程序锁定；旧Bank开发对照通过只读兼容视图，不原地修改其程序hash或转移正向记录。兼容代码只做字段/序列化兼容，不为旧错误程序暗中替换参数或注入答案。

---

## 16. 被动日志：足以定位，不重新增加证明负担

沿用现有请求、Trace、usage、artifact和错误日志。只补运行时直接可取得的字段：

| 记录 | 必要内容 |
|---|---|
| 真实LLM请求 | role、messages/tools、finish_reason、实际completion cap、usage、request ID、repair原因与序号 |
| 原生调用 | 请求/响应、accepted、是否发送、原生额度与环境步、batch及子call ID |
| 状态更新 | event_id、变更字段、known/unknown、进入/退出逻辑state key；不每步另存整套证明 |
| 资源 | result_id、类型、大小、来源event、预览/完整数据位置，字段单位 |
| 程序调用 | program版本、输入、输出、起止原生event、真实status、Agent选择或自动执行 |
| 学习作业 | skill、program意图、pending原因、这次选择的案例、恢复额度、实际试用结果 |
| 提交 | plan完成、finish-only或直接封存、workspace逻辑版本、完整bundle是否存在、scorer原始结果 |
| 完成 | phase完成状态、Frozen前后hash、旧/新代码身份 |

日志不新增LLM；usage缺失记unknown/null，不估零。reasoning只作completion细分，不重复加；缓存命中仍算token数量。保留普通失败与所有真实计费重试，后处理不能只筛成功任务。

私密API凭证不落盘；模型私有推理内容不作为实施所需日志要求，保留已有安全处理和Provider提供的reasoning token数即可。

---

## 17. 防回归验收：针对真实错误序列，不用测试数量代替内容

必须调用生产模块；mock只替代外部模型/环境传输。每个已确认缺陷尽量保留旧代码反例或故障注入，使旧错误重新出现时测试必然失败。

### 17.1 不调用LLM的固定测试矩阵

| ID | 必测输入/序列 | 通过条件 |
|---|---|---|
| T01 | TAKE→MOVE→INVENTORY空 | 持有缓存为空且known；模型视图和Program观察一致 |
| T02 | TAKE→拒绝MOVE | 不移除持有对象，不伪造放置 |
| T03 | INVENTORY无法解析 | 不向模型提供旧列表为确定事实；局部检查不假通过 |
| T04 | TAKE→MOVE后执行取得局部检查 | 不因旧TAKE记录返回passed |
| T05 | Agent调用与Program内部调用相同动作 | 同一个状态/记忆入口，只记录一次事件 |
| T06 | 原生解析→ToolSpec→实际HTTP | MOVE/PUT/HEAT的名字、参数和公开语义没有被中途重写/丢失 |
| T07 | 所有已注册工具handler的有效返回 | 符合实际result_schema；列表/文本不假标为object |
| T08 | grep→read，含CRLF和非ASCII文本 | 返回offset直接读到命中；不把line当offset |
| T09 | 已发布两个文件→files=[]只读检查 | 两个文件仍登记、仍能seal；不空manifest |
| T10 | 登记文件被修改/删除但未声明 | stage拒绝、原提交不变；不是静默继承旧hash |
| T11 | 显式修改/删除、发布中断恢复 | manifest和文件一致，只见完整旧版或新版，无越界链接 |
| T12 | 提交型图最后节点返回最终answer | 立即submit；不启动dynamic_1重新求解 |
| T13 | 最后节点产出完整solution bundle | 直接seal/evaluate；不是LLM自检判正确 |
| T14 | 只产出中间结果 | 不自动冒充最终answer；按原预算继续 |
| T15 | Office第24次read后无answer | 至多一次finish-only，无第25次检索，无结构repair |
| T16 | ALFWorld done/won或single_answer结束 | 不新增finish-only或第二次求解 |
| T17 | 非法正则/缺失只读文件/越界路径 | 普通反馈、计调用、不杀campaign；不泄漏宿主路径 |
| T18 | 语料根故障/未知在途操作 | 仍作为基础设施问题停止，不吞异常 |
| T19 | 2—3个合法Office纯读ToolCall | 按序执行、分别计额度/结果，0次多调用结构repair |
| T20 | 多调用含写/Program/finish、超过3个 | 执行前阻止该批；原一次结构repair，不部分偷偷执行 |
| T21 | 批内合法调用之一正则错误 | 单项普通失败，其余独立纯读继续 |
| T22 | 批执行中耗尽额度/故障/resume | 剩余明确未执行；已完成项不重发、不重复usage |
| T23 | 大glob/文档结果→complete_node | 通过output_refs传递，模型不抄写完整大值 |
| T24 | 结果引用缺失/越题/未来批次/非法path | 普通结构或范围拒绝，不获取新数据、不伪造值 |
| T25 | 大结果原始内容再读取 | 与原始结果一致；preview不冒充全部；单轮QA不新增工具 |
| T26 | 当前Skill有guidance、Program id为哈希 | HTTP可读目标/指导；无全源码/全Bank |
| T27 | 第9个程序最相关 | 进入菜单；Frozen无candidate/disabled |
| T28 | 第1路线缺输入、第2路线ready | 直接执行第2条，0次Runtime调用；不跨授权目标 |
| T29 | found_object→object，随后程序ready | 显式映射一次、自动接续；非法映射不伪造节点完成 |
| T30 | revision递增但状态相同的连续拒绝 | 保留2次限制；真实新状态可以重试 |
| T31 | 已有Skill但Builder曾失败；reuse_existing+generate_program | 实现作业仍被调度，不仅更新文字/Workflow |
| T32 | Look能力只给无light_source的Heat案例 | preflight标不适用且不执行；等待适用Train案例 |
| T33 | 新适用案例到来、同任务重复到来 | 前者补未用试用槽；后者不产生独立信用 |
| T34 | guidance_only单轮QA | 不因缺Program反复构建；只一次正常solver |
| T35 | Builder length＋无ToolCall，两种异常路径 | 一次恢复cap65,536、同budget；第三次生成不可能 |
| T36 | Builder恢复已用后语法/试用又失败 | 不叠加第三次修复；作业deferred |
| T37 | 截断响应已有完整合法提案 | 不重复生成；正常试用 |
| T38 | Builder恢复前后崩溃、其他角色请求 | repair额度不重置；Runtime上限未污染；计费不丢失 |
| T39 | File程序试用直接产出最终bundle | 无整题Dynamic重做；真实变体scorer结果决定正向 |
| T40 | 新题/新Trial/Frozen/三seed/图像路径 | 状态隔离、清单未变、Frozen只读，真实图像保留；不支持模型不伪造通过 |

T06、T19、T23、T26、T35必须检查最终Provider请求/解析路径；只检查helper函数返回不足以通过。T09—T13调用生产Workspace和提交代码；T01—T05调用生产SimpleAlfWorld与实际录制反馈，随后真实环境小链确认。

### 17.2 生产隔离与打包回归

运行当前受影响测试＋现有可执行完整套件；轮子在非checkout目录安装后检查入口。真实Docker验证原有隔离、资源上限、写失败回滚、未知副作用停止。条件缺失标记not_run，不能用普通本地子进程代替Docker后写passed。

这不是新增复杂准入制度，而是对本轮修改的执行行为回归。不得通过删除仍然有效的状态/评分/隔离断言来让测试变绿；已废弃的旧证明政策不恢复。

---

## 18. 真实验证与效果判定：有限次数，不无限返工

### 18.1 先复用已有材料验证确定性行为

本轮提供的录制请求、反馈和文件状态用于T01—T40的反例，不重新请求模型补齐日志。文件类旧模型响应中已经生成的代码可用于定位提交和接口错误；不得把valid/Test代码或答案导入新的learned Bank。

源码缺陷复现和最小对照可以证明某一错误被消除，不证明替换状态后模型会选择同样或更短的轨迹。重放原模型决策不得冒充新模型实验。

### 18.2 一次固定知识状态的执行对照

工程回归通过后，使用原六题的只读Frozen副本、新执行器、同模型high、同数据和原预算，运行原6题一次，新的diagnostic目录保存全部结果。

这一步只检验执行层降本，不重新支付完整Train12。原Bank程序不手改；不把metadata改名后说是新学习。若兼容读视图必要，记录映射和原Bank hash，不迁移成功记录。

六题目标：6/6成功且总tokens≤300,000为目标达成；6/6且总tokens≤360,000为60k上界内；二者未满足如实报告。全部题都计入，不能去掉Heat或失败题。开发题已经用于分析，不当正式泛化结果。

### 18.3 一个真实准备能力的学习补齐验证

沿用已批准的有界局部学习验证：从已完成Train公开经验提出一个有意义准备能力，使用两个不同物理Train案例，验证：

```text
已有Skill但Program缺失的作业可被选中
→ Builder实际产生程序
→ 原定两个适用案例真实试用
→ 同版本取得usable或如实保留candidate
→ 普通Planner/Runtime实际调用
→ 多动作在Program内执行，后继自动接续
```

使用真实Train实例建立起点；不得手写一个通过测试的程序放入正式Bank冒充学习结果。Builder若再次length只按第12节恢复一次；失败后不要自动扩大Train或更换模型。

这个小链是实现机制的证据，不要求一次运行自然覆盖六类ALFWorld任务，更不要求所有Benchmark都产生Python工具。只有工具确实多动作执行时才能报告多动作复用。

### 18.4 文件类和其他Benchmark验证

优先用已提供的Office/Spreadsheet原始请求做结构和提交回归；之后只跑受改动的原smoke任务链一次，保留既定split与实际评分。单轮SearchQA/LiveMath做原定最小smoke；DocVQA用视觉路径的非LLM请求构造测试，当前不支持模型不发额外请求、不改成支持。

不要求小smoke满分，不因一题错就改方法或选更简单题；普通策略失败照常返回且链路结束。真实结构错误、状态矛盾、评分接错或数据泄漏必须修到根因。

### 18.5 三种状态分别交付

```json
{
  "engineering_passed": false,
  "mechanism_observed": "unmeasured",
  "cost_target_met": "unmeasured",
  "quality_target_met": "unmeasured",
  "formal_score": false
}
```

上述是未验证时的示例初始值，不是本轮结果。实现者填入真实状态与产物路径：

- engineering_passed：回归、实际接口、隔离、恢复、评分是否正确。
- mechanism_observed：真实生成程序是否被正常路径使用并自动接续。
- cost_target_met：每个支持的Benchmark分别计算冻结执行成本，≤50k、50—60k、>60k分开标记；不能用便宜QA掩盖昂贵文件任务。
- quality_target_met：同条件baseline结果齐全后再比较，缺对照为unmeasured；不能将历史93%自动当本次同条件指标。

工程通过可以进入有界诊断；效果未达标不能写“已经满足用户目标”。若出现未达标，应明确是哪一条已定义链路未形成，不自动新增另一套架构、证据系统或无限收费pilot。正式矩阵按用户授权在独立新策略目录启动。

---

## 19. 同轮清理：删除已被替代的分叉，保留公共边界

本次不是再次大范围删除工程。完成新接线后删除：

- SimpleAlfWorld里分散的动作名称说明与重复状态更新分支，迁为第3—4节唯一入口。
- Planner/Executor两份`all(program)[:8]`无关菜单生成。
- Runtime每步重复整段history、已完成大列表和discovery审计字段的投影。
- 全局强制“只能一个ToolCall”的Runtime分支，替换为第7节有条件规则；其他角色单提交仍保留。
- 仅凭`decision==propose_skill_and_program_spec`才生成程序的条件；改为实际实现作业调度。
- Builder截断、结构和执行分别独立增加repair次数的分支；只保留共享计数。
- `Workspace.publish`以本次files替换所有登记的旧语义。
- 提交型任务图结束后无条件回到整题Dynamic的分支。
- 已实现并通过测试后，删除与新ToolSpec/结果单位相反的旧prompt说明。

保留：原始日志、旧源码版本、Bank和实验目录、原生执行器、Program隔离、Broker未知副作用保护、独立scorer、三seed与冻结边界。旧报告不改成新结果。

README和有效的项目执行说明只保留本文件版本为本轮实施入口，旧CostFix文档标为superseded；不要删历史记录，也不要让Codex再次拼接相互冲突的旧要求。

---

## 20. 给实施者的最终交付清单

完成本轮后只需要交付以下内容，不另造证明报告体系：

1. **代码提交与改动表：** F01—F13分别改到哪个函数、哪些旧分支已删除；公共接口变动及ABI版本明确。
2. **真实测试日志：** T01—T40逐项结果与生产模块路径；缺运行条件明确not_run；Docker/真实环境与替身测试分开。
3. **实际HTTP示例：** 正确HEAT语义/当前库存、Office两项纯读不repair、Builder截断恢复及真实cap、单轮QA真实内容不丢失。
4. **状态一致性证据：** TAKE/MOVE/INVENTORY、grep→read、只读检查→文件提交、图结束→收尾的输入输出与结果，不只写passed。
5. **一次有界真实对照结果：** 所有任务分数、计费、程序启动/动作、失败原因；分清旧Frozen执行对照与新生成程序小链。
6. **新配置与运行入口：** 已实现并通过`--help`/安装检查的命令、输入路径、输出根及停止/resume方式；不提供尚不存在的参数。
7. **未完成项：** 如仍有工程错误或效果未达标，列真实事实；不以“能结束”掩盖错误，也不因普通低分自动增加新门槛。

源码修改和实际执行均以本文件为准。已经批准的设计不再要求用户反复确认；只有确实触及未批准的模型切换、预算扩张、训练/测试数据改变、评分改变或新外部权限时，先问用户，不自行决定。

---

## 21. 来源、版本与审计口径

### S1：本次新版ALFWorld原始材料

`valid_seen6_materials_20261005_015255.zip`：

- `pilot_terminal_rule/valid_seen/summary.json`
- `pilot_terminal_rule/valid_seen/requests/*_attempt1.json`
- `pilot_terminal_rule/valid_seen/traces/*.json`
- `pilot_terminal_rule/train/summary.json`
- `pilot_terminal_rule/train/frozen_bank/bank.sqlite3`
- `pilot_terminal_rule/selection.json`与运行manifest

逐题统计来自实际requests/usage；不要替换成之前含R10.x `compiler_diagnostics`的旧summary或Heat长Trace。

### S2：本次文件类原始材料

`file_adapter_smoke_bff5350_materials_20261005_023527.zip/file_adapter_smoke_bff5350/`：

- Office Train：UID0187、UID0220；Val：UID0095。
- Spreadsheet Train：120-24、353-29；Val：40757。
- 各自config、execution_manifest、summary、requests、traces、bank/frozen_bank。

源码起点为该运行记录的`bff5350fa7fd170c33d7d6c543e7789cb5c737fd`；缺陷是否仍存在按本文件基线源码核查。包中URI编码的文件名`officeqa%3A...`与`spreadsheet%3A...`按原样使用，不能把目录命名差异当任务ID变化。

### S3：此前已完成的离线核查

`SkillCompiler_新材料逐题核算与源码复现_20261005.json`记录原始文件校验、实际六题用量、唯一Frozen Program及11项离线源码/录制反馈检查。其`performance_target_verified=false`，不把它重标为本轮补丁效果报告。

此前Workspace、Office位置与终止路径的核查均用于确定本文件回归目标；本文不虚构一份尚不存在的新全量测试成绩。

### S4：已冻结的数据对齐报告

`main_experiment_v1_alignment(1).json`：固定六个Benchmark成员、三个Train顺序、DocVQA实际180/22/201/131、scorer hash和Frozen边界。实际源报告记录代码`4f21233…`；不因本轮修复重新切分。[文件内对应各benchmark counts与split_sha256]

### C：本轮源码合同

固定仓库：`https://github.com/YU-S3/AtomicSkill-ToolGraph_v3`  
固定ref：`77959e89afce13e67402b6dd6672f3dbadff0d2e`

关键已核对文件：

| 路径 | 关键现状 |
|---|---|
| `src/atomic_skillgraph/harness/alfworld_simple.py` | MOVE漏更新、语义投影丢失、held局部检查 |
| `src/atomic_skillgraph/harness/alfworld.py` | 原生命令表及已有public_semantics |
| `src/atomic_skillgraph/harness/benchmarks.py` | read字符offset、grep行号、FileAdapter和Spreadsheet提交 |
| `src/atomic_skillgraph/empirical/workspace.py` | publish当前files覆盖登记 |
| `src/atomic_skillgraph/empirical/executor.py` | routes[0]、recent[-3:]、大completed_results、图末尾Dynamic、原生额度收尾 |
| `src/atomic_skillgraph/empirical/learner.py` | 构建decision限定、两案例选择、一次生成/修复流程 |
| `src/atomic_skillgraph/empirical/system.py` | len(tool_calls)==1、length错误返回、阶段预算、试用续程 |
| `src/atomic_skillgraph/empirical/prompts.py` | 当前REF/STEP/BUILD及角色说明 |
| `src/atomic_skillgraph/empirical/program_worker.py` | 隔离、RPC、成功publish及失败discard |
| `configs/default.yaml` | 当前high、角色额度、worker上限与经验式机制固定参数 |

**本文件是实施定稿，不是生产补丁。** 本轮没有自动修改GitHub、启动任务或恢复被用户停止的实验。效果目标必须由实现后的实际运行确认；已知接口一致性问题则必须通过上述反例回归严格拦住。
