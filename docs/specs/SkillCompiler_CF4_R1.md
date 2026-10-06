# SkillCompiler：非 ALFWorld 收尾修复与并行正式运行实施文档

**版本：v3.1-CF4-R1｜日期：2026-10-06**  
**源码基线：`YU-S3/AtomicSkill-ToolGraph_v3@65718eb506c0add39983f77f3a2a8d59b198843c`**  
**受测生产父提交：`8550285a19da9b56a885728fa0ee65eaeab33f41`。**  
**性质：待实施规范；不是已经合并的补丁或补丁后的效果报告。**

> 本轮暂停 ALFWorld 的新增收费实验，保留其代码、已有 Bank、请求、费用和成绩；不为了启动其他基准再跑 ALFWorld 12＋6。只收尾三个已经定位的问题，继续使用 CF4 empirical 主线。修复与生产回归完成后，先并行运行 SearchQA、LiveMath、OfficeQA、SpreadsheetBench 的单模型 seed42 正式 Train→Frozen→Val，正常暂停；同版本恢复执行 Test。DocVQA 保留在总体数据规范中，但当前模型能力锁下记为 unsupported，不启动求解、不用 OCR 替代。

本文是本轮新增修改与启动范围的唯一实施单。CF4 已完成的选项正常化、公共答案契约、范围 grep、实现待办、模型视图、单 cell 入口，以及 CF3 的可纠正规划与已验证 Program 接管均保留。不要重新执行旧缺陷修复，不要从聊天中额外拼入另一套方案。

---

## 0. 结论与范围：先消除误解

### 0.1 ALFWorld 不是“学习完全没生效”

最新真实记录已有 7 个 Program 版本、2 个 usable。Heat 程序在后续 5 道 Train 中实际完成任务；双对象程序在 Val 接管并成功。Look、普通取放、Clean、Cool 各有一个候选通过一次独立 Train 试用，暂不具备 Frozen 的可用条件。[S1]

但固定 Val6 均值为 129,606 tokens，未达到争取 50k、上界目标 60k 的效果要求。Cool 仍是逐步搜索；Heat 程序在本题先消耗 32 次调用并返回 not_found，再由 Runtime 接续。两个 Train 正向表示达到现行可用规则，不保证所有新场景高效，也不证明模型以后必然选择正确路线。增加适用训练有改善覆盖的依据，但没有证据保证增加数量后成本一定下降。[S1]

**因此本轮不扩大 ALFWorld，不为它提高预算或降低可用条件；不删除其数据，也不宣布该方法已经在这一基准达标。** 是否最终缩减论文中 ALFWorld 的模型/方法矩阵，留作后续资源决策；目前只是停止新增运行，不改历史事实。

### 0.2 当前可运行范围不是五个收费组合

| 基准 | 正式 Train / Val / Test | 本轮状态 |
|---|---:|---|
| SearchQA | 300 / 24 / 1400 | 修复单轮学习消费后，运行 seed42 |
| LiveMath | 60 / 17 / 100 | 保留 CF4 完整候选材料，修复单轮学习消费后运行 seed42 |
| OfficeQA | 120 / 24 / 102 | 修复最终提交与 Builder 接口后运行 seed42 |
| SpreadsheetBench | 200 / 20 / 180 | 修复 Builder 接口后运行 seed42 |
| DocVQA | 180 / 22 / 201；另留 131 | 当前锁为 text，unsupported；不是 0 分 |
| ALFWorld | 原规范 120 / 24 / 134 不变 | excluded_by_user_cost，本轮不启动 |

当前单模型可收费启动 **4 个**组合，而不是把 DocVQA 当成第五个已支持组合。代码按 `capabilities.interaction / final_submission_kind / tool_surface` 分支，不按题号、Heat/Cool 或答案写特例。

### 0.3 保持不变的科学与执行边界

- 保持当前 DeepSeek 服务、模型锁、high、各角色额度、两份不同物理 Train 正向、每题至多一个实现工作及原修复次数；不专门改模型命名或视觉能力。
- Builder 首次上限 32,768；length 且无可用提案时，原有一次恢复上限 65,536；所有结构、代码、试用恢复共享该次机会，不新增第三次生成。共享学习总额度不扩大。
- 每题预算不是 50k/60k 硬截断。费用目标独立于停止条件；所有题、失败、修复、隔离试用均保留费用。
- Train/Val/Test 成员、顺序规则、42/43/44 和公共 authority 不变。本轮只调度 seed42；43/44 是后续独立运行，不从 seed42 的 Bank 初始化。
- Val/Test 只读，不更新知识、排序、成功率或程序资格。保留 Program 多动作接管、自动交接、局部修正、一次整图重规划和一次原状态 Dynamic 退路。
- 不增加 LLM reviewer、witness、历史因果证明、四层发布证书或跨节点语义证明。
- 任务正确性只由原评分器决定；正常返回、文件存在、JSON 合法与任务正确分别记录。
- 本轮只改 Ours 和其薄适配接入。现有公共契约包不改变检索语义；不声称比较方法已全部接通，也不以全 baseline 完成作为 Ours 启动门槛。

---

## 1. 修改依据：确定问题与普通效果问题分开

| 编号 | 真实材料/源码中的问题 | 本轮处理 |
|---|---|---|
| R1 | Office finish-only 的文字要求“无工具，输出答案标签”，接收代码却要求 finish_answer ToolCall；有效最终文本被丢弃 | 统一为一次 text-only 收尾，独立评分不变 |
| R2 | Builder 生成阶段与未来 Program 的工具混在材料中；候选 allowed_tools 排除 execute_python，但 ABI/示例仍可展示它 | 三个阶段的接口分开，Program 工具视图与实际权限同源 |
| R3 | 单轮 QA Learner 可只保存 Workflow；solver 只消费 guidance Skill，导致训练产物不进入后续求解 | 单轮能力使用受限 guidance 提议协议，保存后被同一检索函数消费 |
| E1 | SearchQA Val 短答案格式合法但内容不对 | 正常错误，不改评分、不为该题加规则 |
| E2 | Office 最后文本即使接收也不符合金标准；检索/计算质量仍可能不足 | 修接收不重标旧分数；正式运行测效果 |
| E3 | Spreadsheet 某程序只有一次成功试用 | 继续现行 Train 适用案例补齐，不直接 usable |
| E4 | ALFWorld 程序覆盖不完整、搜索效率不稳定 | 暂停新增收费；只保留受影响的无模型回归 |

本轮不得因为修复后仍有普通答错、no_change 或 candidate，就继续扩成无期限架构返工。也不得把可用状态、真实试用或终态保护删掉，来让日志表面上“没有错误”。

---

## 2. 生产修改清单

| 文件/位置 | 修改内容 |
|---|---|
| `src/atomic_skillgraph/empirical/executor.py`，工具耗尽后的 text 收尾分支 | 使用无工具文本接收；明确持久化 finish-only 决策身份与结果 |
| `empirical/system.py::agent` | 增加显式决策 purpose 覆盖；finish-only 文本响应做最小类型/非空/无 ToolCall 检查；保留统一日志与预算 |
| `empirical/prompts.py` | 新增单轮 guidance 专用协议及提示；重写 Builder 的生成期/运行期说明；删去收尾时相互矛盾的要求 |
| `empirical/learner.py::learn` | completed experience 保存后按 interaction 进入单轮学习分支；工具型原流程不变 |
| `empirical/learner.py::_realize` | 首次与修复都使用同一份实际 Program 权限契约；错误工具响应使用原有一次恢复 |
| `empirical/program_worker.py::public_program_abi` 及一个小的同源权限 helper | 从实际允许集合生成未来 RPC 表面、当前样例、资源限制；不新建执行后端 |
| `harness/simple_protocol.py::Broker.rpc/call` | 复用同源“Program 禁止递归工具”常量；保留租约、计数与越权拒绝，不开放 execute_python |
| `empirical/bank.py` | 新增只检索可消费 guidance 的纯查询；普通路由和 Program 生命周期不改 |
| `experiments/formal_log.py` | 记录 guidance 来源ID、产生/复用/检索/注入事实，识别 guidance 提议拒绝；不新增模型调用 |
| `empirical/__init__.py`、`configs/default.yaml` 与校验/身份相关测试 | 更新实现 revision 为 `empirical-v3.1-CF4-R1`；模型视图、数据和公共包版本不无理由改动 |
| `scripts/run_non_alfworld_formal.sh` | 薄调度脚本，调用已有 run_single_cell；4 cell 独立并行，DocVQA排除，ALFWorld暂停 |
| `tests/` | 第 8 节的生产回归；保留既有214项中仍适用的行为 |

若字段在当前仓库已存在，修改现有实现，不造同义字段或重复 helper。新函数在本文中是待新增的明确接口，不表示已经存在。

---

## 3. R1：一次 text-only 最终提交，不能再同时要求 ToolCall

### 3.1 触发条件与范围

只修改 `Executor.run()` 中：原生调用额度已经耗尽、任务 `final_submission_kind == 'text'`、当前没有可直接提交的完整最终答案时的收尾分支。

正常图内 `runtime_step.action=finish` 仍可使用原结构化调用。文件类已经具备完整产物时直接封存；ALFWorld 的真实环境终态处理不变；单轮 QA 原来的一次 solver 不因此多获得一次收尾。

### 3.2 最终请求

以现有 `EmpiricalSystem.agent(..., name=None, schema=None)` 文本路径构造请求：

```python
answer = self.agent(
    'runtime',
    finish_text_prompt(adapter.answer_contract()),
    finish_material,
    None,
    None,
    repair_limit=0,
    owner_state_version=owner_version,
    decision_purpose='finish_only',   # 本轮新增可选参数
)
```

`finish_material` 只包含原始目标、当前公开状态、已获得的工作记忆、近期反馈、完成结果和已可见的必要证据。不得读评分答案。工具额度耗尽后不能发起 read_result 取得尚未送入该次材料的新信息；模型应据已有材料交付答案。

最终 HTTP 的要求：

- 不注册 `finish_answer`、runtime_step 或原生检索工具；`tools` 省略或为空。
- 不设置依赖工具的 forced tool_choice；不修改其他阶段的 Provider 策略。
- 系统提示明确“直接提交最终文本；不检索、不执行、不重新规划”，并附已有 `answer_contract`，例如 `<answer>...</answer>`。
- 仍使用 Runtime 的原额度和 high；无第二次收尾生成或结构修复。

### 3.3 接收规则固定

| 响应 | 处理 |
|---|---|
| `content` 为非空字符串、没有 ToolCall | 原样作为最终预测交给现有 Adapter/评分器；不先判它是否正确 |
| 包含 ToolCall | 不执行任何调用，记录 `finish_only_unexpected_tool_call`，本次结束，无 repair |
| content 为 null、空白或非字符串 | 记录 `finish_only_empty_answer` 或 `finish_only_invalid_text`；空提交由原评分器处理，无补解题 |
| length 且已有非空最终文本 | 保留文本和 length 元数据，按已有最终文本评分；不从 reasoning 补内容、不再生成 |
| Provider/网络/未知费用异常 | 现有基础设施异常与usage处理不变，不伪造预测或费用 |

禁止从隐藏 reasoning 中提取数值，禁止为了给旧失败“补成绩”改写旧预测。历史 UID0127 的文本格式修复不代表其数值正确。

### 3.4 决策与恢复不能遗漏

当前 purpose 由 `name == 'finish_answer'` 推断。移除该名字时必须同时修改：

1. `agent` 新增可选 `decision_purpose=None`，默认保留原推断；仅此调用显式传 `finish_only`。
2. purpose 写入 request、checkpoint、响应身份计算和被动调用记录。scope仍为当前任务/隔离试用，不重置 Runtime 用量。
3. 成功响应通过现有 Executor `save(status='applied')` 与最终预测原子提交；拒绝也提交这次逻辑决策，不能下一轮恢复成未处理响应。
4. 收到响应后进程退出：恢复同一个决策，不再付费；已提交后恢复：直接走后续提交/评分，不再请求一次。
5. 不能仅因调用签名改成 `None, None`，就失去 last_decision_id 或使用上一条正常 Runtime 的决策ID。

`finish_only_protocol_error` 的旧运行读取仍兼容；新实现不再把“没有 ToolCall”当作文本收尾错误。仅当 FINISH 常量不再被任何生产调用使用时删除其生产引用；正常 runtime finish 不删。

---

## 4. R2：Builder 生成权限、Program RPC 权限和本地Python能力分开

### 4.1 三个执行域

| 域 | 实际允许项 | 禁止误解 |
|---|---|---|
| 当前 Builder 模型调用 | 一个 `submit_program` 提案 | 不能当场调用 grep/read/glob/execute_python；这些不是该生成阶段的实时工具 |
| 将来的 Program 的 `ctx.call` / `ctx.available_tools` | 此 Program 版本的真实 allowed_tools 与当前环境可用项交集 | 不能调用权限列表之外的工具；不能递归 execute_python |
| Program 自身Python | 已锁容器中的普通代码、已授权工作区与预装依赖 | 不能访问宿主、网络、密钥或私有评分；不把工作区能力误写成RPC工具 |

Office 程序可在未来通过 ctx 调用授权的 glob/read/grep（包含 paths 参数），数值计算用自身 Python。Spreadsheet 程序直接读取 `/workspace/inputs`、生成结果及完整 `solution.py`；不存在语料或网络默认挂载。文件内部绝对路径与 `outputs.files/deleted_files` 相对发布名仍分开。

### 4.2 权限定义只写一次

在当前 host 侧增加一个小型纯 helper（建议与 `public_program_abi` 邻接），输入 Adapter 的稳定定义、当前定义、TaskContext工具和现有禁止集合，返回：

```text
allowed_names          # 去重排序；与最终 candidate.allowed_tools 完全相同
runtime_tool_definitions
current_runtime_tools  # 先按 allowed_names 过滤，再取少量形状示例
public_program_abi
workspace_capabilities # 公开目录、只读输入、发布契约；不包含宿主路径或秘密
```

现有禁止集合中 `execute_python` 继续保留。读取自身 Python文件或使用openpyxl等不需要把它加入RPC。

要求：

- `Learner._realize` 的 candidate.allowed_tools、首次Builder材料、恢复材料和ABI样例都从同一个返回对象取值。
- `Broker.rpc('available_tools')` 继续只返回当前可用且被授权的工具。不能展示execute_python却在调用时拒绝它。
- `Broker.call` 的权限边界仍然保留：即使模型生成了越权代码也拒绝、计入调用日志，不因提示已经过滤就删除运行守卫。
- 共享禁止名称常量放在 host 可导入的小模块/既有模块中；不要把依赖项目包的代码搬进容器独立worker，使 `python -I /program/worker.py` 无法启动。
- 不是按ALFWorld/OfficeQA猜权限，按已有 ToolSpec、能力类型、工作区和禁止递归规则生成。

### 4.3 Builder 实际材料布局

首次与所有恢复统一为（这些是材料分区，不是新执行协议）：

```text
build_request：Skill范围、输入/输出、入口和固定试用绑定
submission_contract：本次唯一提交工具 submit_program 及真实Schema
future_program_api：未来ctx接口、真实allowed_tools、当前参数形状示例
workspace_capabilities：本地Python和公开文件能力
examples：既有真实Train记录，仅作为历史材料，不是当前可调用工具
previous_failure：仅恢复时提供，注明错误来自生成期还是试用期
```

历史经验中可以出现 `execute_python`，不能删除真实历史。必须标成“该操作过去由Runtime Agent执行；本次Program应把可复用代码写进run，而非递归调用该工具”。**历史记录不是Program权限声明。** 不机械重复整份工具表。

程序ABI保留真实 `current_arguments` 与 named_tools 两种形状、`status/outputs`外壳、实际调用和wall预算。不得把program参数中的`max_steps`解释成执行器额度提升。

两个接线点必须同时处理，不能再次只更新提示正文：

- `model_view.project('tool_builder', ...)` 必须保留上述分区；当前投影对这个阶段是普通透传，不能改名后又被调用方从旧`tools`字段取回宽接口。最终HTTP逐字段检查，不以中间helper输出为准。
- `FormalLog.requests`仍应能读取顶层`examples[].case_id`统计实际试用来源；`FormalLog.asset`当前通过顶层`source`识别程序修订父版本。若新材料把源码放到`previous_failure.source`，必须修改同一个日志读取函数支持该路径及旧日志路径，不复制两份长源码，不丢失父子关系。

`run(ctx, inputs)`中的文件路径必须来自该Program声明的inputs或授权固定工作区，**不假设Runtime execute_python包装器的INPUT_PATH/OUTPUT_PATH全局变量也自动存在于持久Program中**。生成的独立`solution.py`则遵循已有评分重执行的INPUT_PATH/OUTPUT_PATH契约。两者说明分开，不能再从历史execute_python例子直接复制不存在的全局变量。

### 4.4 错误接收与一次恢复

- Builder返回错误工具名或多个提交，记录准确错误：`builder_submission_tool_mismatch`，列出期望 submit_program 与收到的名字；不执行收到的工具。
- 仍使用同一个实现job的原恢复额度。length恢复和普通结构/执行修复共享一次，不新增调用。
- 若首次已经因length用过恢复，第二次又发grep/read，本题仍结束为deferred，不能临时开放工具或第三次生成。
- 已有32,768/65,536上限和共享学习预算不变。恢复材料使用同一future_program_api，不再退回宽Adapter工具表。
- 不需要用AST证明程序中所有动态字符串调用都安全；实际Broker权限是最终边界。可测已知固定错误样例，但不得扩大为又一套形式证明器。

### 4.5 旧Program与测试记录

此补丁不原地更改旧Program白名单、源码、hash或usable状态。正式运行使用fresh Bank。旧candidate可留存做回放测试；需要修复时生成新版本，并重新按现有Train标准验证，不能复制旧成功记录。

---

## 5. R3：单轮QA必须学习可被一次solver消费的产物

### 5.1 分支条件

在 `Learner.learn()` 保存本次真实完成经验后，若 `adapter.capabilities.interaction == 'single_answer'`，进入一个轻量 `_learn_guidance()` 分支。

它覆盖SearchQA、LiveMath，以及未来使用已批准视觉模型时的DocVQA。不是按基准名为某题加答案规则。

该分支：

- 一次Extractor调用及原有至多一次结构/语义修复；继续计入Extractor bucket和共享学习预算。
- 不调用Planner、Builder、ProgramWorker或隔离Program试用；不创建Workflow/Implementation/Program/realization_job。
- 不继续排队或执行旧的工具型pending job；正式fresh Bank不会带入旧job，回放遇到旧job也不在单轮分支补跑。
- `no_change`是合法结果，不强制每题产出Skill，不为追求资产数量追加LLM。

### 5.2 单独的小Schema，不再发送完整WORKFLOW和BUILD协议

新增 `GUIDANCE_LEARNING` 与 `GUIDANCE_LEARNER_PROMPT`。根仍为object，`additionalProperties=false`。统一保留 `submit_learning` 作为提交工具；根据interaction只注册这一份Schema。

```json
{
  "decision": "no_change | reuse_existing | upsert_guidance",
  "existing_skill_id": "仅复用或修订时填写的真实guidance Skill ID",
  "guidance_skill": {
    "goal": "可检索的适用范围描述",
    "guidance": "可复用的简洁文字步骤、原则或易错点"
  },
  "rationale": "可选，说明本次为何保留/更新"
}
```

精确组合规则：

| decision | 必需 | 不允许 |
|---|---|---|
| no_change | decision | existing_skill_id、guidance_skill |
| reuse_existing | 已存在、非空guidance的 guidance_only Skill ID | guidance_skill |
| upsert_guidance | goal与guidance均为非空字符串；existing_skill_id可选 | 模型自造资产ID、Workflow、Program、case_bindings等字段 |

不用新增长度很小的硬上限制造截断/修复。提示要求简洁，但不为压字数丢掉真实方法限定。禁止输出本题专用答案查表；不在文本中塞入评分器金标准、测试题号、私有正确选项标记。仍只使用当前Train经历与已有公共输入，评分反馈按现有范围提供；不增加读取私有参考解的权限。

### 5.3 保存为现有Skill，不新造知识类型

由代码构造而非模型手填以下固定字段：

```python
{
    'goal': proposal['guidance_skill']['goal'],
    'guidance': proposal['guidance_skill']['guidance'],
    'execution_intent': 'guidance_only',
    'result_role': 'final_answer',
    'input_schema': {'type': 'object'},
    'output_schema': {'type': 'object'},
}
```

这些I/O仅维持普通资产结构，不表示单轮solver必须执行程序或输出这个字典；最终答案仍遵守Adapter答案契约。

- 新提议使用`Bank.put('skill', ...)`的内容ID，不让模型指定id。
- 修订指定已有guidance Skill时，内容相同则no-op；内容改变保存新Skill版本，可增加由代码赋值的`parent_skill_id`指向旧ID。旧内容不覆盖。
- 父子关系写入既有artifact日志；不新增生命周期/证书表。若使用`parent_skill_id`，查询时只排除被已存在后继替代的同链旧版本，不能同时注入两份冲突版本。正式Frozen只是读取已保存关系，不写“最新使用”统计。
- `reuse_existing`不修改内容，不制造一次新“成功验证”。
- 提议非法时在原有一次修复内反馈；最终仍非法则`decision='rejected'`，保存响应和原因，不影响已评分任务、不重跑solver。
- 复用 `_receive` 和 TaskCheckpoint；相同已接收提议的恢复应幂等，Bank put不重复创建不同内容ID。不能因为新增分支而绕过日志或checkpoint。

### 5.4 生产与消费使用同一guidance查询

在Bank中增加纯查询 `retrieve_guidance(query, limit)`：

1. 从skill资产中先筛选`execution_intent == guidance_only`且guidance是非空字符串的可消费项。
2. 按现有词匹配排名、ID打破平局；先过滤再截断。旧Workflow不能占满前8个后又被丢掉。
3. 单轮Learner相关材料取现有上限8；solver保持最多3条，不增加默认上下文规模或额外selector。
4. solver接收 `{skill_id, goal, guidance}`；ID写入被动使用记录，原文直接注入同一次请求。材料已有图像则保留原content parts，不新增OCR。
5. no_change导致Bank空时，guidance=[]正常；有不相关Skill时不声称一定受益。受控测试使用确实相关的Skill，检查注入而非只检查落库。

不把历史的未消费Workflow自动转换为新训练成果。旧运行仍保留为旧记录；新fresh训练从受限协议产生正确产物。

### 5.5 被动记录最低要求

在现有`learning`与正式日志中记录：

```text
decision、persisted_skill_id、reused_skill_id、parent_skill_id（如有）
retrieved_guidance_ids、injected_guidance_ids
learning_rejected_reason
```

单轮无Program不是方法失败；但若多题选择学习且产生的产物从不进入求解，应能从日志查出。`FormalLog.learning_end`不能把`decision='rejected'`误写成成功学习更新；已有episode评分保持不变。

上述记录不需要LLM，不增加新的反思器或成功判定器。

---

## 6. 这是收尾，不代表其他方法效果已解决

三项修复分别消除“正确文本被协议丢弃”“生成材料与未来权限相矛盾”“学习产物不被消费者使用”。它们不保证模型检索正确、生成代码无错误或guidance一定提升精度。

Office已经发生的错误数值仍是错误；SearchQA合法短答案选错实体仍是错误；单轮数学一次输出length仍按现有结果处理，不额外解题。Spreadsheet一题67,100不能外推全Test均值。ALFWorld不因暂停而被标记达标。

本轮不顺带实施新的搜索算法、提升程序预算、追加Builder重试或调整晋升门槛。需要对效果负责，但不能把每一次普通模型失败变成新的工程放行门槛。

对未来结果的判定：

- **工程问题**：字段、接口、权限、恢复、评分或数据隔离不一致，修到具体模块。
- **机制问题**：产物已生成却未保存/未检索/未进入消费路径，定位接线与可达性。
- **效果问题**：合法完整路径下模型答错、选择不佳或程序在新任务不适用，保留为真实结果，不改历史分数。

---

## 7. 版本、兼容与清理

### 7.1 固定新运行身份

将实现revision更新为 `empirical-v3.1-CF4-R1`，在当前常量、默认配置、严格校验及入口中一致更新。**不要只改YAML而保留POLICY_DEFAULTS旧值，导致新入口被拒绝。**

本轮未改变CF4模型视图编码、LiveMath材料、grep语义或公共包，因此其现有版本/hash保持；只有实际发生变更的部分才更新。正式manifest继续记录真实源码、配置、公共包、材料和语料hash。

### 7.2 保留且不得重做的内容

CF4的完整选项、单次答案格式、paths省略/空数组/非法路径语义、40总命中、原始offset单位、纯读每批最多3个且逐项计费、文件增量发布、解法封存和全变体评分不变。

CF3的三种execution_mode、参考与绑定分离、真实Program多动作接管、自动交接、patch事务、一次重规划/退路以及逻辑决策恢复不变。

ALFWorld仍保留可导入和既有受影响测试，不再次大规模清理目录。暂停它不要求重写全局benchmark枚举或公共authority。

### 7.3 只删除被替代的分支

- 删除text收尾的finish_answer强制接收及相互冲突的提示；旧日志读取兼容不删。
- 删除Builder首次/恢复各自拼接不一致工具列表的重复代码；使用同源helper。
- 单轮QA不再构造工具型完整LEARNING Schema、不写Workflow、不调Program作业；工具型路径保留。
- 不创建用于隐藏旧资产的批量删除脚本，不修改旧Bank与旧checkpoint，不把smoke库接成正式初始库。

---

## 8. 必须完成的生产回归：小范围、无新增收费pilot

下表是实现后要求，不是本文声称已经通过的生产成绩。既有214项仍适用的测试保留；模型路径断言随新契约更新，不用统一skip掩盖回归。

| ID | 实际调用路径 | 必须断言 |
|---|---|---|
| R01 | Executor额度耗尽→真实agent接收→Adapter提交 | 合法最终文本成功进入原评分器，tools为空，只有一次收尾请求 |
| R02 | 同一路径，content空/意外ToolCall | 不执行任何动作，不补请求，明确终止并保留usage |
| R03 | content含错误数值但合法格式 | 不因格式修好直接hard=true，评分仍为0 |
| R04 | finish-only收到后中断恢复 | 同逻辑ID复用一次；提交后恢复不重复付费/执行；purpose仍finish_only |
| R05 | 文件ready和环境done | 不误走text收尾，不重做任务 |
| R06 | Builder首次及每种原恢复路径 | 原生HTTP唯一工具submit_program，未来工具只是数据说明 |
| R07 | Program契约helper→candidate→Broker | allowed_names一致；execute_python从所有未来RPC表面排除，但Runtime原执行工具仍在 |
| R08 | 录制的Office Builder错误grep/read响应 | 不派发，首次可用原repair，恢复仍错则停止；最多两次生成 |
| R09 | Spreadsheet持久Program直接Python | 使用真实worker/工作区/封存链完成受控文件产物；越权递归仍backend_invoked=false |
| R10 | Program含正常read_result及Office scoped grep | 授权项不因过滤误删，paths传递/原调用预算保持 |
| R11 | 单轮Learner upsert→Bank→freeze→新任务solver | 同一guidance内容和ID实际进入一次最终HTTP；无Planner/Builder/Workflow/Program |
| R12 | 单轮no_change/reuse/rejected | no_change不造资产；reuse不改版本；一次repair用完返回拒绝，episode分数不变 |
| R13 | 单轮模型仍提交workflow/realization_request | 专用Schema与纯语义校验拒绝，不持久化半个Workflow或排队Program |
| R14 | guidance修订及checkpoint重放 | 新内容新ID、旧版本只读；同一提交不重复写入；检索不同时注入被替代同链版本 |
| R15 | Bank同时有8个Workflow和相关guidance | guidance查询先过滤后top-k，相关Skill不被Workflow挤掉 |
| R16 | LiveMath与DocVQA公开输入 | 完整选项、私有金标准隔离；图像保留；当前锁仍unsupported，零模型调用 |
| R17 | 现成CF3/CF4多程序自动接续夹具 | 已授权Program连续接管和动态纠偏保留；不是所有节点改为dynamic |
| R18 | 单cell CLI/模块/安装路径 | `--help`、版本、参数一致；支持stop-after-val与同版本resume |
| R19 | 四个隔离伪campaign并行 | 输出/Bank/checkpoint/临时目录不串；一项失败不终止或重启另三项 |
| R20 | 启动计划与resume | ALFWorld无子进程；DocVQA当前无子进程；重复输出拒绝或显式同版本resume；Val完成不被记Test完成 |
| R21 | 原始usage与新记录字段 | finish-only/Extractor/Builder/试用归属正确；恢复不重复计费；未知费用不填0 |
| R22 | 固定数据和预算快照差分 | 题目成员、数量、high、修复次数、Program门槛、worker和native预算未变 |

执行顺序：定向生产回归→现有完整套件→入口/安装检查→固定干净提交→正式启动。无需再做一轮收费2＋1或12＋6来确认本可用现成响应复现的格式问题。

**不得以以下指标作为新增工程门槛：**某一题必须答对、两道Train必须学出usable、所有正式Val必须≤60k、每个基准都必须生成Python工具。相反，字段错配、权限放宽、图像丢失、评分错接或学习产物永远不可消费必须阻止相应cell启动。

---

## 9. 并行正式启动：除去ALFWorld，先4个支持的seed42

### 9.1 不修改总规范来筛选本轮运行

`configs/main_experiment_v1.yaml`仍保留六benchmark与三个seed的规范。当前`run_single_cell`会校验这份总规范；不能为了排除ALFWorld把列表改短，否则入口会报错。[S7]

通过已有单cell入口选择本轮四个组合，**不调用会枚举全部6×3的run_formal主入口**。

每个组合：

```text
fresh empty Bank
→完整固定Train（每题完成一次原定流程）
→Frozen
→完整Val
→awaiting_test正常暂停
```

本轮四组合：`searchqa / spreadsheetbench / officeqa / livemath`，模型沿用当前锁，seed42。四个组合相互独立；每个组合内部仍按固定任务顺序，不并行训练同一个Bank。

### 9.2 启动前一次性检查

1. 当前修改已提交，源码干净；公共包、依赖和容器镜像都已存在，不能临时拉另一个模型/镜像。
2. 使用CF4材料：`/home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1`。以manifest/hash实际检查为准，不凭目录名声称有效。
3. Office语料路径在本次真实配置中为：`/mnt/d/T3S_exp/SkillCompiler_resources_20261003/raw/officeqa/treasury_bulletins_parsed/transformed`。启动时确认授权路径存在，不能自动换语料。
4. 四个输出目录全新；不存在同cell活进程。未跟踪目录里不能有另一份遮蔽src的可执行代码；确认安装CLI实际导入当前源码。
5. 每个进程独立TMPDIR，文件评分/解法回放使用自己的workspace/临时目录，语料和公开材料只读共享。
6. API并发、机器RAM/CPU和磁盘足够。默认4个cell进程，无额外每题并发。资源不够可少同时运行进程，不能改科学预算来适应机器。
7. 旧 `formal_main_experiment_v1_20261005/matrix.json` 不续用、不清空。本轮输出单独组织，避免旧queued任务顺便启动ALFWorld。

### 9.3 确定可用的单cell命令

以下CLI及参数在基线实际存在。[S7] `CODE`必须指向实施完成后的**新、固定、干净源码目录**，不能热修改正在运行的旧checkout。

```bash
export CODE=/home/yangchengyu/asg_cf4_r1_20261006
export PYTHON=/home/yangchengyu/asg_alfworld_venv/bin/python
export DATASETS=/home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1
export CORPUS_ROOT=/mnt/d/T3S_exp/SkillCompiler_resources_20261003/raw/officeqa/treasury_bulletins_parsed/transformed
export ENV_FILE=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
export MODEL_KEY=deepseek-v4-flash
export OUTPUT_ROOT=/home/yangchengyu/formal_cf4_r1_nonalf_seed42_20261006
export EXPECTED_SHA="$(git -C "$CODE" rev-parse HEAD)"  # 仅在补丁验收、提交完成后设置

# 先查看四条命令和本轮排除项；不调用模型。
bash run_non_alfworld_formal.sh

# 确认CODE是已验收CF4-R1之后执行。终端/会话应保持运行，或置于本机已有会话管理工具中。
bash run_non_alfworld_formal.sh --execute
```

随附脚本不修改源码/Bank，不包含密钥；默认dry-run。`--execute`才启动四个真实cell；脚本使用每cell外部锁、独立日志与TMPDIR，一项退出不会自动重跑它或终止其他项。启动时仍由现有campaign做源码/配置/数据hash验证。

**脚本不是生产补丁，不能在修改前用它绕过本文第8节。** `PYTHON`路径名称含alfworld只表示现有环境名称，不会因此启动ALFWorld。DocVQA不创建求解进程；排除状态记在本轮计划说明中，不伪造completed。

### 9.4 Test和后续seed

四个cell到 `awaiting_test` 后，先核查版本、调用/费用完整性、Frozen未变和是否存在确定工程故障。**不能根据Val高低选择一个更好的Bank、改参数后仍冒充同版本继续。**

同版本的Test使用原输出目录和同一命令，去掉`--stop-after-val`、增加`--resume`。例如：

```bash
cd "$CODE"
PYTHONPATH="$CODE/src" "$PYTHON" -m atomic_skillgraph.experiments.run_single_cell \
  --config configs/main_experiment_v1.yaml --benchmark searchqa --seed 42 \
  --datasets "$DATASETS" --output "$OUTPUT_ROOT/searchqa/seed42" \
  --model-key "$MODEL_KEY" --env-file "$ENV_FILE" --resume
```

Office命令另带`--corpus-root "$CORPUS_ROOT"`。不是重新Train、不是新attempt，不复制其他cell的Bank。若发现确切需要改代码的错误，受影响cell使用新版本新身份；不能篡改旧manifest绕过一致性检查。

最终需要42/43/44的正式重复时，43、44各自fresh Bank和独立输出。当前只先启动seed42，不因单seed较好就把它当三seed结论，不自动展开多个模型。

### 9.5 运行中暂停条件

只对实际工程问题暂停对应cell：输入缺字段/金标准泄露、权限异常、未知副作用、版本不符、费用无法对齐、已保存guidance按相同查询也不进入消费路径、恢复反复执行已完成操作。

一般答错、正常not_found、no_change、candidate等待案例、个别高费用题不是基础设施故障；如预算承受不了，明确在任务边界暂停并保留全部已花费用。不要选择性删题或把贵的失败记为“不计入”。

不得停止一个cell时执行`pkill python`或删除所有Docker容器；使用该cell的PID/容器身份。未知在途副作用按现有恢复规则人工确认，不盲目自动重试。

---

## 10. 交付与放行记录

实现者应提交：

- 生产补丁commit与源码hash、实际配置差分、仍保留的模型/预算/数据hash。
- R01—R22对应的测试节点、真实stdout/JUnit；旧行为复现与新行为对照，标明测试替身与真实worker的范围。
- 文本收尾最终HTTP、Builder生成/未来RPC权限对齐样例、guidance保存→冻结→注入同一次solver请求的完整受控链。
- 四cell的准确启动命令、独立目录及排除清单；命令与实际CLI一致，不打印密钥。
- 明确状态：`implementation_ready`、`natural_learning_observed`、`cost_quality_measured`分开，禁止只写“全部通过”。

最终费用汇总按benchmark分别列Train执行、Extractor、Builder/修复、隔离试用、Frozen Val/Test及未知usage，不以低成本SearchQA稀释高成本OfficeQA。算法效果仍以完整独立评分为准。

**本轮正式启动放行条件：三个收尾点与对应生产回归完成，原边界保持，四个受支持cell可在固定版本独立运行。不是要求先证明所有benchmark达到50k或最高准确率。** 这两个是需要后续真实结果检验的方法目标，而非文档或少量受控测试能够预先保证的事实。

---

## 11. 可追溯来源与本轮核验界限

本实施单依据以下原始材料和固定源码，未访问新的收费运行：

- **[S1]** `CF4_completed_smokes_review_summary_20261006_184904(1).json`：ALFWorld Train/Val、程序正向来源、Heat未命中、四个单例候选及其他基准成绩；`CF4_65718eb_正式启动核查与剩余问题_20261006.json`为上轮独立审计。
- **[S2]** `CF4_交付报告.md`、`SkillCompiler_CF4_review_20261006.zip`：受测版本、214项工程回归、公共材料和入口，不把它们冒充本补丁后的测试。
- **[S3]** `empirical/executor.py`工具耗尽收尾；`empirical/system.py::agent`目的推断、响应接收、usage、checkpoint及single_answer路径。
- **[S4]** `empirical/learner.py::_realize`、`program_worker.py::public_program_abi`、`harness/simple_protocol.py::Broker.rpc/call`：Program权限与当前Builder材料。
- **[S5]** `empirical/prompts.py`的SKILL/LEARNING/BUILDER_PROMPT；`learner.py::learn`与`bank.py::retrieve`：单轮产物和消费者错配。
- **[S6]** `benchmark_profiles.json`、`configs/main_experiment_v1.yaml`、`models.lock.json`、实际Office train/config.json：固定数量、预算、模型锁和路径。
- **[S7]** `experiments/run_single_cell.py`、`run_formal.py::campaign`及`formal_log.py`：单cell参数、Val暂停、同版本resume、记录与异常处理。

以上源码基准统一为本文首页65718eb完整SHA；仓库位置为 `https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/tree/65718eb506c0add39983f77f3a2a8d59b198843c`。所有“新增”函数/字段是本轮实施要求。

本轮附带22项参考检查已通过，仅验证轻量函数的边界和调度参数；它们并非R01—R22的生产集成结果，不修改生产仓库、不运行真实模型、不执行新ALFWorld。它们不能代替R01—R22的生产集成，也不能证明新程序覆盖或最终成绩。后续发现事实错误应按本版接口修复，不再把一次普通模型失败扩展为下一套架构。
