# SkillCompiler 与 Baseline 六 Benchmark 适配实施文档 v3.1

日期：2026-10-04。**第二阶段：完成《01_核心机制精简重构实施文档_v3.1》并固定 mechanism_release 后实施。**

本文件是v3.0经代码接入审查后的具体化规范，不声称当前已完成六Benchmark适配。v3.1不改变已同意的精简方法、数据量和算法预算；只补齐Provider、最终产物、恢复与原生runner的实际接线。当前主仓库核查基线为 `597cfc8d02906055bb5f28fb96ce6afdea8e2fff`；数据/评分参考源码为 `microsoft/SkillOpt@fa4ca184573e42ec11472959dd57422381418096`。

## 0. 适配原则

六个Benchmark共享第一阶段的Agent生成、普通I/O、程序执行、实际试用和冻结机制。**不再为每个Benchmark制造一套ALFWorld式谓词、效果见证、canonical owner、形式化TaskContract和资格证书。**

Adapter只承担真实任务接口：公开输入、工具、执行反馈、可选的便宜局部检查、结果提交与独立评分。缺少局部检查不阻止运行；语义理解和规划允许由Agent负责。

必须继续区分：程序返回了 ≠ 答案正确；Agent认为节点完成 ≠ 官方整题成功。没有独立局部判断时，通过Train端到端结果测试程序，不声称局部正确性已经被证明。

ScienceWorld不进入新正式矩阵。其旧结果只存档，不新增运行或放行门槛。

## 1. 数据与重复实验保持原方案

| Benchmark | Train | Val | Test | Reserve | 来源 |
|---|---:|---:|---:|---:|---|
| SearchQA | 300 | 24 | 1400 | 276 | SkillOpt固定2000题Universe，保留原Test1400 |
| SpreadsheetBench | 200 | 20 | 180 | 0 | Verified400 |
| OfficeQA | 120 | 24 | 102 | 0 | Full246 |
| DocVQA | 180 | 24 | 200 | 130 | SkillOpt固定的官方validation 10% subset，共534题 |
| LiveMathematicianBench | 60 | 17 | 100 | 0 | 2025-11至2026-02固定177题 |
| ALFWorld | 120 | 24 | 134 | 其余源任务 | Train每类20；valid_seen每类4；valid_unseen全134 |

每个`model × method × benchmark`独立运行seed42/43/44；三个seed使用同一份数据ID清单，从独立初始状态开始。固定Bank的额外重复推理仅作为另一个子集实验，不自动乘进这里。

总Test量每个model-method-seed为2116。三seed不是三套不同划分，也不是选择最好一次。

### 1.1 不在机制改造中悄悄改数据

已经冻结并获得用户认可的manifest优先，适配器按其实际ID加载。表中是既定目标规模；发现与现有正式lock不同，列明实际counts，不在执行中自动重切或补样。

若尚未构造清单，沿用前面已定的SearchQA固定子抽样、Spreadsheet Cell/Sheet、OfficeQA Easy/Hard比例和ALFWorld每类配额。

DocVQA/LiveMath的multilabel划分应注意：`MultilabelStratifiedShuffleSplit`不能被当作严格人数保证。目标必须精确时，在分层初分配后，用确定性的成员移动修正数量：只从超额split移到不足split，优先选标签比例误差增幅最小的一题，平局按`SHA256(split_seed|task_id)`排序，直到配额满足。所有方法共用最终清单，不各自重切。

DocVQA沿用question-level条件，不能描述为document-disjoint测试。备用题不反馈学习，也不为失败任务临时替补。

## 2. 公共Adapter合同：不要继承旧ValidatorChannel要求

建议定义下面的最小接口；名称是待实现合同，可按项目命名规范落位，但语义不能缺省为旧证据系统。

```text
reset(public_task) -> public_observation
available_tools() -> public tool specs / current catalog
call(tool_name, arguments) -> ordinary ToolResult
submit(final_output) -> sealed output reference
close()
```

可选能力：

```text
check_local(call, result, public_state) -> passed / failed / unavailable
checkpoint() / restore(checkpoint)
```

独立Evaluator接口：

```text
evaluate(sealed_output, private_gold) -> raw score + fixed hard/soft mapping
```

Evaluator不注册到Agent/Program可调用工具中。ALFWorld正常公开反馈中的`done/won`可按原接口处理；问答的gold相似度不得在提交前或提交后再反馈给同一求解过程改答。

### 2.1 普通数据对象

| 对象 | 内容 |
|---|---|
| PublicTask | 稳定task_id、goal、公开文本/图像/文件引用、任务接口配置 |
| ToolSpec | 名称、简短用途、普通input schema、output schema、资源权限 |
| ToolResult | accepted/status、真实输出或资源句柄、公开observation、error、done |
| FinalResult | sealed prediction、scorer原始结果、固定hard/soft、usage、结束原因 |

不要求`semantic_predicate_schema()`、`resolve_atomic_effect()`、`contract_matcher()`和完整世界状态证明。ALFWorld已有相关能力可隐藏在适配实现中复用，但不是其他Adapter必须实现的抽象方法。

### 2.2 按能力而不是Benchmark名分支

Registry声明：

```text
interaction: single_answer | tool_loop
input_modalities: text / image / files
tool_surface: exact_catalog | named_tools | none
checkpoint_mode: replay | workspace_copy | none
local_check: available | unavailable
```

Planner、Learner、Runtime不出现按`benchmark==alfworld/...`选择科学策略的分支。基准差别留在Adapter和任务输入profile中。

## 3. 新程序后端在第一阶段已完成，本阶段只连接工具

所有新增Program使用第一阶段的`sandbox_python_v1`，入口`run(ctx, inputs)`。ALFWorld、文件处理、文本检索通过同一个ctx/broker运行。

本阶段新增的是具体工具与依赖，不再开发第二套Tool生成、证明、晋升或执行器。

- ALFWorld：原生动作的当前精确解析。
- Spreadsheet：允许的工作簿读写、代码执行与输出文件。
- OfficeQA：全语料只读glob/read/grep。
- 单轮QA：公开输入及一次答案提交；内部可用确定性文本/格式处理，不隐藏LLM。

隔离进程不能访问答案、Evaluator、其他任务目录或模型密钥。无默认网络；OfficeQA本方案也不用在线搜索。大文件的实际路径由broker映射，Agent只看到授权路径/句柄。

共享容器Python worker默认60秒、1GiB内存；Spreadsheet预先固定为120秒、2GiB，使用同一公开执行环境的所有方法一致。wall包含RPC等待；父进程必须先使在途调用结束或终止其owner，才能恢复Agent，不允许超时worker的旧操作在恢复后继续修改状态。真实环境资源不足是适配错误，不用悄悄更换为更少能力的接口。资源设置在运行前写入benchmark lock，不能根据结果临时放宽。

## 4. 各Benchmark的输入、执行与评分

### 4.1 SearchQA

**数据：** 保留SkillOpt Test1400；Train300和Val24来自先前指定子池，不从Test补入。

**公开输入：** 问题和数据提供的context，不开放联网搜索。沿用锁定上游以`[DOC]`边界截断、默认6000字符的公开输入构造；所有方法收到相同材料。

**执行：** `single_answer`。已有指导可加入本次solver输入，确定性程序可处理文本格式；没有可用程序不强行构图。正常求解只产生一次答案，不先Planner解一次再让Runtime重答。

**评分：** 复用`skillopt/envs/searchqa/evaluator.py`及rollout的最终答案提取；hard为EM，soft为F1。gold answers只给Evaluator。

**局部检查：** 可检查返回类型、已读取公开片段、格式处理是否完成；不需要证明语义检索/推理正确。没有这种局部oracle时，用完整Train任务的最终分数评估候选。

### 4.2 SpreadsheetBench

**数据：** Verified400，200/20/180。同一任务的全部输入变体归同一split；不能当作多个独立训练来源。

**公开输入：** instruction、允许的input workbooks和上游该设置公开的输出定位信息。reference/answer workbook和隐藏评分材料只挂载Evaluator。

**执行：** `tool_loop`，上限30个原生工具调用；Agent普通动态代码和持久Program使用相同Python执行资源。可以用openpyxl等该环境预装库，不把任意代码转换成复杂证据IR。

**输出：** 必须同时封存首个输入的结果文件和可在其他规定输入变体上重运行的`solution.py`（或同ABI的完整冻结解法）。只保存首个xlsx不够；固定上游ReAct流程在其他变体会重新运行solution.py，缺失会记no-solution-py。文件存在只表示生成了文件，不代表工作簿正确。详细封存合同见第12节。

**评分：** 复用锁定上游工作簿评测流程，包括全部规定变体、检查范围、公式重算与任务聚合规则；记录真实评分软件版本。不要另写“打开成功/文件非空=成功”的替代器。

**恢复：** 本题独立工作区；程序写入临时副本，正常完成再提交。失败不得污染下题或下个变体。多个变体的测试方式保持上游评测语义，不给Agent隐藏变体反馈后逐个修答案。

**计数：** 一次代码执行记一次工具调用；程序内部改一万个cell不记一万次环境动作，也不能声称省去一万次LLM。

### 4.3 OfficeQA

**数据/信息条件：** Full246，120/24/102。沿用前面已选择的`officeqa_full_offline_no_oracle_v1`：本地全语料，不用oracle页预选、source_docs提示或联网搜索。

**公开接口：** 问题、全部授权语料根和glob/read/grep。可对实际取得的文本做普通Python计算。题目对应的答案来源标注不能预先缩小检索范围。

**执行：** 最多24个原生工具调用。程序可以批量查找、读取、过滤并返回片段；片段可作为普通输出使用，不必建立`document_window_read`形式化效果。

**评分的明确澄清：** 锁定参考版本`SkillOpt@fa4ca184.../skillopt/envs/officeqa/evaluator.py`实际是答案归一化后的EM和token F1，不是数值相对误差评分。本版默认复用这个实际源码口径，命名为`officeqa_skillopt_em_f1_v1`；hard=EM，soft=F1。前文“原评分器包含数字容差”的泛称不能据此自行实现新阈值。

本设置不能与Databricks另一种信息条件或另一评分器的已发表数值直接混表。若项目已有不同的用户已冻结scorer，实施者必须先报告差异，不静默替换；本文件不授权改用LLM判题。

### 4.4 DocVQA

**数据：** 固定534题Universe，目标180/24/200/130；topic只用于分层和日志，不是答案提示。

**公开输入：** 题目和真实文档图像。图像要以实际多模态消息发送；仅发文件路径字符串不能算视觉接入。图像尺寸/detail处理在所有方法中固定。

**执行：** `single_answer`。视觉理解是目标模型本来的求解调用，不能藏到`read_image`工具后宣称零LLM。没有局部视觉oracle也可以执行和学习指导，不要求先实现形式化视觉效果。

**模型能力：** 不支持图像的实际model ID不能静默调用另一个视觉模型/OCR代理。该模型-数据集组合记不支持并单独报告，不当作答错记0，也不迫使其他已就绪组合停工。

**评分：** 复用锁定ANLS实现：归一化编辑距离达到0.5即该答案分数0，对多个gold答案取最大。soft=ANLS；沿用原适配的hard规则`ANLS >= 0.999`。不能改成`ANLS>0`就算成功。

### 4.5 LiveMathematicianBench

**数据：** 四个月177题，60/17/100，不扩成187题。

**公开输入：** 本次采用SkillOpt的数学选择题形式：question+choices。`use_theorem=false`、`use_sketch=false`；correct_choice与解释只用于Evaluator。不得将其悄悄改成自由证明题。

**执行：** `single_answer`，保留数学solver本身的reasoning配置；不用ALFWorld低成本动作策略限制解题。程序只能完成确定性计算/处理，不凭格式验证宣布数学正确。

**评分：** 复用`parse_choice_label()`及同一Evaluator。当前参考实现em/f1/sub_em均来自是否选对标签，不是自由文本相似度或另一个LLM的判断。

### 4.6 ALFWorld

**数据：** 120/24/134保持不变。

**公开接口：** 当前公共observation、合法原生动作、可选结构化公开发现。Program通过broker执行，不接触hidden object tree。最多100个环境动作，包含Program内部实际动作。

**结果：** 官方环境won是最终成功，不要求所有形式节点同时R1成功，也不以节点自报完成替代won。

**局部检查：** 可以复用现有公开反馈上的取物/移动等简洁判断；不强制为Agent参数生成证据链。不得把“当前不允许TAKE”解释为“刚看到的对象不存在”。

**恢复：** 正常搜索没有结果、普通拒绝继续使用真实当前环境；不自动回滚重搜。隔离Train测试可用现有reset/prefix replay重建环境，物理恢复动作计入恢复开销，不冒充新学习任务。

新正式实验仍从空Bank开始，不能导入人工参考资产或前一seed结果。

## 5. 单轮与多步任务的预算语义

| Benchmark | 正常执行预算 |
|---|---|
| SearchQA | 一次solver回答 |
| DocVQA | 一次视觉solver回答 |
| LiveMath | 一次数学solver回答 |
| SpreadsheetBench | 最多30次原生工具调用 |
| OfficeQA | 最多24次原生工具调用 |
| ALFWorld | 最多100个环境动作 |

本版把“工具turn”明确为实际原生调用；一个持久Program包住多个原生调用时，内部调用仍消耗同一个预算。一次Provider响应包含多条原生调用也逐条计数。上游入口若只有max_turns字段，bridge还须维护实际调用计数，不借助程序批处理获得额外环境预算。

规划、学习、反思和候选评估请求另记LLM成本，不能从总费用中删除。單答任务无效格式按原解析处理，不能借“协议修复”反复重新解题。

## 6. 七个baseline只做数据和执行桥接，不用Ours的新机制替代其算法

方法：No skill、Human skill、Trace2Skill、EvoSkill、EmbodiSkill、GEPA、SkillOpt。

共享：task IDs、公开输入、数据权限、原生工具能力、评分器、运行预算和日志。

不共享：方法自己的检索/反思/候选生成/选择流程。保持各上游原生算法入口，不能换成新SkillCompiler Runtime后仍标原方法。

### 6.1 已定预算，按每个seed使用

| Benchmark | Trace2Skill Train轨迹 | EmbodiSkill每轮chunk（4轮） | GEPA metric calls | SkillOpt batch40/4epochs逻辑batch数 |
|---|---:|---:|---:|---:|
| SearchQA | 300 | 75 | 1800 | 32 |
| Spreadsheet | 200 | 50 | 1200 | 20 |
| OfficeQA | 120 | 30 | 720 | 12 |
| DocVQA | 180 | 45 | 1080 | 20 |
| LiveMath | 60 | 15 | 360 | 8 |
| ALFWorld | 120 | 30 | 720 | 12 |

- Trace2Skill：MAP1、merge5、max merge levels5，一次完整Train evolution；公共Val不做选择。
- EvoSkill：skill_only，failure_samples3、frontier3、最多20 iterations、patience5，完整固定Val；不再按ratio切分。
- EmbodiSkill：四个不重叠chunk合计用Train一次；每轮Val时禁止更新manual，选best snapshot。
- GEPA：reflection minibatch3，`max_metric_calls=6*N_train`，完整Val；metric call按上游定义统计，不等于LLM请求数。
- SkillOpt：rollout batch40、4epochs、gradient minibatch8/merge8、论文式gated slow update、完整Val；末batch不足不补样；训练时不Test。
- No skill/Human skill：不训练，不选Val最佳。Human必须确有非空且来源明确的固定人工Skill；空initial.md不能冒充有效Human对照。

表中SkillOpt的batch数是`4*ceil(N/40)`，不是声称所有batch都产生被接受的更新。实际更新次数由原算法决定。

以上是实验语义。配置适配时核对真实上游参数名，不把文档英文标签直接作为不存在的kwargs传入。

### 6.2 模型和初始化

延续已冻结的实际模型ID；同模型块内算法角色使用同一模型的规则不变。API capability、源码版本和生效参数保存在lock中。不再从过去表格简称猜当前可用ID，不替用户重新选模型。

三个seed各自从相同初始资产独立训练；不能继承其他seed库。已有baseline自己的程序后端可以保留，不强制改为Ours的sandbox程序资产格式；公共资源权限和任务信息必须一致。

## 7. Freeze和实验lane精简

### SkillCompiler

```text
固定Train → 经验学习/实际候选试用 → 写出usable程序和指导/模板
        → 一次冻结快照 → Val只读 → 同一长期库Test
```

原来证明型Train-only Compiler不再是一个必须创作/证明的额外Agent阶段。新Compiler仅做普通装载检查、程序状态筛选、依赖文件打包、固定内容版本和输出Frozen目录。

Val/Test不产生长期学习写入。既定协议允许的task-local临时程序保留，但只活在当前题；不能计为Train学得的程序，也不能加入下题检索。

### 需Val选择的baseline

按各自原Train/Val优化 → best snapshot → Freeze → Test。不把它们改成Ours的Val只读选择语义。

### 不学习baseline

固定状态 → Test。三个重复仍各自运行，不复制一次结果三份。

## 8. 源码接入地图

| 位置 | 修改 |
|---|---|
| `harness/registry.py` | 增加六benchmark工厂与能力声明；新profile不要求旧ValidatorChannel |
| 新增 `harness/simple_protocol.py` | 第2节普通输入、工具结果、提交与独立Evaluator接口 |
| `harness/alfworld.py`或其薄wrapper | 把原生执行/public feedback接入simple接口；旧证明对象不进入Agent |
| 新增 `harness/searchqa.py` | 固定公开context、单答、EM/F1桥接 |
| 新增 `harness/spreadsheetbench.py` | 独立工作区、Python/工作簿接口、原评分器与变体隔离 |
| 新增 `harness/officeqa.py` | 全语料本地工具、无oracle输入、固定EM/F1评分profile |
| 新增 `harness/docvqa.py` | 真图像消息、单答、ANLS |
| 新增 `harness/livemath.py` | 数学选项输入、单答、标签评分 |
| 第一阶段 `empirical/system.py` | 只使用Adapter能力，不引入benchmark策略分支 |
| 新增 `experiments/multibench_manifest.py` | 读取/冻结清单和资源，不生成训练答案 |
| 新增 `experiments/run_multibench.py` | 按method分派原生runner或Ours新System，统一resume/结果 |
| 新增 `experiments/multibench_preflight.py` | 小型资源/模型/评分检查，不重复大规模训练 |
| baseline各自bridge | 原生训练和选择流程的I/O接线 |

上述新增文件可以依现有目录结构合并，但不能复制六份Planner/Learner/晋升器。

## 9. 最小验收和放行

公共必测：隐藏答案不能被Agent/Program读取；同一预测与锁定scorer分数一致；多模态图片确实发送；所有内部调用计入共同预算；程序错误不会污染下一题；三seed状态独立；Frozen长期状态不变；resume不重复任务与费用信用。

每个新Benchmark用固定Train前2题与Val前1题完成一次生产smoke。训练方法仅验证最小原生更新→选择/冻结→预测能闭合；不要求这些题全部答对。ALFWorld复用第一阶段测试，仅对适配改动做回归，不重新开ScienceWorld。

**真正阻塞：** 缺资源、图像未发送、隐藏答案泄漏、工具不可执行、评分不一致、预算统计错误、冻结或恢复错误。

**不作为适配阻塞：** 普通答错、某个局部语义无法独立检查、未学出特定程序、QA主要依赖solver、没有某种形式效果或证书。

逐Benchmark发布就绪状态；一个数据集/模型不支持不阻止其他已就绪组合运行。只有六项都实际通过后，才称全部适配完成。

## 10. 输出文件保持少量

```text
mechanism_release.json          第一阶段版本，第二阶段不暗改
benchmark_profiles.json         输入/工具/评分/资源条件
splits/<benchmark>/*.json       固定ID与数量
models.lock.json                实际模型和能力
baseline_sources.lock.json     上游源码和实际参数映射
adapter_smoke/<benchmark>/     固定小样本结果
multibench_release.json         各组合就绪状态
```

运行时沿用必要被动日志：Run、task、LLM request/usage、工具结果、候选/版本、评分、异常和恢复。日志不新增LLM，也不需要每个数据集生成独立证据本体或证明文件。

## 11. 来源与范围

- 用户前面确认的数据规模、三seed与baseline预算，及《02_六Benchmark适配实施文档_v1.0》。本版保留这些条件，替换其中证明型核心依赖。
- 主仓库597cfc8的`harness/protocol.py/registry.py`仍包含旧接口要求；本文simple接口尚待实施。
- SkillOpt参考源码固定为fa4ca184573e42ec11472959dd57422381418096：`skillopt/envs/{searchqa,spreadsheetbench,officeqa,docvqa,livemathematicianbench}/{adapter,rollout,evaluator}.py`（各目录按其真实文件提供）。
- 本轮重新读取的DocVQA、LiveMath和OfficeQA evaluator确认了ANLS、选项标签和OfficeQA归一化EM/F1语义；OfficeQA rollout确实导入该evaluator。
- Databricks官方README区分OfficeQA Full246、Pro133与ProV2，本版不跟随最新Pro替换固定Full。

固定参考链接前缀：
`https://github.com/microsoft/SkillOpt/blob/fa4ca184573e42ec11472959dd57422381418096/`

该文件是适配设计，不是新的测量结果，不保证所有模型全部支持所需模态，也不保证该精简方法在每个Benchmark都提升。


## 12. SpreadsheetBench实际产物与上游模式锁定

### 12.1 不能只交一个xlsx

已核对`SkillOpt@fa4ca184.../skillopt/envs/spreadsheetbench/rollout.py::process_one`：先在case1运行ReAct；保存`solution.py`；其余case重新指定INPUT_PATH/OUTPUT_PATH执行同一方案；hard要求全部case通过，soft为通过case比例。

因此本项目最终输出锁定为：

```text
solution_bundle/
  solution.py              无Provider调用的可重运行解法
  solution_manifest.json   入口/依赖镜像/代码hash/本题公开输入约定
  case1_result.xlsx        首case实际输出
```

Evaluator在独立目录对其他规定case运行同一封存解法，不调用Agent再适配、不向solver反馈gold比较后改答。相同solution hash用于全部变体。候选程序只解决局部步骤时，最终solver仍需按统一规则生成完整solution.py；不能把某个取表头helper当整题解法。

### 12.2 原生模式必须显式

当前上游`SpreadsheetBenchAdapter`默认`mode=single`，不是30-turn工具循环；它另有`multi/react`分支。延续本项目已经批准的tool_loop条件，bridge显式选`mode=react`与对应原生ReAct入口，不能只写max_turns=30却实际跑single。训练优化算法不变。

这属于本实验统一执行条件，不应声称等同上游默认single结果。若某baseline没有相应原生工具接口，通过其环境适配层接入相同工具，不替换它的优化器。

### 12.3 无gold执行反馈

Test/Val求解过程禁用`use_eval_feedback`和`_auto_verify_output()`一类会返回expected cell值的辅助读出。最终评分函数可以在封存之后使用gold；反馈不能回到当前题继续求解。Train候选选择仍可使用原算法允许的Train分数，但原生执行阶段共享相同非oracle公开信息。

## 13. 多模态与Provider接线范围

修改不止`harness/docvqa.py`，还包括：
- `agents/provider.py`：将当前仅str的DeepSeek验证隔离成dialect实现；新消息支持content parts。
- `agents/session.py`或新轻量session：用户消息不要强制str；一张图不得因日志清洗丢失。
- `empirical/system.py`：按interaction选择single_answer入口，不先进入旧P1/C1。
- baseline自己的model binding：确认收到同一图像、context截断和固定模型ID，不继承默认更强teacher。

锁定`models.lock.json`记录endpoint、实际model_id、支持text/image/tools、允许generation参数、usage字段。unsupported必须是能力检查结果，不能把Provider模型名字符串当能力证明；也不因此新增一个隐蔽OCR/视觉模型。

单答模式：一次正常solver调用；库检索/确定性预处理不新增LLM；没有工具的单答题禁止临时Builder/Planner解题后再solver。Train之后Learner可以更新指导，属于学习成本，不是该题第二次答案。指导学习和程序学习使用同一Bank，但不要求每个benchmark自然出现Program。

## 14. 公共Runner与baseline的真正边界

`run_multibench`负责数据/seed/输出目录/阶段切换和结果汇总；内部通过method bridge调用上游方法。不要让SkillOpt/GEPA/EvoSkill/EmbodiSkill全部变成调用Ours的WorkflowExecutor。

- Ours返回新EpisodeResult，不向旧r103 runner伪造database、Trace和certificate字段。
- baseline把自身evaluation/rollout callback接到锁定的公开任务与scorer；算法的candidate/epoch/frontier逻辑保持原实现。
- resume必须按已完成task IDs恢复，不能因为results.jsonl存在任意一行就把整个batch视为完成。若原生入口只有整batch缓存，bridge校验ID集合，不完整时仅补缺失任务并恢复原顺序。
- 一个模型块的训练者/solver等角色规则不变；实际SDK/工具限制不能通过悄悄替换模型解决。

## 15. 少量评分与资源合同，不重新建立验证本体

每个Adapter的smoke要验证：同一sealed prediction交给原评分函数得到同一raw/hard/soft；这只是回归测试，不增加在线LLM判题。

- SearchQA：锁定`[DOC]`截断后的公开context与EM/F1提取。
- DocVQA：真实图像消息与ANLS边界；图片句柄传给Provider前必须真正解析为图像内容。
- LiveMath：只给question/choices；correct_choice/use_theorem/use_sketch不进入solver。
- OfficeQA：当前Full离线无oracle资料条件与EM/F1口径不变；不自动给题目对应source_docs。
- Spreadsheet：同题全部变体共享物理key；解法封存、重新执行、公式/数值比较按锁定上游规则，缺重算依赖就报资源问题，不改成文件可打开即通过。
- ALFWorld：沿用当前原生parser、合法目录和实际won；不用隐藏对象状态构建Find结果。

这些测试只证明接线，没有替任务内容做正确性证明。原生评分是最终结果的唯一来源；程序测试里的独立局部check是可选项。

## 16. 冻结、状态和预算与第一阶段严格同一实现

Frozen期间不改Program状态/排名/全局use_count；当前题失败只使用episode-local避用表。临时代码只活在本题，不写入下题检索。不同seed、不同benchmark使用独立长期Bank目录。

Program内实际原生操作仍由broker计数。ALFWorld分别记录call attempts与真实env.step，不把Python循环次数当动作；Spreadsheet一次native Python执行是一个工具调用，但隐藏变体执行属于评分成本且必须单独记账，不混为solver额外探索。

训练试用使用新Adapter实例/工作区，不复用已经terminal的训练实例。Val/Test没有“为补usable资格再学习”的特殊通道。

## 17. 本轮已验证与仍需上线验证

本轮已在受控原型测试普通文件版本发布、路径范围、单答只调用一次及图像对象不被str化；也通过真实仓库AlfWorldAdapter方法与受控backend验证了RPC动作接线。**未运行实际六数据集、生产Docker、多模型API或上游baseline完整训练。**

上线验证仍按正文第9节固定的小样本smoke完成，不扩成另一轮大规模预实验。每个benchmark/model实际通过就发布自己的就绪状态；不得将一个fixture的通过宣传成所有模型和所有数据集均已通过。

本版冻结后，不因其他benchmark没有局部谓词、没有多动作工具或单答主要依赖模型而重加旧证据机制。下一轮只核对已经列出的接入项和固定smoke结果。
