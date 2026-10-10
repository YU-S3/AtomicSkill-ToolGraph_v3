# 2026-10-10 统一方法修复与有限验证

实际收费批次固定执行 `a745f569dfaca4b4bd7fb1100e875366005f45d2`；最终交付源码 `79aa1c006e39987d88b664cb2ae75d8c1e49be5d`。后续提交补正式矩阵预算静态分配、provider 参数审计、稳定原生权限及真实操作效果核验（`atomic.local-validation.v2`）；未热更新运行副本，未新增收费批次。本批没有 Builder/Program，后两项资格分支在收费批次未触发；由真实 ALFWorld 零模型回归确认。ALFWorld 不参加本次四项收费批次。

批次状态 `completed`，已评分 60/60，缺失 0。实际 267 次 HTTP、2,147,418 total tokens；局部 Worker 验收 0/24。原评分保留，budget_censored 单独标记。

## 已完成修复

- 非字符串/unhashable choice proposal 叶子统一受控拒绝；learning choice 开启而 runtime choice 关闭时启动前拒绝。B 未运行时配对数 0、收益 null。
- 单答题同一 Runtime 支持直接答题、最多两次临时代码或已验收 Program、读取及消费结果，不新增 Planner。
- 一条真实源局部经验即可构建；独立源参照、版本/环境/合同身份及真实 Worker 执行通过才 usable。仅返回 accepted=True 不足：原生操作名/参数必须真实重放，额外状态写入拒绝。缺证据、纯常量返回和错误文件效果拒绝。跨来源只统计。
- candidate 不进入 Runtime/Frozen；动态节点中间输出不提前结束；M4 消费、替换与丢弃保留。默认局部验收无付费整题 continuation；每 job 源码 repair 合计最多一次。
- 公共原生输入及六适配器接口保留。真实 DocVQA 像素及只读 Worker 做零模型核验；当前文本模型的付费 DocVQA 不接入。真实 ALFWorld 零模型局部操作/验收/继续执行已通过。
- 同一父任务及整批 durable Governor 汇总全部角色、retry/repair/trial；正式矩阵另静态分配声明总额度。provider global→stage→purpose 设置传播，HTTP token 字段及实际参数新增审计。

## 验证与边界

- 主要实现固定副本全量回归：367 passed、2 skipped；两项跳过为未配置旧 CF3 12+6 自然资产路径。
- 固定源码 `8bb36b6b08d0c8bf002672b4916e18beea3a808b` 全量回归：372 passed、2 skipped、0 failed；见 `verification/atomic_unified_qualification_final_tests_20261010.log` 与 XML。中途未设置 CF4_DATASETS 有 9 个资源配置失败，旧材料目录另触发 1 个身份拒绝；换回本次已有的正确材料变量，不修改数据或放宽核验。另一次使用共享 2 秒 Worker 的文件发布正向测试出现 execution_error：隔离重跑通过，正向 fixture 改用生产 60 秒墙钟界限，生产超时和越界拒绝保持原值。失败原日志一并保留。
- 最后身份记录补丁 `79aa1c006e39987d88b664cb2ae75d8c1e49be5d` 将局部政策/答题协议写入 resolved 配置、checkpoint/trace 与 Frozen；设置变化的 checkpoint 在模型请求前拒绝。受影响回归分别 27 与 69 passed。它仅补版本记录与一致性检查，未改收费批次或求解/学习提示，未再重复全量回归。
- 真实 ALFWorld 再现了 GO_TO 来源却仅调用 LOOK 仍因 accepted=True 被验收的缺陷；修复后错误操作拒绝，正确 GO_TO 源验收和 intermediate 后继续执行通过。局部资格 policy v2 使 v1 记录不能在新方法中自动成为 usable；旧记录和评分不改。此次局部效果/文件接口回归 19 passed；最终全量另覆盖补丁最终源码。
- choice 统计/内容身份收尾回归：34 passed、2 skipped（该次未设置历史 LM1 资源变量；历史回放已另通过）；局部 discovery 收尾 9 passed；矩阵预算/参数审计影响分支 67 passed。重复测试数不累加。
- StubProvider + 真实 Worker 检查完整 source→Extractor→Builder→usable→Frozen→非源调用消费；人工 fixture 与截获 HTTP 不计自然 Program 或真实费用。
- 旧 LM1：31 学习响应、34 Val 记录回放；历史 65 HTTP / 688,595 tokens、9 资产均未改。153 Val 匹配与 Train60 排除自身来源选择均 0 曝光；B=`not_run_no_exposure`、paired_n=0、gain=null；此次新增模型调用 0。archive_unchanged=True。

## 实际逐臂汇总

| Benchmark | 臂 | 已评分 | 正确 | soft 均值 | Tokens | budget_censored | Program / 实际消费 |
|---|---|---:|---:|---:|---:|---:|---:|
| searchqa | Train | 3 | 3 | 1.0 | 27,643 | 0 | 0 / 0 |
| searchqa | NoSkill | 4 | 2 | 0.6666666666666666 | 8,014 | 0 | 0 / 0 |
| searchqa | Guidance-only | 4 | 2 | 0.6666666666666666 | 15,876 | 0 | 0 / 0 |
| searchqa | Atomic-full | 4 | 2 | 0.6666666666666666 | 19,186 | 0 | 0 / 0 |
| livemath | Train | 3 | 2 | 0.6666666666666666 | 70,170 | 0 | 0 / 0 |
| livemath | NoSkill | 4 | 0 | 0.0 | 85,321 | 0 | 0 / 0 |
| livemath | Guidance-only | 4 | 0 | 0.0 | 108,029 | 0 | 0 / 0 |
| livemath | Atomic-full | 4 | 0 | 0.0 | 107,924 | 0 | 0 / 0 |
| officeqa | Train | 3 | 0 | 0.0 | 420,954 | 2 | 0 / 0 |
| officeqa | NoSkill | 4 | 0 | 0.0 | 176,482 | 2 | 0 / 0 |
| officeqa | Guidance-only | 4 | 0 | 0.0 | 166,070 | 3 | 0 / 0 |
| officeqa | Atomic-full | 4 | 0 | 0.0 | 183,873 | 2 | 0 / 0 |
| spreadsheetbench | Train | 3 | 1 | 0.3333333333333333 | 381,489 | 2 | 0 / 0 |
| spreadsheetbench | NoSkill | 4 | 3 | 0.75 | 139,944 | 3 | 0 / 0 |
| spreadsheetbench | Guidance-only | 4 | 4 | 1.0 | 102,059 | 2 | 0 / 0 |
| spreadsheetbench | Atomic-full | 4 | 3 | 0.75 | 134,384 | 4 | 0 / 0 |

## 资产与完整配对

| Benchmark | Program 生成/usable | Skill | Frozen 前后 |
|---|---:|---:|---|
| searchqa | 0 / 0 | 2 | 一致 |
| livemath | 0 / 0 | 2 | 一致 |
| officeqa | 0 / 0 | 0 | 一致 |
| spreadsheetbench | 0 / 0 | 0 | 一致 |

Full-NoSkill / normal：n=8，胜/负/平=0/0/8，Full−对照总 token 差=33,775，中位数=1800.0。

Full-NoSkill / censored：n=8，胜/负/平=0/0/8，Full−对照总 token 差=1,831，中位数=1333.0。

Full-Guidance-only / normal：n=8，胜/负/平=0/0/8，Full−对照总 token 差=3,205，中位数=-14.5。

Full-Guidance-only / censored：n=8，胜/负/平=0/1/7，Full−对照总 token 差=50,128，中位数=7154.0。

Train 费用仅计一次，三臂不重复训练。逐 benchmark 的训练摊销与正常配对的描述性 break-even 见 `verification/training_amortization.json`；无在线节省时 break-even 为 null，不能把预算截断当节省。

## 失败与后续决策

本批 Train 的局部证据、提案/拒绝原因、预算延后、Extractor finish_reason 与 Builder 次数见 `verification/learning_and_execution_audit.json`。SearchQA/LiveMath 未执行临时代码，走文字学习；文件类虽有真实局部操作，未形成有效 Program 构建。不能将未观察到自然复用归因为对应 benchmark 永久不适合 Program。
具体断点：SearchQA 2 条 guidance 发布、1 条因额外 rationale 字段拒绝；LiveMath 2 条经来源检查的 guidance 发布、1 条错误来源跳过。OfficeQA 1 条提案因截断的工具参数 JSON 拒绝、2 条因父预算无法准入而延后；Spreadsheet 1 条因未提交唯一 submit_learning 工具调用拒绝、2 条因父预算延后。Builder 为 0，不能将未执行的构建/验收声称为自然学习成功。Val 中各臂的预算截断仍保留已有真实分数。
实际 HTTP 错误：{'runtime_agent_schema_error': 2}。批次终止错误：None。Train 学习状态：{'completed': 4, 'rejected': 3, 'deferred_budget': 4, 'skipped_policy': 1}。

非源 Val 自然 Program 实际消费 0 次。本批未观察到自然程序复用；工程正向 fixture 不能替代这一方法结果。

本次不启动或恢复正式矩阵，不追加 seed/题目/预算。仅 DeepSeek 已配置并参加本批；其他模型按用户要求留空。ALFWorld 付费验证仍暂缓，DocVQA 当前模型不支持图像。是否扩大正式实验需结合本批消费、成本、失败原因另行决定。

金额为 null：本批未冻结价格身份，不能猜测美元费用。原始 cached/reasoning/raw_usage 留供核算。

## 包内材料

`finite_batch/` 保存原始 manifest、逐题 config/execution_manifest/trace/requests、checkpoint 状态与 native events、训练/冻结 Bank、总账及 formal 日志；不复制原始 corpus/数据集或重复 workspace。缓存私有 replay 响应不导出，公开 HTTP 响应在 requests 中。
`zero_model_engineering/` 是明确标记的工程证据；`closed_lm1_readonly_export/` 为旧闭环的只读解释导出。`executed_source/source.tar` 是实际受测 commit 的源码快照。`SHA256SUMS.json` 覆盖包内全部交付文件。
