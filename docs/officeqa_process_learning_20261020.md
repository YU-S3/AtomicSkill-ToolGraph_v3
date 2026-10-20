# OfficeQA：向已验证人工 Bank 能力收敛的自然学习链路

本轮正式流程是 **空 Bank → 全部 canonical Train120 → Frozen → 既有 Val24 → 停止审查**。没有 Train 内部 selection、反馈池或 gate 子集。来源重放、参数变化检查及离线诊断属于工程测试，不改变正式数据成员。

人工 HumanStrong V4 Bank 保持冻结。它在已完成的同条件 DeepSeek Test102 中为 69/102（67.65%），NoSkill 为 59/102（57.84%）。这证明该人工资产在该次实验中的效果；自然学习版本是否超过 65%，必须另外验证。

## 本轮实现

- 新方法身份为 `officeqa.process-learning.v1`。自然 Bank 从空库开始，不导入人工 Skill、Program 或 Test 轨迹。
- Runtime、Learner、Builder、隔离来源试用共享同一公开题面和完整提供页。公开页有内容 hash 和实际 `result_id`；来源试用只恢复初始公开页，再重放真实前缀，不预载待验证的未来输出。
- 使用与 V4 双臂相同的输出格式合同、48 次工具额度、216,576 求解 tokens、正常 32,768 completion，以及禁用 thinking 的 32,768 收尾 completion。收尾格式合同在容量测量前加入。
- Learner 可以积累互补的 guidance 和 Program；不再限定只生成检索程序或一张全局策略。可学习实际执行过的检索、表格处理、时期/单位绑定、计算和有依据的答案输出。
- 每题仍至多一个提议与一个 realization job。新引用用 `$new`，已有引用必须是真实 ID；既有单次结构 repair、Builder 初次生成加一次 repair、最多两个真实来源试用槽保持不变。
- 材料按完整操作选取，保留已选操作的完整代码、公开业务输出、题面和源页，并记录未选操作。128 KiB 是完整请求容量上限；不会把半张表伪装成完整证据。容量不足会保留零 HTTP 拒绝收据。
- Builder 收到完整公开题面、源页、原执行代码、Host 输入绑定及实际观察到的目标输出。Program 只有通过真实 Docker 来源试用后才可调用。
- 对可变数值/数列参数，再改动一个已执行字面量，独立执行原 Python 和候选 Program 比较结果，拒绝仅记住样例输出的程序。这是测试，不是新的 Train 划分。
- 整题失败仍可提供成功的局部操作；`final_answer` 另外要求来源题正确，且存在实际执行过的非空 `answer:string`。不能把中间数字或页面文本改名为答案。
- Runtime 记录实际 guidance ID 与适用条件，忽略已被新版 guidance 替代的旧卡；Program 使用原有资格检查和真实工具调用。Val 使用只读冻结库，并检查前后 hash。

## 数据、费用与入口

公共资源沿用已批准的 `officeqa_external_minimax_resources_20261020_v1`。Train/Test 有完整外部题面与源页；Val24 仍是此前明确允许的、按 ID 对齐的本地旧版本，没有外部源页。Val 只用于内部验证，不能声称与外部 Test 输入等价，也不能将其写成论文中的外部版本结果。

正式入口是 `experiments/run_office_process_learning.py`。它校验 canonical 成员、固定 seed 的 Train 顺序、完整资源 hash、源码身份、空初始 Bank、累计费用以及父账本锁；不自动运行 Test。

Train 每题总上限 400,000，其中求解 216,576、学习 183,424；Val 每题 216,576。Train120 + Val24 理论总封顶 **53,197,824 tokens**，不是预期花费。新的正式运行必须在结转账本后能够完整预留，不能借用正在运行的人工 Bank 重复 Test 的余额。

两次额外人工 Bank Test 使用原 `7d93d14` 实现、原 Bank、DeepSeek 和原 seed42，按顺序启动。控制器为 `experiments/run_office_humanstrong_v4_repeats.py`；每轮启动前按上一轮实际结算费用检查 155M 原累计上限。自然学习源码修改不会热更新这两轮实验。

2026-10-20 用户已批准累计 **190M tokens**，离线验收后启动上述自然学习流程。`after-repeats` 入口先等待两次重复 Test 完成及新增计费全部结算，再从 `repeat_3` 账本结转历史费用和保留项，完整预留 53,197,824 tokens；实际累计仍超 190M 时拒绝启动。等待期间不分配费用、不发模型请求。启动状态保存在运行目录同级的 `.launch.json`，`status --root <运行目录>` 同时读取等待状态、逐题进度及真实账本。

## 验证口径

离线门禁覆盖：完整 Train120/Val24 成员；真实 Train12 原材料请求回放；完整公开页和来源作用域；Learner→Builder→Docker→后续输入调用；拒绝常量样例程序；失败整题不能晋升答案 Program；收尾容量包含格式合同；Val 不学习且冻结 hash 不变。离线模型响应为测试替身，不算自然学习成功率，不新增模型 HTTP。

付费自然学习的报告应区分：学习尝试、结构拒绝/容量拒绝、生成数、来源试用通过数、后续调用数、独立题复用结果、各阶段 token/HTTP，以及完整 Val24 结果。不能用资产数量或测试替身通过来代替准确率。
