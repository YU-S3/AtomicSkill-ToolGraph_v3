# OfficeQA 任务内程序采用策略修复与离线验收

本轮完成 Runtime 程序采用策略的最小修改和零模型请求验收。正式 OfficeQA 继续暂停；本轮没有付费调用，也没有重跑旧任务。工程回归通过不代表自然模型已经采用临时程序或实现降本。

## 修改依据与实现

已完成的 `5a1fd6c` 单题观察中，临时 Python 的检索入口确实送入模型，但 Agent 没有使用它：30 次原生调用、287,988 tokens、最终答错；Learner 使用 4,333 tokens 后选择 no_change。缺少自然程序采用证据，不能归因为学习 pool 耗尽。

新实现为 `empirical-v3.3-task-program-priority`，Runtime 投影为 `empirical.model-view.public-windows.v7`。学习材料仍为 v10。

- 将执行优先级放在 Runtime 提示开头：兼容且输入就绪的 usable Program 优先；否则，有实际临时执行权限时，循环检索、筛选和读取优先由 Agent 编写有界脚本处理。孤立检查、未知输入或尚不能在代码中处理的反馈仍允许普通工具调用。
- Runtime 的 `calls.task_local_program` 只在适配器实际声明 broker-bound Python 且工具当前可用时出现。它公开现有 ctx 方法签名、可调用工具名、Worker 的真实内部调用和时间上限，以及与当前题共享预算的事实。普通计算入口不会因此获得检索权限。
- 提示说明提交顶层脚本、在脚本内处理依赖调用及循环、检查 accepted/error 和剩余调用数、输出有来源的紧凑 JSON。脚本不自动写 Bank；学习、独立来源试用和晋升继续走既有流程。

没有题目 ID、答案、Treasury 表名或搜索词规则进入通用代码。没有修改 scorer、gold、预算分配、repair 次数、晋升标准、旧 run、旧 Bank 或费用账本。没有新增模型决策或 LLM 请求阶段。

## 验收

在本地 Ubuntu WSL、现有虚拟环境和 digest 固定的真实 Docker Worker 中：

| 核验 | 结果 |
| --- | --- |
| 临时检索、来源学习与 R3 回归 | 22 passed |
| 共享执行、预算、文件、复用、Runtime handoff 和已有 LiveMath 生成合同回归 | 186 passed，9 initially skipped |
| 配置真实语料和正式前缀后补验 | 2 passed |
| 固定 v5 绑定归档的额度与一次修复回放 | 2 passed |
| 上述核心与共享检查去重 | 212 passed，5 个历史归档测试未配置 |
| 合入 main 的 LiveMath R4 后相关集成检查 | 20 passed，10 skipped，与上述有重叠 |
| 全部检查按测试名取最后状态并去重 | 218 passed，0 failed，14 个指定历史归档测试未配置 |

新回归经实际 Provider payload 组装和模拟 HTTP 响应，确认空 Bank 请求中具备真实 API 和限制；脚本在 Docker 中 glob→grep→read，其启动与三次内部调用共扣四次，实际父程序结果回到下一轮 Runtime，且无自动资产创建。旧的路径越界、递归执行、耗尽工具预算、owner deadline、失败整题的成功局部操作独立试用等回归均通过。模拟 Provider 的选择不作为自然模型效果证据。

真实已完成 UID0115 原始训练记录的独立离线回放：模拟 Learner/Builder 响应、真实生产来源验证及 Docker 运行通过。完整请求仍为 Extractor 16,250 B、Builder 16,287 B，均≤16,384 B；参数化程序在不同输入下执行并被模拟消费者使用，Frozen 前后一致。该 Bank 仅为夹具，禁止用于实验。

旧归档回放初次指向外层目录导致缺少所需 trace；改为实际 seed42 目录后发现旧测试要求与旧版本修复提示逐字相等。当前修复提示、错误 expected 信息及额外真实输入来源在本轮之前已更新，生产 system.py 与基线 `945ad83` 完全一致。测试改为保留原提案、原错误及出处、全部原绑定来源、工具 Schema、请求参数和一次 repair 的核验，允许新增说明与真实来源。旧请求仍为 9,415 B，新修复请求为 10,448 B，均独立记录；没有改写归档或增加 repair。初次失败输出与最终通过记录都保留在交付证据中。

未配置测试是四个指定旧 OfficeQA Test 故障回放及相关 LiveMath 实际归档核对，逐项列在交付 test_index JSON 中。当前执行、完整真实来源学习和冻结链路已单独验收；不把这些跳过计入通过数。

## 身份、费用与后续

本轮源码基于已有 main，保留并合入其它对话的 LiveMath 提交。采用独立 worktree；没有改动原项目目录的未跟踪文件或已固定运行副本。

实际离线来源回放目录：

`/home/yangchengyu/officeqa_task_program_priority_actual_record_offline_20261012`

旧正式及三次既有付费记录的账本 SHA256 均与先前一致，累计仍为 24,192,418 tokens／2,993 HTTP。本轮新增付费 tokens=0、HTTP=0，Office 正式进程未运行。

下一步仍是待授权的一次独立空 Bank 1 Train＋1 Val 验收：固定 UID0208 Train 验证自然临时脚本和来源学习，存在 usable Program 才冻结进入固定 UID0127 Val 检查实际复用和消费。建议 token 上限 400,000＋96,000；请求上限及启动身份应在授权后冻结。没有 Program 时保存卡点并停止，不手工晋升、不继承诊断 Bank、不据此自动启动正式矩阵。

上述题号只用于公开固定诊断题单，不参与方法分支。自然采用、准确率和降本效果尚未经过新版付费验证。
