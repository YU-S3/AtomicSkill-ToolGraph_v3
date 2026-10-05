# CF3 交付与核验

修改基线：`2528f53b222dfbb79a54e37ad7514eb3f160fd59`。最终受测源码：`a1e6b00980ccddf0641df70fc4b4e6dc7e42aeda`；初始实现为 `cebf24e`，后续补齐隔离试用的 Planner checkpoint 作用域。报告补充不改变受测源码、配置或测试。

按 [CF3 实施规范](../docs/specs/SkillCompiler_CF3.md)完成 Ours 修改。**最终干净 WSL 检出运行 160 项回归，160 通过、0 失败、0 跳过，耗时 197.65 秒。** 本轮未调用真实模型，未启动 fresh CF3 训练或正式矩阵，未重标旧结果。

## 修改结果

- `dynamic/skill/program` 共用节点接口解析。参考 Skill 不授予自动路线或输出约束；明确绑定读取真实资产目标和 I/O。ready usable Program 保持自动接管、准备后自动继续与新图组合执行。
- Dynamic 交接字段由实际后继和顶层输出引用生成。缺字段保存真实 `pending_outputs` 和结构化 `handoff_error`。普通业务字典不拆包；别名仅引用实际值，原字段、原程序结果及消费信用保留。
- `patch_node` 在副本校验后提交本题实例，支持 args/handoff/detach。禁止改已执行实参、伪造输出、改 Bank 或重排节点；已完成输出的修正使用新交接视图记录。
- 完成失败、绑定失败、准备映射错误、明确调用失败和无进展信号进入有界恢复。相同实际失败沿用两次上限，显式和系统重规划共用一次，随后最多一次剩余 Dynamic；保留原环境、结果、预算和 usable 程序菜单。
- checkpoint v2 在请求前保存逻辑决策身份；已应用或最终拒绝的回答不能供下一决策复用。Executor 状态与决策完成状态原子提交；独立原生 journal 保护已完成批次子项，未知副作用继续停止。隔离试用使用自己的 checkpoint，返回时恢复父任务作用域。
- Planner 请求包含 Adapter 的稳定工具语义及当前可调用目录。Builder 初次与修复共用由实际 Context 方法、ToolSpec 和返回常量生成的 ABI；必须返回 `status/outputs`，缺外壳按普通执行失败记录。
- 冻结投影中删除不可用 Program 绑定时生成新 Workflow ID 并保留来源；不以同一资产 ID 覆写不同内容。旧明确 Program 节点只读兼容，旧 Skill 歧义在既有 select 请求中用 node_modes 指定。

没有 benchmark 题号或动作路线特判进入 empirical 核心。原晋升条件、candidate 权限、Builder 共享一次恢复、文件越界拒绝、状态更新、单轮 QA、视觉锁和评分规则保留。

## T01—T28

下表的通过状态来自最终 JUnit。逐个测试名称、运行时间和修改路径见 [机器报告](cf3_verification.json)；原始日志、HTTP 样本、完整 resolved config 和两条真实执行轨迹在交付 ZIP 中。

| ID | 核验内容 | 结果 |
|---|---|---|
| T01 | Dynamic 参考整题 Skill，不继承输出 schema、不自动运行其 Program | 通过 |
| T02 | 明确 Skill 绑定保留完整接口与自动接管 | 通过 |
| T03 | ready Program 无 Runtime 确认直接执行 | 通过 |
| T04 | 新组合两程序连续执行与普通传值 | 通过 |
| T05 | 准备输出映射后自动执行目标程序 | 通过 |
| T06 | 跳过第一条不 ready 路线，执行第二条 ready 路线 | 通过 |
| T07 | 搜索节点只交接 resource_id，不被整题字段约束 | 通过 |
| T08 | 缺字段保存 pending，局部修正后继续 | 通过 |
| T09 | 普通 literal/from 业务字典保留，明确引用正常解析 | 通过 |
| T10 | 别名源与冲突检查、原值保留、实际消费信用 | 通过 |
| T11 | args/handoff/detach、拒绝无效补丁、已完成结果新增视图 | 通过 |
| T12 | 重复交接错误进入一次 replan 和一次 escape，不裸循环 | 通过 |
| T13 | 相同 prompt 的两个新决策产生不同逻辑身份和响应 | 通过 |
| T14 | 未应用响应恢复不重复请求/usage；隔离试用恢复作用域独立 | 通过 |
| T15 | 完成或协议拒绝与反馈原子提交，崩溃后使用新决策 | 通过 |
| T16 | 三项 Office 读批次第二项后崩溃，仅完成剩余项，额度一致 | 通过 |
| T17 | 重规划用尽后仅一次剩余 Dynamic，状态和预算保留 | 通过 |
| T18 | Dynamic 退路仍能显式调用 usable Program | 通过 |
| T19 | 被接受但无进展的循环升级；不同文档窗口不误触发 | 通过 |
| T20 | 稳定 HEAT 语义进入 Planner HTTP，当前目录仍如实提供 | 通过 |
| T21 | 两种真实 ToolView、ToolResult 和初次/修复 HTTP ABI 一致 | 通过 |
| T22 | 正常返回外壳通过；缺 status 或 outputs 不包装成成功 | 通过 |
| T23 | 原自然学出的 Heat/双对象程序，在真实 ALFWorld 保留接管 | 通过 |
| T24 | 真实程序多动作内部零 Runtime LLM，原生事件与预算完整 | 通过 |
| T25 | Frozen 执行失败后的本题避用/解除，hash、资格和路线排序不变 | 通过 |
| T26 | 在途 Program 副作用未知停止，不重发、不写第二次成功 | 通过 |
| T27 | 单轮 solver 响应恢复不重解；图像 HTTP 与能力锁保留 | 通过 |
| T28 | CF2 文件、位置单位、状态、批量、finish-only、试用映射等 | 通过 |

回归中的 HTTP 使用生产 Provider 的实际序列化入口后截获，保存最终 payload；不是收费模型输出。工程 fixture 的 usage 数值用于验证记账去重，不能作为真实实验费用。手写工程程序只存在于临时测试 Bank，未导入实验初始资产。自然学习效果未以这些 fixture 替代。

## 真实已验证能力的执行层对照

使用原 `d046c1de` 独立 12 Train＋6 Val 中自然形成的同一个只读 Frozen Bank，真实 reset 对应两题；原生操作和官方评分重新执行，Planner HTTP 被工程 fixture 截获。没有反馈回放、手改程序、导入准备能力诊断库或重新训练。

| 场景 | 程序 ID 前缀 | 原生调用 | Program 调用 | Runtime LLM | 结果 |
|---|---|---:|---:|---:|---|
| heat apple → diningtable | `program_723f61a4` | 9 | 1 | 0 | 官方终态成功 |
| two creditcard → dresser | `program_7c2d8d52` | 18 | 1 | 0 | 官方终态成功 |

每题只有一条截获的 Planner 请求。原 Frozen 树 SHA256 在前后均为 `55f35ae1084e2c985a98e7589f50f04070a633e2acd8e240b3981d4c37da8de5`。

**工程通过：** 接口、连续执行、有界恢复、隔离、评分和恢复记账回归通过。

**机制发生：** 原自然学出程序在新版执行层自动接管，真实执行 9/18 次原生动作；确定性工程用例完成跨节点传值与自动接续。这不是新版从空 Bank 自然学习的效果测量。

**效果测量：** fresh CF3 的成功率、每题真实费用、≤50k/60k 目标与 baseline 质量比较均未测量。没有据回归通过声称降本或质量达标。

## 配置与历史保全

Ubuntu WSL / Python 3.12 / Docker Desktop，镜像固定为 `sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783`。干净受测检出位于 `/home/yangchengyu/asg_cf3_20261005`，原始最终证据位于 `/home/yangchengyu/cf3_verification_20261005/final_a1e6b00`。

两个 YAML 相对基线只改 implementation_revision，并加入 `runtime.plan_execution_policy` 与 `runtime.dynamic_escape_limit=1`；模型、high、各角色 token 限额、动作/时间/内存限制、Builder 初次与恢复额度全部逐字段比较一致。既有 `planning.task_replan_limit=1`、`runtime.repeated_unchanged_failure_limit=2` 继续作为唯一同义配置，没有增加重复字段。

六个 canonical manifest verify 通过；authority SHA256 保持 `b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95`。正式题号、成员、seed 顺序及独立 Bank 规则未改。当前旧矩阵仍为 interrupted=1、queued=14、unsupported=3，未自动恢复或重复启动。

原 12＋6 完成结果仍为 Train 11/12、Val 5/6；本轮未覆盖其轨迹、费用或分数。上一批审查 ZIP 保持 SHA256 `67e85b530654ae042f715562349f39d599795f120c11fd38fb4978a47bf0a485`。

新运行必须固定新源码，使用独立输出目录和 fresh 空 Bank；旧策略 checkpoint 不跨版本迁移。CF3 第 13.3 节明确不新增收费试验授权，本次请求完成代码修改和无 LLM 验收，没有扩大模型矩阵或循环 pilot。

交付材料包含最终 source identity、完整配置、修改清单、T01—T28 映射、全部最终测试日志/JUnit、Planner/Runtime/Builder/恢复请求样本、两题真实执行原生记录、原只读 Frozen 副本、保全哈希与最新矩阵快照。已知工程未完成项为空；新版学习与成本效果属于未测量项。
