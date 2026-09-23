# Release5 实施说明

基线：`a08ec998e2026e0dc88c021b2b129a8da561c9ac`，仅 main。
需求来源：`R10.3_Release5_全Seed协议修复与旧冷却图调整_完整交付包`。
最终测试、代码指纹与放行结论见 `RELEASE5_VALIDATION.md`。

## 共用协议修复

- 原生接口构造器声明内部 kind/scope/result owner；每请求保留不可变 schema 快照。HTTP wire 仍仅使用标准 name/description/parameters。
- Session 原严格校验先决定拒绝，再生成只读结构化诊断。一次反馈包含同层全部额外字段和缺失字段；保留嵌套路径及组合分支，不猜参数、不删参数、不合并分支 required。
- Implementation 描述明确扁平 Atomic 输入，执行后输出不是调用者的候选输出。current/lean 共享 Node 正确性说明，task/draft 不混入不存在的父节点接口。
- 原一次 repair 限额、共享预算、high reasoning、动作权限、官方成功判定不变。非法调用不进入环境；修复后仍走原 preflight/ToolRunner。
- Session、最终 HTTP、Trace 逐请求对账；记录全部生成调用（包括未执行及结构化提交）、拒绝和修复费用。拒绝/修复费用是总量的重叠切片，不重复相加。
- Support 决策费用改用 session + provider request identity 关联，避免一次修复后 accepted turn 序号与物理计费序号错位。

## Bank 修改边界

从 Release4 实际 source/input.zip 和 edit_plan.lock 重建，保留原有发布与验证机制，不复制 SQLite 后改 hash。

| Seed | 原资产版本 | 新资产版本 | 允许差异 |
|---|---:|---:|---|
| 42 | 202 | 203 | 仅新增旧冷却 Composite 的 1.0.1 |
| 43 | 185 | 185 | 所有资产正文、文件哈希、状态不变 |
| 44 | 199 | 199 | 所有资产正文、文件哈希、状态不变 |

seed42：`skill://composite_ec077ae0a450fd66efdbc8ac@1.0.0` 保留，新增 `@1.0.1`。
原五个 Atomic、Implementation、Tool 不改；根据原 NAV/OPEN/COOL 的 input_identity 声明核对改线依据。

- `station_nav.location → station_open.container` 保留。
- `station_nav.location → cool.station` 替代 OPEN 的间接来源。
- `cool.object → deliver.object` 替代 TAKE 的直接来源。
- station_nav 不预置最终 destination；每输入唯一生产者；goal_contract 不改。
- 新版部署偏好指向 1.0.1；旧版与失败历史保留。

没有加入运行时 task ID、对象名、容器名或 family→动作规则。固定任务和显式动作选择仅位于开发测试驱动，不能充当 Runtime 策略。

## 旧运行留存和新版停止

旧 Release4 三个已暂停监督进程已核实 argv 后终止。旧根未移动、未覆盖：
`/home/yangchengyu/asg_oldfirst_20260923_release4`。
留存目录：`/home/yangchengyu/asg_release4_stop_20260923`，含停止记录、SQLite backup、任务/attempt/Trace/费用核查。

| Seed | 已完成题 | 最后完成索引 | 在途索引 | 已记录 tokens |
|---|---:|---:|---:|---:|
| 42 | 70 | 69 | 70 | 3,707,435 |
| 43 | 119 | 118 | 119 | 4,774,946 |
| 44 | 115 | 114 | 115 | 4,701,774 |

在途远端费用未知，未填零，未混入新版结果。不得用 Release5 resume Release4。

新版 `scripts/request_released_stop.sh ROOT all|42|43|44` 仅写控制文件。运行器在任务 begin 前和完整 capture/digest/终态提交后检查；supervisor 退出整个 seed 流，不进入下一 rep。保留原 RunState 枚举，独立 operator_stop/progress 说明暂停。恢复需显式清除对应控制请求，再用同版本 `--resume`。

## 发布与验证入口

最终根：`/home/yangchengyu/asg_oldfirst_20260923_release5_final`。
原 `..._release5` 是审计补齐前的中间构建，尚未运行真实开发题或正式题，不用于启动。另建 final 根保证冻结指纹与最终代码一致；中间材料未删除。

`prepare_release5` 完成三库差异、原生接口审计、35条历史拒绝离线回归及新代码重封存。
`validate_release5.sh` 完成全 pytest、diff/shell 检查、三库无模型专项、冷却两状态专项、固定 current 18题、五轮配置生成和只读放行；不启动 Test134。

最终矩阵仍是 seed42 rep01→02→03 顺序，seed43/44 各 rep01，与42并行。主表仅各 seed rep01；42三轮另列可靠性，不选最好轮。
