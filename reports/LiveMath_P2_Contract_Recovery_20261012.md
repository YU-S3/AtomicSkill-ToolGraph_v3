# LiveMath P2 本地接口修复与明确版本恢复

原 P2 run `p2_a56eea61b1e945b8a11d58d518c676f4` 在 `0499d8f` 下结束为 `completed_no_candidate`，花费 9,800 tokens、2 次 HTTP；未执行候选提案、适用性或 Skill-off/on 配对。两次响应具有完整原始计费，不能据此判断 P2 方法无效。

第一组返回的 `case_id` 为模型实际可见的 task_id，Span 确实位于该题公开输入中；原 Host 只接受 physical_key，造成误拒。引用现在接受材料中唯一对应的 task_id 或 physical_key，并统一按物理来源计数。未知 ID、跨题且不属于本题的 Span、别名歧义、重复物理来源仍拒绝，不做模糊匹配，不添加题目特判。

第二组返回 `no_change` 加两个真实来源引用，请求 Schema 允许该组合，Host 却要求只有 decision。现在允许合法诊断引用，仍校验原文和两个独立物理来源；附带能力、操作或指导内容的 no_change 仍拒绝。合法 no_change 不生成、不发布任何资产。

`recover --plan OLD --output NEW` 为这次已结束的 induction-only run 创建明确新代码身份的子 run。它绑定原 plan、来源材料、实际发送的 public_cases／Schema／参数、原始 provider 响应、解析结果及原账本文件 SHA。只有两组原始未 repair、已计费、未继续候选学习的 induction 响应可以复用。旧 run、P1 轨迹、评分、Bank 和请求不修改；普通模型失败没有获得新的 repair 或选题机会。

新执行从独立派生 Bank 开始，48 来源案例、两组固定来源、6 初筛＋6 确认、候选排序和晋升规则保持原样。先重新校验原始 paid response；第一组可以进入候选提案，第二组记为正常 no_change。候选仍需通过原来源检查、适用性覆盖和两阶段配对门槛，不能手工晋升。

恢复账本完整继承原 2 次 HTTP，保持原 P2 总额度 1,800,000 tokens／80 HTTP，包含原候选父 scope 的花费。真实剩余额度为 1,790,200 tokens／78 HTTP；累计旧费用为 7,105,558 tokens／366 HTTP。新结果分别记录本次新增费用、前次 P2 费用及合计，避免重算或清零。未知计费和额度不足继续停止，不补发请求。

零模型验收：P2 专项 28 passed，公开过程及 LM1 相关回归 27 passed，共 55 passed、0 failed、0 skipped。实际 paid 响应通过生产编译器回放；后续候选提案／来源检查响应采用明确拦截夹具，不能当作自然候选或性能结果。包含错误 Span／未知 ID／别名冲突／同物理来源、no_change 附带能力拒绝、原请求／响应／费用篡改拒绝、原账本费用保持、无二次 induction 请求，以及剩余请求额度耗尽停止。P1 和原 P2 文件逐字节未变。

新入口的启动脚本为 `scripts/run_livemath_p2_recovery_seed42.sh`，在其所属固定干净源码中检查新 plan 并持有执行锁，然后启动有限付费后续流程。本轮不自动运行 Val17、Test100 或 seed43/44；有界最终回答保持关闭，OfficeQA 的独立固定源码运行不受此次更新影响。
