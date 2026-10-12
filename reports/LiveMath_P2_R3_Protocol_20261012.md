# P2-R3 字段协议修正与零 HTTP 回放

本轮只完成协议修改和离线核验。新方法身份为 `choice-capability-batch.v2`，材料身份为 `choice-capability-batch.material.v2`，提示修订为 `procedural-actions.r3`。旧 P2-R2 的失败结论、原始响应、检查点、Bank 与费用均保留。离线重验结果单独保存，不将旧 rejected 改写为 accepted，也不把旧响应冒充新版提示生成的响应。

| 约束 | 原值 | 本次 |
| --- | ---: | ---: |
| required_conditions | 180 字符 | 256 字符 |
| guidance | 180 字符 | 256 字符 |
| applicability | 80 字符 | 80 字符 |
| 完整 Runtime 卡片，含 ID 和 qualification | 512 字符 | 512 字符 |
| Extractor 完整请求容量 | 16,384 B | 16,384 B |
| guidance repair | 1 次 | 1 次 |

字符串长度仍按 Unicode 字符计数，与 token、HTTP 序列化字节数分开。生成提示明确声明每个文字字段的限制及完整卡片限制。P2 结构检查在同一反馈中汇总各字段、嵌套引用、缺失字段和完整卡片容量错误；一次 repair 能看到全部这些已检测错误。完整卡片门槛独立保留，例如 guidance=256 与 applicability=80 的组合可能超过 512，仍会拒绝，不会自动截字。

不同物理来源、真实 Span 引用、禁止指定答案或复制数学结论、独立模型内容检查、selection 条件和晋升门槛保持原规则。Runtime 的 32,768 tokens 上限、原两组来源、6 初筛＋6 确认、最多两个候选、P2 总额度均未变。公共适配处识别新的方法和材料身份，响应缓存按实际版本隔离；历史 v1 资产仍按原 180 字合同校验，禁止直接把旧 v1 run 当成 v2 resume。

离线入口为 `atomic_skillgraph.experiments.replay_choice_protocol`，只有 `replay` 与 `check`，不接受 env-file、不构造模型客户端或可执行的付费 plan。它核对原始 provider ToolCall、工具 Schema、公开输入、精确 Span、原逻辑决策、repair 序号和物理费用，再按新合同重验已付费的最终响应；不根据结果回退挑选旧提案。输出为 `protocol_replay.json`、`receipt_identity.json` 和原样复制的 `inherited_p2_budget.json`，不创建 Bank 或 Frozen。

实际来源为 `/home/yangchengyu/livemath_p2_r2_capacity_continuation_seed42_20261012/plan.json`。真实材料的生产模块零 HTTP 测试结果：

| 单元 | 原始响应在新合同下的结果 | 后续待验证 |
| --- | --- | --- |
| 第一组 | guidance=248、applicability=75，完整卡片=500；结构、真实来源及静态内容限制通过 | 独立模型内容检查，再按原规则进行适用性与 off/on |
| 第二组 | required_conditions=231 通过能力需求准入；没有已生成的 Skill 提案 | 候选指导生成，再经过内容与效用检查 |

第一组的自然来源检查、适用性和 off/on 均没有真实模型成绩。离线 receipt 对 independent_content_check、utility_check 明确记为 not_run，published=false。第二组只验证到能力需求，不能声称已经生成候选 Skill。

原 P2 已计费 **42,057 tokens／8 HTTP**，剩余 **1,757,943 tokens／72 HTTP**；包括 P1 在内的原 seed42 累计 **7,137,815 tokens／372 HTTP**。本轮新增 **0 tokens／0 HTTP**。复制账本只保留既有物理请求；检查会拒绝改变原始响应、来源、Schema、逻辑决策、费用或未知计费状态。未启动 Val、Test 或其它 seed。

核验：P2 与 R3 专项 **43 passed**，相关公开过程、LM1、恢复边界、CF2 和正式记录回归 **103 passed、6 skipped**。跳过项为四项缺少 Docker 回归环境参数、两项缺少历史 LM1 审查目录。测试明确拦截 HTTP，覆盖真实两组响应、500／512／513 字卡片边界、256 字字段边界、一次 repair 的聚合反馈、原费用继承、改动原始证据的拒绝和旧版本隔离。模拟响应只用于检验工程路径。

提交后会在独立干净 WSL checkout 上运行同一离线入口并保存源码身份、完整输入哈希及核验收据；这些原始输出随本轮审查包提供。下一次付费工作应从已记录的待验证阶段开始，并继续计入原剩余额度，不能重复解 P1 Train60 或自动改换 selection。
