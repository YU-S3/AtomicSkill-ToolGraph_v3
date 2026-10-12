# P2 生成目标 180 字符与 Host 验收容差分离

按用户澄清，模型仍应生成不超过 180 个 Unicode 字符的能力需求和指导。此前 R3 同时将生成提示、发送的 Schema 和 Host 验证改成了 256，超出了本次要求。本轮恢复生成目标，保留已经授权的 Host 容差。

| 字段 | 新请求提示及发送的 Schema | Host 验证上限 |
| --- | ---: | ---: |
| required_conditions | 180 | 256 |
| guidance | 180 | 256 |
| 内容审查的 reason（可选） | 180 | 240 |
| applicability | 80 | 80 |
| proposal_quote（原文摘录） | 240 | 240 |
| 完整 Runtime 卡片 | 512 | 512 |

240 是独立内容审查解释 `reason` 的原有上限，与候选 `guidance` 的长度分开。实跑 R3 两次解释为 380 和 436 字符，因该字段超限拒绝。第一组复用的指导为 248 字符，第二组新生成的指导为 160 字符；第二组复用的能力需求为 231 字符。这些不是同一个字段的长度变化。

实现仅修改 P2 的请求合同：`generation_schema` 生成独立的发送 Schema，Host 继续使用原验证 Schema。提示修订记录为 `procedural-actions.r4`；Host 方法身份仍为 `choice-capability-batch.v2`。按历史提示修订核对旧请求的原始发送 Schema，不改写旧请求。不同提示修订的完成单元不能冒充同版本检查点恢复。

超过 Host 上限仍拒绝，不截字、不删除解释、不手工晋升。单次 repair、来源与原文 Span 检查、独立内容审查、固定两组来源、6＋6 selection 和晋升条件保持原设置；HTTP 材料上限仍为 16,384 B，原 1,800,000 tokens／80 HTTP 总额度不变。

旧 R3 两条内容审查原始 `status` 均为 `unsupported`。长度合同修正不会将这些否决改成通过。旧实跑目录 `/home/yangchengyu/livemath_p2_r3_paid_seed42_20261012`、固定源码 checkout、原始响应、账本及空 Frozen 保留；累计 P2 成本仍为 58,444 tokens／11 HTTP。

离线验收通过 54 项测试，包括真实 R2 付费响应重验、真实 R3 超限响应与费用保留、生产请求发送 180 而 Host 接受容差内的响应、240／241 边界、内容否决不晋升、一次 repair 汇总错误、跨提示修订拒绝恢复，以及原 selection／预算／Frozen 链路。所有模拟 HTTP 均被拦截；本轮没有新增模型请求或实验启动。JUnit：`D:/T3S_exp/deliveries/outputs/livemath_p2_generation_180_20261012/p2.xml`。
