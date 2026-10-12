# LiveMath P2-R2 真实付费验收与故障修复

本轮按用户明确授权完成一次 P2-R2，已经实际发出模型请求并正常结束。最终状态为 `completed_no_candidate`，没有合格 Skill，没有公开匹配或解题配对成绩。故障处的真实 HTTP 已恢复；剩余配对路径仅有生产模块拦截回放证据，不能称为真实付费配对验收通过。

P2 使用已经结束的 P1 Train60 保存轨迹进行跨题能力提炼。固定划分为 48 道来源池、6 道初筛和 6 道确认；实际候选来源仍是预先固定的两组、每组两道不同物理题，只直接使用四道来源题。合法候选要先经过来源内容检查，再对 12 道独立 Train 公开输入判断适用性，达到覆盖门槛才执行同输入、同模型参数的 Skill-off/on 配对。没有重新运行 P1 Train60，没有运行 Val17、Test100、seed43/44。

## 各次退出的真实原因

| 运行 | 实跑提交 | 实际状态与原因 | 本次新增 tokens／HTTP |
| --- | --- | --- | ---: |
| 初次 P2 | `0499d8f` | 引用接口拒绝：真实 task_id 引用与物理键检查不兼容；另一组 no_change 引用被 Host 拒绝 | 9,800／2 |
| P2-R1 接口恢复 | `27ea23a` | 正常无候选：第一组提案被内容检查判为 domain_fact；第二组 no_change | 11,671／2 |
| P2-R2 学习 | `4ef2456` | 第一组诱导成功，guidance 初答 applicability 93 字超过 80 字；唯一 repair 请求 17,070 B 超过 16,384 B，零 HTTP 本地拒绝。第二组 required_conditions 231 字超过 180 字。旧结果将容量故障归为正常拒绝，原记录保留 | 15,843／3 |
| P2-R2 容量修复子 run | `805f9c6` | 原 repair 成功发送，HTTP 200、finish_reason=tool_calls。applicability 修为 75 字，但 guidance 248 字超过 180 字，因此按原规则拒绝；第二组原拒绝保持不变。正常 completed_no_candidate | 4,743／1 |

R2 合计新增 20,586 tokens／4 HTTP；包含原 P2、R1 在内的 P2 总费用为 **42,057 tokens／8 HTTP**。所有物理请求均已计量，unknown_billing=false。剩余原 P2 额度为 1,757,943 tokens／72 HTTP，没有自动追加实验。包含 P1 的原 seed42 累计为 7,137,815 tokens／372 HTTP，P1 训练 Runtime 不重新执行、不重复收费。

## 已完成的最小修复

容量修复将同一公开 task 中重复出现的精确 Span 引文改为位置引用。逐字核验 quote 等于完整原文在 task_path/start/end 指定的 Unicode 字符片段，才移除同值 quote；题干、全部候选项、Span ID、来源身份和观察全部保留。只在该材料版本超限时使用，不改 16,384 B 上限，也不截短模型提案。真实发送的 repair 参数共 **13,309 B**；公开审计重构共 13,284 B，审计按既有规则去掉私有 reasoning 字段，不能将重构内容声称为逐字原始线缆参数。

子 run 显式记录新代码身份、父 run、输入及响应哈希、原逻辑决策 ID和账本。复用原已付费诱导与初次提案响应，只继续原 repair_index=1、零 HTTP 的待发送修复。第二组失败不重试。material_too_large 以后明确记录为工程停止。没有降低来源检查或效用晋升条件，没有增加 repair 次数，没有给模型产物手工删字或晋升。

## 核验结果与边界

- P2 专项 33 passed；公开过程、LM1、费用与恢复边界、CF2、正式记录相关回归 103 passed、6 skipped。测试没有发出模型 HTTP；跳过项为相应测试的环境条件。
- 用这次真实未发送 repair 的生产检查点回放，验证已付费响应复用、仅一次 repair、来源检查、全部 12 道公开匹配、24 次 off/on 调用和费用结转。拦截响应与分数仅是工程测试，不能作为自然候选或效用结果。
- 真实子 run 只新增一个物理 HTTP，原逻辑决策 ID 保持一致，继承的七个物理请求及费用不变。
- 62 个绑定的历史输入、旧结果与 Bank 文件哈希未变；运行结束重新 check 为 passed。
- 最终 Frozen 的 Skill／Program／Workflow 均为 0，digest 为 `3d1d11b0825ed8ffc0a2efde9a7ee473da9b5f99bbadf9b9ed13176ddda9f7d2`。
- 一组共享公开能力诱导成功，另一组诱导格式无效；第一组 guidance 初答与一次 repair 都未满足原字段长度。合法自然候选 0、通过来源检查候选 0、实际适用性匹配 0、实际 Skill 注入 0、实际配对 0、usable 0。格式无效的原始提案保存在 requests.json 中，不隐藏为“模型从未提出过文本”。

本次再次结束的原因是模型输出违反固定字段限制，不是 API 连接、计费、读取路径或本地容量故障。不能为得到候选而自动改变提示、放宽字符上限或追加第三次 repair；也不能把零候选解释为方法收益已经得到验证。

## 实际记录位置

修复子 run：`/home/yangchengyu/livemath_p2_r2_capacity_continuation_seed42_20261012/`。

实跑源码：`/home/yangchengyu/asg_livemath_p2_r2_capacity_fix_seed42_20261012/`，提交 `805f9c66608110d98a0b97f7d21f5166db86fd67`，源码 SHA256 `6bc9a79db68e5f12cfd7545389a659debf23cced6104207c0ee9a8005576127d`。

run_id：`p2_636024aa7cec4d38b6f6e874323ac083`；plan_hash：`b634bf581fba89d8952423a5564684922d8f3b839abefccc7408ddb876d779fc`。

重点材料为 result.json、paid_verification.json、plan.json、config.json、launch_receipt.json、precheck.json、learning/0 和 learning/1 的 learning.json／requests.json／checkpoint、budget.json、candidate_order.json、Frozen Bank、execute.log。旧 R2 容量拒绝现场在 `/home/yangchengyu/livemath_p2_r2_paid_seed42_20261012/`，旧结果未改写。审查包保留上述现场、各父 run 身份和真实请求、费用及测试 XML；不包含密钥或整个项目／数据集。

同一 plan 已正常结束，无需再启动。再次 execute 会返回已保存结果，不会自动进入下一轮学习、Val 或 Test。
