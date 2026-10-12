# P2-R4：独立解释长度修正与下一次有限候选学习

当前已准备独立 P2-R4 plan，状态为 `prepared_not_started`；真实源码、历史材料及账本预检通过。本轮新增 0 tokens／0 HTTP，尚未发出真实模型请求。它复用已完成的 P1 Train 轨迹开展候选学习，不能把旧 R3 内容否决当成同版本异常恢复或直接晋升。

本轮将内容审查解释 `reason` 的生成提示、发送 Schema 和 Host 验证统一设为 512 个 Unicode 字符。它是结构化审查解释，不注入 Runtime Skill 卡片，也不是模型隐藏思维预算；已有解释为 380、436 字符，512 足以容纳这两次输出。Extractor 的 2,048 completion tokens 和 16,384 B 请求材料上限保持原配置。

能力需求 `required_conditions` 与候选指导 `guidance` 的生成要求仍为 180，Host 容差仍为 256。`applicability` 80、完整 Runtime 卡片 512、原文引用 240、一次 guidance repair、原物理来源与 Span 检查、内容分类标准、selection 和晋升条件保持原配置。提示修订为 `procedural-actions.r5`，Host 方法为 `choice-capability-batch.v2`；历史 r3/r4 的发送 Schema 与 240 字解释验证仍可明确核对，禁止跨提示修订伪装检查点 resume。

已用真实 R3 原始响应完成零 HTTP 回放。两条解释的结构均可通过新版 512 上限，但原始内容分类均为 `unsupported`，故内容检查仍不通过。回放只出具独立收据，不改写旧 rejected、请求、费用、Bank 或 Frozen，也不重做旧检查直至得到有利分类。

P2-R4 重新从原两组固定来源进行能力需求和候选指导生成，最多两个候选；两组各两道不同物理题，共直接使用四道来源题，48 道来源池继续保留。内容通过的候选才进入原 6 初筛＋6 确认 Train 的公开适用性判断，实际适用少于四题即停止配对；按原 off/on 门槛决定是否发布并 Frozen。没有自然候选或内容全部否决时，正常结束为 `completed_no_candidate`；晋升标准不保证产生 Skill。

不重新解 P1 Train60，不自动进入 Val17、Test100 或 seed43/44；`choice_bounded_finalization=false`。R4 入口只接受已经完成的原 R3，不能再由 R4 自动衍生一轮 R4。固定来源、成员顺序、两臂参数、现有推理 token cap 和费用上限均冻结；最终分析应分别报告候选数、内容通过数、适用数、实际注入、配对胜负及费用。

旧 P2 全部费用 **58,444 tokens／11 HTTP** 原样结转。原 P2 总上限仍为 **1,800,000 tokens／80 HTTP**，本次剩余 **1,741,556 tokens／69 HTTP**。包括 P1 的 seed42 累计仍为 **7,154,202 tokens／375 HTTP**；额度不足或计费未知即按原规则停止，不自动扩额。

固定源码：`/home/yangchengyu/asg_livemath_p2_r4_paid_seed42_20261012`，commit `3c38842efe13ad6dace2818f5d3e7087fc7611f2`，源码 SHA256 `323fd007e4f8db1d99879ddfaf5a4c51abcf8f5acd290476f20933b74b504b44`。

新运行目录：`/home/yangchengyu/livemath_p2_r4_paid_seed42_20261012`；plan hash `fabbc201a0b0c59418a58e002de6b8b37036d7db713e912f49aa8d9c7e0f1b8b`。`launch_precheck.json` 记录实际源码及来源/账本预检，`prior_content_checks.json` 记录旧响应的新合同重验。旧 R3 的 27 项文件哈希前后相同，旧累计费用没有被重复计算或遗漏。

P2 专项与生产链路测试 58 passed；最终账本关联检查的两项针对性回归再次通过。测试中 HTTP 全部被拦截，没有真实模型成绩。覆盖了 512／513 字符边界、旧 240 合同保留、180 字生成与 Host 容差、真实 R2/R3 原始记录、全部 11 次旧请求的结转、候选内容否决、固定 selection 的 off/on 与 Frozen、重复 execute 不新增调用及禁止自动衍生再试。JUnit：`D:/T3S_exp/deliveries/outputs/livemath_p2_r4_ready_20261012/p2.xml`。

在 WSL 启动这一次付费实验：

```bash
nohup bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_livemath_p2_r4_paid_seed42.sh \
  >> /home/yangchengyu/livemath_p2_r4_paid_seed42_20261012/execute.log 2>&1 &
```

只做零付费预检：

```bash
bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_livemath_p2_r4_paid_seed42.sh --check-only
```

监控阶段与最后结果：

```bash
tail -f /home/yangchengyu/livemath_p2_r4_paid_seed42_20261012/execute.log
cat /home/yangchengyu/livemath_p2_r4_paid_seed42_20261012/progress.json
cat /home/yangchengyu/livemath_p2_r4_paid_seed42_20261012/result.json
```

启动脚本先核验固定源码与 plan，执行使用 flock 避免同一目录并发重复启动。结束后审查 `learning/*/{learning,requests}.json`、`candidate_order.json`、`candidates/*/{preflight,utility_receipt}.json`、`selection/`、`budget.json` 和 `frozen_bank/freeze.json`；未进入的阶段保留实际未执行状态，不补造成绩。
