# LiveMath P2：真实付费验收与协议修复（2026-10-12）

本轮已完成实际付费候选学习、公开适用性判断、初筛 off/on 及后续格式验证。结果不是“始终没有候选”：第一组生成了一个通过结构、来源和独立内容检查的候选，实际注入五道独立 Train 题；但没有准确率收益，空答还增加，未晋升。第二组在精简提示下两次选择 `no_change`，最后的独立付费格式验证确认该决定能以合法提交正常结束。最终 Frozen 中 usable Skill 为 0。

不同代码版本及失败记录全部保留。本轮未重解 P1 Train60，未运行 Val17、Test100 或 seed43/44，未换来源题、selection、晋升条件或额外最终回答设置。后两次协议验证没有 selection 或资产晋升，也不能算成最新提示版本的完整方法效果实验。

## 原来的 no_candidate 与实际修复

R4 的第一组 `reason` 实际为 546 字符，超过已经放宽至 512 的 Host 上限；原始分类同时是 `domain_fact`，因此仅消除长度错误也不能将它晋升。第二组需求带有具体数学结论 `C_β=1`，按原内容规则拒绝。原 `completed_no_candidate` 和 25,848 tokens／5 HTTP 保留。

需求、指导、内容检查的生成提示与发送 Schema 均要求短文本：`required_conditions`／`guidance` 180 字符，目标约 140；`applicability` 上限 80，目标约 50，采用简短类别；可选 `reason` 180，目标约 140。Host 容差保持需求／指导 256、解释 512，完整 Runtime 卡片 512、原文引用 240。没有继续提高这些上限，字符统计使用 Python 字符串长度，包含空格、标点。

R5 首次付费尝试暴露了真实的额度作用域问题：历史账本与新学习单元都使用 `candidate:0`，新请求被错误计入旧单元的 80,000 tokens 池。旧账本中该作用域已累计 65,347 tokens，下一请求的保守预留因此被拒。它应记为 `stopped_budget_censored`，不应称为自然无候选。修复后学习和 routing 额度按 run ID 隔离；旧费用仍全部进入全局额度。Runtime 配对原本就按独立输出目录隔离。新版本复用了已经付费的同一 induction 响应，仅继续未发出的提案及后续步骤，未增加 repair 或重复收费。

续跑第二组的 applicability 初次 92、唯一 repair 后 81，仍超过 80。新版长度反馈一次性报告字段错误，要求明显压缩、保留含义和引用；格式 repair 只重放原提议、能力身份、合法引用清单及错误，不重复完整历史轨迹。独立内容检查仍使用完整来源，资格标准不变。请求材料上限仍为 16,384 B，Extractor completion cap 仍为 2,048。

精简提示的付费验证中，该组两次原生响应都选择 `no_change`，但夹带 applicability／guidance，按原合同被拒。最终补丁明确该分支提交 `{"decision":"no_change"}`；修复反馈保留弃权决定，删除其他字段，不诱导改成候选。最后一次付费格式验证一次成功返回这个完整原生 ToolCall，payload 为 2,695 B，新增 977 tokens／1 HTTP，未触发 repair。旧 rejected 没有被改写为 accepted。

## 真正的候选效用结果

受测提交 `d1329c49edc7fc78290fcd50b1cbee2ea64c5226`，提示 `procedural-actions.r6`。候选为 `skill_704a2f32ee767587ada21382f0b657c40ec29005f46600c3450909ea0b4206c5`。其 guidance 165 字符、applicability 52，独立检查解释 126，分类为 `procedural_only`，无长度拒绝。

该候选对固定 6 初筛＋6 确认的公开输入判断为 7/12 适用，其中初筛 5/6、确认 2/6。初筛仅执行适用的五题；所有 on 臂实际注入该 ID，off 臂不注入。两臂的题目、其他指导、参数和 Frozen Bank 哈希通过生产 `pair_fact` 检查，均无 HTTP 重试。

| 初筛臂 | 正确 | 空答／截断 | 实际解题 HTTP | 总 tokens | input | output | reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 2/5 | 1 | 5 | 117,714 | 10,124 | 107,590 | 107,558 |
| on | 2/5 | 2 | 5 | 128,025 | 10,669 | 117,356 | 117,332 |

五组均为正确性 tie，净增正确数 0；on 多一次空答，并多用 10,311 tokens。按冻结门槛为 `screen_rejected`，不进入确认、不发布 Skill。完整状态为 `completed_selection`，不是 `completed_no_candidate`。正常模型截断按原规则计失败，没有为此追加最终回答或放宽门槛。

后续 `procedural-actions.r7` 只对原第二组来源复用真实需求做候选生成协议验收；没有重跑第一组配对。`procedural-actions.r8` 仅验证原 no_change 的合法格式，保留决定，不重新提取候选。因此不能声称 r8 获得了新的配对分数，也不能将前一个候选视为有迁移收益。

## 身份、费用和位置

请求模型名为 `deepseek-v4-flash`，实际响应 model 为 `deepseek-flash`；最终发送参数、thinking／effort／completion cap 和 raw_usage 均在物理请求记录中。Runtime 的已有预算及模型参数保留，`choice_bounded_finalization=false`。

| 实际阶段 | 固定源码提交 | 新增 tokens | 新增 HTTP | 结果 |
|---|---|---:|---:|---|
| R5 原尝试 | `afa269b1ae31c0bc2f02f9512ca15d8d505d45cc` | 5,352 | 1 | 局部额度作用域拒绝；保留需求响应 |
| R5 额度修复续跑 | `d1329c49edc7fc78290fcd50b1cbee2ea64c5226` | 311,302 | 27 | 1 个合法候选，初筛未通过 |
| r7 精简协议验收 | `63f6a7c77ab00e25cfff3906d314c329d8ed8321` | 10,241 | 2 | no_change 两次夹带字段，原样拒绝 |
| r8 no_change 格式验收 | `aa41aa896fc52a5e030726243ccd80e674cf58d7` | 977 | 1 | 合法 no_change，一次通过 |
| 本轮合计 | 各自保留身份 | **327,872** | **31** | 不发布、不扩大预算 |

包括所有历史 P2 的累计为 **412,164 tokens／47 HTTP**，所有请求计费已知。原 1,800,000 tokens／80 HTTP 额度剩余 **1,387,836／33**，没有清零旧费用。包括 P1 的 seed42 累计为 **7,507,922 tokens／411 HTTP**。旧响应、账本、来源和各阶段结果保留；衍生时逐项哈希检查，旧账本物理记录原样继承，历史文件未修改。

R5 完整结果：`/home/yangchengyu/livemath_p2_r5_budget_continuation_seed42_20261012`。源码 SHA256 为 `9ac4000a3009626070d54f031e3b820cc9917dae3cdff2bed1434111c113e2d6`，plan hash 为 `9711f2544d6f7fcb1beb14b78732588889e9a49c85da9be9fecec869b61b9f92`。

r7 协议结果：`/home/yangchengyu/livemath_p2_r7_protocol_seed42_20261012`。源码 SHA256 为 `34b350704eb588c5d6a36a607daab7a15d82368c703079e0fd324eaf8869c758`，plan hash 为 `4c9404f1fa7a7cd70e44925681227f83332bce2f8ea11ea60669b1bde87b6efa`。

r8 最终格式结果：`/home/yangchengyu/livemath_p2_r8_no_change_format_seed42_20261012`。源码 SHA256 为 `998f9fe2221c276faa160023cfd8a52554ad8b95d72880ef5162290d6c73e463`，plan hash 为 `5eaaf0508184db58a21594d8ad81e6054962d9d324ebef13ef5bda619ca950cf`。

最终三份 Frozen 的 digest 均为 `236a2ad1ec1a4c0f41a08d4765d90c6629883371b1ceff14edf4be0ecb029660`，不含 usable Skill。协议诊断 Bank 不能成为后续 fresh 正式实验的初始资产。

## 验收与交付

最终专项及相关生产链路测试 **69 passed，0 failed，0 skipped**。测试覆盖真实高费用前缀、账本全部结转、固定来源及 selection、字段错误汇总、Host 容差、512 卡片限制、唯一 repair、只读 off/on、内容否决及真实 no_change。最终 JUnit 在 `D:/T3S_exp/deliveries/outputs/livemath_p2_r5_actual_20261012/final.xml`。测试不代替上表的真实付费记录；模型响应也不保证未来每次都合规，超限或无效输出仍须按原规则拒绝。

审查汇总为 `D:/T3S_exp/deliveries/outputs/livemath_p2_r5_actual_20261012/review_summary.json`。审查包 `D:/T3S_exp/deliveries/LiveMath_P2_R5_R8_实际验收审查包_20261012.zip` 包含实际计划、配置、账本、原始请求响应、逐题配对 trace、效用收据、Bank／Frozen、对应固定源码、测试与各次付费诊断脚本，并附逐文件 SHA256。不含 `.env` 或完整数据集。

本轮已结束，无待继续的 P2 请求。不能因 usable=0 自动再换题、反复生成直到通过、扩大额度或启动 Val／Test。现在得到的真实研究结论是：工程链路能够生成、检查并实际试用候选，但受测候选没有通过原定效用门槛；第二组原始提议选择 no_change。
