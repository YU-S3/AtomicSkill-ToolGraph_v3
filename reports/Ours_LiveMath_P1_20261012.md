# LiveMath P1：公开过程、过程指导与单 seed 观察

本轮只处理 Ours。Office P0 已独立提交为 `0a7963732f1e2c86e439360c930bb09684983aed`；本报告随独立 P1 提交。实际受测提交、源码身份和启动收据保存于 `D:/T3S_exp/deliveries/outputs/ours_p0_p1_20261012/`。本报告不声称已经提高准确率或降低成本。

`choice-public-procedure.v1` 仅由新 LiveMath 配置显式开启。finish 可附不超过 512 Unicode 字符的公开过程及四种有界步骤；纯文本标签仍按原 scorer 评分，无额外反思请求。Host 核对实际题文位置、精确引用与实际消费的公共结果，记录时间位置和请求 ID；来源不合法只拒绝过程学习，不改答题评分。该记录是 self_reported_process，来源兼容不等于独立执行或数学证明。真实 Worker 操作继续优先进入原 Program 来源绑定与独立重放资格链。

新 Skill 在现有 Bank 内以 executable_operation、procedural_advisory、domain_fact 区分用途。过程指导须有成功 Train 的实际公开步骤、匹配操作类别、必要前提与原文引用；每次最多一条、完整注入不超过 512 字符。定理内容仍要求至少两个来源主题词及 IDF coverage≥0.5，保留原数量/长度额度，不因过程路由放宽。公开意图只来自题干和完整公开选项，无 UID、gold、Test 特判；不相关主题、正式证明要求、虚构结果和越过前提的资产均不能获得过程资格。

身份固定为 empirical-v3.2-public-procedure-r1、choice-guidance.public-process.v2、choice-guidance.public-process-material.v2、choice-guidance.procedure-scope.v2、single-answer.public-process.v3；普通学习材料仍为 learner.material.public-directory.v5。来源投影与 hash 从已验证的 materialization 捕获。新 Bank 必须为空并绑定方法版本；旧 Bank/Frozen 不迁移、不补写过程、不冒名同版本 resume。

length 且无合法答案规范为 completion_truncated_no_answer / truncated_empty，原评分保持，保存实际答案请求及 usage。可选 choice_bounded_finalization.v1 已实现，但默认关闭：启用时仅在原父预算内预留一次 0 工具最终回答，不重答已有合法答案、不增加修复或 HTTP retry；已付费响应的 Checkpoint 恢复不重复发送。该模式的完整请求与费用由原账本记录。本次付费配置保持关闭。

旧 `80762b4` Train60 的只读覆盖审计：60 题中 16 题 hard success，公开过程、自报步骤、来源兼容过程均为 0；保存了 1 次成功 Worker 操作，但输出消费为 0。不能用这些标签替代新方法的过程训练。旧 Test16/100、另版本 Test20/100 和旧 Frozen 原样保留。

零模型 HTTP 验收在本地 WSL、真实锁定 Docker Worker 下完成：

- P0 回归：68 passed / 7 skipped。
- P1 主回归：121 passed / 2 skipped，覆盖新来源/流程、旧 LM1、正式 split/seed、原 Worker/公共接口。
- P1 文件与公共链路回归：78 passed / 7 skipped。
- 最终受影响的公开过程与 gate 回归：31 passed / 0 skipped；包含同位置双引用拒绝、旧 domain_fact 边界保留及成本/源码/来源/发送材料 gate。

上述不同调用含重叠测试，不相加作为唯一测试数。2 个 LM1 与 7 个统一复审历史夹具因另一历史有限包未配置而跳过，未计为通过；本次真实旧 UID0115、Office 四个预算案例和实际 Worker 夹具已执行。HTTP 传输均由测试拦截，未请求模型。

原 UID0115 无绑定材料的真实生产重构由 8,662 B 降至 7,490 B。新 Office v5 六题中的同 UID 是另一份带 case_bindings 的真实提案，必要材料为 9,415 B；完整原提案/来源错误/工具 Schema 无法按无绑定目录删减，已零 HTTP 拒绝并保存收据。用户明确要求 Office 保持暂停；不调整 8,192 B、repair=1 或六题 gate，不另跑六题。Office 累计 11,510,372 tokens / 1,279 HTTP，包含原继承 10,986,884 / 1,193，本前缀新增 523,488 / 86；未知费用为 false。六题、Bank、评分与费用均保留。

用户授权的 LiveMath 观察采用一次 fresh seed42：固定 canonical Train60 顺序，从空 Bank 开始，前六题作为同次正式 Train 的工程验收前缀，通过后同源码、同目录、同 Bank 继续 Train60 → Frozen → Val17；不跑 Test。工程 gate 不以准确率、过程覆盖或 Program 数量放行，也不增加一轮 12+6。已查看过的 Val17 标明为开发集，结果不作为未触碰评估。

原 LiveMath 当前正式 Governor 的 5,046,558 tokens / 273 HTTP 只结转一次费用，不继承资产。整段预算仍为 23,232,000 tokens / 9,165 HTTP，各题原上限不扩大。启动器要求干净固定提交、独立目录、锁定镜像、全量计量及账本限额一致；已有请求/Bank 不从旧 run 复制为新训练。实际新运行的过程、注入、Program 消费和成本需按后续记录统计，未完成前不作研究效果结论。
