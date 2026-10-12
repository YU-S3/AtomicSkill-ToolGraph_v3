# LiveMath P2-R2 一次有限付费验收

用户已明确授权一次 P2-R2 付费验收。原 `0499d8f` 的引用接口故障已在 `27ea23a` 修复；随后实际恢复运行完成为 `completed_no_candidate`，没有异常堆栈。其第一组真实提案将“强结论、少依赖、不加假设／等号”作为选项偏好，被内容检查判为 domain_fact；第二组为正常 no_change。原两次运行共 21,471 tokens／4 HTTP，负结果保留，不能改名为工程报错或手工放行旧候选。

P2 使用已有 P1 Train60 的真实经验，固定 48 来源池和独立的 6 screen＋6 confirmation。本轮仍只从原定两组不同物理题诱导最多两个候选。它不重解 P1 Train60，不继承 P1 或旧 P2 资产，不更换来源，不使用 Val/Test。合格候选要通过原内容门槛，全部 12 题先做公开输入适用性匹配；达到原覆盖门槛后才进行同输入、同模型参数的 Skill-off/on 配对，按原净正确数、空答和回退条件决定晋升。

本次提示身份为 `procedural-actions.r2`，新 plan、源码身份和父 run 全部明确记录。提案需要描述可执行检查，不能仅凭强弱、简洁、依赖、额外假设或等号措辞判断数学真假。内容检查明确区分提案自身和来源题中的数学文本，区分可执行程序、特定数学结论及未经支持的选项捷径，并允许记录短的分类理由。原来源引用、字符上限、一次 guidance repair、零 induction／routing／grounding repair、Runtime high／32768 cap 和所有效用门槛保留。

内容检查材料删除 proposal 与顶层 guidance/applicability 的同值重复，仅保留一次指导、适用条件、真实引用和完整来源记录。candidate_hash 仍绑定完整不可变候选；不截掉题干、候选项或来源证据，上限仍为 16,384 B。两组真实来源以最大合法卡片长度，以及全部 12 道真实公开匹配输入，均经过生产调用路径的拦截容量检查。

`revise --plan OLD --output NEW` 只接受已经正常结束、无候选且全部费用已计量的旧 run，创建新代码身份和独立输出。不能将工程失败或未知费用伪装为修订，也不允许该入口自动再修订 R2。新运行重新执行本轮两组自然候选学习，不复用旧候选分数。保留原物理账本全部 4 次 HTTP 和父 scope 的花费；原 P2 1,800,000 tokens／80 HTTP 是包含前次费用的总上限，当前剩余 1,778,529 tokens／76 HTTP；旧累计费用为 7,117,229 tokens／368 HTTP，不重算 P1 Runtime。

新增阶段进度文件与即时日志，明确记录学习单元、公开匹配、pair_started／pair_completed 和最终原因。summary 分别报告自然提案数、通过来源检查数、实际配对数及旧／新／累计费用。正常 no_change 或内容拒绝仍可结束为 completed_no_candidate；额度不足明确 censored，接口异常明确工程错误，不将所有退出称为报错。

离线验收 31 项 P2 专项＋27 项公开过程／LM1 回归通过，模型 HTTP 为零。随后仅在这次用户授权及剩余额度内执行一次真实 P2-R2，实际付费结果以新运行目录的 result.json、learning、preflight、selection、utility_receipt、budget 和 Frozen 原始产物为准。拦截测试分数不能作为自然性能结果，也不会触发扩大预算、换题或自动再学习。

真实 R2 在 `4ef2456` 下新增 15,843 tokens／3 HTTP。第一组诱导成功，初次 guidance 超过原 80 字 applicability 上限；唯一一次 repair 的完整请求为 17,070 B，超过 16,384 B，因而在 HTTP 之前拒绝。第二组诱导的 required_conditions 超过原 180 字上限，保留为普通模型失败。旧结果错误地将第一组的本地容量故障并入 completed_no_candidate；原始结果不改写，本报告明确区分工程问题与模型拒绝。

容量修复仅对完全位于同一公开 task 文本中的精确 Span 引文使用 task_path＋Unicode start/end 引用，保留完整题干、选项、Span ID、原文字段和观察。只有逐字校验成功才去除同值 quote，不截短、不改上限。通用请求在该材料版本超限时才考虑此无损表示，已付费的原始请求内容保持一致。material_too_large 此后明确记录为工程停止。

`continue-repair` 创建独立、清洁提交的新源码身份和子 run，绑定旧账本及所有原响应哈希。只对原单元中已经准备、repair_index=1、零 HTTP 的那次 guidance repair 恢复；复制检查点、显式保留原响应命名空间并复用已付费原响应，第二组终止记录保持原样。禁止追加 induction、改变来源／门槛／repair 次数，也禁止该入口再次延长这次实验。恢复前 P2 累计 37,314 tokens／7 HTTP，余量 1,762,686 tokens／73 HTTP；原 seed42 累计 7,133,072 tokens／371 HTTP。真实拒绝现场的生产模块回放覆盖缓存复用、一次 repair、来源检查、12 次公开匹配、24 次配对及费用结转；回放使用拦截响应，其分数仅用于工程验证。
