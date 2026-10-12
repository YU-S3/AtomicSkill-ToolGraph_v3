# OfficeQA 临时编程与学习来源修复、实际付费核验

代码修复已提交至 main，最终受测源码为 `e4db6a4147612ff6faeab2d09cc6e168fd420480`，Linux 源码 SHA256 为 `a71b90f664ad7e5b7af4e4805aad803587bae1a12ef77a6c579ba44e0baf5f2c`，受测副本无 tracked 修改。已经实际完成有界付费测试：修复了收尾丢失计算结果的问题，真实收尾诊断答对；自然 Learner 仍选择 no_change，跨题 Program 复用门槛未通过，正式实验保持暂停。

## 本轮实际结果

| 实际事件 | 源码 | 结果 | tokens | HTTP |
| --- | --- | --- | ---: | ---: |
| fresh 空 Bank，seed42 首个 canonical Train：UID0208 | 878f240 | 完成，答错；7 个临时脚本；Learner no_change | 288,525 | 15 |
| 修正来源目录后的同记录学习诊断，不重做求解 | f776446 | no_change，无 usable Program | 4,466 | 1 |
| 保留完整计算代码及澄清参数化后的同记录学习诊断 | 4fa6d40 | no_change，无 usable Program | 4,931 | 1 |
| 原定 canonical 首个 Val：UID0127，只读无资产执行检查 | 4fa6d40 | 完成，答错；4 个临时脚本，没有原生调用接口错误 | 52,500 | 6 |
| 修复收尾材料后的独立最终回答诊断，不重跑检索 | e4db6a4 | 返回 0.162，原评分器判定正确 | 5,881 | 1 |
| 用真实新收尾结果派生的独立学习诊断 | e4db6a4 | no_change，无错误、无 usable Program，跳过复用 Val | 4,901 | 1 |
| 合计 | 各阶段源码分别保存 | 旧 Train 0/1、旧 Val 0/1；新收尾诊断 1/1；自然 usable Program 0 | **361,204** | **25** |

旧 Train 最终回答是 `<answer>0.245</answer>`；旧 Val 最终回答是 `<answer>1,124,857,250.00</answer>`。两题沿用原评分器 `officeqa_skillopt_em_f1_v1`，hard=False、raw_score=0。新收尾诊断输出 `<answer>0.162</answer>`，同一评分器 hard=True、raw_score=1。它单独保存，未修改旧答案、评分、轨迹或费用，也不算新版本完整 fresh Train 成功。

fresh Train 使用原 seed42 canonical 顺序的第一题，Bank 初始为空。随后学习诊断分别使用新源码、新目录及独立空 Bank；它们不是同版本 resume，也不是重新完成的 Train 题。Val 是无资产只读执行验证，不算 Program 复用门槛通过。收尾诊断只重组原已取得的公开结果，未重新检索、执行程序或求解；之后的学习来源是明确标记的派生诊断记录，连接原工具轨迹与此次真实新收尾请求、响应和评分，原记录仍完整保留。所有自然学习诊断的 usable gate 均未通过，其内部复用 Val 自动跳过。未运行 Test，没有重开正式矩阵。

本次全部 HTTP 已计量，没有 unknown billing。累计费用仍在本次 496,000 tokens／50 HTTP 和原 Office 总额度内。包含历史账本的 Office 累计为 **24,553,622 tokens／3,018 HTTP**；所有旧账本 SHA256 和来源记录保持不变。原 Train 求解实际 284,243 tokens；四次自然学习合计 **18,580 tokens**，低于授权的 80,000 学习池，未发生学习 token 预算耗尽。此次最终回答诊断上限 24,000 tokens／3 HTTP；随后学习上限 60,000 tokens／10 HTTP，复用 Val 上限 40,000 tokens／6 HTTP，因无 usable Program 而未发出该 Val 请求。另有诊断启动脚本遗漏 logger begin_task 的零 HTTP 失败，已保留失败目录并在新目录修正，未改变生产模块。

## 确认并修复的工程问题

1. 原学习目录按 `execute_python` 名称合并，第一次读目录页的脚本覆盖了后来真实执行成功的数值计算。目录现按执行方式保留代表，优先真实 typed computation，再保留调用工具的代码；仍最多四张卡，没有按题号或 benchmark 分支。
2. 数组被重复展开成大量元素来源，挤掉后面的独立标量。目录保留数组聚合 AST 路径，并合并相同的 literal 值；公开任务字段及原生参数字段不按值合并。核心来源解析、绑定和晋升规则保持原样。
3. 前缀构造跳过临时程序内部调用，而 Learner 的 preflight 又把内部调用算入前缀，产生 `Trial prefix must be a real action prefix`。两处现使用同样的父动作序列；原始内部调用仍全部保留并计费计预算。
4. 原来源试用遇到早先已经发生的脚本失败就拒绝重放，即使后续真实操作成功。现只在原动作、finished 状态、accepted=False，以及 observation/data/error/done 全部精确一致时认可该失败已复现；不同错误、原先成功现在失败仍拒绝。后续程序输出仍须按原规则完全匹配真实来源输出。
5. Runtime 提示明确 `ctx.call` 返回原生业务类型，内存中的 preview 包装不能照搬为 RPC 字段访问；同步公开既有 stdout 4,096 B 上限。没有增加输出上限、工具预算、权限、repair 次数或发布路径权限。
6. 容量压缩先移除次要目录卡，再缩短唯一主要证据的代码预览；精简重复说明，仍保留完整工具 Schema、真实输入和输出引用。最终真实 Learner 请求为 **16,368 B ≤ 16,384 B**，完整 926 字符计算源码可见，未发生 material_too_large。
7. 原 Train 实际已经算出 `data.computed.abs_diff_3dp=0.162`，但原 finish_only 请求中没有该值：长表格占满输入空间，计算结果只剩不可解析的引用。公共收尾现先保留已获得的短计算／Program 业务输出，再拟合长文本窗口；同优先级保持原顺序，来源引用、Host 字段过滤、allowed_result_ids 和字节上限不变。没有按 benchmark、题号或正确答案分支。新版真实收尾 wire 为 **16,275 B ≤ 16,384 B**，包含原取得的计算结果，评分 gold 未注入请求。之后真实 Learner wire 为 **16,367 B**，无容量错误。

Train 原始调用共 58 次：19 个外层、39 个内部；其中 7 个临时脚本，发生两次普通模型执行失败：把字符串按 `['content']` 下标访问、单个脚本用尽内部调用预算。这些错误原样保存。4fa6d40 的 Val 共 24 次调用：5 个外层、19 个内部，含 4 个临时脚本，全部无接口错误，最终用满原定 Val 工具预算并进入 finish_only。Train 错答有明确收尾材料工程缺陷；不能将它全部归为模型能力，也不能据这次单点修复推断整个 benchmark 准确率已经恢复。

## 验证及其边界

最终固定 e4db6a4 源码共 **193 项针对性回归通过、25 项按历史材料／旧配置条件跳过**：扩展学习、临时 RPC、来源绑定、发布权限、冻结检查为 156/11；收尾、公开材料、原恢复／预算检查为 37/14。报告附 JUnit 原始结果。此前另有真实 Train 记录的零 HTTP 回归：模拟 Learner／Builder 响应，使用生产来源绑定、真实 Docker 和独立来源重放，生成的参数化计算通过 `source_output_match`，按原规则成为 usable。输入、源码和参考输出来自真实执行记录，没有使用私有 gold，没有手工晋升。

测试曾因从 WSL 直接读取 Windows Git worktree 失败，改用 Linux 仓库；扩展测试的首次调用未设置真实 Docker image 环境变量，使用了测试占位镜像 aaaa，随后按项目的锁定镜像配置重跑。一次历史预算 fixture 的 1,000 tokens 无法容纳当前完整收尾协议，调整为 1,500，仅改变测试场景，生产预算不变。失败和最终日志均保留。真实现场的收尾零 HTTP 回归另有独立目录，不将模拟请求的 fixture 用量计入实际费用。

这个 fixture Bank 明确为测试资产，不是自然学习成功，不进入任何正式初始 Bank。修复后的自然 Learner 三次均选择 no_change；最后一次已看到正确最终提交、完整计算代码、typed 输出和来源引用，仍认为临时原生代码已完成参数化计算，没有新的可复用资产缺口。此决策无结构／容量／来源错误、没有耗尽预算。无自然 Builder 调用、无自然 usable Program，因此 **跨题 Program 复用与成本下降尚未被本轮证明**。没有强制提议、手工晋升、增加 repair 或继续抽样直到成功。

上述已确认的工程缺陷已修复，临时检索入口、严格来源试用和收尾计算输出可以运行；真实单点收尾结果已通过。但 Program 自然学习／复用门槛未通过，不能宣称 OfficeQA 已整体恢复准确率或完整新版本 fresh Train→Frozen→Val 已获验证。当前 formal 的 STOP_AFTER_TASK 标记仍存在，旧费用及 Bank 保留。

## 审查材料

`OfficeQA_临时程序与学习来源修复_实际付费审查包_20261012.zip` 包括六个实际事件的身份、配置、任务身份、已有 summary、逐题 trace／派生诊断来源、物理请求／响应／raw_usage、账本、checkpoint、实际冻结库及哈希验证；另外提供零 HTTP fixture、零 HTTP 启动失败记录、源码、回归结果、启动脚本、公共 manifest 和逐文件 SHA256。没有为旧记录补造不存在的 summary 或新版本 fresh 实验。

各事件分别保存源码身份；原文件内容不回填、不改写。为便于 Windows 解压，压缩包内文件名中的 `:` 改为 `_`，映射表记录原路径和原内容 SHA256。不包含密钥、完整语料或整个项目。此前离线包和失败记录均保留。
