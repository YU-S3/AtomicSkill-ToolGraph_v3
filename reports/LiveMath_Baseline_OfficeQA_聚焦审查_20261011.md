# LiveMath、Baseline 与 OfficeQA 聚焦审查报告

本轮发现了明确的比较协议问题和仍未关闭的 Office 构包问题。Ours 新旧 Test 的题目与评分一致；其 16/100 由 54 个合法错答、29 个截断空答及 1 个其他空答构成。旧 NoSkill 的 95/100 使用了暴露 gold 的输入，不能作为同协议数学能力对照。当前 Office v4 已停止在六题工程 gate，未继续完整 Train。

审查使用既有记录与确定性构包/准入回放，新增模型 HTTP 为 0。生产代码、任务顺序、评分、Bank、预算和 repair 次数均未修改；审查前后 2,245 个封存源文件 hash 一致。SpreadsheetBench 保持原独立代码运行；seed43/44 保持当前暂停状态。本轮审查不等于实施新的生产修复。

## 1. 运行身份与材料边界

| 对象 | 实际身份 |
| --- | --- |
| Ours 旧 20/100 | commit `133f88ea809f7c14616641b6dca0a66bd38f2a63`；`cf4_r3_fresh_formal_deepseek_4bench_3seed_20261009_m4/livemath/seed42` |
| Ours 当前 16/100 | commit `80762b4c84fc71d56a42d9783f1fdd3e70ce43ab`；`atomic_unified_r1_selector_seed42_formal_20261010/livemath/seed42` |
| Office 旧三题前缀 | commit `a354ef5b7e7541cd5c8152c21f70a6458d521003`；`officeqa_r1_fresh_seed42_formal_20261011/officeqa/seed42`；已停止，单独保留 |
| Office 当前六题 v4 | commit `17989e473315c9a04f1723c5ac0fd1d2d8e111e6`；`officeqa_r1_v4_fresh_seed42_formal_20261011/officeqa/seed42`；`learner.material.public-directory.v4` |
| NoSkill seed42 | 归档实际选定 `formal_ds_retry1/deepseek_v4_1_flash/no_skill/livemath/42` |
| SkillOpt seed42 | 归档实际选定 `formal_ds_retry2/deepseek_v4_1_flash/skillopt/livemath/42`；不能混用此前失败的 retry1 |

WSL run 的绝对根目录为 `/home/yangchengyu/`。Office 实际干净 checkout 是 `/home/yangchengyu/asg_officeqa_r1_v4_seed42_20261011`。Baseline 原归档位于 `D:/T3S_exp/deepseek_2026-10-07/deepseek_2026-10-07`；只审查 NoSkill 与 SkillOpt seed42，没有扩展到其余方法。

## 2. LiveMath 新旧同题与评分核查

新旧各 100 条记录的 task ID、physical_key、公开题干与显示选项的投影 hash、gold 全部一致。用现有纯评分函数重算 200 条预测，与原 hard score 全部一致。两份 Frozen 的原文件 SHA 均与冻结 manifest 相符，Test 前后 Host frozen hash 相同。本轮未发现题目错配、选项投影/gold 错位或合法标签被错误计分的证据。

显示 gold 分布为 A=22、B=20、C=24、D=19、E=15。实际 100 题中 55 题含所指 meta-option，而且这 55 题的 gold 正是该选项；其余 45 题是普通题。这是当前资源的事实，不用于增加 Test 选项特判。

| 指标 | 旧 20/100 | 当前 16/100 |
| --- | ---: | ---: |
| 正确 | 20 | 16 |
| 合法标签错答 | 62 | 54 |
| 截断空答 | 18 | 29 |
| 其他空答 | 0 | 1 |
| meta-option 题正确 | 2/55 | 2/55 |
| 普通题正确 | 18/45 | 14/45 |
| Test 实际注入 guidance 的题数 | 100 | 3 |
| Frozen Skill 数 | 7 | 9 |
| Frozen Program 数 / Test 调用题数 | 0 / 0 | 0 / 0 |
| Test input tokens | 225,100 | 212,370 |
| Test output tokens | 1,625,854 | 2,521,964 |
| Test total tokens | 1,850,954 | 2,734,334 |

同题变化为：13 题均正确，77 题均错误，7 题对→错，3 题错→对。55 道 meta-option 题没有正确性变化；净减少的四题都来自普通题。可以描述这个逐题变化，不能仅凭净减少四题确定某项修改导致了退化。

当前实际 Test 审计有 132 次 enabled/high、max_tokens=32768 请求，另有 5 次 disabled、高 effort 字段仍发送、max_tokens=2048 请求。请求 model 为 `deepseek-v4-flash`，保存的响应 model 为 `deepseek-flash`。旧请求保留 completion_cap=32768、配置和模型记录，但没有 serialized_parameters；不回填完整 wire 参数。

当前 raw_usage 已报告的 reasoning 总和为 2,496,117；5 个 disabled 请求没有该字段，总 reasoning 保持未知，不将缺失强填为零。旧 run 的已报告 reasoning 为 1,625,234。output tokens 已包含 reasoning，不能再相加。

判断：低分同时包含大量合法错答、空答/截断损失和有限的实际经验覆盖。当前只有三题注入新 guidance，Frozen 中无 Program。现有记录不足以区分基础模型能力、提示/执行路径与学习选择各自的因果贡献；尤其没有同公开输入、同参数且无泄漏的 NoSkill 对照。本轮不基于 Test 再筛 prompt 或增加 meta-option 规则。

证据：包内 `01_livemath/*/test100_light.jsonl`、`new_old_same_task_comparison100.jsonl`、`audit_summary.json`、独立公开/私有资源与原身份/Frozen 文件。

## 3. Baseline 高分与比较协议

两方法与 Ours 的物理题重合 100/100。题干和归一化后的五个候选全文一致，但实际送入模型的结构不同：

- 两方法全部 100 道实际 user JSON 都包含 `mcq.correct_choice`，其 label 固定 A，连正确候选全文也直接可见。
- 实际 `mcq.choices` 全部只含 B、C、D、E 四个候选，A 被独立放在 `correct_choice`。离线评分导出后来补入缺失正确候选，没有重新推理，也没有随机化标签。
- 全部 100 题还向模型展示了 theorem、sketch、选项生成 `meta`/`sketch_usage_meta`。Ours 公开求解输入为题干与完整、固定 seed42 投影后的五个候选，gold、theorem 和 proof sketch 不作为公开求解输入。
- NoSkill 没有 skill；SkillOpt 的冻结 skill 正文确实完整注入了每个保存的 system prompt，正文还将 theorem 视为权威依据。这是输入与训练机制差异的实际证据，但不能据此证明原本地 adapter 完整复现了上游算法。

因此这是明确的 gold 泄漏和比较协议不一致。即使离线评分数学上按其原标签成立，这些高分也不能用于支持同协议盲测数学推理能力。无需用 meta-option 捷径假设解释 95%，因为原请求已经直接提供答案；也不能据此判断 NoSkill 在 Ours 的公开输入下会有同样成绩。

| 选定 seed42 | 原交付离线分数 | 原解析结果 | meta / 普通题原分数 |
| --- | ---: | --- | --- |
| NoSkill | 95/100 | A=95；其余 5 条原输出/解析形式保留 | 52/55；43/45 |
| SkillOpt | 0/100 | 全部解析为字符串 `FINAL ANSWER: A` | 0/55；0/45 |

SkillOpt 的 100 个保存回答均以 `Final answer: A` 结尾（其中一条之前有解释）。原解析逻辑不识别该格式，所以 0/100 不能解释为模型选错了 100 题。包里另列仅用于诊断的 anchored answer label=A=100，不替换原评分，也不把它改称为可比的 100%。所选 SkillOpt seed42 原导出并非高分；其他 seed/方法的成绩不混入此次审查。

模型方面：Baseline transport_snapshot 配置为 `deepseek-flash`、thinking=true、max_tokens=16384；实际 provider 记录字段名只有 model/messages/thinking/max_tokens，没有 reasoning_effort。Ours 当前实际为 high、Runtime 32768。Baseline 未保存完整 wire 参数值或响应 model，不能证明两者已是同配置实验，也不能据响应别名差异推断模型一定配错。

Baseline 归档缺少完整序列化 HTTP payload、原始响应 envelope、finish_reason、响应 model；`resolved_plan.source_commit` 仅为 `local-baseline-adapter-v1`，原本地 adapter 源码也未随原归档交付。报告与表中保留这些缺失。8 道不同物理题的真实已保存 prompt、响应、provider 元数据及 Ours 原请求已一并打包，没有将推断的 payload 冒充实际 wire。

证据：包内 `02_baseline/*/input_field_audit100.jsonl`、`test100_light.jsonl`、`original_offline_scores100.jsonl`、冻结 skill/transport、`samples/index.json` 与 8 组真实记录。

## 4. UID0115 结构修复现场

实际代码是 Office 六题 v4 的 `17989e4`，不是 `a354ef5` 三题前缀。求解已完成，hard=false；学习 rejected。普通 Extractor 已付费响应提出一个 workflow，首个节点缺少必需 `args`。按生产 Schema 复验精确重现：`$.workflow.nodes[0]: missing required properties ['args']`。

生产 `structure_repair_material()` 对原提案、该 Host 校验错误和原可见操作目录重放，输出与保存的修复材料完全一致。再用同提交实际 Provider `_build_payload()` 构包，并按生产 `json.dumps(payload, allow_nan=False)` 的 ASCII 转义及空格规则计量，准确重现 8,662 字节，超出 8,192 字节 470 字节。

| 部分 | 字节 | 占比 |
| --- | ---: | ---: |
| system 提示原始 UTF-8 内容 | 208 | 2.40% |
| user 修复材料原始 UTF-8 内容 | 3,724 | 42.99% |
| tools 的实际 JSON 序列化 | 4,107 | 47.41% |
| 字符串转义、message 外壳与其他参数 | 623 | 7.19% |
| 完整 Provider payload | **8,662** | **100%** |

最后一项开销是完整 wire 减前三部分的互斥余量，包含 JSON 转义/外壳及 model、cap、thinking、effort、tool_choice 等其他参数，不是 token 数。只有 user 材料大小不够判断准入；本例工具声明本身就占 47.41%。

该结构修复 admission_status=`local_refusal`，http_attempts=0，未产生这次修复的模型费用。已有求解与普通 Extractor 费用仍完整计量。无效提案没有被强行补成成功，也没有新增 repair。

普通 Extractor 的 v4 修复已在真实前缀中送达，不足以证明结构修复 8 KiB 路径关闭。本轮只是定位并完整保存现场；生产实现仍有该缺陷，后续修复应先用本例离线构包验收，再明确版本与续跑身份。不能仅放宽 gate、改整题预算或把 payload 上限等同于 token 上限。

证据：包内 `03_office_UID0115/` 的原提案/错误/请求/trace、完整 checkpoint、明确标注离线重建的拟发送 payload 和字节分解。

## 5. 当前 Office v4 六题前缀

实际 completed Train=6/120，状态 `stopped_at_task_boundary`，无正在执行的 Office 题。六题 hard score 均为 false；停止触发条件是第六题材料超限，工程 gate 没有使用准确率或 Program 数量放行。

模型已实际成功执行 21 次 read/read_result。没有执行 execute_python，不能宣称 compute 路径已获真实模型验收。普通 Extractor 均已发出并完成；当前六题的 finish wire 均低于 16,384 字节，disabled/2048 收尾已实际执行；原 glob/读取与动态 Runtime cap 的工程路径已有真实记录。结构修复仍出现 material_too_large，故整体工程链路尚未全部关闭。

| 题目 | 学习结果 |
| --- | --- |
| UID0208 | no_change，完成 |
| UID0170 | rejected，program_answer_schema_invalid |
| UID0221 | rejected，proposal_invalid；结构修复模型返回非法 ToolCall JSON |
| UID0122 | no_change，完成 |
| UID0148 | no_change，完成 |
| UID0115 | rejected，material_too_large；HTTP 前拒绝 |

UID0170 与 UID0221 的现有拒绝按原规则保留，不将普通模型输出失败转换为成功。当前 `prefix_engineering_audit.json` 不存在；生产 gate 的只读回放仍在同一处失败。包内独立失败回放明确写为 failed，没有补造通过记录。

| 费用来源 | tokens | HTTP |
| --- | ---: | ---: |
| 更早旧 Office 已计量账本 | 10,221,986 | 1,069 |
| a354ef5 三题前缀新增 | 249,778 | 41 |
| 本次 v4 六题前缀新增 | 515,120 | 83 |
| 当前整段已结转总额 | **10,986,884** | **1,193** |

原总 cap 仍为 36,096,000 tokens / 284,070 HTTP。全部 1,193 条账本为 metered，unknown_billing=false。新 run 的初始 Bank 为空；继承的是费用而非旧 Bank，原三题与六题属于不同提交/目录。账本复制链路只结转一次，未重复累计更早费用。

## 6. 离线核验与后续边界

固定 `17989e4` 下的相关回归 16 passed、0 skipped，包括 UID0091/0125/0164/0241 四个原预算拒绝案例的实际生产准入和有界收尾回放、原始首题 Extractor 回放、错误引用/缺失输入拒绝，以及合法既有 Skill 进入 Builder 的路径。Builder 使用现有锁定 Docker 镜像；模型 HTTP 由响应夹具拦截。最初一次 Builder 夹具因未设置 PROGRAM_IMAGE_DIGEST 使用占位镜像而失败；原日志保留，设置既有锁定镜像后通过，未修改生产代码。

这些通过验证的是已有机制与旧案例。UID0115 的重放正确证明当前仍拒绝 8,662>8,192，当前六题 gate 也仍失败；不能把其他回归通过当成该缺陷修复完成。

可确定的后续工作是 Office 结构修复材料的有界构包修复和零 HTTP 回放；本轮不擅自改上限或重启新 run。LiveMath 应先统一 Baseline 公开输入与解析/评分协议，现有泄漏成绩暂不进入同协议方法比较。没有依据为追求旧高分加入 Test meta-option 特判，或再启动矩阵寻找这些已能从原始记录识别的问题。

OfficeQA 与 LiveMath 的这批 Test 反馈已用于开发诊断，后续同 Test 的新版本成绩应披露该开发使用事实，不能称为从未触碰的盲测；泛化结论需另有未参与开发的评估依据。本轮报告不修改已有成绩，也不把当前六题低分升级为完整 Office 实验结果。

## 7. 交付与复核

材料目录：`D:/T3S_exp/deliveries/outputs/focused_audit_20261011/审查材料/`。压缩包：同目录上一级的 `LiveMath_Baseline_OfficeQA_审查材料_20261011.zip`。包含要求的四组材料及单独的无模型回放证据，没有整个仓库、其他七方法全量日志或新的收费实验。

`verification.json` 记录条数、源文件未变校验、缺失项和回归结果；`SHA256SUMS.json` 校验每个包内文件。公开题目与私有 gold 分开保存。旧字段缺失保持缺失，原始评分保持原值，拟发送 payload 与真实已保存请求明确区分。
