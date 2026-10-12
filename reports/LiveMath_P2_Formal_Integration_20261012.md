# LiveMath seed42 正式 P2 接入交付

用户已冻结正式流程：独立空 Bank → canonical Train60（run seed42 顺序）→ 一次 P2 学习与 Train 内部配对 → Frozen → canonical Val17 → canonical Test100。

P1/P2 是 Ours 的方法版本。本次新正式 run 在同一源码身份、同一费用账本中完成上述流程，不用两个独立实验拼接，不继承历史 Bank，不用旧答复替代新的 off 臂。

## 实现与边界

入口为 `scripts/run_livemath_p2_formal_seed42.sh`，配置为 `configs/livemath_p2_formal_seed42.yaml`。唯一 selection 合同仍是用户上传并冻结的 `configs/livemath_p2_selection_seed42.yaml`。历史文档中的 `fff483d` 是旧 P1 来源身份；新正式阶段明确记录新的实跑提交、源码内容 SHA、合同 SHA、运行 ID 和 fresh Train 来源。

正式 Train 保存全部 60 道真实轨迹。12 道内部 selection 不产生学习材料或学习统计；48 道来源保留实际公共操作支持的 Program 学习与原本地资格检查。没有公共局部操作时，文字型跨题候选推迟到用户指定的 Train 后一次 P2 阶段，不再走逐题 P1 指导提取。

P2 复用已有 production batch：48 道经验作为来源池，按原固定的公开 task_id 顺序尝试两组来源，每组两道不同物理题，即本次直接生成仍只用四道题。未临时换题或扩展聚合次数。独立内容检查通过的候选才进入公开适用性判断；覆盖不足不做配对，初筛不通过不进入确认。晋升要求实际暴露至少 4 题、净增正确至少 2 题、空答不增加、正确转错误最多 1 题。正常无候选或效用不通过如实结束 P2，再做冻结评估；工程错误、预算停止、未知费用不自动放行评估。

最终 Frozen 在 P2 完成后生成，保留本次 Train 的合格 Program 及本次晋升资产。`train/summary.json` 的 knowledge_digest 是 P2 前的 Train 身份，其 frozen 字段为 null；最终资产身份以 `p2/result.json`、`train/frozen_bank/freeze.json` 和 `final_frozen_manifest.json` 为准。试用 snapshot 不作为最终 Frozen 发布。

Val/Test 使用目标公开题干与选项执行有界 capability_routing，仅匹配已满足来源和效用合同的资产；记录匹配结果、拒绝原因及实际注入 ID。判断不读取目标 gold、评分或答案，也不更新 Bank。每道题的匹配请求费用计入该题原 96,000 tokens 上限及累计账本；不是免费调用。没有 usable 资产时不额外发匹配请求。

生成要求继续为 required_conditions/guidance 180 字符、applicability 80、可选 reason 180；Host 容差分别为 256/256/80/512，完整卡片 512、原文引用 240。保持 16,384 B 材料、2,048 completion、一次指导 repair、selection 零重试、`choice_bounded_finalization=false`。

## 费用与恢复

用户确认新正式阶段的 P2 专项上限为 1,800,000 tokens／80 HTTP；原累计上限仍为 23,232,000 tokens／9,165 HTTP。本地旧 P1 与截至 r8 的 P2 账本按物理 request ID 去重结转 **7,507,922 tokens／411 HTTP**。累计尚余 **15,724,078 tokens／8,754 HTTP**，不代表保证能用完每题理论上限。

P2 通过一个阶段预算代理向唯一 campaign governor 记账，同时受专项与累计上限约束。P2 的 budget.json 是本阶段物理尝试的旁路投影，不是额外可花的第二份额度。作用域带新 run ID。训练完成但 P2 未完成的同版本恢复会进入 P2；已付费完成的学习单元、匹配与配对用原 checkpoint 恢复。已完成 P2、Val 或 Test 不重复调用。未知费用和未完成的外部副作用继续按原规则停下，不做盲重试。

## 零付费验收

- 核心 P2、历史真实响应回放、正式对齐、公开过程和新正式链路：**105 passed，0 failed，0 skipped**。
- 实际 Docker Program、参数变化、执行证据与嵌套来源回放：**32 passed，0 failed，0 skipped**。
- 新正式链路覆盖：Train 前缀停止且无早期 Frozen/P2；一次 Train60；有候选与无候选；真实 off/on 注入与资格发布；Val17/Test100；Frozen hash 不变；学习单元完成后的中断恢复；重复 resume 零新增调用；专项预算不能恢复后重置；历史费用去重与未知费用拒绝。
- 用已有合法来源候选和本地真实 117 道 Val/Test 公开输入构造 production 路由请求，全部通过现有 16,384 B 合同；HTTP 已拦截，未付费。
- 固定 WSL checkout 的启动预检核对实际模型、完整配置、公开数据文件 SHA、60/17/100 身份、48/6/6 合同、历史账本和真实 Docker 镜像；零模型请求。

这些是工程验收，不是新的准确率、自然候选生成率或迁移收益结果。本次没有启动正式付费 run。

## 启动与产物

独立 WSL 源码目录：`/home/yangchengyu/asg_livemath_p2_formal_seed42_20261012`。使用交付固定提交运行 `bash scripts/run_livemath_p2_formal_seed42.sh --check <commit>`，再用 `--execute <commit>` 启动。脚本禁止非空输出重新开始；同版本恢复用 `--resume <commit>`。不切换其他正在运行实验的 checkout。

默认输出：`/home/yangchengyu/livemath_p2_formal_seed42_20261012/livemath/seed42`。本次预检没有创建该 run 或初始 Bank。脚本只启动单模型单 seed42，连续跑到 Test100，不启动其他 seed。

审查时读取 `cost_carryforward_manifest.json`、`run_manifest.json`、`resolved_config.json`、`budget.json`；Train 的 execution_manifest/config/summary/traces/requests；P2 的 plan、source_records、learning、candidate_order、preflight、selection、utility_receipt、result 和阶段 budget；最终 frozen_bank 及 final_frozen_manifest；Val/Test 的逐题原始轨迹与请求。FormalLog 另外记录 P2 学习单元、配对评分、原始请求费用及发布资产。
