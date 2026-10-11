# OfficeQA 学习容量与工程 gate 修复

本轮按用户提供的判断落实两项 P0 修改：统一完整学习请求的材料容量，并区分受控学习拒绝与工程阻断。没有修改 LiveMath P1 学习方法，也没有重新启动 Office 付费实验。

## 受测合同

| 项目 | 新版本 |
| --- | --- |
| implementation_revision | `empirical-v3.2-learning-capacity-r1` |
| learning material_version | `learner.material.public-directory.v6` |
| Extractor / Builder / structural repair 完整 Provider payload | 全部 16,384 B |
| structural repair | 仍一次；thinking disabled；输出 2,048 tokens |
| Office Train 单题 | 200,000 tokens；solve 120,000 + learning 80,000 |
| Office Val / Test 单题 | 96,000 tokens |
| Office seed42 整段累计授权 | 36,096,000 tokens / 284,070 HTTP |
| 数据划分和顺序 | 原 canonical Train120 / Val24 / Test102；run_seed42 |

容量政策用于公共学习请求；通用代码没有题目 ID、答案或 benchmark 特判。完整请求包括 system、user、工具 Schema 和 Provider 参数。额外容量不会绕过角色、任务父预算、learning pool 或整段 Governor。

## 修复结果

1. 保存的真实 9,415 B 带绑定请求，在生产 Provider 构包与 Host validator 的拦截式重放中逐字节保持原 payload，落在 16 KiB 合同内。原始提案仍因无效来源被 Host 拒绝；固定测试响应为 `no_change`，没有构造引用或晋升旧候选。
2. 无绑定、一条绑定、两条绑定都通过单次 repair 回归，保留模型提案和必要来源卡。错误输出来源继续被 Host 拒绝。真实 Docker 工作簿生成、发布、来源试用以及缺实际文件拒绝均通过；程序内 `/workspace/...` 与发布相对文件名仍保持各自语义。原有越界路径拒绝回归通过。
3. 超限前先做等值 JSON 紧凑序列化、重复合同驻留和现有可选历史裁减。只有识别到合法学习材料合同、可选材料耗尽、等值最小完整 wire 仍超限的本地拒绝，才附加 `learning_capacity_rejected`。原始 `material_too_large` 原因和完整拒绝 payload 保留；不完整证明继续视作未核清的工程错误。
4. Learner、Builder 和正式学习记录保存归一化原因。新 Train 的 `summary.json.learning_capacity` 分别记录学习请求拒绝数／比例、已完成任务拒绝数／比例；gate 同样单列这些指标。请求分母包含普通学习与结构修复的逻辑请求，含本地拒绝；任务分母为已完成 Train 题目。旧 pilot 不补填字段。
5. Office 工程 gate 不以分数、Program 数量或所有学习成功作为放行条件。六题固定前缀的受控容量拒绝需零 HTTP、完整材料和来源、费用与账本一致；模型非法结构或无效来源仍按原规则拒绝。可避免的容量问题、未知计费、缺来源／状态、Bank 链异常继续阻断。
6. `office-engineering-gate.v2` 收据绑定 clean commit、源码内容、config、execution manifest、run identity、六题 trace、真实账本与原上限、费用结转、gate 时 Bank digest。启动器检查通过状态、自校验 hash 及绑定；文件存在不能直接跳过核验。初次 gate 在持久化任务边界进行；已通过 gate 后，Bank 未变、源码／配置匹配且保留有效 checkpoint 的中断任务仍交给原 Runner 恢复。缺 checkpoint 或未解释的 Bank 变化继续阻断，不另造恢复协议。

## 离线验证

所有模型响应由固定 fixture 在真实 Provider HTTP 发送接口处拦截；没有网络模型请求。Host、Runner、Governor、文件适配器及 Docker Worker 按生产代码执行。fixture Bank 和人工 fixture Program 没有进入任何正式 run。

| 核验 | 结果 |
| --- | --- |
| Office / CF2 / Provider / 执行快照 / 统一方法回归 | 84 passed，7 skipped |
| 公共过程 / choice / LM1 / formal logger 回归，含容量回归 | 79 passed，2 skipped |
| 最终容量、费用一致性、失效收据、同版本 checkpoint 与发布材料回归 | 18 passed，0 skipped |
| 发布路径拒绝：输入目录、`../`、越界符号链接 | 3 passed，0 skipped |
| 启动脚本 | `bash -n` 通过 |
| git patch | `git diff --check` 通过 |

跳过项分别依赖另一个历史统一方法审查目录（7）和 LM1 历史深读目录（2）；本轮必需的真实 9,415 B、来源绑定、发布、预算和 gate 回归均实际执行。零 HTTP fixture 六题全部评分失败、其中一次容量拒绝，工程 gate 可通过：这是工程控制流验证，不能视为真实付费六题验收或准确率提升。

离线原始请求和收据、测试日志/XML、版本记录位于：

`D:/T3S_exp/deliveries/outputs/office_capacity_v6_20261012/`

## 已有实验与启动边界

旧 Office 六题保持暂停：

`/home/yangchengyu/officeqa_r2_v5_fresh_seed42_formal_20261012/officeqa/seed42`

其 commit 为 `0a7963732f1e2c86e439360c930bb09684983aed`，保留六题原评分、Bank、原失败 gate，以及累计 **11,510,372 tokens / 1,279 HTTP**（包含既有结转费用）。本轮没有重命名旧拒绝或重写旧 gate。LiveMath 继续使用独立 clean clone 的 `fff483d85a9a251fd71bf647dbce8b535b29c2f1`，不热修改受测代码。

Office 继续保持用户此前要求的暂停。下次启动必须使用新 clean commit 和独立空 Bank；启动器仅复制最新旧账本、明确记录 `bank_inherited=false`，历史费用在原额度内结转一次。前六题是新正式 Train120 的正常前缀，工程 gate 通过后同版本自动续至 Frozen、Val24、Test102；没有额外独立收费六题诊断。旧六题不作为新在线训练的初始资产，也不跨版本伪装 resume。

新 Office 默认输出目录：

`/home/yangchengyu/officeqa_capacity_v6_fresh_seed42_formal_20261012/officeqa/seed42`

在交付的固定 clone `/home/yangchengyu/asg_officeqa_capacity_v6_seed42_20261012` 中，用户决定开始时可运行：

```bash
cd /home/yangchengyu/asg_officeqa_capacity_v6_seed42_20261012
bash scripts/run_officeqa_r1_seed42.sh --execute "$(git rev-parse HEAD)"
```

应在持久终端（例如既有 tmux 会话）执行。`--print` 仅打印底层命令，不进行付费实验；使用 `--execute` 才会执行费用结转、正常六题前缀 gate 和完整正式流程。只有同目录同版本中断才使用 `--resume`。
