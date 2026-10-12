# LiveMath P2-R3 有限付费验证启动说明

下一步运行一次 P2-R3 的内容检查、适用性判断和 Train 内部 off/on 配对。旧 P1 Train60、旧 P2 结果和本轮离线收据完整保留。已生成待启动 plan，并通过零 HTTP 预检；当前状态为 prepared_not_started。

## 实际实验范围

- 第一组：复用已付费的最终提案（guidance 248、applicability 75、完整卡片 500 字），从独立模型内容检查开始，不重复诱导、生成或 repair。
- 第二组：复用已付费的能力需求（required_conditions 231 字），先生成候选指导，仍最多一次结构 repair，再进行独立内容检查。
- 内容检查通过的候选才进入原固定 12 道 Train 公开输入的适用性判断。实际适用少于 4 题，停止该候选的配对。
- 按原固定 6 道初筛＋6 道确认做 off/on；只运行实际适用题，初筛未通过则停止确认。晋升仍需原净增正确数、空答和回退门槛全部满足。
- 最多两个候选；只有通过本次真实验证的资产才发布并 Frozen。完成后先分析结果。

不重复运行 P1 Train60，不运行 Val17、Test100 或 seed43/44，choice_bounded_finalization=false。48 道来源池和原两组实际来源保持冻结。日志分别记录新自然候选与旧提案重验候选；不能把复用提案计成新版提示自然生成的候选。

## 身份与额度

固定源码目录：`/home/yangchengyu/asg_livemath_p2_r3_paid_seed42_20261012`。

固定提交：`19fc6d47140bc4bf0d79a350e9a1df318c6fa831`；源码 SHA256：`83ed2426442d89cd73d4b7f54ea5277a12f62636862fd4e9c45c725352ccc26f`。该版本基于已验收的 P2-R3 源码，仅补充付费续跑入口；远程后续 OfficeQA 修改在 main 合并保留，固定 P2 checkout 保持上述版本。

实验目录：`/home/yangchengyu/livemath_p2_r3_paid_seed42_20261012`；plan hash：`5d146ef000e0a9fe88878d859016ce09500e00eec0a9065359833382546e00c4`。

方法版本 `choice-capability-batch.v2`、提示版本 `procedural-actions.r3`。plan 显式记录原回放收据、旧源码身份、原始响应与物理请求、配置变迁和费用继承。新旧源码身份分别记录，旧响应没有冒充新版生成响应。

原 P2 已花费 **42,057 tokens／8 HTTP**。本次可用剩余 **1,757,943 tokens／72 HTTP**；沿用原 P2 总上限 **1,800,000 tokens／80 HTTP**，不会为完成配对自动加预算。原 seed42 累计 **7,137,815 tokens／372 HTTP**，本次新增费用继续结转。额度不足或计费状态未知，按既有规则停止并记录截尾。

预检记录为实验目录的 `launch_precheck.json`：check=passed，81 个历史/回放文件哈希一致，新增 0 tokens／0 HTTP。固定版本测试为 P2 专项 47 passed，相关回归 119 passed、16 skipped；跳过项为 14 项未提供容器参数和 2 项未提供历史 LM1 审查目录。真实来源材料经过 HTTP 拦截的生产模拟执行，验证了内容检查、两组候选、适用性、16 次 off/on、晋升/Frozen、独立拒绝、旧费用保留和重复 execute 不增调用。模拟答案和分类仅验证工程路径，未产生真实 P2-R3 效用成绩。

交付 main 合并远程 OfficeQA 更新后，P2 与新增 bootstrap 相关集成回归 50 passed、9 skipped；这 9 项需要专用容器参数。固定 P2 checkout 和 plan 身份未因此改变。启动脚本的语法及 --check-only 已实际通过，执行互斥锁工具可用。

## 启动与监控

在 WSL 中执行以下命令。脚本先检查固定提交、干净源码、plan、来源及剩余额度；flock 避免同一目录同时启动两个进程。

```bash
nohup bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_livemath_p2_r3_paid_seed42.sh \
  >> /home/yangchengyu/livemath_p2_r3_paid_seed42_20261012/execute.log 2>&1 &
```

只重新运行零付费预检：

```bash
bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_livemath_p2_r3_paid_seed42.sh --check-only
```

监控：

```bash
tail -f /home/yangchengyu/livemath_p2_r3_paid_seed42_20261012/execute.log
```

`progress.json` 是最近阶段，`result.json` 记录结束状态和新增/累计费用。结束后审查 learning/0、learning/1 的 learning.json/requests.json、candidate_order.json、preflight.json、utility_receipt.json、selection 的 off/on trace 与请求、budget.json 和 frozen_bank/freeze.json。分别统计重验候选、新自然候选、内容检查通过数、适用题数、实际注入、配对胜负和费用。
