# OfficeQA 收尾窗口越界修复与重新启动

本次 OfficeQA 实际停在正式 Train 的前六题工程 gate，未开始 Val/Test，没有存活实验进程。不是 HTTP 连接失败。旧运行：

`/home/yangchengyu/officeqa_capacity_v6_fresh_seed42_formal_20261012/officeqa/seed42`

旧提交 `406e7ca6f1dc7cf2f4fa66acb4a2d4b811e322cf`，六题已完成，正确 1/6。本次前缀 517,866 tokens / 81 HTTP；含历史结转累计 12,028,238 tokens / 1,360 HTTP，均已计量，无未知计费。

## 原因与最小修改

UID0208 与 UID0148 的 finish_only 完整请求分别为 16,403 B 与 16,402 B，超过固定 16,384 B。两次均在本地拒绝，未发 HTTP。`TaskContext.fit_window()` 原先检查材料是否容纳后，再追加 `truncated`，导致已通过检查的窗口最终多出 18–19 B。gate 拒绝这个未解决的运行时构包错误是正确行为。

公共窗口函数现在在每次大小检查前计算截断标记；仍按完整源单位裁剪，保持偏移、后续读取位置和内容 hash。gate 仅补充非学习材料越界的明确错误消息，不放行旧失败。实现版本更新为 `empirical-v3.2-finish-window-r1`。

原 16 KiB 输入合同、finish 输出 2,048、学习一次结构修复、角色/父任务/整段预算、来源校验和晋升规则均未改变。原模型非法 JSON、未绑定来源和非法 Workflow 仍记录为普通学习拒绝，不增加修复次数，不修补模型答案。LiveMath 的独立 `fff483d` checkout 与已完成观察未修改。

## 零模型请求验证

- 本次新增 5 项测试全部通过，覆盖文本与数组的最终 metadata、两份真实失败材料经真实 Provider 构包/准入/拦截式发送，以及 gate 继续拒绝运行时越界。
- 两份真实记录重新投影后，完整请求分别为 16,267 B / 16,213 B；fixture 各一次发送，finish 输出仍为 2,048。固定 fixture 回答只验证接口，不评估准确率，也未写回正式 Bank。
- Office R1/R2/容量、CF2 接口、Provider、公用过程与 choice 回归共 115 passed / 6 skipped。跳过项为四个未配置旧请求档案的历史准入回放，以及两个未配置旧 UID0115 档案的结构修复回放；本次真实失败两题和 9,415 B 带绑定容量检查均已执行。
- 原始离线日志、JUnit 与 fixture 收据：`D:/T3S_exp/deliveries/outputs/office_gate_failure_20261012/`。

## 正式运行边界

代码改变后不得对旧目录伪装同版本 resume。新版本从空 Bank 在独立目录执行原 seed42 canonical Train120 → Frozen → Val24 → Test102。前六题仍是正常正式训练前缀；工程 gate 通过后同版本自动续跑，没有另付费运行的诊断六题。

启动器新默认输出：

`/home/yangchengyu/officeqa_finish_window_r1_fresh_seed42_formal_20261012/officeqa/seed42`

默认费用源为上述 `officeqa_capacity_v6_fresh_seed42_formal_20261012`。仅原账本结转，Bank 不继承，累计额度仍为 36,096,000 tokens / 284,070 HTTP。旧评分、请求、Bank、费用不改写。

交付固定 checkout：`/home/yangchengyu/asg_officeqa_finish_window_r1_seed42_20261012`。在该目录运行 `bash scripts/run_officeqa_r1_seed42.sh --execute <固定提交>` 开始 fresh；`--print` 只打印命令。仅同提交同目录的后续中断可使用 `--resume`。本轮没有发出新的模型 HTTP。
