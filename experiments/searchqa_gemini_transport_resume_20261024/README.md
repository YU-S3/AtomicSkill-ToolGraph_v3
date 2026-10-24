# Gemini Test1400 传输故障修复与续跑

旧 run 在第19题遭遇两次 HTTP504，未返回 usage。保守预留分别为81,072和26,919 tokens；合计107,991，加收尾预留20,009，恰好占满单题128,000。旧重试把第二次输出 cap 从65,536压到11,383，随后无法继续。根因是上游504触发了本地传输费用与答题预算混算的问题；该题尚未进入提交 Program。

## 修复范围

- 已确认的传输失败仍完整计入累计总账；仅在单题预算中与有效求解费用分开，保留可核验的失败记录 hash 和金额。不将未知费用记为0。
- 重试保留原始输入、模型参数和输出 cap，不因传输失败压缩正常求解预算。最多初次加3次重试，退避30/90/180秒；计数和等待时间持久化。
- 原失败题已用2次物理请求，续跑最多再用2次，未重置次数。原始 payload 在发出续跑请求前进行 hash 核验。
- 总额度不足或重试耗尽时停止、不评分。重启时先检查未完成的传输重试，避免误转入额外收尾回答。
- 保留现有 Docker 检查；仅在可证明纯 Program 未启动时复用已付费模型回答恢复提交。Docker 控制器恢复最多3次，次数跨进程保留。

正常单题上限128,000、最大输出65,536、high reasoning、公开输入、评分器和 GEPA 风格人工 Bank 均保持原配置。新的控制代码在独立交付目录；冻结运行源码仍为`28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`。本次没有修改其他模型的已完成结果。

## 运行与费用身份

- 父 run：`/home/yangchengyu/searchqa_target_accuracy_r2_20261024/gemini`
- 子 run：`/home/yangchengyu/searchqa_gemini_transport_resume_20261024`
- 完整保留父 run 前18题（17题正确），子 run 只执行剩余1382题；报告仍以1400为分母。
- 累计额度保持10,272,289 tokens / 2,257 HTTP。保留此前全部请求、已计量费用及未知预留。启动前按保守预留计算，剩余8,054,759 tokens / 1,846 HTTP。
- 启动及费用结转见子 run 的`continuation_manifest.json`、`cost_carryforward_manifest.json`、`budget.json`。原目录和原 ledger 不改写。

## 验证

`offline_acceptance.json`记录11类零真实HTTP检查，使用生产 Provider、原生图、账本和真实 Docker 提交：普通成功、连续两次504后成功、中断后恢复、原第19题历史失败迁移、重试耗尽、总额度不足、含usage断流、缺usage断流、读取超时、Docker提交恢复，以及拒答/认证/余额错误不重试。

检查确认：20个拦截请求；正常输入与所有重试 payload 相同；历史失败请求计数不清零；历史账本条目不变；未知预留不删除；总额度不能绕过；已完成任务重复执行不产生HTTP；Frozen hash不变。

真实启动核验另见`live_acceptance.json`。自动重试处理短暂故障；若上游持续不可用、进程出现未能确认计费的中断，或既定总额度耗尽，会保留现场停止，不以无界重试或删除费用保证表面完成。

WSL监控：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_gemini_transport_resume_20261024/continue_test.py status
tail -f /home/yangchengyu/searchqa_gemini_transport_resume_20261024/run.log
```

代码、Bank和计划 hash 已冻结，运行中不要编辑。`check`为只读身份核验，`status`为只读进度查询；不要重复启动`run`。
