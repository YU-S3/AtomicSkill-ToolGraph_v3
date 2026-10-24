# Gemini Test1400：追加50万 tokens后续跑

用户批准“追加50万 tokens，继续跑完”。累计 tokens 上限由10,272,289提高至10,772,289；HTTP上限仍为2,257。正常单题128,000、输出65,536、high reasoning和既有重试策略均保持不变。

在任务边界停稳后保留1316题（1174题正确）。新目录只运行剩余84题，完整结果继续以1400为分母。旧账本1710条物理请求、已计量费用、全部未知费用预留和重试记录完整结转。没有重跑已完成题目。

- 初始18题：`/home/yangchengyu/searchqa_target_accuracy_r2_20261024/gemini`
- 后续1298题：`/home/yangchengyu/searchqa_gemini_transport_resume_20261024`
- 本次剩余84题：`/home/yangchengyu/searchqa_gemini_budget_extension_20261024`

运行复用已验证、已冻结的`searchqa_gemini_transport_resume_20261024`代码；原源码提交仍为`28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`。Bank、公共输入、官方评分器均保持原样。`budget_extension.py`仅负责明确授权的预算扩展、剩余题单和分段结果合并。

`offline_acceptance.json`记录零真实HTTP验收：准确增加500,000 tokens、HTTP限制不变、旧费用/预留/重试记录不变、canonical题目无重复遗漏、真实原生图和Docker提交通过、完成后resume零额外HTTP、Frozen hash不变。实际新配置仅改变运行目录和累计tokens上限。

预算授权与结转见新run根目录的`authorization.json`、`budget_extension_manifest.json`、`cost_carryforward_manifest.json`、`continuation_manifest.json`。原账本保存在原目录及新目录`parent_budget.json`，未改写。

WSL监控（本次路径已更新）：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_gemini_budget_extension_20261024/budget_extension.py status
tail -f /home/yangchengyu/searchqa_gemini_budget_extension_20261024/run.log
```

完成后自动写入`completion_report.json`，合并三段唯一题目及整个Test的失败请求成本。禁止运行中编辑冻结代码或重复启动`run`。
