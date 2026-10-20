# OfficeQA HumanStrong V4

独立人工参考 Bank：12 张指导卡、7 个 Program、7 条 Implementation，原 4 个 Composite 仅保存。保留全部 V3 Program；不修改自然学习、晋升标准、旧 Bank 或旧运行。

新增 `table_calculate` 接收已获得文本的 `source_ref`、明确行列与单位，以及具名计算参数。它通过原有 `read_result` 分页读取并缓存同一窗口，复用既有表格提取、统计和多项式实现，返回精确来源行与计算结果。它不接受抄写的数值作为数据，不猜测缺失年份或目标实际值，不转换含混的表头。

该 Program 是显式 `final_answer`：Agent 选择完整任务计算并传 `submit_answer=true` 后，成功输出通过现有 Executor 提交；缺项或歧义返回 `needs_input`，不提交。原有中间 Program 继续可用。没有自动图调度或强制所有题走固定流程。

`bindings` 把计算字段映射至表格 `key` 列表。例如 `numerator: ["a"], denominator: ["b"]`，或 `observations: ["series"], actual: ["target"]`。源窗口 `offset/length` 按 Unicode 字符计；只解析完整行。同一窗口每次调用只读一次。数据单位必须一致；不提供隐式单位转换。`absolute_relative_change` 明确计算基于首值的绝对相对变化，其余统计字段遵守原 `series_metrics` 的操作合同，不能传入被静默忽略的参数。

两臂共同使用一条标量/日期/列表答案格式说明，避免仅因 Bank 提醒去除百分号而造成对照偏差。说明在正常请求及收尾请求的实际容量计算前加入。题面、来源页和评分仍使用已批准的外部完整版本；正常 thinking/high/32768、收尾不启用 thinking/32768、单题 216576 tokens、48 次原生调用均保持一致。

`experiments/run_office_humanstrong_v4.py` 固定所有 Test102、seed42 打乱每题两臂先后顺序，共 204 次运行。用户已批准累计 155M tokens；本轮理论上限 44,181,504 tokens，HTTP 上限由现有循环和重试计算。保留 V3 及全部祖先费用和未知计费预留，禁止旧账本变动后继续用过时额度。不增加 Train/Val，不自动重复。

这是已经分析过部分历史 Test 失败后的人工参考 Bank 观察，不能称为完全未见的 Test 或自然学习的 Ours 结果。离线注入固定调用只验证生产链路，真实模型能否选择这些 Program 要在独立的付费记录中报告。

先运行零 HTTP 验收，再在已提交的固定 WSL 源码目录中执行：

```bash
PYTHONPATH=src python experiments/run_office_humanstrong_v4.py prepare --root <新目录>
PYTHONPATH=src python experiments/run_office_humanstrong_v4.py check --root <新目录>
PYTHONPATH=src python experiments/run_office_humanstrong_v4.py launch --root <新目录>
PYTHONPATH=src python experiments/run_office_humanstrong_v4.py status --root <该目录>
```
