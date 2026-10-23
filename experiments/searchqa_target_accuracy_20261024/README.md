# SearchQA 三模型准确率迭代

目标：MiniMax ≥82%、Gemini ≥88%、GLM ≥84%。仅完整 canonical Test1400 的官方 EM 用于判断是否达到目标；目标是验收条件，不是预先保证的结论。

已核对的基线：MiniMax R2 1146/1400（81.8571%）；GLM V7 最近一次1153/1400（82.3571%）。Gemini Ours V7仅完成56题，没有完整基线。提供的GEPA Gemini seed44为1233/1400（88.0714%），属于参考方法。

## 候选

- `scope`：明确缺失角色、证据关系、最小完整答案及例外；保留此前3组有来源的政府关系卡。
- `reference`：相同规则，加上从提供的GEPA最终Skill整理的34条背景卡，按公开题干选择，最多4条。记录实际注入卡片ID。
- 两版均为人工编写的Ours原生Atomic / Workflow / Implementation / Program资产：一次模型solve，真实Docker submit。Program保留原始/转换后答案与规则收据，只进行有界格式转换。
- 对历史Test及参考成品的检查已用于开发，全部结果标记为post-Test development；不能作为未见测试集上的泛化证明或自然学习晋升证据。

## 固定实验

按20261024随机种子，在1400题中不按结果筛选，固定96题。各模型同时期对比旧基线、scope、reference，公共输入、模型参数及官方评分器保持一致。每模型选择正确题数最多的新候选，平手取更少tokens；随后独立重新请求一次完整Test1400。96题用于候选选择，最终目标只看完整1400题。不自动启动第二轮完整Test或将新库替换为生产默认库。

沿用用户的付费迭代授权，当前轮冻结新增上限：MiniMax 8M tokens/2000 HTTP、GLM 8M/2000、Gemini 10M/2200；合计26M/6200。该上限包括比较、完整Test及有限重试，不是预期用量。历史请求与未知计费预留逐项继承；未知费用保留原预留，不冒充零费用。

请求限流、瞬时服务错误和明确的SSE连接未完整结束，最多3次额外请求，10/30/60秒退避，重试次数跨resume保留。未知费用仍占本题和全局额度；重试输出cap只能在原额度内缩小。鉴权/额度/拒答不按瞬时故障无限重试。Docker不可用时保留已支付回答与checkpoint，并有限等待恢复。模型答错、合法拒答及生成失败如实计分。

## 本地运行与监控

在WSL执行：

```bash
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
D=/mnt/d/T3S_exp/deliveries/searchqa_target_accuracy_20261024
$PY -B "$D/run_campaign.py" status --model minimax
$PY -B "$D/run_campaign.py" status --model glm
$PY -B "$D/run_campaign.py" status --model gemini
```

输出目录：`/home/yangchengyu/searchqa_target_accuracy_20261024/{minimax,glm,gemini}`。
每模型含`budget.json`、费用结转清单、probe96、候选决策、fulltest1400、请求/轨迹/SQLite、Frozen及源码hash。
完整结束后生成`completion_report.json`，含是否达到目标。`supervisor.json`记录运行或停止原因。

生产源码固定为`28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`，本轮变更位于独立交付目录。旧运行与Bank保持原身份。
