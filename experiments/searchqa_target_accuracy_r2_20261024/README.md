# SearchQA 三模型目标准确率：实质不同策略的对照

目标为 MiniMax≥82%、Gemini≥88%、GLM≥84%，以完整Test1400官方EM验收。当前仅完成改库、工程验收和真实启动，尚未证明达到目标。

## 本轮候选

- `scope`：重点约束缺失目标关系、答案范围、标题空缺、常用称呼与单复数；保留3组公共政府事实卡。
- `gepa`：将用户提供的GEPA Gemini seed44最终Skill完整策略，原文分配给solve、evidence、surface三条Atomic。其Train背景事实块完整进入evidence，避免稀疏选择漏掉上下文作用。没有导入Test逐题gold或按题号查表。
- 两者都经Ours原生Workflow编译，正常为一次模型solve→真实Docker submit Program。维持7 Atomic、4 Program、4 Implementation、1 Workflow。人工资产与Program状态沿用author_trusted/candidate，不冒充自然学习晋升。
- Program保留原始输出与格式变换收据，处理HTML/标点、姓名首字母间距、重音及有限数量格式；不判断答案语义是否正确。

已检查过历史Test输出，属于明确披露的post-Test development。参考方法成绩不能算作Ours成绩，局部比较不能代替完整Test。

## 实验与账本

使用先前冻结的同一96题（随机种子20261024，不按成绩筛选），比较旧baseline、scope、gepa。各模型选择答对数更多的新候选，平手取tokens较少者，然后各自重新请求完整1400题。原模型参数、公开输入、评分器保持不变。

先前`searchqa_target_accuracy_20261024`的稀疏卡片对照已在任务边界停止：两候选在96题中95题提示相同，不适合用来比较卡片效果。全部前缀、费用及未知预留保留在父run，当前子run显式继承。没有热改父run，也没有增加本轮总额度。

整个当前迭代（含已停前缀）新增封顶：MiniMax8M tokens/2000 HTTP；GLM8M/2000；Gemini10M/2200，合计26M/6200。真实调用按账本计量；没有usage的错误保留完整预留。有限重试沿用已验收实现。

源码固定`28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`。本目录为独立实验控制与Bank资产，旧源代码与Bank均不热改。

## 监控

WSL中执行：

```bash
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
D=/mnt/d/T3S_exp/deliveries/searchqa_target_accuracy_r2_20261024
for M in minimax glm gemini; do
  "$PY" -B "$D/run_campaign.py" status --model "$M"
done
```

结果目录：`/home/yangchengyu/searchqa_target_accuracy_r2_20261024/{minimax,glm,gemini}`。对照完成后自动生成`selection_result.json`并进入一次fulltest1400。完成时生成`completion_report.json`，其中`target_met`才是完整测试是否达标的判断。所有错误与重试保留。
