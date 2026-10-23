# MiniMax R2 完整 Test1400

按用户“能否启动test？可以就启动”的本次指令，独立运行完整 canonical SearchQA Test1400。复用已完成96题验收的冻结 R2 Bank，保留 Atomic / Workflow / Implementation / Program 结构；Bank 内容、MiniMax-M3 参数、公开输入和官方评分器均保持一致。此前小规模测试的成绩不继承，本轮1400题均重新求解。

这是接触旧 Test 后的人工 Bank 开发验证，不是盲测，也不能用来宣称自然学习泛化。目标82%—85%尚未得到完整测试验证。

单题128K tokens，completion cap65536，adaptive thinking，stream，generation seed不支持。正常每题一次模型求解，再执行真实 submit Program。旧账本计量7,666,801 tokens／2195 HTTP，未知预留154,912 tokens全部结转。根据本次完整测试启动指令，执行端为本轮设置新增6,000,000 tokens／1600 HTTP停止上限；累计上限13,821,713 tokens／3795 HTTP。超限停止，不自动扩大。旧Docker200题仍不计入旧有效结果成本，但物理账本保留。

实际源码28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9。新目录：`/home/yangchengyu/searchqa_minimax_r2_test1400_seed42_20261023`。Docker不可用时停止并保留已付费回答，不继续让题目批量失败。

WSL监控（20秒刷新）：

```bash
watch -n 20 /home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_minimax_r2_test1400_20261023/run_test.py status
```

任务边界暂停：

```bash
touch /home/yangchengyu/searchqa_minimax_r2_test1400_seed42_20261023/STOP_AFTER_TASK
```

完成后生成根目录和交付目录的 `completion_report.json`。原始请求、轨迹、SQLite、Frozen和配置保存在运行目录。本轮不自动改Bank、不启动第二轮。
