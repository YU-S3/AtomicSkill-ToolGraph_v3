# SearchQA V8 完整 Test1400

用户授权：“新结果试试test呢？局部题不能代表全部的整体效果”。

两版均沿用已完成192题对照的原始 Bank，不修改提示、Program 或输入。各自从独立目录重新执行 canonical Test1400；按原题单顺序运行，在同一题内以 seed42 交错两版，保持共享累计费用账本。主仓库和冻结核心源码不变。

- `v8_guidance`：具体文字经验。
- `v8_programmatic`：相同文字经验，加条件字段提取 Program。
- DeepSeek `deepseek-v4-flash`，thinking enabled，effort high，cap32768，单题96K。
- 新增上限16M tokens／6000 HTTP；承接此前全部费用及未知用量预留，原累计上限不变。
- 已有 V7 完整 Test1400 用作历史参考，本轮不重跑 V7。跨运行比较包含模型重复波动。
- 本轮是开发后的完整Test评估，不是新盲测；不自动晋升资产。
- 保留已知字段解析局限：纯标点占位内容可能被作为候选返回。本次测试原版本，不能将这类返回计作有效证据。

WSL监控：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B \
  /mnt/d/T3S_exp/deliveries/searchqa_v8_fulltest1400_20261020/run_fulltest.py status
```

运行产物：
`/home/yangchengyu/searchqa_v8_fulltest1400_seed42_20261020`

`preflight.json` 记录零HTTP启动前检查；`startup_verification.json` 记录实际付费前缀的模型参数、公开输入、原生图和Program执行检查。
