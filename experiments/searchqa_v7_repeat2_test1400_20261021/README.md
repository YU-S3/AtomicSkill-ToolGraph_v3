# V7 两次独立完整 Test1400 重复实验

用户授权：“V7再启动两次重复的test”。

使用原完整 Test 达到1190/1400（85.00%）的 `v7_text_compiler`，新增 `v7_repeat_1` 和 `v7_repeat_2`。两次均重新请求全部1400题，不复用历史回答；分别保存输入、Frozen、SQLite、请求、轨迹和评分。两次按同一canonical顺序逐题交错运行。

Bank、核心源码、公开输入、评分器和模型参数保持一致：DeepSeek `deepseek-v4-flash`、thinking enabled、effort high、cap32768、单题96K。仍使用run_seed42；供应商不支持generation seed，因此这不是seed43/44消融。实际响应model和backend fingerprint逐请求记录。

本次合计新增额度16M tokens／6000 HTTP。继承已结束的192题实验的累计账本，并为仍运行的V8完整测试保留整个16M／6000额度，避免复制其在途请求或遗漏它的费用。原累计上限不提高，R3及旧未知用量预留保留。合并审查时，按物理请求ID去重，不能直接把两个带共同历史费用的账本相加。

本轮不编辑Bank、不训练、不按中途成绩停止或晋升。完成后比较原始运行和两次重复的EM/F1、tokens、原始预测变化及对错翻转。

WSL监控：

```bash
watch -n 30 '/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_repeat2_test1400_20261021/run_repeats.py status'
```

运行目录：
`/home/yangchengyu/searchqa_v7_repeat2_test1400_seed42_20261021`

`preflight.json` 记录零HTTP的身份、配置、题单和额度检查；`startup_verification.json` 核验真实新请求与原V7的可见messages逐字相同，以及独立请求身份、原生图和Program执行。
