# MiniMax SearchQA 准确率 R1

在完整 V7 Test1400 的 1116/1400（79.7143%）基础上开发的独立人工 Bank。用户目标为至少 82%，争取 85%；该目标不是保证或当前实测结论。

主要修改：更明确的缺失关系推断、答案范围和常用姓名形式；公开来源的机构隶属、副总统任职、参议院制度事实卡片；真实提交 Program 对 HTML、引号/破折号和严格限定数字格式做确定性处理，并记录修改前后内容。常规执行仍是原生 solve → submit，每题一次模型调用。没有把 Test ID、逐题 gold 或答案映射装入运行资产。

这是接触既有 Test 后的人工开发，不能当成未见测试集上的方法泛化或自然学习成果。旧 1400 题、评分、请求、Bank 和费用保持原样。字符修复的旧答案回放是离线诊断，不替换旧实验结果。

`packages/minimax_r1` 包含 7 Atomic Skill、4 Program、4 Implementation、1 Workflow。三个常驻指导 Skill 分别负责目标/证据、答案范围/事实卡片和表达/提交。额外文本工具沿用 V7 的公开 context 长度策略，6000 字符以内一般只执行 submit。人工资产为 author_trusted，Program 原生导入状态为 candidate，不伪造学习晋升。

`paired128` 在完整 canonical Test1400 中按 seed20261023 均匀无放回选 128 题，保留 canonical 顺序；没有按旧答对/答错筛选题目。同期 V7 和 R1 逐题轮转，两边使用相同 MiniMax-M3 adaptive thinking、stream、65536 completion cap、每题128K 和官方评分。generation seed 不受支持。试验结束后报告净得失及成本，不自动启动 Test1400。

实际源码为 `28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`，source SHA256 `ffe84c8a1718ae7286d479f07b4828a9613a0f1bb0c7cd9f8326c2ed805a9486`。相对上一 MiniMax 源码包含 Gemini 适配和缓存请求审计恢复补丁；MiniMax 的最终参数在离线验收中已逐项核对，同期两臂源码一致。运行身份与所有输入/Frozen/hash 由 plan 锁定。

试验沿用原 MiniMax 累计 8M tokens 上限，不增加额度。启动时物理计量 6,494,941 tokens，加保留的 154,912 unknown 预留，剩余 1,350,147。新请求最多300次（正常256）；旧 Docker200 费用仍从有效结果指标中排除，但物理账本完整保留。达到额度或出现新未知计费、Docker故障即停止。

零 HTTP 验收覆盖实际编译、Docker Worker、真实提交输出消费、字符/数字边界、完整公开输入和真实 MiniMax payload、完成后 resume 不重复请求、收到模型回答后 Docker 故障的缓存续跑，以及传输失败停止。`offline_acceptance.json` 保存收据。

WSL 只读监控：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B \
  /mnt/d/T3S_exp/deliveries/searchqa_minimax_accuracy_r1_20261023/monitor.py
```

运行目录：`/home/yangchengyu/searchqa_minimax_accuracy_r1_20261023`。`STOP_AFTER_TASK` 文件可请求任务边界暂停。只读统计 `analyze.py` 的输出严格使用两边均完成的同题集合。

事实卡片来源：[美国内政部](https://www.doi.gov/bureaus)、[参议院副总统历史表](https://www.senate.gov/about/officers-staff/vice-president/vice-presidents.htm)、[参议院宪法说明](https://www.senate.gov/artandhistory/history/common/briefing/Constitution_Senate.htm)。来源、查阅日期与适用关系随 Bank 冻结，模型可使用这些事实，但必须服从题目实际询问方向和年代。
