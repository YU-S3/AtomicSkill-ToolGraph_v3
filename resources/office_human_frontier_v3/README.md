# Office HumanFrontier V3

独立人工 Bank，源自用户提供的 HumanFrontier Bank 与 2026-10-17 复审建议。
这不是 Ours 自然学习产物，也不生成 usable 晋升或消费成功凭证。

实际 Runtime 读取 `bank/bank.json` 的 guidance。`SKILL.md` 和 `skills/` 与其同步。
五张过程卡按已取得的公开证据选择，每轮最多两张；`finish_only` 仅接收提前按真实 payload
计算的材料。检索结果、表格匹配和局部算术分别报告其有限证明范围。

计算 Program 的 ratio/difference/change 使用具名参数。旧 Test Bank 和旧请求保持原样，
不把旧位置参数自动改成新参数。旧固定参数的可选 helper graphs 与新 ABI 不兼容，
新版仅保留动态过程图及更新后的 Implementation；本次运行不强制执行图。

候选来自实际结果引用，引用及全部已执行操作数匹配后才置 `ready`。
它只表示引用一致性，不表示题意、来源选择或最终答案已被 Host 认证。
不自动替模型提交，不改写 `outputs_consumed`。

入口：`experiments/run_office_human_phase_v3.py`。三条 canonical Train 前缀分别进行
off/manual 配对，单题同为 192,000 tokens / 48 native calls，保留原重试设置。
六次执行总上限 1,152,000 tokens / 600 physical HTTP，并计入原 91M 累计额度。
两臂都使用相同文本语料、模型、原生工具及评分器；manual 臂增加本 Bank 与其证据投影。
不新增视觉输入，不运行 Test；这是方法包的对照观察，不是纯 Skill 文本单因素消融。

训练样本仅用于观察冻结人工 Bank，`readonly=True`，不会学习或跨题更新 Bank。
原 Test 已用于接口审查，后续泛化结论需披露接触史。

复核命令（在固定 WSL checkout 中运行）：

```bash
PYTHONPATH=src python experiments/build_office_human_phase_v3.py
OFFICE_HUMAN_ARCHIVE=/path/to/completed_test PYTHONPATH=src python -m pytest tests/test_office_human_phase_v3.py -q
PYTHONPATH=src python experiments/audit_office_human_phase_v3.py --archive /path/to/completed_test --output offline_replay.json
```

包内 Program 使用标准库和原有 Docker Worker，没有外部网络或模型调用。
