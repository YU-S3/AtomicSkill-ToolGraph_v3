# Office HumanFrontier V3.1

独立人工 Bank；5 Skill、3 Program、3 Implementation、1 动态 Workflow。
真实 Runtime 使用 bank.json guidance，SKILL.md 与 skills/ 同步生成。
保留 V3 原件，不将人工资产冒充自然学习的 usable Program。

阶段卡仍最多两张。简短的原生提交规则持续呈现；百分比只规范符号，不自动改变数值尺度。
分轮操作数累积去重，冲突不覆盖。replaces_source_ref 可更正同一语义槽的明确旧来源；
容量为40，超出明确反馈，不丢弃旧年份。operand_bindings 可消歧当前计算的来源。
明确财政/自然年度连续区间才产生期间覆盖；unknown 不冒充 complete，覆盖只检查期间集合。
recent_scalar_result_refs 展示真实计算子字段，候选必须匹配实际执行的必要输入。
ready 只表示引用与期间覆盖一致，不证明题意正确，也不阻止模型直接 finish。

scoped_search 获得窗口时返回局部 ok，result_role=intermediate；分页完整性保持实际值，
目标期间与表头不自动齐全。table_extract 的完整性只针对传入 rows，不证明整题覆盖。
所有计算公式沿用 V3。使用原生动态 Executor，不强制走图，不新增 Planner 或 LLM 调用。

入口 experiments/run_office_human_phase_v31.py。固定 canonical Train 前3或12题，
各题 manual/off 配对，保持单题192,000 tokens／48 native calls及原模型设置。
运行器从已完成 V3 继承完整账本和未知计费保留，不重置费用；默认累计授权仍为91M。
若新实验理论上界超过授权，prepare 拒绝，必须先确认新的累计额度。
12题双臂上界4,608,000 tokens／2,400 HTTP；3题双臂上界1,152,000／600 HTTP。
这是冻结人工方法包（Bank加桥接投影）的观察，不进行学习，不跑Val/Test。

构建：PYTHONPATH=src python experiments/build_office_human_phase_v31.py
验收：tests/test_office_human_phase_v3.py 与 tests/test_office_human_phase_v31.py。
OFFICE_HUMAN_ARCHIVE 指向原已完成Test目录以读取固定配置；OFFICE_PHASE_V3_ARCHIVE
指向已完成Train3配对目录用于真实材料的零HTTP Provider回放。
