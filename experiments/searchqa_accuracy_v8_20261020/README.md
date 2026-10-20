# SearchQA V8 人工经验 Bank 与三臂对照

日期：2026-10-20。用户授权：“按照上面的规划修改好bank，启动相关实验。”

## 修改

- 六类具体经验：目标关系、完整证据/字段匹配、证据冲突、答案范围、词形/数量、针对性提交检查。每类写明操作、判断条件及虚构反例。
- 归入三个原生 Atomic Skill：solve（目标与冲突）、evidence、surface（范围、表达及检查），满足现有原生 Skill 引用数量限制。
- 保留原生 Workflow 编译、Implementation、Docker Program Worker、ResultRef 绑定与 submit；不改主方法、公共输入或评分器。
- 原生字段提取 Program 复用已测试的明确 Q/A、管道符/制表符字段解析。仅在公开 context 存在结构标记且包含完整归一化题干时选择该分支。提取结果须由完整线索匹配产生；找不到匹配属于正常结果。
- 字段结果真实绑定到 solve 并发送给模型；记录“已送入求解”，不把它夸大为模型实际采纳或准确率收益。
- 不保存真实题目对应答案、真实训练事实卡片、任务 ID 分支或 gold。所有例子均为虚构的关系/表达对比。
- 人工资产为 author_trusted；Program 导入后保留 candidate 身份，不伪造自然学习或晋升。

## 已冻结实验

| 臂 | 内容 |
|---|---|
| v7_text_compiler | 已完成完整 Test 的原 V7，原样复制 |
| v8_guidance | 新具体经验；solve → submit |
| v8_programmatic | 与 v8_guidance 相同经验；按公开结构选择 source_fields → solve → submit |

三个实验臂均正常每题一次模型请求，无额外投票或反思。V8 两臂除 Program 条件分支、版本身份外，文字经验相同。

统一使用 deepseek-v4-flash、thinking enabled、high、cap32768、单题96K。generation seed 不受支持，保留该事实。完整题干/context 与同一官方 scorer 保持不变。

从 canonical Test1400 以抽样 seed20261020 均匀抽192题，按 canonical 顺序运行，每题三臂顺序固定随机交错。没有按历史正确/错误筛题。因之前已查看 Test 结果，本次明确为开发后的评估，不声称新的盲测。

新增额度封顶4,000,000 tokens／720 HTTP，继承已结束 DeepSeek fulltest 的费用和历史未知用量预留，保留 R3 余量及原累计总上限。与正在进行的 GLM 运行目录、模型配置和账本分离。

本轮不会自动推广候选或启动完整 Test1400。最终应报告三臂全部结果、改对和改错、tokens 比率、Program 实际绑定/字段匹配次数；准确率改善与不超过 V7 约1.5倍 tokens 是评估目标，不是逐题动态修改或筛结果规则。

## 离线验证

`offline_acceptance.json` 记录零真实 HTTP 验收：

- 八个纯解析用例：实体替换、关系变化、否定词、跨文档边界、明确标签、含糊逗号及无匹配。
- 三种真实公开输入（匹配字段／有结构无匹配／普通文本），三臂共9次拦截请求；实际原生导入、Workflow、Docker 执行、请求 payload、计费记录通过。
- 已完成 resume 不重复请求。
- 模拟网络错误后停止、保留未知用量并阻止未解决情况下继续计费。
- 全部经验实际进入 system 一次，完整 context 原样保留；真实 Program 输出送入 solve，Frozen hash 不变。

测试证明工程链路，不证明准确率改善。

## 运行位置与监控

部署控制器：
`/mnt/d/T3S_exp/deliveries/searchqa_accuracy_v8_20261020`

结果：
`/home/yangchengyu/searchqa_accuracy_v8_seed42_20261020/comparison192`

冻结基础源码：
`/home/yangchengyu/asg_searchqa_trace_r3_sources_fixed_20261015`

基础源码 commit：
`b1305aeb7075757be2888813e54e3f7aa7e88683`

控制器复用既有 `searchqa_accuracy_v7_20261019/campaign.py` 和 `searchqa_human_frontier_20261015/runner.py`；依赖身份记录在运行 plan。仓库副本用于归档，实际运行采用上述部署路径及已冻结环境，不依赖当前 main 的代码状态。

WSL 查看：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_accuracy_v8_20261020/monitor.py
```

持续查看：

```bash
watch -n 30 '/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_accuracy_v8_20261020/monitor.py'
```

运行日志：

```bash
tail -f /home/yangchengyu/searchqa_accuracy_v8_seed42_20261020/run.log
```

停止只需创建结果根目录的 `STOP_AFTER_TASK`，当前任务结束后停。续跑需先确认停止原因、账本及文件 hash，再显式使用 `run_experiment.py start --resume`；不要在进程存活时重复启动。
