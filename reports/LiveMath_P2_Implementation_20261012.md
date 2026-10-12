# LiveMath P2 配置与离线实现

本轮按用户明确要求完成修改与离线验收，新模型 HTTP 为零。P1 `fff483d` 的真实 Train60→Frozen→Val17 已结束，原目录、Bank、评分和费用保持原样；启动前复审文档中“P1 尚无完整轨迹”的描述是早期审查状态，不作为重新运行 P1 的指令。

## 固定合同

`configs/livemath_p2_selection_seed42.yaml` 原样保存用户提供的合同。公共 Train manifest SHA256 为 `31c1ea371fb967d9e933ea76f6edf5bc8c46255f8e33d4f79f968c036a3252ca`。按声明的 SHA256 salt、task_id 排序固定 48 来源题、6 screen、6 confirmation；两个 selection 阶段的反馈不进入候选材料、Bank 案例或候选统计。

新方法身份为 `choice-capability-batch.v1`。来源允许正确、合法错答和截断记录；没有过程时不推测隐藏推理原因。先从公开任务需求诱导能力，再从实际保存的案例提出程序性指导，并做独立、有界的来源与内容检查。精确 Span 保留原始 LaTeX／Unicode，只表示来源兼容；候选不宣称数学证明。

有限实现最多生成两组候选，每组采用按公开 task_id 排序的两个独立物理来源，完整来源池仍为 48 题；组成员在模型请求前写入 plan，不按结果挑题或另开轮次。有效候选按实际独立支持来源数降序、不可变候选 digest 升序固定配对顺序。两组预声明来源为候选池排序前四题：202511:15、202511:17；202511:22、202511:25。每组最多四个来源的合同上限保留；本轮选用最小的两个来源。

全部 12 题先从公开输入做适用性检查，匹配少于 4 题时零配对 HTTP。只配对匹配题。两臂使用同一只读基线、同一求解提示和有效模型参数，保留其他 Program 与指导，只改变当前候选可见性。每个候选／任务／arm 使用独立请求和 checkpoint；第二候选反转调用顺序，不复用第一候选的 off 答案。筛选净增正确数 ≥0、空答不增加、正确变错误 ≤1；确认后累计实际暴露 ≥4、净增正确数 ≥2、空答不增加、正确变错误 ≤1 才发布。费用下降不独立触发晋升。

新派生 Bank 不继承 P1 资产。未经来源校验及两阶段效用验收的候选不能进入 Frozen。原 Program 单源 Worker 资格、一次 repair、Runtime high thinking／32,768 cap 均保留。额外最终回答保持关闭；对旧 Train15 个截断记录仅导出预算可行性，不补答，不声称能改善正确率。

## 身份、费用与使用方式

新入口为 `python -m atomic_skillgraph.experiments.run_choice_capability_batch`：`prepare` 和 `check` 不读密钥、不调用模型；`execute` 是后续明确启动付费学习／配对时使用的独立入口，当前未运行。不会让原 `run_single_cell` 自动启用 P2，也不会自动运行 Val/Test。

prepare 绑定实际 canonical 文件、P1 commit、模型／seed、原执行题单、原 Bank digest、原账本及文件 SHA；派生 config／材料和 plan 也固定 hash。旧账本 7,095,758 tokens／364 HTTP 计入一次，其中旧 Train Runtime 费用已经包含在这笔累计值中。新增有限额度为 1,800,000 tokens／80 HTTP，作为原 23,232,000 tokens／9,165 HTTP 总上限内的子额度，不清零旧费用。selection retry=0。未知计费或额度耗尽会停止、标明 censored，未计量额度以保守预留单列，不伪装实际 tokens。

复用本次只读 P1 来源准备的例子（输出目录必须不存在）：

```bash
python -m atomic_skillgraph.experiments.run_choice_capability_batch prepare \
  --settings configs/livemath_p2_selection_seed42.yaml \
  --source-run /home/yangchengyu/livemath_public_process_seed42_formal_20261012/livemath/seed42 \
  --output /home/yangchengyu/livemath_p2_recompile_seed42_offline_20261012
python -m atomic_skillgraph.experiments.run_choice_capability_batch check \
  --plan /home/yangchengyu/livemath_p2_recompile_seed42_offline_20261012/plan.json
```

必须在 prepare 所绑定的干净源码版本内继续；源码、参数、材料、Frozen 或账本变化会拒绝恢复。已经完成的相同 arm 可按原 checkpoint 身份恢复，完整请求、response metadata 与 raw usage 保持原记录，不产生新的 off 调用。

## 零模型验收

P2 专项 19 项：固定 48/6/6、来源／split 隔离、重复物理来源、虚构 Span、答案字母和数学结论拒绝、全部配对门槛边界、零 screen 暴露、覆盖不足、拒绝晋升、预算与未知计费停止、实际注入、Frozen 后独立消费、两候选顺序和恢复请求隔离。真实 P1 Train 输入已通过生产编译器的固定模拟候选及完整 12 题拦截配对到 Frozen 的路径，原文件读前读后相同。

原有相关回归 94 项通过，12 项因未指定历史回放目录而跳过；另有两项单来源真实 Worker 资格→Frozen→后题消费回归通过。镜像 `sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783`。截断、源码隔离与旧公开过程校验保持原规则。

真实来源回放修正了新入口的重复材料容量：短 Span 为原文首／尾 160 字符，内容检查只发送必需提案与原始来源，不重复发送候选内部状态和能力元数据；完整题干、选项、观测记录不被截掉，上限仍为 16,384 B。

所有正向配对分数是拦截夹具结果，用于核验控制流，不能作为 P2 自然生成率、准确率或泛化收益。当前没有付费候选，没有新的 Val17 或 Test100 结果。OfficeQA 固定运行副本未修改。
