# Atomic Unified R1 修复与单 seed 运行安排

依据本轮统一方法复审规范，仅修改 Ours。生产版本为 `empirical-v3.2-atomic-unified-r1`，实际提交由新运行的 `source_commit.txt` 与 `run_manifest.json` 固定。

## 修改

- 学习材料改为紧凑的真实操作目录，当前题不重复进入历史。Builder 携带选中操作的完整源码／参数；完整 trace、重放前缀与原产物身份留在 Host。最终 provider payload 包括 schema 和 JSON 转义，分别按 Extractor／Builder 16,384 bytes、结构修复 8,192 bytes 检查；无法完整表达时记录 `material_too_large`，HTTP 为零。
- Host 从模型选定的真实来源及输入／输出引用生成起点、前缀和绑定哈希。模型不可自填这些权威字段。多个绑定错误一起报告，不替换来源。Extractor 结构修复仍最多一次，使用 disabled thinking、2,048 completion 和强制 `submit_learning`；首次提取与 Builder 原 cap 保留。
- XLSX 文件效果仅规范化 `docProps/core.xml` 的 `modified` 时间，同时保存原始 SHA256。单元格、公式、样式、工作表和其他 metadata 改动仍拒绝。真实源捕获与生产 Worker 发布使用同一入口；旧 digest 不自动更换政策标签。
- 正式与有限验证共用 benchmark 配置合并函数。LiveMath learning/runtime choice guidance 成对开启，guidance learning／grounding 为 disabled thinking、2,048／1,536；SearchQA 不继承该适配。新增受版本控制的单 seed spec、预算分配及薄启动器。
- Train 200,000 分为 solve 120,000／learning 80,000，finish、retry 和学习 trial 按真实归属扣费，并同时满足父任务及整段额度。请求保留实际收尾额度；同任务重启不重置预算。额度耗尽记为 `budget_stopped`，不扩大预算或补造余题分数。

新政策进入 resolved config、请求审计、checkpoint 和 Frozen 身份。首次 usable 仍须真实源局部 Worker 验收；candidate 不进入 Runtime 或 Frozen。旧运行、Bank、评分及费用保留。

## 无 LLM 核验

完整生产代码回归得到 389 passed、1 failed、2 skipped；唯一失败来自历史 Learner 测试仍使用当前接口。将该夹具改为同时加载固定旧提交的真实 prompt schema 后，对本轮与历史模块重跑得到 21 passed。真实启动发现下述收尾返回值问题，修复后新增 Office 完整生产入口回归，对受影响模块再跑得到 **126 passed**。按不同用例合并为 **391 passed、0 failed、2 skipped**。两项跳过需要通过 `CF3_REVIEW_RUN` 指定旧 CF2 自然 Bank；本次没有指定该路径。

回放真实 Office 截断与 Spreadsheet reasoning-only 响应，保留已存公开内容和 raw usage；归档已删除私有 reasoning，仅为 HTTP 夹具补空的 reasoning 字符串，不重构私有内容。未恢复结构受控拒绝，没有额外求解或晋升。

| 旧真实 Train 材料 | Extractor 完整 bytes | Builder 完整 bytes | 两次准入预留合计 |
| --- | ---: | ---: | ---: |
| OfficeQA UID0170 | 14,092 | 13,417 | 52,085 |
| OfficeQA UID0221 | 13,954 | 13,279 | 51,809 |
| Spreadsheet 44296 | 13,017 | 7,581 | 45,174 |
| Spreadsheet 58032 | 12,972 | 7,944 | 45,492 |

上述均为保存材料和确定性工程响应的 HTTP 截获夹具，不是新增自然学习结果。实际生产 Docker Worker 的 XLSX 发布正向、错误原生动作反例、源验收→usable→Frozen→非源消费主链、两池准入和身份恢复回归通过。收费模型请求为 **0**。

原审查 ZIP SHA256 仍为 `6b2d297093666e4d3db2327e3fc8d6ffab25c8c3393b3873485376e9e0a97433`。机器核验见 [verification](atomic_unified_r1_verification_20261010.json)，完整日志和逐项结果保存在 `D:/T3S_exp/deliveries/outputs/atomic_unified_r1_verification_20261010/`。

## 用户确认的预算

按每题上限之和冻结。HTTP 上限包含既有求解循环、最多一次结构修复、最多两次 Builder 生成和每请求最多四次重试，不新增循环或修复次数。

| seed42 cell | 整段 token 硬上限 | HTTP 硬上限 |
| --- | ---: | ---: |
| SearchQA | 196,704,000 | 83,580 |
| LiveMath | 23,232,000 | 9,165 |
| OfficeQA | 36,096,000 | 284,070 |
| SpreadsheetBench | 59,200,000 | 558,000 |
| 合计 | **315,232,000** | **934,815** |

这些是理论封顶，不是预期消耗。整段 `finish_reserve=0`，单题普通请求仍预留实际 bounded finish 的输入界和输出 cap。每个 cell 独立预算文件；不使用多进程共同写入的 JSON 总账。

## 运行

仅当前 DeepSeek、四个 benchmark、seed42，新目录与独立空 Bank。ALFWorld 暂缓收费；文本模型的 DocVQA 仍为 unsupported；seed43／44 不启动。

初次前缀使用提交 `7e4225e1d2d138a9620f6660967d62dcf7bbc602`，在 `/home/yangchengyu/asg_atomic_unified_r1_20261010` 的独立源码副本运行。输出为 `/home/yangchengyu/atomic_unified_r1_seed42_formal_20261010`。前五题是正常 Train 前缀，不追加 Val；已有 Bank 和请求保持原身份。

真实启动确认 OfficeQA 的 Runtime 收尾预留将 `build_finish_evidence` 返回字典错误地解包为二元组，在 HTTP 前发生 `too many values to unpack`。这是 Host 工程错误，不是模型执行失败或预算拒绝。现改为读取真实 `materials` 字段，新增实际 `run_task` + HTTP 截获回归，确认可以正常完成 Runtime 与评分且实际 solve/finish 预留被记录。

OfficeQA 已通过 `STOP_AFTER_TASK` 在第 4 题后停止，原记录、原分数和费用不改。已有 **73,749 tokens、9 HTTP**，均已计量；`SystemExit(75)` 是任务边界停止。收尾接口修复提交为 `724d3e91fab82c7f63e0eb4ad7d43540d094f097`。用户随后明确要求“先不重开 OfficeQA”；保持停止，没有新建收费 run，也没有继承旧 Office Bank。

SearchQA、LiveMath、SpreadsheetBench 均已完成各自五题前缀，并在任务边界停止。这三项保留原固定源码和目录；当前无实验进程在运行。**当前不使用旧四项一键 resume 继续 OfficeQA**。对已完成且工程记录正常的其他 cell，可分别使用真实 CLI 继续同一 Train→Frozen→Val→Test，例如：

```bash
cd /home/yangchengyu/asg_atomic_unified_r1_20261010
PYTHONPATH="$PWD/src" /home/yangchengyu/asg_alfworld_venv/bin/python -m atomic_skillgraph.experiments.run_single_cell \
  --config configs/atomic_unified_seed42.yaml --benchmark searchqa --seed 42 \
  --model-key deepseek-v4-flash \
  --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1 \
  --output /home/yangchengyu/atomic_unified_r1_seed42_formal_20261010/searchqa/seed42 \
  --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env --resume
```

LiveMath／SpreadsheetBench 只替换对应 benchmark 和输出目录，先确认前缀进程结束。不要在同一 cell 上同时运行两个进程。通用薄启动器仍用于干净的新四项启动；默认只打印，`--execute` 前五题、`--resume` 同目录后续。使用修复提交启动的新运行必须在实际 source manifest 中明确该提交，不能热改旧 checkout 或旧 manifest。

观察 `logs/<benchmark>.seed42.log`、各 cell 的 `run_manifest.json`、`episodes.jsonl`、`llm_calls.jsonl`、`train/summary.json` 和 `completion.json`。状态和费用以这些实际记录为准。没有自然 Program、正常失败或低分不构成工程停止门槛；确认合同、账本或基础设施错误时只停止受影响 cell。
