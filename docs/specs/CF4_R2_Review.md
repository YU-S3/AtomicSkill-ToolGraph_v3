# CF4 R2 复审修复范围

本轮仅实现 20261008 复审文档的 M01/M02。

- M01：明确的 trial 宿主操作以现有 infrastructure_failure 包装未知异常，保留 cause、操作、trial stage 和 execution ID。learn_trace 保存原 solve score，写 failed_engineering 和 task_execution_finished，再上抛。输入不适用、声明拒绝、输出合同失败和已知预算仍沿原通路处理。
- M02：候选筛选和前八项顺序不变，只投影 Extractor 的 related 材料。Program 去源码；job 保留版本、固定绑定和状态；独立结果保留物理任务、outcome、评分及有界失败摘要。两槽信息只反映已有事实。Bank、Builder、Worker、评分和学习资格不变。
- Extractor 材料和请求审计记录 candidate.v1、规范 UTF-8 JSON 的投影前后字节数及候选 ID 序列。字节数不代表实测 tokens。

验收使用 tests/test_cf4_r2_review.py 的 Q1–Q5 八场景；禁止 HTTP、Docker 和生成源码执行。Q4 只读旧恢复 Bank 和第 58/59 题公开输入；Q5 只构造 Test 配置，不读取 Test 个体。现有 tests/test_cf4_r2.py 的 Z 回归单独统计，其中保存 Worker 的真实容器试用允许按原 fixture 执行。

正式续跑从原 R1 saved-score 第 57 题创建新修复 child，不普通 resume 755f980 子 Run，不导入诊断库。固定原 authority/profiles/model lock/datasets 的绝对路径，以新 child train/config.json 作 base。先核对 source identity、三个 split 全配置 hash、Train execution config，再交付 --max-new-tasks 2 --resume --stop-after-val 的单 SpreadsheetBench seed42 命令；两题边界通过后去掉 max-new-tasks 完成剩余 Train 和 Val。

历史解释沿复审文档修正：D1 Spreadsheet 两个既有正向不等于导出脚本独立复现；Office D1 失败且 D4 不适用。D2 两个非空结束响应未经独立评分，不能称为 2/2 正确。D3 配对结果不能推断因果。D4 能力覆盖不足、Program 调用为零，保留实际增量 176098 tokens，不能据此断言接管缺陷。旧源码 ZIP 含 CRLF，不称为 Git 原始 blob；新源码导出须逐项校验 blob bytes。

不重跑 D0–D4、旧 Train/Val、12+6，也不扩展修复次数、资格或正式费用 Governor。完成后的实际测试日志、恢复 receipt、配置预检和 Git blob 校验在本机复审交付包中提供。
