# OfficeQA HumanStrong V2

版本：`office.humanstrong.programmatic.v2.20261020`。

这是独立的人工参考 Bank，用于展示学习链路后续需要覆盖的能力。它不是 Ours 自然学习生成的资产，也没有经过新的付费准确率评估。旧 V1 与历史运行保持原样。

## 实际启用的内容

- 12 张 Atomic 指导卡，正文在 `guidance.md`。7 张核心卡始终提供，5 张专题卡按公开问题多标签选择；阶段信息不再隐藏指导。
- 5 个 Python Program 全程可见。Runtime 自主选择调用，执行仍经现有 Broker、Docker、输入输出 Schema 和同一工具预算。
- 5 个 Implementation 关联保留。4 个 Composite 作为存储中的参考图保留，**不会自动执行**；运行路径仍是原生动态工具循环。
- 使用原生 Runtime action Schema，不添加 `evidence_state` 报告字段。源页和实际执行结果有可读引用，不为填状态表额外请求模型。
- `series_metrics` 增加明确的 `symmetric_difference`、`share_difference`；保留原有公式。公开选项展示完整输入 Schema 与输出字段名，完整输出 Schema 留在资产和执行校验中。

核心指导覆盖：目标定义、表头层级、精确行列、时期与报告版本、单位与百分比尺度、公式选择、缺失证据处理及及时提交。不要强制初始检索，不对充分证据重复复核，不把普通变化率替代对称差异。

## 公开输入与评分版本

用户批准采用外部 MiniMax Trace2Skill 包中的完整数据版本。三个 seed 的 Train120/Test102 题目、标准答案及实际公开请求文本逐项一致。

`tools/materialize_inputs.py` 只导出白名单公开字段：题面、完整相关源页、候选文件与源链接。答案单独写入 `evaluator_records_*.json`，不进入 Task.inputs 或 Bank。原始成员 SHA256、文件路径映射与新文件 SHA256 记录在资源 manifest 中。不得从运行结果提取预测答案补充公开输入。

Val24 经用户同意，按外部 UID 使用现有本地题面与答案，**没有外部解析页，仅供内部检查，不作为外部设置完全一致的结果或论文结果**。

完整外部 transformed 语料未随 ZIP 提供。所有候选 basename 在本地授权语料中存在，但全文件字节一致性未获验证。直接提供的 Train/Test 源页正文与外部运行一致；进一步全文检索沿用本地语料，不能宣称两套完整工具环境完全相同。原生工具接口仍为 Unicode offset/regex/paths，指导明确区分外部 line-window/literal/path API。

## 运行接入

在项目 WSL Python 环境中执行，使用全新输出目录：

```bash
python resources/office_humanstrong_v2/tools/inspect_bundle.py
python resources/office_humanstrong_v2/tools/prepare_runtime.py \
  --base-config <现有已解析的Office配置.json> \
  --resources /home/yangchengyu/officeqa_external_minimax_resources_20261020_v1 \
  --output <新的独立准备目录>
```

准备步骤生成原生 Bank SQLite、导入身份和 Train/Val/Test 三份有效配置，零模型请求。不要把这些配置交给缺少人工视图的普通 Bank 运行器；调用 `integration/evaluate.py` 的 `evaluate(config, tasks, output, budget_governor=..., arm='manual'|'off')`，它绑定 `HumanBankView` 与输入 bridge，复用原生 evaluator、checkpoint、请求账本和 frozen 执行。调用方应使用当前累计账本的 Governor、固定源码和独立输出目录。准备步骤不分配新的付费额度。

2026-10-20 已授权的固定 Train12 双臂使用 `experiments/run_office_humanstrong_v2.py`：

```bash
python experiments/run_office_humanstrong_v2.py prepare --root <新的运行目录>
python experiments/run_office_humanstrong_v2.py check --root <新的运行目录>
python experiments/run_office_humanstrong_v2.py launch --root <新的运行目录>
python experiments/run_office_humanstrong_v2.py status --root <运行目录>
```

该入口绑定相同 V2 人工视图和输入 bridge，复用已有双臂运行器；继承最新已完成证据交接实验的账本及历史锁，固定 24 次解题、每次 192,000 tokens／48 次原生工具调用。本轮最多 4,608,000 tokens／2,400 HTTP，累计仍限 116M。只运行已开发暴露的 canonical Train 前 12 题，不学习、不运行 Val/Test、不按成绩选择题目或自动重跑。源码须为已提交的独立 WSL 固定 checkout。

配置启用公共证据交接版本和无损紧凑 JSON。正常请求保留全部公开源页，不继承原生文件输入预览的截断；收尾仅从已提供/已获得的内容中打包，容量不足仍有明确收据。源页登记为 `provided_public_input`，不会伪造工具调用。两臂共享同一公开输入。模型、thinking、任务总额度和工具调用上限没有被偷换成外部模型的设置。

## 验收与解释范围

`tests/test_office_humanstrong_v2.py` 覆盖：三份 split、逐题双臂最终请求、最长输入、私有答案隔离、无工具时收尾、5 个真实 Docker Program、两种新增公式、原生评估入口/账本/checkpoint、冻结 Bank 不变及错误版本拒绝。

离线 HTTP 用确定性响应代替模型，**只验证工程接口，不提供准确率或智能效果证据**。新 Bank 的 60%/70% 准确率均未被证明。外部三 seed OfficeQA 为 62/102、59/102、65/102，平均 60.78%；旧本机人工 V1 完整 Test 为 27/102（26.47%），模型、公开输入和预算不同，不能作为同条件 Bank 增益。旧 Test 与部分 Train 已被分析，后续应如实记录开发暴露。

构建 `tools/build_bundle.py` 在仓库内以 V1 的接口为基础生成独立 V2；运行和导入 V2 不依赖 V1 的 Bank 内容。
