# OfficeQA HumanStrong V5

版本：`office.humanstrong.programmatic.v5.20261020`。

这是依据 V4 人工 Bank 的重复 Test 轨迹修订的人工参考资产，含 12 个 Atomic、9 个 Program、9 个 Implementation 和 4 个存档 Composite。Composite 不自动执行；运行仍由原生动态 Agent 选择 Program。它不是自然学习产物，也不能称为未见 Test 后得到的设计。旧 V4 和正在运行的自然学习不加载此目录。

## 修订内容

- `table_extract` 校验完整表头层次，记录实际行、列路径、表头与精确原文位置。父列与叶列冲突时返回 `needs_input`，不会自动推测错位列。重复叶列需提供唯一的完整 `header_path`。月份末尾句点仅在唯一匹配时兼容，不删除年份、内部标点或脚注来凑匹配。
- `table_calculate` 使用上述提取器，在一次 Program 中完成取数、计算、格式化和最终提交。来源单位、统计约定、显示格式分开；不隐式换币种或转换源单位。越界 offset 返回真实 source_size；拒绝结果带具体恢复信息。
- 新增 `computed_scalar` 和 `computed_list`。它们读取已获得的数字字段，通过原生 `final_answer` 路径直接提交，不增加一次模型改写。列表可分别指定每项精度和格式。读取字典、解释正文、非有限数字、截断结果或未知引用会拒绝。
- 新增显式 CV、z-score、成对利差均值计算。样本/总体标准差由 `ddof` 明确指定，差分由 `transform` 明确指定；利差两组必须具有相同的期序。未指定或不匹配时拒绝，不按题号猜统计约定。
- 引导 Agent 使用已提供源页先完成目标计算。运行材料记录实际数字结果引用及最近拒绝，检索脚本成功不会被直接标成任务计算完成。恢复仍使用既有 Runtime/预算，不新增 repair 或学习流程。

## 输出合同

三个最终 Program 都使用显式格式：

```json
{"style":"number","decimals":2,"scale":"as_is"}
{"style":"percent","decimals":2,"scale":"as_is"}
{"style":"percent","decimals":2,"scale":"fraction_to_percent"}
{"style":"unit_suffix","decimals":0,"scale":"as_is","suffix":"million euros"}
```

`number` 只输出数字；`percent` 只附加 `%`，不自动乘 100；`fraction_to_percent` 明确乘 100。已经由计算步骤乘 100 的结果不能再次缩放。数值在最终显示时才按 ROUND_HALF_UP 舍入；负零归为同精度零。`computed_list` 的 `layout` 为 `brackets`、`comma` 或 `lines`，元素顺序等于调用顺序。

`table_calculate` 中的精度从旧 `calculation.decimals` 迁至 `format.decimals`，旧 `answer_unit` 已移除。`tables[].unit` 仍须引用源表单位；显示后缀属于 `format`。这是 V5 的独立接口，不会套在 V4 运行中。

格式合同可以保证 **给定格式得到确定答案字符串**，不能证明模型选择了题目真正要求的格式、行或统计约定。题面没有明确指定百分号而标准答案采用某种约定的情形仍有歧义；没有读取 gold 来自动挑选格式，也没有修改评分。

## 接入及预算

使用本目录 `integration/configuration.py` 的 `configure` 和 `integration/frontier.py` 的 `native_import`、`HumanBankView`、`bridge`，在新目录导入 `bank/bank.json`。Bank 内嵌所有 Program 源码；SQLite 为同资产的只读导出。

`integration/evaluate.py` 提供 `evaluate(config, tasks, output, budget_governor=已有账本, arm='manual')`。它复用原生冻结评估器，校验资源和 Bank 版本，记录请求、轨迹、运行身份及费用；不创建或扩大付费授权，不自动启动下一轮。

维持外部公开输入版本、原评分、DeepSeek 配置、单题 216,576 tokens、48 次原生调用和 32,768 收尾输出上限。两臂共用输出说明，NoSkill 不加载资产。

全部 9 个 Program 始终可见。模型材料对重复 Schema 使用共享 `defs` 与 `$ref`；展开后与原生校验 Schema 逐项相同。存储的 Schema、Docker 执行和参数校验不使用压缩替代物。基础指导为目标、表格、单位和统计；有数字产物时加入提交指导，缺源页时加入检索指导，发生拒绝时加入恢复指导，另外最多加入两张相关专项卡。没有通过省略公开题面或源页腾预算。

## 重建和验证

在项目根目录、原 WSL Python 环境运行：

```bash
PYTHONPATH=src python resources/office_humanstrong_v5/tools/build_bundle.py
PYTHONPATH=src python -m pytest tests/test_office_humanstrong_v5.py -q
```

涉及真实历史记录的用例读取已有外部资源及固定运行目录；模型响应由测试夹具提供，真实模型 HTTP 为零。Docker、工具读取、Schema 校验、原生提交与评分实际执行。此验收不是新的准确率实验。
