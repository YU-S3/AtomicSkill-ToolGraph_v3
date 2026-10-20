# OfficeQA HumanStrong V6

版本 `office.humanstrong.programmatic.v6.20261021`。独立人工参考 Bank：14 Atomic、9 Program、9 Implementation、4 存档 Composite。Composite 不自动执行。V5 的资产与已完成实验不变；本目录不替换自然学习算法。

## 本轮修改

- 将“先计算”改为有条件的操作：所需来源、期间、统计约定齐全时计算，否则补具体缺项。允许原生 Python 和直接提交，不强制把已成功计算再做一遍。
- Atomic 的 goal 与指导同步；相关性／回归、多项式、离散程度分别描述适用条件。期间、财政口径、发布版本独立选入，不再使用最多两张专项卡的截断。
- 默认仅保留目标、单位和当前证据阶段的指导。计算、提交、恢复按公开任务和已有执行结果加入；候选数字引用不代表题目已完成。
- 不把明确的 inputs、operands、参数和诊断计数列为候选最终结果。同 Program 后续成功会清除旧拒绝；后续有实际计算产物时不继续要求重做旧的拒绝操作。历史原始结果始终保留。
- 所有 9 个 Program 仍有可见目录和真实 ID。常用当前接口内联，其余完整 Schema 通过普通 `read_result(schema_ref)` 获取，仍计工具预算。内联 `$ref` 使用当前材料的 `defs`，原生校验始终使用完整 Schema。不添加模型请求阶段或特殊检索工具。
- 表格提取器把分组标签、空白／缺失值数据行识别为数据区边界，保留多层表头。拒绝重复行、错误父／叶表头、越界窗口及缺失数值；不将缺失值当零。
- `table_calculate` 的 Schema 显式区分 series_metrics／statistics／polynomial_forecast；例如均值属于 series_metrics，CV／z-score 需要各自的约定。保留三个最终答案 Program 的格式合同。

## 验证及付费计划

离线检查读取保存的公开轨迹，真实执行 Docker、原生 Schema 读取和最终提交；模型响应由夹具给出，不调用模型。UID 仅出现在测试中。

本轮付费观察为 canonical **Val24 × Bank/NoSkill**，seed42 固定双臂顺序。全量 Val，无 Train 再划分，无学习或自动后续 Test。两臂使用同一 DeepSeek 模型、题面、语料、评分、工具、单题 216,576 tokens、48 原生调用和 32,768 收尾上限。

沿用用户批准的本地 Val 对齐版本：有源文件与页码提示，但没有外部完整公开页正文。因此是内部检索／效果检查，不能直接声称证明外部完整公开页 Test 的效果。Bank 编写已参考过去 Test，属于人工开发参考，不能作为未见 Test 的自然学习结果。

费用继承已完成 V5 双臂全部费用和未知计费预留。在累计 190M 授权内，本轮 48 次任务理论封顶 10,395,648 tokens。所有请求、原始 usage、轨迹、实际 Bank 快照及冻结前后 hash 单独记录。

```bash
PYTHONPATH=src python resources/office_humanstrong_v6/tools/build_bundle.py
PYTHONPATH=src python -m pytest tests/test_office_humanstrong_v6.py tests/test_office_humanstrong_v6_launch.py -q
```

付费启动器为 `experiments/run_office_humanstrong_v6.py`。必须从提交后的固定源码、新目录执行 prepare → check → launch；不复用旧实验目录。
