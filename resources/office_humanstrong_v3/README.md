# OfficeQA HumanStrong V3

独立人工参考 Bank：12 张指导卡、6 个 Program、6 个 Implementation；4 个 Composite 仍仅保存，不自动执行。旧 V2 资源与运行保持原样。V3 复用 V2 的原生视图、公开输入 bridge 和五个既有 Program，新增源码仅在 `programs/polynomial_forecast.py`。

修改依据是已暴露的 Train 失败：搜索命中未继续读取正文、拟合序列与目标实际值未分别补齐、未在工具阶段执行回归。指导强调缺失年份/错位表格不能凭位置猜测；新增通用带截距多项式 OLS（次数 1..5），覆盖检查、预测、可选绝对误差和末步舍入。计算正确不证明操作数来源正确。资产没有题目 UID、固定历史年份/数值、标准答案或评分特判。

`tools/build_bundle.py` 从冻结 V2 生成新内容 ID 和引用；SQLite 是分发快照。人工作者权限仍只在 HumanBankView 生效，不改变自然学习的晋升标准。无模型训练或候选晋升。

本轮 `experiments/run_office_humanstrong_v3.py` 固定相同 canonical Train12，DeepSeek seed42、人工 Bank／NoSkill 双臂。两臂均保留正常 high thinking、32,768 输出 cap；收尾关闭 thinking、32,768 正文 cap；单题总额度 216,576、原生调用 48。最多新增 5,197,824 tokens／2,400 HTTP，继承最新收尾诊断账本，包括历史费用和未知计费保留，累计不超过已授权 116M。不启动 Val/Test，不自动重复实验。

公开输入与评分继续使用已批准的外部资源冻结版，不把人工核验得到的缺失值或原 PDF 转录偷偷补进题面。无法从现有文本消除的歧义应如实保留。与旧 V2 的纵向变化同时受收尾设置影响，主要比较本轮匹配的两臂；Train12 已用于 Bank 开发，不能当作盲测或完整 Test 准确率。

运行前固定已提交 WSL checkout，然后执行：

```bash
python experiments/run_office_humanstrong_v3.py prepare --root <全新目录>
python experiments/run_office_humanstrong_v3.py check --root <全新目录>
python experiments/run_office_humanstrong_v3.py launch --root <全新目录>
python experiments/run_office_humanstrong_v3.py status --root <该目录>
```

验收应覆盖原生导入、实际请求中的新能力、真实 Docker 执行、缺实际值/缺期/秩不足拒绝、账本结转与冻结身份。报告严格评分与格式差异分别统计；不要把确定性离线响应当成模型成功。
